"""Goal continuation lifecycle contracts over durable gateway state."""
from __future__ import annotations

import json

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_automation import admit_route_wake
from gateway.session_contract import Principal
from hermes_state_runtime import list_session_admissions


GOAL_TEXT = "[Continuing toward your standing goal]\nGoal: ship it"


def _goal_row(rows):
    return next(row for row in rows if row["request_id"].startswith('["goal-continuation"'))


def test_only_queued_trusted_goal_rows_are_withdrawable():
    from gateway.session_automation import is_goal_continuation_row

    route = "agent:main:local:route"
    request_id = json.dumps(["goal-continuation", route, "uuid"], separators=(",", ":"))
    row = {
        "status": "queued",
        "principal_id": "automation:" + route,
        "request_id": request_id,
        "payload": {"local_automation_v1": {"identity": request_id}},
    }
    assert is_goal_continuation_row(row, route)
    for status in ("started", "unknown"):
        assert not is_goal_continuation_row({**row, "status": status}, route)
    assert not is_goal_continuation_row({**row, "principal_id": "human"}, route)
    assert not is_goal_continuation_row({**row, "request_id": "goal-continuation"}, route)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stale_epoch", "permission_denied", "runtime_draining", "not_found"])
async def test_withdraw_refusals_are_not_silently_treated_as_cancellation(
    owner, monkeypatch, tmp_path, reason,
):
    from gateway.session_automation import _withdraw
    from hermes_state_runtime import RuntimeStoreError, get_session_admission
    from tests.gateway.test_goal_claim_boundaries import queued_goal

    _ref, _live, _entry, wake, _user = await queued_goal(owner, monkeypatch, tmp_path)
    monkeypatch.setattr(owner, "cancel_queued", AsyncMock(side_effect=RuntimeStoreError(reason)))
    with pytest.raises(RuntimeStoreError, match=reason):
        await _withdraw(owner, wake)
    saved = get_session_admission(owner.db, admission_id=wake["admission_id"])
    assert (saved["status"], saved["outcome"]) == ("queued", None)


@pytest.mark.asyncio
async def test_withdraw_claimed_row_accepts_stale_generation(owner, monkeypatch, tmp_path):
    from gateway.session_automation import _withdraw
    from hermes_state_runtime import RuntimeStoreError, get_session_admission
    from tests.gateway.test_goal_claim_boundaries import queued_goal

    _ref, _live, _entry, wake, _user = await queued_goal(owner, monkeypatch, tmp_path)
    owner.db._execute_write(lambda conn: conn.execute(
        "UPDATE session_admissions SET status='started',owner_epoch=?,generation=? WHERE admission_id=?",
        (owner.epoch, 1, wake["admission_id"])))
    monkeypatch.setattr(owner, "cancel_queued", AsyncMock(side_effect=RuntimeStoreError("stale_generation")))
    assert await _withdraw(owner, wake) is False
    saved = get_session_admission(owner.db, admission_id=wake["admission_id"])
    assert saved["status"] == "started"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["pause", "clear"])
async def test_pause_or_clear_cancels_persisted_goal_continuation_and_keeps_user_input(
    owner, monkeypatch, tmp_path, action,
):
    """Pause withdraws the durable goal wake while a normal queued row survives."""
    from gateway.session_local import create_local_session
    from hermes_cli import goals

    owner.runner.adapters = {}
    owner.runner._adapters_for_profile = lambda _profile=None: owner.runner.adapters
    owner.runner._adapter_for_source = lambda _source: owner.runner.adapters[Platform.LOCAL]
    actor = Principal("owner", owner.profile_id, frozenset({"session:create", "session:submit", "session:control"}), "test")
    ref = create_local_session(owner, actor, {"request_id": "goal", "source": "cli", "cwd": str(tmp_path), "model": "m", "toolsets": []})
    source = owner.sessions[ref.session_id].source
    entry = owner.runner.session_store.lookup_by_session_key(owner.sessions[ref.session_id].route)
    owner.runner._session_key_for_source = lambda _source: entry.session_key
    owner.runner._delivery_adapter_for = lambda _source: owner.runner.adapters[Platform.LOCAL]
    owner.runner._adapter_and_key_for = lambda _event: (owner.runner.adapters[Platform.LOCAL], entry.session_key)
    owner.runner._clear_goal_pending_continuations = lambda _key, _adapter: (_ for _ in ()).throw(RuntimeError('adapter cleanup failed'))
    monkeypatch.setattr(goals, "_get_session_db", lambda: owner.db)
    goals.GoalManager(entry.session_id).set("ship it")
    adapter = owner.runner.adapters[Platform.LOCAL]
    assert await admit_route_wake(owner.runner, adapter, source, GOAL_TEXT, "goal-continuation")
    row = _goal_row(list_session_admissions(owner.db, session_id=owner.logical_owner(entry.session_id), pending_only=False))
    from hermes_state_runtime import admit_session_input
    user = admit_session_input(owner.db, epoch=owner.epoch, principal_id="human", session_id=row["target_session_id"],
                               request_id="user-queued", payload={"text": "keep me"})
    assert row["status"] == "queued"
    mgr = goals.GoalManager(entry.session_id)
    getattr(mgr, action)()
    event = MessageEvent(text=f"/goal {action}", message_type=MessageType.TEXT, source=source)
    from gateway.slash_commands_goals import GatewayGoalCommandsMixin
    await GatewayGoalCommandsMixin._clear_goal_continuations(owner.runner, event, action)
    rows = list_session_admissions(owner.db, session_id=row["target_session_id"], pending_only=False)
    by_request = {r["request_id"]: r for r in rows}
    assert by_request[row["request_id"]]["outcome"] == "cancelled"
    assert by_request[user["request_id"]]["status"] == "queued"


@pytest.mark.asyncio
async def test_pause_during_goal_judge_does_not_get_overwritten(owner, monkeypatch):
    """The judge's stale GoalManager snapshot cannot reactivate a paused goal."""
    from hermes_cli import goals

    sid = "judge-pause"
    monkeypatch.setattr(goals, "_get_session_db", lambda: owner.db)
    mgr = goals.GoalManager(sid)
    mgr.set("ship it")
    def pause_inside_judge(*_args, **_kwargs):
        goals.GoalManager(sid).pause()
        return "continue", "more", False, None, False

    monkeypatch.setattr(goals, "judge_goal", pause_inside_judge)
    mgr.evaluate_after_turn("partial")
    assert goals.GoalManager(sid).state.status == "paused"


@pytest.mark.asyncio
async def test_new_goal_during_goal_judge_is_not_overwritten(owner, monkeypatch):
    """An old judge result must not overwrite a different goal that became active meanwhile."""
    from hermes_cli import goals

    sid = "judge-replaced-goal"
    monkeypatch.setattr(goals, "_get_session_db", lambda: owner.db)
    mgr = goals.GoalManager(sid)
    mgr.set("old goal")

    def replace_inside_judge(*_args, **_kwargs):
        goals.GoalManager(sid).set("new goal")
        return "continue", "more", False, None, False

    monkeypatch.setattr(goals, "judge_goal", replace_inside_judge)
    mgr.evaluate_after_turn("partial")
    latest = goals.GoalManager(sid).state
    assert latest.goal == "new goal"
    assert latest.turns_used == 0


def test_failed_and_interrupted_results_are_ineligible_for_goal_judge():
    from gateway.run import GatewayRunner

    assert not GatewayRunner._goal_hook_result_is_eligible({"final_response": "error", "failed": True})
    assert not GatewayRunner._goal_hook_result_is_eligible({"final_response": "stopped", "interrupted": True})
    assert not GatewayRunner._goal_hook_result_is_eligible({"final_response": "partial", "completed": False})
    assert GatewayRunner._goal_hook_result_is_eligible({"final_response": "done", "completed": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"final_response": "failed copy", "failed": True},
    {"final_response": "interrupted copy", "interrupted": True},
    {"final_response": "incomplete copy", "completed": False},
])
async def test_post_turn_hooks_skip_goal_for_noncompleted_visible_outcomes(monkeypatch, result):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="s")),
    )
    runner._post_turn_goal_continuation = AsyncMock()
    runner._post_turn_loop_completion = AsyncMock()
    runner._publish_local_control = lambda *_args, **_kwargs: None
    monkeypatch.setattr("gateway.session_authorities.active_authority", lambda _runner: None)
    monkeypatch.setattr("gateway.session_viewer_automation.viewer_driven", lambda *_args: False)

    await runner._run_post_turn_hooks(
        agent_result=result,
        source=SimpleNamespace(platform=Platform.DISCORD, chat_id="chat"),
        is_internal=False,
    )

    runner._post_turn_goal_continuation.assert_not_awaited()
    runner._post_turn_loop_completion.assert_awaited_once()
