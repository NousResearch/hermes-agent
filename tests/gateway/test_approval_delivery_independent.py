"""A /goal never treats work that waits on, or was refused, an approval as progress.

A background child whose gated action was denied, withdrawn, timed out or never reached the user
ends ``blocked`` with ``approval_blockers``. The gateway's post-turn goal hook pauses the goal on
that deterministically (no judge call, no automatic continuation) and says why in the chat. An
explicit ``/goal resume`` is a new boundary: the refusal it already reported does not pause it again.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key
from tests.waiting import async_wait_until


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    from tools import async_delegation as ad
    goals._DB_CACHE.clear()
    goals._get_session_db()  # warm from sync context; the loop thread will not build it inline
    ad._reset_for_tests()
    yield home
    ad._reset_for_tests()
    goals._DB_CACHE.clear()


class _RecordingAdapter:
    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[str] = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sends.append(content)
        return type("R", (), {"success": True, "message_id": "m"})()


def _runner():
    from gateway.run import GatewayRunner
    src = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="tester", chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner._running_agents = {}
    entry = SessionEntry(session_key=build_session_key(src), session_id=f"goal-{uuid.uuid4().hex[:8]}",
                         created_at=datetime.now(), updated_at=datetime.now(),
                         platform=Platform.TELEGRAM, chat_type="dm")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)
    adapter = _RecordingAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, adapter, entry, src


def _blocked_child(parent_session_id, delegation_id="deleg_blocked"):
    from tools import async_delegation as ad
    ad._persist_dispatch({"delegation_id": delegation_id, "session_key": "", "origin_ui_session_id": "",
                          "parent_session_id": parent_session_id, "dispatched_at": time.time(), "goal": "deploy"})
    ad._persist_completion({"delegation_id": delegation_id, "status": "blocked", "completed_at": time.time()},
                           {"status": "blocked", "exit_reason": "approval_required", "error": "approval",
                            "approval_blockers": [{"status": "blocked", "approval_outcome": "denied"}]})


async def _turn(runner, adapter, entry, src, sent_before):
    await runner._post_turn_goal_continuation(session_entry=entry, source=src, final_response="Done, deployed.")
    await async_wait_until(lambda: len(adapter.sends) > sent_before, timeout=5)


@pytest.mark.asyncio
async def test_a_blocked_child_pauses_the_goal_without_asking_the_judge(hermes_home):
    from hermes_cli.goals import GoalManager
    runner, adapter, entry, src = _runner()
    GoalManager(entry.session_id).set("ship the release")
    _blocked_child(entry.session_id)

    with patch("hermes_cli.goals.judge_goal") as judge:
        await _turn(runner, adapter, entry, src, 0)

    judge.assert_not_called()
    state = GoalManager(entry.session_id).state
    assert state.status == "paused"
    assert state.last_verdict == "blocked"
    assert "deleg_blocked" in adapter.sends[0]
    assert not adapter._pending_messages, "a blocked goal must not enqueue a continuation"


@pytest.mark.asyncio
async def test_resume_is_not_paused_again_by_the_refusal_it_already_reported(hermes_home):
    from hermes_cli.goals import GoalManager
    runner, adapter, entry, src = _runner()
    GoalManager(entry.session_id).set("ship the release")
    _blocked_child(entry.session_id)
    with patch("hermes_cli.goals.judge_goal") as judge:
        await _turn(runner, adapter, entry, src, 0)
        judge.assert_not_called()

    GoalManager(entry.session_id).resume()
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)) as judge:
        await _turn(runner, adapter, entry, src, 1)

    judge.assert_called_once()
    assert GoalManager(entry.session_id).state.status == "active"


@pytest.mark.asyncio
async def test_a_new_refusal_after_resume_pauses_again(hermes_home):
    from hermes_cli.goals import GoalManager
    runner, adapter, entry, src = _runner()
    GoalManager(entry.session_id).set("ship the release")
    _blocked_child(entry.session_id, "deleg_first")
    with patch("hermes_cli.goals.judge_goal"):
        await _turn(runner, adapter, entry, src, 0)
    GoalManager(entry.session_id).resume()

    _blocked_child(entry.session_id, "deleg_second")
    with patch("hermes_cli.goals.judge_goal") as judge:
        await _turn(runner, adapter, entry, src, 1)

    judge.assert_not_called()
    assert GoalManager(entry.session_id).state.status == "paused"
    assert "deleg_second" in adapter.sends[1]
