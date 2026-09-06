"""Behavior oracle from #76230, exercised against the unit/claim surface of #104299.

SQLite, transcript row construction, tool-boundary hook and spill storage are real.
Only child/provider execution and explicit failure injection are replaced.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent import delegation_inject as inject
from agent.session_persistence import _db_flush_row, _db_flush_write
from agent.tool_executor import _flush_session_db_after_tool_progress
from hermes_state import SessionDB
from tools import async_delegation as ad
from tools.budget_config import BudgetConfig


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(ad, "_records", {})
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="parent", source="cli")
    agent = SimpleNamespace(
        session_id="parent", _active_turn_id="turn-a", _session_db=db, _persist_disabled=False,
        max_iterations=10, iteration_budget=SimpleNamespace(remaining=5), _api_call_count=1,
        _interrupt_requested=False, _budget_grace_call=False,
    )

    def persist(messages):
        pending = [m for m in messages if m.get("_db_persisted") is not True]
        rows = [_db_flush_row(agent, m, False) for m in pending]
        _db_flush_write(agent, rows, pending)
        return True

    agent._flush_messages_to_session_db = persist
    counter = 0

    def event(**overrides):
        nonlocal counter
        counter += 1
        now = time.time()
        evt = dict(
            type="async_delegation", delegation_id=f"deleg_unit_{counter}", session_key="parent",
            parent_session_id="parent", parent_turn_id="turn-a", result_delivery="inject",
            goal="Review the change", status="completed", summary=f"Review evidence {counter}",
            context=None, toolsets=None, role="leaf", model="dummy", api_calls=1,
            dispatched_at=now - 1, completed_at=now, duration_seconds=1,
        )
        evt.update(overrides)
        record = {**evt, "status": "completed"}
        ad._records[evt["delegation_id"]] = record
        ad._persist_dispatch(record)
        ad._persist_completion(evt, {"status": "completed", "summary": evt.get("summary")})
        return evt

    yield SimpleNamespace(agent=agent, db=db, event=event, persist=persist, path=tmp_path)
    db.close()


def messages(contents=("tool output",)):
    return [
        {"role": "system", "content": "cached system prefix", "_db_persisted": True},
        {"role": "user", "content": "original user request"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call-{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
            for i in range(len(contents))]},
        *[{"role": "tool", "tool_call_id": f"call-{i}", "tool_name": "terminal", "content": content}
          for i, content in enumerate(contents)],
    ]


def flush(rig, history, budget=None):
    return _flush_session_db_after_tool_progress(
        rig.agent, history, stage="test", storage_env=None,
        budget_config=budget or BudgetConfig(default_result_size=10_000, turn_budget=16_000, preview_size=128),
    )


def state(event):
    return ad.get_durable_delegation(event["delegation_id"])


def test_new_last_result_carries_all_ready_units_atomically(rig):
    events = [rig.event(), rig.event()]
    history = messages(("first result", "last result"))
    prefix = deepcopy(history[0])
    assert flush(rig, history)
    assert history[0] == prefix
    assert history[-2]["content"] == "first result"
    assert all(e["summary"] in history[-1]["content"] for e in events)
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool", "tool"]
    assert history[-1]["tool_call_id"] == "call-1"
    assert history[-1]["display_metadata"]["delegation_event_ids"] == [e["delegation_id"] for e in events]
    assert inject._CLAIMS not in history[-1]
    assert all(state(e)["delivery_state"] == "delivered" for e in events)
    assert all(ad.claim_event_delivery(e, "late-idle-consumer") is None for e in events)
    assert ad.restore_undelivered_completions(queue.Queue()) == 0
    stored = rig.db.get_messages("parent")
    assert all(e["summary"] in stored[-1]["content"] for e in events)
    assert "tool-boundary:" not in json.dumps(stored)


@pytest.mark.parametrize("overrides", [
    {"result_delivery": "after_turn"}, {"result_delivery": None},
    {"result_delivery": "future-policy"}, {"parent_turn_id": "previous-turn"},
    {"parent_session_id": "foreign-session"}, {"parent_turn_id": ""},
])
def test_noneligible_units_stay_on_existing_idle_rail(rig, overrides):
    evt = rig.event(**overrides)
    history = messages()
    assert flush(rig, history)
    assert history[-1]["content"] == "tool output"
    assert state(evt)["delivery_attempts"] == 0
    token = ad.claim_event_delivery(evt, "idle")
    assert token
    ad.release_event_delivery(evt, token)


@pytest.mark.parametrize("guard", ["no_db", "disabled", "interrupt", "grace", "exhausted"])
def test_ineligible_boundary_never_spends_a_delivery_attempt(rig, guard):
    if guard == "no_db":
        rig.agent._session_db = None
        rig.agent._flush_messages_to_session_db = lambda _messages: None
    elif guard == "disabled":
        rig.agent._persist_disabled = True
        rig.agent._flush_messages_to_session_db = lambda _messages: None
    elif guard == "interrupt":
        rig.agent._interrupt_requested = True
    elif guard == "grace":
        rig.agent._budget_grace_call = True
    else:
        rig.agent.iteration_budget.remaining = 0
    evt = rig.event()
    history = messages()
    assert flush(rig, history)
    assert history[-1]["content"] == "tool output"
    assert state(evt)["delivery_attempts"] == 0


@pytest.mark.parametrize("outcome", [False, None, "exception"])
def test_failed_or_noop_flush_restores_carrier_and_releases_claim(rig, outcome):
    evt = rig.event()
    history = messages()
    history[-1]["display_metadata"] = {"existing": {"nested": [1, 2]}}
    original = deepcopy(history[-1])

    def fail(_messages):
        if outcome == "exception":
            raise OSError("simulated disk failure")
        return outcome

    rig.agent._flush_messages_to_session_db = fail
    assert flush(rig, history) is (outcome is None)
    history[-1].pop(inject._CHECKED, None)
    assert history[-1] == original
    assert state(evt)["delivery_state"] == "pending"
    assert ad.claim_event_delivery(evt, "idle-after-failure")
    assert rig.db.get_messages("parent") == []


def test_lost_claim_rolls_back_transcript_and_other_unit_ack(rig):
    first, second = rig.event(), rig.event()
    history = messages()

    def lose_one_then_persist(msgs):
        claims = msgs[-1][inject._CLAIMS]
        ad.release_completion_delivery(second["delegation_id"], claims[second["delegation_id"]])
        return rig.persist(msgs)

    rig.agent._flush_messages_to_session_db = lose_one_then_persist
    assert not flush(rig, history)
    assert history[-1]["content"] == "tool output"
    assert rig.db.get_messages("parent") == []
    assert state(first)["delivery_state"] == state(second)["delivery_state"] == "pending"
    assert ad.claim_event_delivery(first, "retry-first")
    assert ad.claim_event_delivery(second, "retry-second")


def test_crash_after_commit_before_ram_markers_cannot_replay_result(rig, monkeypatch):
    evt = rig.event()
    history = messages()

    def marker_failure(*_args):
        raise RuntimeError("crash after COMMIT, before marker publication")

    monkeypatch.setattr("agent.session_persistence.sync_flushed_message_markers", marker_failure)
    assert not flush(rig, history)
    assert evt["summary"] in history[-1]["content"]
    assert state(evt)["delivery_state"] == "delivered"
    assert ad.restore_undelivered_completions(queue.Queue()) == 0
    assert ad.claim_event_delivery(evt, "restart") is None
    assert evt["summary"] in rig.db.get_messages("parent")[-1]["content"]


def test_preparation_exception_releases_acquired_claim_without_mutating_content(rig, monkeypatch):
    evt = rig.event()
    history = messages()

    def fail(*_args):
        raise RuntimeError("failure immediately after acquiring claims")

    monkeypatch.setattr(inject, "_Prepared", fail)
    assert flush(rig, history)
    assert history[-1]["content"] == "tool output"
    assert state(evt)["delivery_state"] == "pending"
    assert ad.claim_event_delivery(evt, "idle")


def test_large_report_spills_full_text_and_stays_within_budget(rig):
    report = "complete child evidence\n" * 2000
    evt = rig.event(summary=report)
    history = messages()
    budget = BudgetConfig(default_result_size=2000, turn_budget=3000, preview_size=128)
    assert flush(rig, history, budget)
    assert "<persisted-output>" in history[-1]["content"]
    assert len(history[-1]["content"]) <= budget.default_result_size
    from tools.tool_result_storage import get_spillover_dir
    assert any(report in p.read_text() for p in get_spillover_dir().glob("*.txt"))
    assert state(evt)["delivery_state"] == "delivered"


def test_failed_spill_defers_without_claim_or_truncation(rig, monkeypatch):
    evt = rig.event(summary="long evidence " * 2000)
    monkeypatch.setattr("tools.tool_result_storage._write_to_spillover", lambda *_args: None)
    history = messages()
    assert flush(rig, history, BudgetConfig(default_result_size=1000, turn_budget=2000))
    assert history[-1]["content"] == "tool output"
    assert state(evt)["delivery_attempts"] == 0


def test_existing_batch_output_counts_against_aggregate_budget(rig):
    evt = rig.event()
    history = messages(("a" * 1600, "b" * 1600))
    assert flush(rig, history, BudgetConfig(default_result_size=4000, turn_budget=3000))
    assert history[-1]["content"] == "b" * 1600
    assert state(evt)["delivery_attempts"] == 0


def test_multimodal_carrier_preserves_existing_image_blocks(rig):
    evt = rig.event()
    blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
              {"type": "text", "text": "original image description"}]
    history = messages((deepcopy(blocks),))
    assert flush(rig, history)
    assert history[-1]["content"][:-1] == blocks
    assert evt["summary"] in history[-1]["content"][-1]["text"]
    assert state(evt)["delivery_state"] == "delivered"


def test_incomplete_or_duplicate_call_tail_is_not_a_boundary():
    incomplete = messages(("one", "two"))[:-1]
    assert inject._completed_tool_batch_size(incomplete) == 0
    complete = messages(("one", "two"))
    complete[-1]["tool_call_id"] = complete[-2]["tool_call_id"]
    assert inject._completed_tool_batch_size(complete) == 0


def test_late_result_does_not_rewrite_already_persisted_tool_message(rig):
    history = messages()
    assert flush(rig, history)
    snapshot = deepcopy(history)
    evt = rig.event()
    assert flush(rig, history)
    assert history == snapshot
    assert state(evt)["delivery_attempts"] == 0


def test_model_dispatch_forwards_policy_and_preserves_orchestrator_sync(monkeypatch):
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    calls = []
    monkeypatch.setattr(dt, "delegate_task", lambda **kw: calls.append(kw) or "{}")
    for depth in (0, 1):
        parent = SimpleNamespace(_delegate_depth=depth)
        AIAgent._dispatch_delegate_task(parent, {"tasks": [{"goal": "review"}], "result_delivery": "inject"})
        assert calls[-1]["result_delivery"] == "inject"
        assert calls[-1]["background"] is (depth == 0)


def test_units_keep_grouping_and_inherit_policy_and_origin_turn(rig, monkeypatch):
    from tools import delegate_tool_dispatch as dispatch
    tasks = [{"goal": "independent"}, {"goal": "group-a", "group": "review"},
             {"goal": "group-b", "group": "review"}]
    batch = dispatch._Batch(
        tasks, [(i, task, SimpleNamespace()) for i, task in enumerate(tasks)], rig.agent,
        {"model": "dummy"}, None, "leaf", 3, "deleg_call", [], [], "", "", None, None,
        time.monotonic(), result_delivery="inject",
    )
    units = dispatch._units_of(batch)
    assert [[i for i, _, _ in unit.children] for unit in units] == [[0], [1, 2]]
    assert [unit.group for unit in units] == [None, "review"]
    captured = []
    monkeypatch.setattr(ad, "dispatch_async_delegation_batch", lambda **kw: captured.append(kw) or {})
    for i, unit in enumerate(units):
        dispatch._dispatch_unit(unit, f"deleg_call-{i + 1}", "one-slot", {
            "session_key": "parent", "parent_session_id": "parent", "parent_turn_id": "turn-a",
            "result_delivery": unit.result_delivery,
        })
    assert all(c["slot_key"] == "one-slot" and c["result_delivery"] == "inject" for c in captured)
    assert [c["task_indexes"] for c in captured] == [[0], [1, 2]]
    assert all(c["parent_turn_id"] == "turn-a" for c in captured)


def test_tui_rejected_claim_releases_reserved_turn(monkeypatch):
    from tui_gateway import server
    monkeypatch.setattr(ad, "claim_event_delivery", lambda *_args: None)
    session = {"running": True, "history_lock": threading.RLock()}
    server._notif_dispatch_event("tab", session, {"type": "async_delegation"}, "already carried")
    assert session["running"] is False
