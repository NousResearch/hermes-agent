"""Delayed approval notices remain scoped to their original delegation (#130231)."""

import json
import sqlite3
from unittest.mock import Mock

import pytest

from agent.approval_outcomes import APPROVAL_OUTCOME_NOTICES
from tests.agent.test_native_pre_tool_approval import (
    TOOL,
    _child,
    _run_child,
    handler,  # noqa: F401 - shared effect-boundary fixture
    native_hook,  # noqa: F401 - real plugin hook registration fixture
)
from tests.agent.test_tool_call_incremental_persistence import _mock_response, _mock_tool_call
from tools import approval, async_delegation as ad
from tools.process_registry import process_registry
from tools.process_registry_notifications import _format_async_delegation


@pytest.mark.parametrize("old_outcome", ["timeout", "cancelled"])
@pytest.mark.parametrize("fresh_denial", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_delayed_approval_completion_cannot_replace_fresh_run(
    native_hook, handler, monkeypatch, old_outcome, fresh_denial, batch
):
    native_hook({"action": "approve", "message": "same user task"})
    gate = Mock(side_effect=[
        approval._denied("old gate", outcome=old_outcome,
                         pattern_key="old-request", description="fixture"),
        (approval._denied("fresh gate", outcome="denied",
                          pattern_key="fresh-request", description="fixture")
         if fresh_denial else {"approved": True}),
    ])
    monkeypatch.setattr(approval, "request_tool_approval", gate)
    records, entries = [], []
    for attempt in ("old", "fresh"):
        record = {"delegation_id": f"approval-{attempt}", "goal": "same user task",
                  "is_batch": batch, "dispatched_at": 1.0}
        ad._persist_dispatch(record)
        child = _child()
        create = child.client.chat.completions.create
        create.side_effect = [
            _mock_response("", "tool_calls", [_mock_tool_call(TOOL, call_id=attempt)]),
            _mock_response('{"answer":"fresh authorized work"}'),
        ]
        entry = _run_child(child)
        expected_calls = 2 if attempt == "fresh" and not fresh_denial else 1
        assert create.call_count == expected_calls
        records.append(record)
        entries.append(entry)

    assert gate.call_count == 2
    assert handler.call_count == (0 if fresh_denial else 1)
    assert entries[0]["approval_outcome"] == old_outcome
    if fresh_denial:
        assert entries[1]["status"] == "failed"
        assert entries[1]["approval_outcome"] == "denied"
    else:
        assert entries[1]["status"] == "completed"
        assert "approval_outcome" not in entries[1]

    def publish(index):
        entry = entries[index]
        ad._push_completion_event(records[index], {"results": [entry]} if batch else entry,
                                  entry["status"])
        return process_registry.completion_queue.get(timeout=5)

    def durable(delegation_id):
        # Reopen SQLite for each check; do not rely on an in-process result cache.
        with sqlite3.connect(ad._db_path()) as db:
            return db.execute(
                "SELECT state, event_json, result_json FROM async_delegations WHERE delegation_id=?",
                (delegation_id,),
            ).fetchone()

    fresh_event = publish(1)
    fresh_before = durable("approval-fresh")
    old_event = publish(0)  # Deliberately delivered after fresh work settles.
    assert old_event["delegation_id"] == "approval-old"
    assert fresh_event["delegation_id"] == "approval-fresh"
    assert durable("approval-fresh") == fresh_before
    assert fresh_before[0] == entries[1]["status"]
    assert durable("approval-old")[0] == "failed"
    persisted = json.loads(durable("approval-old")[1])
    assert (persisted["results"][0] if batch else persisted)["approval_outcome"] == old_outcome
    old_notice = _format_async_delegation(old_event)
    fresh_notice = _format_async_delegation(fresh_event)
    assert "approval-old" in old_notice and "approval-fresh" not in old_notice
    assert APPROVAL_OUTCOME_NOTICES[old_outcome] in old_notice
    assert "approval-fresh" in fresh_notice
    assert APPROVAL_OUTCOME_NOTICES[old_outcome] not in fresh_notice
