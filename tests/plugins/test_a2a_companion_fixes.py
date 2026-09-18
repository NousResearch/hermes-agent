"""Regression coverage for PR #92494 review follow-ups."""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol, security
from plugins.platforms.a2a.adapter import A2AAdapter


def _record(task_id: str, state: str, *, created_at: float | None = None) -> dict:
    return {
        "task_id": task_id,
        "context_id": f"ctx-{task_id}",
        "peer": "peer-a",
        "agent_slug": "",
        "tenant": "",
        "state": state,
        "reply": "",
        "created_at": time.time() if created_at is None else created_at,
        "created_iso": protocol.now_iso(),
        "push_url": "",
        "push_config_id": "",
    }


def test_pending_completion_commits_safe_semantics_and_effects_once(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    ledger = tmp_path / "a2a_task_ledger.json"
    record = _record("task-safe", protocol.STATE_WORKING)
    assert adapter.tasks.publish_durable(ledger, "task-safe", record).published
    future = adapter._add_pending("task-safe", "ctx-task-safe")

    persisted: list[tuple] = []
    audited: list[tuple] = []
    pushed: list[tuple] = []
    monkeypatch.setattr(protocol, "persist_message", lambda *args, **kwargs: persisted.append(args))
    monkeypatch.setattr(security, "audit", lambda *args, **kwargs: audited.append(args))
    monkeypatch.setattr(adapter, "_send_push_notification", lambda *args: pushed.append(args))
    completed_before = protocol.metrics.tasks_completed
    outbound_before = protocol.metrics.outbound_total

    raw_reply = "[INPUT_REQUIRED] contact secret@example.com"
    ok, error = adapter._durable_complete_pending(
        "task-safe", "ctx-task-safe", raw_reply, "message-1"
    )

    assert ok, error
    state, reply = future.result(timeout=0)
    assert state == protocol.STATE_INPUT_REQUIRED
    assert reply == "contact [redacted-email]"
    durable = adapter.tasks.get("task-safe")
    assert durable is not None
    assert durable["state"] == state
    assert durable["reply"] == reply

    pending = {
        "task_id": "task-safe",
        "context_id": "ctx-task-safe",
        "peer": "peer-a",
        "started": record["created_at"],
        "created_iso": record["created_iso"],
    }
    assert adapter._finalize_task(pending, state, reply) == (state, reply)

    assert len(persisted) == 1
    assert len(audited) == 1
    assert pushed == [("task-safe", "ctx-task-safe", reply, state)]
    assert protocol.metrics.tasks_completed == completed_before + 1
    assert protocol.metrics.outbound_total == outbound_before + 1


def test_durable_snapshot_keeps_live_tasks_and_caps_terminal_history(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "a2a_task_ledger.json"
    store = protocol.TaskStore()
    records: OrderedDict[str, dict] = OrderedDict()
    records["live"] = _record(
        "live", protocol.STATE_WORKING, created_at=time.time() - 86_400
    )
    for index in range(protocol.TaskStore._MAX_TERMINAL + 1):
        task_id = f"done-{index:03d}"
        records[task_id] = _record(task_id, protocol.STATE_COMPLETED)
    store._tasks = records

    store.persist(ledger)
    snapshot = json.loads(ledger.read_text())

    assert "live" in snapshot
    terminal_ids = [
        task_id
        for task_id, record in snapshot.items()
        if record["state"] in protocol.TERMINAL_STATES
    ]
    assert len(terminal_ids) == protocol.TaskStore._MAX_TERMINAL
    assert "done-000" not in snapshot
    assert f"done-{protocol.TaskStore._MAX_TERMINAL:03d}" in snapshot


def test_publish_durable_uses_the_same_retention_policy(tmp_path: Path) -> None:
    ledger = tmp_path / "a2a_task_ledger.json"
    store = protocol.TaskStore()
    records: OrderedDict[str, dict] = OrderedDict()
    records["live"] = _record(
        "live", protocol.STATE_WORKING, created_at=time.time() - 86_400
    )
    for index in range(protocol.TaskStore._MAX_TERMINAL + 1):
        task_id = f"done-{index:03d}"
        records[task_id] = _record(task_id, protocol.STATE_COMPLETED)
    store._tasks = records

    candidate = _record("new", protocol.STATE_COMPLETED)
    outcome = store.publish_durable(ledger, "new", candidate)
    assert outcome.published
    snapshot = json.loads(ledger.read_text())

    assert "live" in snapshot
    terminal_ids = [
        task_id
        for task_id, record in snapshot.items()
        if record["state"] in protocol.TERMINAL_STATES
    ]
    assert len(terminal_ids) == protocol.TaskStore._MAX_TERMINAL
    assert "done-000" not in snapshot
    assert "new" in snapshot


def test_late_completion_is_retained_when_terminal_history_is_full(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "a2a_task_ledger.json"
    store = protocol.TaskStore()
    started = time.time() - 1_000
    records: OrderedDict[str, dict] = OrderedDict()
    records["old-live"] = _record(
        "old-live", protocol.STATE_WORKING, created_at=started
    )
    for index in range(protocol.TaskStore._MAX_TERMINAL):
        task_id = f"done-{index:03d}"
        record = _record(task_id, protocol.STATE_COMPLETED)
        record["completed_at"] = started + index
        records[task_id] = record
    store._tasks = records

    completed = dict(records["old-live"])
    completed["state"] = protocol.STATE_COMPLETED
    completed["reply"] = "late result"
    completed["completed_at"] = time.time()
    outcome = store.publish_durable(ledger, "old-live", completed)

    assert outcome.published
    snapshot = json.loads(ledger.read_text())
    assert snapshot["old-live"]["reply"] == "late result"
    assert store.get("old-live") is not None


def test_watchdog_does_not_replace_a_concurrent_terminal_publication(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ledger = tmp_path / "a2a_task_ledger.json"
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    other = protocol.TaskStore()

    orphan = _record("orphan", protocol.STATE_WORKING)
    shared = _record("shared", protocol.STATE_WORKING)
    assert adapter.tasks.publish_durable(ledger, "orphan", orphan).published
    assert other.publish_durable(ledger, "shared", shared).published
    adapter.tasks._tasks["orphan"]["created_at"] = time.time() - 1_000

    original_fail = adapter._fail_orphans_once

    def fail_then_complete_other() -> list[str]:
        failed = original_fail()
        current = other.get("shared")
        assert current is not None
        completed = dict(current)
        completed["state"] = protocol.STATE_COMPLETED
        completed["reply"] = "finished elsewhere"
        completed["completed_at"] = time.time()
        assert other.publish_durable(ledger, "shared", completed).published
        return failed

    monkeypatch.setattr(adapter, "_fail_orphans_once", fail_then_complete_other)

    calls = 0

    def wait(_timeout: float) -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setattr(adapter._watchdog_stop, "wait", wait)
    adapter._watchdog_loop()

    snapshot = json.loads(ledger.read_text())
    assert snapshot["orphan"]["state"] == protocol.STATE_FAILED
    assert snapshot["shared"]["state"] == protocol.STATE_COMPLETED
    assert snapshot["shared"]["reply"] == "finished elsewhere"
