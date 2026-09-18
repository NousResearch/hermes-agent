"""Extraction seams and lifecycle coverage for API run idempotency."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


@pytest.mark.asyncio
async def test_api_server_constructor_uses_module_run_store_binding(monkeypatch):
    store = MagicMock()
    store_factory = MagicMock(return_value=store)
    monkeypatch.setattr(api_server, "RunIdempotencyStore", store_factory)

    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True))
    try:
        store_factory.assert_called_once_with()
        assert adapter._run_idempotency_store is store
    finally:
        await adapter.disconnect()
    store.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_disconnect_tolerates_bare_fixture_without_run_idempotency_store():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter.platform = Platform.API_SERVER
    adapter._mark_disconnected = MagicMock()
    adapter._close_cached_session_dbs = MagicMock()
    adapter._response_store = MagicMock()
    adapter._site = None
    adapter._runner = None
    adapter._app = object()

    assert not hasattr(adapter, "_run_idempotency_store")
    await adapter.disconnect()

    adapter._mark_disconnected.assert_called_once_with()
    adapter._response_store.close.assert_called_once_with()
    adapter._close_cached_session_dbs.assert_called_once_with()
    assert adapter._app is None


def _reserve(store: RunIdempotencyStore, scope: str = "scope-a", run_id: str = "run-a") -> None:
    store.reserve(
        scope,
        "key-a",
        "fingerprint-a",
        run_id,
        {"run_id": run_id, "status": "running"},
    )


def test_run_events_replay_after_store_reopen_without_duplicates(tmp_path):
    """Contract: an idempotent run's event cursor survives gateway replacement.

    Replaying from the first cursor returns only later events, while replaying
    from the tail accepts the valid empty result.
    """
    path = tmp_path / "runs.db"
    first = RunIdempotencyStore(str(path))
    _reserve(first)
    one = first.append_event("run-a", {"event": "message.delta", "delta": "one"})
    two = first.append_event("run-a", {"event": "run.completed", "output": "done"})
    first.close()

    reopened = RunIdempotencyStore(str(path))
    try:
        assert one["sequence"] == 1
        assert two["sequence"] == 2
        assert reopened.events_after("scope-a", "run-a", one["sequence"]) == [two]
        assert reopened.events_after("scope-a", "run-a", two["sequence"]) == []
        assert reopened.events_after("scope-b", "run-a", 0) == []
    finally:
        reopened.close()


def test_pending_approval_and_resolution_receipt_survive_store_reopen(tmp_path):
    """Contract: a pending approval and its exact decision receipt are durable.

    Retrying the same decision is accepted as replay; a conflicting decision
    is rejected and cannot rewrite the receipt.
    """
    path = tmp_path / "runs.db"
    first = RunIdempotencyStore(str(path))
    _reserve(first)
    first.save_approval_request(
        "run-a",
        {"request_id": "approval-a", "command": "safe-redacted-command"},
    )
    first.close()

    reopened = RunIdempotencyStore(str(path))
    try:
        pending = reopened.pending_approval("scope-a", "run-a", "approval-a")
        assert pending == {
            "run_id": "run-a",
            "request_id": "approval-a",
            "request": {"request_id": "approval-a", "command": "safe-redacted-command"},
            "state": "pending",
        }
        outcome, receipt = reopened.resolve_approval(
            "scope-a",
            "run-a",
            "approval-a",
            "deny",
            applied=False,
            resolved=0,
        )
        assert outcome == "created"
        assert receipt == {
            "run_id": "run-a",
            "request_id": "approval-a",
            "choice": "deny",
            "applied": False,
            "resolved": 0,
        }
        assert reopened.resolve_approval(
            "scope-a", "run-a", "approval-a", "deny", applied=False, resolved=0
        ) == ("replayed", receipt)
        conflict, unchanged = reopened.resolve_approval(
            "scope-a", "run-a", "approval-a", "once", applied=False, resolved=0
        )
        assert conflict == "conflict"
        assert unchanged == receipt
    finally:
        reopened.close()


def test_approval_request_failure_rolls_back_waiting_state_and_partial_rows(tmp_path, monkeypatch):
    """A crash-shaped failure cannot expose a wait state without its request."""
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    _reserve(store)

    def fail_event(*_args, **_kwargs):
        raise RuntimeError("simulated process loss before commit")

    monkeypatch.setattr(store, "_append_event_locked", fail_event)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        store.record_approval_request(
            "run-a",
            {"run_id": "run-a", "status": "waiting_for_approval"},
            {"event": "approval.request", "request_id": "approval-a"},
            tool={
                "tool_call_id": "call-a",
                "tool_name": "terminal",
                "tool_args": {"command": "printf safe"},
            },
        )

    assert store.status_for_run("scope-a", "run-a")["status"]["status"] == "running"
    assert store.approval_for_run("scope-a", "run-a", "approval-a") is None
    store.close()


def test_recovery_plan_binds_frozen_tool_and_creates_one_successor(tmp_path):
    """A decided approval creates one durable successor with the exact tool input."""
    path = tmp_path / "runs.db"
    store = RunIdempotencyStore(str(path))
    _reserve(store)
    store.save_run_launch(
        "run-a",
        {
            "session_id": "session-a",
            "model": "provider/model-a",
            "requested_model": "provider/model-a",
        },
    )
    store.record_approval_request(
        "run-a",
        {"run_id": "run-a", "status": "waiting_for_approval"},
        {
            "event": "approval.request",
            "request_id": "approval-a",
            "command": "printf '<redacted>'",
        },
        tool={
            "tool_call_id": "call-a",
            "tool_name": "terminal",
            "tool_args": {"command": "printf '雪'"},
        },
    )
    outcome, _ = store.resolve_approval(
        "scope-a", "run-a", "approval-a", "once", applied=False, resolved=0
    )
    assert outcome == "created"

    first = store.reserve_recovery_successor(
        "scope-a",
        "run-a",
        successor_run_id="run-successor",
        owner_pid=101,
        owner_started=202,
    )
    replay = store.reserve_recovery_successor(
        "scope-a",
        "run-a",
        successor_run_id="run-other",
        owner_pid=303,
        owner_started=404,
    )

    assert first["created"] is True
    assert replay["created"] is False
    assert replay["successor_run_id"] == "run-successor"
    assert first["tool"] == {
        "tool_call_id": "call-a",
        "tool_name": "terminal",
        "tool_args": {"command": "printf '雪'"},
    }
    assert first["decision"] == {"request_id": "approval-a", "choice": "once"}
    assert first["launch"]["session_id"] == "session-a"
    assert store.status_for_run("scope-a", "run-a")["status"]["status"] == "superseded"
    assert store.status_for_run("scope-a", "run-successor")["status"]["status"] == "recovery_pending"
    store.close()


def test_recovery_refuses_to_repeat_uncertain_tool_effect(tmp_path):
    """Once native application starts, a missing result is intervention work."""
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    _reserve(store)
    store.save_run_launch("run-a", {"session_id": "session-a", "model": "m"})
    store.record_approval_request(
        "run-a",
        {"run_id": "run-a", "status": "waiting_for_approval"},
        {"event": "approval.request", "request_id": "approval-a", "command": "danger"},
        tool={"tool_call_id": "call-a", "tool_name": "terminal", "tool_args": {"command": "danger"}},
    )
    store.resolve_approval("scope-a", "run-a", "approval-a", "once", applied=False, resolved=0)
    assert store.mark_approval_applied("run-a", "approval-a", resolved=1)
    assert store.mark_tool_dispatching("run-a", "approval-a")

    recovery = store.reserve_recovery_successor(
        "scope-a",
        "run-a",
        successor_run_id="run-successor",
        owner_pid=101,
        owner_started=202,
    )

    assert recovery["created"] is False
    assert recovery["state"] == "unrecoverable"
    status = store.status_for_run("scope-a", "run-a")["status"]
    assert status["status"] == "unrecoverable"
    assert status["intervention_reason"] == "tool_effect_uncertain"
    assert [event["event"] for event in store.events_after("scope-a", "run-a", 0)][-1:] == [
        "run.unrecoverable"
    ]
    repeated = store.reserve_recovery_successor(
        "scope-a",
        "run-a",
        successor_run_id="run-other",
        owner_pid=303,
        owner_started=404,
    )
    assert repeated["state"] == "unrecoverable"
    assert [
        event["event"] for event in store.events_after("scope-a", "run-a", 0)
        if event["event"] == "run.unrecoverable"
    ] == ["run.unrecoverable"]
    store.close()


def test_completed_tool_receipt_is_reused_after_restart(tmp_path):
    """A committed result closes the dispatch window and is returned verbatim."""
    path = tmp_path / "runs.db"
    first = RunIdempotencyStore(str(path))
    _reserve(first)
    first.save_run_launch("run-a", {"session_id": "session-a", "model": "m"})
    first.record_approval_request(
        "run-a",
        {"run_id": "run-a", "status": "waiting_for_approval"},
        {"event": "approval.request", "request_id": "approval-a", "command": "write marker"},
        tool={"tool_call_id": "call-a", "tool_name": "terminal", "tool_args": {"command": "write marker"}},
    )
    first.resolve_approval("scope-a", "run-a", "approval-a", "once", applied=True, resolved=1)
    first.mark_tool_dispatching("run-a", "approval-a")
    first.mark_tool_completed("run-a", "call-a", {"output": "marker", "exit_code": 0})
    first.close()

    reopened = RunIdempotencyStore(str(path))
    recovery = reopened.reserve_recovery_successor(
        "scope-a",
        "run-a",
        successor_run_id="run-successor",
        owner_pid=101,
        owner_started=202,
    )

    assert recovery["created"] is True
    assert recovery["state"] == "tool_completed"
    assert recovery["tool_result"] == {"output": "marker", "exit_code": 0}
    reopened.close()
