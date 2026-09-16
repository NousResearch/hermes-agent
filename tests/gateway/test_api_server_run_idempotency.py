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
