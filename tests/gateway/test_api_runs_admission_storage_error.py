"""POST /v1/runs whose admission write loses the state.db writer answers a retryable 503 and
keeps no reservation: an exact Idempotency-Key retry then runs the work, never replays a ghost."""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from hermes_state import SessionDB
from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _adapter, _runs_client


@pytest.mark.asyncio
async def test_locked_admission_write_is_503_and_exact_retry_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = SessionDB(tmp_path / "state.db")
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()["result"] = {"final_response": "ACK", "messages": []}
        return "ACK"

    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False, _handle_message=answer)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="lock-owner")
    adapter = _adapter(runner, db)
    runner._adapter_for_source = lambda source: adapter
    monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.3)
    client = _runs_client(adapter)
    await client.start_server()
    headers = {**_AUTH_HEADERS, "Idempotency-Key": "locked-run"}
    body = {"input": "LOCKED", "session_id": "api-session"}
    # Another process holds the state.db writer past this owner's write patience.
    holder = sqlite3.connect(tmp_path / "state.db", isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        refused = await client.post("/v1/runs", json=body, headers=headers)
        assert refused.status == 503, await refused.text()
        assert (await refused.json())["error"]["code"] == "storage_unavailable"
        assert refused.headers.get("Retry-After")
        assert not adapter._run_statuses and not adapter._run_owners and not adapter._run_idempotency_ids
        holder.execute("ROLLBACK")
        retry = await client.post("/v1/runs", json=body, headers=headers)
        assert retry.status == 202, await retry.text()
        assert (await retry.json())["replayed"] is False
        await asyncio.wait_for(asyncio.gather(*adapter._active_run_tasks.values()), 10)
        assert executed == ["LOCKED"]
    finally:
        holder.close()
        for task in adapter._active_run_tasks.values():
            task.cancel()
        await client.close()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        authority.db.close()
        store.close_all_db_handles()
