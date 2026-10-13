"""Room rejection redaction and asynchronous idempotency fingerprinting."""

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms import api_server, api_server_room_dispatch, api_server_runs


def test_room_dispatch_errors_never_expose_local_paths():
    message, code = api_server_room_dispatch._public_dispatch_error(
        RuntimeError("failed at /Users/private/.hermes/state.db"))
    assert message == "Room dispatch was rejected."
    assert code == "invalid_room_dispatch"
    assert "/Users/private" not in message


@pytest.mark.asyncio
async def test_run_fingerprint_yields_to_loop_and_replays_legacy_hash(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "idem.db"))
    body = {"input": [{"role": "user", "content": [
        {"type": "text", "text": "caf\u00e9\n\"quoted\""},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 100_000}},
    ]}]}
    request = SimpleNamespace(
        headers={"Idempotency-Key": "legacy-image"}, json=AsyncMock(return_value=body))
    scope = adapter._run_idempotency_scope(request)
    encoded = json.dumps({"body": body, "gateway_session_key": ""},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    adapter._run_idempotency_store.reserve(
        scope, "legacy-image", hashlib.sha256(encoded).hexdigest(), "run_original",
        {"run_id": "run_original", "status": "completed"})
    loop = asyncio.get_running_loop()
    responsive = []
    dumps, sha256 = json.dumps, hashlib.sha256

    def checked_dumps(value, *args, **kwargs):
        if isinstance(value, dict) and "gateway_session_key" in value and "body" in value:
            released = threading.Event()
            loop.call_soon_threadsafe(released.set)
            responsive.append(released.wait(2))
        return dumps(value, *args, **kwargs)

    def checked_sha256(value=b"", *args, **kwargs):
        if value == encoded:
            released = threading.Event()
            loop.call_soon_threadsafe(released.set)
            responsive.append(released.wait(2))
        return sha256(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", checked_dumps)
    monkeypatch.setattr(hashlib, "sha256", checked_sha256)
    try:
        response = await api_server_runs._handle_runs(adapter, request, _api_server=api_server)
        assert response.status == 202
        assert json.loads(response.text)["run_id"] == "run_original"
        assert response.headers["Idempotency-Replayed"] == "true"
        assert responsive == [True, True]
        assert not adapter._active_run_tasks
    finally:
        adapter._run_idempotency_store.close()
