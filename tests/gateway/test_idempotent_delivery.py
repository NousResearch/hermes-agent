"""A ledgered reply keeps its identity and body after an ambiguous send."""

import sqlite3
from contextlib import closing
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _Adapter(BasePlatformAdapter):
    def __init__(self, *, idempotent=False):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.idempotent_delivery = idempotent
        self.send = AsyncMock(return_value=SendResult(success=True, message_id="accepted"))

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    runner.session_store = None
    runner._async_session_store = MagicMock()
    runner._async_session_store.clear_resume_pending = AsyncMock()
    runner._async_session_store._store = None
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("idempotent", [False, True])
@pytest.mark.parametrize("recovery", ["restart", "reconnect"])
async def test_accepted_reply_is_deduplicated_after_ack_loss(tmp_path, monkeypatch, idempotent, recovery):
    """Real producer and recovery paths, with a durable fake receiving endpoint."""
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "ledger.db")
    endpoint = tmp_path / "endpoint.db"
    with closing(sqlite3.connect(endpoint)) as conn, conn:
        conn.execute("CREATE TABLE messages (delivery_id TEXT PRIMARY KEY, chat TEXT, thread TEXT, body TEXT)")

    def transport(*, lose_ack):
        async def send(chat_id, content, reply_to=None, metadata=None):
            metadata = metadata or {}
            delivery_id = metadata.get("_delivery_obligation_id") if idempotent else None
            if idempotent:
                assert delivery_id, "the ledger identity must reach the adapter"
            payload = (chat_id, metadata.get("thread_id"), content)
            with closing(sqlite3.connect(endpoint)) as conn, conn:
                existing = conn.execute(
                    "SELECT chat, thread, body FROM messages WHERE delivery_id=?", (delivery_id,)
                ).fetchone()
                if existing is None:
                    conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?)", (delivery_id, *payload))
                else:
                    assert existing == payload, "one delivery ID cannot address different content"
            if lose_ack:
                raise ConnectionError("accepted by endpoint, acknowledgement lost")
            return SendResult(success=True, message_id=delivery_id or "accepted")

        return send

    session_key = "agent:main:slack:channel:C1:thread:T1"
    event = MessageEvent(
        text="hello", message_id="inbound-1",
        source=SessionSource(platform=Platform.SLACK, chat_id="C1", thread_id="T1", chat_type="channel"),
    )
    content = "The completed answer."
    metadata = {"thread_id": "T1", "_delivery_obligation_id": "untrusted-caller-value"}
    adapter = _Adapter(idempotent=idempotent)
    adapter.send.side_effect = transport(lose_ack=True)
    with pytest.raises(ConnectionError, match="acknowledgement lost"):
        await adapter.send_final_ledgered(event, session_key, content, metadata, reply_to=event.message_id)
    assert metadata["_delivery_obligation_id"] == "untrusted-caller-value"
    obligation_id = dl.compute_obligation_id(session_key, event.message_id, content)
    if idempotent:
        assert adapter.send.call_args.kwargs["metadata"]["_delivery_obligation_id"] == obligation_id

    # No adapter state survives: the endpoint and ledger are the only durable state.
    replacement = _Adapter(idempotent=idempotent)
    replacement.send.side_effect = transport(lose_ack=False)
    runner = _runner(replacement)
    if recovery == "restart":
        with dl._connect() as conn:
            conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1")
        assert await runner._redeliver_pending_obligations() == 1
    else:
        dl.mark_failed(obligation_id, "send_path_degraded")
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1

    with closing(sqlite3.connect(endpoint)) as conn:
        messages = conn.execute("SELECT delivery_id, chat, thread, body FROM messages").fetchall()
    if idempotent:
        assert messages == [(obligation_id, "C1", "T1", content)]
    else:
        assert len(messages) == 2
        assert messages[0][3] == content
        assert messages[1][3] != content  # The legacy ambiguity marker is retained.
    runner._async_session_store.clear_resume_pending.assert_awaited_once_with(session_key)
    with dl._connect() as conn:
        assert conn.execute("SELECT state FROM delivery_obligations").fetchone() == ("delivered",)


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [
    [SendResult(success=False, error="bad formatting")],
    [SendResult(success=False, error="connection reset", retryable=True)] * 3,
    [SendResult(success=False, error="connection reset", retryable=True),
     SendResult(success=False, error="bad formatting")],
])
async def test_retry_failure_never_changes_content_under_one_delivery_id(failures):
    adapter = _Adapter(idempotent=True)
    adapter.send.side_effect = [*failures, SendResult(success=True, message_id="wrong-fallback")]
    content = "The full answer. " * 300
    metadata = {"thread_id": "T1", "_delivery_obligation_id": "durable-obligation"}
    with patch("gateway.platforms.base.asyncio.sleep", new_callable=AsyncMock):
        result = await adapter._send_with_retry("C1", content, metadata=metadata)

    assert result is failures[-1]
    assert len(adapter.send.call_args_list) == len(failures)
    assert all(call.kwargs["content"] == content for call in adapter.send.call_args_list)
    assert all(call.kwargs["metadata"] == metadata for call in adapter.send.call_args_list)
