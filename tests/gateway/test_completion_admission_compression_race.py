"""Regression for the compression-after-admission result-loss window in #136225."""

import asyncio
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from tools import async_delegation as ad


class _QueuedTransport(BasePlatformAdapter):
    """Use the production adapter's admission and busy queue, without external I/O."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("the queued completion must not invoke external transport")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route_change,transient_read,expected",
    [
        ("compression", True, "recoverable"),
        ("compression", False, "tip"),
        ("new", False, None),
        ("resume", False, None),
        ("unchanged", True, "owner"),
        ("unchanged", False, "owner"),
    ],
)
async def test_queued_completion_after_admission_preserves_result_or_user_boundary(
    tmp_path, monkeypatch, private_db_probe_cleanup, route_change, transient_read, expected,
):
    """A completed SQLite claim cannot strand a queued turn behind a transient lineage read."""
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        adapter = _QueuedTransport()
        runner.adapters[Platform.TELEGRAM] = adapter
        try:
            entry = store.get_or_create_session(SessionSource(
                platform=Platform.TELEGRAM, chat_id="post-admission", chat_type="dm",
            ))
            owner = entry.session_id
            db = cast(Any, store._db)
            db.create_session("worker", source="subagent", model_config={"_delegate_from": owner})

            async def unused_handler(event):
                raise AssertionError("the busy guard must keep the completion queued")

            adapter.set_message_handler(unused_handler)
            adapter._active_sessions[entry.session_key] = asyncio.Event()
            event = {
                "type": "async_delegation", "delegation_id": "queued-owner",
                "session_key": entry.session_key, "parent_session_id": "worker",
                "dispatched_at": 1.0, "status": "completed", "summary": "completed result",
            }
            ad._persist_dispatch(event)
            ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})

            assert await runner._deliver_completion_notification("completed result", event) is True
            accepted = ad.get_durable_delegation(event["delegation_id"])
            assert accepted is not None and accepted["delivery_state"] == "delivered"
            queued = adapter.get_pending_message(entry.session_key)
            assert queued is not None and queued.internal
            assert queued.metadata["gateway_session_id"] == owner
            assert queued.metadata["gateway_proven_session_id"] == owner

            if route_change == "compression":
                db.create_session("tip", source="telegram", parent_session_id=owner)
                db.end_session(owner, end_reason="compression")
                assert store.advance_compression_session(entry.session_key, owner, "tip") is not None
            elif route_change == "new":
                assert store.reset_session(entry.session_key) is not None
            elif route_change == "resume":
                db.create_session("other", source="telegram")
                assert store.switch_session(entry.session_key, "other", expected_session_id=owner) is not None

            original_get = runner._session_db.get_session
            fault_reads = []

            async def transient_owner_read(sid):
                if transient_read and sid == owner:
                    fault_reads.append(sid)
                    raise RuntimeError("transient SQLite session-row read failure")
                return await original_get(sid)

            monkeypatch.setattr(runner._session_db, "get_session", transient_owner_read)
            resolved = await runner._hmwa_resolve_session(queued, queued.source)
            delivered = ad.get_durable_delegation(event["delegation_id"])
            assert delivered is not None
            if expected == "recoverable":
                assert fault_reads == [owner]
                # Either consume into B, or restore the durable completion to a retryable state.
                assert (resolved is not None and resolved[1].session_id == "tip") or (
                    delivered["delivery_state"] == "pending"
                ), "queued completion vanished after admission while its SQLite row stayed delivered"
            elif expected is None:
                assert resolved is None
                assert delivered["delivery_state"] == "delivered"
            else:
                target = owner if expected == "owner" else expected
                assert resolved is not None and resolved[1].session_id == target
                assert delivered["delivery_state"] == "delivered"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()
