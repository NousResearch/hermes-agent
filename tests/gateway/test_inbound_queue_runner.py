"""GatewayRunner integration tests for the durable inbound queue."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.inbound_queue import GatewayInboxStore, INBOX_METADATA_KEY
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _CapturingAdapter:
    def __init__(self) -> None:
        self._active_sessions: dict[str, asyncio.Event] = {}
        self.started: list[tuple[str, MessageEvent]] = []

    def _start_session_processing(self, event: MessageEvent, session_key: str) -> bool:
        self._active_sessions[session_key] = asyncio.Event()
        self.started.append((session_key, event))
        return True


def _event(message_id: str, chat_id: str) -> MessageEvent:
    return MessageEvent(
        text=f"prompt {message_id}",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id=chat_id,
            chat_type="dm",
            user_id=f"user-{chat_id}",
        ),
        message_id=message_id,
    )


def _runner(tmp_path, *, max_concurrent_sessions: int | None = None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")},
        max_concurrent_sessions=max_concurrent_sessions,
    )
    runner._inbox_store = GatewayInboxStore(hermes_home=tmp_path)
    runner._inbox_wakeup = None
    runner._running = True
    runner._running_agents = {}
    runner._profile_adapters = {}
    runner.adapters = {Platform.DISCORD: _CapturingAdapter()}
    return runner


def test_runner_queue_helpers_preserve_claim_lifecycle(tmp_path):
    runner = _runner(tmp_path)
    event = _event("message-1", "chat-a")

    async def exercise() -> None:
        enqueued = await runner._inbox_enqueue_event(
            event,
            "agent:main:discord:dm:chat-a",
            origin="direct",
        )
        assert enqueued is not None and enqueued.accepted and enqueued.inserted
        marker = event.metadata[INBOX_METADATA_KEY]
        assert marker["queue_id"] == enqueued.row.queue_id
        assert marker["trigger_identity"] == "message-1"

        claimed = await runner._inbox_claim_event(event)
        assert claimed is not None and claimed.state == "claimed"
        assert marker["claim_token"] == claimed.claim_token

        assert await runner._inbox_bind_event(event, "session-db-a") is True
        assert runner._inbox_store.get(claimed.queue_id).session_id == "session-db-a"

        assert await runner._inbox_retry_event(event, "transient setup failure") is True
        assert runner._inbox_store.get(claimed.queue_id).state == "queued"

        claimed_again = await runner._inbox_claim_event(event)
        assert claimed_again is not None
        assert claimed_again.claim_token != claimed.claim_token
        assert await runner._inbox_complete_event(event) is True
        assert runner._inbox_store.get(claimed.queue_id).state == "completed"

    asyncio.run(exercise())


def test_dispatcher_claims_fairly_across_sessions(tmp_path):
    runner = _runner(tmp_path)
    a1 = _event("a1", "chat-a")
    a2 = _event("a2", "chat-a")
    b1 = _event("b1", "chat-b")

    async def exercise() -> None:
        await runner._inbox_enqueue_event(a1, "session-a", origin="busy")
        await runner._inbox_enqueue_event(a2, "session-a", origin="busy")
        await runner._inbox_enqueue_event(b1, "session-b", origin="busy")

        assert await runner._dispatch_one_inbox_event() is True
        assert await runner._dispatch_one_inbox_event() is True

    asyncio.run(exercise())

    adapter = runner.adapters[Platform.DISCORD]
    assert [key for key, _event_value in adapter.started] == ["session-a", "session-b"]
    assert runner._inbox_store.get(
        a1.metadata[INBOX_METADATA_KEY]["queue_id"]
    ).state == "claimed"
    assert runner._inbox_store.get(
        a2.metadata[INBOX_METADATA_KEY]["queue_id"]
    ).state == "queued"


def test_dispatcher_leaves_rows_queued_at_global_session_capacity(tmp_path):
    runner = _runner(tmp_path, max_concurrent_sessions=1)
    runner._running_agents["already-running"] = MagicMock()
    event = _event("queued", "chat-c")

    async def exercise() -> None:
        await runner._inbox_enqueue_event(event, "session-c", origin="direct")
        assert await runner._dispatch_one_inbox_event() is False

    asyncio.run(exercise())

    queue_id = event.metadata[INBOX_METADATA_KEY]["queue_id"]
    assert runner._inbox_store.get(queue_id).state == "queued"
    assert runner.adapters[Platform.DISCORD].started == []


def test_dispatcher_preserves_durable_resume_control_marker(tmp_path):
    """A durable trigger is resumed without replaying its text, but the
    claimed control event must retain the typed resume marker so the agent
    continues the recorded task rather than treating it as an empty request.
    """
    runner = _runner(tmp_path)
    runner.session_store = MagicMock()
    runner.session_store.mark_resume_pending.return_value = True
    event = _event("durable-resume", "chat-resume")

    async def exercise() -> None:
        await runner._inbox_enqueue_event(event, "session-resume", origin="direct")
        claimed = await runner._inbox_claim_event(event)
        assert claimed is not None
        assert await runner._inbox_retry_event(
            event,
            "gateway stopped after trigger persistence",
            resume_only=True,
        )
        assert await runner._dispatch_one_inbox_event() is True

    asyncio.run(exercise())

    adapter = runner.adapters[Platform.DISCORD]
    assert len(adapter.started) == 1
    session_key, resumed_event = adapter.started[0]
    assert session_key == "session-resume"
    assert resumed_event.text == ""
    assert resumed_event.internal is True
    assert resumed_event.metadata[INBOX_METADATA_KEY]["resume_only"] is True
    runner.session_store.mark_resume_pending.assert_called_once_with(
        "session-resume", "restart_interrupted"
    )
