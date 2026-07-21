from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import WebSocketRelayTransport
from gateway.session import SessionSource
from plugins.platforms.slack.ingress import (
    FollowStore,
    SlackIngressPolicy,
    SlackIngressServer,
)


class _FakeSlackAdapter:
    def __init__(self) -> None:
        self.admission_handler = None
        self.reaction_handler = None
        self.dispatch_reaction_trigger = AsyncMock()
        self.message_handler = None
        self.sent: list[dict] = []
        self.connected = False
        self.connect_calls = 0
        self.connected_event = asyncio.Event()
        self.disconnected_event = asyncio.Event()

    def set_external_admission_handler(self, handler) -> None:
        self.admission_handler = handler

    def set_external_reaction_handler(self, handler) -> None:
        self.reaction_handler = handler

    def set_message_handler(self, handler) -> None:
        self.message_handler = handler

    async def connect(self) -> bool:
        self.connect_calls += 1
        self.connected = True
        self.disconnected_event.clear()
        self.connected_event.set()
        return True

    async def disconnect(self) -> None:
        self.connected = False
        self.connected_event.clear()
        self.disconnected_event.set()

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": dict(metadata or {}),
        })
        return SendResult(success=True, message_id="200.1")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"name": "general", "type": "group"}


def test_default_sidecar_lock_is_shared_across_profiles(tmp_path, monkeypatch):
    profile_home = tmp_path / "hermes-root" / "profiles" / "work"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    sidecar = SlackIngressServer(
        _FakeSlackAdapter(),
        SlackIngressPolicy(
            FollowStore(tmp_path / "state.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
    )

    assert sidecar._instance_lock.path == (
        tmp_path / "hermes-root" / "slack-ingress.lock"
    )


@pytest.mark.asyncio
async def test_cancelled_start_releases_machine_lock(tmp_path, monkeypatch):
    import plugins.platforms.slack.ingress as ingress_module

    lock_path = tmp_path / "slack-ingress.lock"
    first = SlackIngressServer(
        _FakeSlackAdapter(),
        SlackIngressPolicy(
            FollowStore(tmp_path / "first.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
        lock_path=lock_path,
    )
    second = SlackIngressServer(
        _FakeSlackAdapter(),
        SlackIngressPolicy(
            FollowStore(tmp_path / "second.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
        lock_path=lock_path,
    )
    real_serve = ingress_module.websockets.serve
    monkeypatch.setattr(
        ingress_module.websockets,
        "serve",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await first.start()

    monkeypatch.setattr(ingress_module.websockets, "serve", real_serve)
    await second.start()
    await second.stop()


@pytest.mark.asyncio
async def test_only_one_sidecar_can_own_slack_for_a_profile(tmp_path):
    lock_path = tmp_path / "slack-ingress.lock"
    first = SlackIngressServer(
        _FakeSlackAdapter(),
        SlackIngressPolicy(
            FollowStore(tmp_path / "first.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
        lock_path=lock_path,
    )
    second = SlackIngressServer(
        _FakeSlackAdapter(),
        SlackIngressPolicy(
            FollowStore(tmp_path / "second.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
        lock_path=lock_path,
    )

    await first.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await second.start()
    finally:
        await first.stop()

    await second.start()
    await second.stop()


@pytest.mark.asyncio
async def test_sidecar_closes_listener_and_releases_lock_when_slack_disconnect_fails(
    tmp_path,
):
    lock_path = tmp_path / "slack-ingress.lock"
    slack = _FakeSlackAdapter()
    first = SlackIngressServer(
        slack,
        SlackIngressPolicy(
            FollowStore(tmp_path / "first.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
        lock_path=lock_path,
    )
    second = SlackIngressServer(
        _FakeSlackAdapter(),
        SlackIngressPolicy(
            FollowStore(tmp_path / "second.db", ttl_seconds=60, max_threads=100)
        ),
        host="127.0.0.1",
        port=0,
        lock_path=lock_path,
    )

    await first.start()
    first._slack_connected = True
    slack.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

    with pytest.raises(RuntimeError, match="disconnect failed"):
        await first.stop()
    assert first._server is None

    await second.start()
    await second.stop()


@pytest.mark.asyncio
async def test_slack_ingress_round_trips_over_production_relay(tmp_path):
    slack = _FakeSlackAdapter()
    store = FollowStore(
        tmp_path / "ingress.db",
        ttl_seconds=30 * 24 * 60 * 60,
        max_threads=10_000,
    )
    sidecar = SlackIngressServer(
        slack,
        SlackIngressPolicy(
            store,
            reaction_user_ids={"U_OWNER"},
            reaction_names={"eyes"},
        ),
        host="127.0.0.1",
        port=0,
    )
    await sidecar.start(connect_slack=False)
    assert slack.connected is False

    placeholder = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=39_000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=False,
        markdown_dialect="slack",
        len_unit="chars",
    )
    transport = WebSocketRelayTransport(
        sidecar.url,
        "slack",
        "U_BOT",
        authorization_is_upstream=False,
    )
    relay = RelayAdapter(PlatformConfig(enabled=True), placeholder, transport=transport)
    received = asyncio.Event()
    inbound: list[MessageEvent] = []

    async def _capture(event: MessageEvent) -> None:
        inbound.append(event)
        received.set()

    relay.set_message_handler(_capture)
    try:
        assert await relay.connect() is True
        await asyncio.wait_for(slack.connected_event.wait(), timeout=2)
        assert slack.connected is True
        assert slack.connect_calls == 1
        assert slack.reaction_handler is not None

        duplicate_transport = WebSocketRelayTransport(
            sidecar.url,
            "slack",
            "U_DUPLICATE",
            authorization_is_upstream=False,
        )
        duplicate_relay = RelayAdapter(
            PlatformConfig(enabled=True),
            placeholder,
            transport=duplicate_transport,
        )
        try:
            with pytest.raises(Exception, match="4429"):
                await duplicate_relay.connect()
        finally:
            await duplicate_relay.disconnect()
        assert slack.connect_calls == 1

        reaction = {
            "type": "reaction_added",
            "user": "U_OTHER",
            "reaction": "eyes",
            "item": {"type": "message", "channel": "C1", "ts": "100.1"},
        }
        await slack.reaction_handler(reaction, body={"team_id": "T1"})
        slack.dispatch_reaction_trigger.assert_not_awaited()
        reaction["user"] = "U_OWNER"
        await slack.reaction_handler(reaction, body={"team_id": "T1"})
        slack.dispatch_reaction_trigger.assert_awaited_once_with(
            reaction, body={"team_id": "T1"}
        )

        await sidecar.forward_event(
            MessageEvent(
                text="plain follow-up",
                message_type=MessageType.TEXT,
                source=SessionSource(
                    platform=Platform.SLACK,
                    chat_id="C1",
                    chat_type="group",
                    user_id="U1",
                    thread_id="100.1",
                    scope_id="T1",
                    message_id="100.2",
                ),
                message_id="100.2",
                reply_to_message_id="100.1",
                metadata={"slack_team_id": "T1", "slack_thread_ts": "100.1"},
            )
        )
        await asyncio.wait_for(received.wait(), timeout=2)

        assert inbound[0].source.platform is Platform.SLACK
        assert inbound[0].source.delivered_via_relay is True
        assert inbound[0].source.delivered_via_upstream_relay is False

        result = await relay.send(
            "C1",
            "Hermes reply",
            reply_to="100.2",
            metadata={"thread_id": "100.1", "slack_team_id": "T1"},
        )
        assert result.success is True
        assert slack.sent == [
            {
                "chat_id": "C1",
                "content": "Hermes reply",
                "reply_to": "100.2",
                "metadata": {
                    "thread_id": "100.1",
                    "slack_team_id": "T1",
                    "scope_id": "T1",
                    "user_id": "U1",
                },
            }
        ]
    finally:
        await relay.disconnect()
        await asyncio.wait_for(slack.disconnected_event.wait(), timeout=2)
        assert slack.connected is False
        await sidecar.stop()
