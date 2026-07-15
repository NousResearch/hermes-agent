from __future__ import annotations

import asyncio
import os
import socket
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.canonical_writer_client import (
    ExactServerMainPidAuthorizer,
    ServerPeerCredentials,
)
from gateway.config import Platform, PlatformConfig
from gateway.discord_connector_protocol import (
    DiscordConnectorEvent,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryMessage,
    DiscordConnectorHistoryPage,
    DiscordConnectorTarget,
    DiscordConnectorTargetType,
)
from gateway.discord_connector_service import (
    DEFAULT_DISCORD_CONNECTOR_SOCKET,
    DiscordConnectorAcceptedMessage,
    DiscordConnectorRuntime,
    DiscordConnectorUnixServer,
    DurableDiscordConnectorJournal,
)
from gateway.discord_edge_service import DiscordEdgePeerCredentials
from gateway.relay.discord_connector_transport import (
    DiscordConnectorRelayTransport,
    DiscordConnectorTransportError,
)
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor


@pytest.fixture
def short_socket_dir():
    # Darwin's sockaddr_un.sun_path is only 104 bytes. Codex worktrees and
    # pytest's default per-test directories can exceed it before the filename.
    with tempfile.TemporaryDirectory(prefix="muncho-sock-", dir="/tmp") as value:
        yield Path(value).resolve(strict=True)


class _PidProvider:
    def __init__(self) -> None:
        self.calls = 0

    def main_pid(self, unit_name: str) -> int:
        self.calls += 1
        return os.getpid()


class _Backend:
    def __init__(self) -> None:
        self.sends = 0
        self.target = DiscordConnectorTarget(
            DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL,
            "100",
            "200",
        )

    def prove_public_target(self, channel_id: str) -> DiscordConnectorTarget:
        if channel_id != self.target.channel_id:
            raise PermissionError("forbidden")
        return self.target

    def fetch_guild_history(
        self,
        channel_id: str,
        *,
        limit: int,
        before_message_id: str | None,
        after_message_id: str | None,
        authority: DiscordConnectorHistoryAuthority,
    ) -> DiscordConnectorHistoryPage:
        if channel_id != self.target.channel_id:
            raise PermissionError("forbidden")
        assert authority == DiscordConnectorHistoryAuthority.authenticated_user(
            "400"
        )
        return DiscordConnectorHistoryPage(
            target=self.target,
            messages=(
                DiscordConnectorHistoryMessage(
                    message_id="301",
                    author_id="400",
                    author_name="Emo",
                    author_is_bot=False,
                    content="continue",
                    content_truncated=False,
                    created_at_unix_ms=1_000,
                    reply_to_message_id="300",
                ),
            ),
            limit=limit,
            before_message_id=before_message_id,
            after_message_id=after_message_id,
            has_more=False,
        )

    def send_public_message(
        self,
        target: DiscordConnectorTarget,
        content: str,
        *,
        reply_to_message_id: str | None,
        deadline_unix_ms: int,
    ) -> DiscordConnectorAcceptedMessage:
        self.sends += 1
        return DiscordConnectorAcceptedMessage("500", True)


def _event(target: DiscordConnectorTarget) -> DiscordConnectorEvent:
    return DiscordConnectorEvent.from_mapping(
        {
            "event_id": "300",
            "target": target.to_mapping(),
            "author_id": "400",
            "author_name": "Emo",
            "author_is_bot": False,
            "content": "continue the approved plan",
            "created_at_unix_ms": int(time.time() * 1_000),
            "reply_to_message_id": None,
        }
    )


def test_history_receipt_replay_is_bound_to_exact_internal_authority() -> None:
    target = DiscordConnectorTarget(
        DiscordConnectorTargetType.GUILD_CHANNEL,
        "100",
        "200",
    )
    page = DiscordConnectorHistoryPage(
        target=target,
        messages=(),
        limit=1,
        before_message_id=None,
        after_message_id=None,
        has_more=False,
    )
    first = DiscordConnectorHistoryAuthority.authenticated_user("400")
    second = DiscordConnectorHistoryAuthority.authenticated_user("401")
    transport = object.__new__(DiscordConnectorRelayTransport)
    transport._request_sync = lambda *_args, **_kwargs: {
        "status": "ok",
        "result": {
            "page": page.to_mapping(),
            "page_sha256": page.sha256,
            "authority_sha256": first.sha256,
        },
    }

    with pytest.raises(
        DiscordConnectorTransportError,
        match="connector_history_receipt_invalid",
    ):
        transport.read_guild_history(
            "200",
            limit=1,
            authority=second,
        )


@pytest.mark.asyncio
async def test_existing_relay_adapter_transport_has_exact_receipts_and_event_ack(
    short_socket_dir, monkeypatch
) -> None:
    # Production UIDs are non-root; skip only an unusual root-only test runner.
    if os.getuid() == 0:
        pytest.skip("exact production-shaped UID boundary requires non-root")
    journal = DurableDiscordConnectorJournal.bootstrap(
        short_socket_dir / "journal.sqlite3"
    )
    backend = _Backend()
    runtime = DiscordConnectorRuntime(backend=backend, journal=journal)
    server_pid = _PidProvider()
    socket_path = short_socket_dir / "connector.sock"
    server = DiscordConnectorUnixServer(
        socket_path,
        runtime=runtime,
        expected_gateway_uid=os.getuid(),
        gateway_unit="hermes-cloud-gateway.service",
        main_pid_provider=server_pid,
        peer_getter=lambda _sock: DiscordEdgePeerCredentials(
            os.getpid(), os.getuid(), os.getgid()
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not socket_path.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    client_pid = _PidProvider()
    authorizer = ExactServerMainPidAuthorizer(
        server_unit="muncho-discord-connector.service",
        expected_server_uid=os.getuid(),
        main_pid_provider=client_pid,
    )
    transport = DiscordConnectorRelayTransport(
        socket_path,
        server_authorizer=authorizer,
        server_peer_getter=lambda _sock: ServerPeerCredentials(
            os.getpid(), os.getuid(), os.getgid()
        ),
        event_wait_ms=10,
    )
    accepted = asyncio.Event()
    events = []

    async def _accept(event) -> None:
        events.append(event)
        accepted.set()

    placeholder = CapabilityDescriptor(
        contract_version=1,
        platform="relay",
        label="Relay",
        max_message_length=4_096,
        supports_draft_streaming=False,
        supports_edit=False,
        supports_threads=False,
        markdown_dialect="plain",
        len_unit="chars",
    )
    adapter = RelayAdapter(PlatformConfig(), placeholder, transport=transport)
    adapter.handle_message = _accept
    try:
        assert await adapter.connect() is True
        assert adapter.descriptor.platform == "discord"
        assert journal.offer_event(_event(backend.target)) is True
        await asyncio.wait_for(accepted.wait(), timeout=2)
        assert events[0].source.chat_type == "channel"
        assert events[0].source.platform is Platform.DISCORD
        assert events[0].source.delivered_via_upstream_relay is True

        result = await adapter.send(
            "200",
            "done",
            reply_to="300",
            metadata={
                "scope_id": "100",
                "connector_idempotency_key": "ordinary-reply:case:1",
            },
        )
        assert result.success is True
        assert result.message_id == "500"
        assert backend.sends == 1

        history = transport.read_guild_history(
            "200",
            limit=10,
            after_message_id="300",
            authority=DiscordConnectorHistoryAuthority.authenticated_user("400"),
        )
        assert history["target"] == backend.target.to_mapping()
        assert history["query"] == {
            "limit": 10,
            "before_message_id": None,
            "after_message_id": "300",
        }
        assert history["messages"][0]["content"] == "continue"

        # One logical BasePlatformAdapter retry retains one connector key.
        real_send = transport.send_outbound
        retry_actions = []

        async def _fail_before_dispatch_once(action, *, platform=None):
            retry_actions.append(action)
            if len(retry_actions) == 1:
                return {
                    "success": False,
                    "error": "connector_transport_failed",
                    "error_kind": "transient",
                    "retryable": True,
                }
            return await real_send(action, platform=platform)

        monkeypatch.setattr("gateway.platforms.base.random.uniform", lambda *_: 0)
        monkeypatch.setattr(transport, "send_outbound", _fail_before_dispatch_once)
        retry_result = await adapter._send_with_retry(
            "200",
            "safe retry",
            reply_to="300",
            metadata={"scope_id": "100"},
            max_retries=1,
            base_delay=0,
        )
        assert retry_result.success is True
        assert len(retry_actions) == 2
        retry_keys = {
            action["metadata"]["connector_idempotency_key"]
            for action in retry_actions
        }
        assert len(retry_keys) == 1
        assert backend.sends == 2

        # A response lost after a verified dispatch is structurally ambiguous.
        # The generic retry/plain-text/notice paths must not execute again.
        uncertain_actions = []

        async def _lose_receipt_after_dispatch(action, *, platform=None):
            uncertain_actions.append(action)
            accepted = await real_send(action, platform=platform)
            assert accepted["success"] is True
            return {
                "success": False,
                "error": "Discord dispatch outcome is uncertain",
                "error_kind": "dispatch_uncertain",
                "retryable": False,
            }

        monkeypatch.setattr(transport, "send_outbound", _lose_receipt_after_dispatch)
        uncertain = await adapter._send_with_retry(
            "200",
            "single ambiguous send",
            reply_to="300",
            metadata={"scope_id": "100"},
            max_retries=2,
            base_delay=0,
        )
        assert uncertain.success is False
        assert uncertain.error_kind == "dispatch_uncertain"
        assert len(uncertain_actions) == 1
        assert backend.sends == 3

        # The same ambiguity can happen after a proven pre-dispatch failure
        # triggers one safe retry. It must still stop before generic formatting
        # fallback or a delivery-failure notice can create a second operation.
        transition_actions = []

        async def _transport_then_ambiguous(action, *, platform=None):
            transition_actions.append(action)
            if len(transition_actions) == 1:
                return {
                    "success": False,
                    "error": "connector_transport_failed",
                    "error_kind": "transient",
                    "retryable": True,
                }
            accepted = await real_send(action, platform=platform)
            assert accepted["success"] is True
            return {
                "success": False,
                "error": "Discord dispatch outcome is uncertain",
                "error_kind": "dispatch_uncertain",
                "retryable": False,
            }

        monkeypatch.setattr(transport, "send_outbound", _transport_then_ambiguous)
        transitioned = await adapter._send_with_retry(
            "200",
            "ambiguous only after retry",
            reply_to="300",
            metadata={"scope_id": "100"},
            max_retries=2,
            base_delay=0,
        )
        assert transitioned.success is False
        assert transitioned.error_kind == "dispatch_uncertain"
        assert len(transition_actions) == 2
        assert len(
            {
                action["metadata"]["connector_idempotency_key"]
                for action in transition_actions
            }
        ) == 1
        assert backend.sends == 4

        # Signed Canonical route-back requests belong exclusively to the
        # existing discord_edge_* protocol and cannot enter the normal connector.
        monkeypatch.setattr(transport, "send_outbound", real_send)
        forbidden_route_back = await adapter.send(
            "200",
            "canonical route-back",
            metadata={"discord_edge_request": {"signed": True}},
        )
        assert forbidden_route_back.success is False
        assert forbidden_route_back.error_kind == "blocked_before_dispatch"
        assert backend.sends == 4

        ack_deadline = time.monotonic() + 2
        state = ""
        while time.monotonic() < ack_deadline:
            with sqlite3.connect(journal.path) as conn:
                state = conn.execute(
                    "SELECT state FROM connector_events_v1 WHERE event_id='300'"
                ).fetchone()[0]
            if state == "acked":
                break
            await asyncio.sleep(0.01)
        assert state == "acked"
        # Both sides query the exact current MainPID more than once per frame.
        assert client_pid.calls >= 4
        assert server_pid.calls >= 6
    finally:
        await adapter.disconnect()
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_wrong_connector_service_identity_is_blocked_before_request(
    short_socket_dir,
) -> None:
    socket_path = short_socket_dir / "wrong.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    accepted = threading.Event()

    def _serve() -> None:
        conn, _ = listener.accept()
        accepted.set()
        conn.close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    authorizer = ExactServerMainPidAuthorizer(
        server_unit="muncho-discord-connector.service",
        expected_server_uid=max(1, os.getuid()),
        main_pid_provider=_PidProvider(),
    )
    transport = DiscordConnectorRelayTransport(
        socket_path,
        server_authorizer=authorizer,
        server_peer_getter=lambda _sock: ServerPeerCredentials(
            os.getpid(), os.getuid() + 1, os.getgid()
        ),
    )
    try:
        with pytest.raises(
            DiscordConnectorTransportError, match="connector_server_unauthorized"
        ):
            await transport.connect()
        assert accepted.wait(timeout=1)
    finally:
        listener.close()
        thread.join(timeout=1)


def test_pinned_unix_endpoint_builds_existing_relay_adapter_consumer(monkeypatch) -> None:
    import pwd

    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from gateway.relay import register_relay_adapter
    from gateway.relay.adapter import RelayAdapter

    platform_registry.unregister("relay")
    monkeypatch.setattr(
        pwd,
        "getpwnam",
        lambda _name: SimpleNamespace(pw_uid=max(1, os.getuid())),
    )
    try:
        assert register_relay_adapter(
            url=f"unix://{DEFAULT_DISCORD_CONNECTOR_SOCKET}"
        )
        adapter = platform_registry.create_adapter("relay", PlatformConfig())
        assert isinstance(adapter, RelayAdapter)
        assert isinstance(adapter._transport, DiscordConnectorRelayTransport)
    finally:
        platform_registry.unregister("relay")


def test_local_connector_has_no_http_provision_or_semantic_policy_side_channel(
    monkeypatch,
) -> None:
    import gateway.relay as relay

    local_url = f"unix://{DEFAULT_DISCORD_CONNECTOR_SOCKET}"
    monkeypatch.setattr(relay, "relay_url", lambda: local_url)
    monkeypatch.setattr(
        relay,
        "_resolve_relay_identity_token",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve credentials")),
    )
    monkeypatch.setattr(
        relay,
        "_post_policy",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not POST policy")),
    )
    assert relay.self_provision_relay() is False
    assert relay.send_relay_policy() is False
