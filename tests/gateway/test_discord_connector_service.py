from __future__ import annotations

import os
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path

import pytest

from gateway.discord_connector_protocol import (
    DiscordConnectorEvent,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryMessage,
    DiscordConnectorHistoryPage,
    DiscordConnectorKind,
    DiscordConnectorTarget,
    DiscordConnectorTargetType,
    canonical_json_bytes,
    decode_frame,
    parse_request,
    request_message,
)
from gateway.discord_connector_service import (
    DiscordConnectorAcceptedMessage,
    DiscordConnectorHistoryReaderPeer,
    DiscordConnectorRuntime,
    DiscordConnectorServiceError,
    DiscordConnectorUnixServer,
    DurableDiscordConnectorJournal,
)
from gateway.discord_edge_service import DiscordEdgePeerCredentials


@pytest.fixture
def unix_socket_tmp_path():
    # macOS limits AF_UNIX addresses to 104 bytes; pytest's nested tmp_path can
    # exceed that before the fixed socket filename is appended.
    base = "/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
    with tempfile.TemporaryDirectory(prefix="mdc-", dir=base) as directory:
        yield Path(directory)


def _target() -> DiscordConnectorTarget:
    return DiscordConnectorTarget(
        DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL,
        "100",
        "200",
    )


def _event(content: str = "all meaning stays model-owned") -> DiscordConnectorEvent:
    return DiscordConnectorEvent.from_mapping({
        "event_id": "300",
        "target": _target().to_mapping(),
        "author_id": "400",
        "author_name": "Emo",
        "author_is_bot": False,
        "content": content,
        "created_at_unix_ms": int(time.time() * 1_000),
        "reply_to_message_id": None,
    })


class _Backend:
    def __init__(self, *, verified: bool = True) -> None:
        self.verified = verified
        self.sends = 0
        self.history_reads = 0

    def prove_public_target(self, channel_id: str) -> DiscordConnectorTarget:
        if channel_id != "200":
            raise PermissionError("blocked")
        return _target()

    def fetch_guild_history(
        self,
        channel_id: str,
        *,
        limit: int,
        before_message_id: str | None,
        after_message_id: str | None,
        authority: DiscordConnectorHistoryAuthority,
    ) -> DiscordConnectorHistoryPage:
        if channel_id != "200":
            raise PermissionError("blocked")
        self.history_reads += 1
        assert authority == DiscordConnectorHistoryAuthority.authenticated_user(
            "400"
        )
        return DiscordConnectorHistoryPage(
            target=_target(),
            messages=(
                DiscordConnectorHistoryMessage(
                    message_id="301",
                    author_id="400",
                    author_name="Emo",
                    author_is_bot=False,
                    content="public evidence; GPT interprets it",
                    content_truncated=False,
                    created_at_unix_ms=1_000,
                    reply_to_message_id=None,
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
        return DiscordConnectorAcceptedMessage("500", self.verified)


def _send_request(*, content: str = "do the task", key: str = "case:1"):
    return parse_request(
        request_message(
            DiscordConnectorKind.MESSAGE_SEND,
            {
                "idempotency_key": key,
                "target": _target().to_mapping(),
                "content": content,
                "reply_to_message_id": None,
                "deadline_unix_ms": int(time.time() * 1_000) + 5_000,
            },
        )
    )


class _MainPids:
    def __init__(self, values: dict[str, int]) -> None:
        self.values = values

    def main_pid(self, unit_name: str) -> int:
        return self.values[unit_name]


def _history_message(user_id: str = "400") -> dict:
    return request_message(
        DiscordConnectorKind.HISTORY_FETCH,
        {
            "channel_id": "200",
            "limit": 1,
            "before_message_id": None,
            "after_message_id": None,
            "authority": DiscordConnectorHistoryAuthority.authenticated_user(
                user_id
            ).to_mapping(),
        },
    )


def _send_message(key: str = "canary-boundary:1") -> dict:
    return request_message(
        DiscordConnectorKind.MESSAGE_SEND,
        {
            "idempotency_key": key,
            "target": _target().to_mapping(),
            "content": "must not dispatch",
            "reply_to_message_id": None,
            "deadline_unix_ms": int(time.time() * 1_000) + 5_000,
        },
    )


def _exchange(server: DiscordConnectorUnixServer, message: dict) -> dict | None:
    client, accepted = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    body = canonical_json_bytes(message)
    client.sendall(struct.pack("!I", len(body)) + body)
    worker = threading.Thread(
        target=server._handle_connection,
        args=(accepted,),
        daemon=False,
    )
    worker.start()
    try:
        try:
            header = client.recv(4)
        except ConnectionResetError:
            return None
        if not header:
            return None
        size = struct.unpack("!I", header)[0]
        chunks = bytearray()
        while len(chunks) < size:
            try:
                chunk = client.recv(size - len(chunks))
            except ConnectionResetError:
                return None
            if not chunk:
                break
            chunks.extend(chunk)
        return decode_frame(bytes(chunks)) if len(chunks) == size else None
    finally:
        client.close()
        worker.join(timeout=1)
        assert worker.is_alive() is False


def _scoped_server(
    tmp_path: Path,
    *,
    peer: DiscordEdgePeerCredentials,
    main_pid_provider,
) -> tuple[DiscordConnectorUnixServer, _Backend]:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    backend = _Backend()
    server = DiscordConnectorUnixServer(
        tmp_path / "connector.sock",
        runtime=DiscordConnectorRuntime(backend=backend, journal=journal),
        expected_gateway_uid=501,
        gateway_unit="hermes-cloud-gateway.service",
        main_pid_provider=main_pid_provider,
        peer_getter=lambda _sock: peer,
        history_reader_peer=DiscordConnectorHistoryReaderPeer(
            service_unit="muncho-capability-producer-discord-edge.service",
            expected_uid=502,
            requester_user_id="400",
        ),
    )
    return server, backend


def test_event_journal_is_first_wins_leased_and_exactly_acked(tmp_path) -> None:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    event = _event()
    assert journal.offer_event(event) is True
    assert journal.offer_event(event) is False
    with pytest.raises(
        DiscordConnectorServiceError, match="event_idempotency_conflict"
    ):
        journal.offer_event(_event("different payload"))

    delivery = journal.next_event(now_unix_ms=1_000)
    assert delivery is not None
    assert journal.next_event(now_unix_ms=1_001) is None
    with pytest.raises(DiscordConnectorServiceError, match="event_ack_binding_invalid"):
        journal.ack_event(
            delivery_id=delivery["delivery_id"],
            event_id=event.event_id,
            event_sha256="0" * 64,
            now_unix_ms=1_002,
        )
    assert (
        journal.ack_event(
            delivery_id=delivery["delivery_id"],
            event_id=event.event_id,
            event_sha256=event.sha256,
            now_unix_ms=1_003,
        )
        is False
    )
    assert journal.next_event(now_unix_ms=100_000) is None


def test_cleanup_blocks_retirement_until_inbound_events_are_acked(tmp_path) -> None:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    event = _event()
    journal.offer_event(event)

    pending = journal.cleanup_snapshot()
    assert pending["unacked_event_count"] == 1
    assert pending["unresolved_dispatch_count"] == 0
    assert pending["safe_to_retire"] is False

    delivery = journal.next_event(now_unix_ms=1_000)
    assert delivery is not None
    delivering = journal.cleanup_snapshot()
    assert delivering["unacked_event_count"] == 1
    assert delivering["safe_to_retire"] is False

    journal.ack_event(
        delivery_id=delivery["delivery_id"],
        event_id=event.event_id,
        event_sha256=event.sha256,
        now_unix_ms=1_001,
    )
    complete = journal.cleanup_snapshot()
    assert complete["unacked_event_count"] == 0
    assert complete["safe_to_retire"] is True


def test_unix_listener_readiness_proves_accepting_fd_and_rejects_closed_fd(
    unix_socket_tmp_path,
) -> None:
    tmp_path = unix_socket_tmp_path
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    runtime = DiscordConnectorRuntime(backend=_Backend(), journal=journal)
    server = DiscordConnectorUnixServer(
        tmp_path / "connector.sock",
        runtime=runtime,
        expected_gateway_uid=max(os.getuid(), 1),
    )
    server.start()
    try:
        identity = server.readiness_identity()
        assert identity["listening"] is True
        assert identity["socket_path"] == str(tmp_path / "connector.sock")
        assert server._listener is not None
        server._listener.close()
        with pytest.raises(
            DiscordConnectorServiceError,
            match="connector_socket_not_ready",
        ):
            server.readiness_identity()
    finally:
        server.shutdown()


def test_shutdown_drains_tracked_request_threads(unix_socket_tmp_path) -> None:
    tmp_path = unix_socket_tmp_path
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    runtime = DiscordConnectorRuntime(backend=_Backend(), journal=journal)
    server = DiscordConnectorUnixServer(
        tmp_path / "connector.sock",
        runtime=runtime,
        expected_gateway_uid=max(os.getuid(), 1),
        connection_timeout_seconds=1,
    )
    server.start()
    release = threading.Event()
    worker = threading.Thread(target=lambda: release.wait(1), daemon=False)
    with server._handler_lock:
        server._handlers.add(worker)
    worker.start()
    release.set()
    server.shutdown()
    assert worker.is_alive() is False


def test_canary_history_reader_peer_can_only_read_as_exact_canary_user(
    tmp_path,
) -> None:
    peer = DiscordEdgePeerCredentials(pid=7001, uid=502, gid=900)
    server, backend = _scoped_server(
        tmp_path,
        peer=peer,
        main_pid_provider=_MainPids({
            "hermes-cloud-gateway.service": 6001,
            "muncho-capability-producer-discord-edge.service": 7001,
        }),
    )
    assert server.history_reader_identity() == {
        "service_unit": "muncho-capability-producer-discord-edge.service",
        "expected_uid": 502,
        "authority_sha256": (
            DiscordConnectorHistoryAuthority.authenticated_user("400").sha256
        ),
        "operation": DiscordConnectorKind.HISTORY_FETCH.value,
    }

    response = _exchange(server, _history_message("400"))
    assert response is not None
    assert response["status"] == "ok"
    assert backend.history_reads == 1

    assert _exchange(server, _history_message("401")) is None
    assert backend.history_reads == 1


@pytest.mark.parametrize(
    "operation",
    ["hello", "event_next", "target_get", "message_send"],
)
def test_canary_history_reader_peer_cannot_poll_discover_or_send(
    tmp_path,
    operation,
) -> None:
    server, backend = _scoped_server(
        tmp_path,
        peer=DiscordEdgePeerCredentials(pid=7001, uid=502, gid=900),
        main_pid_provider=_MainPids({
            "hermes-cloud-gateway.service": 6001,
            "muncho-capability-producer-discord-edge.service": 7001,
        }),
    )
    message = {
        "hello": lambda: request_message(
            DiscordConnectorKind.HELLO,
            {"consumer": "forbidden"},
        ),
        "event_next": lambda: request_message(
            DiscordConnectorKind.EVENT_NEXT,
            {"wait_ms": 0},
        ),
        "target_get": lambda: request_message(
            DiscordConnectorKind.TARGET_GET,
            {"channel_id": "200"},
        ),
        "message_send": _send_message,
    }[operation]()

    assert _exchange(server, message) is None
    assert backend.history_reads == 0
    assert backend.sends == 0


@pytest.mark.parametrize(
    "peer",
    [
        DiscordEdgePeerCredentials(pid=7001, uid=503, gid=900),
        DiscordEdgePeerCredentials(pid=7002, uid=502, gid=900),
    ],
)
def test_canary_history_reader_peer_fails_closed_on_uid_or_pid_drift(
    tmp_path,
    peer,
) -> None:
    server, backend = _scoped_server(
        tmp_path,
        peer=peer,
        main_pid_provider=_MainPids({
            "hermes-cloud-gateway.service": 6001,
            "muncho-capability-producer-discord-edge.service": 7001,
        }),
    )
    assert _exchange(server, _history_message()) is None
    assert backend.history_reads == 0


def test_canary_history_reader_pid_is_rechecked_before_discord_read(
    tmp_path,
) -> None:
    class _PidDrift:
        calls = 0

        def main_pid(self, unit_name: str) -> int:
            assert unit_name == "muncho-capability-producer-discord-edge.service"
            self.calls += 1
            return 7001 if self.calls == 1 else 7002

    server, backend = _scoped_server(
        tmp_path,
        peer=DiscordEdgePeerCredentials(pid=7001, uid=502, gid=900),
        main_pid_provider=_PidDrift(),
    )
    assert _exchange(server, _history_message()) is None
    assert backend.history_reads == 0


def test_gateway_peer_contract_remains_exact_and_separate(tmp_path) -> None:
    server, backend = _scoped_server(
        tmp_path,
        peer=DiscordEdgePeerCredentials(pid=6001, uid=501, gid=901),
        main_pid_provider=_MainPids({
            "hermes-cloud-gateway.service": 6001,
            "muncho-capability-producer-discord-edge.service": 7001,
        }),
    )
    response = _exchange(server, _send_message("gateway-contract:1"))
    assert response is not None
    assert response["status"] == "ok"
    assert backend.sends == 1

    server.peer_getter = lambda _sock: DiscordEdgePeerCredentials(
        pid=6001,
        uid=504,
        gid=901,
    )
    assert _exchange(server, _send_message("gateway-contract:2")) is None
    assert backend.sends == 1


def test_send_is_idempotent_and_changed_payload_never_redispatches(tmp_path) -> None:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    backend = _Backend()
    runtime = DiscordConnectorRuntime(backend=backend, journal=journal)
    request = _send_request()

    first = runtime.handle(request)
    replay = runtime.handle(request)
    assert first["status"] == "ok"
    assert replay["status"] == "ok"
    assert replay["replayed"] is True
    assert backend.sends == 1

    with pytest.raises(DiscordConnectorServiceError, match="send_idempotency_conflict"):
        runtime.handle(_send_request(content="changed", key="case:1"))
    assert backend.sends == 1


def test_unverified_dispatch_is_uncertain_and_never_blindly_retried(tmp_path) -> None:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    backend = _Backend(verified=False)
    runtime = DiscordConnectorRuntime(backend=backend, journal=journal)
    request = _send_request()

    first = runtime.handle(request)
    replay = runtime.handle(request)
    assert first["status"] == "dispatch_uncertain"
    assert replay["status"] == "dispatch_uncertain"
    assert replay["replayed"] is True
    assert backend.sends == 1


def test_public_history_is_exactly_query_bound_and_private_shape_cannot_enter(
    tmp_path,
) -> None:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    runtime = DiscordConnectorRuntime(backend=_Backend(), journal=journal)
    request = parse_request(
        request_message(
            DiscordConnectorKind.HISTORY_FETCH,
            {
                "channel_id": "200",
                "limit": 5,
                "before_message_id": None,
                "after_message_id": "300",
                "authority": DiscordConnectorHistoryAuthority.authenticated_user(
                    "400"
                ).to_mapping(),
            },
        )
    )

    result = runtime.handle(request)
    assert result["status"] == "ok"
    page = DiscordConnectorHistoryPage.from_mapping(result["result"]["page"])
    assert page.target.target_type is (
        DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL
    )
    assert page.after_message_id == "300"
    assert page.messages[0].content == "public evidence; GPT interprets it"
    assert result["result"]["page_sha256"] == page.sha256
    assert result["result"]["authority_sha256"] == (
        DiscordConnectorHistoryAuthority.authenticated_user("400").sha256
    )

    forged = object.__new__(DiscordConnectorHistoryPage)
    object.__setattr__(
        forged,
        "target",
        {
            "target_type": "dm",
            "guild_id": "100",
            "channel_id": "200",
        },
    )
    runtime.backend.fetch_guild_history = lambda *_args, **_kwargs: forged
    blocked = runtime.handle(request)
    assert blocked["status"] == "blocked"
    assert blocked["result"] == {}


@pytest.mark.parametrize(
    ("message_id", "readback_verified"),
    [("0", True), ("500", 1), (500, True)],
)
def test_forged_backend_receipt_can_never_commit_verified(
    tmp_path, message_id, readback_verified
) -> None:
    journal = DurableDiscordConnectorJournal.bootstrap(tmp_path / "journal.sqlite3")
    backend = _Backend()
    forged = object.__new__(DiscordConnectorAcceptedMessage)
    object.__setattr__(forged, "message_id", message_id)
    object.__setattr__(forged, "readback_verified", readback_verified)
    backend.send_public_message = lambda *_args, **_kwargs: forged
    runtime = DiscordConnectorRuntime(backend=backend, journal=journal)
    request = _send_request()

    result = runtime.handle(request)
    replay = runtime.handle(request)
    assert result["status"] == "dispatch_uncertain"
    assert result["result"]["readback_verified"] is False
    assert replay["status"] == "dispatch_uncertain"
    assert replay["replayed"] is True
