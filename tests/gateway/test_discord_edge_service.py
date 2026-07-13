import json
import os
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.discord_edge_client import (
    DiscordEdgeClient,
    DiscordEdgeClientError,
    DiscordEdgeServerPeer,
)
from gateway.discord_edge_protocol import (
    MAX_DEADLINE_SECONDS,
    MAX_REQUEST_BYTES,
    DiscordEdgeAuthorityKind,
    DiscordEdgeErrorCode,
    DiscordEdgeIntent,
    DiscordEdgeOperation,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeReconciliationQuery,
    DiscordPublicTarget,
    SignedDiscordEdgeEnvelope,
    canonical_json_bytes,
    make_request,
    sign_capability,
    verify_receipt,
    verify_request_capability,
)
from gateway.discord_edge_runtime import (
    DiscordEdgeRuntime,
    DiscordLivePublicTargetProof,
    DiscordMutationAccepted,
    DiscordMutationReadback,
    DurableDiscordEdgeJournal,
)
from gateway.discord_edge_service import (
    SOCKET_MODE,
    DiscordEdgePeerCredentials,
    DiscordEdgeUnixServer,
)

NOW_MS = 2_000_000_000_000
GUILD_ID = "100000000000000001"
CHANNEL_ID = "100000000000000002"
MESSAGE_ID = "100000000000000003"
BOT_USER_ID = "100000000000000004"
UNIT = "hermes-cloud-gateway.service"
_FRAME_HEADER = struct.Struct("!I")


@pytest.fixture
def private_edge_dir():
    base = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/tmp")
    path = Path(tempfile.mkdtemp(prefix="de-", dir=base))
    path.chmod(0o700)
    try:
        yield path
    finally:
        path.chmod(0o700)
        shutil.rmtree(path, ignore_errors=True)


class StaticMainPidProvider:
    def __init__(self, pid):
        self.pid = pid
        self.calls = 0

    def main_pid(self, unit_name):
        assert unit_name == UNIT
        self.calls += 1
        return self.pid


class SequencedMainPidProvider:
    def __init__(self, *pids):
        self.pids = list(pids)
        self.calls = 0
        self._lock = threading.Lock()

    def main_pid(self, unit_name):
        assert unit_name == UNIT
        with self._lock:
            value = self.pids[min(self.calls, len(self.pids) - 1)]
            self.calls += 1
            return value


class FixedPublicProof:
    def _proof(self, operation, target, now_unix_ms):
        return DiscordLivePublicTargetProof(
            operation=operation,
            target=target,
            bot_user_id=BOT_USER_ID,
            observed_at_unix_ms=now_unix_ms,
            publicly_viewable=True,
            bot_can_view=True,
            bot_has_required_permission=True,
        )

    def prove_public_message_send(
        self,
        target,
        *,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert deadline_unix_ms > now_unix_ms
        return self._proof(
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target,
            now_unix_ms,
        )

    def prove_public_message_edit(
        self,
        target,
        *,
        deadline_unix_ms,
        now_unix_ms,
    ):
        return self._proof(
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
            target,
            now_unix_ms,
        )

    def prove_public_thread_create(
        self,
        target,
        *,
        has_initial_message,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert isinstance(has_initial_message, bool)
        return self._proof(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target,
            now_unix_ms,
        )

    def prove_public_readback(
        self,
        operation,
        target,
        *,
        require_message_history,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert isinstance(operation, DiscordEdgeOperation)
        assert isinstance(require_message_history, bool)
        assert deadline_unix_ms > now_unix_ms
        return self._proof(operation, target, now_unix_ms)


class FixedDiscordTransport:
    def __init__(self):
        self.send_calls = 0
        self.read_calls = 0
        self.content = None
        self.readback_content_override = None

    def send_public_message(
        self,
        target,
        *,
        content,
        reply_to_message_id,
        deadline_unix_ms,
    ):
        assert deadline_unix_ms == NOW_MS + 15_000
        assert reply_to_message_id is None
        self.send_calls += 1
        self.content = content
        return DiscordMutationAccepted(
            operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target=target,
            discord_object_id=MESSAGE_ID,
            bot_user_id=BOT_USER_ID,
        )

    def edit_public_message(
        self,
        target,
        *,
        message_id,
        content,
        deadline_unix_ms,
    ):
        raise AssertionError("unexpected edit operation")

    def create_public_thread(
        self,
        target,
        *,
        name,
        initial_message,
        auto_archive_minutes,
        deadline_unix_ms,
    ):
        raise AssertionError("unexpected thread operation")

    def read_public_message(
        self,
        target,
        *,
        operation,
        message_id,
        expected_reply_to_message_id,
    ):
        assert operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
        assert message_id == MESSAGE_ID
        assert expected_reply_to_message_id is None
        self.read_calls += 1
        return DiscordMutationReadback(
            operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target=target,
            discord_object_id=message_id,
            author_user_id=BOT_USER_ID,
            content=(
                self.content
                if self.readback_content_override is None
                else self.readback_content_override
            ),
        )

    def read_created_public_thread(self, target, *, thread_id, expected_content):
        raise AssertionError("unexpected thread readback")


def _build_runtime(private_edge_dir):
    writer_key = Ed25519PrivateKey.generate()
    edge_key = Ed25519PrivateKey.generate()
    transport = FixedDiscordTransport()
    runtime = DiscordEdgeRuntime(
        writer_public_key=writer_key.public_key(),
        edge_private_key=edge_key,
        journal=DurableDiscordEdgeJournal.bootstrap(
            private_edge_dir / "journal.sqlite3"
        ),
        target_prover=FixedPublicProof(),
        transport=transport,
        clock_ms=lambda: NOW_MS,
    )
    return runtime, writer_key, edge_key, transport


def _request(writer_key):
    target = DiscordPublicTarget.from_mapping(
        {
            "target_type": "public_guild_channel",
            "guild_id": GUILD_ID,
            "channel_id": CHANNEL_ID,
        }
    )
    intent = DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        target=target,
        payload={"content": "Exact service response"},
        idempotency_key="case-1:service:1",
    )
    capability = sign_capability(
        writer_key,
        intent,
        authority_kind=DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
        authority_ref="routeauth:case-1:service",
        issued_at_unix_ms=NOW_MS,
        expires_at_unix_ms=NOW_MS + 60_000,
    )
    return make_request(intent, capability, now_unix_ms=NOW_MS)


def _reconciliation_query(request):
    return DiscordEdgeReconciliationQuery(
        idempotency_key=request.intent.idempotency_key,
        operation=request.intent.operation,
        target=request.intent.target,
        request_sha256=request.intent.request_sha256,
        content_sha256=request.intent.content_sha256,
    )


class AllowExactEdgeServer:
    def __init__(self, expected_peer):
        self.expected_peer = expected_peer

    def authorize(self, peer):
        return peer == self.expected_peer


def _start_server(
    private_edge_dir,
    runtime,
    *,
    peer,
    provider=None,
    expected_uid=None,
):
    provider = provider or StaticMainPidProvider(peer.pid)
    server = DiscordEdgeUnixServer(
        private_edge_dir / "edge.sock",
        runtime=runtime,
        expected_client_uid=(peer.uid if expected_uid is None else expected_uid),
        gateway_unit=UNIT,
        main_pid_provider=provider,
        peer_credentials_getter=lambda _conn: peer,
        connection_timeout_seconds=2,
    )
    server.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, provider


def _stop_server(server, thread):
    server.shutdown()
    thread.join(timeout=3)
    assert not thread.is_alive()


def _connect(path):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(2)
    conn.connect(str(path))
    return conn


def _request_frame(request):
    body = canonical_json_bytes(request.to_message())
    return _FRAME_HEADER.pack(len(body)) + body


def _receive_exact(conn, size):
    chunks = []
    while sum(map(len, chunks)) < size:
        chunk = conn.recv(size - sum(map(len, chunks)))
        if not chunk:
            raise AssertionError("connection closed before a complete response")
        chunks.append(chunk)
    return b"".join(chunks)


def _response(conn):
    header = _receive_exact(conn, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    body = _receive_exact(conn, size)
    response = json.loads(body.decode("utf-8"))
    assert set(response) == {"state", "blocker", "replayed", "receipt"}
    return response


def test_wrong_uid_is_rejected_before_any_request(
    private_edge_dir,
):
    runtime, _writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid() + 1, os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
        expected_uid=os.geteuid(),
    )
    try:
        with _connect(server.socket_path) as conn:
            assert conn.recv(1) == b""
        assert transport.send_calls == 0
    finally:
        _stop_server(server, thread)


def test_forked_child_pid_is_rejected_even_with_the_expected_uid(private_edge_dir):
    runtime, _writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    peer = DiscordEdgePeerCredentials(child.pid, os.geteuid(), os.getgid())
    provider = StaticMainPidProvider(os.getpid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
        provider=provider,
    )
    try:
        with _connect(server.socket_path) as conn:
            assert conn.recv(1) == b""
        assert transport.send_calls == 0
    finally:
        _stop_server(server, thread)
        child.terminate()
        child.wait(timeout=3)


def test_current_main_pid_is_rechecked_before_every_request(private_edge_dir):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    provider = SequencedMainPidProvider(peer.pid, peer.pid, peer.pid + 10_000)
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
        provider=provider,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_request_frame(request))
            first = _response(conn)
            assert first["state"] == "verified"

            conn.sendall(_request_frame(request))
            assert conn.recv(1) == b""
        assert provider.calls == 3
        assert transport.send_calls == 1
    finally:
        _stop_server(server, thread)


@pytest.mark.parametrize(
    "body",
    [
        b'{"operation":"first","operation":"second"}',
        b'{"target":{},"target":{}}',
    ],
)
def test_duplicate_json_keys_close_without_runtime_dispatch(private_edge_dir, body):
    runtime, _writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_FRAME_HEADER.pack(len(body)) + body)
            assert conn.recv(1) == b""
        assert transport.send_calls == 0
    finally:
        _stop_server(server, thread)


def test_oversized_frame_is_rejected_before_body_read(private_edge_dir):
    runtime, _writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_FRAME_HEADER.pack(MAX_REQUEST_BYTES + 1))
            assert conn.recv(1) == b""
        assert transport.send_calls == 0
    finally:
        _stop_server(server, thread)


def test_exact_replay_returns_same_signed_receipt_with_fixed_schema(private_edge_dir):
    runtime, writer_key, edge_key, transport = _build_runtime(private_edge_dir)
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_request_frame(request))
            first = _response(conn)
            conn.sendall(_request_frame(request))
            second = _response(conn)

        assert first["state"] == second["state"] == "verified"
        assert first["blocker"] is second["blocker"] is None
        assert first["replayed"] is False
        assert second["replayed"] is True
        assert second["receipt"] == first["receipt"]
        assert transport.send_calls == 1

        envelope = SignedDiscordEdgeEnvelope.from_mapping(
            second["receipt"],
            code=DiscordEdgeErrorCode.INVALID_RECEIPT,
            label="service receipt",
        )
        capability = verify_request_capability(
            request,
            writer_key.public_key(),
            now_unix_ms=NOW_MS,
        )
        receipt = verify_receipt(
            envelope,
            edge_key.public_key(),
            expected_request=request,
            expected_capability=capability,
            now_unix_ms=NOW_MS,
        )
        assert receipt.outcome is DiscordEdgeReceiptOutcome.VERIFIED
    finally:
        _stop_server(server, thread)


def test_expired_exact_replay_returns_same_signed_receipt(private_edge_dir):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_request_frame(request))
            first = _response(conn)
            runtime.clock_ms = lambda: NOW_MS + 120_000
            conn.sendall(_request_frame(request))
            replay = _response(conn)

        assert replay["replayed"] is True
        assert replay["receipt"] == first["receipt"]
        assert transport.send_calls == 1
    finally:
        _stop_server(server, thread)


def test_new_expired_request_is_not_revived_for_dispatch(private_edge_dir):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    runtime.clock_ms = lambda: NOW_MS + 120_000
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_request_frame(request))
            assert conn.recv(1) == b""
        assert transport.send_calls == 0
    finally:
        _stop_server(server, thread)


def test_far_future_request_never_uses_reconciliation_parser(private_edge_dir):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    request = _request(writer_key)
    message = request.to_message()
    message["deadline_unix_ms"] = NOW_MS + (MAX_DEADLINE_SECONDS + 1) * 1_000
    body = canonical_json_bytes(message)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_FRAME_HEADER.pack(len(body)) + body)
            assert conn.recv(1) == b""
        assert transport.send_calls == 0
    finally:
        _stop_server(server, thread)


def test_service_replay_can_reconcile_by_readback_without_resend(private_edge_dir):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    transport.readback_content_override = "Wrong content"
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(server.socket_path) as conn:
            conn.sendall(_request_frame(request))
            first = _response(conn)
            assert first["state"] == "dispatching"
            transport.readback_content_override = None
            conn.sendall(_request_frame(request))
            reconciled = _response(conn)

        assert reconciled["state"] == "verified"
        assert reconciled["replayed"] is True
        assert reconciled["receipt"] != first["receipt"]
        assert transport.send_calls == 1
        assert transport.read_calls == 2
        assert len(runtime.journal.receipt_history(request.intent.idempotency_key)) == 3
    finally:
        _stop_server(server, thread)


def test_fresh_client_recovers_after_server_loss_and_request_expiry_without_resend(
    private_edge_dir,
):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    transport.readback_content_override = "Wrong content"
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    first_server, first_thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    try:
        with _connect(first_server.socket_path) as conn:
            conn.sendall(_request_frame(request))
            first = _response(conn)
        assert first["state"] == "dispatching"
        assert transport.send_calls == 1
    finally:
        _stop_server(first_server, first_thread)

    transport.readback_content_override = None
    restarted_runtime = DiscordEdgeRuntime(
        writer_public_key=writer_key.public_key(),
        edge_private_key=_edge_key,
        journal=DurableDiscordEdgeJournal(runtime.journal.path),
        target_prover=FixedPublicProof(),
        transport=transport,
        clock_ms=lambda: NOW_MS + 120_000,
    )
    second_server, second_thread, _provider = _start_server(
        private_edge_dir,
        restarted_runtime,
        peer=peer,
    )
    server_peer = DiscordEdgeServerPeer(7001, 7002, 7003)
    client = DiscordEdgeClient(
        second_server.socket_path,
        server_authorizer=AllowExactEdgeServer(server_peer),
        server_peer_getter=lambda _sock: server_peer,
    )
    try:
        recovered = client.reconcile(_reconciliation_query(request))
    finally:
        client.close()
        _stop_server(second_server, second_thread)

    assert recovered.request.to_message() == request.to_message()
    assert recovered.state == "verified"
    assert recovered.replayed is True
    assert transport.send_calls == 1
    assert transport.read_calls == 2


def test_claim_without_edge_envelope_stays_not_available_not_blocked(
    private_edge_dir,
):
    runtime, writer_key, _edge_key, transport = _build_runtime(private_edge_dir)
    request = _request(writer_key)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    server_peer = DiscordEdgeServerPeer(7101, 7102, 7103)
    client = DiscordEdgeClient(
        server.socket_path,
        server_authorizer=AllowExactEdgeServer(server_peer),
        server_peer_getter=lambda _sock: server_peer,
    )
    try:
        with pytest.raises(DiscordEdgeClientError) as exc:
            client.reconcile(_reconciliation_query(request))
    finally:
        client.close()
        _stop_server(server, thread)

    assert exc.value.code == "discord_edge_reconciliation_not_available"
    assert exc.value.dispatch_uncertain is False
    assert "blocked" not in exc.value.code
    assert runtime.journal.get(request.intent.idempotency_key) is None
    assert transport.send_calls == 0
    assert transport.read_calls == 0


def test_owned_socket_has_fixed_mode_and_is_removed_on_shutdown(private_edge_dir):
    runtime, _writer_key, _edge_key, _transport = _build_runtime(private_edge_dir)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    socket_stat = server.socket_path.lstat()
    assert stat.S_ISSOCK(socket_stat.st_mode)
    assert stat.S_IMODE(socket_stat.st_mode) == SOCKET_MODE

    _stop_server(server, thread)

    assert not server.socket_path.exists()
    assert not server.socket_path.is_symlink()


def test_cleanup_never_unlinks_replacement_at_socket_path(private_edge_dir):
    runtime, _writer_key, _edge_key, _transport = _build_runtime(private_edge_dir)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server, thread, _provider = _start_server(
        private_edge_dir,
        runtime,
        peer=peer,
    )
    server.socket_path.unlink()
    server.socket_path.write_text("unknown replacement", encoding="utf-8")

    _stop_server(server, thread)

    assert server.socket_path.read_text(encoding="utf-8") == "unknown replacement"


def test_start_never_replaces_preexisting_path(private_edge_dir):
    runtime, _writer_key, _edge_key, _transport = _build_runtime(private_edge_dir)
    socket_path = private_edge_dir / "edge.sock"
    socket_path.write_text("keep", encoding="utf-8")
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    server = DiscordEdgeUnixServer(
        socket_path,
        runtime=runtime,
        expected_client_uid=peer.uid,
        gateway_unit=UNIT,
        main_pid_provider=StaticMainPidProvider(peer.pid),
        peer_credentials_getter=lambda _conn: peer,
    )

    with pytest.raises(FileExistsError):
        server.start()

    assert socket_path.read_text(encoding="utf-8") == "keep"


def test_socket_parent_must_remain_protected(private_edge_dir):
    runtime, _writer_key, _edge_key, _transport = _build_runtime(private_edge_dir)
    private_edge_dir.chmod(0o770)
    peer = DiscordEdgePeerCredentials(os.getpid(), os.geteuid(), os.getgid())
    try:
        with pytest.raises(PermissionError, match="parent"):
            DiscordEdgeUnixServer(
                private_edge_dir / "edge.sock",
                runtime=runtime,
                expected_client_uid=peer.uid,
                gateway_unit=UNIT,
                main_pid_provider=StaticMainPidProvider(peer.pid),
                peer_credentials_getter=lambda _conn: peer,
            )
    finally:
        private_edge_dir.chmod(0o700)
