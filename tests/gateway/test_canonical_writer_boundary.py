import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import pytest

from gateway.canonical_writer_client import (
    CanonicalWriterClient,
    CanonicalWriterClientError,
    ExactServerMainPidAuthorizer,
    ServerPeerCredentials,
    SystemctlServerMainPidProvider,
)
from gateway.canonical_writer_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PROTOCOL_VERSION,
    CanonicalWriterOperation,
    ErrorCode,
    ProtocolError,
    canonical_json_bytes,
    encode_frame,
    make_request,
    parse_response,
    receive_message,
)
from scripts.canonical_writer_service import (
    CanonicalWriterServer,
    DispatchResult,
    OperationDispatcher,
    PeerCredentials,
    SystemdMainPidAuthorizer,
)


class _StaticMainPidProvider:
    def __init__(self, pid, expected_unit=None):
        self.pid = pid
        self.expected_unit = expected_unit
        self.calls = 0

    def main_pid(self, unit_name):
        if self.expected_unit is not None:
            assert unit_name == self.expected_unit
        self.calls += 1
        return self.pid


class _QueuedPeerCredentials:
    def __init__(self, *peers):
        self._peers = deque(peers)
        self._lock = threading.Lock()

    def __call__(self, _conn):
        with self._lock:
            return self._peers.popleft()


def _ping_dispatcher(calls=None):
    def ping(payload, context):
        if calls is not None:
            calls.append((payload, context))
        return DispatchResult(result={"pong": True})

    return OperationDispatcher({CanonicalWriterOperation.PING: ping})


def _start_server(
    tmp_path,
    *,
    peer_getter=None,
    dispatcher=None,
    main_pid_provider=None,
):
    # macOS limits AF_UNIX paths to 104 bytes; pytest's per-test temp paths can
    # exceed that before the filename is added.
    socket_path = Path("/tmp") / f"cw-{uuid.uuid4().hex}.sock"
    peer = PeerCredentials(os.getpid(), os.getuid(), os.getgid())
    provider = main_pid_provider or _StaticMainPidProvider(
        os.getpid(),
        "muncho-gateway.service",
    )
    server = CanonicalWriterServer(
        socket_path,
        authorizer=SystemdMainPidAuthorizer(
            "muncho-gateway.service",
            provider,
            expected_uid=os.getuid(),
        ),
        dispatcher=dispatcher or _ping_dispatcher(),
        peer_credentials_getter=peer_getter or (lambda _conn: peer),
        connection_timeout_seconds=2,
    )
    server.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, socket_path


def _stop_server(server, thread):
    server.shutdown()
    thread.join(timeout=3)
    assert not thread.is_alive()


def _client(socket_path, *, writer_provider=None, writer_peer=None, **kwargs):
    provider = writer_provider or _StaticMainPidProvider(
        os.getpid(),
        "muncho-canonical-writer.service",
    )
    peer = writer_peer or ServerPeerCredentials(
        os.getpid(),
        os.getuid(),
        os.getgid(),
    )
    return CanonicalWriterClient(
        socket_path,
        server_authorizer=ExactServerMainPidAuthorizer(
            server_unit="muncho-canonical-writer.service",
            expected_server_uid=os.getuid(),
            main_pid_provider=provider,
        ),
        server_peer_credentials_getter=lambda _sock: peer,
        **kwargs,
    )


def _connect(path):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(2)
    conn.connect(str(path))
    return conn


def _response(conn):
    return parse_response(receive_message(conn, max_bytes=MAX_RESPONSE_BYTES))


def test_exact_main_pid_is_accepted_and_child_pid_is_rejected(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    peers = _QueuedPeerCredentials(
        PeerCredentials(os.getpid(), os.getuid(), os.getgid()),
        PeerCredentials(child.pid, os.getuid(), os.getgid()),
    )
    server, thread, socket_path = _start_server(tmp_path, peer_getter=peers)
    try:
        with _client(socket_path, max_reconnect_attempts=0) as client:
            result = client.call(CanonicalWriterOperation.PING, {}, runtime={})
            assert result.result == {"pong": True}

        with _connect(socket_path) as rejected:
            response = _response(rejected)
        assert response.ok is False
        assert response.error_code == ErrorCode.UNAUTHORIZED_PEER
    finally:
        _stop_server(server, thread)
        child.terminate()
        child.wait(timeout=3)


def test_malformed_and_oversized_frames_are_rejected(tmp_path):
    server, thread, socket_path = _start_server(tmp_path)
    try:
        with _connect(socket_path) as malformed:
            body = b"{not-json"
            malformed.sendall(struct.pack("!I", len(body)) + body)
            response = _response(malformed)
            assert response.error_code == ErrorCode.INVALID_JSON

        with _connect(socket_path) as oversized:
            oversized.sendall(struct.pack("!I", MAX_REQUEST_BYTES + 1))
            response = _response(oversized)
            assert response.error_code == ErrorCode.FRAME_TOO_LARGE
    finally:
        _stop_server(server, thread)


def test_unknown_operation_and_raw_sql_escape_hatch_are_rejected(tmp_path):
    server, thread, socket_path = _start_server(tmp_path)
    request = make_request(
        CanonicalWriterOperation.PING,
        {},
        runtime={},
        sequence=1,
        timeout_seconds=2,
    ).to_message()
    request["operation"] = "sql.execute"
    request["payload"] = {"sql": "DELETE FROM canonical_events"}
    try:
        with _connect(socket_path) as conn:
            conn.sendall(encode_frame(request, max_bytes=MAX_REQUEST_BYTES))
            response = _response(conn)
        assert response.ok is False
        assert response.error_code == ErrorCode.UNKNOWN_OPERATION
    finally:
        _stop_server(server, thread)


def test_duplicate_request_id_and_nonincreasing_sequence_are_rejected(tmp_path):
    calls = []
    server, thread, socket_path = _start_server(
        tmp_path,
        dispatcher=_ping_dispatcher(calls),
    )
    first = make_request(
        CanonicalWriterOperation.PING,
        {"value": 1},
        runtime={},
        sequence=1,
        timeout_seconds=2,
    ).to_message()
    second = make_request(
        CanonicalWriterOperation.PING,
        {"value": 2},
        runtime={},
        sequence=1,
        timeout_seconds=2,
    ).to_message()
    try:
        with _connect(socket_path) as conn:
            conn.sendall(encode_frame(first, max_bytes=MAX_REQUEST_BYTES))
            assert _response(conn).ok is True

            conn.sendall(encode_frame(first, max_bytes=MAX_REQUEST_BYTES))
            assert _response(conn).error_code == ErrorCode.REPLAYED_REQUEST

            conn.sendall(encode_frame(second, max_bytes=MAX_REQUEST_BYTES))
            assert _response(conn).error_code == ErrorCode.REPLAYED_REQUEST
        assert len(calls) == 1
    finally:
        _stop_server(server, thread)


def test_replay_guard_rejects_same_request_id_after_reconnect(tmp_path):
    calls = []
    server, thread, socket_path = _start_server(
        tmp_path,
        dispatcher=_ping_dispatcher(calls),
    )
    request = make_request(
        CanonicalWriterOperation.PING,
        {},
        runtime={},
        sequence=1,
        timeout_seconds=2,
    ).to_message()
    try:
        with _connect(socket_path) as first:
            first.sendall(encode_frame(request, max_bytes=MAX_REQUEST_BYTES))
            assert _response(first).ok is True
        with _connect(socket_path) as replay:
            replay.sendall(encode_frame(request, max_bytes=MAX_REQUEST_BYTES))
            assert _response(replay).error_code == ErrorCode.REPLAYED_REQUEST
        assert len(calls) == 1
    finally:
        _stop_server(server, thread)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"protocol": "canonical-writer.v0"}, ErrorCode.UNSUPPORTED_VERSION),
        ({"request_id": "not-a-uuid"}, ErrorCode.INVALID_REQUEST_ID),
        ({"deadline_unix_ms": 1}, ErrorCode.DEADLINE_EXPIRED),
        (
            {"deadline_unix_ms": int((time.time() + 60) * 1000)},
            ErrorCode.DEADLINE_TOO_FAR,
        ),
        ({"token": "must-not-be-accepted"}, ErrorCode.INVALID_REQUEST),
    ],
)
def test_strict_request_metadata_is_enforced(tmp_path, mutation, expected):
    server, thread, socket_path = _start_server(tmp_path)
    request = make_request(
        CanonicalWriterOperation.PING,
        {},
        runtime={},
        sequence=1,
        timeout_seconds=2,
    ).to_message()
    request.update(mutation)
    try:
        with _connect(socket_path) as conn:
            conn.sendall(encode_frame(request, max_bytes=MAX_REQUEST_BYTES))
            assert _response(conn).error_code == expected
    finally:
        _stop_server(server, thread)


def test_duplicate_json_keys_are_rejected(tmp_path):
    server, thread, socket_path = _start_server(tmp_path)
    body = (
        b'{"protocol":"canonical-writer.v1","protocol":"canonical-writer.v1"}'
    )
    try:
        with _connect(socket_path) as conn:
            conn.sendall(struct.pack("!I", len(body)) + body)
            assert _response(conn).error_code == ErrorCode.INVALID_JSON
    finally:
        _stop_server(server, thread)


def test_client_and_server_socket_descriptors_are_not_inheritable(tmp_path):
    server, thread, socket_path = _start_server(tmp_path)
    try:
        client = _client(socket_path, max_reconnect_attempts=0)
        client.call(CanonicalWriterOperation.PING, {}, runtime={})
        client_fd = client.fileno
        assert client_fd >= 0
        assert os.get_inheritable(client_fd) is False
        assert os.get_inheritable(server.fileno) is False

        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os,sys; fd=int(sys.argv[1]); "
                    "\ntry: os.fstat(fd)"
                    "\nexcept OSError: raise SystemExit(0)"
                    "\nraise SystemExit(1)"
                ),
                str(client_fd),
            ],
            check=False,
            close_fds=False,
        )
        assert probe.returncode == 0
        client.close()
    finally:
        _stop_server(server, thread)


def test_canonical_json_is_deterministic_and_never_carries_client_token(tmp_path):
    assert canonical_json_bytes({"z": 1, "a": "ю"}) == (
        '{"a":"ю","z":1}'.encode()
    )
    server, thread, socket_path = _start_server(tmp_path)
    calls = []
    try:
        client = _client(socket_path, max_reconnect_attempts=0)
        result = client.call(
            CanonicalWriterOperation.PING,
            {"safe": True},
            runtime={"platform": "discord"},
        )
        assert result.status == "ok"
        assert "token" not in json.dumps(result.result)
        client.close()
    finally:
        _stop_server(server, thread)


def test_runtime_envelope_is_separate_and_service_binds_authenticated_peer(tmp_path):
    calls = []
    server, thread, socket_path = _start_server(
        tmp_path,
        dispatcher=_ping_dispatcher(calls),
    )
    built = []

    def build_context(runtime):
        built.append(dict(runtime))
        return {**runtime, "context_built": True}

    try:
        client = _client(
            socket_path,
            max_reconnect_attempts=0,
            request_context_builder=build_context,
        )
        result = client.request(
            "ping",
            {"safe": True},
            {"platform": "discord"},
        )
        assert result == {"pong": True}
        assert built == [{"platform": "discord"}]
        payload, context = calls[0]
        assert payload == {"safe": True}
        assert context.runtime["platform"] == "discord"
        assert context.runtime["context_built"] is True
        assert dict(context.runtime["peer"]) == {
            "pid": os.getpid(),
            "uid": os.getuid(),
            "gid": os.getgid(),
        }

        with pytest.raises(ProtocolError) as payload_error:
            client.request("ping", {"runtime": {"peer_pid": 1}}, {})
        assert payload_error.value.code == ErrorCode.INVALID_REQUEST
        with pytest.raises(ProtocolError) as runtime_error:
            client.request("ping", {}, {"peer_pid": 1})
        assert runtime_error.value.code == ErrorCode.INVALID_REQUEST
        client.close()
    finally:
        _stop_server(server, thread)


def test_client_has_no_unauthenticated_server_mode(tmp_path):
    with pytest.raises(TypeError):
        CanonicalWriterClient(tmp_path / "writer.sock")


def test_systemctl_server_pid_provider_uses_strict_bounded_command(monkeypatch):
    observed = []

    class Completed:
        returncode = 0
        stdout = f"{os.getpid()}\n"

    def run(command, **kwargs):
        observed.append((command, kwargs))
        return Completed()

    monkeypatch.setattr("gateway.canonical_writer_client.subprocess.run", run)
    provider = SystemctlServerMainPidProvider(
        systemctl_path="/usr/bin/systemctl",
        timeout_seconds=1.5,
    )
    assert provider.main_pid("muncho-canonical-writer.service") == os.getpid()
    assert observed == [
        (
            [
                "/usr/bin/systemctl",
                "show",
                "--property=MainPID",
                "--value",
                "--",
                "muncho-canonical-writer.service",
            ],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 1.5,
            },
        )
    ]
    assert provider.main_pid("../../attacker.service") is None
    assert len(observed) == 1


@pytest.mark.parametrize(
    "writer_peer",
    [
        ServerPeerCredentials(os.getpid() + 10_000, os.getuid(), os.getgid()),
        ServerPeerCredentials(os.getpid(), os.getuid() + 1, os.getgid()),
    ],
)
def test_client_rejects_wrong_writer_pid_or_uid_before_sending(tmp_path, writer_peer):
    calls = []
    server, thread, socket_path = _start_server(
        tmp_path,
        dispatcher=_ping_dispatcher(calls),
    )
    try:
        client = _client(
            socket_path,
            writer_peer=writer_peer,
            max_reconnect_attempts=0,
        )
        with pytest.raises(CanonicalWriterClientError) as rejected:
            client.request("ping", {}, {})
        assert rejected.value.code == ErrorCode.UNAUTHORIZED_PEER
        assert calls == []
        client.close()
    finally:
        _stop_server(server, thread)


def test_client_rechecks_current_writer_main_pid_before_every_request(tmp_path):
    calls = []
    writer_provider = _StaticMainPidProvider(os.getpid())
    server, thread, socket_path = _start_server(
        tmp_path,
        dispatcher=_ping_dispatcher(calls),
    )
    try:
        client = _client(
            socket_path,
            writer_provider=writer_provider,
            max_reconnect_attempts=0,
        )
        assert client.request("ping", {}, {}) == {"pong": True}
        writer_provider.pid = os.getpid() + 10_000
        with pytest.raises(CanonicalWriterClientError) as rejected:
            client.request("ping", {}, {})
        assert rejected.value.code == ErrorCode.UNAUTHORIZED_PEER
        assert len(calls) == 1
        # Connect authentication plus one check before each request.
        assert writer_provider.calls == 3
        client.close()
    finally:
        _stop_server(server, thread)


def test_persistent_connection_rechecks_current_systemd_main_pid(tmp_path):
    provider = _StaticMainPidProvider(os.getpid())
    calls = []
    server, thread, socket_path = _start_server(
        tmp_path,
        dispatcher=_ping_dispatcher(calls),
        main_pid_provider=provider,
    )
    try:
        client = _client(socket_path, max_reconnect_attempts=0)
        assert client.request("ping", {}, {}) == {"pong": True}
        provider.pid = os.getpid() + 10_000
        with pytest.raises(CanonicalWriterClientError) as rejected:
            client.request("ping", {}, {})
        assert rejected.value.code == ErrorCode.UNAUTHORIZED_PEER
        assert len(calls) == 1
        # Once on connection accept, then once for each received request.
        assert provider.calls == 3
        client.close()
    finally:
        _stop_server(server, thread)


def test_protocol_version_constant_is_explicit():
    assert PROTOCOL_VERSION == "canonical-writer.v1"
    assert "sql" not in {operation.value for operation in CanonicalWriterOperation}
    assert {
        "capability.grant",
        "capability.consume",
        "capability.revoke",
        "capability.revoke_session",
        "projection.read_events",
    }.issubset({operation.value for operation in CanonicalWriterOperation})
