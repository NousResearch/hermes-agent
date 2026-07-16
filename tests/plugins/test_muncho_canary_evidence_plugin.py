from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import stat
import struct
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

import plugins.muncho_canary_evidence as canary_module
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.discord_edge_protocol import (
    DiscordEdgeErrorCode,
    DiscordEdgeProtocolError,
    parse_request,
)
from plugins.muncho_canary_evidence import (
    ACK_SCHEMA,
    CONFIG_SCHEMA,
    FRAME_SCHEMA,
    GOAL_CONFIG_SCHEMA,
    GOAL_FRAME_SCHEMA,
    CanaryEvidenceError,
    CanaryEvidencePlugin,
    CanaryEvidenceHookMultiplexer,
    CollectorEndpoint,
    EdgeEndpoint,
    GoalContinuationEvidencePlugin,
    GoalObserverConfig,
    PeerIdentity,
    SocketIdentity,
    _collector_exchange,
    _run_private_probe,
    _sha256_bytes,
    _socket_identity,
    load_config,
    load_goal_config,
)


NOW = dt.datetime(2026, 7, 13, 12, 0, tzinfo=dt.timezone.utc)
NOW_MS = int(NOW.timestamp() * 1_000)
RUN_ID = "11111111-1111-4111-8111-111111111111"
CASE_ID = "case:full-canary-1"
SESSION_ID = "session-full-canary-1"
TURN_ID = "turn-full-canary-1"
TASK_ID = "task-full-canary-1"
RELEASE_SHA = "a" * 40
RELEASE_SHA256 = "b" * 64
SESSION_SHA256 = "d" * 64
SERVICE_SHA256 = "f" * 64
SOCKET_SHA256 = "1" * 64
MODULE_SHA256 = "2" * 64
EVENT_ID = "22222222-2222-4222-8222-222222222222"
REQUEST_ID = "33333333-3333-4333-8333-333333333333"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _fixture() -> dict[str, Any]:
    prompt = "Complete the owner-approved multi-step live canary."
    return {
        "schema": "muncho-full-canary-e2e-fixture.v1",
        "canary_run_id": RUN_ID,
        "release_sha": RELEASE_SHA,
        "release_artifact_sha256": RELEASE_SHA256,
        "api_session_key_sha256": SESSION_SHA256,
        "valid_from_unix_ms": NOW_MS - 10_000,
        "valid_until_unix_ms": NOW_MS + 120_000,
        "case_id": CASE_ID,
        "owner_discord_user_id": "1279454038731264061",
        "source": {
            "platform": "api_server",
            "control_protocol": "authenticated_loopback_api_server.v1",
            "host": "127.0.0.1",
            "port": 8642,
            "session_create_endpoint": "/api/sessions",
            "chat_stream_endpoint_template": (
                "/api/sessions/{session_id}/chat/stream"
            ),
        },
        "model_route": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "model": "gpt-5.6-sol",
            "initial_effort": "high",
            "elevated_effort": "max",
        },
        "task_policy": {
            "minimum_completed_steps": 3,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        },
        "public_routeback": {
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": "1279454038731264061",
                "channel_id": "1279454038731264062",
            },
            "canonical_idempotency_key": "full-canary-routeback-1",
        },
        "discord_public_keys": {
            "writer_capability_ed25519_hex": "3" * 64,
            "edge_receipt_ed25519_hex": "4" * 64,
        },
    }


def _write_config(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    fixture = _fixture()
    fixture_body = _canonical_bytes(fixture)
    fixture_sha256 = hashlib.sha256(fixture_body).hexdigest()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(fixture_body)
    os.chown(fixture_path, -1, os.getgid())
    fixture_path.chmod(0o440)
    config = {
        "schema": CONFIG_SCHEMA,
        "release_sha": RELEASE_SHA,
        "release_sha256": RELEASE_SHA256,
        "canary_run_id": RUN_ID,
        "case_id": CASE_ID,
        "fixture_path": str(fixture_path),
        "fixture_sha256": fixture_sha256,
        "collector": {
            "socket_path": "/run/muncho-full-canary/collector.sock",
            "expected_pid": 41,
            "expected_uid": 0,
            "expected_gid": 0,
            "socket_owner_uid": 0,
            "socket_owner_gid": os.getgid(),
            "socket_mode": "0660",
            "service_identity_sha256": SERVICE_SHA256,
            "connect_timeout_ms": 1_000,
            "ack_timeout_ms": 3_000,
        },
        "discord_edge": {
            "socket_path": "/run/muncho-discord-egress/edge.sock",
            "expected_pid": 42,
            "expected_uid": 1001,
            "expected_gid": 1001,
            "socket_owner_uid": 1001,
            "socket_owner_gid": os.getgid(),
            "socket_mode": "0660",
            "service_identity_sha256": "5" * 64,
            "connect_timeout_ms": 1_000,
            "response_timeout_ms": 2_000,
        },
    }
    config_path = tmp_path / "observer.json"
    config_path.write_bytes(_canonical_bytes(config))
    os.chown(config_path, -1, os.getgid())
    config_path.chmod(0o440)
    return config_path, config, fixture


def _rewrite_sealed(path: Path, body: bytes) -> None:
    path.chmod(0o640)
    path.write_bytes(body)
    os.chown(path, -1, os.getgid())
    path.chmod(0o440)


def _loaded_config(tmp_path: Path):
    config_path, _, _ = _write_config(tmp_path)
    return load_config(
        config_path,
        expected_owner_uid=os.getuid(),
        expected_owner_gid=os.getgid(),
    )


def _goal_config() -> GoalObserverConfig:
    return GoalObserverConfig(
        config_sha256="3" * 64,
        release_sha=RELEASE_SHA,
        release_sha256=RELEASE_SHA256,
        run_id=RUN_ID,
        fixture_sha256="4" * 64,
        valid_from_unix_ms=NOW_MS - 10_000,
        valid_until_unix_ms=NOW_MS + 120_000,
        public_target={
            "target_type": "public_guild_channel",
            "guild_id": "1282725267068157972",
            "channel_id": "1526858760100909066",
        },
        owner_user_id="1279454038731264061",
        model_route={
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "model": "gpt-5.6-sol",
            "fallback_configured": False,
        },
        collector=CollectorEndpoint(
            socket_path=canary_module.DEFAULT_GOAL_COLLECTOR_SOCKET_PATH,
            expected_peer=PeerIdentity(41, 0, 0),
            socket_identity=SocketIdentity(0, os.getgid(), 0o660),
            service_identity_sha256=SERVICE_SHA256,
            connect_timeout_ms=1_000,
            ack_timeout_ms=3_000,
        ),
        api_observer_retirement={
            "marker_path": str(
                canary_module.DEFAULT_API_OBSERVER_RETIREMENT_PATH
            ),
            "marker_sha256": "5" * 64,
            "marker_file_sha256": "9" * 64,
            "api_observer_config_path": str(canary_module.DEFAULT_CONFIG_PATH),
            "api_observer_config_sha256": "6" * 64,
            "goal_config_authority_sha256": "7" * 64,
        },
    )


def test_default_config_group_rejects_missing_posix_gid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(canary_module.os, "getgid")
    with pytest.raises(CanaryEvidenceError, match="config_invalid"):
        canary_module._process_gid()


@pytest.fixture
def short_socket_dir():
    # macOS has a much shorter sockaddr_un path limit than Linux. Resolve the
    # /tmp symlink up front because the production identity check rejects a
    # symlinked parent path by design.
    with tempfile.TemporaryDirectory(prefix="mce-", dir="/tmp") as raw:
        yield Path(raw).resolve()


class RecordingCollector:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.raw: list[bytes] = []

    def __call__(
        self, _endpoint: CollectorEndpoint, body: bytes
    ) -> Mapping[str, Any]:
        frame = json.loads(body)
        self.raw.append(body)
        self.frames.append(frame)
        digest = hashlib.sha256(body).hexdigest()
        return {
            "schema": ACK_SCHEMA,
            "sequence": frame["sequence"],
            "accepted": True,
            "frame_sha256": digest,
            "collector_receipt_sha256": hashlib.sha256(
                b"collector-receipt:" + body
            ).hexdigest(),
        }


class Writer:
    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []

    def __call__(self, operation, payload, **kwargs):
        self.calls.append((operation, dict(payload), dict(kwargs)))
        if operation is CanonicalWriterOperation.CASE_QUERY:
            return {
                "request_id": "44444444-4444-4444-8444-444444444444",
                "status": "ok",
                "success": True,
                "case_id": CASE_ID,
                "events": [{"event_type": "route_back.sent"}],
            }
        raise AssertionError(operation)


def _runtime_envelope() -> Mapping[str, Any]:
    return {
        "platform": "api_server",
        "session_id": SESSION_ID,
        "session_key_sha256": SESSION_SHA256,
    }


def _probe(_endpoint, _fixture, identity, observed_at):
    return {
        "discord_edge_service_identity_sha256": "5" * 64,
        "socket_identity_sha256": identity,
        "attempt_frame_sha256": "6" * 64,
        "attempted_operation": "public.message.send",
        "attempted_target_type": "private_channel",
        "connection_closed_without_response": True,
        "signed_receipt_observed": False,
        "observed_at_unix_ms": observed_at,
    }


def _plugin(
    tmp_path: Path,
    *,
    runtime_envelope=_runtime_envelope,
):
    config = _loaded_config(tmp_path)
    collector = RecordingCollector()
    writer = Writer(config)
    plugin = CanaryEvidencePlugin(
        config,
        collector_transport=collector,
        socket_inspector=lambda _path, _identity: SOCKET_SHA256,
        edge_probe=_probe,
        writer_call=writer,
        runtime_envelope=runtime_envelope,
        clock_ms=lambda: NOW_MS,
    )
    plugin.start(module_origin="/sealed/release/plugin.py", module_sha256=MODULE_SHA256)
    return plugin, collector, writer


def _start(plugin: CanaryEvidencePlugin) -> None:
    plugin.on_session_start(
        session_id=SESSION_ID,
        platform="api_server",
        model="gpt-5.6-sol",
    )


def _pre(plugin: CanaryEvidencePlugin, api_id: str, count: int, effort: str) -> None:
    plugin.pre_api_request(
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        task_id=TASK_ID,
        api_request_id=api_id,
        platform="api_server",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        api_call_count=count,
        started_at=NOW.timestamp(),
        request={"body": {"reasoning": {"effort": effort}}},
    )


def _post(
    plugin: CanaryEvidencePlugin, api_id: str, tool_call_ids: list[str]
) -> None:
    plugin.post_api_request(
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        task_id=TASK_ID,
        api_request_id=api_id,
        platform="api_server",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        ended_at=NOW.timestamp(),
        finish_reason="tool_calls",
        response_model="gpt-5.6-sol",
        response={
            "assistant_message": {
                "tool_calls": [{"id": value} for value in tool_call_ids]
            }
        },
    )


def test_config_is_exact_root_gateway_dac_and_read_only(tmp_path, monkeypatch):
    config_path, _, _ = _write_config(tmp_path)
    opened_flags: list[int] = []
    real_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        opened_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    loaded = load_config(
        config_path,
        expected_owner_uid=os.getuid(),
        expected_owner_gid=os.getgid(),
    )

    assert loaded.release_sha256 == RELEASE_SHA256
    assert loaded.collector.expected_peer == PeerIdentity(41, 0, 0)
    assert loaded.collector.socket_identity == SocketIdentity(0, os.getgid(), 0o660)
    assert loaded.fixture.value["release_artifact_sha256"] == RELEASE_SHA256
    assert opened_flags
    assert all(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT) == 0 for flags in opened_flags)

    config_path.chmod(0o640)
    with pytest.raises(CanaryEvidenceError, match="config_invalid"):
        load_config(
            config_path,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=os.getgid(),
        )


def test_config_rejects_gid_session_digest_and_release_binding_drift(tmp_path):
    config_path, config, fixture = _write_config(tmp_path)
    with pytest.raises(CanaryEvidenceError, match="config_invalid"):
        load_config(
            config_path,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=os.getgid() + 1,
        )

    fixture["api_session_key_sha256"] = "wrong"
    fixture_body = _canonical_bytes(fixture)
    _rewrite_sealed(Path(config["fixture_path"]), fixture_body)
    config["fixture_sha256"] = hashlib.sha256(fixture_body).hexdigest()
    _rewrite_sealed(config_path, _canonical_bytes(config))
    with pytest.raises(CanaryEvidenceError, match="fixture_invalid"):
        load_config(
            config_path,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=os.getgid(),
        )

    fixture["api_session_key_sha256"] = SESSION_SHA256
    fixture["release_artifact_sha256"] = "9" * 64
    fixture_body = _canonical_bytes(fixture)
    _rewrite_sealed(Path(config["fixture_path"]), fixture_body)
    config["fixture_sha256"] = hashlib.sha256(fixture_body).hexdigest()
    _rewrite_sealed(config_path, _canonical_bytes(config))
    with pytest.raises(CanaryEvidenceError, match="config_fixture_binding_invalid"):
        load_config(
            config_path,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=os.getgid(),
        )


def test_peer_and_socket_identity_are_separate_and_substitution_is_rejected(
    short_socket_dir,
):
    socket_path = short_socket_dir / "collector.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chown(socket_path, -1, os.getgid())
    socket_path.chmod(0o660)
    server.listen(1)
    body = b'{"mechanical":"frame"}'
    frame_sha256 = hashlib.sha256(body).hexdigest()

    def serve() -> None:
        connection, _ = server.accept()
        try:
            header = connection.recv(4)
            (size,) = struct.unpack("!I", header)
            assert connection.recv(size) == body
            ack = _canonical_bytes(
                {
                    "schema": ACK_SCHEMA,
                    "sequence": 1,
                    "accepted": True,
                    "frame_sha256": frame_sha256,
                    "collector_receipt_sha256": "8" * 64,
                }
            )
            connection.sendall(struct.pack("!I", len(ack)) + ack)
        finally:
            connection.close()

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        socket_identity = SocketIdentity(os.getuid(), os.getgid(), 0o660)
        endpoint = CollectorEndpoint(
            socket_path=socket_path,
            expected_peer=PeerIdentity(999, 0, 0),
            socket_identity=socket_identity,
            service_identity_sha256=SERVICE_SHA256,
            connect_timeout_ms=1_000,
            ack_timeout_ms=1_000,
        )
        assert endpoint.expected_peer.uid == 0
        assert endpoint.socket_identity.owner_gid == os.getgid()
        assert _socket_identity(socket_path, socket_identity)
        ack = _collector_exchange(
            endpoint,
            body,
            peer_getter=lambda _sock: endpoint.expected_peer,
        )
        assert ack["frame_sha256"] == frame_sha256
        with pytest.raises(CanaryEvidenceError, match="socket_identity_invalid"):
            _socket_identity(
                socket_path,
                SocketIdentity(os.getuid(), os.getgid() + 1, 0o660),
            )
    finally:
        worker.join(timeout=2)
        server.close()
    assert not worker.is_alive()


def test_plugin_ready_precedes_session_binding_and_private_probe_ack_barrier(tmp_path):
    plugin, collector, _writer = _plugin(tmp_path)
    _start(plugin)

    assert [frame["event"] for frame in collector.frames] == [
        "plugin_ready",
        "api_session_bound",
        "private_target_probe_ready",
        "private_target_probe_result",
    ]
    ready = collector.frames[0]
    assert ready["schema"] == FRAME_SCHEMA
    assert ready["sequence"] == 1
    assert ready["session_id"] is None and ready["turn_id"] is None
    assert ready["payload"]["gateway_pid"] == os.getpid()
    assert ready["payload"]["module_sha256"] == MODULE_SHA256
    assert ready["payload"]["collector_socket_identity_sha256"] == SOCKET_SHA256
    assert collector.frames[2]["payload"]["collector_snapshot_barrier"] == "before_probe"
    assert collector.frames[3]["payload"]["connection_closed_without_response"] is True


def test_session_binding_failure_is_evidenced_and_stops_boundary_probe(tmp_path):
    plugin, collector, writer = _plugin(
        tmp_path,
        runtime_envelope=lambda: {
            "platform": "api_server",
            "session_id": SESSION_ID,
            "session_key_sha256": "0" * 64,
        },
    )
    _start(plugin)

    assert [frame["event"] for frame in collector.frames] == [
        "plugin_ready",
        "api_session_bound",
    ]
    assert collector.frames[-1]["payload"] == {
        "success": False,
        "failure_code": "session_binding_failed",
    }
    assert writer.calls == []


def test_high_to_max_hooks_project_receipts_and_end_with_case_readback(tmp_path):
    plugin, collector, writer = _plugin(tmp_path)
    _start(plugin)
    private_semantic_step = "do-not-copy-this-model-authored-step"

    _pre(plugin, "api-request-1", 0, "high")
    _post(plugin, "api-request-1", ["todo-call-1"])
    plugin.post_tool_call(
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        task_id=TASK_ID,
        api_request_id="api-request-1",
        tool_call_id="todo-call-1",
        tool_name="todo",
        args={
            "todos": [{"content": private_semantic_step, "status": "pending"}],
            "reasoning": {"effort": "max", "reason_code": "complexity"},
        },
        result={
            "success": True,
            "reasoning_control": {
                "requested_effort": "max",
                "applied_effort": "max",
            },
        },
        duration_ms=4,
        status="success",
    )

    _pre(plugin, "api-request-2", 1, "max")
    _post(plugin, "api-request-2", ["canonical-call-1", "routeback-call-1"])
    plugin.post_tool_call(
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        task_id=TASK_ID,
        api_request_id="api-request-2",
        tool_call_id="canonical-call-1",
        tool_name="canonical_event_append",
        args={"case_id": CASE_ID},
        result={"success": True, "event_id": EVENT_ID},
        duration_ms=8,
        status="success",
    )
    plugin.post_tool_call(
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        task_id=TASK_ID,
        api_request_id="api-request-2",
        tool_call_id="routeback-call-1",
        tool_name="route_back_execute",
        args={"case_id": CASE_ID},
        result={
            "success": True,
            "outcome": "sent",
            "receipt": {"readback_verified": True},
        },
        duration_ms=12,
        status="success",
    )
    plugin.on_session_end(
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        task_id=TASK_ID,
        completed=True,
        interrupted=False,
        model="gpt-5.6-sol",
        platform="api_server",
    )

    pre_frames = [frame for frame in collector.frames if frame["event"] == "pre_api_request"]
    assert [frame["payload"]["reasoning_effort"] for frame in pre_frames] == [
        "high",
        "max",
    ]
    tool_frames = [frame for frame in collector.frames if frame["event"] == "post_tool_call"]
    assert tool_frames[0]["payload"]["reasoning_control"]["applied_effort"] == "max"
    assert "todos" not in tool_frames[0]["payload"]
    assert private_semantic_step.encode() not in b"\n".join(collector.raw)
    assert tool_frames[1]["payload"]["result_projection"]["event_id"] == EVENT_ID
    assert tool_frames[2]["payload"]["result_projection"]["outcome"] == "sent"
    assert [frame["event"] for frame in collector.frames[-2:]] == [
        "canonical_case_readback",
        "session_end",
    ]
    assert writer.calls[-1][0] is CanonicalWriterOperation.CASE_QUERY
    assert collector.frames[-1]["payload"]["terminal_fields_source"] == (
        "authenticated_loopback_sse"
    )


def test_private_probe_reaches_protocol_forbidden_target_and_observes_no_receipt(
    tmp_path, short_socket_dir
):
    config = _loaded_config(tmp_path)
    socket_path = short_socket_dir / "edge.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chown(socket_path, -1, os.getgid())
    socket_path.chmod(0o660)
    server.listen(1)
    observed: dict[str, Any] = {}

    def serve() -> None:
        connection, _ = server.accept()
        try:
            header = connection.recv(4)
            (size,) = struct.unpack("!I", header)
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                chunk = connection.recv(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            value = json.loads(b"".join(chunks))
            try:
                parse_request(value, now_unix_ms=NOW_MS)
            except DiscordEdgeProtocolError as exc:
                observed["code"] = exc.code
        finally:
            connection.close()

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        identity = SocketIdentity(os.getuid(), os.getgid(), 0o660)
        endpoint = EdgeEndpoint(
            socket_path=socket_path,
            expected_peer=PeerIdentity(os.getpid(), os.getuid(), os.getgid()),
            socket_identity=identity,
            service_identity_sha256="5" * 64,
            connect_timeout_ms=1_000,
            response_timeout_ms=1_000,
        )
        expected = _socket_identity(socket_path, identity)
        receipt = _run_private_probe(
            endpoint,
            config.fixture,
            expected,
            NOW_MS,
            peer_getter=lambda _sock: endpoint.expected_peer,
        )
    finally:
        worker.join(timeout=2)
        server.close()

    assert not worker.is_alive()
    assert observed["code"] is DiscordEdgeErrorCode.FORBIDDEN_TARGET
    assert receipt["connection_closed_without_response"] is True
    assert receipt["signed_receipt_observed"] is False


def _goal_plugin() -> tuple[GoalContinuationEvidencePlugin, RecordingCollector]:
    collector = RecordingCollector()
    plugin = GoalContinuationEvidencePlugin(
        _goal_config(),
        collector_transport=collector,
        socket_inspector=lambda _path, _identity: SOCKET_SHA256,
        clock_ms=lambda: NOW_MS,
    )
    plugin.start(
        module_origin="/sealed/release/plugin.py",
        module_sha256=MODULE_SHA256,
    )
    plugin.on_session_start(
        session_id="discord-session-1",
        model="gpt-5.6-sol",
        platform="discord",
    )
    return plugin, collector


def _goal_pre_post(
    plugin: GoalContinuationEvidencePlugin,
    *,
    api_id: str = "goal-api-1",
    turn_id: str = "goal-turn-1",
    tool_ids: tuple[str, ...] = ("goal-todo-1",),
) -> None:
    plugin.pre_api_request(
        session_id="discord-session-1",
        turn_id=turn_id,
        task_id="goal-task-1",
        api_request_id=api_id,
        platform="discord",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        api_call_count=0,
        started_at=NOW.timestamp(),
        request={
            "body": {
                "instructions": "private stable system prompt",
                "tools": [{"type": "function", "name": "todo"}],
                "reasoning": {"effort": "max"},
            }
        },
    )
    plugin.post_api_request(
        session_id="discord-session-1",
        turn_id=turn_id,
        api_request_id=api_id,
        platform="discord",
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        ended_at=NOW.timestamp(),
        finish_reason="tool_calls",
        response_model="gpt-5.6-sol",
        response={
            "assistant_message": {
                "content": "private model prose",
                "tool_calls": [{"id": item} for item in tool_ids],
            }
        },
    )


def test_goal_observer_projects_structured_outcome_without_prose() -> None:
    plugin, collector = _goal_plugin()
    _goal_pre_post(plugin)
    plugin.post_tool_call(
        session_id="discord-session-1",
        turn_id="goal-turn-1",
        api_request_id="goal-api-1",
        tool_call_id="goal-todo-1",
        tool_name="todo",
        args={
            "goal_outcome": {
                "status": "continue",
                "reason": "private model-authored reason",
            },
            "todos": [{"content": "private task step", "status": "pending"}],
        },
        result={"goal_outcome": {"recorded": True}},
        status="ok",
    )
    plugin.on_session_end(
        session_id="discord-session-1",
        turn_id="goal-turn-1",
        completed=True,
        interrupted=False,
        model="gpt-5.6-sol",
        platform="discord",
    )

    assert [item["event"] for item in collector.frames] == [
        "goal_plugin_ready",
        "goal_pre_api_request",
        "goal_post_api_request",
        "goal_model_outcome",
        "goal_turn_end",
    ]
    assert collector.frames[1]["payload"]["reasoning_effort"] == "max"
    assert collector.frames[3]["payload"]["outcome"] == "continue"
    joined = b"\n".join(collector.raw)
    for prohibited in (
        b"private stable system prompt",
        b"private model prose",
        b"private model-authored reason",
        b"private task step",
    ):
        assert prohibited not in joined
    assert all(item["schema"] == GOAL_FRAME_SCHEMA for item in collector.frames)


def test_goal_observer_projects_verified_routeback_terminal_event() -> None:
    plugin, collector = _goal_plugin()
    _goal_pre_post(plugin, tool_ids=("route-tool-1",))
    canonical_content_sha256 = hashlib.sha256(b"route-content").hexdigest()
    plugin.post_tool_call(
        session_id="discord-session-1",
        turn_id="goal-turn-1",
        api_request_id="goal-api-1",
        tool_call_id="route-tool-1",
        tool_name="route_back_execute",
        args={
            "case_id": "case:goal-routeback",
            "message": "private delivered message",
        },
        result={
            "success": True,
            "status": "ROUTE_BACK_EXECUTE_SENT",
            "route_back_record": {
                "event_id": "61111111-1111-4111-8111-111111111111",
                "case_id": "case:goal-routeback",
                "event_type": "route_back.sent",
                "canonical_content_sha256": canonical_content_sha256,
                "idempotency_key": "goal-routeback-terminal",
                "readback_verified": True,
            },
        },
        status="ok",
    )

    frame = collector.frames[-1]
    assert frame["event"] == "goal_canonical_event"
    assert frame["payload"]["event_type"] == "route_back.sent"
    assert frame["payload"]["canonical_content_sha256"] == (
        canonical_content_sha256
    )
    assert b"private delivered message" not in collector.raw[-1]


def test_goal_observer_rejects_unseen_duplicate_and_cross_turn_tool_ids() -> None:
    plugin, _collector = _goal_plugin()
    _goal_pre_post(plugin)
    common = {
        "session_id": "discord-session-1",
        "api_request_id": "goal-api-1",
        "tool_name": "todo",
        "args": {"goal_outcome": {"status": "continue", "reason": "reason"}},
        "result": {"goal_outcome": {"recorded": True}},
        "status": "ok",
    }
    with pytest.raises(CanaryEvidenceError, match="goal_tool_observation_invalid"):
        plugin.post_tool_call(
            **common,
            turn_id="goal-turn-1",
            tool_call_id="unseen-tool-id",
        )
    with pytest.raises(CanaryEvidenceError, match="goal_tool_observation_invalid"):
        plugin.post_tool_call(
            **common,
            turn_id="another-turn",
            tool_call_id="goal-todo-1",
        )
    plugin.post_tool_call(
        **common,
        turn_id="goal-turn-1",
        tool_call_id="goal-todo-1",
    )
    with pytest.raises(CanaryEvidenceError, match="goal_tool_observation_invalid"):
        plugin.post_tool_call(
            **common,
            turn_id="goal-turn-1",
            tool_call_id="goal-todo-1",
        )


def test_goal_config_rejects_non_uuid_run_and_route_substitution(tmp_path: Path) -> None:
    config = _goal_config()
    raw = {
        "schema": GOAL_CONFIG_SCHEMA,
        "release_sha": config.release_sha,
        "release_sha256": config.release_sha256,
        "run_id": config.run_id,
        "fixture_sha256": config.fixture_sha256,
        "valid_from_unix_ms": config.valid_from_unix_ms,
        "valid_until_unix_ms": config.valid_until_unix_ms,
        "public_target": dict(config.public_target),
        "owner_user_id": config.owner_user_id,
        "model_route": dict(config.model_route),
        "collector": {
            "socket_path": str(config.collector.socket_path),
            "expected_pid": config.collector.expected_peer.pid,
            "expected_uid": config.collector.expected_peer.uid,
            "expected_gid": config.collector.expected_peer.gid,
            "socket_owner_uid": config.collector.socket_identity.owner_uid,
            "socket_owner_gid": config.collector.socket_identity.owner_gid,
            "socket_mode": "0660",
            "service_identity_sha256": config.collector.service_identity_sha256,
            "connect_timeout_ms": config.collector.connect_timeout_ms,
            "ack_timeout_ms": config.collector.ack_timeout_ms,
        },
    }
    goal_authority_sha256 = hashlib.sha256(_canonical_bytes(raw)).hexdigest()
    marker_unsigned = {
        "schema": canary_module.API_OBSERVER_RETIREMENT_SCHEMA,
        "release_sha": config.release_sha,
        "release_sha256": config.release_sha256,
        "run_id": config.run_id,
        "fixture_sha256": config.fixture_sha256,
        "api_observer_config_path": str(canary_module.DEFAULT_CONFIG_PATH),
        "api_observer_config_sha256": "6" * 64,
        "goal_config_authority_sha256": goal_authority_sha256,
        "historical_api_observer_terminal": True,
        "message_content_recorded": False,
    }
    marker_sha256 = hashlib.sha256(_canonical_bytes(marker_unsigned)).hexdigest()
    marker_value = {**marker_unsigned, "marker_sha256": marker_sha256}
    raw["api_observer_retirement"] = {
        "marker_path": str(canary_module.DEFAULT_API_OBSERVER_RETIREMENT_PATH),
        "marker_sha256": marker_sha256,
        "marker_file_sha256": hashlib.sha256(
            _canonical_bytes(marker_value)
        ).hexdigest(),
        "api_observer_config_path": str(canary_module.DEFAULT_CONFIG_PATH),
        "api_observer_config_sha256": "6" * 64,
        "goal_config_authority_sha256": goal_authority_sha256,
    }
    path = tmp_path / "goal-observer.json"
    path.write_bytes(_canonical_bytes(raw))
    os.chown(path, -1, os.getgid())
    path.chmod(0o440)
    loaded = load_goal_config(
        path,
        expected_owner_uid=os.getuid(),
        expected_owner_gid=os.getgid(),
    )
    assert loaded.run_id == RUN_ID

    for mutate in (
        lambda value: value.__setitem__("run_id", "not-a-uuid"),
        lambda value: value["model_route"].__setitem__(
            "base_url", "https://example.invalid"
        ),
    ):
        candidate = json.loads(json.dumps(raw))
        mutate(candidate)
        _rewrite_sealed(path, _canonical_bytes(candidate))
        with pytest.raises(CanaryEvidenceError, match="goal_config_invalid"):
            load_goal_config(
                path,
                expected_owner_uid=os.getuid(),
                expected_owner_gid=os.getgid(),
            )


def test_api_observer_retirement_marker_is_exact_and_fail_closed(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o700)
    api_path = sealed / "api-observer.json"
    api_raw = _canonical_bytes({"schema": "historical-api-observer"})
    api_path.write_bytes(api_raw)
    api_path.chmod(0o440)
    marker_path = sealed / "retired.json"
    base = _goal_config()
    api_sha256 = hashlib.sha256(api_raw).hexdigest()
    goal_authority_sha256 = "8" * 64
    marker_unsigned = {
        "schema": canary_module.API_OBSERVER_RETIREMENT_SCHEMA,
        "release_sha": base.release_sha,
        "release_sha256": base.release_sha256,
        "run_id": base.run_id,
        "fixture_sha256": base.fixture_sha256,
        "api_observer_config_path": str(api_path),
        "api_observer_config_sha256": api_sha256,
        "goal_config_authority_sha256": goal_authority_sha256,
        "historical_api_observer_terminal": True,
        "message_content_recorded": False,
    }
    marker = {
        **marker_unsigned,
        "marker_sha256": hashlib.sha256(
            _canonical_bytes(marker_unsigned)
        ).hexdigest(),
    }
    marker_path.write_bytes(_canonical_bytes(marker))
    marker_path.chmod(0o440)
    config = GoalObserverConfig(
        **{
            **base.__dict__,
            "api_observer_retirement": {
                "marker_path": str(marker_path),
                "marker_sha256": marker["marker_sha256"],
                "marker_file_sha256": hashlib.sha256(
                    _canonical_bytes(marker)
                ).hexdigest(),
                "api_observer_config_path": str(api_path),
                "api_observer_config_sha256": api_sha256,
                "goal_config_authority_sha256": goal_authority_sha256,
            },
        }
    )
    owner_gid = marker_path.lstat().st_gid

    observed = canary_module.validate_api_observer_retirement(
        config,
        expected_owner_uid=os.getuid(),
        expected_owner_gid=owner_gid,
    )
    assert observed == marker

    stale = GoalObserverConfig(
        **{
            **config.__dict__,
            "run_id": "22222222-2222-4222-8222-222222222222",
        }
    )
    with pytest.raises(CanaryEvidenceError, match="api_observer_retirement_invalid"):
        canary_module.validate_api_observer_retirement(
            stale,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=owner_gid,
        )

    marker["message_content_recorded"] = True
    _rewrite_sealed(marker_path, _canonical_bytes(marker))
    with pytest.raises(CanaryEvidenceError, match="api_observer_retirement_invalid"):
        canary_module.validate_api_observer_retirement(
            config,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=owner_gid,
        )
    marker_path.unlink()
    with pytest.raises(CanaryEvidenceError, match="api_observer_retirement_invalid"):
        canary_module.validate_api_observer_retirement(
            config,
            expected_owner_uid=os.getuid(),
            expected_owner_gid=owner_gid,
        )


def test_hook_multiplexer_exposes_one_bound_exact_surface() -> None:
    api = object.__new__(CanaryEvidencePlugin)
    goal = object.__new__(GoalContinuationEvidencePlugin)
    api._session_id = None
    goal._session_id = None
    multiplexer = CanaryEvidenceHookMultiplexer(api, goal)
    for name in (
        "pre_api_request",
        "post_api_request",
        "post_tool_call",
        "on_session_start",
        "on_session_end",
    ):
        callback = getattr(multiplexer, name)
        assert callback.__self__ is multiplexer
        assert callback.__name__ == name


def test_manifest_has_only_observer_hooks_and_no_capability_surface():
    root = Path(__file__).parents[2]
    manifest = yaml.safe_load(
        (root / "plugins/muncho_canary_evidence/plugin.yaml").read_text()
    )
    assert manifest == {
        "name": "muncho_canary_evidence",
        "version": "0.1.0",
        "description": "Collect sealed, non-semantic Muncho canary evidence.",
        "author": "SkyVision / Adventico",
        "kind": "standalone",
        "hooks": [
            "pre_api_request",
            "post_api_request",
            "post_tool_call",
            "on_session_start",
            "on_session_end",
        ],
    }
    assert "tools" not in manifest
    assert "middleware" not in manifest
    assert "context" not in manifest


def test_register_requires_ready_ack_before_hook_only_registration(monkeypatch):
    calls: list[tuple[str, Any]] = []

    class StubPlugin:
        def __init__(self, config):
            calls.append(("construct", config))

        def start(self, **identity):
            calls.append(("ready", identity))

        def pre_api_request(self, **_kwargs):
            return None

        def post_api_request(self, **_kwargs):
            return None

        def post_tool_call(self, **_kwargs):
            return None

        def on_session_start(self, **_kwargs):
            return None

        def on_session_end(self, **_kwargs):
            return None

    class Context:
        def register_hook(self, name, callback):
            calls.append(("hook:" + name, callback))

    sentinel = object()
    monkeypatch.setattr(canary_module, "_PLUGIN", None)
    monkeypatch.setattr(canary_module, "_API_PLUGIN", None)
    monkeypatch.setattr(canary_module, "_GOAL_PLUGIN", None)
    monkeypatch.setattr(canary_module, "load_config", lambda: sentinel)
    monkeypatch.setattr(
        canary_module,
        "_module_identity",
        lambda: ("/sealed/release/plugin.py", MODULE_SHA256),
    )
    monkeypatch.setattr(canary_module, "CanaryEvidencePlugin", StubPlugin)

    canary_module.register(Context())

    assert [name for name, _value in calls] == [
        "construct",
        "ready",
        "hook:pre_api_request",
        "hook:post_api_request",
        "hook:post_tool_call",
        "hook:on_session_start",
        "hook:on_session_end",
    ]
    assert calls[1][1] == {
        "module_origin": "/sealed/release/plugin.py",
        "module_sha256": MODULE_SHA256,
    }


def test_register_skips_only_historical_api_after_exact_retirement(monkeypatch):
    calls: list[str] = []

    class GoalStub:
        def __init__(self, _config):
            calls.append("goal_construct")
            self._session_id = None

        def start(self, **_identity):
            calls.append("goal_ready")

        def pre_api_request(self, **_kwargs):
            return None

        def post_api_request(self, **_kwargs):
            return None

        def post_tool_call(self, **_kwargs):
            return None

        def on_session_start(self, **_kwargs):
            return None

        def on_session_end(self, **_kwargs):
            return None

    class ApiForbidden:
        def __init__(self, _config):
            raise AssertionError("retired API observer was reconstructed")

    class Context:
        def register_hook(self, _name, _callback):
            calls.append("hook")

    sentinel = _goal_config()
    monkeypatch.setattr(canary_module, "_PLUGIN", None)
    monkeypatch.setattr(canary_module, "_API_PLUGIN", None)
    monkeypatch.setattr(canary_module, "_GOAL_PLUGIN", None)
    monkeypatch.setattr(
        canary_module.os.path,
        "lexists",
        lambda path: Path(path)
        in {
            canary_module.DEFAULT_GOAL_CONFIG_PATH,
            canary_module.DEFAULT_CONFIG_PATH,
            canary_module.DEFAULT_API_OBSERVER_RETIREMENT_PATH,
        },
    )
    monkeypatch.setattr(canary_module, "load_goal_config", lambda: sentinel)
    monkeypatch.setattr(
        canary_module,
        "validate_api_observer_retirement",
        lambda config: calls.append("retirement_validated") or config,
    )
    monkeypatch.setattr(
        canary_module,
        "_module_identity",
        lambda: ("/sealed/release/plugin.py", MODULE_SHA256),
    )
    monkeypatch.setattr(canary_module, "CanaryEvidencePlugin", ApiForbidden)
    monkeypatch.setattr(canary_module, "GoalContinuationEvidencePlugin", GoalStub)

    canary_module.register(Context())

    assert calls[:3] == [
        "retirement_validated",
        "goal_construct",
        "goal_ready",
    ]
    assert canary_module._API_PLUGIN is None
    assert calls.count("hook") == 5
