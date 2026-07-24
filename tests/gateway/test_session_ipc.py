from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import socket
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gateway import run as gateway_run
from gateway import session_ipc
from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PendingEventQueue,
    _new_gateway_session_ipc_event,
)
from gateway.run import GatewayRunner
from gateway.session import Platform, SessionEntry, SessionSource, build_session_key
from gateway.session_ipc import (
    GatewaySessionIPCServer,
    SessionIPCRequestError,
    inject_gateway_session,
)
from hermes_cli.gateway import _gateway_command_inner
from hermes_cli.subcommands.gateway import build_gateway_parser


SESSION_KEY = "agent:jasper:telegram:dm:7873700813"
SESSION_ID = "20260723_201200_deadbeef"


class _TestAdapter(BasePlatformAdapter):
    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, *args, **kwargs):
        return None

    async def get_chat_info(self, *args, **kwargs):
        return {}


def _source(*, profile: str = "jasper") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="7873700813",
        chat_type="dm",
        user_id="7873700813",
        user_name="Tyler",
        profile=profile,
    )


def _entry(*, key: str = SESSION_KEY, profile: str = "jasper") -> SessionEntry:
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_key=key,
        session_id=SESSION_ID,
        created_at=now,
        updated_at=now,
        origin=_source(profile=profile),
        platform=Platform.TELEGRAM,
    )


async def _inject(
    runner,
    message: str,
    *,
    expected_session_id: str = SESSION_ID,
    idempotency_key: str | None = None,
):
    return await runner._inject_exact_session(
        profile="jasper",
        session_key=SESSION_KEY,
        expected_session_id=expected_session_id,
        idempotency_key=idempotency_key or str(uuid.uuid4()),
        message=message,
    )


def _runner(entry: SessionEntry | None = None):
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace(
        _lock=threading.RLock(),
        resolve_session_entry=Mock(return_value=entry),
        _loaded=True,
        _entries={} if entry is None else {SESSION_KEY: entry},
        _ensure_loaded_locked=lambda: None,
        _generate_session_key=lambda source: build_session_key(
            source,
            profile=source.profile,
        ),
        _is_session_expired=Mock(return_value=False),
        _is_session_ended_in_db=Mock(return_value=False),
        _compression_tip_for_session_id=Mock(side_effect=lambda value: value),
    )
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._draining = False
    runner._running = True
    runner._active_profile_name = lambda: "jasper"
    pending_events = PendingEventQueue()
    adapter = SimpleNamespace(
        pending_events=pending_events,
        _pending_messages=pending_events.compatibility_heads,
        _active_sessions={},
        accept_internal_task=Mock(return_value=True),
    )
    adapter.get_pending_message = pending_events.pop_next
    runner._adapter_for_source = lambda source: adapter
    return runner, adapter


def _raw_request(socket_path, payload: bytes) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2.0)
        client.connect(str(socket_path))
        client.sendall(payload)
        raw = b""
        while b"\n" not in raw:
            chunk = client.recv(8192)
            if not chunk:
                break
            raw += chunk
    return json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))


def _request_payload(
    *,
    profile: str = "jasper",
    session_key: str = SESSION_KEY,
    expected_session_id: str = SESSION_ID,
    idempotency_key: str | None = None,
    message: str = "Exact local task",
) -> bytes:
    return (
        json.dumps(
            {
                "operation": "inject",
                "profile": profile,
                "session_key": session_key,
                "expected_session_id": expected_session_id,
                "idempotency_key": idempotency_key or str(uuid.uuid4()),
                "message": message,
            }
        ).encode("utf-8")
        + b"\n"
    )


@pytest.mark.asyncio
async def test_exact_profile_local_idle_route_returns_session_proof_and_dispatches_in_order():
    runner, adapter = _runner(_entry())

    first = await _inject(runner, "Inspect WORK-8491 and report status.")
    second = await _inject(runner, "Then preserve FIFO order.")

    assert first == {
        "ok": True,
        "profile": "jasper",
        "session_key": SESSION_KEY,
        "session_id": SESSION_ID,
        "disposition": "queued",
    }
    assert second["session_id"] == SESSION_ID
    assert second["disposition"] == "queued"
    events = [call.args[0] for call in adapter.accept_internal_task.call_args_list]
    assert [event.text for event in events] == [
        "Inspect WORK-8491 and report status.",
        "Then preserve FIFO order.",
    ]
    assert all(isinstance(event, MessageEvent) for event in events)
    assert all(event.internal is True for event in events)
    assert all(event.source.profile == "jasper" for event in events)
    assert all(event.metadata["gateway_session_id"] == SESSION_ID for event in events)


@pytest.mark.asyncio
async def test_active_exact_route_steers_without_platform_send_or_queue():
    runner, adapter = _runner(_entry())
    agent = SimpleNamespace(steer=Mock(return_value=True))
    runner._running_agents[SESSION_KEY] = agent

    proof = await _inject(runner, "Use the corrected constraint.")

    assert proof["disposition"] == "steered"
    assert proof["session_id"] == SESSION_ID
    agent.steer.assert_called_once_with("Use the corrected constraint.")
    adapter.accept_internal_task.assert_not_called()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_route_steer_fallback_uses_bounded_fifo_queue():
    runner, adapter = _runner(_entry())
    agent = SimpleNamespace(steer=Mock(return_value=False))
    runner._running_agents[SESSION_KEY] = agent

    for message in ("First queued task.", "Second queued task."):
        proof = await _inject(runner, message)
        assert proof["disposition"] == "queued"

    assert [call.args for call in agent.steer.call_args_list] == [
        ("First queued task.",),
        ("Second queued task.",),
    ]
    adapter.accept_internal_task.assert_not_called()
    assert [event.text for event in adapter.pending_events.snapshot(SESSION_KEY)] == [
        "First queued task.", "Second queued task.",
    ]


@pytest.mark.asyncio
async def test_internal_task_uses_separate_fifo_and_queue_full_preserves_user_media():
    runner, adapter = _runner(_entry())
    runner._running_agents[SESSION_KEY] = SimpleNamespace(steer=Mock(return_value=False))
    pending_user_event = MessageEvent(
        text="Original user caption",
        message_type=MessageType.PHOTO,
        source=_source(),
        media_urls=["https://example.invalid/original.jpg"],
        media_types=["image/jpeg"],
    )
    adapter.pending_events.enqueue_fifo(SESSION_KEY, pending_user_event)
    for index in range(runner._BUSY_QUEUE_MAX_PENDING - 2):
        adapter.pending_events.enqueue_fifo(
            SESSION_KEY,
            MessageEvent(
                text=f"queued-{index}", message_type=MessageType.TEXT, source=_source()
            ),
        )
    before = pickle.dumps(pending_user_event, protocol=pickle.HIGHEST_PROTOCOL)

    accepted = await _inject(runner, "Separate internal FIFO task")

    assert accepted["disposition"] == "queued"
    assert pickle.dumps(pending_user_event, protocol=pickle.HIGHEST_PROTOCOL) == before
    assert adapter.pending_events.snapshot(SESSION_KEY)[-1].text == "Separate internal FIFO task"
    full_before = pickle.dumps(pending_user_event, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await _inject(runner, "Rejected internal FIFO task")

    assert exc_info.value.code == "queue_full"
    assert pickle.dumps(pending_user_event, protocol=pickle.HIGHEST_PROTOCOL) == full_before
    assert all(
        event.text != "Rejected internal FIFO task"
        for event in adapter.pending_events.snapshot(SESSION_KEY)
    )


def test_concurrent_user_and_ipc_admission_never_exceeds_busy_queue_cap(monkeypatch):
    runner, adapter = _runner(_entry())
    monkeypatch.setattr(runner, "_BUSY_QUEUE_MAX_PENDING", 2)
    adapter.pending_events.enqueue_fifo(
        SESSION_KEY,
        MessageEvent(
            text="already queued",
            message_type=MessageType.TEXT,
            source=_source(),
        ),
    )
    user_event = MessageEvent(
        text="concurrent user follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
    )
    ipc_event = _new_gateway_session_ipc_event(
        text="concurrent IPC follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        internal=True,
        metadata={"gateway_session_ipc_task": True},
    )
    admission_barrier = threading.Barrier(2)
    original_depth = runner._queue_depth
    first_depth_call = threading.local()

    def synchronized_depth(session_key, *, adapter=None):
        depth = original_depth(session_key, adapter=adapter)
        if not getattr(first_depth_call, "seen", False):
            first_depth_call.seen = True
            try:
                admission_barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
        return depth

    monkeypatch.setattr(runner, "_queue_depth", synchronized_depth)
    ipc_result = []
    user_thread = threading.Thread(
        target=runner._queue_or_replace_pending_event,
        args=(SESSION_KEY, user_event),
    )
    ipc_thread = threading.Thread(
        target=lambda: ipc_result.append(
            runner._enqueue_internal_fifo_event(SESSION_KEY, ipc_event, adapter)
        )
    )

    user_thread.start()
    ipc_thread.start()
    user_thread.join(timeout=1.0)
    ipc_thread.join(timeout=1.0)

    assert not user_thread.is_alive()
    assert not ipc_thread.is_alive()
    assert original_depth(SESSION_KEY, adapter=adapter) <= runner._BUSY_QUEUE_MAX_PENDING
    assert len(ipc_result) == 1


def test_trusted_slash_task_survives_drain_as_inert_text_while_user_slash_is_blocked():
    runner, adapter = _runner(_entry())
    trusted = _new_gateway_session_ipc_event(
        text="/restart trusted task",
        message_type=MessageType.TEXT,
        source=_source(),
        internal=True,
        metadata={"gateway_session_ipc_task": True},
    )
    user = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=_source(),
    )

    runner._enqueue_fifo(SESSION_KEY, trusted, adapter)
    drained_trusted = runner._dequeue_and_promote_queued_event(SESSION_KEY, adapter)
    assert drained_trusted is trusted
    assert gateway_run._pending_slash_is_command_leak(
        drained_trusted.text, drained_trusted
    ) is False
    assert drained_trusted.is_command() is False

    runner._enqueue_fifo(SESSION_KEY, user, adapter)
    drained_user = runner._dequeue_and_promote_queued_event(SESSION_KEY, adapter)
    assert drained_user is user
    assert gateway_run._pending_slash_is_command_leak(
        drained_user.text, drained_user
    ) is True
    assert drained_user.is_command() is True


@pytest.mark.asyncio
async def test_missing_exact_route_fails_closed():
    runner, adapter = _runner()

    with pytest.raises(SessionIPCRequestError, match="No live gateway route") as exc_info:
        await _inject(runner, "Do not create a session.")

    assert exc_info.value.code == "route_not_found"
    adapter.accept_internal_task.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_route_with_mismatched_embedded_key_fails_closed():
    runner, adapter = _runner(_entry(key="agent:other:telegram:dm:7873700813"))

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await _inject(runner, "Do not guess the route.")

    assert exc_info.value.code == "route_ambiguous"
    adapter.accept_internal_task.assert_not_called()


@pytest.mark.asyncio
async def test_cross_profile_route_is_rejected_before_dispatch():
    runner, adapter = _runner(_entry())

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await runner._inject_exact_session(
            profile="cleo",
            session_key=SESSION_KEY,
            expected_session_id=SESSION_ID,
            idempotency_key=str(uuid.uuid4()),
            message="Do not cross profile boundaries.",
        )

    assert exc_info.value.code == "profile_mismatch"
    adapter.accept_internal_task.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("task_text", ["restart gateway", "/restart", "/approve always"])
async def test_trusted_internal_task_is_never_command_coerced_or_dispatched_as_control(task_text):
    adapter = _TestAdapter(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
    adapter._message_handler = AsyncMock(return_value="must not run inline")
    adapter._busy_session_handler = AsyncMock(return_value=False)
    adapter._active_sessions = {SESSION_KEY: asyncio.Event()}
    adapter._pending_messages = {}
    adapter._session_tasks = {}
    adapter._background_tasks = set()
    adapter._expected_cancelled_tasks = set()
    adapter._text_debounce = {}
    adapter._busy_text_mode = "interrupt"
    adapter._busy_text_debounce_seconds = 0.0
    adapter._busy_text_hard_cap_seconds = 0.0
    adapter._topic_recovery_fn = None

    event = _new_gateway_session_ipc_event(
        text=task_text,
        message_type=MessageType.TEXT,
        source=_source(),
        internal=True,
        metadata={
            "gateway_session_ipc_task": True,
            "gateway_session_key": SESSION_KEY,
            "gateway_session_id": SESSION_ID,
        },
    )

    with (
        patch("gateway.platforms.base.coerce_plaintext_gateway_command") as coerce,
        patch("gateway.platforms.base.build_session_key", return_value=SESSION_KEY),
    ):
        await adapter.handle_message(event)

    coerce.assert_not_called()
    adapter._message_handler.assert_not_awaited()
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages[SESSION_KEY].text == task_text


@pytest.mark.asyncio
async def test_expected_session_rotation_is_rejected_before_steer_or_queue():
    runner, adapter = _runner(_entry())
    runner._running_agents[SESSION_KEY] = SimpleNamespace(steer=Mock(return_value=True))

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await _inject(runner, "Must not land", expected_session_id="rotated-session")

    assert exc_info.value.code == "route_rotated"
    runner._running_agents[SESSION_KEY].steer.assert_not_called()
    adapter.accept_internal_task.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_state", "expected_code"),
    [
        ("suspended", "session_suspended"),
        ("expired", "session_expired"),
        ("ended", "session_ended"),
        ("stale_parent", "route_rotated"),
    ],
)
async def test_inactive_or_stale_routes_are_rejected_before_acceptance(route_state, expected_code):
    entry = _entry()
    runner, adapter = _runner(entry)
    if route_state == "suspended":
        entry.suspended = True
    elif route_state == "expired":
        runner.session_store._is_session_expired.return_value = True
    elif route_state == "ended":
        runner.session_store._is_session_ended_in_db.return_value = True
    elif route_state == "stale_parent":
        runner.session_store._compression_tip_for_session_id.side_effect = None
        runner.session_store._compression_tip_for_session_id.return_value = "compressed-child"

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await _inject(runner, "Must not land")

    assert exc_info.value.code == expected_code
    adapter.accept_internal_task.assert_not_called()


@pytest.mark.asyncio
async def test_idle_route_does_not_report_queued_when_adapter_claim_fails():
    runner, adapter = _runner(_entry())
    adapter.accept_internal_task.return_value = False

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await _inject(runner, "Claim must be synchronous")

    assert exc_info.value.code == "acceptance_failed"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_profile_socket_and_runtime_directory_are_owner_only_and_unix_only(tmp_path):
    home = tmp_path / "profiles" / "jasper"

    async def handler(**request):
        return {
            "ok": True,
            "profile": "jasper",
            "session_key": request["session_key"],
            "session_id": SESSION_ID,
            "disposition": "queued",
        }

    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=home)
    await server.start()
    try:
        assert server.socket_path == home.resolve() / "run" / "gateway-session.sock"
        assert stat.S_IMODE(server.socket_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(server.socket_path.parent.stat().st_mode) == 0o700
        assert server._server is not None
        assert server._server.sockets
        assert {sock.family for sock in server._server.sockets} == {socket.AF_UNIX}

        proof = await asyncio.to_thread(
            inject_gateway_session,
            profile="jasper",
            session_key=SESSION_KEY,
            expected_session_id=SESSION_ID,
            idempotency_key=str(uuid.uuid4()),
            message="Exact local task",
            hermes_home=home,
        )
        assert proof["session_id"] == SESSION_ID
    finally:
        await server.stop()

    assert not server.socket_path.exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
def test_client_rejects_permissive_profile_socket(tmp_path):
    home = tmp_path / "profiles" / "jasper"
    socket_path = home / "run" / "gateway-session.sock"
    socket_path.parent.mkdir(parents=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    socket_path.chmod(0o666)
    try:
        with pytest.raises(SessionIPCRequestError) as exc_info:
            inject_gateway_session(
                profile="jasper",
                session_key=SESSION_KEY,
                expected_session_id=SESSION_ID,
                idempotency_key=str(uuid.uuid4()),
                message="Unsafe socket must be rejected",
                hermes_home=home,
            )
    finally:
        listener.close()

    assert exc_info.value.code == "unsafe_socket"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_socket_rejects_cross_profile_and_malformed_json(tmp_path):
    calls = []

    async def handler(**request):
        calls.append(request)
        return {"ok": True}

    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=tmp_path)
    await server.start()
    try:
        malformed = await asyncio.to_thread(_raw_request, server.socket_path, b"{not-json}\n")
        assert malformed["ok"] is False
        assert malformed["error"]["code"] == "invalid_request"

        cross_profile = await asyncio.to_thread(
            _raw_request,
            server.socket_path,
            _request_payload(profile="cleo", message="Cross-profile attempt"),
        )
        assert cross_profile["ok"] is False
        assert cross_profile["error"]["code"] == "profile_mismatch"
        assert calls == []
    finally:
        await server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_socket_rejects_json_beyond_request_bound_without_dispatch(tmp_path):
    calls = []

    async def handler(**request):
        calls.append(request)
        return {"ok": True}

    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=tmp_path)
    await server.start()
    try:
        oversized = b"{" + b"x" * session_ipc._MAX_REQUEST_BYTES + b"}\n"
        response = await asyncio.to_thread(_raw_request, server.socket_path, oversized)
        assert response["ok"] is False
        assert response["error"]["code"] == "request_too_large"
        assert calls == []
    finally:
        await server.stop()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_unknown_peer_uid_fails_closed_without_dispatch(tmp_path, monkeypatch):
    handler = AsyncMock(return_value={"ok": True})
    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=tmp_path)
    monkeypatch.setattr(server, "_peer_uid", lambda _writer: None)
    await server.start()
    try:
        response = await asyncio.to_thread(
            _raw_request,
            server.socket_path,
            _request_payload(),
        )
    finally:
        await server.stop()

    assert response["ok"] is False
    assert response["error"]["code"] == "peer_credentials_unavailable"
    handler.assert_not_awaited()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_client_timeout_retry_with_same_key_delivers_once(tmp_path):
    calls = 0

    async def handler(**request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {
            "ok": True,
            "profile": request["profile"],
            "session_key": request["session_key"],
            "session_id": request["expected_session_id"],
            "disposition": "queued",
        }

    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=tmp_path)
    await server.start()
    key = "timeout-retry-same-task"
    try:
        with pytest.raises((TimeoutError, socket.timeout, SessionIPCRequestError)):
            await asyncio.to_thread(
                inject_gateway_session,
                profile="jasper",
                session_key=SESSION_KEY,
                expected_session_id=SESSION_ID,
                idempotency_key=key,
                message="Deliver exactly once",
                hermes_home=tmp_path,
                timeout=0.01,
            )
        await asyncio.sleep(0.08)
        proof = await asyncio.to_thread(
            inject_gateway_session,
            profile="jasper",
            session_key=SESSION_KEY,
            expected_session_id=SESSION_ID,
            idempotency_key=key,
            message="Deliver exactly once",
            hermes_home=tmp_path,
        )
    finally:
        await server.stop()

    assert proof["disposition"] == "queued"
    assert calls == 1


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_server_timeout_cancels_handler_and_returns_closed_failure(tmp_path, monkeypatch):
    cancelled = asyncio.Event()

    async def handler(**_request):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(session_ipc, "_HANDLER_TIMEOUT_SECONDS", 0.01)
    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=tmp_path)
    await server.start()
    try:
        response = await asyncio.to_thread(
            _raw_request,
            server.socket_path,
            _request_payload(idempotency_key="server-timeout"),
        )
    finally:
        await server.stop()

    assert response["ok"] is False
    assert response["error"]["code"] == "request_timeout"
    assert cancelled.is_set()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_blocking_production_lookup_times_out_without_later_injection(tmp_path, monkeypatch):
    runner, adapter = _runner(_entry())
    lookup_started = threading.Event()

    def blocking_lookup(_session_id):
        lookup_started.set()
        time.sleep(0.2)
        return False

    runner.session_store._is_session_ended_in_db = Mock(side_effect=blocking_lookup)
    monkeypatch.setattr(session_ipc, "_HANDLER_TIMEOUT_SECONDS", 0.01)
    server = GatewaySessionIPCServer(
        runner._inject_exact_session,
        profile="jasper",
        hermes_home=tmp_path,
    )
    await server.start()
    try:
        started = time.monotonic()
        response = await asyncio.to_thread(
            _raw_request,
            server.socket_path,
            _request_payload(idempotency_key="blocking-production-lookup"),
        )
        elapsed = time.monotonic() - started
        assert lookup_started.is_set()
        assert response["ok"] is False
        assert response["error"]["code"] == "request_timeout"
        assert elapsed < 0.1
        await asyncio.sleep(0.25)
    finally:
        await server.stop()

    adapter.accept_internal_task.assert_not_called()
    assert adapter.pending_events.snapshot(SESSION_KEY) == ()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_reused_idempotency_key_with_different_payload_is_rejected(tmp_path):
    handler = AsyncMock(
        return_value={
            "ok": True,
            "profile": "jasper",
            "session_key": SESSION_KEY,
            "session_id": SESSION_ID,
            "disposition": "queued",
        }
    )
    server = GatewaySessionIPCServer(handler, profile="jasper", hermes_home=tmp_path)
    await server.start()
    try:
        first = await asyncio.to_thread(
            _raw_request,
            server.socket_path,
            _request_payload(idempotency_key="conflict", message="first"),
        )
        second = await asyncio.to_thread(
            _raw_request,
            server.socket_path,
            _request_payload(idempotency_key="conflict", message="different"),
        )
    finally:
        await server.stop()

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "idempotency_conflict"
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_unique_idempotency_admission_never_exceeds_hard_cap(monkeypatch):
    release = asyncio.Event()

    async def handler(**request):
        await release.wait()
        return {"ok": True, "idempotency_key": request["idempotency_key"]}

    monkeypatch.setattr(session_ipc, "_MAX_IDEMPOTENCY_RECORDS", 3)
    server = GatewaySessionIPCServer(handler, profile="jasper")
    tasks = [
        asyncio.create_task(
            server._dispatch_idempotent(
                f"concurrent-{index}",
                {
                    "profile": "jasper",
                    "session_key": SESSION_KEY,
                    "expected_session_id": SESSION_ID,
                    "idempotency_key": f"concurrent-{index}",
                    "message": f"task-{index}",
                },
            )
        )
        for index in range(5)
    ]
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(server._idempotency_tasks) <= session_ipc._MAX_IDEMPOTENCY_RECORDS
        assert (
            len(server._idempotency_tasks) + len(server._idempotency_results)
            <= session_ipc._MAX_IDEMPOTENCY_RECORDS
        )
        assert len(server._idempotency_fingerprints) <= session_ipc._MAX_IDEMPOTENCY_RECORDS
    finally:
        release.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    await asyncio.sleep(0)
    assert server._idempotency_tasks == {}
    assert (
        len(server._idempotency_tasks) + len(server._idempotency_results)
        <= session_ipc._MAX_IDEMPOTENCY_RECORDS
    )
    assert len(server._idempotency_fingerprints) <= session_ipc._MAX_IDEMPOTENCY_RECORDS
    assert sum(
        isinstance(outcome, SessionIPCRequestError) and outcome.code == "server_busy"
        for outcome in outcomes
    ) == 2


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (ConnectionRefusedError("connect rejected"), "gateway_unavailable"),
        (BrokenPipeError("peer rejected before send"), "transport_error"),
        (socket.timeout("response deadline"), "request_timeout"),
    ],
)
def test_client_socket_failures_are_normalized_and_cli_exits_nonzero(
    tmp_path, monkeypatch, capsys, failure, expected_code
):
    socket_path = session_ipc.gateway_session_socket_path(tmp_path)
    socket_path.parent.mkdir(parents=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    socket_path.chmod(0o600)
    try:
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        if isinstance(failure, ConnectionRefusedError):
            client.connect.side_effect = failure
        elif isinstance(failure, BrokenPipeError):
            client.sendall.side_effect = failure
        else:
            client.recv.side_effect = failure
        monkeypatch.setattr(session_ipc.socket, "socket", Mock(return_value=client))

        with pytest.raises(SessionIPCRequestError) as exc_info:
            inject_gateway_session(
                profile="jasper",
                session_key=SESSION_KEY,
                expected_session_id=SESSION_ID,
                idempotency_key="stable-client-error",
                message="Must fail stably",
                hermes_home=tmp_path,
            )
        assert exc_info.value.code == expected_code

        monkeypatch.setattr(
            session_ipc,
            "inject_gateway_session",
            Mock(side_effect=exc_info.value),
        )
        with (
            patch("hermes_cli.profiles.get_active_profile_name", return_value="jasper"),
            pytest.raises(SystemExit) as cli_exit,
        ):
            _gateway_command_inner(
                SimpleNamespace(
                    gateway_command="inject",
                    session_key=SESSION_KEY,
                    expected_session_id=SESSION_ID,
                    idempotency_key="stable-client-error",
                    message="Must fail stably",
                )
            )
        assert cli_exit.value.code == 1
        assert json.loads(capsys.readouterr().out)["error"]["code"] == expected_code
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_stop_does_not_unlink_replacement_socket(tmp_path):
    server = GatewaySessionIPCServer(
        AsyncMock(return_value={"ok": True}),
        profile="jasper",
        hermes_home=tmp_path,
    )
    await server.start()
    original = server.socket_path.lstat()
    server.socket_path.unlink()

    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    replacement.bind(str(server.socket_path))
    replacement.listen(1)
    replacement_stat = server.socket_path.lstat()
    try:
        assert (replacement_stat.st_dev, replacement_stat.st_ino) != (
            original.st_dev,
            original.st_ino,
        )
        with pytest.raises(SessionIPCRequestError) as exc_info:
            await server.stop()
        assert exc_info.value.code == "socket_replaced"
        assert server.socket_path.exists()
        current = server.socket_path.lstat()
        assert (current.st_dev, current.st_ino) == (
            replacement_stat.st_dev,
            replacement_stat.st_ino,
        )
    finally:
        replacement.close()
        server.socket_path.unlink(missing_ok=True)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
def test_startup_stale_cleanup_never_unlinks_replacement_between_check_and_cleanup(
    tmp_path, monkeypatch
):
    server = GatewaySessionIPCServer(
        AsyncMock(return_value={"ok": True}),
        profile="jasper",
        hermes_home=tmp_path,
    )
    server.socket_path.parent.mkdir(parents=True)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(server.socket_path))
    stale.close()
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    replacement_path = server.socket_path.with_name("replacement.sock")
    replacement.bind(str(replacement_path))
    replacement.listen(1)
    replacement_identity = replacement_path.lstat().st_dev, replacement_path.lstat().st_ino
    original_replace = session_ipc.os.replace
    replaced = False

    def replace_before_quarantine(source, destination):
        nonlocal replaced
        if source == server.socket_path and not replaced:
            replaced = True
            server.socket_path.unlink()
            original_replace(replacement_path, server.socket_path)
        return original_replace(source, destination)

    monkeypatch.setattr(session_ipc.os, "replace", replace_before_quarantine)
    try:
        with pytest.raises(SessionIPCRequestError):
            server._prepare_socket_directory()
        assert replaced is True
        assert server.socket_path.exists()
        assert stat.S_ISSOCK(server.socket_path.lstat().st_mode)
        current = server.socket_path.lstat()
        assert (current.st_dev, current.st_ino) == replacement_identity
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.5)
            client.connect(str(server.socket_path))
    finally:
        replacement.close()
        server.socket_path.unlink(missing_ok=True)
        for quarantined in server.socket_path.parent.glob(
            f".{server.socket_path.name}.stale-*"
        ):
            quarantined.unlink(missing_ok=True)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
@pytest.mark.asyncio
async def test_stop_quarantines_before_identity_check_and_preserves_live_replacement(
    tmp_path, monkeypatch
):
    server = GatewaySessionIPCServer(
        AsyncMock(return_value={"ok": True}), profile="jasper", hermes_home=tmp_path
    )
    await server.start()
    replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    replacement_path = server.socket_path.with_name("stop-replacement.sock")
    replacement.bind(str(replacement_path))
    replacement.listen(1)
    info = replacement_path.lstat()
    replacement_identity = info.st_dev, info.st_ino
    original_replace = session_ipc.os.replace
    replaced = False

    def replace_before_quarantine(source, destination):
        nonlocal replaced
        if source == server.socket_path and not replaced:
            replaced = True
            server.socket_path.unlink()
            original_replace(replacement_path, server.socket_path)
        return original_replace(source, destination)

    monkeypatch.setattr(session_ipc.os, "replace", replace_before_quarantine)
    try:
        with pytest.raises(SessionIPCRequestError) as exc_info:
            await server.stop()
        assert exc_info.value.code == "socket_replaced"
        assert replaced is True
        current = server.socket_path.lstat()
        assert (current.st_dev, current.st_ino) == replacement_identity
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.5)
            client.connect(str(server.socket_path))
    finally:
        replacement.close()
        server.socket_path.unlink(missing_ok=True)
        for quarantined in server.socket_path.parent.glob(
            f".{server.socket_path.name}.*-*"
        ):
            quarantined.unlink(missing_ok=True)


def test_adapter_loop_and_ipc_worker_share_one_atomic_fifo_protocol(monkeypatch):
    runner, _ = _runner(_entry())
    adapter = _TestAdapter(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
    runner._adapter_for_source = lambda source: adapter
    runner._running_agents[SESSION_KEY] = SimpleNamespace(steer=Mock(return_value=False))
    runner._BUSY_QUEUE_MAX_PENDING = 3
    seed = MessageEvent(text="seed", message_type=MessageType.PHOTO, source=_source())
    user = MessageEvent(text="user", message_type=MessageType.PHOTO, source=_source())
    ipc = _new_gateway_session_ipc_event(
        text="ipc", message_type=MessageType.TEXT, source=_source(), internal=True,
        metadata={"gateway_session_ipc_task": True},
    )
    adapter.pending_events.enqueue_fifo(SESSION_KEY, seed, max_depth=3)
    merge_entered = threading.Event()
    release_merge = threading.Event()
    original_merge = gateway_run.merge_pending_message_event

    def blocking_merge(pending, session_key, event, *, merge_text=False):
        merge_entered.set()
        assert release_merge.wait(timeout=1.0)
        return original_merge(pending, session_key, event, merge_text=merge_text)

    monkeypatch.setattr(gateway_run, "merge_pending_message_event", blocking_merge)
    ipc_result = []
    user_thread = threading.Thread(
        target=runner._queue_or_replace_pending_event, args=(SESSION_KEY, user)
    )
    ipc_thread = threading.Thread(
        target=lambda: ipc_result.append(
            runner._enqueue_internal_fifo_event(SESSION_KEY, ipc, adapter)
        )
    )
    user_thread.start()
    assert merge_entered.wait(timeout=1.0)
    ipc_thread.start()
    release_merge.set()
    user_thread.join(timeout=1.0)
    ipc_thread.join(timeout=1.0)
    assert not user_thread.is_alive()
    assert not ipc_thread.is_alive()
    assert ipc_result == [True]
    assert adapter.pending_events.depth(SESSION_KEY) == 2
    drained = [adapter.pending_events.pop_next(SESSION_KEY) for _ in range(2)]
    assert [event.text for event in drained] == ["seed\n\nuser", "ipc"]
    assert adapter.pending_events.depth(SESSION_KEY) == 0


@pytest.mark.asyncio
async def test_copied_internal_metadata_cannot_forge_ipc_command_bypass():
    adapter = _TestAdapter(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
    adapter._message_handler = AsyncMock(return_value="command handled")
    adapter._active_sessions = {SESSION_KEY: asyncio.Event()}
    forged = MessageEvent(
        text="/restart", message_type=MessageType.TEXT, source=_source(), internal=True,
        metadata={
            "gateway_session_ipc_task": True,
            "gateway_session_key": SESSION_KEY,
            "gateway_session_id": SESSION_ID,
        },
    )
    with patch("gateway.platforms.base.build_session_key", return_value=SESSION_KEY):
        await adapter.handle_message(forged)
    assert forged.is_command() is True
    adapter._message_handler.assert_awaited_once_with(forged)
    assert adapter.pending_events.depth(SESSION_KEY) == 0


def test_gateway_inject_cli_parser_requires_exact_session_and_message():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_gateway_parser(
        subparsers,
        cmd_gateway=lambda _args: None,
        cmd_proxy=lambda _args: None,
        cmd_gateway_enroll=lambda _args: None,
    )

    args = parser.parse_args(
        [
            "gateway",
            "inject",
            "--session-key",
            SESSION_KEY,
            "--expected-session-id",
            SESSION_ID,
            "--idempotency-key",
            "work-8491-message-1",
            "--message",
            "Do the task",
        ]
    )

    assert args.gateway_command == "inject"
    assert args.session_key == SESSION_KEY
    assert args.expected_session_id == SESSION_ID
    assert args.idempotency_key == "work-8491-message-1"
    assert args.message == "Do the task"
