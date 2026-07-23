from __future__ import annotations

import argparse
import asyncio
import json
import socket
import stat
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import session_ipc
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import Platform, SessionEntry, SessionSource, build_session_key
from gateway.session_ipc import (
    GatewaySessionIPCServer,
    SessionIPCRequestError,
    inject_gateway_session,
)
from hermes_cli.subcommands.gateway import build_gateway_parser


SESSION_KEY = "agent:jasper:telegram:dm:7873700813"
SESSION_ID = "20260723_201200_deadbeef"


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


def _runner(entry: SessionEntry | None = None):
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _loaded=True,
        _entries={} if entry is None else {SESSION_KEY: entry},
        _ensure_loaded_locked=lambda: None,
        _generate_session_key=lambda source: build_session_key(
            source,
            profile=source.profile,
        ),
    )
    runner._running_agents = {}
    runner._queued_events = {}
    runner._active_profile_name = lambda: "jasper"
    adapter = SimpleNamespace(
        _pending_messages={},
        _active_sessions={},
        handle_message=AsyncMock(),
    )
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


@pytest.mark.asyncio
async def test_exact_profile_local_idle_route_returns_session_proof_and_dispatches_in_order():
    runner, adapter = _runner(_entry())

    first = await runner._inject_exact_session(
        profile="jasper",
        session_key=SESSION_KEY,
        message="Inspect WORK-8491 and report status.",
    )
    second = await runner._inject_exact_session(
        profile="jasper",
        session_key=SESSION_KEY,
        message="Then preserve FIFO order.",
    )

    assert first == {
        "ok": True,
        "profile": "jasper",
        "session_key": SESSION_KEY,
        "session_id": SESSION_ID,
        "disposition": "queued",
    }
    assert second["session_id"] == SESSION_ID
    assert second["disposition"] == "queued"
    events = [call.args[0] for call in adapter.handle_message.await_args_list]
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

    proof = await runner._inject_exact_session(
        profile="jasper",
        session_key=SESSION_KEY,
        message="Use the corrected constraint.",
    )

    assert proof["disposition"] == "steered"
    assert proof["session_id"] == SESSION_ID
    agent.steer.assert_called_once_with("Use the corrected constraint.")
    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_route_steer_fallback_uses_bounded_fifo_queue():
    runner, adapter = _runner(_entry())
    agent = SimpleNamespace(steer=Mock(return_value=False))
    runner._running_agents[SESSION_KEY] = agent

    for message in ("First queued task.", "Second queued task."):
        proof = await runner._inject_exact_session(
            profile="jasper",
            session_key=SESSION_KEY,
            message=message,
        )
        assert proof["disposition"] == "queued"

    assert [call.args for call in agent.steer.call_args_list] == [
        ("First queued task.",),
        ("Second queued task.",),
    ]
    adapter.handle_message.assert_not_awaited()
    assert adapter._pending_messages[SESSION_KEY].text == "First queued task."
    assert [event.text for event in runner._queued_events[SESSION_KEY]] == [
        "Second queued task."
    ]


@pytest.mark.asyncio
async def test_missing_exact_route_fails_closed():
    runner, adapter = _runner()

    with pytest.raises(SessionIPCRequestError, match="No live gateway route") as exc_info:
        await runner._inject_exact_session(
            profile="jasper",
            session_key=SESSION_KEY,
            message="Do not create a session.",
        )

    assert exc_info.value.code == "route_not_found"
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_route_with_mismatched_embedded_key_fails_closed():
    runner, adapter = _runner(_entry(key="agent:other:telegram:dm:7873700813"))

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await runner._inject_exact_session(
            profile="jasper",
            session_key=SESSION_KEY,
            message="Do not guess the route.",
        )

    assert exc_info.value.code == "route_ambiguous"
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_cross_profile_route_is_rejected_before_dispatch():
    runner, adapter = _runner(_entry())

    with pytest.raises(SessionIPCRequestError) as exc_info:
        await runner._inject_exact_session(
            profile="cleo",
            session_key=SESSION_KEY,
            message="Do not cross profile boundaries.",
        )

    assert exc_info.value.code == "profile_mismatch"
    adapter.handle_message.assert_not_awaited()


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
            json.dumps(
                {
                    "operation": "inject",
                    "profile": "cleo",
                    "session_key": SESSION_KEY,
                    "message": "Cross-profile attempt",
                }
            ).encode("utf-8")
            + b"\n",
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
        ["gateway", "inject", "--session-key", SESSION_KEY, "--message", "Do the task"]
    )

    assert args.gateway_command == "inject"
    assert args.session_key == SESSION_KEY
    assert args.message == "Do the task"
