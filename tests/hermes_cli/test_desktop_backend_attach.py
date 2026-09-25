from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_active_session_registry_home_override_is_opt_in_and_wins_explicit(monkeypatch, tmp_path):
    from hermes_cli import active_sessions

    explicit = tmp_path / "explicit"
    override = tmp_path / "mobile-registry"
    monkeypatch.delenv("HERMES_ACTIVE_SESSIONS_HOME", raising=False)
    assert active_sessions._registry_home(explicit) == explicit

    monkeypatch.setenv("HERMES_ACTIVE_SESSIONS_HOME", str(override))
    assert active_sessions._registry_home(explicit) == override
    assert active_sessions._state_path(explicit) == override / "runtime" / "active_sessions.json"


def test_active_session_override_isolates_an_existing_owner(monkeypatch, tmp_path):
    from hermes_cli.active_sessions import try_acquire_active_session

    root = tmp_path / "root"
    mobile = tmp_path / "mobile"
    monkeypatch.delenv("HERMES_ACTIVE_SESSIONS_HOME", raising=False)
    desktop, refusal = try_acquire_active_session(
        session_id="shared", surface="desktop", config={}, registry_home=root
    )
    assert desktop is not None and refusal is None
    duplicate, refusal = try_acquire_active_session(
        session_id="shared", surface="tui", config={}, registry_home=root
    )
    assert duplicate is None and refusal is not None

    monkeypatch.setenv("HERMES_ACTIVE_SESSIONS_HOME", str(mobile))
    phone, refusal = try_acquire_active_session(
        session_id="shared", surface="tui", config={}, registry_home=root
    )
    try:
        assert phone is not None and refusal is None
        assert phone.state_path == mobile / "runtime" / "active_sessions.json"
    finally:
        if phone is not None:
            phone.release()
        desktop.release()


def _prepare_chat_argv(monkeypatch, tmp_path):
    import hermes_cli.config as config
    import hermes_cli.main_tui_launch as tui_launch
    import hermes_cli.web_server_chat as chat
    import hermes_cli.web_server_profiles as profiles
    import tools.environments.local as local_env

    monkeypatch.setattr(
        tui_launch,
        "_make_tui_argv",
        lambda *_args, **_kwargs: (["node", "fake-tui.js"], tmp_path),
    )
    monkeypatch.setattr(tui_launch, "_apply_tui_python_env", lambda _env: None)
    monkeypatch.setattr(local_env, "build_subprocess_env", lambda **_kwargs: {})
    monkeypatch.setattr(config, "apply_terminal_config_to_env", lambda **_kwargs: None)
    monkeypatch.setattr(config, "read_raw_config", lambda: {})
    monkeypatch.setattr(chat, "_build_gateway_ws_url", lambda: "ws://stock/api/ws?token=stock")
    monkeypatch.setattr(profiles, "_resolve_profile_dir", lambda _name: tmp_path / "profiles" / "worker")
    return chat


def test_pty_attach_env_unset_keeps_stock_gateway(monkeypatch, tmp_path):
    chat = _prepare_chat_argv(monkeypatch, tmp_path)
    monkeypatch.delenv("HERMES_ATTACH_DESKTOP_BACKEND", raising=False)

    _argv, _cwd, env = chat._resolve_chat_argv()

    assert env["HERMES_TUI_GATEWAY_URL"] == "ws://stock/api/ws?token=stock"


def test_pty_attach_env_set_uses_proven_desktop_backend(monkeypatch, tmp_path):
    chat = _prepare_chat_argv(monkeypatch, tmp_path)
    from hermes_cli import desktop_backend_attach as attach

    backend = attach.DesktopBackend(pid=17, port=54321, token="desktop-token")
    monkeypatch.setenv("HERMES_ATTACH_DESKTOP_BACKEND", "1")
    monkeypatch.setattr(attach, "find_desktop_backend", lambda: backend)

    _argv, _cwd, env = chat._resolve_chat_argv()

    assert env["HERMES_TUI_GATEWAY_URL"] == backend.ws_url()


def test_pty_attach_excludes_profile_scoped_chat(monkeypatch, tmp_path):
    chat = _prepare_chat_argv(monkeypatch, tmp_path)
    from hermes_cli import desktop_backend_attach as attach

    monkeypatch.setenv("HERMES_ATTACH_DESKTOP_BACKEND", "1")
    monkeypatch.setattr(
        attach,
        "find_desktop_backend",
        lambda: (_ for _ in ()).throw(AssertionError("profile chat must not discover Desktop")),
    )

    _argv, _cwd, env = chat._resolve_chat_argv(profile="worker")

    assert "HERMES_TUI_GATEWAY_URL" not in env
    assert env["HERMES_HOME"] == str(tmp_path / "profiles" / "worker")


def test_desktop_discovery_rejects_dashboard_and_unproven_serve(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach
    from hermes_cli import process_identity
    import psutil

    default_home = os.environ["HERMES_HOME"]
    processes = [
        SimpleNamespace(info={"pid": 11, "cmdline": ["python", "-m", "hermes_cli.main", "dashboard"]}),
        SimpleNamespace(
            info={"pid": 12, "cmdline": ["python", "-m", "hermes_cli.main", "serve"]},
            environ=lambda: {"HERMES_DESKTOP": "1", "HERMES_HOME": default_home},
        ),
    ]
    probed = []
    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: processes)
    monkeypatch.setattr(
        process_identity,
        "ledger_entries",
        lambda: [{"pid": 12, "purpose": "serve", "profile": "default", "port": 40012}],
    )
    monkeypatch.setattr(attach, "_listening_ports", lambda proc: [40000 + proc.info["pid"]])
    monkeypatch.setattr(attach, "_probe", lambda port: probed.append(port) or None)

    assert attach.find_desktop_backend() is None
    assert probed == [40012]


def test_desktop_discovery_skips_named_profile_and_selects_default(monkeypatch, tmp_path):
    from hermes_cli import desktop_backend_attach as attach
    from hermes_cli import process_identity
    import psutil

    default_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    processes = [
        SimpleNamespace(
            info={
                "pid": 101,
                "cmdline": ["python", "-m", "hermes_cli.main", "--profile", "secret", "serve"],
            },
            # psutil sees the launch environment on macOS. Python's later
            # --profile re-home does not rewrite this process-table value.
            environ=lambda: {"HERMES_DESKTOP": "1", "HERMES_HOME": str(default_home)},
        ),
        SimpleNamespace(
            info={
                "pid": 102,
                "cmdline": ["python", "-m", "hermes_cli.main", "serve"],
            },
            environ=lambda: {"HERMES_DESKTOP": "1", "HERMES_HOME": str(default_home)},
        ),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: processes)
    monkeypatch.setattr(attach, "_listening_ports", lambda proc: [40000 + proc.info["pid"]])
    monkeypatch.setattr(attach, "_probe", lambda port: f"token-for-{port}")
    monkeypatch.setattr(
        process_identity,
        "ledger_entries",
        lambda: [
            {"pid": 101, "purpose": "serve", "profile": "secret", "port": 40101},
            {"pid": 102, "purpose": "serve", "profile": "", "port": 40102},
        ],
    )

    assert attach.find_desktop_backend() == attach.DesktopBackend(
        pid=102, port=40102, token="token-for-40102"
    )


@pytest.mark.parametrize("candidate_home", ["custom", None])
def test_desktop_discovery_denies_custom_or_unknown_home(
    monkeypatch, tmp_path, candidate_home
):
    from hermes_cli import desktop_backend_attach as attach
    from hermes_cli import process_identity
    import psutil

    default_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    environment = {"HERMES_DESKTOP": "1"}
    if candidate_home is not None:
        environment["HERMES_HOME"] = str(tmp_path / candidate_home)
    process = SimpleNamespace(
        info={"pid": 103, "cmdline": ["python", "-m", "hermes_cli.main", "serve"]},
        environ=lambda: environment,
    )
    probed = []
    monkeypatch.setattr(psutil, "process_iter", lambda _attrs: [process])
    monkeypatch.setattr(attach, "_listening_ports", lambda _proc: [40103])
    monkeypatch.setattr(attach, "_probe", lambda port: probed.append(port) or "token")
    monkeypatch.setattr(
        process_identity,
        "ledger_entries",
        lambda: [{"pid": 103, "purpose": "serve", "profile": "default", "port": 40103}],
    )

    assert attach.find_desktop_backend() is None
    assert probed == []


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('<script>__HERMES_SESSION_TOKEN__ = "token"</script>', None),
        ("<h1>Headless backend</h1>", None),
        ('<h1>Headless backend</h1><script>__HERMES_SESSION_TOKEN__ = "token"</script>', "token"),
    ],
)
def test_desktop_probe_requires_headless_marker_and_token(monkeypatch, body, expected):
    from hermes_cli import desktop_backend_attach as attach

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return body.encode()

    monkeypatch.setattr(attach.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    assert attach._probe(43210) == expected


class _ClientSocket:
    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.accepted = False
        self.subprotocol = None
        self.sent = []
        self.block = asyncio.Event()

    async def accept(self, subprotocol=None):
        self.accepted = True
        self.subprotocol = subprotocol

    async def receive(self):
        if self.incoming:
            return self.incoming.pop(0)
        await self.block.wait()
        return {"type": "websocket.disconnect"}

    async def send_text(self, frame):
        self.sent.append(frame)

    async def send_bytes(self, frame):
        self.sent.append(frame)


class _Upstream:
    def __init__(self, incoming=None):
        self.incoming = list(incoming or [])
        self.sent = []
        self.closed = False
        self.block = asyncio.Event()

    async def send(self, frame):
        self.sent.append(frame)

    async def recv(self):
        if self.incoming:
            return self.incoming.pop(0)
        await self.block.wait()
        return None

    async def close(self):
        self.closed = True


def _install_fake_websockets(monkeypatch, connect):
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=connect))


@pytest.mark.asyncio
async def test_sidecar_relay_env_unset_is_inert_and_unaccepted(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach

    client = _ClientSocket()
    monkeypatch.delenv(attach.ATTACH_ENV_VAR, raising=False)
    monkeypatch.setattr(
        attach,
        "find_desktop_backend",
        lambda: (_ for _ in ()).throw(AssertionError("disabled attach must not discover")),
    )

    assert await attach.proxy_sidecar_to_desktop(client) is False
    assert client.accepted is False


@pytest.mark.asyncio
async def test_sidecar_relay_success_forwards_client_frame_and_completes(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach

    backend = attach.DesktopBackend(pid=17, port=54321, token="desktop-token")
    upstream = _Upstream()
    client = _ClientSocket([
        {"type": "websocket.receive", "text": '{"id":"1"}'},
        {"type": "websocket.disconnect"},
    ])

    async def connect(*_args, **_kwargs):
        return upstream

    monkeypatch.setenv(attach.ATTACH_ENV_VAR, "1")
    monkeypatch.setattr(attach, "find_desktop_backend", lambda: backend)
    _install_fake_websockets(monkeypatch, connect)

    assert await attach.proxy_sidecar_to_desktop(client, "hermes-jsonrpc") is True
    assert client.accepted is True
    assert client.subprotocol == "hermes-jsonrpc"
    assert upstream.sent == ['{"id":"1"}']
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_sidecar_relay_dial_failure_is_unaccepted_stock_fallback(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach

    backend = attach.DesktopBackend(pid=17, port=54321, token="desktop-token")
    client = _ClientSocket()

    async def connect(*_args, **_kwargs):
        raise OSError("backend vanished")

    monkeypatch.setenv(attach.ATTACH_ENV_VAR, "1")
    monkeypatch.setattr(attach, "find_desktop_backend", lambda: backend)
    _install_fake_websockets(monkeypatch, connect)

    assert await attach.proxy_sidecar_to_desktop(client) is False
    assert client.accepted is False


@pytest.mark.asyncio
async def test_gateway_ws_uses_stock_handler_when_relay_declines(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach
    from hermes_cli.web_routers import chat_ws
    from tui_gateway import ws as tui_ws

    socket = SimpleNamespace(_hermes_ws_subprotocol="json", _hermes_auth_identity="identity")
    calls = []

    async def allowed(_ws):
        return True

    async def decline(_ws, _subprotocol):
        return False

    async def stock(_ws, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(chat_ws, "_close_unless_sidecar_allowed", allowed)
    monkeypatch.setattr(attach, "proxy_sidecar_to_desktop", decline)
    monkeypatch.setattr(tui_ws, "handle_ws", stock)

    await chat_ws.gateway_ws(socket)

    assert calls == [{"auth_identity": "identity", "subprotocol": "json"}]


@pytest.mark.asyncio
async def test_sidecar_relay_remote_completion_is_forwarded_and_closed(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach

    backend = attach.DesktopBackend(pid=17, port=54321, token="desktop-token")
    upstream = _Upstream(["done", None])
    client = _ClientSocket()

    async def connect(*_args, **_kwargs):
        return upstream

    monkeypatch.setenv(attach.ATTACH_ENV_VAR, "1")
    monkeypatch.setattr(attach, "find_desktop_backend", lambda: backend)
    _install_fake_websockets(monkeypatch, connect)

    assert await attach.proxy_sidecar_to_desktop(client) is True
    assert client.sent == ["done"]
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_sidecar_relay_cancellation_closes_upstream(monkeypatch):
    from hermes_cli import desktop_backend_attach as attach

    backend = attach.DesktopBackend(pid=17, port=54321, token="desktop-token")
    upstream = _Upstream()
    client = _ClientSocket()

    async def connect(*_args, **_kwargs):
        return upstream

    monkeypatch.setenv(attach.ATTACH_ENV_VAR, "1")
    monkeypatch.setattr(attach, "find_desktop_backend", lambda: backend)
    _install_fake_websockets(monkeypatch, connect)

    task = asyncio.create_task(attach.proxy_sidecar_to_desktop(client))
    for _ in range(20):
        if client.accepted:
            break
        await asyncio.sleep(0)
    assert client.accepted is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert upstream.closed is True
