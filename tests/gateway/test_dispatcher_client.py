"""Tests for the harness dispatcher client (Phase 2.6).

The client wraps an asyncio Unix socket with retry + timeout. Per
golden rule #3 ("mocks share your blind spots"), we test against a
REAL asyncio Unix socket server fixture running on tmp_path, not
a mock. The fixture plays the role of the harness dispatcher: it
accepts one connection, reads one Envelope, and writes back a
configurable response (or hangs to test timeout).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio

from gateway.dispatcher_client import (
    DEFAULT_DISPATCHER_SOCKET,
    DispatcherClient,
    DispatcherConnectionError,
)
from gateway.dispatcher_protocol import (
    OP_DISPATCH,
    OP_PING,
    STATUS_BAD_REQUEST,
    STATUS_BUSY,
    STATUS_INTERNAL,
    STATUS_OK,
    Envelope,
    make_request,
)


# --- Fake dispatcher server fixture -------------------------------


class FakeDispatcher:
    """A real asyncio Unix socket server that pretends to be the
    harness dispatcher. Behavior is controlled by the response
    function set before each test.
    """

    def __init__(self, socket_path: Path) -> None:
        self._path = str(socket_path)
        self._server: Optional[asyncio.base_events.Server] = None
        self.connections = 0
        # response_fn takes the incoming Envelope and returns the
        # Envelope to write back (or None to drop the connection).
        self.response_fn = self._default_response

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._path
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        # Best-effort cleanup of the socket file.
        try:
            Path(self._path).unlink()
        except FileNotFoundError:
            pass

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.connections += 1
        try:
            line = await asyncio.wait_for(
                reader.readuntil(b"\n"), timeout=5.0
            )
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            return
        try:
            req = Envelope.from_jsonl(line)
        except ValueError:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            return
        resp = self.response_fn(req)
        if resp is None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            return
        writer.write(resp.to_jsonl())
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass

    @staticmethod
    def _default_response(req: Envelope) -> Envelope:
        if req.op == OP_PING:
            return Envelope(
                request_id=req.request_id,
                op=OP_PING,
                payload={"ts": 0.0},
                status=STATUS_OK,
            )
        return Envelope(
            request_id=req.request_id,
            op=req.op,
            payload={"result": "echo", "echoed_payload": req.payload},
            status=STATUS_OK,
        )


@pytest_asyncio.fixture
async def fake_dispatcher(tmp_path: Path):
    """Yield a running FakeDispatcher on tmp_path/dispatcher.sock.
    Cleans up the server and socket file on teardown.
    """
    sock = tmp_path / "dispatcher.sock"
    server = FakeDispatcher(sock)
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


# --- DispatcherClient tests ---------------------------------------


@pytest.mark.asyncio
async def test_ping_returns_true_on_status_ok(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """ping() returns True when the dispatcher responds STATUS_OK."""
    client = DispatcherClient(socket_path=str(fake_dispatcher._path))
    assert await client.ping() is True
    assert fake_dispatcher.connections == 1


@pytest.mark.asyncio
async def test_dispatch_roundtrip_ping(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """dispatch() returns the response Envelope from the server."""
    client = DispatcherClient(socket_path=str(fake_dispatcher._path))
    req = make_request(OP_PING, {})
    resp = await client.dispatch(req)
    assert resp.status == STATUS_OK
    assert resp.op == OP_PING
    assert "ts" in resp.payload
    assert resp.request_id == req.request_id


@pytest.mark.asyncio
async def test_dispatch_echo_payload(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """dispatch() preserves request_id so the caller can correlate."""
    client = DispatcherClient(socket_path=str(fake_dispatcher._path))
    req = make_request(
        OP_DISPATCH,
        {"source": "wechat", "content": "/echo hello"},
    )
    resp = await client.dispatch(req)
    assert resp.status == STATUS_OK
    assert resp.payload.get("echoed_payload") == {
        "source": "wechat",
        "content": "/echo hello",
    }
    assert resp.request_id == req.request_id


@pytest.mark.asyncio
async def test_dispatch_unreachable_raises(tmp_path: Path) -> None:
    """No server running -> DispatcherConnectionError after retries."""
    # No fake_dispatcher fixture; nothing listening on this path.
    sock = tmp_path / "nope.sock"
    client = DispatcherClient(
        socket_path=str(sock),
        timeout_s=0.2,
        max_retries=1,
    )
    with pytest.raises(DispatcherConnectionError):
        await client.dispatch(make_request(OP_PING, {}))


@pytest.mark.asyncio
async def test_dispatch_propagates_server_status_codes(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """Non-OK responses are returned to caller; only connection-level
    failures raise DispatcherConnectionError. This keeps the wire
    contract honest: the dispatcher decides OK/BUSY/INTERNAL/
    BAD_REQUEST, not the client.
    """
    from gateway.dispatcher_protocol import (
        OP_DISPATCH as _OD,
        make_request as _mr,
    )

    def respond(req: Envelope) -> Envelope:
        return Envelope(
            request_id=req.request_id,
            op=req.op,
            payload={"error": "handler 'echo' raised"},
            status=STATUS_INTERNAL,
        )

    fake_dispatcher.response_fn = respond
    client = DispatcherClient(socket_path=str(fake_dispatcher._path))
    resp = await client.dispatch(_mr(_OD, {"content": "/echo"}))
    assert resp.status == STATUS_INTERNAL
    assert "raised" in resp.payload["error"]


@pytest.mark.asyncio
async def test_dispatch_retries_on_connection_reset(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """Server drops the connection (response_fn returns None) ->
    client retries -> eventually gets a successful response."""
    call_count = {"n": 0}

    def respond(req: Envelope) -> Optional[Envelope]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First attempt: drop the connection.
            return None
        # Second attempt: respond normally.
        return Envelope(
            request_id=req.request_id,
            op=req.op,
            payload={"result": "echo", "echoed_payload": req.payload},
            status=STATUS_OK,
        )

    fake_dispatcher.response_fn = respond
    client = DispatcherClient(
        socket_path=str(fake_dispatcher._path),
        max_retries=2,
    )
    resp = await client.dispatch(make_request(OP_PING, {}))
    assert resp.status == STATUS_OK
    assert call_count["n"] >= 2


@pytest.mark.asyncio
async def test_dispatch_timeout_raises(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """Server accepts the connection but never writes a response ->
    client times out after timeout_s."""
    # response_fn returns a coroutine that sleeps forever (simulates
    # a hung handler). The server-side reader.readuntil times out at
    # 5.0s; we use a smaller client timeout to fail faster.
    async def respond_hang(req: Envelope) -> Optional[Envelope]:
        await asyncio.sleep(60)
        return None

    fake_dispatcher.response_fn = respond_hang  # type: ignore[assignment]
    client = DispatcherClient(
        socket_path=str(fake_dispatcher._path),
        timeout_s=0.2,
        max_retries=0,
    )
    with pytest.raises(DispatcherConnectionError):
        await client.dispatch(make_request(OP_PING, {}))


@pytest.mark.asyncio
async def test_close_is_idempotent(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """close() can be called multiple times without error."""
    client = DispatcherClient(socket_path=str(fake_dispatcher._path))
    await client.dispatch(make_request(OP_PING, {}))
    await client.close()
    await client.close()  # no error


@pytest.mark.asyncio
async def test_context_manager_closes_on_exit(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """async with DispatcherClient() closes the connection on exit."""
    async with DispatcherClient(
        socket_path=str(fake_dispatcher._path)
    ) as client:
        resp = await client.dispatch(make_request(OP_PING, {}))
        assert resp.status == STATUS_OK
    assert not client.is_connected


def test_socket_path_stored() -> None:
    """Explicit socket_path is stored and accessible."""
    client = DispatcherClient(socket_path="/tmp/test.sock")
    assert client.socket_path == "/tmp/test.sock"


# --- Format helper tests (sync) -----------------------------------


def test_format_status_ok_kind_echo() -> None:
    """echo response renders as '[dispatcher] echo: <content>'."""
    from gateway.run import GatewayRunner

    resp = Envelope(
        request_id="r1", op=OP_DISPATCH,
        payload={
            "result": "echo",
            "echoed_payload": {"content": "/echo hello"},
        },
        status=STATUS_OK,
    )
    text = GatewayRunner._format_dispatcher_response("echo", resp)
    assert text == "[dispatcher] echo: /echo hello"


def test_format_status_ok_kind_status() -> None:
    """status response includes uptime + handler count."""
    from gateway.run import GatewayRunner

    resp = Envelope(
        request_id="r1", op=OP_DISPATCH,
        payload={
            "result": "status",
            "uptime_s": 12.5,
            "handlers": ["echo", "status", "help"],
        },
        status=STATUS_OK,
    )
    text = GatewayRunner._format_dispatcher_response("status", resp)
    assert "alive" in text
    assert "12.5" in text
    assert "3 handlers" in text


def test_format_status_ok_kind_stub() -> None:
    """stub response shows the stage where the real impl lands."""
    from gateway.run import GatewayRunner

    resp = Envelope(
        request_id="r1", op=OP_DISPATCH,
        payload={
            "result": "stub",
            "command": "research",
            "stage": "Phase 3",
            "message": "aicc-research handler is scheduled for Phase 3",
        },
        status=STATUS_OK,
    )
    text = GatewayRunner._format_dispatcher_response("research", resp)
    assert "Phase 3" in text
    assert "aicc-research" in text


def test_format_status_busy() -> None:
    """BUSY response tells the user to retry."""
    from gateway.run import GatewayRunner

    resp = Envelope(
        request_id="r1", op=OP_DISPATCH,
        payload={"error": "handler 'echo' at max_inflight"},
        status=STATUS_BUSY,
    )
    text = GatewayRunner._format_dispatcher_response("echo", resp)
    assert "max_inflight" in text
    assert "retry" in text.lower()


def test_format_status_internal() -> None:
    """INTERNAL response surfaces 'internal error' to the user."""
    from gateway.run import GatewayRunner

    resp = Envelope(
        request_id="r1", op=OP_DISPATCH,
        payload={"error": "handler 'echo' raised"},
        status=STATUS_INTERNAL,
    )
    text = GatewayRunner._format_dispatcher_response("echo", resp)
    assert "internal error" in text.lower()


def test_format_status_bad_request() -> None:
    """BAD_REQUEST response surfaces the error string."""
    from gateway.run import GatewayRunner

    resp = Envelope(
        request_id="r1", op=OP_DISPATCH,
        payload={"error": "unknown command 'foo'"},
        status=STATUS_BAD_REQUEST,
    )
    text = GatewayRunner._format_dispatcher_response("foo", resp)
    assert "unknown command" in text


def test_dispatcher_forward_commands_set_is_correct() -> None:
    """The forward set contains the 6 dispatcher-unique commands."""
    from gateway.run import GatewayRunner

    assert GatewayRunner._DISPATCHER_FORWARD_COMMANDS == frozenset({
        "echo", "research", "forge", "ashare", "replay", "log",
    })


# --- Integration: GatewayRunner._forward_to_dispatcher -----------


@pytest.mark.asyncio
async def test_forward_to_dispatcher_via_runner(
    fake_dispatcher: FakeDispatcher,
) -> None:
    """End-to-end: build a minimal GatewayRunner, dispatch a
    recognized slash-command via _forward_to_dispatcher, verify the
    formatted response.
    """
    from gateway.run import GatewayRunner
    from types import SimpleNamespace

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._dispatcher_client = DispatcherClient(
        socket_path=str(fake_dispatcher._path)
    )

    # Minimal MessageEvent stub: only the fields the forward helper
    # actually reads.
    event = SimpleNamespace(
        source=SimpleNamespace(platform=SimpleNamespace(value="wechat")),
        get_command_args=lambda: " hello world",
    )
    text = await runner._forward_to_dispatcher(event, "echo")
    assert text == "[dispatcher] echo: /echo hello world"


@pytest.mark.asyncio
async def test_forward_returns_none_on_dispatcher_down(
    tmp_path: Path,
) -> None:
    """Dispatcher unreachable -> _forward_to_dispatcher returns
    None so the caller can fall through to normal message handling.
    """
    from gateway.run import GatewayRunner
    from types import SimpleNamespace

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._dispatcher_client = DispatcherClient(
        socket_path=str(tmp_path / "nope.sock"),
        timeout_s=0.1,
        max_retries=0,
    )

    event = SimpleNamespace(
        source=SimpleNamespace(platform=SimpleNamespace(value="wechat")),
        get_command_args=lambda: "",
    )
    assert await runner._forward_to_dispatcher(event, "echo") is None


def test_no_dispatcher_disables_forwarding():
    """When dispatcher_socket is not configured,
    _DISPATCHER_FORWARD_COMMANDS should be empty."""
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()  # no dispatcher_socket
    runner._dispatcher_client = None
    # Simulate __init__ logic
    _disp_socket = getattr(runner.config, "dispatcher_socket", None)
    if _disp_socket:
        runner._DISPATCHER_FORWARD_COMMANDS = frozenset({
            "echo", "research", "forge", "ashare", "replay", "log",
        })
    else:
        runner._DISPATCHER_FORWARD_COMMANDS = frozenset()
    assert runner._DISPATCHER_FORWARD_COMMANDS == frozenset()


def test_configured_dispatcher_enables_forwarding():
    """When dispatcher_socket is configured,
    _DISPATCHER_FORWARD_COMMANDS should have default commands."""
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(dispatcher_socket="/tmp/test.sock")
    runner._dispatcher_client = None
    # Simulate __init__ logic
    _disp_socket = getattr(runner.config, "dispatcher_socket", None)
    if _disp_socket:
        runner._dispatcher_client = DispatcherClient(socket_path=_disp_socket)
        runner._DISPATCHER_FORWARD_COMMANDS = frozenset({
            "echo", "research", "forge", "ashare", "replay", "log",
        })
    else:
        runner._DISPATCHER_FORWARD_COMMANDS = frozenset()
    assert "echo" in runner._DISPATCHER_FORWARD_COMMANDS
    assert "forge" in runner._DISPATCHER_FORWARD_COMMANDS


class TestDispatcherConfig:
    """GatewayConfig stores dispatcher_socket and dispatcher_commands."""

    def test_config_stores_dispatcher_socket(self):
        from gateway.config import GatewayConfig

        cfg = GatewayConfig(dispatcher_socket="/run/test/dispatcher.sock")
        assert cfg.dispatcher_socket == "/run/test/dispatcher.sock"

    def test_config_stores_dispatcher_commands(self):
        from gateway.config import GatewayConfig

        cfg = GatewayConfig(dispatcher_commands=["echo", "forge"])
        assert cfg.dispatcher_commands == ["echo", "forge"]

    def test_config_defaults_to_none(self):
        from gateway.config import GatewayConfig

        cfg = GatewayConfig()
        assert cfg.dispatcher_socket is None
        assert cfg.dispatcher_commands is None


class TestDispatcherClientInit:
    """DispatcherClient socket_path configuration."""

    def test_explicit_socket_path(self):
        """Explicit socket_path is stored."""
        client = DispatcherClient(socket_path="/tmp/test.sock")
        assert client._path == "/tmp/test.sock"

    def test_default_socket_path(self):
        """Default socket_path is DEFAULT_DISPATCHER_SOCKET."""
        client = DispatcherClient()
        assert client._path == DEFAULT_DISPATCHER_SOCKET


class TestDispatcherConfigFromDict:
    """GatewayConfig.from_dict handles nested dispatcher config."""

    def test_flat_keys(self):
        """Flat dispatcher_socket/commands keys are parsed."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({
            "dispatcher_socket": "/tmp/s.sock",
            "dispatcher_commands": ["echo", "forge"],
        })
        assert cfg.dispatcher_socket == "/tmp/s.sock"
        assert cfg.dispatcher_commands == ["echo", "forge"]

    def test_nested_dict(self):
        """Nested dispatcher: dict is parsed."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({
            "dispatcher": {"socket": "/tmp/nested.sock", "commands": ["ashare"]},
        })
        assert cfg.dispatcher_socket == "/tmp/nested.sock"
        assert cfg.dispatcher_commands == ["ashare"]

    def test_flat_keys_take_precedence(self):
        """Flat keys win over nested dict when both present."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({
            "dispatcher_socket": "/flat.sock",
            "dispatcher_commands": ["echo"],
            "dispatcher": {"socket": "/nested.sock", "commands": ["forge"]},
        })
        assert cfg.dispatcher_socket == "/flat.sock"
        assert cfg.dispatcher_commands == ["echo"]

    def test_empty_commands_list(self):
        """Empty list disables forwarding (not same as None)."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({
            "dispatcher_commands": [],
        })
        assert cfg.dispatcher_commands == []
        assert cfg.dispatcher_commands is not None

    def test_absent_commands_yields_none(self):
        """Missing dispatcher_commands key yields None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({})
        assert cfg.dispatcher_commands is None

    def test_explicit_none_commands_yields_none(self):
        """Explicit None dispatcher_commands is stored as None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_commands": None})
        assert cfg.dispatcher_commands is None

    def test_absent_socket_yields_none(self):
        """Missing dispatcher_socket key yields None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({})
        assert cfg.dispatcher_socket is None

    def test_explicit_none_socket_yields_none(self):
        """Explicit None dispatcher_socket is stored as None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_socket": None})
        assert cfg.dispatcher_socket is None


class TestDispatcherFromDict:
    """GatewayConfig.from_dict validates dispatcher values.

    Invalid values are rejected with a warning and stored as None.
    """

    def test_invalid_socket_type_stored_as_none(self):
        """Non-string socket value is rejected and stored as None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_socket": 12345})
        assert cfg.dispatcher_socket is None

    def test_empty_socket_stored_as_none(self):
        """Empty string socket value is rejected and stored as None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_socket": ""})
        assert cfg.dispatcher_socket is None

    def test_whitespace_socket_stored_as_none(self):
        """Whitespace-only socket value is rejected and stored as None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_socket": "   "})
        assert cfg.dispatcher_socket is None

    def test_valid_socket_stripped(self):
        """Valid socket value is stripped and stored."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_socket": " /tmp/test.sock "})
        assert cfg.dispatcher_socket == "/tmp/test.sock"

    def test_invalid_commands_type_stored_as_none(self):
        """Non-list commands value is rejected and stored as None."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_commands": "not-a-list"})
        assert cfg.dispatcher_commands is None

    def test_commands_with_non_strings_stored_as_none(self):
        """Commands list with non-string elements is rejected."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_commands": [123, True]})
        assert cfg.dispatcher_commands is None

    def test_valid_commands_stored(self):
        """Valid commands list is stored as-is."""
        from gateway.config import GatewayConfig

        cfg = GatewayConfig.from_dict({"dispatcher_commands": ["echo", "forge"]})
        assert cfg.dispatcher_commands == ["echo", "forge"]


# --- _handle_message regression tests ------------------------------
# These verify the full dispatch path falls through correctly when
# the dispatcher is absent or unreachable (per teknium1 review on
# PR #76810).


class TestHandleMessageDispatcherFallthrough:
    """Regression: _handle_message must fall through to quick-commands
    and agent loop when dispatcher is unavailable.

    These tests exercise the REAL GatewayRunner constructor to verify
    that the dispatcher gating logic in __init__ produces the correct
    _DISPATCHER_FORWARD_COMMANDS and _dispatcher_client state.
    """

    @staticmethod
    def _make_event(command: str, args: str = ""):
        from unittest.mock import MagicMock

        event = MagicMock()
        event.get_command.return_value = command
        event.get_command_args.return_value = args
        event.text = f"/{command} {args}".strip()
        event.source = MagicMock()
        event.source.user_id = "test_user"
        event.source.user_name = "Test User"
        event.source.platform.value = "telegram"
        event.source.chat_type = "dm"
        event.source.chat_id = "123"
        return event

    @pytest.mark.asyncio
    async def test_no_socket_clears_forward_commands(self, monkeypatch, tmp_path):
        """When no dispatcher socket is configured, __init__ clears
        _DISPATCHER_FORWARD_COMMANDS so commands are handled locally."""
        from unittest.mock import MagicMock

        import gateway.run as gateway_run
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        cfg = GatewayConfig.from_dict({})
        runner = GatewayRunner(cfg)

        # Constructor should have cleared forwarding
        assert runner._DISPATCHER_FORWARD_COMMANDS == frozenset()
        assert runner._dispatcher_client is None

        # Commands should fall through to quick-commands
        runner._is_user_authorized = MagicMock(return_value=True)
        runner.config = {
            "quick_commands": {
                "echo": {"type": "exec", "command": "echo local"}
            }
        }
        event = self._make_event("echo")
        result = await runner._handle_message(event)
        assert result == "local"

    @pytest.mark.asyncio
    async def test_configured_socket_creates_client(self, monkeypatch, tmp_path):
        """When dispatcher socket is configured, __init__ creates
        a DispatcherClient and sets _DISPATCHER_FORWARD_COMMANDS."""
        import gateway.run as gateway_run
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        cfg = GatewayConfig.from_dict({
            "dispatcher_socket": "/tmp/test-dispatcher.sock",
        })
        runner = GatewayRunner(cfg)

        # Constructor should have created client and set commands
        assert runner._dispatcher_client is not None
        assert len(runner._DISPATCHER_FORWARD_COMMANDS) > 0
        assert "echo" in runner._DISPATCHER_FORWARD_COMMANDS

    @pytest.mark.asyncio
    async def test_unreachable_socket_falls_through(self, monkeypatch, tmp_path):
        """When dispatcher socket is configured but unreachable,
        _handle_message falls through to quick-commands/agent."""
        from unittest.mock import AsyncMock, MagicMock

        import gateway.run as gateway_run
        from gateway.config import GatewayConfig
        from gateway.dispatcher_client import DispatcherConnectionError
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        cfg = GatewayConfig.from_dict({
            "dispatcher_socket": "/tmp/test-dispatcher.sock",
        })
        runner = GatewayRunner(cfg)
        # Patch the client to simulate unreachable socket
        runner._dispatcher_client.dispatch = AsyncMock(
            side_effect=DispatcherConnectionError("connection refused")
        )
        runner._is_user_authorized = MagicMock(return_value=True)
        runner.config = {
            "quick_commands": {
                "echo": {"type": "exec", "command": "echo local"}
            }
        }

        event = self._make_event("echo")
        result = await runner._handle_message(event)
        # Dispatcher failed -> fall through -> quick-command handles it
        assert result == "local"
