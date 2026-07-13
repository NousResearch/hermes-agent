"""Tests for the IRC platform adapter plugin."""

import asyncio
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/irc/adapter.py under a unique module name
# (plugin_adapter_irc) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same xdist worker.
_irc_mod = load_plugin_adapter("irc")

_parse_irc_message = _irc_mod._parse_irc_message
_extract_nick = _irc_mod._extract_nick
IRCAdapter = _irc_mod.IRCAdapter
check_requirements = _irc_mod.check_requirements
validate_config = _irc_mod.validate_config
register = _irc_mod.register
_standalone_send = _irc_mod._standalone_send


class TestIRCProtocolHelpers:

    def test_parse_simple_command(self):
        msg = _parse_irc_message("PING :server.example.com")
        assert msg["command"] == "PING"
        assert msg["params"] == ["server.example.com"]
        assert msg["prefix"] == ""

    def test_parse_prefixed_message(self):
        msg = _parse_irc_message(":nick!user@host PRIVMSG #channel :Hello world")
        assert msg["prefix"] == "nick!user@host"
        assert msg["command"] == "PRIVMSG"
        assert msg["params"] == ["#channel", "Hello world"]

    def test_parse_numeric_reply(self):
        msg = _parse_irc_message(":server 001 hermes-bot :Welcome to IRC")
        assert msg["prefix"] == "server"
        assert msg["command"] == "001"
        assert msg["params"] == ["hermes-bot", "Welcome to IRC"]

    def test_parse_nick_collision(self):
        msg = _parse_irc_message(":server 433 * hermes-bot :Nickname is already in use")
        assert msg["command"] == "433"

    def test_extract_nick_full_prefix(self):
        assert _extract_nick("nick!user@host") == "nick"

    def test_extract_nick_bare(self):
        assert _extract_nick("server.example.com") == "server.example.com"


# ── IRC Adapter ──────────────────────────────────────────────────────────


class TestIRCAdapterInit:

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_PORT", "6667")
        monkeypatch.setenv("IRC_NICKNAME", "testbot")
        monkeypatch.setenv("IRC_CHANNEL", "#test")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True)
        adapter = IRCAdapter(cfg)

        assert adapter.server == "irc.test.net"
        assert adapter.port == 6667
        assert adapter.nickname == "testbot"
        assert adapter.channel == "#test"
        assert adapter.use_tls is False

    def test_init_from_config_extra(self, monkeypatch):
        # Clear any env vars
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)

        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "irc.libera.chat",
                "port": 6697,
                "nickname": "hermes",
                "channel": "#hermes-dev",
                "use_tls": True,
            },
        )
        adapter = IRCAdapter(cfg)

        assert adapter.server == "irc.libera.chat"
        assert adapter.port == 6697
        assert adapter.nickname == "hermes"
        assert adapter.channel == "#hermes-dev"
        assert adapter.use_tls is True

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "env-server.net")

        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"server": "config-server.net", "channel": "#ch"},
        )
        adapter = IRCAdapter(cfg)
        assert adapter.server == "env-server.net"


class TestIRCAdapterSend:

    @pytest.fixture
    def adapter(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "testbot",
                "channel": "#test",
                "use_tls": False,
            },
        )
        return IRCAdapter(cfg)

    @pytest.mark.asyncio
    async def test_send_not_connected(self, adapter):
        result = await adapter.send("#test", "hello")
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_success(self, adapter):
        writer = MagicMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        adapter._writer = writer

        result = await adapter.send("#test", "hello world")
        assert result.success is True
        assert result.message_id is not None
        # Verify PRIVMSG was sent
        writer.write.assert_called()
        sent_data = writer.write.call_args[0][0]
        assert b"PRIVMSG #test :hello world" in sent_data

    @pytest.mark.asyncio
    async def test_send_splits_long_messages(self, adapter):
        writer = MagicMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        adapter._writer = writer

        long_msg = "x" * 1000
        result = await adapter.send("#test", long_msg)
        assert result.success is True
        # Should have been split into multiple PRIVMSG calls
        assert writer.write.call_count > 1


class TestIRCAdapterMessageParsing:

    @pytest.fixture
    def adapter(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "hermes",
                "channel": "#test",
                "use_tls": False,
            },
        )
        a = IRCAdapter(cfg)
        a._current_nick = "hermes"
        a._registered = True
        return a

    @pytest.mark.asyncio
    async def test_handle_ping(self, adapter):
        writer = MagicMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        adapter._writer = writer

        await adapter._handle_line("PING :test-server")
        sent = writer.write.call_args[0][0]
        assert b"PONG :test-server" in sent

    @pytest.mark.asyncio
    async def test_handle_welcome(self, adapter):
        adapter._registered = False
        adapter._registration_event = asyncio.Event()

        await adapter._handle_line(":server 001 hermes :Welcome to IRC")
        assert adapter._registered is True
        assert adapter._registration_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_nick_collision(self, adapter):
        writer = MagicMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        adapter._writer = writer

        await adapter._handle_line(":server 433 * hermes :Nickname in use")
        assert adapter._current_nick == "hermes_"
        sent = writer.write.call_args[0][0]
        assert b"NICK hermes_" in sent

    @pytest.mark.asyncio
    async def test_handle_addressed_channel_message(self, adapter):
        """Messages addressed to the bot (nick: msg) should be dispatched."""
        handler = AsyncMock(return_value="response")
        adapter._message_handler = handler

        # Mock handle_message to capture the event
        dispatched = []
        original_dispatch = adapter._dispatch_message

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch

        await adapter._handle_line(":user!u@host PRIVMSG #test :hermes: hello there")
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "hello there"
        assert dispatched[0]["chat_id"] == "#test"

    @pytest.mark.asyncio
    async def test_ignores_unaddressed_channel_message(self, adapter):
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":user!u@host PRIVMSG #test :just talking")
        assert len(dispatched) == 0

    @pytest.mark.asyncio
    async def test_handle_dm(self, adapter):
        """DMs (target == bot nick) should always be dispatched."""
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":user!u@host PRIVMSG hermes :private message")
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "private message"
        assert dispatched[0]["chat_type"] == "dm"
        assert dispatched[0]["chat_id"] == "user"

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self, adapter):
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":hermes!bot@host PRIVMSG #test :my own msg")
        assert len(dispatched) == 0

    @pytest.mark.asyncio
    async def test_ctcp_action_converted(self, adapter):
        """CTCP ACTION (/me) should be converted to text."""
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":user!u@host PRIVMSG hermes :\x01ACTION waves\x01")
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "* user waves"

    @pytest.mark.asyncio
    async def test_allowed_users_case_insensitive(self, monkeypatch):
        """Allowlist should match nicks case-insensitively."""
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "hermes",
                "channel": "#test",
                "use_tls": False,
                "allowed_users": ["Admin", "BOB"],
            },
        )
        adapter = IRCAdapter(cfg)
        adapter._current_nick = "hermes"
        adapter._registered = True
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        # "admin" matches "Admin" in allowlist
        await adapter._handle_line(":admin!u@host PRIVMSG #test :hermes: hello")
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_unauthorized_user_blocked(self, monkeypatch):
        """Nicks not in allowlist should be ignored."""
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "hermes",
                "channel": "#test",
                "use_tls": False,
                "allowed_users": ["Admin", "BOB"],
            },
        )
        adapter = IRCAdapter(cfg)
        adapter._current_nick = "hermes"
        adapter._registered = True
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":eve!u@host PRIVMSG #test :hermes: hello")
        assert len(dispatched) == 0

    @pytest.mark.asyncio
    async def test_nick_collision_retry(self, adapter):
        """Multiple 433 responses should keep incrementing the suffix."""
        writer = MagicMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        adapter._writer = writer

        await adapter._handle_line(":server 433 * hermes :Nickname in use")
        assert adapter._current_nick == "hermes_"
        await adapter._handle_line(":server 433 * hermes_ :Nickname in use")
        assert adapter._current_nick == "hermes_1"
        await adapter._handle_line(":server 433 * hermes_1 :Nickname in use")
        assert adapter._current_nick == "hermes_2"


class TestIRCAdapterSplitting:

    def test_split_respects_byte_limit(self):
        """Multi-byte characters should not exceed IRC byte limit."""
        # 100 japanese chars = 300 bytes in utf-8
        text = "あ" * 100
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"server": "x", "channel": "#x"})
        adapter = IRCAdapter(cfg)
        adapter._current_nick = "bot"
        lines = adapter._split_message(text, "#test")
        for line in lines:
            overhead = len(f"PRIVMSG #test :{line}\r\n".encode("utf-8"))
            assert overhead <= 512, f"line over 512 bytes: {overhead}"

    def test_split_prefers_word_boundary(self):
        text = "hello world foo bar baz qux"
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"server": "x", "channel": "#x"})
        adapter = IRCAdapter(cfg)
        adapter._current_nick = "bot"
        lines = adapter._split_message(text, "#test")
        # Should not split in the middle of "world"
        assert any("hello" in ln for ln in lines)
        assert any("world" in ln for ln in lines)


class TestIRCProtocolHelpersExtra:

    def test_parse_malformed_no_space(self):
        """A line starting with : but no space should not crash."""
        msg = _parse_irc_message(":justaprefix")
        assert msg["prefix"] == "justaprefix"
        assert msg["command"] == ""
        assert msg["params"] == []

    def test_parse_empty(self):
        msg = _parse_irc_message("")
        assert msg["prefix"] == ""
        assert msg["command"] == ""
        assert msg["params"] == []


class TestIRCAdapterMarkdown:

    def test_strip_bold(self):
        assert IRCAdapter._strip_markdown("**bold**") == "bold"

    def test_strip_italic(self):
        assert IRCAdapter._strip_markdown("*italic*") == "italic"

    def test_strip_code(self):
        assert IRCAdapter._strip_markdown("`code`") == "code"

    def test_strip_link(self):
        result = IRCAdapter._strip_markdown("[click here](https://example.com)")
        assert result == "click here (https://example.com)"

    def test_strip_image(self):
        result = IRCAdapter._strip_markdown("![alt](https://example.com/img.png)")
        assert result == "https://example.com/img.png"


# ── Requirements / validation ────────────────────────────────────────────


class TestIRCRequirements:

    def test_check_requirements_with_env(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#test")
        assert check_requirements() is True

    def test_check_requirements_missing_server(self, monkeypatch):
        monkeypatch.delenv("IRC_SERVER", raising=False)
        monkeypatch.setenv("IRC_CHANNEL", "#test")
        assert check_requirements() is False

    def test_check_requirements_missing_channel(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.delenv("IRC_CHANNEL", raising=False)
        assert check_requirements() is False

    def test_validate_config_from_extra(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(extra={"server": "irc.test.net", "channel": "#test"})
        assert validate_config(cfg) is True

    def test_validate_config_missing(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(extra={})
        assert validate_config(cfg) is False


# ── Plugin registration ──────────────────────────────────────────────────


class TestIRCPluginRegistration:
    """Test the register() entry point."""

    def test_register_adds_to_registry(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#test")

        from gateway.platform_registry import platform_registry

        # Clean up if already registered
        platform_registry.unregister("irc")

        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        call_kwargs = ctx.register_platform.call_args
        assert call_kwargs[1]["name"] == "irc" or call_kwargs[0][0] == "irc" if call_kwargs[0] else call_kwargs[1]["name"] == "irc"


# ── _standalone_send (out-of-process cron delivery) ──────────────────────


class _FakeIRCConnection:
    """A scripted reader/writer pair used to simulate an IRC server.

    Construct with the lines the server should respond with (already
    framed by ``\\r\\n``).  Captures every line written by the client so
    tests can assert NICK/USER/PRIVMSG/QUIT order.
    """

    def __init__(self, scripted_lines):
        self.writes: list[bytes] = []
        self._closed = False
        self._scripted = list(scripted_lines)
        self._buffer = b""

    # writer side ────────────────────────────────────────────────────
    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self._closed

    # reader side ────────────────────────────────────────────────────
    async def readuntil(self, separator: bytes = b"\r\n") -> bytes:
        if not self._scripted:
            raise asyncio.IncompleteReadError(b"", None)
        line = self._scripted.pop(0)
        if not line.endswith(b"\r\n"):
            line = line + b"\r\n"
        return line

    async def read(self, n: int = -1) -> bytes:
        return b""


class TestIRCStandaloneSend:

    @pytest.mark.asyncio
    async def test_standalone_send_completes_handshake_and_sends_privmsg(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        # Server greets us with 001 RPL_WELCOME, then nothing for QUIT drain.
        conn = _FakeIRCConnection([b":server 001 hermesbot-cron :Welcome"])

        async def _fake_open(host, port, **kwargs):
            return conn, conn  # reader and writer share the same fake

        monkeypatch.setattr(_irc_mod.asyncio, "open_connection", _fake_open)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron",
            "hello from cron",
        )

        assert result["success"] is True
        assert "message_id" in result

        sent_lines = b"".join(conn.writes).decode("utf-8").splitlines()
        # NICK uses the cron-suffixed identity to avoid colliding with the
        # long-running gateway adapter that may already hold the nickname.
        assert any(line.startswith("NICK hermesbot-cron") for line in sent_lines)
        assert any(line.startswith("USER hermesbot-cron 0 * :Hermes Agent (cron)")
                   for line in sent_lines)
        assert any(line == "PRIVMSG #cron :hello from cron" for line in sent_lines)
        assert any(line.startswith("QUIT ") for line in sent_lines)

    @pytest.mark.asyncio
    async def test_standalone_send_returns_error_when_unconfigured(self, monkeypatch):
        from gateway.config import PlatformConfig

        for var in ("IRC_SERVER", "IRC_CHANNEL"):
            monkeypatch.delenv(var, raising=False)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "",
            "hi",
        )

        assert "error" in result
        assert "IRC_SERVER" in result["error"] or "IRC_CHANNEL" in result["error"]

    @pytest.mark.asyncio
    async def test_standalone_send_returns_error_on_registration_timeout(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        # No 001 response: the readuntil call returns IncompleteReadError so
        # the registration loop times out via the asyncio wait_for inside.
        conn = _FakeIRCConnection([])

        async def _fake_open(host, port, **kwargs):
            return conn, conn

        monkeypatch.setattr(_irc_mod.asyncio, "open_connection", _fake_open)

        # Patch wait_for to raise TimeoutError immediately so the test is fast
        async def _fast_timeout(coro, timeout):
            try:
                return await coro
            except asyncio.IncompleteReadError:
                raise asyncio.TimeoutError()

        monkeypatch.setattr(_irc_mod.asyncio, "wait_for", _fast_timeout)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron",
            "hi",
        )

        assert "error" in result
        assert "registration" in result["error"].lower() or "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_standalone_send_rejects_crlf_in_chat_id(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        # Attempt to inject a second IRC command via CRLF in chat_id
        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron\r\nKICK #cron hermesbot",
            "hi",
        )

        assert "error" in result
        assert "illegal IRC characters" in result["error"]

    @pytest.mark.asyncio
    async def test_standalone_send_strips_crlf_from_message_body(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        conn = _FakeIRCConnection([b":server 001 hermesbot-cron :Welcome"])

        async def _fake_open(host, port, **kwargs):
            return conn, conn

        monkeypatch.setattr(_irc_mod.asyncio, "open_connection", _fake_open)

        # A bare \r in message content tries to inject a NICK command.
        # Our control-char stripper must blank \r so the line stays one PRIVMSG.
        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron",
            "hello\rNICK eviltwin",
        )

        sent_lines = b"".join(conn.writes).decode("utf-8").splitlines()
        # No injected NICK command after the legitimate registration NICK
        nick_lines = [line for line in sent_lines if line.startswith("NICK ")]
        # Only the original registration NICK should be present (no injected one)
        assert all(line.startswith("NICK hermesbot-cron") for line in nick_lines)
        # The PRIVMSG should contain "hello NICK eviltwin" as one line (with \r blanked)
        assert any("PRIVMSG #cron :hello NICK eviltwin" in line for line in sent_lines)

    @pytest.mark.asyncio
    async def test_standalone_send_joins_channel_before_privmsg(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        # Register, then accept JOIN with 366 RPL_ENDOFNAMES, then PRIVMSG.
        conn = _FakeIRCConnection([
            b":server 001 hermesbot-cron :Welcome",
            b":server 366 hermesbot-cron #cron :End of /NAMES list.",
        ])

        async def _fake_open(host, port, **kwargs):
            return conn, conn

        monkeypatch.setattr(_irc_mod.asyncio, "open_connection", _fake_open)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron",
            "hello",
        )

        assert result["success"] is True
        sent_lines = b"".join(conn.writes).decode("utf-8").splitlines()
        join_idx = next((i for i, line in enumerate(sent_lines) if line.startswith("JOIN #cron")), None)
        privmsg_idx = next((i for i, line in enumerate(sent_lines) if line.startswith("PRIVMSG #cron")), None)
        assert join_idx is not None, "JOIN must be sent for channel targets"
        assert privmsg_idx is not None
        assert join_idx < privmsg_idx, "JOIN must precede PRIVMSG"
