"""Tests for the IRC platform adapter plugin."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/irc/adapter.py under a unique module name
# (plugin_adapter_irc) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same process.
_irc_mod = load_plugin_adapter("irc")

_parse_irc_message = _irc_mod._parse_irc_message
_extract_nick = _irc_mod._extract_nick
IRCAdapter = _irc_mod.IRCAdapter
check_requirements = _irc_mod.check_requirements
validate_config = _irc_mod.validate_config
register = _irc_mod.register
_standalone_send = _irc_mod._standalone_send
is_connected = _irc_mod.is_connected
_env_enablement = _irc_mod._env_enablement


class TestIRCProtocolHelpers:

    def test_parse_simple_command(self):
        msg = _parse_irc_message("PING :server.example.com")
        assert msg["command"] == "PING"
        assert msg["params"] == ["server.example.com"]
        assert msg["prefix"] == ""


    def test_extract_nick_full_prefix(self):
        assert _extract_nick("nick!user@host") == "nick"


# ── IRC Adapter ──────────────────────────────────────────────────────────




class TestIRCAdapterLockConflict:

    @pytest.mark.asyncio
    async def test_connect_fails_when_identity_lock_held(self, monkeypatch):
        """``acquire_scoped_lock`` returns ``(acquired, existing)``; a live foreign holder must stop
        connect() before any socket is opened (the tuple is truthy, so a bare ``if not`` never fired)."""
        import gateway.status as gateway_status
        from gateway.config import PlatformConfig

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, {"pid": 4242, "profile": "other"}))
        opened = []

        async def _no_connect(*a, **k):
            opened.append(a)
            raise AssertionError("socket must not be opened on a lock conflict")
        monkeypatch.setattr(asyncio, "open_connection", _no_connect)
        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={
            "server": "irc.example", "nickname": "hermes", "channel": "#x"}))
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "irc_lock"
        assert "other" in adapter._fatal_error_message
        assert opened == []


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


class TestIRCProtocolHelpersExtra:

    def test_parse_malformed_no_space(self):
        """A line starting with : but no space should not crash."""
        msg = _parse_irc_message(":justaprefix")
        assert msg["prefix"] == "justaprefix"
        assert msg["command"] == ""
        assert msg["params"] == []


class TestIRCAdapterMarkdown:


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


    def test_validate_config_from_extra(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(extra={"server": "irc.test.net", "channel": "#test"})
        assert validate_config(cfg) is True


# ── Plugin registration ──────────────────────────────────────────────────




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


# ---------------------------------------------------------------------------
# Multiplex secondary-profile scope
# ---------------------------------------------------------------------------
#
# __init__'s server/port/nickname/channel/use_tls, check_requirements/
# validate_config/is_connected's server/channel, and _env_enablement's
# server/channel/port/nickname/use_tls/home_channel, all previously read raw
# os.getenv unconditionally (only IRC_SERVER_PASSWORD/IRC_NICKSERV_PASSWORD
# were already scoped). Under multiplex, os.environ holds the DEFAULT
# profile's YAML-to-env bridge output -- a secondary profile with its own
# (different or absent) IRC config would silently connect to the default
# profile's server/channel, or (for _env_enablement) get auto-enabled using
# the default's channel as its cron home_channel -- a real message-
# misdelivery risk, not just cosmetic. Mirrors the LINE/Buzz/SimpleX fix for
# #98738.

@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("IRC_SERVER", "default.example.net")
    monkeypatch.setenv("IRC_CHANNEL", "#default")
    monkeypatch.setenv("IRC_PORT", "6667")
    monkeypatch.setenv("IRC_NICKNAME", "default-bot")
    monkeypatch.setenv("IRC_USE_TLS", "false")


class TestMultiplexProfileScope:

    def test_secondary_extra_wins_over_default_profile_env(
        self, multiplex_scope, default_profile_env
    ):
        """The secondary profile's own config.yaml extra is authoritative,
        not the default profile's bridged server/channel/port/nick/tls."""
        from gateway.config import PlatformConfig

        multiplex_scope()
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "profile.example.net",
                "channel": "#profile",
                "port": 6697,
                "nickname": "profile-bot",
                "use_tls": True,
            },
        )
        adapter = IRCAdapter(cfg)
        assert adapter.server == "profile.example.net"
        assert adapter.channel == "#profile"
        assert adapter.port == 6697
        assert adapter.nickname == "profile-bot"
        assert adapter.use_tls is True

    def test_secondary_missing_keys_fail_closed(
        self, multiplex_scope, default_profile_env
    ):
        """Keys absent from the profile's own scope must NOT borrow the
        default profile's bridged env values -- that would silently connect
        the secondary profile's bot to the wrong IRC server/channel."""
        from gateway.config import PlatformConfig

        multiplex_scope()
        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter.server == ""
        assert adapter.channel == ""
        assert adapter.port != 6667
        assert adapter.nickname != "default-bot"
        assert adapter.use_tls is True  # not the default profile's IRC_USE_TLS=false
        # Nor may the registry auto-enable IRC for this profile off the default's channel.
        assert _env_enablement() is None
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False


class TestIRCOutboundSafety:
    @pytest.fixture
    def adapter(self, monkeypatch):
        from gateway.config import PlatformConfig
        for key in ('IRC_SERVER', 'IRC_CHANNEL', 'IRC_NICKNAME', 'IRC_SERVER_PASSWORD', 'IRC_NICKSERV_PASSWORD'):
            monkeypatch.delenv(key, raising=False)
        obj = IRCAdapter(PlatformConfig(enabled=True, extra={'server': 'irc.test', 'channel': '#c', 'nickname': 'bot', 'use_tls': False}))
        obj._writer = _FakeIRCConnection([])
        monkeypatch.setattr(_irc_mod.asyncio, 'sleep', AsyncMock())
        return obj

    @staticmethod
    def assert_frames(writes):
        for wire in writes:
            assert wire.endswith(b'\r\n')
            assert not any(ch in wire[:-2] for ch in (b'\r', b'\n', b'\x00'))
            assert len(wire) <= 512

    @pytest.mark.asyncio
    @pytest.mark.parametrize('content', ['hi\rQUIT :bye', 'hi\r\nQUIT :bye', 'hi\nQUIT :bye', 'hi\x00there', 'a\r\nb\rc\nd\x00e'])
    async def test_content_cannot_inject_commands(self, adapter, content):
        assert (await adapter.send('#c', content)).success
        self.assert_frames(adapter._writer.writes)
        assert all(w.startswith(b'PRIVMSG #c :') for w in adapter._writer.writes)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('target', ['', '#c\rQUIT :bye', '#c\nQUIT :bye', '#c\x00', '#c other'])
    async def test_hostile_target_is_rejected_without_write(self, adapter, target):
        assert not (await adapter.send(target, 'hello')).success
        assert adapter._writer.writes == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize('target', ['#channel', '&channel', 'Nick'])
    async def test_normal_targets_and_unicode_separators_unchanged(self, adapter, target):
        content = 'hello\u2028world\u0085again'
        assert (await adapter.send(target, content)).success
        assert adapter._writer.writes == [f'PRIVMSG {target} :{content}\r\n'.encode()]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('content', ['', '   ', '\r', '\x00'])
    async def test_empty_content_still_sends_one_empty_payload(self, adapter, content):
        assert (await adapter.send('#c', content)).success
        assert adapter._writer.writes == [b'PRIVMSG #c :\r\n']

    @pytest.mark.asyncio
    @pytest.mark.parametrize('command', ['PASS pw', 'NICK nick', 'USER nick 0 * :Hermes Agent', 'JOIN #chan', 'PRIVMSG NickServ :IDENTIFY pw', 'PONG :server'])
    async def test_raw_registration_and_control_lines_are_framed_once(self, adapter, command):
        await adapter._send_raw(command)
        await adapter._send_raw(command + '\r\nQUIT :injected\x00')
        assert adapter._writer.writes[0] == (command + '\r\n').encode()
        self.assert_frames(adapter._writer.writes)
        assert adapter._writer.writes[1] == (command + '  QUIT :injected\r\n').encode()

    @pytest.mark.asyncio
    async def test_utf8_chunks_and_clean_content_budgets_unchanged(self, adapter):
        content = '你好 ' * 400
        expected = _irc_mod._split_lines([content], min(adapter.max_message_length, _irc_mod._privmsg_budget('#c')))
        assert adapter._split_message(content, '#c') == expected
        assert (await adapter.send('#c', content + '\x00')).success
        self.assert_frames(adapter._writer.writes)
        assert len(adapter._writer.writes) == len(expected)
        assert b''.join(adapter._writer.writes).decode('utf-8')

    @pytest.mark.asyncio
    async def test_standalone_matches_live_sanitization(self, adapter, monkeypatch):
        from gateway.config import PlatformConfig
        monkeypatch.setenv('IRC_SERVER', 'irc.test')
        monkeypatch.setenv('IRC_CHANNEL', '#c')
        monkeypatch.setenv('IRC_NICKNAME', 'bot')
        monkeypatch.setenv('IRC_USE_TLS', 'false')
        conn = _FakeIRCConnection([b':server 001 bot-cron :Welcome'])
        monkeypatch.setattr(_irc_mod.asyncio, 'open_connection', AsyncMock(return_value=(conn, conn)))
        content = 'hello\rQUIT :bye\r\nnext\nlast\x00\u2028end'
        assert (await adapter.send('#c', content)).success
        result = await _standalone_send(PlatformConfig(enabled=True, extra={}), '#c', content)
        assert result['success']
        assert [w for w in conn.writes if w.startswith(b'PRIVMSG #c :')] == adapter._writer.writes
        self.assert_frames(conn.writes)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('hostile', [False, True])
    async def test_connect_sanitizes_every_registration_field(self, adapter, monkeypatch, hostile):
        suffix = '\r\nQUIT :bad\x00' if hostile else ''
        adapter.nickname = 'bot' + suffix
        adapter.channel = '#c' + suffix
        adapter.server_password = 'serverpw' + suffix
        adapter.nickserv_password = 'nickpw' + suffix
        monkeypatch.setattr(adapter, '_acquire_platform_lock', lambda *args: True)
        monkeypatch.setattr(adapter, '_wire_plugin_handlers', lambda *args: None)
        monkeypatch.setattr(adapter, '_mark_connected', lambda: None)
        async def receive():
            adapter._registration_event.set()
        monkeypatch.setattr(adapter, '_receive_loop', receive)
        conn = _FakeIRCConnection([])
        monkeypatch.setattr(_irc_mod.asyncio, 'open_connection', AsyncMock(return_value=(conn, conn)))
        assert await adapter.connect()
        clean = _irc_mod._strip_irc_control_chars
        assert conn.writes == [
            f'PASS {clean(adapter.server_password)}\r\n'.encode(),
            f'NICK {clean(adapter.nickname)}\r\n'.encode(),
            f'USER {clean(adapter.nickname)} 0 * :Hermes Agent\r\n'.encode(),
            f'PRIVMSG NickServ :IDENTIFY {clean(adapter.nickserv_password)}\r\n'.encode(),
            f'JOIN {clean(adapter.channel)}\r\n'.encode(),
        ]
        self.assert_frames(conn.writes)

    @pytest.mark.asyncio
    async def test_send_reports_write_failure_and_stops(self, adapter):
        adapter._writer.write = MagicMock(side_effect=OSError('connection closed'))
        result = await adapter.send('#c', 'first\nsecond')
        assert not result.success
        assert result.error == 'connection closed'
        adapter._writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_when_disconnected_has_no_writes(self, adapter):
        adapter._writer.close()
        assert not (await adapter.send('#c', 'hello')).success
        assert adapter._writer.writes == []
