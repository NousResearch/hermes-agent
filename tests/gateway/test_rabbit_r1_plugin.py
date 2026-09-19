"""Tests for the Rabbit R1 platform adapter plugin.

The Rabbit R1 adapter is a bundled platform plugin
(``plugins/platforms/rabbit_r1/``) that runs a WebSocket server speaking the
clawdbot-gateway protocol. These tests validate its logic without a real R1
device or a live gateway:

1. register() metadata + hook wiring (zero core-file integration points)
2. check_requirements / validate_config / is_connected
3. _env_enablement seeding from RABBIT_R1_* env vars
4. _standalone_send returns a descriptive error (server-only platform)
5. adapter init (token resolution, port/tunnel/keepalive config)
6. format_message markdown stripping for the small screen
7. QR payload construction (LAN vs public URL)
8. auth: timing-safe token comparison + per-IP rate limiting
9. device-ID redaction in log strings
10. send / send_typing behavior over a mocked WebSocket
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/rabbit_r1/adapter.py under a unique module name so it
# cannot collide with sibling platform-plugin tests in the same xdist worker.
_r1 = load_plugin_adapter("rabbit_r1")

RabbitR1Adapter = _r1.RabbitR1Adapter
register = _r1.register
check_requirements = _r1.check_requirements
validate_config = _r1.validate_config
is_connected = _r1.is_connected
_env_enablement = _r1._env_enablement
_standalone_send = _r1._standalone_send
_redact = _r1._redact


# All RABBIT_R1_* env vars this plugin reads — cleared before each test so
# the environment can't leak configuration into assertions.
_R1_ENV_VARS = (
    "RABBIT_R1_TOKEN",
    "RABBIT_R1_PORT",
    "RABBIT_R1_TUNNEL",
    "RABBIT_R1_PUBLIC_URL",
    "RABBIT_R1_KEEPALIVE_INTERVAL",
    "RABBIT_R1_ALLOWED_USERS",
    "RABBIT_R1_ALLOW_ALL_USERS",
    "RABBIT_R1_HOME_CHANNEL",
)


@pytest.fixture(autouse=True)
def _clear_r1_env(monkeypatch):
    for var in _R1_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def _make_config(**extra):
    from gateway.config import PlatformConfig
    return PlatformConfig(enabled=True, extra=extra or {})


# ---------------------------------------------------------------------------
# 1. register() metadata + hook wiring
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self):
        self.kwargs = None

    def register_platform(self, **kw):
        self.kwargs = kw


class TestRegister:
    def test_register_calls_register_platform(self):
        ctx = _FakeCtx()
        register(ctx)
        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "rabbit_r1"
        assert ctx.kwargs["label"] == "Rabbit R1"

    def test_register_requires_no_credential_env(self):
        ctx = _FakeCtx()
        register(ctx)
        # The R1 auto-generates its token; nothing is a hard requirement.
        assert ctx.kwargs["required_env"] == []

    def test_register_wires_allowlist_envs(self):
        ctx = _FakeCtx()
        register(ctx)
        assert ctx.kwargs["allowed_users_env"] == "RABBIT_R1_ALLOWED_USERS"
        assert ctx.kwargs["allow_all_env"] == "RABBIT_R1_ALLOW_ALL_USERS"

    def test_register_wires_cron_home_channel(self):
        ctx = _FakeCtx()
        register(ctx)
        assert ctx.kwargs["cron_deliver_env_var"] == "RABBIT_R1_HOME_CHANNEL"

    def test_register_provides_hooks(self):
        ctx = _FakeCtx()
        register(ctx)
        assert callable(ctx.kwargs["standalone_sender_fn"])
        assert callable(ctx.kwargs["env_enablement_fn"])
        assert callable(ctx.kwargs["setup_fn"])
        assert callable(ctx.kwargs["check_fn"])
        assert callable(ctx.kwargs["validate_config"])
        assert callable(ctx.kwargs["is_connected"])

    def test_register_provides_platform_hint(self):
        ctx = _FakeCtx()
        register(ctx)
        hint = ctx.kwargs["platform_hint"]
        assert "Rabbit R1" in hint
        # The hint should steer the model away from markdown on the small screen.
        assert "markdown" in hint.lower()

    def test_register_sets_message_length(self):
        ctx = _FakeCtx()
        register(ctx)
        assert ctx.kwargs["max_message_length"] == RabbitR1Adapter.MAX_MESSAGE_LENGTH

    def test_register_factory_yields_r1_adapter(self):
        ctx = _FakeCtx()
        register(ctx)
        ad = ctx.kwargs["adapter_factory"](_make_config())
        assert isinstance(ad, RabbitR1Adapter)


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_check_requirements_returns_bool(self):
        assert isinstance(check_requirements(), bool)

    def test_validate_config_returns_bool(self):
        assert isinstance(validate_config(_make_config()), bool)

    def test_is_connected_returns_bool(self):
        assert isinstance(is_connected(_make_config()), bool)

    def test_check_requirements_false_without_websockets(self, monkeypatch):
        monkeypatch.setattr(_r1, "WEBSOCKETS_AVAILABLE", False)
        assert check_requirements() is False

    def test_validate_config_gated_on_websockets(self, monkeypatch):
        monkeypatch.setattr(_r1, "WEBSOCKETS_AVAILABLE", False)
        assert validate_config(_make_config()) is False


# ---------------------------------------------------------------------------
# 3. _env_enablement
# ---------------------------------------------------------------------------

class TestEnvEnablement:
    def test_returns_none_without_any_env(self):
        assert _env_enablement() is None

    def test_seeds_token(self, monkeypatch):
        monkeypatch.setenv("RABBIT_R1_TOKEN", "abc123")
        assert _env_enablement() == {"token": "abc123"}

    def test_seeds_port_and_tunnel(self, monkeypatch):
        monkeypatch.setenv("RABBIT_R1_PORT", "9000")
        monkeypatch.setenv("RABBIT_R1_TUNNEL", "cloudflare")
        result = _env_enablement()
        assert result["port"] == 9000
        assert result["tunnel"] == "cloudflare"

    def test_ignores_invalid_port(self, monkeypatch):
        monkeypatch.setenv("RABBIT_R1_TUNNEL", "none")
        monkeypatch.setenv("RABBIT_R1_PORT", "not-a-number")
        result = _env_enablement()
        assert "port" not in result
        assert result["tunnel"] == "none"

    def test_seeds_home_channel(self, monkeypatch):
        monkeypatch.setenv("RABBIT_R1_HOME_CHANNEL", "device-42")
        assert _env_enablement()["home_channel"] == "device-42"


# ---------------------------------------------------------------------------
# 4. _standalone_send — server-only platform, returns descriptive error
# ---------------------------------------------------------------------------

class TestStandaloneSend:
    def test_returns_error(self):
        result = asyncio.run(_standalone_send(_make_config(), "device-1", "hi"))
        assert "error" in result
        assert "gateway" in result["error"].lower()

    def test_accepts_optional_kwargs(self):
        # Signature parity with other plugins — extra kwargs must not raise.
        result = asyncio.run(
            _standalone_send(
                _make_config(), "device-1", "hi",
                thread_id="t", media_files=["/tmp/x"], force_document=True,
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# 5. Adapter init
# ---------------------------------------------------------------------------

class TestAdapterInit:
    def test_token_autogenerated_when_absent(self):
        ad = RabbitR1Adapter(_make_config())
        # 32 bytes hex == 64 chars.
        assert len(ad._token) == 64

    def test_token_from_env_wins(self, monkeypatch):
        monkeypatch.setenv("RABBIT_R1_TOKEN", "envtoken")
        ad = RabbitR1Adapter(_make_config(token="extratoken"))
        assert ad._token == "envtoken"

    def test_token_from_extra_when_no_env(self):
        ad = RabbitR1Adapter(_make_config(token="extratoken"))
        assert ad._token == "extratoken"

    def test_defaults(self):
        ad = RabbitR1Adapter(_make_config())
        assert ad._port == _r1.DEFAULT_PORT
        assert ad._tunnel_mode == _r1.DEFAULT_TUNNEL
        assert ad._keepalive_interval == _r1.DEFAULT_KEEPALIVE_INTERVAL

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("RABBIT_R1_PORT", "12345")
        monkeypatch.setenv("RABBIT_R1_TUNNEL", "NONE")
        monkeypatch.setenv("RABBIT_R1_KEEPALIVE_INTERVAL", "60")
        ad = RabbitR1Adapter(_make_config(port=1, tunnel="tailscale", keepalive_interval=999))
        assert ad._port == 12345
        assert ad._tunnel_mode == "none"  # lowercased
        assert ad._keepalive_interval == 60

    def test_config_extra_when_no_env(self):
        ad = RabbitR1Adapter(_make_config(port=7777, tunnel="cloudflare", keepalive_interval=120))
        assert ad._port == 7777
        assert ad._tunnel_mode == "cloudflare"
        assert ad._keepalive_interval == 120


# ---------------------------------------------------------------------------
# 6. format_message — markdown stripping
# ---------------------------------------------------------------------------

class TestFormatMessage:
    def setup_method(self):
        self.ad = RabbitR1Adapter(_make_config())

    def test_strips_bold_italic(self):
        assert self.ad.format_message("**bold** and _italic_") == "bold and italic"

    def test_links_become_text_and_url(self):
        assert self.ad.format_message("[docs](https://x.com)") == "docs (https://x.com)"

    def test_strips_headers(self):
        assert self.ad.format_message("# Title\nbody") == "Title\nbody"

    def test_strips_code_fences(self):
        out = self.ad.format_message("```python\ncode\n```")
        assert "```" not in out
        assert "code" in out

    def test_strips_inline_code(self):
        assert self.ad.format_message("run `ls` now") == "run ls now"


# ---------------------------------------------------------------------------
# 7. QR payload construction
# ---------------------------------------------------------------------------

class TestQrPayload:
    def test_lan_payload(self, monkeypatch):
        ad = RabbitR1Adapter(_make_config())
        ad._public_url = None
        monkeypatch.setattr(_r1, "_get_lan_ip", lambda: "192.168.1.5")
        payload = json.loads(ad._build_qr_payload())
        assert payload["type"] == "clawdbot-gateway"
        assert payload["protocol"] == "ws"
        assert payload["ips"] == ["192.168.1.5"]
        assert payload["port"] == ad._port
        assert payload["token"] == ad._token

    def test_public_url_payload(self):
        ad = RabbitR1Adapter(_make_config())
        ad._public_url = "wss://me.ts.net"
        payload = json.loads(ad._build_qr_payload())
        assert payload["protocol"] == "wss"
        assert payload["ips"] == ["me.ts.net"]
        assert payload["port"] == 443


# ---------------------------------------------------------------------------
# 8. Auth: timing-safe comparison + per-IP rate limiting
# ---------------------------------------------------------------------------

class TestAuth:
    def _make_ws(self):
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        ws.remote_address = ("10.0.0.9", 5555)
        return ws

    def test_valid_token_pairs_device(self):
        ad = RabbitR1Adapter(_make_config(token="secret-token"))
        ws = self._make_ws()
        msg = {"id": "1", "method": "connect",
               "params": {"auth": {"token": "secret-token"},
                          "device": {"id": "dev-1"}}}
        device_id = asyncio.run(ad._handle_connect(ws, msg, "10.0.0.9:5555"))
        assert device_id == "dev-1"
        assert ad._clients["dev-1"] is ws
        ad._stop_keepalive("dev-1")

    def test_bad_token_rejected(self):
        ad = RabbitR1Adapter(_make_config(token="secret-token"))
        ws = self._make_ws()
        msg = {"id": "1", "method": "connect",
               "params": {"auth": {"token": "wrong"}, "device": {"id": "dev-1"}}}
        device_id = asyncio.run(ad._handle_connect(ws, msg, "10.0.0.9:5555"))
        assert device_id is None
        assert ad._clients == {}
        ws.close.assert_awaited()

    def test_rate_limit_after_repeated_failures(self):
        ad = RabbitR1Adapter(_make_config(token="secret-token"))
        remote = "10.0.0.9:5555"

        async def _run():
            # 5 bad attempts trip the limiter; the 6th is rejected as 429.
            for _ in range(ad._max_auth_failures):
                ws = self._make_ws()
                msg = {"id": "x", "method": "connect",
                       "params": {"auth": {"token": "bad"}, "device": {"id": "d"}}}
                await ad._handle_connect(ws, msg, remote)
            ws = self._make_ws()
            msg = {"id": "x", "method": "connect",
                   "params": {"auth": {"token": "secret-token"}, "device": {"id": "d"}}}
            result = await ad._handle_connect(ws, msg, remote)
            # Even a *correct* token is refused once the IP is rate-limited.
            sent = json.loads(ws.send.call_args[0][0])
            return result, sent

        result, sent = asyncio.run(_run())
        assert result is None
        assert sent["error"]["code"] == 429


# ---------------------------------------------------------------------------
# 9. Device-ID redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_masks_64_char_hex(self):
        device_id = "a" * 64
        assert _redact(f"paired {device_id} ok") == "paired [R1_DEVICE_ID] ok"

    def test_leaves_short_ids_alone(self):
        assert _redact("device r1-10.0.0.9:5555") == "device r1-10.0.0.9:5555"


# ---------------------------------------------------------------------------
# 10. send / send_typing over a mocked WebSocket
# ---------------------------------------------------------------------------

class TestSend:
    def _make_ws(self):
        ws = MagicMock()
        ws.send = AsyncMock()
        return ws

    def test_send_to_unknown_device_fails(self):
        ad = RabbitR1Adapter(_make_config())
        result = asyncio.run(ad.send("nope", "hi"))
        assert result.success is False

    def test_send_delivers_final_chat_event(self):
        ad = RabbitR1Adapter(_make_config())
        ws = self._make_ws()
        ad._clients["dev-1"] = ws
        result = asyncio.run(ad.send("dev-1", "hello there"))
        assert result.success is True
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["event"] == "chat"
        assert sent["payload"]["state"] == "final"
        assert sent["payload"]["message"]["content"][0]["text"] == "hello there"

    def test_send_typing_emits_thinking_state(self):
        ad = RabbitR1Adapter(_make_config())
        ws = self._make_ws()
        ad._clients["dev-1"] = ws
        asyncio.run(ad.send_typing("dev-1"))
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["payload"]["state"] == "thinking"

    def test_send_typing_noop_for_unknown_device(self):
        ad = RabbitR1Adapter(_make_config())
        # Should not raise when the device isn't connected.
        asyncio.run(ad.send_typing("nope"))

    def test_get_chat_info(self):
        ad = RabbitR1Adapter(_make_config())
        ad._clients["dev-1"] = self._make_ws()
        info = asyncio.run(ad.get_chat_info("dev-1"))
        assert info["type"] == "dm"
        assert info["connected"] is True
        assert asyncio.run(ad.get_chat_info("other"))["connected"] is False
