"""Tests for the XMPP platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_xmpp`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_xmpp = load_plugin_adapter("xmpp")

XMPPAdapter = _xmpp.XMPPAdapter
check_requirements = _xmpp.check_requirements
validate_config = _xmpp.validate_config
is_connected = _xmpp.is_connected
register = _xmpp.register
_env_enablement = _xmpp._env_enablement
_standalone_send = _xmpp._standalone_send
_strip_resource = _xmpp._strip_resource
_strip_markdown = _xmpp._strip_markdown
_strip_control_chars = _xmpp._strip_control_chars
_split_message = _xmpp._split_message
_parse_bool = _xmpp._parse_bool
_parse_comma_list = _xmpp._parse_comma_list
MUC_PREFIX = _xmpp.MUC_PREFIX


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------

def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform
    p = Platform("xmpp")
    assert p.value == "xmpp"
    assert Platform("xmpp") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------

def test_check_requirements_needs_credentials(monkeypatch):
    monkeypatch.delenv("XMPP_JID", raising=False)
    monkeypatch.delenv("XMPP_PASSWORD", raising=False)
    assert check_requirements() is False

    monkeypatch.setenv("XMPP_JID", "hermes@example.org")
    assert check_requirements() is False  # password still missing


def test_check_requirements_true_when_fully_configured(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "hermes@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "secret")
    try:
        import slixmpp  # noqa: F401
        slixmpp_present = True
    except ImportError:
        slixmpp_present = False
    assert check_requirements() is slixmpp_present


def test_validate_config_uses_env_or_extra(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("XMPP_JID", raising=False)
    monkeypatch.delenv("XMPP_PASSWORD", raising=False)
    cfg = PlatformConfig(enabled=True)
    assert validate_config(cfg) is False

    cfg2 = PlatformConfig(enabled=True, extra={"jid": "hermes@example.org", "password": "secret"})
    assert validate_config(cfg2) is True


def test_is_connected_mirrors_validate(monkeypatch):
    from gateway.config import PlatformConfig
    monkeypatch.delenv("XMPP_JID", raising=False)
    monkeypatch.delenv("XMPP_PASSWORD", raising=False)
    cfg = PlatformConfig(enabled=True, extra={"jid": "x@x", "password": "y"})
    assert is_connected(cfg) is True
    assert is_connected(PlatformConfig(enabled=True)) is False


# ---------------------------------------------------------------------------
# 3. _env_enablement seeds PlatformConfig.extra
# ---------------------------------------------------------------------------

def test_env_enablement_none_when_unset(monkeypatch):
    monkeypatch.delenv("XMPP_JID", raising=False)
    monkeypatch.delenv("XMPP_PASSWORD", raising=False)
    assert _env_enablement() is None


def test_env_enablement_needs_password(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "hermes@example.org")
    monkeypatch.delenv("XMPP_PASSWORD", raising=False)
    assert _env_enablement() is None


def test_env_enablement_seeds_minimal(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "hermes@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "secret")
    for var in ("XMPP_HOST", "XMPP_PORT", "XMPP_FORCE_STARTTLS",
                "XMPP_NICKNAME", "XMPP_ROOMS", "XMPP_HOME_CHANNEL"):
        monkeypatch.delenv(var, raising=False)
    seed = _env_enablement()
    assert seed == {"jid": "hermes@example.org"}


def test_env_enablement_seeds_full(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "hermes@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "secret")
    monkeypatch.setenv("XMPP_HOST", "chat.example.org")
    monkeypatch.setenv("XMPP_PORT", "5223")
    monkeypatch.setenv("XMPP_FORCE_STARTTLS", "false")
    monkeypatch.setenv("XMPP_NICKNAME", "h2")
    monkeypatch.setenv("XMPP_ROOMS", "ops@conf, dev@conf")
    monkeypatch.setenv("XMPP_HOME_CHANNEL", "alice@example.org")
    monkeypatch.setenv("XMPP_HOME_CHANNEL_NAME", "Alice")

    seed = _env_enablement()
    assert seed["jid"] == "hermes@example.org"
    assert seed["host"] == "chat.example.org"
    assert seed["port"] == 5223
    assert seed["force_starttls"] is False
    assert seed["nickname"] == "h2"
    assert seed["rooms"] == ["ops@conf", "dev@conf"]
    assert seed["home_channel"] == {"chat_id": "alice@example.org", "name": "Alice"}


def test_env_enablement_home_channel_defaults_name_to_id(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "hermes@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "secret")
    monkeypatch.setenv("XMPP_HOME_CHANNEL", "alice@example.org")
    monkeypatch.delenv("XMPP_HOME_CHANNEL_NAME", raising=False)

    seed = _env_enablement()
    assert seed["home_channel"] == {
        "chat_id": "alice@example.org",
        "name": "alice@example.org",
    }


# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------

def test_strip_resource():
    assert _strip_resource("user@example.org/laptop") == "user@example.org"
    assert _strip_resource("user@example.org") == "user@example.org"
    assert _strip_resource("") == ""


def test_strip_markdown_removes_basic_formatting():
    out = _strip_markdown("**bold** *italic* `code` ~~strike~~")
    assert "*" not in out
    assert "`" not in out
    assert "~~" not in out
    assert "bold" in out and "italic" in out and "code" in out and "strike" in out


def test_strip_markdown_keeps_link_target():
    out = _strip_markdown("see [docs](https://example.org/x)")
    assert "https://example.org/x" in out
    assert "[" not in out


def test_strip_markdown_strips_code_fences():
    out = _strip_markdown("```python\nprint(1)\n```")
    assert "```" not in out
    assert "print(1)" in out


def test_strip_control_chars_drops_null_and_normalizes_crlf():
    out = _strip_control_chars("a\x00b\r\nc\rd")
    assert "\x00" not in out
    assert "\r" not in out
    assert out == "ab\nc\nd"


def test_split_message_short_returns_single_chunk():
    chunks = _split_message("hello world", max_bytes=100)
    assert chunks == ["hello world"]


def test_split_message_long_paragraph_splits_under_limit():
    para = "a" * 10_000
    chunks = _split_message(para, max_bytes=500)
    assert len(chunks) >= 20
    for chunk in chunks:
        assert len(chunk.encode("utf-8")) <= 500


def test_split_message_multi_paragraph():
    text = "para one\npara two"
    chunks = _split_message(text, max_bytes=100)
    assert chunks == ["para one", "para two"]


def test_parse_bool_handles_strings_and_booleans():
    assert _parse_bool("true", default=False) is True
    assert _parse_bool("FALSE", default=True) is False
    assert _parse_bool(None, default=True) is True
    assert _parse_bool(True, default=False) is True


def test_parse_comma_list_strips_and_drops_empty():
    assert _parse_comma_list("a, b , ,c") == ["a", "b", "c"]
    assert _parse_comma_list("") == []


# ---------------------------------------------------------------------------
# 5. Adapter init
# ---------------------------------------------------------------------------

def test_adapter_init_reads_extra(monkeypatch):
    from gateway.config import PlatformConfig
    for var in ("XMPP_JID", "XMPP_PASSWORD", "XMPP_HOST", "XMPP_PORT",
                "XMPP_FORCE_STARTTLS", "XMPP_NICKNAME", "XMPP_ROOMS"):
        monkeypatch.delenv(var, raising=False)

    cfg = PlatformConfig(enabled=True, extra={
        "jid": "hermes@example.org",
        "password": "s3cret",
        "host": "chat.example.org",
        "port": 5223,
        "force_starttls": False,
        "nickname": "h",
        "rooms": "ops@conf, dev@conf",
    })
    adapter = XMPPAdapter(cfg)
    assert adapter.jid == "hermes@example.org"
    assert adapter.host == "chat.example.org"
    assert adapter.port == 5223
    assert adapter.force_starttls is False
    assert adapter.nickname == "h"
    assert adapter.rooms == ["ops@conf", "dev@conf"]


def test_adapter_init_env_overrides_extra(monkeypatch):
    from gateway.config import PlatformConfig
    monkeypatch.setenv("XMPP_JID", "env@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "envpass")
    monkeypatch.setenv("XMPP_PORT", "5223")
    monkeypatch.setenv("XMPP_FORCE_STARTTLS", "false")

    cfg = PlatformConfig(enabled=True, extra={
        "jid": "extra@example.org",
        "password": "xpass",
        "port": 5222,
        "force_starttls": True,
    })
    adapter = XMPPAdapter(cfg)
    assert adapter.jid == "env@example.org"
    assert adapter.password == "envpass"
    assert adapter.port == 5223
    assert adapter.force_starttls is False


def test_adapter_init_defaults(monkeypatch):
    from gateway.config import PlatformConfig
    for var in ("XMPP_JID", "XMPP_PASSWORD", "XMPP_HOST", "XMPP_PORT",
                "XMPP_FORCE_STARTTLS", "XMPP_NICKNAME", "XMPP_ROOMS"):
        monkeypatch.delenv(var, raising=False)

    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@example.org", "password": "x"})
    adapter = XMPPAdapter(cfg)
    assert adapter.port == 5222
    assert adapter.force_starttls is True
    assert adapter.host is None
    assert adapter.nickname == "bot"  # local part of the JID
    assert adapter.rooms == []


def test_adapter_init_invalid_port_falls_back(monkeypatch):
    from gateway.config import PlatformConfig
    monkeypatch.delenv("XMPP_PORT", raising=False)
    cfg = PlatformConfig(enabled=True, extra={"jid": "x@x", "password": "y", "port": "not-a-port"})
    adapter = XMPPAdapter(cfg)
    assert adapter.port == 5222


def test_adapter_platform_identity():
    from gateway.config import Platform, PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "x@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    assert adapter.platform is Platform("xmpp")


def test_resolve_target_dm():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "x@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    target, mtype = adapter._resolve_target("alice@example.org")
    assert target == "alice@example.org"
    assert mtype == "chat"


def test_resolve_target_muc():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "x@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    target, mtype = adapter._resolve_target(f"{MUC_PREFIX}ops@conf.example.org")
    assert target == "ops@conf.example.org"
    assert mtype == "groupchat"


def test_resolve_target_empty():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "x@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    target, mtype = adapter._resolve_target("")
    assert target == ""
    assert mtype == "chat"


# ---------------------------------------------------------------------------
# 6. Outbound send (mocked client)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_dm_calls_send_message():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = MagicMock()

    result = await adapter.send("alice@example.org", "**hi**")
    assert result.success is True
    adapter._client.send_message.assert_called_once()
    kwargs = adapter._client.send_message.call_args.kwargs
    assert kwargs["mto"] == "alice@example.org"
    assert kwargs["mtype"] == "chat"
    assert "**" not in kwargs["mbody"]


@pytest.mark.asyncio
async def test_send_muc_uses_groupchat():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = MagicMock()

    result = await adapter.send(f"{MUC_PREFIX}ops@conf", "hello")
    assert result.success is True
    kwargs = adapter._client.send_message.call_args.kwargs
    assert kwargs["mto"] == "ops@conf"
    assert kwargs["mtype"] == "groupchat"


@pytest.mark.asyncio
async def test_send_when_disconnected_returns_error():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    result = await adapter.send("alice@example.org", "hi")
    assert result.success is False
    assert "Not connected" in (result.error or "")


@pytest.mark.asyncio
async def test_send_empty_body_after_strip_succeeds_without_call():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = MagicMock()

    result = await adapter.send("alice@example.org", "\x00 \r\n  ")
    assert result.success is True
    adapter._client.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_splits_long_message():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y", "max_message_length": 100})
    adapter = XMPPAdapter(cfg)
    adapter._client = MagicMock()

    body = "lorem ipsum " * 50
    result = await adapter.send("alice@example.org", body)
    assert result.success is True
    assert adapter._client.send_message.call_count >= 2


# ---------------------------------------------------------------------------
# 7. get_chat_info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_chat_info_dm():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    info = await adapter.get_chat_info("alice@example.org")
    assert info["type"] == "dm"
    assert info["chat_id"] == "alice@example.org"


@pytest.mark.asyncio
async def test_get_chat_info_muc():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    info = await adapter.get_chat_info(f"{MUC_PREFIX}ops@conf")
    assert info["type"] == "group"
    assert info["name"] == "ops@conf"


# ---------------------------------------------------------------------------
# 8. Inbound message dispatch
# ---------------------------------------------------------------------------

class _FakeMsg:
    def __init__(self, *, mtype="chat", body="", frm="", mid="", mucnick=""):
        self._data = {
            "type": mtype,
            "body": body,
            "from": frm,
            "id": mid,
            "mucnick": mucnick,
        }

    def __getitem__(self, key):
        return self._data.get(key, "")

    def get(self, key, default=None):
        v = self._data.get(key)
        return v if v else default


@pytest.mark.asyncio
async def test_on_message_dispatches_chat():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@example.org", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter.handle_message = AsyncMock()
    adapter._message_handler = lambda evt: None

    msg = _FakeMsg(mtype="chat", body="hello", frm="alice@example.org/laptop", mid="m1")
    await adapter._on_message(msg)
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.source.chat_id == "alice@example.org"
    assert event.source.chat_type == "dm"


@pytest.mark.asyncio
async def test_on_message_ignores_wrong_type():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter.handle_message = AsyncMock()
    adapter._message_handler = lambda evt: None

    await adapter._on_message(_FakeMsg(mtype="error", body="oops", frm="alice@x"))
    await adapter._on_message(_FakeMsg(mtype="groupchat", body="hi", frm="ops@conf"))
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_ignores_empty_body():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter.handle_message = AsyncMock()
    adapter._message_handler = lambda evt: None

    await adapter._on_message(_FakeMsg(mtype="chat", body="", frm="alice@x"))
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_message_ignores_self_echo():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@example.org", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter.handle_message = AsyncMock()
    adapter._message_handler = lambda evt: None

    await adapter._on_message(_FakeMsg(mtype="chat", body="hi", frm="bot@example.org/desk"))
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_groupchat_requires_addressed_prefix():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@example.org", "password": "y", "nickname": "hermes"})
    adapter = XMPPAdapter(cfg)
    adapter.handle_message = AsyncMock()
    adapter._message_handler = lambda evt: None

    # Not addressed → ignored
    await adapter._on_groupchat_message(_FakeMsg(
        mtype="groupchat", body="hello world", frm="ops@conf/alice", mucnick="alice",
    ))
    adapter.handle_message.assert_not_awaited()

    # Addressed → dispatched and prefix stripped
    await adapter._on_groupchat_message(_FakeMsg(
        mtype="groupchat", body="hermes: status?", frm="ops@conf/alice", mucnick="alice",
    ))
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "status?"
    assert event.source.chat_id == f"{MUC_PREFIX}ops@conf"
    assert event.source.chat_type == "group"
    assert event.source.user_id == "alice"


@pytest.mark.asyncio
async def test_on_groupchat_ignores_own_nick():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y", "nickname": "hermes"})
    adapter = XMPPAdapter(cfg)
    adapter.handle_message = AsyncMock()
    adapter._message_handler = lambda evt: None

    await adapter._on_groupchat_message(_FakeMsg(
        mtype="groupchat", body="hermes: ping", frm="ops@conf/hermes", mucnick="hermes",
    ))
    adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# 9. Standalone (out-of-process) send for cron
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_standalone_send_missing_slixmpp(monkeypatch):
    """When slixmpp is unimportable, return a clean error dict."""
    import sys
    saved = {name: sys.modules.pop(name)
             for name in list(sys.modules) if name == "slixmpp" or name.startswith("slixmpp.")}
    saved_meta = list(sys.meta_path)

    class _Blocker:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "slixmpp" or name.startswith("slixmpp."):
                raise ImportError("slixmpp blocked for test")
            return None

    sys.meta_path.insert(0, _Blocker())
    try:
        pconfig = MagicMock()
        pconfig.extra = {"jid": "bot@x", "password": "y"}
        result = await _standalone_send(pconfig, "alice@x", "hi")
        assert isinstance(result, dict)
        assert "error" in result
        assert "slixmpp" in result["error"].lower()
    finally:
        sys.meta_path[:] = saved_meta
        sys.modules.update(saved)


@pytest.mark.asyncio
async def test_standalone_send_missing_credentials(monkeypatch):
    monkeypatch.delenv("XMPP_JID", raising=False)
    monkeypatch.delenv("XMPP_PASSWORD", raising=False)
    pconfig = MagicMock()
    pconfig.extra = {}
    try:
        import slixmpp  # noqa: F401
    except ImportError:
        pytest.skip("slixmpp not installed")
    result = await _standalone_send(pconfig, "alice@x", "hi")
    assert isinstance(result, dict)
    assert "error" in result
    assert "XMPP_JID" in result["error"] or "required" in result["error"].lower()


@pytest.mark.asyncio
async def test_standalone_send_rejects_injection_chars(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "bot@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "secret")
    try:
        import slixmpp  # noqa: F401
    except ImportError:
        pytest.skip("slixmpp not installed")
    pconfig = MagicMock()
    pconfig.extra = {}
    result = await _standalone_send(pconfig, "alice@x\nINJECT", "hi")
    assert "error" in result
    assert "illegal" in result["error"].lower()


# ---------------------------------------------------------------------------
# 10. register() — plugin-side metadata
# ---------------------------------------------------------------------------

def test_register_calls_register_platform():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs

    assert kwargs["name"] == "xmpp"
    assert kwargs["label"] == "XMPP"
    assert kwargs["required_env"] == ["XMPP_JID", "XMPP_PASSWORD"]
    assert kwargs["allowed_users_env"] == "XMPP_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "XMPP_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "XMPP_HOME_CHANNEL"
    assert kwargs["max_message_length"] == _xmpp.MAX_MESSAGE_LENGTH
    assert kwargs["pii_safe"] is False
    assert callable(kwargs["check_fn"])
    assert callable(kwargs["validate_config"])
    assert callable(kwargs["is_connected"])
    assert callable(kwargs["env_enablement_fn"])
    assert callable(kwargs["standalone_sender_fn"])
    assert callable(kwargs["adapter_factory"])
    assert callable(kwargs["setup_fn"])
    assert "XMPP" in kwargs["platform_hint"]
