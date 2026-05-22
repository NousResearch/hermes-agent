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
_markdown_to_xhtml_im = _xmpp._markdown_to_xhtml_im
_markdown_to_xhtml_im_fallback = _xmpp._markdown_to_xhtml_im_fallback
_html_escape = _xmpp._html_escape
_sanitize_link_url = _xmpp._sanitize_link_url
_content_type_for = _xmpp._content_type_for
_UploadUnavailable = _xmpp._UploadUnavailable
MUC_PREFIX = _xmpp.MUC_PREFIX


class _FakeStanza:
    """Minimal slixmpp stanza stand-in that records ``stanza['html']['body']``."""

    class _Slot:
        def __init__(self, parent: "_FakeStanza", key: str):
            self._parent = parent
            self._key = key

        def __setitem__(self, k, v):
            if self._key == "html" and k == "body":
                self._parent._html_body = v
            elif self._key == "oob" and k == "url":
                self._parent._oob_url = v

        def __getitem__(self, k):  # pragma: no cover - not used by adapter
            return None

    def __init__(self, mto: str, mbody: str, mtype: str):
        self.mto = mto
        self.mbody = mbody
        self.mtype = mtype
        self._html_body = None
        self._oob_url = None
        self._sent = False
        self._sent_by: list = []

    def __getitem__(self, key):
        return _FakeStanza._Slot(self, key)

    def send(self):
        self._sent = True
        self._sent_by.append(self)


def _make_mock_client(*, has_html: bool = False):
    """Build a MagicMock that mirrors slixmpp's surface for tests.

    The adapter calls ``client.make_message(mto=, mbody=, mtype=).send()``
    (returning the stanza for ``stanza["html"]["body"]`` assignment), and
    consults ``"xep_0071" in client.plugin`` to decide whether to attach
    XHTML-IM.
    """
    client = MagicMock()
    client.plugin = {"xep_0071": MagicMock()} if has_html else {}
    sent_stanzas: list = []

    def _make_message(*, mto, mbody, mtype, **_kwargs):
        stanza = _FakeStanza(mto=mto, mbody=mbody, mtype=mtype)
        original_send = stanza.send

        def _record():
            sent_stanzas.append(stanza)
            return original_send()

        stanza.send = _record  # type: ignore[assignment]
        return stanza

    client.make_message.side_effect = _make_message
    client._sent_stanzas = sent_stanzas
    return client


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
async def test_send_dm_strips_markdown_in_body():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()

    result = await adapter.send("alice@example.org", "**hi**")
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert len(sent) == 1
    assert sent[0].mto == "alice@example.org"
    assert sent[0].mtype == "chat"
    assert "**" not in sent[0].mbody


@pytest.mark.asyncio
async def test_send_muc_uses_groupchat():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()

    result = await adapter.send(f"{MUC_PREFIX}ops@conf", "hello")
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert sent[0].mto == "ops@conf"
    assert sent[0].mtype == "groupchat"


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
    adapter._client = _make_mock_client()

    result = await adapter.send("alice@example.org", "\x00 \r\n  ")
    assert result.success is True
    assert adapter._client.make_message.call_count == 0


@pytest.mark.asyncio
async def test_send_splits_long_message():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y", "max_message_length": 100})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()

    body = "lorem ipsum " * 50
    result = await adapter.send("alice@example.org", body)
    assert result.success is True
    assert len(adapter._client._sent_stanzas) >= 2


@pytest.mark.asyncio
async def test_send_emits_xhtml_im_for_single_chunk():
    """Single-chunk markdown messages should carry both <body> and <html>."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client(has_html=True)

    result = await adapter.send("alice@example.org", "**hello** _world_")
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert len(sent) == 1
    assert sent[0]._html_body, "single-chunk markdown should set the html body"
    assert "<strong>" in sent[0]._html_body or "<b>" in sent[0]._html_body


@pytest.mark.asyncio
async def test_send_skips_xhtml_im_when_no_formatting():
    """Plain text messages should not waste bytes on an empty <html> block."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client(has_html=True)

    result = await adapter.send("alice@example.org", "plain text reply")
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert sent[0]._html_body in (None, "")


@pytest.mark.asyncio
async def test_send_skips_xhtml_im_on_multi_chunk():
    """Chunked sends should fall through to plain text only."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y", "max_message_length": 80})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client(has_html=True)

    result = await adapter.send("alice@example.org", "**bold** " + ("lorem " * 30))
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert len(sent) >= 2
    for stanza in sent:
        assert stanza._html_body in (None, "")


@pytest.mark.asyncio
async def test_send_respects_html_formatting_disabled():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={
        "jid": "bot@x", "password": "y", "html_formatting": False,
    })
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client(has_html=True)

    result = await adapter.send("alice@example.org", "**hi**")
    assert result.success is True
    assert adapter._client._sent_stanzas[0]._html_body in (None, "")


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
    # Adapter advertises media support via XEP-0363; the hint should
    # tell the agent it can emit MEDIA: tags.
    assert "MEDIA" in kwargs["platform_hint"]


# ---------------------------------------------------------------------------
# 11. XHTML-IM (XEP-0071) helpers
# ---------------------------------------------------------------------------

def test_html_escape_basic():
    assert _html_escape("<a>&\"x") == "&lt;a&gt;&amp;&quot;x"


def test_sanitize_link_url_rejects_dangerous_schemes():
    assert _sanitize_link_url("javascript:alert(1)") == ""
    assert _sanitize_link_url("DATA:text/html,<script>") == ""
    assert _sanitize_link_url("vbscript:msgbox") == ""
    assert _sanitize_link_url("https://example.org/x") == "https://example.org/x"


def test_markdown_to_xhtml_im_bold_italic_code():
    html = _markdown_to_xhtml_im("**bold** _italic_ `code`")
    assert "bold" in html and "italic" in html and "code" in html
    assert "<strong>" in html or "<b>" in html
    assert "<em>" in html or "<i>" in html
    assert "<code>" in html


def test_markdown_to_xhtml_im_link():
    html = _markdown_to_xhtml_im("see [docs](https://example.org/x)")
    assert "<a href=\"https://example.org/x\">" in html
    assert ">docs</a>" in html


def test_markdown_to_xhtml_im_escapes_html_in_code():
    html = _markdown_to_xhtml_im("`<script>alert(1)</script>`")
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_markdown_to_xhtml_im_fenced_code_block():
    html = _markdown_to_xhtml_im("```python\nprint('hi')\n```")
    assert "<pre>" in html
    # The python ``markdown`` library may emit ``<code class="language-…">``;
    # the regex fallback emits a bare ``<code>``. Both are valid XHTML-IM.
    assert "<code" in html
    assert "print" in html


def test_markdown_to_xhtml_im_empty_returns_empty():
    assert _markdown_to_xhtml_im("") == ""
    assert _markdown_to_xhtml_im("   ").strip() == ""


def test_markdown_to_xhtml_im_fallback_path_explicit():
    """The fallback path must produce a usable subset even without the
    optional ``markdown`` package available at import time."""
    html = _markdown_to_xhtml_im_fallback("**a** [x](https://y/) `<b>`")
    assert "<strong>" in html
    assert "<a href=\"https://y/\">" in html
    assert "&lt;b&gt;" in html


# ---------------------------------------------------------------------------
# 12. HTTP Upload (XEP-0363)
# ---------------------------------------------------------------------------

def test_content_type_for_known_extensions():
    assert _content_type_for("/tmp/x.png") == "image/png"
    assert _content_type_for("/tmp/x.jpg") == "image/jpeg"
    assert _content_type_for("/tmp/x.ogg") == "audio/ogg"
    assert _content_type_for("/tmp/x.mp4") == "video/mp4"
    assert _content_type_for("/tmp/x.pdf") == "application/pdf"
    assert _content_type_for("/tmp/x.weirdext") == "application/octet-stream"
    assert _content_type_for("/tmp/no-extension") == "application/octet-stream"


@pytest.mark.asyncio
async def test_send_image_url_uses_oob_for_https():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()

    result = await adapter.send_image("alice@example.org", "https://example.org/cat.png", caption="look")
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert "https://example.org/cat.png" in sent[0].mbody
    assert "look" in sent[0].mbody


@pytest.mark.asyncio
async def test_upload_then_send_falls_back_when_no_service(tmp_path):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()
    adapter._upload_service_resolved = False  # no upload component on the server

    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    result = await adapter._upload_then_send("alice@example.org", str(img), caption="cute")
    assert result.success is True
    sent = adapter._client._sent_stanzas
    assert any("cat.png" in s.mbody for s in sent)
    assert any("upload failed" in s.mbody for s in sent)


@pytest.mark.asyncio
async def test_upload_then_send_rejects_missing_file():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()

    result = await adapter._upload_then_send("alice@x", "/tmp/does-not-exist.png")
    assert result.success is False
    assert "not found" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_upload_then_send_emits_oob_url_on_success(tmp_path):
    """Mock the slixmpp upload plugin and verify the resulting send."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()

    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)

    upload_plugin = MagicMock()
    upload_plugin.upload_file = AsyncMock(return_value="https://uploads.example.org/cat.png")
    adapter._client.plugin = {"xep_0363": upload_plugin}
    adapter._upload_service_resolved = "uploads.example.org"

    result = await adapter._upload_then_send("alice@example.org", str(img), caption="cute")
    assert result.success is True
    upload_plugin.upload_file.assert_awaited_once()
    sent = adapter._client._sent_stanzas
    assert "https://uploads.example.org/cat.png" in sent[0].mbody
    assert "cute" in sent[0].mbody


@pytest.mark.asyncio
async def test_upload_file_raises_unavailable_when_service_missing():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"jid": "bot@x", "password": "y"})
    adapter = XMPPAdapter(cfg)
    adapter._client = _make_mock_client()
    adapter._upload_service_resolved = False

    with pytest.raises(_UploadUnavailable):
        await adapter._upload_file("/tmp/anything.png")


@pytest.mark.asyncio
async def test_resolve_upload_service_pinned_skips_discovery(monkeypatch):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={
        "jid": "bot@x", "password": "y", "upload_service": "uploads.example.org",
    })
    adapter = XMPPAdapter(cfg)
    adapter._client = MagicMock()
    # find_upload_service should not be called when the operator pins one
    upload_plugin = MagicMock()
    upload_plugin.find_upload_service = AsyncMock()
    adapter._client.plugin = {"xep_0363": upload_plugin}

    await adapter._resolve_upload_service()
    upload_plugin.find_upload_service.assert_not_awaited()
    assert adapter._upload_service_resolved == "uploads.example.org"


# ---------------------------------------------------------------------------
# 13. New env-vars surface via _env_enablement
# ---------------------------------------------------------------------------

def test_env_enablement_seeds_upload_and_html_keys(monkeypatch):
    monkeypatch.setenv("XMPP_JID", "bot@example.org")
    monkeypatch.setenv("XMPP_PASSWORD", "secret")
    monkeypatch.setenv("XMPP_UPLOAD_SERVICE", "uploads.example.org")
    monkeypatch.setenv("XMPP_HTML_FORMATTING", "false")

    seed = _env_enablement()
    assert seed["upload_service"] == "uploads.example.org"
    assert seed["html_formatting"] is False


def test_adapter_init_reads_new_keys(monkeypatch):
    from gateway.config import PlatformConfig
    for v in ("XMPP_UPLOAD_SERVICE", "XMPP_HTML_FORMATTING"):
        monkeypatch.delenv(v, raising=False)

    cfg = PlatformConfig(enabled=True, extra={
        "jid": "bot@x", "password": "y",
        "upload_service": "uploads.x.org",
        "html_formatting": False,
    })
    adapter = XMPPAdapter(cfg)
    assert adapter.upload_service == "uploads.x.org"
    assert adapter.html_formatting is False
