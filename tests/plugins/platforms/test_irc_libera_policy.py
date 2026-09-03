"""Libera.Chat must not be the recommended IRC host for agentic clients (#61181)."""

from __future__ import annotations

import pytest

from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_irc_mod = load_plugin_adapter("irc")
is_libera_chat_host = _irc_mod.is_libera_chat_host
LIBERA_POLICY_WARNING = _irc_mod.LIBERA_POLICY_WARNING
IRCAdapter = _irc_mod.IRCAdapter


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("irc.libera.chat", True),
        ("IRC.Libera.Chat", True),
        ("irc.libera.chat:6697", True),
        ("libera.chat", True),
        ("chat.libera.chat", True),
        ("127.0.0.1", False),
        ("localhost", False),
        ("irc.oftc.net", False),
        ("notlibera.chat", False),
        ("", False),
    ],
)
def test_is_libera_chat_host(host, expected):
    assert is_libera_chat_host(host) is expected


def test_recommended_server_example_is_not_libera():
    """Setup/env example host must not be a Libera hostname (behavior, not source scan)."""
    desc = OPTIONAL_ENV_VARS["IRC_SERVER"]["description"]
    assert "127.0.0.1" in desc
    assert "libera" not in desc.lower()
    assert not is_libera_chat_host("127.0.0.1")


def test_unconfigured_adapter_does_not_default_to_libera():
    from gateway.config import PlatformConfig

    adapter = IRCAdapter(PlatformConfig(enabled=True, extra={}))
    assert not is_libera_chat_host(adapter.server)


@pytest.mark.asyncio
async def test_connect_warns_when_server_is_libera(caplog, monkeypatch):
    import asyncio

    import gateway.status as status

    from gateway.config import PlatformConfig

    for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
        monkeypatch.delenv(key, raising=False)

    adapter = IRCAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "server": "irc.libera.chat",
                "port": 6667,
                "nickname": "testbot",
                "channel": "#test",
                "use_tls": False,
            },
        )
    )

    async def fake_open(*_a, **_k):
        raise OSError("no network")

    monkeypatch.setattr(asyncio, "open_connection", fake_open)
    monkeypatch.setattr(status, "acquire_scoped_lock", lambda *_a, **_k: True)
    monkeypatch.setattr(status, "release_scoped_lock", lambda *_a, **_k: None)

    with caplog.at_level("WARNING"):
        result = await adapter.connect()

    assert result is False
    assert any(LIBERA_POLICY_WARNING in rec.message for rec in caplog.records)
