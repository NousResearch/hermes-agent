"""The gateway adapter's outbound policy: the notifications mode comes from the active profile's
config.yaml through the real plugin factory (``_build_adapter``), and ``send()`` drops text that has
no visible glyph instead of posting an empty bubble. SDK-free: the adapter's bot is a recorder."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.parametrize("yaml_mode,env_mode,expected", [
    ("all", None, "all"),
    (None, None, "important"),
    ("all", "important", "important"),
    ("loud", None, "important"),
])
def test_factory_reads_notifications_mode_from_config_yaml(tmp_path, monkeypatch, yaml_mode, env_mode, expected):
    from hermes_agent.gateway.config import PlatformConfig
    from hermes_agent.plugins.platforms.telegram.adapter import _build_adapter

    if yaml_mode is not None:
        (tmp_path / "config.yaml").write_text(
            f"display:\n  platforms:\n    telegram:\n      notifications: {yaml_mode}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if env_mode is None:
        monkeypatch.delenv("HERMES_TELEGRAM_NOTIFICATIONS", raising=False)
    else:
        monkeypatch.setenv("HERMES_TELEGRAM_NOTIFICATIONS", env_mode)

    adapter = _build_adapter(PlatformConfig(enabled=True, token="1:x"))

    assert adapter._notifications_mode == expected


@pytest.mark.parametrize("content", ["\u200b", "\u200b\u2060\ufeff", " \u00ad\u200e\n", ""])
def test_send_skips_invisible_only_content(content):
    from hermes_agent.gateway.config import PlatformConfig
    from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="1:x"))
    adapter._bot = MagicMock(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))

    result = asyncio.run(adapter.send("123", content))

    assert (result.success, result.message_id) == (True, None)
    adapter._bot.send_message.assert_not_awaited()
