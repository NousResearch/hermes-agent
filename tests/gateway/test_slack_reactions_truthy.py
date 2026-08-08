"""SLACK_REACTIONS must honor shared truthy/falsy aliases (including 'off')."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_slack_deps() -> None:
    """Mirror tests/gateway/test_slack.py bootstrap for light collection."""
    if "slack_bolt" not in sys.modules:
        slack_bolt = MagicMock()
        slack_bolt.async_app.AsyncApp = MagicMock
        slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
        for name, mod in [
            ("slack_bolt", slack_bolt),
            ("slack_bolt.async_app", slack_bolt.async_app),
            ("slack_bolt.adapter", slack_bolt.adapter),
            ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
            (
                "slack_bolt.adapter.socket_mode.async_handler",
                slack_bolt.adapter.socket_mode.async_handler,
            ),
        ]:
            sys.modules.setdefault(name, mod)
    if "slack_sdk" not in sys.modules:
        slack_sdk = MagicMock()
        slack_sdk.web.async_client.AsyncWebClient = MagicMock
        for name, mod in [
            ("slack_sdk", slack_sdk),
            ("slack_sdk.web", MagicMock()),
            ("slack_sdk.web.async_client", MagicMock()),
        ]:
            sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_deps()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


@pytest.fixture()
def adapter():
    return SlackAdapter(PlatformConfig(enabled=True, token="***"))


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "OFF"])
def test_reactions_falsy_aliases_disable(adapter, monkeypatch, raw):
    monkeypatch.setenv("SLACK_REACTIONS", raw)
    assert adapter._reactions_enabled() is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE"])
def test_reactions_truthy_aliases_enable(adapter, monkeypatch, raw):
    monkeypatch.setenv("SLACK_REACTIONS", raw)
    assert adapter._reactions_enabled() is True


def test_reactions_defaults_on(adapter, monkeypatch):
    monkeypatch.delenv("SLACK_REACTIONS", raising=False)
    assert adapter._reactions_enabled() is True
