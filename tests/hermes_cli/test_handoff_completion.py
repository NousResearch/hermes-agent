"""Tests for /handoff slash-command autocomplete (#33070).

Covers three things the completer must get right:
  1. ``GatewayConfig.get_handoff_platforms()`` returns exactly the platforms
     ``/handoff`` would accept (enabled AND a home channel with a chat_id) —
     the shared eligibility rule the completer and the command both rely on.
  2. The completion's ``display_meta`` surfaces the home channel name, not an
     empty string (``PlatformConfig`` has no ``name``; the human-readable name
     lives on ``HomeChannel``).
  3. The completer caches the gateway config instead of re-loading it (and
     re-running plugin discovery + file I/O) on every keystroke.
"""

from __future__ import annotations

import gateway.config as gwconfig
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from hermes_cli.commands import SlashCommandCompleter
from prompt_toolkit.document import Document


def _gw(platforms):
    return GatewayConfig(platforms=platforms)


def test_get_handoff_platforms_filters_to_enabled_with_home():
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="123", name="My Phone")
    gw = _gw({
        Platform.TELEGRAM: PlatformConfig(enabled=True, home_channel=home),
        # enabled but no home channel → not eligible
        Platform.DISCORD: PlatformConfig(enabled=True, home_channel=None),
        # has a home channel but disabled → not eligible
        Platform.SLACK: PlatformConfig(
            enabled=False,
            home_channel=HomeChannel(Platform.SLACK, "999", "Slack"),
        ),
        # enabled, home present but chat_id empty → not eligible
        Platform.SIGNAL: PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(Platform.SIGNAL, "", "Signal"),
        ),
    })

    assert gw.get_handoff_platforms() == [(Platform.TELEGRAM, home)]


def test_handoff_completion_display_meta_shows_home_name(monkeypatch):
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="123", name="My Phone")
    gw = _gw({Platform.TELEGRAM: PlatformConfig(enabled=True, home_channel=home)})
    monkeypatch.setattr(gwconfig, "load_gateway_config", lambda: gw)

    completer = SlashCommandCompleter()
    comps = list(completer.get_completions(Document("/handoff te"), None))

    metas = {c.text: c.display_meta_text for c in comps}
    assert metas.get("telegram") == "My Phone"


def test_handoff_completion_caches_gateway_config(monkeypatch):
    calls = {"n": 0}
    home = HomeChannel(platform=Platform.TELEGRAM, chat_id="123", name="My Phone")
    gw = _gw({Platform.TELEGRAM: PlatformConfig(enabled=True, home_channel=home)})

    def fake_load():
        calls["n"] += 1
        return gw

    monkeypatch.setattr(gwconfig, "load_gateway_config", fake_load)

    completer = SlashCommandCompleter()
    list(completer.get_completions(Document("/handoff t"), None))
    list(completer.get_completions(Document("/handoff te"), None))

    assert calls["n"] == 1
