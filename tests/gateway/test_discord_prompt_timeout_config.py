"""Tests for the configurable Discord interactive-view timeout.

Previously hardcoded to 300s on ExecApprovalView, SlashConfirmView,
UpdatePromptView, and ClarifyChoiceView. Now reads
``approvals.discord_prompt_timeout`` with the same 300s default, clamped to
``[_DISCORD_PROMPT_TIMEOUT_MIN, _DISCORD_PROMPT_TIMEOUT_MAX]`` so a typo
can't make prompts disappear (too short) or outlive Discord's 15-min
interaction-token expiry (too long).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import (  # noqa: E402
    _DISCORD_PROMPT_TIMEOUT_DEFAULT,
    _DISCORD_PROMPT_TIMEOUT_MAX,
    _DISCORD_PROMPT_TIMEOUT_MIN,
    _read_discord_prompt_timeout,
)


def _patch_config(monkeypatch, cfg):
    """Stub ``hermes_cli.config.read_raw_config`` to return ``cfg``."""
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", lambda: cfg)


def test_explicit_int_value(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_prompt_timeout": 600}})
    assert _read_discord_prompt_timeout() == 600


def test_numeric_string_accepted(monkeypatch):
    """YAML parsers occasionally return numbers as strings; tolerate it."""
    _patch_config(monkeypatch, {"approvals": {"discord_prompt_timeout": "450"}})
    assert _read_discord_prompt_timeout() == 450


def test_malformed_value_falls_back_to_default(monkeypatch):
    _patch_config(
        monkeypatch,
        {"approvals": {"discord_prompt_timeout": "five minutes"}},
    )
    assert _read_discord_prompt_timeout() == _DISCORD_PROMPT_TIMEOUT_DEFAULT


def test_value_clamped_to_minimum(monkeypatch):
    """A typo of e.g. 5 seconds must not make prompts disappear."""
    _patch_config(monkeypatch, {"approvals": {"discord_prompt_timeout": 5}})
    assert _read_discord_prompt_timeout() == _DISCORD_PROMPT_TIMEOUT_MIN


def test_default_matches_previous_hardcoded_value():
    """Behavioral parity assertion: existing installs (no new config) must
    see exactly the 300s timeout the views were hardcoded to before this
    change. Guards against the default drifting in a future refactor.
    """
    assert _DISCORD_PROMPT_TIMEOUT_DEFAULT == 300


