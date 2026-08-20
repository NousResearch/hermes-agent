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


def test_clamp_range_includes_default():
    """Sanity: the default must lie inside the clamp range, or every fresh
    install would hit the clamp on its very first read.
    """
    assert _DISCORD_PROMPT_TIMEOUT_MIN <= _DISCORD_PROMPT_TIMEOUT_DEFAULT <= _DISCORD_PROMPT_TIMEOUT_MAX


# ---------------------------------------------------------------------------
# approvals.discord_clarify_text_grace
# ---------------------------------------------------------------------------
# Grace window granted after a clarify's buttons expire, during which a typed
# reply still answers the prompt. Same reader shape as the timeout above, with
# one semantic difference: 0 is a meaningful value (release the agent as soon
# as the view expires), so it is NOT clamped up to a minimum.

from plugins.platforms.discord.adapter import (  # noqa: E402
    _CLARIFY_TEXT_GRACE_DEFAULT,
    _CLARIFY_TEXT_GRACE_MAX,
    _read_clarify_text_grace,
)


def test_grace_default_when_config_absent(monkeypatch):
    _patch_config(monkeypatch, {})
    assert _read_clarify_text_grace() == _CLARIFY_TEXT_GRACE_DEFAULT


def test_grace_default_when_key_missing(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_prompt_timeout": 600}})
    assert _read_clarify_text_grace() == _CLARIFY_TEXT_GRACE_DEFAULT


def test_grace_explicit_int_value(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_clarify_text_grace": 120}})
    assert _read_clarify_text_grace() == 120


def test_grace_numeric_string_accepted(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_clarify_text_grace": "90"}})
    assert _read_clarify_text_grace() == 90


def test_grace_malformed_value_falls_back_to_default(monkeypatch):
    _patch_config(
        monkeypatch,
        {"approvals": {"discord_clarify_text_grace": "five minutes"}},
    )
    assert _read_clarify_text_grace() == _CLARIFY_TEXT_GRACE_DEFAULT


def test_grace_zero_is_preserved(monkeypatch):
    """0 = release the agent the moment the buttons die. Not a typo guard."""
    _patch_config(monkeypatch, {"approvals": {"discord_clarify_text_grace": 0}})
    assert _read_clarify_text_grace() == 0


def test_grace_negative_floors_at_zero(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_clarify_text_grace": -60}})
    assert _read_clarify_text_grace() == 0


def test_grace_clamped_to_maximum(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_clarify_text_grace": 999999}})
    assert _read_clarify_text_grace() == _CLARIFY_TEXT_GRACE_MAX


def test_grace_empty_string_falls_back_to_default(monkeypatch):
    _patch_config(monkeypatch, {"approvals": {"discord_clarify_text_grace": ""}})
    assert _read_clarify_text_grace() == _CLARIFY_TEXT_GRACE_DEFAULT


def test_grace_config_read_exception_falls_back_to_default(monkeypatch):
    import hermes_cli.config
    def _boom():
        raise RuntimeError("config file corrupt")
    monkeypatch.setattr(hermes_cli.config, "read_raw_config", _boom)
    assert _read_clarify_text_grace() == _CLARIFY_TEXT_GRACE_DEFAULT


def test_default_view_timeout_plus_grace_fits_default_clarify_timeout():
    """The release task must not fire after the agent-side wait already gave
    up — on stock defaults the whole expiry dance has to fit inside
    ``agent.clarify_timeout``. Guards against any of the three defaults
    drifting apart again, which is what left sessions pinned for ~55 min.
    """
    from tools.clarify_gateway import resolve_clarify_timeout
    assert (
        _DISCORD_PROMPT_TIMEOUT_DEFAULT + _CLARIFY_TEXT_GRACE_DEFAULT
        <= resolve_clarify_timeout({})
    )
