"""Tests for the configurable Discord interactive-view timeout.

Previously hardcoded to 300s on ExecApprovalView, SlashConfirmView,
UpdatePromptView, and ClarifyChoiceView. Now reads
``approvals.discord_prompt_timeout`` with the same 300s default, clamped to
``[_DISCORD_PROMPT_TIMEOUT_MIN, _DISCORD_PROMPT_TIMEOUT_MAX]`` so a typo
can't make prompts disappear (too short) or outlive Discord's 15-min
interaction-token expiry (too long).
"""


from tests.discord_mock import ensure_discord_module as _ensure_discord_mock

_ensure_discord_mock()

from plugins.platforms.discord.adapter import (  # noqa: E402
    _DISCORD_PROMPT_TIMEOUT_DEFAULT,
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
