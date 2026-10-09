"""TDD regression test: parse_model_flags must ignore Block Kit newline payload.

Root cause (2026-05-15): Slack appends Block Kit JSON after a double-newline to
every message text. /model command args arriving via gateway contain:

    "deepseek/deepseek-v4-flash\n\n[Slack Block Kit payload...]\n..."

parse_model_flags called .split() on ALL whitespace, producing tokens from the
payload that were then joined with spaces → "deepseek/deepseek-v4-flash [Slack ..."
→ validate_requested_model detected space → "Model names cannot contain spaces."

Fix: strip everything after the first newline before tokenizing.
"""
import pytest
from hermes_cli.model_switch import parse_model_flags


def test_parse_model_flags_strips_block_kit_payload():
    """Model name must not include Block Kit payload appended after newline."""
    raw = "deepseek/deepseek-v4-flash\n\n[Slack Block Kit payload for this message]\nmore stuff"
    model_input, provider, is_global, force_refresh, is_session = parse_model_flags(raw)
    assert force_refresh is False
    assert is_session is False
    assert model_input == "deepseek/deepseek-v4-flash"
    assert " " not in model_input
    assert provider == ""
    assert is_global is False


def test_parse_model_flags_strips_single_newline_payload():
    """Single newline separator also stripped."""
    raw = "claude-3-opus\nsome trailing block content"
    model_input, _, _, force_refresh, is_session = parse_model_flags(raw)
    assert force_refresh is False
    assert is_session is False
    assert model_input == "claude-3-opus"


def test_parse_model_flags_with_flags_before_newline():
    """Flags before the newline are still parsed correctly."""
    raw = "sonnet --global\n\n[Block Kit junk]"
    model_input, provider, is_global, force_refresh, is_session = parse_model_flags(raw)
    assert force_refresh is False
    assert is_session is False
    assert model_input == "sonnet"
    assert is_global is True
    assert provider == ""


def test_parse_model_flags_provider_flag_before_newline():
    """--provider flag before the newline is still extracted."""
    raw = "deepseek/deepseek-v4-flash --provider openrouter\n\n[blocks]"
    model_input, provider, is_global, force_refresh, is_session = parse_model_flags(raw)
    assert force_refresh is False
    assert is_session is False
    assert model_input == "deepseek/deepseek-v4-flash"
    assert provider == "openrouter"
    assert is_global is False


def test_parse_model_flags_no_newline_unchanged():
    """Strings without newlines still work as before."""
    model_input, provider, is_global, force_refresh, is_session = parse_model_flags("gpt-4o --global")
    assert force_refresh is False
    assert is_session is False
    assert model_input == "gpt-4o"
    assert is_global is True
    assert provider == ""


def test_parse_model_flags_empty_string():
    """Empty string still returns empty model."""
    model_input, provider, is_global, force_refresh, is_session = parse_model_flags("")
    assert force_refresh is False
    assert is_session is False
    assert model_input == ""
    assert provider == ""
    assert is_global is False
