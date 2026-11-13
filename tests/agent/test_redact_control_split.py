"""Prefix tokens split by control chars or wrapped in ANSI CSI sequences must mask (#81012)."""

import pytest

from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _redact(text):
    return redact_sensitive_text(text, force=True)


@pytest.mark.parametrize("before, after", [
    ("\x1b[32m", "\x1b[0m"),
    ("\x1b[38;5;200m", "\x1b[?25h"),
    ("\x9b31m", "\x9b0m"),
])
def test_csi_wrapped_token_masks_and_keeps_colour_codes(before, after):
    body = "a" * 24
    result = _redact(f"{before}sk-{body}{after}")
    assert body[:12] not in result
    assert result.startswith(before) and result.endswith(after)


def test_csi_split_inside_token_masks_both_fragments():
    result = _redact("sk-" + "a" * 12 + "\x1b[31m" + "b" * 12)
    assert "a" * 12 not in result
    assert "b" * 12 not in result


def test_escape_split_remainder_before_newline_masks_and_next_line_survives():
    head, mid = "sk-" + "a" * 15, "b" * 25
    for eol in ("\n", "\x1b[0m\r\n"):
        result = _redact(f"{head}\x1b{mid}{eol}button [ref=e3]")
        assert mid not in result
        assert result.endswith(f"{eol}button [ref=e3]")


def test_complete_token_before_newline_does_not_swallow_next_line():
    tok = "ghp_" + "F" * 29
    result = _redact(f"{tok}\x1b[0m\nbutton [ref=e3]")
    assert "F" * 20 not in result
    assert result.endswith("\x1b[0m\nbutton [ref=e3]")


def test_orphan_escape_never_turns_later_prose_into_noise():
    text = "sk-aaaaa\x1bNOT TOKEN, prose!\x1b[31mbbbbbbbbbb end"
    assert _redact(text) == text


def test_coloured_prose_is_untouched():
    text = "\x1b[1mbold\x1b[0m plain \x1b[32msk-learn\x1b[0m"
    assert _redact(text) == text
