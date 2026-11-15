"""Prefix tokens split by control chars or wrapped in ANSI CSI sequences must mask (#81012)."""

import base64

import pytest

from agent.redact import redact_for_egress, redact_sensitive_text


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


@pytest.mark.parametrize("after", ["\x1b[0m", ""])
def test_coloured_token_before_newline_masks_and_next_line_survives(after):
    body = "a" * 24
    result = _redact(f"\x1b[32msk-{body}{after}\nbutton [ref=e3]")
    assert body[:12] not in result
    assert result.startswith("\x1b[32msk-")
    assert result.endswith(f"{after}\nbutton [ref=e3]")


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


@pytest.mark.parametrize("seq", ["\x1b(B\x1b[m", "\x1b(B", "\x1b7", "\x1b8", "\x1bM", "\x1bOA"])
def test_non_csi_escape_before_token_masks(seq):
    # ``tput sgr0`` emits ``ESC ( B ESC [ m``; sc/rc/ri and SS3 cursor keys glue a letter likewise.
    body = "Q" * 40
    text = f"\x1b[31mOPENAI key:{seq}sk-" + "proj-" + body
    assert body[:12] not in _redact(text)
    assert body[:12] not in redact_for_egress(text)


@pytest.mark.parametrize("payload", [
    b'{"o":"1234567","n":"stack-xx","k":"Zq9"}',  # base64 ends in "==" padding
    b'{"o":"1","n":"stack-x","k":"k>>"}',         # base64 carries "+", no padding
])
@pytest.mark.parametrize("split", ["\u200b", "\x1b", "\x1b[32m"])
def test_split_grafana_cloud_token_masks_base64_tail(payload, split):
    # glc_ bodies are standard base64 (#150339): "+", "/" and trailing "=" are body chars.
    token = "glc_" + base64.b64encode(payload).decode()
    result = _redact(token[:30] + split + token[30:] + " done")
    assert token[30:-4] not in result
    assert result.endswith(" done")
