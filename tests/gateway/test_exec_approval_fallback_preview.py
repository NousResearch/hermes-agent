"""The plain-text approval prompt must show the human the command they are approving."""

from gateway.platforms.base_exec_approval import (
    EA_FALLBACK_CMD_BUDGET, EA_FALLBACK_CMD_FLOOR, fit_command_preview)
from gateway.run import _format_exec_approval_fallback


def test_fallback_shows_a_long_command_in_full():
    # 600 characters, several lines: the old 200-char cap hid the destructive tail of exactly this shape.
    command = "ssh -o BatchMode=yes orion '" + "echo step; " * 50 + "doas rm -rf /srv/mcp'"
    assert 200 < len(command) <= EA_FALLBACK_CMD_BUDGET
    text = _format_exec_approval_fallback(command, "remote deletion", "/")
    assert command in text
    assert "more characters" not in text


def test_fallback_marks_a_cut_command_and_says_how_much_is_missing():
    command = "x" * (EA_FALLBACK_CMD_BUDGET + 123)
    text = _format_exec_approval_fallback(command, "big", "/")
    assert "x" * EA_FALLBACK_CMD_BUDGET in text
    assert "x" * (EA_FALLBACK_CMD_BUDGET + 1) not in text
    assert "[123 more characters not shown]" in text


def test_fit_command_preview_is_identity_at_the_budget():
    command = "y" * EA_FALLBACK_CMD_BUDGET
    assert fit_command_preview(command) == command
    assert fit_command_preview("short") == "short"



def test_without_a_cap_the_full_budget_is_used():
    text = _format_exec_approval_fallback("z" * 100_000, "dangerous command", "/")
    assert "z" * EA_FALLBACK_CMD_BUDGET in text
    assert "[97000 more characters not shown]" in text


def test_a_cap_shrinks_the_preview_so_the_prompt_is_one_message():
    # SMS: MAX_MESSAGE_LENGTH 1600 and send() does not split to it; Twilio rejects a longer body.
    text = _format_exec_approval_fallback("z" * 5000, "dangerous command", "/", max_len=1600)
    assert len(text) <= 1600
    assert "more characters not shown]" in text
    assert text.count("z") > EA_FALLBACK_CMD_FLOOR
    # As much as fits: one more character of preview adds one character of prompt (or none, where
    # the cut count loses a digit), so a maximal pick lands exactly on the cap.
    assert len(text) == 1600


def test_a_command_that_fits_the_cap_is_shown_whole():
    command = "echo ok; " * 100
    text = _format_exec_approval_fallback(command, "dangerous command", "/", max_len=1600)
    assert command in text
    assert "more characters" not in text


def test_a_command_just_over_the_cap_is_cut_and_marked():
    command = "q" * 1400
    full = _format_exec_approval_fallback(command, "dangerous command", "/")
    text = _format_exec_approval_fallback(command, "dangerous command", "/", max_len=len(full) - 1)
    assert len(text) <= len(full) - 1
    assert "more characters not shown]" in text


def test_the_cap_is_measured_with_the_adapters_length_function():
    def utf16_len(s: str) -> int:
        return len(s.encode("utf-16-le")) // 2

    command = "\U0001f525" * 2000  # two UTF-16 units each (Telegram measures this way)
    text = _format_exec_approval_fallback(command, "dangerous command", "/", max_len=1600, len_fn=utf16_len)
    assert utf16_len(text) <= 1600
    assert "more characters not shown]" in text


def test_the_preview_never_shrinks_below_the_old_cut():
    # The reason is not truncated (same as the button card), so a long one can overflow the cap;
    # the command still shows at least the 200 characters it always did.
    text = _format_exec_approval_fallback("z" * 5000, "r" * 3000, "/", max_len=1600)
    assert "z" * EA_FALLBACK_CMD_FLOOR in text
    assert "z" * (EA_FALLBACK_CMD_FLOOR + 1) not in text


def test_adapter_limits_come_from_a_real_adapter_only():
    from unittest.mock import MagicMock

    from gateway.run_turn_runner import _approval_text_limits
    from plugins.platforms.sms.adapter import SmsAdapter

    sms = object.__new__(SmsAdapter)  # no connection needed: the cap is a class attribute
    assert _approval_text_limits(sms, "+15550100")["max_len"] == 1600
    assert _approval_text_limits(MagicMock(), "chat") == {}
