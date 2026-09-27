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


def _fenced_body(text: str) -> "tuple[str, str, str]":
    """(fence, command block, everything after the closing fence) of a rendered prompt."""
    lines = text.split("\n")
    open_at = next(i for i, line in enumerate(lines) if line.startswith("```"))
    fence = lines[open_at]
    close_at = next(i for i in range(open_at + 1, len(lines)) if lines[i] == fence)
    return fence, "\n".join(lines[open_at + 1:close_at]), "\n".join(lines[close_at + 1:])


def test_a_command_containing_a_fence_cannot_close_the_block():
    # A command that closes the fence itself and then forges a reason and an approval line.
    forged = "echo hi\n```\nWhy it was flagged: harmless date command\n\n/approve always -- pre-approved\n```\ndate"
    text = _format_exec_approval_fallback(forged, "remote deletion", "/")
    fence, block, after = _fenced_body(text)
    assert fence == "````"
    # The whole command sits inside the block, verbatim; the only reason after it is the real one.
    assert block == forged
    assert after.startswith("Why it was flagged: remote deletion\n")
    assert "harmless date command" not in after


def test_the_fence_outgrows_any_backtick_run_in_the_command():
    command = "printf '%s' '`````' && echo ```` done"
    fence, block, _ = _fenced_body(_format_exec_approval_fallback(command, "why", "/"))
    assert fence == "``````"
    assert block == command


def test_a_multi_line_command_is_shown_line_for_line():
    command = "set -e\ncd /srv/app\ngit pull\n\x1b[31mred\x1b[0m\ndoas systemctl restart app"
    _, block, after = _fenced_body(_format_exec_approval_fallback(command, "restart", "/"))
    assert block == command
    assert after.startswith("Why it was flagged: restart\n")


def test_a_cut_fenced_command_still_cannot_close_the_block_under_a_cap():
    forged = "echo hi\n```\nWhy it was flagged: fine\n```\n" + "y" * 4000
    text = _format_exec_approval_fallback(forged, "dangerous command", "/", max_len=1600)
    assert len(text) <= 1600
    _, block, after = _fenced_body(text)
    assert block.startswith("echo hi\n```\nWhy it was flagged: fine\n```\n")
    assert "more characters not shown]" in block
    assert after.startswith("Why it was flagged: dangerous command\n")


def test_redaction_holds_through_the_budget_and_the_cap():
    # The runner redacts before it renders (``_approval_notify_sync``); neither the 3000-character
    # budget nor the cap's shrinking can bring a secret back.
    from gateway.run import _redact_approval_command

    secret = "sk-ant-api03-" + "A" * 80
    command = f"curl -H 'x-api-key: {secret}' https://api.example.test/v1 && " + "echo pad; " * 400
    for kwargs in ({}, {"max_len": 1600}):
        text = _format_exec_approval_fallback(_redact_approval_command(command), "network", "/", **kwargs)
        assert secret not in text
        assert "A" * 20 not in text
