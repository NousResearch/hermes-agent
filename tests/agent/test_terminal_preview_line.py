"""One-line terminal previews must spend their budget on content.

The non-verbose progress preview has a single line to describe a shell
command. It used to take only the *first* source line of the command, so a
120-character budget could render as ``set +e`` — shell commands routinely
open with boilerplate (``set -e``, a ``cd``, a variable assignment), which
is exactly when a preview is least able to afford wasting characters.

These tests pin the property rather than the wording: whatever the budget
is, a preview should carry real content up to it.
"""

import pytest

from agent.display import build_terminal_preview_line

# The command that exposed the bug, in the shape it actually arrives.
REAL_WORLD = """set +e
LIVE=/usr/local/lib/hermes-agent
echo "=== The fenced-block path in live run.py ==="
sed -n '4180,4265p' "$LIVE/gateway/run.py"
"""


class TestBudgetIsSpentOnContent:
    """The regression that started this: boilerplate must not eat the line."""

    def test_leading_boilerplate_does_not_consume_the_preview(self):
        out = build_terminal_preview_line(REAL_WORLD, 120)
        assert out != "set +e"
        assert not out.startswith("set +e ...")
        # The budget is actually used, not abandoned after the first line.
        assert len(out) > 100, f"only {len(out)} of 120 chars used: {out!r}"

    def test_content_past_the_first_line_is_reachable(self):
        out = build_terminal_preview_line(REAL_WORLD, 120)
        assert "LIVE=/usr/local/lib/hermes-agent" in out

    @pytest.mark.parametrize("boilerplate", ["set -e", "set +e", "cd /tmp", "X=1"])
    def test_any_short_opening_line_still_yields_a_useful_preview(self, boilerplate):
        cmd = f"{boilerplate}\ngrep -rn 'needle' /some/haystack/path"
        out = build_terminal_preview_line(cmd, 80)
        assert "grep -rn" in out


class TestCollapsing:
    """Newlines and indentation become single spaces."""

    def test_lines_are_joined_with_single_spaces(self):
        assert build_terminal_preview_line("echo a\necho b", 80) == "echo a echo b"

    def test_blank_lines_are_dropped(self):
        assert build_terminal_preview_line("echo a\n\n\necho b", 80) == "echo a echo b"

    def test_indentation_is_collapsed(self):
        cmd = "for f in *; do\n    echo $f\ndone"
        assert build_terminal_preview_line(cmd, 80) == "for f in *; do echo $f done"

    def test_tabs_and_runs_of_spaces_collapse(self):
        assert build_terminal_preview_line("echo\t\ta   b", 80) == "echo a b"

    def test_single_line_command_is_unchanged(self):
        assert build_terminal_preview_line("ls -la /tmp", 80) == "ls -la /tmp"


class TestCapping:
    """Truncation matches the shared preview helper, not a private rule."""

    def test_over_budget_is_truncated_with_an_ellipsis(self):
        out = build_terminal_preview_line("echo " + "x" * 200, 40)
        assert len(out) == 40
        assert out.endswith("...")

    def test_exactly_at_budget_keeps_every_character(self):
        cmd = "a" * 40
        out = build_terminal_preview_line(cmd, 40)
        assert out == cmd
        assert not out.endswith("...")

    def test_under_budget_gains_no_misleading_ellipsis(self):
        """A short multi-line command is shown in FULL, so "..." would lie."""
        out = build_terminal_preview_line("echo a\necho b", 80)
        assert not out.endswith("...")

    @pytest.mark.parametrize("cap", [0, None])
    def test_unlimited_budget_does_not_truncate(self, cap):
        cmd = "echo " + "x" * 500
        assert build_terminal_preview_line(cmd, cap) == cmd

    @pytest.mark.parametrize("tiny", [1, 2, 3])
    def test_degenerate_budgets_do_not_crash_or_go_negative(self, tiny):
        out = build_terminal_preview_line("echo hello world", tiny)
        assert len(out) == tiny


class TestDegenerateInput:
    """Empty and whitespace-only commands must not raise."""

    @pytest.mark.parametrize("cmd", ["", "   ", "\n", "\n\n  \t\n"])
    def test_empty_or_whitespace_yields_empty_string(self, cmd):
        assert build_terminal_preview_line(cmd, 80) == ""

    def test_trailing_newline_is_not_rendered_as_an_ellipsis(self):
        assert build_terminal_preview_line("ls -la\n", 80) == "ls -la"
