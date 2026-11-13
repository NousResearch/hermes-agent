"""`!<command>` execution bounds — the cases reviewers reproduced by hand (#2637).

Upstream main already routes `!` through `handle_bang_shell` and renders its output through
``_rich_text_from_ansi``; these tests pin the remaining findings: literal bracketed output on
both streams, real exit codes, a bounded display, and stdin that cannot hang the composer.
"""

import pytest

from hermes_cli.bang_shell import MAX_OUTPUT_BYTES, run_bang_command


class TestOutputFidelity:
    """Reviewer finding: command output must never be parsed as Rich markup."""

    @pytest.mark.parametrize("payload", ["[/]", "[red]", "[bold]x[/bold]", "[INFO] ok"])
    def test_bracket_output_survives_both_streams(self, payload):
        lines = []
        code = run_bang_command(
            f"printf '%s\n' {payload!r}; printf '%s\n' {payload!r} >&2", writer=lines.append)
        assert code == 0
        assert payload in lines

    def test_nonzero_exit_is_returned_even_with_output(self):
        lines = []
        assert run_bang_command("echo boom; exit 42", writer=lines.append) == 42
        assert "boom" in lines

    def test_stderr_is_merged_into_the_stream(self):
        lines = []
        run_bang_command("echo to-stderr >&2", writer=lines.append)
        assert "to-stderr" in lines


class TestBounds:
    def test_cap_is_a_sane_size(self):
        assert 0 < MAX_OUTPUT_BYTES <= 4 * 1024 * 1024

    def test_endless_output_is_truncated(self):
        lines = []
        run_bang_command("yes flood", timeout=30, writer=lines.append, max_output_bytes=2048)
        assert any("truncated" in str(line) for line in lines)

    def test_output_after_the_cap_is_not_emitted(self):
        lines = []
        run_bang_command("yes flood", timeout=30, writer=lines.append, max_output_bytes=2048)
        assert sum(len(str(line)) for line in lines) < 8 * 1024


class TestInteractivity:
    def test_stdin_is_not_inherited(self):
        """A command that reads stdin must see EOF, not the tty."""
        lines = []
        code = run_bang_command("cat", timeout=10, writer=lines.append)
        assert code == 0
        assert lines == []
