"""Inline-shell snippets never turn a silent failure into an empty (looks-fine) result."""
from agent.skill_preprocessing import run_inline_shell


def test_silent_nonzero_exit_returns_marker(tmp_path):
    """rc!=0 with no stdout and no stderr is the interpreter-never-ran signature (#116818)."""
    assert run_inline_shell("exit 3", tmp_path, 5) == "[inline-shell exit 3 with no output: exit 3]"


def test_nonzero_exit_with_stderr_keeps_the_diagnostic(tmp_path):
    assert run_inline_shell('printf "diag" >&2; exit 3', tmp_path, 5) == "diag"
