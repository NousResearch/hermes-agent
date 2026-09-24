"""Inline-shell snippets never turn a silent failure into an empty (looks-fine) result."""
from agent.skill_preprocessing import run_inline_shell


def test_silent_nonzero_exit_returns_marker(tmp_path):
    """rc!=0 with no stdout and no stderr is the interpreter-never-ran signature (#116818)."""
    out = run_inline_shell("exit 3", tmp_path, 5)
    assert out.strip() and "3" in out


def test_nonzero_exit_with_stderr_keeps_the_diagnostic(tmp_path):
    assert run_inline_shell('printf "diag" >&2; exit 3', tmp_path, 5) == "diag"

def test_long_output_is_elided_with_non_imitable_marker(tmp_path):
    """#121548: inline-shell output is embedded into skill bodies on every load."""
    from agent.skill_preprocessing import _INLINE_SHELL_MAX_OUTPUT

    out = run_inline_shell("printf 'a%.0s' {1..5000}", tmp_path, 5)
    assert len(out) <= _INLINE_SHELL_MAX_OUTPUT
    assert "HERMES-CONTEXT-COMPRESSION" in out
    assert out.endswith("⟫")
