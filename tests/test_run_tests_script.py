from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TESTS = REPO_ROOT / "scripts" / "run_tests.sh"


def test_missing_virtualenv_error_mentions_shared_worktree_fallback():
    script = RUN_TESTS.read_text(encoding="utf-8")
    expected = "$HOME/.hermes/hermes-agent/venv"

    assert expected in script

    error_lines = [line for line in script.splitlines() if "error: no virtualenv found" in line]
    assert error_lines, "run_tests.sh should emit a clear missing-virtualenv error"
    assert expected in error_lines[0]
