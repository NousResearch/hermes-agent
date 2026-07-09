"""Tests for verify-before-commit.py."""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\verify-before-commit.py")


def run(script_args: list[str], stdin: str = "") -> subprocess.CompletedProcess:
    """Run the script with given args + stdin."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *script_args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_no_completion_claims_passes():
    """A message with no triggers returns exit 0."""
    r = run([], stdin="Hello world, this is a normal sentence.\nNothing special here.\n")
    assert r.returncode == 0
    assert "no unverified claims" in r.stdout


def test_unverified_done_claim_warns():
    """A 'done' claim without verification returns exit 1 in strict mode."""
    r = run(["--strict"], stdin="All done, see you tomorrow.\n")
    assert r.returncode == 1
    assert "unverified" in r.stderr.lower()
    assert "done" in r.stdout.lower()


def test_verified_claim_passes():
    """A 'done' claim followed by Verified: passes."""
    r = run(["--strict"], stdin="All done.\nVerified: pytest exit 0, output '85 passed'\n")
    assert r.returncode == 0


def test_verified_by_phrase_passes():
    """Alternative 'Verified by' phrasing is recognized."""
    r = run(["--strict"], stdin="Tests pass.\nVerified by: pytest tests/\n")
    assert r.returncode == 0


def test_backtick_command_satisfies_verification():
    """Backtick-quoted command counts as verification."""
    r = run(["--strict"], stdin="PR opened: https://...\n`gh pr view 1` returned title and state.\n")
    assert r.returncode == 0


def test_fenced_code_block_satisfies_verification():
    """``` code blocks count as verification."""
    r = run(["--strict"], stdin="Tests pass.\n```\n85 passed\n```\n")
    assert r.returncode == 0


def test_strict_flag_changes_exit_code():
    """Without --strict, warnings are emitted but exit code is 0."""
    r = run([], stdin="All done.\n")
    assert r.returncode == 0
    assert "unverified" in r.stderr.lower()


def test_multiple_unverified_claims_reported():
    """Multiple triggers → multiple warnings."""
    r = run(["--strict"], stdin="Done.\nTests pass.\nPR opened.\n")
    assert r.returncode == 1
    assert r.stdout.count("⚠") == 3


def test_verification_window_is_two_lines():
    """Verification block 3+ lines after claim does NOT satisfy."""
    r = run(["--strict"], stdin="Done.\n\n\nVerified: pytest passed\n")
    assert r.returncode == 1  # too far away


def test_far_away_verification_in_strict_mode_fails():
    """Even with --strict, verification must be in the window."""
    r = run(["--strict"], stdin="Done.\nblah blah\nblah blah\nVerified: pytest\n")
    assert r.returncode == 1


def test_x_is_fixed_pattern():
    """'X is now fixed' without verification triggers warning."""
    r = run(["--strict"], stdin="The bug is now fixed.\n")
    assert r.returncode == 1


def test_x_is_healthy_pattern():
    """'X is healthy' without verification triggers warning."""
    r = run(["--strict"], stdin="Gateway is healthy.\n")
    assert r.returncode == 1


def test_exit_0_pattern():
    """'exit code 0' claim triggers warning."""
    r = run(["--strict"], stdin="Compiled with exit code 0.\n")
    assert r.returncode == 1


def test_exit_0_with_verification_passes():
    """'exit code 0' followed by verified passes."""
    r = run(["--strict"], stdin="Compiled with exit code 0.\nVerified: build log shows SUCCESS\n")
    assert r.returncode == 0


def test_quiet_suppresses_per_finding_warnings():
    """--quiet prints summary but not per-finding details."""
    r = run(["--strict", "--quiet"], stdin="Done.\n")
    assert r.returncode == 1
    assert "⚠" not in r.stdout  # no per-finding line
    assert "1 unverified" in r.stderr


def test_stdin_dash_reads_stdin():
    """Passing '-' as filename reads from stdin."""
    r = run(["--strict", "-"], stdin="Done.\n")
    assert r.returncode == 1


def test_missing_file_returns_2():
    """Missing file returns exit 2."""
    r = run(["--strict", "/nonexistent/path/to/file.txt"])
    assert r.returncode == 2


def test_real_conversation_message_with_proper_verification_passes():
    """A realistic chat message with explicit verification blocks passes."""
    msg = """\
All 85 tests pass.
Verified: `python -m pytest tests/tools/test_memory_tool.py --timeout=30` → 85 passed in 2.59s
PR opened: https://github.com/bbasketballer75/hermes-agent/pull/1
Verified: `gh pr view 1` shows state=OPEN, additions=107
"""
    r = run(["--strict"], stdin=msg)
    assert r.returncode == 0


def test_real_conversation_message_with_weak_verification_fails():
    """A message with claims but no proper verification fails strict mode."""
    msg = """\
All 85 tests pass. PR opened. The fix is working.
"""
    r = run(["--strict"], stdin=msg)
    assert r.returncode == 1


def test_quiet_real_conversation():
    """The smoke test of the script I just wrote should pass under --strict."""
    msg = """\
- Build 7: PENDING.md for explicit-decision items (cross-session, 10 min)
Verified: `write_file` wrote 4929 bytes, sections present per scan

- Build 1: NSSM truth helper
Verified: `pytest tests/hermes_cli/test_nssm_truth.py --timeout=30` → 20 passed in 1.21s
"""
    r = run(["--strict"], stdin=msg)
    assert r.returncode == 0