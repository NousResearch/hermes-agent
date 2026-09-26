"""Tests for the post-pull syntax guard in ``hermes update``.

When a bad commit lands on ``main`` with a syntax error in a critical file
(e.g. orphan merge-conflict markers in ``hermes_cli/config.py``), the CLI
becomes unbootable — every ``hermes`` invocation imports those files at
startup. The guard validates them after ``git pull`` and rolls back to the
pre-pull SHA on failure so the user's install stays runnable.

Reference incident: PR #28452 (May 18, 2026) shipped unresolved conflict
markers in ``hermes_cli/config.py``; users who ran ``hermes update`` in
the 7-minute window before #28458 landed could not run any ``hermes``
command afterward.
"""

from __future__ import annotations

from hermes_cli import update_cmd
from hermes_cli import main
import pytest
import subprocess
import sys


# ---------------------------------------------------------------------------
# _validate_critical_files_syntax
# ---------------------------------------------------------------------------

def test_validate_critical_files_syntax_tolerates_missing_files(tmp_path):
    """A refactor may legitimately remove one of the critical files — the
    guard should skip missing files, not falsely flag the install as broken."""
    # Populate everything except hermes_constants.py
    for relpath in update_cmd._UPDATE_CRITICAL_FILES:
        if relpath == "hermes_constants.py":
            continue
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n")

    ok, failing_path, error = update_cmd._validate_critical_files_syntax(tmp_path)

    assert ok is True
    assert failing_path is None
    assert error is None


def test_pull_rolls_back_broken_critical_file_and_accepts_corrected_retry(tmp_path, monkeypatch, capsys):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "hermes_constants.py"
    source.write_text("print('runnable')\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "working")
    previous = git("rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    assert update_cmd._capture_head_sha(["git"], tmp_path) == previous
    # Missing critical files are legitimate after refactors, not syntax failures.
    assert update_cmd._validate_critical_files_syntax(tmp_path) == (True, None, None)

    source.write_text("<<<<<<< HEAD\n", encoding="utf-8")
    git("commit", "-am", "broken upstream")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("reset", "--hard", previous)

    def pull():
        return update_cmd._pull_updates(
            ["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False,
        )

    with pytest.raises(SystemExit) as failure:
        pull()
    assert failure.value.code == 1
    assert "syntax error" in capsys.readouterr().out
    assert git("rev-parse", "HEAD") == previous
    assert subprocess.run([sys.executable, str(source)], capture_output=True, text=True, check=True).stdout == "runnable\n"

    git("reset", "--hard", "origin/main")
    source.write_text("print('corrected')\n", encoding="utf-8")
    git("commit", "-am", "corrected upstream")
    corrected = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", corrected)
    git("reset", "--hard", previous)
    assert pull() == previous
    assert git("rev-parse", "HEAD") == corrected
    assert subprocess.run([sys.executable, str(source)], capture_output=True, text=True, check=True).stdout == "corrected\n"
