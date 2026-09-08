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

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# _capture_head_sha
# ---------------------------------------------------------------------------

def test_capture_head_sha_returns_stripped_sha(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        assert cmd[-2:] == ["rev-parse", "HEAD"]
        return SimpleNamespace(stdout="deadbeefcafe\n", returncode=0)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    assert update_cmd._capture_head_sha(["git"], tmp_path) == "deadbeefcafe"


# ---------------------------------------------------------------------------
# _validate_critical_files_syntax
# ---------------------------------------------------------------------------

def _populate_critical_tree(root: Path, *, broken_file: str | None = None) -> None:
    """Create stub files for every entry in ``_UPDATE_CRITICAL_FILES``.

    If ``broken_file`` is given, that file gets orphan merge-conflict markers
    (the exact failure mode from PR #28452).
    """
    broken_payload = (
        "x = {\n"
        '    "a": 1,\n'
        "<<<<<<< HEAD\n"
        '    "b": 2,\n'
        "=======\n"
        '    "c": 0b6d673e7,\n'  # invalid binary literal — the actual error users saw
        ">>>>>>> 0b6d673e7\n"
        "}\n"
    )
    for relpath in update_cmd._UPDATE_CRITICAL_FILES:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if relpath == broken_file:
            path.write_text(broken_payload)
        else:
            path.write_text("# stub\n")




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


# ---------------------------------------------------------------------------
# Repo invariant — the production tree itself must always pass the guard.
# This catches the case where ``main`` ships a syntax error before the next
# release; if a future ``hermes update`` would brick users, this test fails
# in CI first.
# ---------------------------------------------------------------------------

