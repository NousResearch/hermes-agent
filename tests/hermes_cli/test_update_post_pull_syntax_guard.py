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


# ---------------------------------------------------------------------------
# _capture_head_sha
# ---------------------------------------------------------------------------

def test_capture_head_sha_returns_stripped_sha(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        assert cmd[-2:] == ["rev-parse", "HEAD"]
        return SimpleNamespace(stdout="deadbeefcafe\n", returncode=0)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    assert hermes_main._capture_head_sha(["git"], tmp_path) == "deadbeefcafe"


def test_capture_head_sha_returns_none_on_git_failure(monkeypatch, tmp_path):
    import subprocess as _sp

    def fake_run(cmd, **kwargs):
        raise _sp.CalledProcessError(returncode=128, cmd=cmd)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    assert hermes_main._capture_head_sha(["git"], tmp_path) is None


def test_capture_head_sha_returns_none_on_empty_output(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(stdout="\n", returncode=0)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    assert hermes_main._capture_head_sha(["git"], tmp_path) is None


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
    for relpath in hermes_main._UPDATE_CRITICAL_FILES:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if relpath == broken_file:
            path.write_text(broken_payload)
        else:
            path.write_text("# stub\n")


def test_validate_critical_files_syntax_ok_when_all_files_parse(tmp_path):
    _populate_critical_tree(tmp_path)

    ok, failing_path, error = hermes_main._validate_critical_files_syntax(tmp_path)

    assert ok is True
    assert failing_path is None
    assert error is None


def test_validate_critical_files_syntax_detects_conflict_markers(tmp_path):
    """The exact PR #28452 failure mode: orphan ``<<<<<<<`` in config.py."""
    _populate_critical_tree(tmp_path, broken_file="hermes_cli/config.py")

    ok, failing_path, error = hermes_main._validate_critical_files_syntax(tmp_path)

    assert ok is False
    assert failing_path is not None and failing_path.endswith("hermes_cli/config.py")
    assert error is not None
    # The error mentions either the syntax error itself or the file path —
    # either is enough proof we caught the bad commit.
    assert "SyntaxError" in str(error) or "config.py" in str(error)


def test_validate_critical_files_syntax_detects_break_in_main_py(tmp_path):
    _populate_critical_tree(tmp_path, broken_file="hermes_cli/main.py")

    ok, failing_path, _ = hermes_main._validate_critical_files_syntax(tmp_path)

    assert ok is False
    assert failing_path is not None and failing_path.endswith("hermes_cli/main.py")


def test_validate_critical_files_syntax_detects_break_in_web_server(tmp_path):
    _populate_critical_tree(tmp_path, broken_file="hermes_cli/web_server.py")

    ok, failing_path, _ = hermes_main._validate_critical_files_syntax(tmp_path)

    assert ok is False
    assert failing_path is not None and failing_path.endswith("hermes_cli/web_server.py")


def test_validate_critical_files_syntax_tolerates_missing_files(tmp_path):
    """A refactor may legitimately remove one of the critical files — the
    guard should skip missing files, not falsely flag the install as broken."""
    # Populate everything except hermes_constants.py
    for relpath in hermes_main._UPDATE_CRITICAL_FILES:
        if relpath == "hermes_constants.py":
            continue
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n")

    ok, failing_path, error = hermes_main._validate_critical_files_syntax(tmp_path)

    assert ok is True
    assert failing_path is None
    assert error is None


# ---------------------------------------------------------------------------
# Repo invariant — the production tree itself must always pass the guard.
# This catches the case where ``main`` ships a syntax error before the next
# release; if a future ``hermes update`` would brick users, this test fails
# in CI first.
# ---------------------------------------------------------------------------

def test_production_tree_passes_syntax_guard():
    """The repo itself must always satisfy the guard the update command runs."""
    repo_root = Path(__file__).resolve().parents[2]

    ok, failing_path, error = hermes_main._validate_critical_files_syntax(repo_root)

    assert ok is True, (
        f"Critical-path file {failing_path} fails to parse on current main; "
        f"hermes update would brick users. Error: {error}"
    )
