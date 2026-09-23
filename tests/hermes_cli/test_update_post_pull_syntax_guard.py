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
