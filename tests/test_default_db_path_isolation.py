"""Regression test: DEFAULT_DB_PATH must not leak into the real state.db.

``DEFAULT_DB_PATH`` is a module-level constant evaluated at import time.
The ``_hermetic_environment`` autouse fixture monkeypatches it to a tempdir
so ``SessionDB()`` without an explicit *db_path* cannot write to the
production ``~/.hermes/state.db``.  See #50681.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import hermes_state


def test_default_db_path_points_to_tempdir(tmp_path: Path) -> None:
    """DEFAULT_DB_PATH must resolve to the test-scoped tempdir, not ``~/.hermes``."""
    # The _hermetic_environment autouse fixture sets HERMES_HOME to tmp_path
    # and monkeypatches DEFAULT_DB_PATH accordingly.
    resolved = hermes_state.DEFAULT_DB_PATH
    assert tmp_path in resolved.parents or resolved.parent == tmp_path, (
        f"DEFAULT_DB_PATH {resolved} should be under tmp_path {tmp_path}"
    )
    assert ".hermes" not in str(resolved), (
        f"DEFAULT_DB_PATH {resolved} still points at the real ~/.hermes"
    )


def test_sessiondb_uses_tempdir_by_default(tmp_path: Path) -> None:
    """SessionDB() with no explicit path must use the tempdir, not production."""
    db = hermes_state.SessionDB()
    try:
        assert tmp_path in db.db_path.parents or db.db_path.parent == tmp_path, (
            f"SessionDB().db_path {db.db_path} should be under tmp_path {tmp_path}"
        )
    finally:
        db.close()


def test_default_db_path_survives_reload(tmp_path: Path) -> None:
    """After reloading hermes_state, DEFAULT_DB_PATH must still be patched."""
    importlib.reload(hermes_state)
    resolved = hermes_state.DEFAULT_DB_PATH
    assert tmp_path in resolved.parents or resolved.parent == tmp_path, (
        f"After reload, DEFAULT_DB_PATH {resolved} should still be under tmp_path"
    )
