"""Regression tests for state.db file-mode hardening.

Background
==========
``SessionDB.__init__`` used to create ``state.db`` via a bare
``sqlite3.connect()``, which opens the file at the process umask (``0o644``
under the common ``022`` default) with no explicit ``chmod``. On the default
posture this is only shielded by ``~/.hermes``'s ``0o700`` mode — if that
parent directory is ever widened (``HERMES_HOME_MODE`` override, managed
mode's ``0o750``), ``state.db``'s own bits become load-bearing and expose
full conversation history to other local accounts.

The fix pre-creates ``state.db`` via ``os.open(O_CREAT | O_EXCL, 0o600)``
before ``sqlite3.connect()`` ever touches it, and chmods the ``-wal``/``-shm``
sidecar files (which SQLite creates itself at the process umask and which
don't inherit the main file's mode) to ``0o600`` once WAL mode engages.

POSIX-only — mode-bit enforcement does not exist on Windows.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

from hermes_state import SessionDB


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits not enforced on Windows",
)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_state_db_created_0o600_under_permissive_umask(tmp_path):
    """A brand-new state.db must land at 0o600 even under umask 0o022."""
    db_path = tmp_path / "state.db"
    old_umask = os.umask(0o022)
    try:
        db = SessionDB(db_path=db_path)
        try:
            assert _mode(db_path) == 0o600
        finally:
            db.close()
    finally:
        os.umask(old_umask)


def test_state_db_wal_and_shm_sidecars_are_0o600(tmp_path):
    """WAL/SHM sidecars don't inherit state.db's mode from SQLite -- must be
    chmod'd explicitly once WAL mode creates them."""
    db_path = tmp_path / "state.db"
    old_umask = os.umask(0o022)
    try:
        db = SessionDB(db_path=db_path)
        try:
            wal_path = tmp_path / "state.db-wal"
            shm_path = tmp_path / "state.db-shm"
            # WAL fallback is best-effort (some filesystems don't support it,
            # see apply_wal_with_fallback) -- only assert on sidecars that
            # actually exist.
            if wal_path.exists():
                assert _mode(wal_path) == 0o600
            if shm_path.exists():
                assert _mode(shm_path) == 0o600
        finally:
            db.close()
    finally:
        os.umask(old_umask)


def test_preexisting_state_db_mode_is_left_untouched(tmp_path):
    """A file that already exists before SessionDB opens it (e.g. restored
    from a backup, or created by an older Hermes version) must not be
    silently widened or narrowed -- _create_owner_only only sets the mode
    on first creation (O_EXCL), it never chmods an existing file."""
    db_path = tmp_path / "state.db"
    db_path.touch()
    os.chmod(db_path, 0o640)
    db = SessionDB(db_path=db_path)
    try:
        assert _mode(db_path) == 0o640
    finally:
        db.close()
