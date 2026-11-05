"""The conftest WAL gate must agree with the backend's WAL-reset predicate.

``tests/conftest.py::_wal_is_usable`` duplicates the SQLite WAL-reset version
predicate of ``persistence.sqlite._state.wal`` instead of importing it, so the
gate runs at collection without loading the store.

Duplication needs a guard, so these tests pin the two implementations in
agreement across the documented upstream boundaries.
"""

import sqlite3

import pytest

from hermes_state_wal import is_sqlite_wal_reset_vulnerable
from tests.conftest import _wal_is_usable


@pytest.mark.parametrize(
    "version_info",
    [
        (3, 6, 23),   # pre-WAL
        (3, 7, 0),    # first WAL release — vulnerable
        (3, 44, 5),   # below the 3.44 backport
        (3, 44, 6),   # 3.44 backport — fixed
        (3, 45, 0),   # next minor, back to vulnerable
        (3, 50, 4),   # the version that exposed this (repo .venv)
        (3, 50, 6),   # below the 3.50 backport
        (3, 50, 7),   # 3.50 backport — fixed
        (3, 51, 2),   # last vulnerable
        (3, 51, 3),   # fixed upstream
        (3, 53, 1),   # the managed runtime
    ],
)
def test_conftest_gate_agrees_with_hermes_state(version_info, monkeypatch):
    """``_wai_is_usable`` must be the exact inverse of the canonical predicate."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", version_info)
    assert _wal_is_usable() is not is_sqlite_wal_reset_vulnerable(version_info), (
        f"conftest gate and the backend predicate disagree for SQLite {version_info}"
    )
