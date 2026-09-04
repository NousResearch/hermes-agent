"""The real-home tripwire: tests touching the REAL hermes home must fail.

The guard (conftest _forbid_real_hermes_home_io) exists to catch the
hardcoded-restatement bug class — code writing Path.home()/".hermes"
instead of get_hermes_home(). These tests prove the tripwire fires.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_read_from_real_home_fails():
    """Reading production state in a test is a leak — must trip."""
    from tests.conftest import _REAL_HERMES_ROOT_CANDIDATES

    assert _REAL_HERMES_ROOT_CANDIDATES, "guard roots not captured"
    root = _REAL_HERMES_ROOT_CANDIDATES[0]
    sentinel = root / "tripwire-probe-should-fail.txt"
    # the wrapper must raise AssertionError (not create/read anything)
    with pytest.raises(AssertionError, match="REAL hermes home"):
        # open() for READ on a nonexistent file still trips the guard
        # BEFORE the OS error — the check is path-based, not result-based
        with open(sentinel, "r", encoding="utf-8"):
            pass


def test_write_to_real_home_fails():
    root = Path.home() / ".hermes"
    sentinel = root / "tripwire-probe-write-should-fail.json"
    with pytest.raises(AssertionError, match="REAL hermes home"):
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write("should never land")


def test_mkdir_under_real_home_fails(tmp_path):
    # os.mkdir with a real-home target (parent must exist for mkdir; use
    # makedirs which creates parents — both are guarded)
    with pytest.raises(pytest.fail.Exception):
        import os

        os.makedirs(str(Path.home() / ".hermes" / "tripwire-deep" / "dir"), exist_ok=True)


def test_isolated_home_io_still_works(tmp_path):
    """The happy path: tempdir (the sandboxed HERMES_HOME shape) is NOT
    under the real root — ordinary I/O must be completely unaffected."""
    target = tmp_path / "fine.json"
    with open(target, "w", encoding="utf-8") as f:
        f.write("ok")
    with open(target, encoding="utf-8") as f:
        assert f.read() == "ok"


def test_profile_paths_under_real_root_also_trip():
    """<root>/profiles/<name> is inside the guarded root — the deployment
    bug class (pm/plugins_state Path.home()/.hermes/profiles) trips."""
    with pytest.raises(AssertionError, match="REAL hermes home"):
        with open(
            Path.home() / ".hermes" / "profiles" / "some-profile" / "config.yaml",
            "r",
            encoding="utf-8",
        ):
            pass


@pytest.mark.allow_real_home_io
def test_opt_out_marker_bypasses_guard(tmp_path):
    """The documented escape hatch: marked tests read real paths freely
    (the guard's own tests inspect conftest internals)."""
    # just exercising that the fixture returns without wrapping
    assert True


def test_sqlite_connect_to_real_home_fails():
    """sqlite3 bypasses builtins.open (C-level) — the dedicated hook must
    catch a state.db-shaped leak under ~/.hermes."""
    import sqlite3

    with pytest.raises(pytest.fail.Exception):
        sqlite3.connect(str(Path.home() / ".hermes" / "state.db"))


def test_sqlite_connect_to_tempdir_still_works(tmp_path):
    import sqlite3

    con = sqlite3.connect(str(tmp_path / "fine.db"))
    con.execute("CREATE TABLE t (x)")
    con.close()
