"""Real-profile snapshot while the browser runs, and the error when cookies can't be read.

A running Chrome on macOS/Linux lets ``Cookies`` back up but holds ``Login Data`` /
``Login Data For Account`` / ``Web Data`` with a write lock. Cookies alone keep the copy signed
in, so those three are skipped and the snapshot succeeds. When the cookie DB itself can't be
read, the error names it and the reason, never a bare count (#111647).
"""
import json
import os
import sqlite3

import pytest

import hermes_cli.browser_connect as bc

_LOCKED = ("Login Data", "Login Data For Account", "Web Data")


def _fake_profile(root):
    (root / "Default" / "Network").mkdir(parents=True)
    (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}), encoding="utf-8")
    (root / "Default" / "Preferences").write_text("{}", encoding="utf-8")
    for name in ("Cookies", *_LOCKED):
        con = sqlite3.connect(root / "Default" / name)
        con.execute("create table t(x)")
        con.execute("insert into t values(1)")
        con.commit()
        con.close()


def test_snapshot_succeeds_while_the_browser_locks_password_and_autofill_dbs(tmp_path, monkeypatch):
    """Exclusive writers on the three login/autofill DBs (what a running Chrome holds) no longer
    fail the snapshot: the cookie DB lands, the locked ones are skipped quickly."""
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    holders = []
    for name in _LOCKED:
        h = sqlite3.connect(root / "Default" / name)
        h.execute("begin exclusive")
        holders.append(h)
    try:
        dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    finally:
        for h in holders:
            h.rollback()
            h.close()
    assert err is None and dst
    assert os.path.isfile(os.path.join(dst, "Default", "Cookies"))


@pytest.mark.parametrize("reason, expect", [
    (None, ("is running and holds the profile's Cookies with a write lock", "Fully quit chrome")),
    ("file is not a database", ("Cookies: file is not a database", "Close chrome")),
])
def test_snapshot_error_names_databases_and_reason(tmp_path, monkeypatch, reason, expect):
    """An unreadable cookie DB fails the snapshot with the DB name plus the reason: the lock
    wording when the failure is the backup deadline, the SQLite error otherwise."""
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    locked_reason = reason or bc._AUTH_DB_LOCKED

    def fake_copy(src, dst_file, **_kw):
        return locked_reason if os.path.basename(src) == "Cookies" else None

    monkeypatch.setattr(bc, "_copy_auth_file", fake_copy)
    dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    assert dst is None and err
    for fragment in expect:
        assert fragment in err, err
    assert "database(s) unavailable" not in err
    assert "Login Data" not in err  # only the databases that actually failed are named
