"""Real-profile snapshot error names the auth databases it could not read and why.

A running Chrome on macOS/Linux lets ``Cookies`` back up but holds ``Login Data`` /
``Login Data For Account`` / ``Web Data`` with a write lock, so their SQLite online backup misses
its deadline. The launch must fail closed with the database NAMES and the lock reason, never a
bare "3 database(s) unavailable" count (#111647).
"""
import json
import os
import sqlite3

import pytest

import hermes_cli.browser_connect as bc

_LOCKED = ("Login Data", "Login Data For Account", "Web Data")


def _fake_profile(root):
    (root / "Default" / "Network").mkdir(parents=True)
    (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}))
    (root / "Default" / "Preferences").write_text("{}")
    for name in ("Cookies", *_LOCKED):
        con = sqlite3.connect(root / "Default" / name)
        con.execute("create table t(x)")
        con.execute("insert into t values(1)")
        con.commit()
        con.close()


def test_mirror_profile_auth_reports_locked_dbs_by_name(tmp_path, monkeypatch):
    """Exclusive writers on the three login/autofill DBs → each is reported with the lock reason;
    the unlocked Cookies DB still lands in the snapshot (no all-or-nothing regression)."""
    root, dst = tmp_path / "real", tmp_path / "copy"
    _fake_profile(root)
    monkeypatch.setattr(bc, "_AUTH_BACKUP_DEADLINE_S", 0.3)  # keep the test fast; 5 s in prod
    holders = []
    for name in _LOCKED:
        h = sqlite3.connect(root / "Default" / name)
        h.execute("begin exclusive")
        holders.append(h)
    try:
        failed = bc._mirror_profile_auth(str(root), str(dst), "Default")
    finally:
        for h in holders:
            h.rollback()
            h.close()
    assert failed == {name: bc._AUTH_DB_LOCKED for name in _LOCKED}
    assert os.path.isfile(dst / "Default" / "Cookies")
    # Locks released → clean re-sync.
    assert bc._mirror_profile_auth(str(root), str(dst), "Default") == {}


@pytest.mark.parametrize("reason, expect", [
    (None, ("is running and holds the profile's Login Data, Login Data For Account, Web Data "
            "with a write lock", "Fully quit chrome")),
    ("file is not a database", ("Login Data: file is not a database", "Close chrome")),
])
def test_snapshot_error_names_databases_and_reason(tmp_path, monkeypatch, reason, expect):
    """The user-facing snapshot error carries the failed DB names plus the reason: the lock
    wording when every failure is the backup deadline, the SQLite error otherwise."""
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    locked_reason = reason or bc._AUTH_DB_LOCKED

    def fake_copy(src, dst_file):
        return locked_reason if os.path.basename(src) in _LOCKED else None

    monkeypatch.setattr(bc, "_copy_auth_file", fake_copy)
    dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    assert dst is None and err
    for fragment in expect:
        assert fragment in err, err
    assert "database(s) unavailable" not in err
    assert "Cookies" not in err  # only the databases that actually failed are named


@pytest.mark.parametrize("system, expect", [
    ("Darwin", ("macOS is blocking the read", "Full Disk Access", "--setup-tcc-identity")),
    ("Linux", ("owner and permissions",)),
])
def test_snapshot_reports_os_denial_not_a_browser_lock(tmp_path, monkeypatch, system, expect):
    """macOS TCC (no Full Disk Access) denies reads of the Chrome profile with EPERM; that must
    surface as a distinct denial — never as the running-browser/profile-lock message that sends
    the user through pointless Chrome restarts (#120396)."""
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    monkeypatch.setattr(bc, "_real_profile_pin", lambda: None)
    monkeypatch.setattr(bc.platform, "system", lambda: system)

    def denied_scandir(path):
        raise PermissionError(1, "Operation not permitted", path)

    monkeypatch.setattr(bc.os, "scandir", denied_scandir)
    dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    assert dst is None and err
    assert not err.startswith(bc._PROFILE_LOCKED_PREFIX)
    assert "is running" not in err
    for fragment in expect:
        assert fragment in err, err


def test_snapshot_reports_denial_from_deep_in_the_copy(tmp_path, monkeypatch):
    """A PermissionError escaping the copy itself (past the fast probe) is reported as the same
    denial message, not as a generic snapshot failure."""
    import shutil
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    monkeypatch.setattr(bc, "_real_profile_pin", lambda: None)
    monkeypatch.setattr(bc.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(shutil, "copytree",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")))
    dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    assert dst is None and err
    assert "Full Disk Access" in err, err
    assert "is running" not in err
