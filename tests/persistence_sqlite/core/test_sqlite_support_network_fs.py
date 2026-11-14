"""Network-filesystem journal-mode policy of ``sqlite_support.apply_wal_with_fallback``.

On an nfs/cifs/smb/sshfs mount every WAL ``-shm`` lock and unlock is a server round trip, and several
connections in one process can park in ``fcntl(F_SETLK)`` on the ``-shm`` for tens of seconds. So:

* fresh store on a network mount, journal_mode unset  -> DELETE, one WARNING naming the override;
* journal_mode explicitly ``wal``                     -> WAL (the operator's choice wins);
* journal_mode explicitly ``delete``                  -> DELETE (unchanged);
* an existing DELETE store on a network mount is not silently upgraded to WAL;
* an existing WAL store is never live-downgraded and gets one WARNING with the offline remedy;
* every other filesystem keeps the WAL default.
"""

import logging
import sqlite3

import pytest

import hermes_yaml as yaml
import sqlite_support
from sqlite_support import apply_wal_with_fallback


def _detect_network_fs(directory, mountinfo_path):
    return sqlite_support._detect_network_fs(directory, mountinfo_path=mountinfo_path)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sqlite_support, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
    monkeypatch.setattr(sqlite_support, "path_on_cross_vm_fs", lambda p: False)
    for name in ("_network_fs_warned_paths", "_network_fs_new_store_warned_paths",
                 "_journal_upgrade_warned_paths", "_delete_overridden_warned_paths"):
        if hasattr(sqlite_support, name):
            getattr(sqlite_support, name).clear()
    getattr(sqlite_support, "_network_fs_cache", {}).clear()
    return home


def _configure(home, journal_mode):
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": journal_mode}}),
                                      encoding="utf-8")


def _on_network_fs(monkeypatch, value=True):
    monkeypatch.setattr(sqlite_support, "_detect_network_fs", lambda directory, *a, **k: value, raising=False)


def _open(path):
    conn = sqlite3.connect(str(path))
    try:
        return apply_wal_with_fallback(conn, db_label=path.name), conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


def _seed(path, mode):
    seed = sqlite3.connect(str(path))
    got = str(seed.execute(f"PRAGMA journal_mode={mode}").fetchone()[0]).lower()
    if got != mode.lower():
        seed.close()
        pytest.skip(f"environment refuses journal_mode={mode}")
    seed.execute("CREATE TABLE t (x)")
    seed.commit()
    seed.close()


def _mountinfo(tmp_path, lines):
    p = tmp_path / "mountinfo"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def test_fresh_store_on_network_fs_starts_in_delete_and_warns_once(tmp_path, monkeypatch, caplog):
    _on_network_fs(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=sqlite_support.logger.name):
        assert _open(tmp_path / "state.db") == ("delete", "delete")
        assert _open(tmp_path / "state.db") == ("delete", "delete")
    lines = [r for r in caplog.records if "network filesystem" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert "database.journal_mode: wal" in lines[0].getMessage()


def test_same_fresh_store_off_network_fs_still_gets_wal(tmp_path, monkeypatch):
    _on_network_fs(monkeypatch, False)
    mode, on_disk = _open(tmp_path / "state.db")
    if mode != "wal":
        pytest.skip("environment refuses WAL for unrelated reasons")
    assert on_disk == "wal"


def test_explicit_wal_config_wins_on_network_fs(tmp_path, monkeypatch, _isolate):
    _configure(_isolate, "wal")
    _on_network_fs(monkeypatch, False)
    if _open(tmp_path / "control.db")[0] != "wal":  # gate the skip on the environment, never on the result
        pytest.skip("environment refuses WAL for unrelated reasons")
    _on_network_fs(monkeypatch)
    sqlite_support._network_fs_cache.clear()
    assert _open(tmp_path / "state.db") == ("wal", "wal")


def test_explicit_delete_config_unchanged_on_network_fs(tmp_path, monkeypatch, _isolate):
    _configure(_isolate, "delete")
    _on_network_fs(monkeypatch)
    assert _open(tmp_path / "state.db") == ("delete", "delete")


def test_existing_delete_store_on_network_fs_is_not_upgraded_to_wal(tmp_path, monkeypatch, caplog):
    db = tmp_path / "kanban.db"
    _seed(db, "DELETE")
    _on_network_fs(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=sqlite_support.logger.name):
        assert _open(db) == ("delete", "delete")
    assert not [r for r in caplog.records if "switched to WAL" in r.getMessage()]


def test_existing_wal_store_on_network_fs_keeps_wal_and_names_the_offline_remedy(tmp_path, monkeypatch, caplog):
    db = tmp_path / "state.db"
    _seed(db, "WAL")
    _on_network_fs(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=sqlite_support.logger.name):
        for _ in range(2):
            assert _open(db) == ("wal", "wal")  # never live-downgraded
    lines = [r for r in caplog.records if "network filesystem" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    text = lines[0].getMessage()
    assert "hermes sessions set-journal-mode delete" in text
    assert "database.journal_mode: delete" in text


@pytest.mark.parametrize("wal_reset_vulnerable", [False, True])
def test_existing_wal_db_on_network_fs_warns_once_at_warning(tmp_path, monkeypatch, caplog, wal_reset_vulnerable):
    # Both already-WAL early-return paths (regular and WAL-reset-vulnerable) warn, never at ERROR.
    monkeypatch.setattr(sqlite_support, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: wal_reset_vulnerable)
    monkeypatch.setattr(sqlite_support, "resolve_journal_mode", lambda: "wal")
    db = tmp_path / "already-wal.db"
    _seed(db, "WAL")
    monkeypatch.setattr(sqlite_support, "_path_on_network_fs", lambda p: True, raising=False)
    with caplog.at_level(logging.WARNING, logger=sqlite_support.logger.name):
        for _ in range(2):
            conn = sqlite3.connect(str(db))
            assert apply_wal_with_fallback(conn, db_label="state.db") == "wal"
            conn.close()
    warnings = [r for r in caplog.records if "network filesystem" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_fresh_db_on_network_fs_unconfigured_starts_in_delete_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sqlite_support, "resolve_journal_mode", lambda: "wal")
    monkeypatch.setattr(sqlite_support, "_path_on_network_fs", lambda p: True, raising=False)
    monkeypatch.setattr(sqlite_support, "_journal_mode_explicitly_configured", lambda: False, raising=False)
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    with caplog.at_level(logging.WARNING, logger=sqlite_support.logger.name):
        mode = apply_wal_with_fallback(conn, db_label="fresh.db")
    conn.close()
    assert mode == "delete"
    assert [r for r in caplog.records if "network filesystem" in r.getMessage()]


# Realistic mountinfo rows (id parent major:minor root mountpoint opts ... - fstype source superopts)
ROOT_EXT4 = "25 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"
BIND_NFS = "620 25 0:60 / /srv/nas rw,relatime - nfs nas:/export/hermes rw"
BIND_CIFS = "621 25 0:61 / /mnt/share rw,relatime - cifs //fileserver/hermes rw"
BIND_FUSE_SSHFS = "622 25 0:62 / /home/remote rw,relatime - fuse.sshfs host:/home/user rw"
BIND_VIRTIOFS = "623 25 0:63 / /data rw,relatime - virtiofs mount0 rw"


@pytest.fixture()
def _linux(monkeypatch):
    monkeypatch.setattr(sqlite_support.sys, "platform", "linux")


@pytest.mark.usefixtures("_linux")
@pytest.mark.parametrize("path,expected", [
    ("/srv/nas/agent", True),
    ("/mnt/share/db", True),
    ("/home/remote/.hermes", True),
    ("/home/local/.hermes", False),
    ("/data/agent", False),  # virtiofs belongs to the cross-VM class, not this one
])
def test_only_network_mounts_are_flagged(tmp_path, path, expected):
    mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_NFS, BIND_CIFS, BIND_FUSE_SSHFS, BIND_VIRTIOFS])
    assert _detect_network_fs(path, mountinfo_path=mi) is expected


@pytest.mark.usefixtures("_linux")
@pytest.mark.parametrize("fstype", [
    "ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay", "apfs", "f2fs", "fuse.ntfs3g", "fuse.rofs",
])
def test_local_and_generic_fuse_filesystems_never_flagged(tmp_path, fstype):
    mi = _mountinfo(tmp_path, [f"25 1 8:1 / / rw,relatime shared:1 - {fstype} /dev/sda1 rw"])
    assert _detect_network_fs("/home/user/.hermes", mountinfo_path=mi) is False


@pytest.mark.usefixtures("_linux")
def test_network_fstypes_cover_nfs_and_smb_variants(tmp_path):
    rows = [ROOT_EXT4] + [
        f"{600 + i} 25 0:{60 + i} / /mnt/{fs.replace('.', '_')} rw - {fs} srv:/x rw"
        for i, fs in enumerate(("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"))]
    mi = _mountinfo(tmp_path, rows)
    for fs in ("nfs", "nfs4", "cifs", "smb3", "fuse_sshfs"):
        assert _detect_network_fs(f"/mnt/{fs}/home", mountinfo_path=mi) is True
    assert _detect_network_fs("/home/user/.hermes", mountinfo_path=mi) is False


def test_missing_mountinfo_conservative_false(tmp_path):
    assert _detect_network_fs("/srv/nas", mountinfo_path=str(tmp_path / "nope")) is False
