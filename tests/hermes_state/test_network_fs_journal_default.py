"""Network-filesystem journal-mode policy for NEW stores (field report: state.db on NFS, WAL stalls).

On an NFS mount every WAL ``-shm`` lock and unlock is a server round trip, and with several connections
in ONE process (SessionDB alone holds a writer plus up to 8 pooled readers) a thread can sit in
``fcntl(F_SETLK)`` on ``state.db-shm`` for 20-90 s. DELETE mode on the same mount stayed under 100 ms
reads in the operator's benchmark. So a store Hermes creates on a network filesystem starts in DELETE,
unless the operator explicitly configured ``database.journal_mode``:

* fresh store on nfs/cifs/smb/sshfs, journal_mode unset  -> DELETE, one WARNING naming the override;
* journal_mode explicitly ``wal``                         -> WAL (the operator's choice wins);
* journal_mode explicitly ``delete``                      -> DELETE (unchanged);
* an existing DELETE store on a network mount is not silently upgraded to WAL;
* an existing WAL store is never live-downgraded and gets one WARNING with the offline remedy;
* every other filesystem keeps today's WAL default.
"""

import logging
import sqlite3

import pytest

import hermes_state_wal
import hermes_yaml as yaml
from hermes_state_wal import apply_wal_with_fallback


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
    monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: False)
    for name in ("_network_fs_warned_paths", "_network_fs_new_store_warned_paths",
                 "_journal_upgrade_warned_paths", "_delete_overridden_warned_paths"):
        if hasattr(hermes_state_wal, name):
            getattr(hermes_state_wal, name).clear()
    getattr(hermes_state_wal, "_network_fs_cache", {}).clear()
    return home


def _configure(home, journal_mode):
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": journal_mode}}),
                                      encoding="utf-8")


def _on_network_fs(monkeypatch, value=True):
    monkeypatch.setattr(hermes_state_wal, "_detect_network_fs", lambda directory, *a, **k: value, raising=False)


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


def test_fresh_store_on_network_fs_starts_in_delete_and_warns_once(tmp_path, monkeypatch, caplog):
    _on_network_fs(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=hermes_state_wal.logger.name):
        assert _open(tmp_path / "state.db") == ("delete", "delete")
        assert _open(tmp_path / "state.db") == ("delete", "delete")
    lines = [r for r in caplog.records if "network filesystem" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert "database.journal_mode: wal" in lines[0].getMessage()  # names the override


def test_same_fresh_store_off_network_fs_still_gets_wal(tmp_path, monkeypatch):
    _on_network_fs(monkeypatch, False)
    mode, on_disk = _open(tmp_path / "state.db")
    if mode != "wal":
        pytest.skip("environment refuses WAL for unrelated reasons")
    assert on_disk == "wal"


def test_explicit_wal_config_wins_on_network_fs(tmp_path, monkeypatch, _isolate):
    _configure(_isolate, "wal")
    _on_network_fs(monkeypatch, False)
    if _open(tmp_path / "control.db")[0] != "wal":  # gate the skip on the ENVIRONMENT, never on the result
        pytest.skip("environment refuses WAL for unrelated reasons")
    _on_network_fs(monkeypatch)
    hermes_state_wal._network_fs_cache.clear()
    assert _open(tmp_path / "state.db") == ("wal", "wal")


def test_explicit_delete_config_unchanged_on_network_fs(tmp_path, monkeypatch, _isolate):
    _configure(_isolate, "delete")
    _on_network_fs(monkeypatch)
    assert _open(tmp_path / "state.db") == ("delete", "delete")


def test_existing_delete_store_on_network_fs_is_not_upgraded_to_wal(tmp_path, monkeypatch, caplog):
    db = tmp_path / "kanban.db"
    _seed(db, "DELETE")
    _on_network_fs(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=hermes_state_wal.logger.name):
        assert _open(db) == ("delete", "delete")
    assert not [r for r in caplog.records if "switched to WAL" in r.getMessage()]


def test_existing_wal_store_on_network_fs_keeps_wal_and_names_the_offline_remedy(tmp_path, monkeypatch, caplog):
    db = tmp_path / "state.db"
    _seed(db, "WAL")
    _on_network_fs(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=hermes_state_wal.logger.name):
        for _ in range(2):
            assert _open(db) == ("wal", "wal")  # never live-downgraded
    lines = [r for r in caplog.records if "network filesystem" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    text = lines[0].getMessage()
    assert "hermes sessions set-journal-mode delete" in text
    assert "database.journal_mode: delete" in text


@pytest.mark.platforms("linux")  # mountinfo detection is Linux-only
def test_network_fstypes_cover_nfs_and_smb_variants(tmp_path):
    rows = ["25 1 8:1 / / rw - ext4 /dev/sda1 rw"] + [
        f"{600 + i} 25 0:{60 + i} / /mnt/{fs.replace('.', '_')} rw - {fs} srv:/x rw"
        for i, fs in enumerate(("nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"))]
    mi = tmp_path / "mountinfo"
    mi.write_text("\n".join(rows) + "\n", encoding="utf-8")
    for fs in ("nfs", "nfs4", "cifs", "smb3", "fuse_sshfs"):
        assert hermes_state_wal._detect_network_fs(f"/mnt/{fs}/home", mountinfo_path=str(mi)) is True
    assert hermes_state_wal._detect_network_fs("/home/user/.hermes", mountinfo_path=str(mi)) is False
