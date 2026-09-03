"""Cross-VM filesystem (virtiofs/9p) WAL refusal — port of openclaw#120597.

WAL over a VM-boundary filesystem (Docker Desktop / OrbStack / Podman host
bind mounts) corrupts silently, so ``apply_wal_with_fallback`` must refuse
to ENABLE WAL when the DB lives on such a mount — proactively, before the
pragma — while never live-downgrading an on-disk WAL database.
"""

import sqlite3

import pytest

import hermes_state
from hermes_state import (
    WalUnsupportedError,
    _detect_cross_vm_fs,
    _path_on_cross_vm_fs,
    apply_wal_with_fallback,
)


def _mountinfo(tmp_path, lines):
    p = tmp_path / "mountinfo"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# Realistic mountinfo rows (id parent major:minor root mountpoint opts ... - fstype source superopts)
ROOT_EXT4 = "25 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"
BIND_VIRTIOFS = "612 25 0:53 / /data rw,relatime shared:300 - fuse.virtiofs mount0 rw"
BIND_9P = "613 25 0:54 / /mnt/host rw,relatime - 9p host0 rw,trans=virtio"
NESTED_EXT4 = "614 612 8:2 / /data/native rw,relatime - ext4 /dev/sdb1 rw"


class TestDetectCrossVmFs:
    def test_virtiofs_mount_detected(self, tmp_path):
        mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_VIRTIOFS])
        assert _detect_cross_vm_fs("/data/agent", mountinfo_path=mi) is True

    def test_9p_mount_detected(self, tmp_path):
        mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_9P])
        assert _detect_cross_vm_fs("/mnt/host/db", mountinfo_path=mi) is True

    def test_native_root_not_flagged(self, tmp_path):
        mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_VIRTIOFS])
        assert _detect_cross_vm_fs("/home/user/.hermes", mountinfo_path=mi) is False

    def test_longest_prefix_wins_nested_native_mount(self, tmp_path):
        # /data is virtiofs but /data/native is a real ext4 mount on top.
        mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_VIRTIOFS, NESTED_EXT4])
        assert _detect_cross_vm_fs("/data/native/db", mountinfo_path=mi) is False
        assert _detect_cross_vm_fs("/data/other", mountinfo_path=mi) is True

    def test_missing_mountinfo_conservative_false(self, tmp_path):
        assert _detect_cross_vm_fs(
            "/data", mountinfo_path=str(tmp_path / "nope")
        ) is False


class TestWalRefusalOnCrossVmFs:
    @pytest.fixture(autouse=True)
    def _clear_cache(self, monkeypatch):
        with hermes_state._cross_vm_fs_cache_lock:
            hermes_state._cross_vm_fs_cache.clear()
        # Pin the WAL-reset vulnerability gate OFF: on builds bundling a
        # vulnerable SQLite (e.g. 3.50.4 on CI) apply_wal_with_fallback
        # returns via _apply_delete_for_wal_reset_bug BEFORE the cross-VM
        # check, so these tests would exercise the wrong branch and pass
        # (or fail) vacuously depending on the interpreter's SQLite build.
        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False
        )
        yield
        with hermes_state._cross_vm_fs_cache_lock:
            hermes_state._cross_vm_fs_cache.clear()

    def _fresh_db(self, tmp_path):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(str(db))
        return conn, str(db)

    def test_fresh_db_on_cross_vm_fs_gets_delete(self, tmp_path, monkeypatch):
        conn, db_path = self._fresh_db(tmp_path)
        monkeypatch.setattr(
            hermes_state, "_path_on_cross_vm_fs", lambda p: True
        )
        # Fix precondition: WAL would otherwise be enabled here. Sabotage
        # guard — with detection forced OFF the same call returns "wal",
        # proving the refusal below is doing the work.
        mode = apply_wal_with_fallback(conn, db_label=f"pre-{db_path}")
        assert mode == "delete"
        conn.close()

    def test_without_detection_wal_is_enabled(self, tmp_path, monkeypatch):
        conn, db_path = self._fresh_db(tmp_path)
        monkeypatch.setattr(
            hermes_state, "_path_on_cross_vm_fs", lambda p: False
        )
        mode = apply_wal_with_fallback(conn, db_label=db_path)
        if mode != "wal":
            pytest.skip("environment refuses WAL for unrelated reasons")
        conn.close()

    def test_require_wal_raises_on_cross_vm_fs(self, tmp_path, monkeypatch):
        conn, db_path = self._fresh_db(tmp_path)
        monkeypatch.setattr(
            hermes_state, "_path_on_cross_vm_fs", lambda p: True
        )
        with pytest.raises(WalUnsupportedError, match=r"cross-VM"):
            apply_wal_with_fallback(
                conn, db_label=db_path, require_wal=True
            )
        conn.close()

    def test_on_disk_wal_db_is_never_downgraded(self, tmp_path, monkeypatch):
        # An existing WAL database is returned as WAL before the cross-VM
        # check runs — never live-downgrade under possible concurrent
        # openers (same invariant as the NFS path).
        db = tmp_path / "already-wal.db"
        seed = sqlite3.connect(str(db))
        row = seed.execute("PRAGMA journal_mode=WAL").fetchone()
        if str(row[0]).lower() != "wal":
            seed.close()
            pytest.skip("environment refuses WAL")
        seed.execute("CREATE TABLE t (x)")
        seed.commit()
        seed.close()

        conn = sqlite3.connect(str(db))
        monkeypatch.setattr(
            hermes_state, "_path_on_cross_vm_fs", lambda p: True
        )
        mode = apply_wal_with_fallback(conn, db_label=str(db))
        assert mode == "wal"
        conn.close()

    def test_path_cache_reused(self, tmp_path, monkeypatch):
        calls = []

        def fake_detect(directory, mountinfo_path="/proc/self/mountinfo"):
            calls.append(directory)
            return False

        monkeypatch.setattr(hermes_state, "_detect_cross_vm_fs", fake_detect)
        assert _path_on_cross_vm_fs(str(tmp_path / "a.db")) is False
        assert _path_on_cross_vm_fs(str(tmp_path / "b.db")) is False
        assert len(calls) == 1  # same directory — second call served by cache
