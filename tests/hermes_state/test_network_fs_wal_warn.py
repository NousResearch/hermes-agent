"""Network filesystem (nfs/cifs/smb/sshfs) existing-WAL warning, the #110848 remainder.

Unlike the cross-VM class these mounts never refuse WAL for fresh databases (single-writer NFS homes are a
legitimate shape), so the contract is narrower: an on-disk WAL database on one is warned about, once, at
WARNING level, and a fresh database keeps upstream behaviour (WAL enabled, no warning).
"""

import logging
import sqlite3

import pytest

import hermes_state_wal
from hermes_state_wal import _detect_network_fs, apply_wal_with_fallback


def _mountinfo(tmp_path, lines):
    p = tmp_path / "mountinfo"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# Realistic mountinfo rows (id parent major:minor root mountpoint opts ... - fstype source superopts)
ROOT_EXT4 = "25 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"
BIND_NFS = "620 25 0:60 / /srv/nas rw,relatime - nfs nas:/export/hermes rw"
BIND_CIFS = "621 25 0:61 / /mnt/share rw,relatime - cifs //fileserver/hermes rw"
BIND_FUSE_SSHFS = "622 25 0:62 / /home/remote rw,relatime - fuse.sshfs host:/home/user rw"
BIND_VIRTIOFS = "623 25 0:63 / /data rw,relatime - virtiofs mount0 rw"


class TestDetectNetworkFs:
    @pytest.mark.parametrize("path,expected", [
        ("/srv/nas/agent", True),         # nfs bind mount
        ("/mnt/share/db", True),          # cifs bind mount
        ("/home/remote/.hermes", True),   # fuse.sshfs bind mount
        ("/home/local/.hermes", False),   # ext4 root
        ("/data/agent", False),           # virtiofs belongs to the cross-VM class, not this one
    ])
    def test_only_network_mounts_are_flagged(self, tmp_path, path, expected):
        mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_NFS, BIND_CIFS, BIND_FUSE_SSHFS, BIND_VIRTIOFS])
        assert _detect_network_fs(path, mountinfo_path=mi) is expected

    @pytest.mark.parametrize("fstype", [
        "ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay", "apfs", "f2fs", "fuse.ntfs3g", "fuse.rofs",
    ])
    def test_local_and_generic_fuse_filesystems_never_flagged(self, tmp_path, fstype):
        # Generic FUSE can be local; flagging it would warn every ntfs3g home (#110848 scope call).
        mi = _mountinfo(tmp_path, [f"25 1 8:1 / / rw,relatime shared:1 - {fstype} /dev/sda1 rw"])
        assert _detect_network_fs("/home/user/.hermes", mountinfo_path=mi) is False

    def test_missing_mountinfo_conservative_false(self, tmp_path):
        assert _detect_network_fs("/srv/nas", mountinfo_path=str(tmp_path / "nope")) is False


class TestNetworkFsExistingWalWarn:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        # Same isolation as the cross-VM suite: pin the WAL-reset gate and the config reader, keep the
        # cross-VM class out of the picture, and clear the network once-per-process dedupe set.
        monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
        monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: False)
        hermes_state_wal._network_fs_warned_paths.clear()

    @pytest.mark.parametrize("wal_reset_vulnerable", [False, True])
    def test_existing_wal_db_on_network_fs_warns_once_at_warning(
        self, tmp_path, monkeypatch, caplog, wal_reset_vulnerable,
    ):
        # Both already-WAL early-return paths (regular and WAL-reset-vulnerable) must warn, never at ERROR:
        # single-writer NFS is a working shape, so this is a risk signal, not an active-corruption alarm.
        monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: wal_reset_vulnerable)
        db = tmp_path / "already-wal.db"
        seed = sqlite3.connect(str(db))
        if str(seed.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
            seed.close()
            pytest.skip("environment refuses WAL")
        seed.execute("CREATE TABLE t (x)")
        seed.commit()
        seed.close()
        monkeypatch.setattr(hermes_state_wal, "_path_on_network_fs", lambda p: True)
        with caplog.at_level(logging.WARNING, logger=hermes_state_wal.logger.name):
            for _ in range(2):
                conn = sqlite3.connect(str(db))
                assert apply_wal_with_fallback(conn, db_label="state.db") == "wal"
                conn.close()
        warnings = [r for r in caplog.records if "network filesystem" in r.getMessage()]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_fresh_db_on_network_fs_keeps_wal_and_stays_silent(self, tmp_path, monkeypatch, caplog):
        # Scope boundary: the fresh-DB refusal is reserved for the cross-VM class, so a brand-new database on
        # NFS keeps WAL (single-writer NFS is legitimate) and gets no warning.
        monkeypatch.setattr(hermes_state_wal, "_path_on_network_fs", lambda p: True)
        conn = sqlite3.connect(str(tmp_path / "fresh.db"))
        mode = apply_wal_with_fallback(conn, db_label="fresh.db")
        conn.close()
        if mode != "wal":
            pytest.skip("environment refuses WAL for unrelated reasons")
        assert not [r for r in caplog.records if "network filesystem" in r.getMessage()]
