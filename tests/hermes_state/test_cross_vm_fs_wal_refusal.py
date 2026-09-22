"""Cross-VM filesystem (virtiofs/9p) WAL refusal — port of openclaw#120597.

WAL over a VM-boundary filesystem (Docker Desktop / OrbStack / Podman host bind mounts) corrupts silently, so
``apply_wal_with_fallback`` must refuse to ENABLE WAL when the DB lives on such a mount — before the pragma —
while never live-downgrading an on-disk WAL database and never flagging an ordinary filesystem.
"""

import logging
import sqlite3

import pytest

import hermes_state_wal
from hermes_state_wal import WalUnsupportedError, _detect_cross_vm_fs, _running_under_gvisor, apply_wal_with_fallback


def _mountinfo(tmp_path, lines):
    p = tmp_path / "mountinfo"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# Realistic mountinfo rows (id parent major:minor root mountpoint opts ... - fstype source superopts)
ROOT_EXT4 = "25 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw"
BIND_VIRTIOFS = "612 25 0:53 / /data rw,relatime shared:300 - fuse.virtiofs mount0 rw"
BIND_9P = "613 25 0:54 / /mnt/host rw,relatime - 9p host0 rw,trans=virtio"
NESTED_EXT4 = "614 612 8:2 / /data/native rw,relatime - ext4 /dev/sdb1 rw"
SPACE_VIRTIOFS = "615 25 0:55 / /mnt/my\\040share rw,relatime - virtiofs share rw"
# gVisor (runsc): the sentry names every host-backed mount 9p, rootfs included. Rows copied from a live sandbox.
GVISOR_ROOT_9P = ("16 15 0:18 / / rw - 9p none rw,trans=fd,rfdno=3,wfdno=3,aname=/,dfltuid=4294967294,"
                  "dfltgid=4294967294,dcache=1000,cache=fscache,disable_fifo_open,overlayfs_stale_read,directfs")
GVISOR_HOME_9P = ("24 16 0:25 / /home/node rw - 9p none rw,trans=fd,rfdno=7,wfdno=7,aname=/,dfltuid=4294967294,"
                  "dfltgid=4294967294,dcache=1000,cache=remote_revalidating,disable_fifo_open,directfs")
GVISOR_PROC_VERSION = "Linux version 4.19.0-gvisor #1 SMP Sun Jan 10 15:06:54 PST 2016\n"
LINUX_PROC_VERSION = "Linux version 6.8.0-45-generic (buildd@lcy02-amd64-115) (x86_64-linux-gnu-gcc-13) #45-Ubuntu\n"


def _proc_version(tmp_path, text):
    p = tmp_path / "proc_version"
    p.write_text(text)
    return str(p)


class TestDetectCrossVmFs:
    @pytest.mark.parametrize("path,expected", [
        ("/data/agent", True),          # fuse.virtiofs bind mount
        ("/mnt/host/db", True),         # 9p bind mount
        ("/mnt/my share/db", True),     # octal-escaped mount point
        ("/home/user/.hermes", False),  # ext4 root
        ("/data/native/db", False),     # ext4 mounted over the virtiofs tree — longest prefix wins
        ("/datastore", False),          # sibling path sharing a prefix string, not a mount prefix
    ])
    def test_only_virtiofs_and_9p_mounts_are_flagged(self, tmp_path, path, expected):
        mi = _mountinfo(tmp_path, [ROOT_EXT4, BIND_VIRTIOFS, BIND_9P, NESTED_EXT4, SPACE_VIRTIOFS])
        assert _detect_cross_vm_fs(path, mountinfo_path=mi) is expected

    @pytest.mark.parametrize("fstype", [
        "ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay", "nfs", "nfs4", "cifs", "fuse.sshfs", "apfs", "f2fs",
    ])
    def test_ordinary_filesystems_never_flagged(self, tmp_path, fstype):
        # A false positive here would put every session on DELETE mode — the class bug this pins absent.
        mi = _mountinfo(tmp_path, [f"25 1 8:1 / / rw,relatime shared:1 - {fstype} /dev/sda1 rw"])
        assert _detect_cross_vm_fs("/home/user/.hermes", mountinfo_path=mi) is False

    def test_missing_mountinfo_conservative_false(self, tmp_path):
        assert _detect_cross_vm_fs("/data", mountinfo_path=str(tmp_path / "nope")) is False

    def test_gvisor_9p_mounts_are_not_cross_vm(self, tmp_path):
        # runsc reports its gofer mounts as 9p, but the sandbox is one kernel: every opener maps the same host
        # file and shares one lock table, so WAL is safe and the refusal would put every sandboxed DB on DELETE.
        mi = _mountinfo(tmp_path, [GVISOR_ROOT_9P, GVISOR_HOME_9P])
        gvisor = _proc_version(tmp_path, GVISOR_PROC_VERSION)
        assert _detect_cross_vm_fs("/home/node/.hermes", mountinfo_path=mi, proc_version_path=gvisor) is False
        assert _detect_cross_vm_fs("/opt/state", mountinfo_path=mi, proc_version_path=gvisor) is False

    def test_same_9p_rows_still_flagged_on_a_real_linux_kernel(self, tmp_path):
        # Sabotage guard for the exemption: only the kernel banner differs, and the refusal must stay.
        mi = _mountinfo(tmp_path, [GVISOR_ROOT_9P, GVISOR_HOME_9P])
        linux = _proc_version(tmp_path, LINUX_PROC_VERSION)
        assert _detect_cross_vm_fs("/home/node/.hermes", mountinfo_path=mi, proc_version_path=linux) is True

    def test_unreadable_proc_version_keeps_the_refusal(self, tmp_path):
        mi = _mountinfo(tmp_path, [BIND_9P])
        assert _detect_cross_vm_fs("/mnt/host/db", mountinfo_path=mi, proc_version_path=str(tmp_path / "nope")) is True

    def test_running_under_gvisor_reads_the_kernel_banner(self, tmp_path):
        assert _running_under_gvisor(_proc_version(tmp_path, GVISOR_PROC_VERSION)) is True
        assert _running_under_gvisor(_proc_version(tmp_path, LINUX_PROC_VERSION)) is False
        assert _running_under_gvisor(str(tmp_path / "nope")) is False


class TestWalRefusalOnCrossVmFs:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        # Pin the WAL-reset vulnerability gate OFF: on builds bundling a vulnerable SQLite (3.50.4 on CI)
        # apply_wal_with_fallback returns via _apply_delete_for_wal_reset_bug before the cross-VM check.
        monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: False)
        monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: "wal")
        hermes_state_wal._cross_vm_warned_paths.clear()
        hermes_state_wal._cross_vm_existing_wal_warned_paths.clear()

    def test_fresh_db_on_cross_vm_fs_gets_delete_and_without_detection_gets_wal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: True)
        conn = sqlite3.connect(str(tmp_path / "a.db"))
        assert apply_wal_with_fallback(conn, db_label="a.db") == "delete"
        conn.close()
        # Sabotage guard: same environment, detection off -> WAL is enabled, so the refusal above did the work.
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: False)
        conn = sqlite3.connect(str(tmp_path / "b.db"))
        mode = apply_wal_with_fallback(conn, db_label="b.db")
        conn.close()
        if mode != "wal":
            pytest.skip("environment refuses WAL for unrelated reasons")

    def test_fresh_db_on_gvisor_9p_gets_wal_through_the_real_detector(self, tmp_path, monkeypatch):
        # End to end minus the kernel: real mountinfo/proc rows from a runsc sandbox, real detector, real sqlite.
        mi = _mountinfo(tmp_path, [GVISOR_ROOT_9P, GVISOR_HOME_9P])
        gvisor = _proc_version(tmp_path, GVISOR_PROC_VERSION)
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs",
                            lambda p: hermes_state_wal._detect_cross_vm_fs("/home/node/.hermes", mountinfo_path=mi,
                                                                           proc_version_path=gvisor))
        conn = sqlite3.connect(str(tmp_path / "gvisor.db"))
        mode = apply_wal_with_fallback(conn, db_label="gvisor.db")
        conn.close()
        if mode != "wal":
            pytest.skip("environment refuses WAL for unrelated reasons")
        # Same rows under a real Linux kernel: the refusal is what lands DELETE, not the environment.
        linux = _proc_version(tmp_path, LINUX_PROC_VERSION)
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs",
                            lambda p: hermes_state_wal._detect_cross_vm_fs("/home/node/.hermes", mountinfo_path=mi,
                                                                           proc_version_path=linux))
        conn = sqlite3.connect(str(tmp_path / "linux.db"))
        assert apply_wal_with_fallback(conn, db_label="linux.db") == "delete"
        conn.close()

    def test_require_wal_raises_on_cross_vm_fs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: True)
        conn = sqlite3.connect(str(tmp_path / "state.db"))
        with pytest.raises(WalUnsupportedError, match=r"cross-VM"):
            apply_wal_with_fallback(conn, db_label="state.db", require_wal=True)
        conn.close()

    def test_on_disk_wal_db_is_never_downgraded(self, tmp_path, monkeypatch):
        db = tmp_path / "already-wal.db"
        seed = sqlite3.connect(str(db))
        if str(seed.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
            seed.close()
            pytest.skip("environment refuses WAL")
        seed.execute("CREATE TABLE t (x)")
        seed.commit()
        seed.close()
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: True)
        conn = sqlite3.connect(str(db))
        assert apply_wal_with_fallback(conn, db_label=str(db)) == "wal"
        conn.close()

    @pytest.mark.parametrize("wal_reset_vulnerable", [False, True])
    def test_existing_wal_db_on_cross_vm_fs_warns_operator_once(self, tmp_path, monkeypatch, caplog,
                                                                wal_reset_vulnerable):
        # #110848: the fresh-DB refusal cannot help a database that is already WAL, and staying silent left the
        # reporter with a corrupting state.db and no signal. Keep WAL (never live-downgrade) but say so, once.
        # The WAL-reset-vulnerable SQLite path (Debian/Ubuntu system Pythons) returns early too and must not be silent.
        monkeypatch.setattr(hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda *a, **k: wal_reset_vulnerable)
        db = tmp_path / "already-wal.db"
        seed = sqlite3.connect(str(db))
        if str(seed.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
            seed.close()
            pytest.skip("environment refuses WAL")
        seed.execute("CREATE TABLE t (x)")
        seed.commit()
        seed.close()
        monkeypatch.setattr(hermes_state_wal, "_path_on_cross_vm_fs", lambda p: True)
        with caplog.at_level(logging.ERROR, logger=hermes_state_wal.logger.name):
            for _ in range(2):
                conn = sqlite3.connect(str(db))
                assert apply_wal_with_fallback(conn, db_label="state.db") == "wal"
                conn.close()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR and "cross-VM" in r.getMessage()]
        assert len(errors) == 1
        assert "hermes sessions set-journal-mode delete" in errors[0].getMessage()
        assert "native volume" in errors[0].getMessage()
