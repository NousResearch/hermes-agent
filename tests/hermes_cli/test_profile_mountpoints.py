"""``--clone-all`` never recurses into a host directory bind-mounted inside HERMES_HOME (#156026).

Invariant, not snapshot: whatever the mount table says, an entry that is itself a mount point
is skipped by the copy filter while ordinary directories survive. The parser is exercised
against a written mountinfo file (injectable path), and the filter against a real directory
tree plus a snapshot naming one of its dirs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.profile_mountpoints import (
    _non_exportable_entries,
    _read_mount_points,
    is_child_mount_point,
)


def _write_mountinfo(tmp_path: Path, rows: list[str]) -> str:
    path = tmp_path / "mountinfo"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_read_mount_points_parses_proc5_rows(tmp_path):
    # Field layout from proc(5); the parser only needs field 5 (the mount point).
    # Spaces arrive octal-escaped (\040) — a literal space cannot appear in the field.
    path = _write_mountinfo(
        tmp_path,
        [
            "36 35 98:0 /mnt1 /mnt/parent rw,noatime master:1 - ext3 /dev/root rw,errors=continue",
            "40 35 0:29 / /tmp/with\\040space rw,relatime - tmpfs tmpfs rw",
        ],
    )
    assert _read_mount_points(path) == frozenset({"/mnt/parent", "/tmp/with space"})


def test_read_mount_points_unescapes_octal_sequences(tmp_path):
    # \040 is an escaped space in the mount point field.
    path = _write_mountinfo(
        tmp_path,
        [
            "22 1 0:20 /opt/data/my\\040dir /opt/data/my\\040dir rw,relatime - ext4 /dev/sda1 rw"
        ],
    )
    assert _read_mount_points(path) == frozenset({"/opt/data/my dir"})


def test_read_mount_points_skips_malformed_rows_and_tolerates_a_missing_file(tmp_path):
    path = _write_mountinfo(tmp_path, ["garbage", "36 35 98:0 /", "", "not enough"])
    assert _read_mount_points(path) == frozenset()
    assert _read_mount_points(str(tmp_path / "absent")) == frozenset()


def test_is_child_mount_point_trusts_the_snapshot_over_ismount(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    snapshot = frozenset({str(plain)})
    # os.path.ismount(plain) is False — the snapshot is what decides on Linux.
    assert not os.path.ismount(str(plain))
    assert is_child_mount_point(str(plain), snapshot)
    assert not is_child_mount_point(str(tmp_path), snapshot)


def test_is_child_mount_point_falls_back_to_ismount_with_an_empty_snapshot(tmp_path):
    # Non-Linux path: no table, so the stdlib call decides (False for a plain dir).
    plain = tmp_path / "plain"
    plain.mkdir()
    assert not is_child_mount_point(str(plain), frozenset())


def test_non_exportable_entries_skips_a_bind_mounted_directory(tmp_path):
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    (host_dir / "huge.bin").write_bytes(b"x" * 10)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "xchg").mkdir()  # stands in for the bind-mounted host dir
    (profile / "skills").mkdir()  # ordinary dir
    (profile / "config.yaml").write_text("model: {}", encoding="utf-8")

    snapshot = frozenset({str(profile / "xchg")})
    ignored = _non_exportable_entries(
        str(profile), sorted(os.listdir(profile)), snapshot
    )

    assert "xchg" in ignored
    assert not {"skills", "config.yaml"} & ignored


def test_non_exportable_entries_keeps_plain_dirs_without_a_table(tmp_path):
    # No snapshot passed and the dirs are not mounts: nothing is dropped, so the
    # non-Linux default behaviour is unchanged.
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "skills").mkdir()
    ignored = _non_exportable_entries(str(profile), ["skills"])
    assert "skills" not in ignored


def test_non_exportable_entries_still_drops_special_files(tmp_path):
    # The pre-existing contract (sockets/FIFOs abort copytree) must survive the move.
    profile = tmp_path / "profile"
    profile.mkdir()
    fifo = profile / "live_pipe"
    os.mkfifo(fifo)
    ignored = _non_exportable_entries(str(profile), ["live_pipe"], frozenset())
    assert "live_pipe" in ignored
