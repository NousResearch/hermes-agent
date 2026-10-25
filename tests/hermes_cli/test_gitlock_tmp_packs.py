"""Aborted-fetch tmp_pack debris sweep (#93732, campaign #91277).

Every git fetch that dies mid-transfer strands a tmp_pack_* file in
.git/objects/pack; git never cleans them. clear_stale_tmp_packs() removes
them with the same age + live-git-process safety contract the lock sweep
uses.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from hermes_cli.gitlock import (
    STALE_TMP_PACK_MIN_AGE_SECONDS,
    clear_stale_tmp_packs,
)


def _mkrepo(tmp_path: Path) -> Path:
    pack = tmp_path / ".git" / "objects" / "pack"
    pack.mkdir(parents=True)
    return tmp_path


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_removes_old_tmp_pack_debris(tmp_path, monkeypatch):
    repo = _mkrepo(tmp_path)
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: False)
    pack = repo / ".git" / "objects" / "pack"

    debris = []
    for name in ("tmp_pack_AbCd12", "tmp_idx_XyZ", "tmp_rev_Q1", "tmp_mtimes_M8"):
        p = pack / name
        p.write_bytes(b"x" * 128)
        _age(p, STALE_TMP_PACK_MIN_AGE_SECONDS + 60)
        debris.append(p)

    removed = clear_stale_tmp_packs(repo)
    assert len(removed) == 4
    for p in debris:
        assert not p.exists()


def test_spares_fresh_debris_and_real_packs(tmp_path, monkeypatch):
    repo = _mkrepo(tmp_path)
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: False)
    pack = repo / ".git" / "objects" / "pack"

    fresh = pack / "tmp_pack_fresh"           # a fetch may be writing this NOW
    fresh.write_bytes(b"y")
    real_pack = pack / "pack-abc123.pack"      # real object data — never touch
    real_pack.write_bytes(b"z" * 64)
    real_idx = pack / "pack-abc123.idx"
    real_idx.write_bytes(b"z")
    _age(real_pack, 10 * 24 * 3600)            # even when ancient
    _age(real_idx, 10 * 24 * 3600)

    removed = clear_stale_tmp_packs(repo)
    assert removed == []
    assert fresh.exists() and real_pack.exists() and real_idx.exists()


def test_skips_sweep_while_git_is_running(tmp_path, monkeypatch):
    repo = _mkrepo(tmp_path)
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: True)
    pack = repo / ".git" / "objects" / "pack"
    p = pack / "tmp_pack_old"
    p.write_bytes(b"x")
    _age(p, STALE_TMP_PACK_MIN_AGE_SECONDS + 60)

    assert clear_stale_tmp_packs(repo) == []
    assert p.exists()


def test_bare_repo_pack_dir_is_swept(tmp_path, monkeypatch):
    """A bare repo (e.g. the checkpoint store) has no .git/ layer — objects/pack hangs directly
    off the repo root, where a gc killed mid-repack strands the same debris (#115410)."""
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: False)
    pack = tmp_path / "objects" / "pack"
    pack.mkdir(parents=True)
    debris = pack / "tmp_pack_killedGc"
    debris.write_bytes(b"x" * 256)
    _age(debris, STALE_TMP_PACK_MIN_AGE_SECONDS + 60)

    removed = clear_stale_tmp_packs(tmp_path)
    assert removed == [str(debris)]
    assert not debris.exists()


def test_no_git_dir_is_a_noop(tmp_path):
    assert clear_stale_tmp_packs(tmp_path) == []




@pytest.mark.platforms("windows")
def test_windows_readonly_debris_is_cleared(tmp_path, monkeypatch):
    """git renames its transfer temps into place read-only, and Windows refuses to unlink a
    read-only file with EACCES — the exact rule that let aborted-fetch debris survive this
    sweep for months (#116384). Runs on a real Windows host: no faked unlink, no faked OS."""
    repo = _mkrepo(tmp_path)
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: False)
    pack = repo / ".git" / "objects" / "pack"

    debris = pack / "tmp_pack_ReadOnly"
    debris.write_bytes(b"x" * 128)
    _age(debris, STALE_TMP_PACK_MIN_AGE_SECONDS + 60)
    os.chmod(debris, 0o444)  # read-only, as git writes its pack temps

    removed = clear_stale_tmp_packs(repo)
    assert removed == [str(debris)]  # write bit cleared first, then unlinked
    assert not debris.exists()


def test_unlink_failure_is_surfaced_at_warning(tmp_path, monkeypatch, caplog):
    """The skip used to be logged at debug, which hid the months-long Windows no-op sweep;
    a cleaner that fails silently is worse than none, so every skip must be visible (#116384)."""
    repo = _mkrepo(tmp_path)
    monkeypatch.setattr("hermes_cli.gitlock._git_proc_running", lambda: False)
    pack = repo / ".git" / "objects" / "pack"
    stuck = pack / "tmp_pack_stuck"
    stuck.write_bytes(b"x")
    _age(stuck, STALE_TMP_PACK_MIN_AGE_SECONDS + 60)

    real_unlink = Path.unlink

    def failing_unlink(self, *a, **k):
        if self.name == "tmp_pack_stuck":
            raise OSError(13, "Permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.gitlock"):
        assert clear_stale_tmp_packs(repo) == []
    assert any(
        r.levelname == "WARNING" and "tmp_pack_stuck" in r.getMessage()
        for r in caplog.records
    )
