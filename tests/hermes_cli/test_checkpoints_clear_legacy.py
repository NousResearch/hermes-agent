"""`hermes checkpoints clear-legacy` exit code must reflect whether the archives are gone.

`clear_legacy()` counts failed deletions as ``errors`` and the CLI turns a non-zero count into
exit 2 plus a "Could not delete" line, so a run that left every archive on disk is no longer
indistinguishable from a clean sweep to scripts (GitHub issue #111776). Both tests run the real
manager against a temp checkpoint base; only ``shutil.rmtree`` is faked for the failure case.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _base_with_archives(tmp_path: Path, monkeypatch, names) -> Path:
    import tools.checkpoint_manager as ckpt_mgr

    base = tmp_path / "checkpoints"
    for name in names:
        (base / name).mkdir(parents=True)
        (base / name / "blob").write_bytes(b"x" * 1024)
    monkeypatch.setattr(ckpt_mgr, "CHECKPOINT_BASE", base)
    return base


def test_undeletable_archive_exits_two_and_is_reported(tmp_path, monkeypatch, capsys):
    import tools.checkpoint_manager as ckpt_mgr
    from hermes_cli import checkpoints as checkpoints_cli

    base = _base_with_archives(tmp_path, monkeypatch, ["legacy-20200101-000000", "legacy-20200102-000000"])
    stuck = base / "legacy-20200101-000000"
    real_rmtree = shutil.rmtree

    def _rmtree_read_only(path, *args, **kwargs):
        if Path(path) == stuck:
            raise OSError(13, "Access is denied", str(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(ckpt_mgr.shutil, "rmtree", _rmtree_read_only)

    rc = checkpoints_cli.cmd_clear_legacy(argparse.Namespace(force=True))

    out = capsys.readouterr().out
    assert rc == 2
    assert "Deleted 1 archive(s), reclaimed" in out
    assert "Could not delete 1 archive(s) (see logs)." in out
    assert stuck.exists()


def test_clean_sweep_keeps_exit_zero_and_success_line(tmp_path, monkeypatch, capsys):
    from hermes_cli import checkpoints as checkpoints_cli

    base = _base_with_archives(tmp_path, monkeypatch, ["legacy-20200101-000000"])

    rc = checkpoints_cli.cmd_clear_legacy(argparse.Namespace(force=True))

    out = capsys.readouterr().out
    assert rc == 0
    assert "Deleted 1 archive(s), reclaimed" in out
    assert "Could not delete" not in out
    assert not list(base.glob("legacy-*"))
