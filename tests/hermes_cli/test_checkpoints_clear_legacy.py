"""Tests for `hermes checkpoints clear-legacy` exit codes.

`clear_legacy()` counts failed deletions as ``errors``; the CLI must not
report success (exit 0) for a run where every archive may still be on disk.
Mirrors the decline/accept flow of the prune tests: store status drives the
preview, `--force` bypasses the prompt, and a patched `clear_legacy` result
drives the final report.
"""

from __future__ import annotations

import argparse

import pytest


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {"force": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


_STATUS_WITH_LEGACY = {
    "legacy_archives": [
        {
            "name": "legacy-20200101-000000",
            "path": "/home/user/.hermes/checkpoints/legacy-20200101-000000",
            "size_bytes": 2048,
        },
    ],
}


def _patch(monkeypatch, clear_result: dict):
    import tools.checkpoint_manager as ckpt_mgr
    import hermes_cli.checkpoints as checkpoints_cli

    monkeypatch.setattr(
        ckpt_mgr, "store_status", lambda *a, **k: dict(_STATUS_WITH_LEGACY)
    )
    monkeypatch.setattr(checkpoints_cli, "_confirmed", lambda args, prompt: True)
    # cmd_clear_legacy imports clear_legacy lazily from the manager module,
    # so patch it where it lives, not where it is imported from.
    monkeypatch.setattr(ckpt_mgr, "clear_legacy", lambda *a, **k: dict(clear_result))


def test_no_legacy_archives_is_success(monkeypatch, capsys):
    import tools.checkpoint_manager as ckpt_mgr
    import hermes_cli.checkpoints as checkpoints_cli

    monkeypatch.setattr(
        ckpt_mgr, "store_status", lambda *a, **k: {"legacy_archives": []}
    )

    rc = checkpoints_cli.cmd_clear_legacy(_ns())

    assert rc == 0
    assert "No legacy archives to clear." in capsys.readouterr().out


def test_clean_sweep_exits_zero(monkeypatch, capsys):
    _patch(monkeypatch, {"bytes_freed": 2048, "deleted": 1, "errors": 0})

    import hermes_cli.checkpoints as checkpoints_cli

    rc = checkpoints_cli.cmd_clear_legacy(_ns(force=True))

    assert rc == 0
    out = capsys.readouterr().out
    assert "Deleted 1 archive(s), reclaimed" in out
    assert "Could not delete" not in out


def test_failed_deletions_exit_two(monkeypatch, capsys):
    _patch(monkeypatch, {"bytes_freed": 0, "deleted": 0, "errors": 1})

    import hermes_cli.checkpoints as checkpoints_cli

    rc = checkpoints_cli.cmd_clear_legacy(_ns(force=True))

    assert rc == 2
    out = capsys.readouterr().out
    assert "Deleted 0 archive(s), reclaimed" in out
    assert "Could not delete 1 archive(s) (see logs)." in out


def test_partial_failure_still_reports_successes(monkeypatch, capsys):
    _patch(monkeypatch, {"bytes_freed": 2048, "deleted": 1, "errors": 1})

    import hermes_cli.checkpoints as checkpoints_cli

    rc = checkpoints_cli.cmd_clear_legacy(_ns(force=True))

    assert rc == 2
    out = capsys.readouterr().out
    assert "Deleted 1 archive(s), reclaimed" in out
    assert "Could not delete 1 archive(s) (see logs)." in out


def test_manager_counts_undeletable_archive(tmp_path, monkeypatch):
    """The counting path is real: a legacy dir that cannot be removed must land in ``errors``."""
    import tools.checkpoint_manager as ckpt_mgr

    base = tmp_path / "checkpoints"
    legacy = base / "legacy-20200101-000000"
    legacy.mkdir(parents=True)
    (legacy / "junk").write_bytes(b"x" * 100)

    def _boom(path, *args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(ckpt_mgr.shutil, "rmtree", _boom)

    result = ckpt_mgr.clear_legacy(checkpoint_base=base)

    assert result["errors"] == 1
    assert result["deleted"] == 0
    assert legacy.exists()
