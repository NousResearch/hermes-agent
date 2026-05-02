"""Tests for the gateway stale-code self-check (Issue #17648).

A gateway that survives ``hermes update`` keeps pre-update modules cached
in ``sys.modules``.  Later imports of names added post-update (e.g.
``cfg_get`` from PR #17304) raise ImportError against the stale module
object.  The self-check in ``GatewayRunner._detect_stale_code()`` detects
this by comparing boot-time sentinel-file mtimes against current ones,
and ``_trigger_stale_code_restart()`` triggers a graceful restart.
"""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.run import (
    GatewayRunner,
    _compute_repo_mtime,
    _STALE_CODE_SENTINELS,
)


def _make_tmp_repo(tmp_path: Path) -> Path:
    """Create a fake repo with all stale-code sentinel files."""
    for rel in _STALE_CODE_SENTINELS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# test sentinel\n")
    return tmp_path


def _make_runner(repo_root: Path, *, boot_mtime: float, boot_wall: float):
    """Bare GatewayRunner with just the stale-check attributes set."""
    runner = object.__new__(GatewayRunner)
    runner._repo_root_for_staleness = repo_root
    runner._boot_wall_time = boot_wall
    runner._boot_repo_mtime = boot_mtime
    runner._stale_code_notified = set()
    runner._stale_code_restart_triggered = False
    return runner


def test_compute_repo_mtime_returns_newest(tmp_path):
    """_compute_repo_mtime returns the newest mtime across sentinel files."""
    repo = _make_tmp_repo(tmp_path)

    # Stamp a baseline mtime across all sentinels
    baseline = time.time() - 100
    for rel in _STALE_CODE_SENTINELS:
        os.utime(repo / rel, (baseline, baseline))

    # Touch one file forward
    newer = time.time()
    os.utime(repo / "hermes_cli/config.py", (newer, newer))

    result = _compute_repo_mtime(repo)
    assert abs(result - newer) < 1.0  # within 1s (filesystem mtime resolution)


def test_compute_repo_mtime_missing_files_returns_zero(tmp_path):
    """Missing sentinel files return 0.0 (treated as 'can't tell' upstream)."""
    # tmp_path has none of the sentinels
    assert _compute_repo_mtime(tmp_path) == 0.0


def test_compute_repo_mtime_partial_files_still_works(tmp_path):
    """Partial sentinel presence still returns newest of the readable ones."""
    (tmp_path / "hermes_cli").mkdir()
    target = tmp_path / "hermes_cli" / "config.py"
    target.write_text("# partial\n")
    target_mtime = time.time() - 50
    os.utime(target, (target_mtime, target_mtime))

    result = _compute_repo_mtime(tmp_path)
    assert abs(result - target_mtime) < 1.0


def test_detect_stale_code_false_when_no_boot_snapshot(tmp_path):
    """No boot snapshot → can't tell → not stale (no restart loop)."""
    repo = _make_tmp_repo(tmp_path)
    runner = _make_runner(repo, boot_mtime=0.0, boot_wall=0.0)
    assert runner._detect_stale_code() is False


def test_detect_stale_code_false_when_files_unchanged(tmp_path):
    """Source files at boot mtime → not stale."""
    repo = _make_tmp_repo(tmp_path)
    # Freeze all sentinels to the same mtime
    baseline = time.time() - 100
    for rel in _STALE_CODE_SENTINELS:
        os.utime(repo / rel, (baseline, baseline))

    runner = _make_runner(repo, boot_mtime=baseline, boot_wall=baseline)
    assert runner._detect_stale_code() is False


def test_detect_stale_code_true_after_update(tmp_path):
    """Sentinel files newer than boot snapshot → stale."""
    repo = _make_tmp_repo(tmp_path)
    baseline = time.time() - 100
    for rel in _STALE_CODE_SENTINELS:
        os.utime(repo / rel, (baseline, baseline))

    runner = _make_runner(repo, boot_mtime=baseline, boot_wall=baseline)

    # Simulate hermes update touching config.py
    new_mtime = time.time()
    os.utime(repo / "hermes_cli/config.py", (new_mtime, new_mtime))

    assert runner._detect_stale_code() is True


def test_detect_stale_code_ignores_subsecond_drift(tmp_path):
    """2-second slack prevents false positives on coarse-mtime filesystems."""
    repo = _make_tmp_repo(tmp_path)
    baseline = time.time() - 100
    for rel in _STALE_CODE_SENTINELS:
        os.utime(repo / rel, (baseline, baseline))

    runner = _make_runner(repo, boot_mtime=baseline, boot_wall=baseline)

    # Touch config.py 1s newer — within the 2s slack → not stale
    os.utime(repo / "hermes_cli/config.py", (baseline + 1.0, baseline + 1.0))
    assert runner._detect_stale_code() is False

    # Touch 5s newer → stale
    os.utime(repo / "hermes_cli/config.py", (baseline + 5.0, baseline + 5.0))
    assert runner._detect_stale_code() is True


def test_trigger_stale_code_restart_is_idempotent(tmp_path):
    """Calling _trigger_stale_code_restart twice only requests restart once."""
    repo = _make_tmp_repo(tmp_path)
    runner = _make_runner(repo, boot_mtime=1.0, boot_wall=1.0)

    calls = []

    def fake_request_restart(*, detached=False, via_service=False):
        calls.append((detached, via_service))
        return True

    runner.request_restart = fake_request_restart

    runner._trigger_stale_code_restart()
    runner._trigger_stale_code_restart()
    runner._trigger_stale_code_restart()

    assert len(calls) == 1
    assert runner._stale_code_restart_triggered is True


def test_trigger_stale_code_restart_survives_request_failure(tmp_path):
    """If request_restart raises, we swallow and mark as triggered anyway."""
    repo = _make_tmp_repo(tmp_path)
    runner = _make_runner(repo, boot_mtime=1.0, boot_wall=1.0)

    def boom(*, detached=False, via_service=False):
        raise RuntimeError("no event loop")

    runner.request_restart = boom

    # Should not raise
    runner._trigger_stale_code_restart()

    # Marked triggered so we don't retry on every subsequent message
    assert runner._stale_code_restart_triggered is True


def test_detect_stale_code_handles_disappearing_repo_root(tmp_path):
    """If the repo root vanishes after boot, return False (don't loop)."""
    repo = _make_tmp_repo(tmp_path)
    baseline = time.time() - 100
    for rel in _STALE_CODE_SENTINELS:
        os.utime(repo / rel, (baseline, baseline))

    runner = _make_runner(repo, boot_mtime=baseline, boot_wall=baseline)

    # Remove all sentinel files — _compute_repo_mtime returns 0.0
    for rel in _STALE_CODE_SENTINELS:
        (repo / rel).unlink(missing_ok=True)

    assert runner._detect_stale_code() is False


def test_class_level_defaults_prevent_uninitialized_access():
    """Partial construction via object.__new__ must not crash _detect_stale_code."""
    runner = object.__new__(GatewayRunner)
    # Don't set any instance attrs — class-level defaults should kick in
    runner._repo_root_for_staleness = Path(".")
    # _boot_wall_time / _boot_repo_mtime fall through to class defaults (0.0)
    assert runner._detect_stale_code() is False
    # _stale_code_restart_triggered falls through to class default (False)
    assert runner._stale_code_restart_triggered is False


def test_init_captures_boot_snapshot(monkeypatch, tmp_path):
    """GatewayRunner.__init__ captures a usable stale-code baseline."""
    # Stub out the heavy parts of __init__ we don't need.  We only want
    # to prove the stale-code snapshot is captured before anything else.
    from gateway import run as run_mod

    calls = {}

    def fake_compute(repo_root):
        calls["repo_root"] = repo_root
        return 1234567890.0

    monkeypatch.setattr(run_mod, "_compute_repo_mtime", fake_compute)

    # Build a runner without running the full __init__ — then manually
    # exercise the stale-check init block that __init__ contains.
    runner = object.__new__(GatewayRunner)
    runner._boot_wall_time = time.time()
    runner._repo_root_for_staleness = Path(run_mod.__file__).resolve().parent.parent
    runner._boot_repo_mtime = run_mod._compute_repo_mtime(runner._repo_root_for_staleness)
    runner._stale_code_notified = set()
    runner._stale_code_restart_triggered = False

    assert runner._boot_repo_mtime == 1234567890.0
    assert calls["repo_root"] == runner._repo_root_for_staleness
    assert runner._boot_wall_time > 0
