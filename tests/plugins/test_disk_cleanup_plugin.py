"""Tests for the disk-cleanup plugin.

Covers the bundled plugin at ``plugins/disk-cleanup/``:

  * ``disk_cleanup`` library: track / forget / dry_run / quick / status,
    ``is_safe_path`` and ``guess_category`` filtering.
  * Plugin ``__init__``: ``post_tool_call`` hook auto-tracks files created
    by ``write_file`` / ``terminal``; ``on_session_end`` hook runs quick
    cleanup when anything was tracked during the turn.
  * Slash command handler: status / dry-run / quick / track / forget /
    unknown subcommand behaviours.
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
"""

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test.

    The global hermetic fixture already redirects HERMES_HOME to a tempdir,
    but we want the plugin to work with a predictable subpath. We reset
    HERMES_HOME here for clarity.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _load_lib():
    """Import the plugin's library module directly from the repo path."""
    repo_root = Path(__file__).resolve().parents[2]
    lib_path = repo_root / "plugins" / "disk-cleanup" / "disk_cleanup.py"
    spec = importlib.util.spec_from_file_location(
        "disk_cleanup_under_test", lib_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_plugin_init():
    """Import the plugin's __init__.py (which depends on the library)."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "disk-cleanup"
    # Use the PluginManager's module naming convention so relative imports work.
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.disk_cleanup",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    # Ensure parent namespace package exists for relative `. import disk_cleanup`
    import types
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.disk_cleanup"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.disk_cleanup"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_git_repo(path: Path) -> Path:
    """Create a real Git worktree for cleanup-boundary tests."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


def _make_linked_worktree(repo: Path, linked: Path) -> Path:
    """Create a real linked Git worktree whose ``.git`` marker is a file."""
    _make_git_repo(repo)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Disk Cleanup Test"],
        check=True,
    )
    (repo / "README.md").write_text("fixture\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", "README.md"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", "-b", "linked", str(linked)],
        check=True,
    )
    assert (linked / ".git").is_file()
    return linked


# ---------------------------------------------------------------------------
# Library tests
# ---------------------------------------------------------------------------

class TestIsSafePath:
    def test_accepts_path_under_hermes_home(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "subdir" / "file.txt"
        p.parent.mkdir()
        p.write_text("x")
        assert dg.is_safe_path(p) is True

    def test_rejects_outside_hermes_home(self, _isolate_env):
        dg = _load_lib()
        assert dg.is_safe_path(Path("/etc/passwd")) is False

    def test_accepts_tmp_hermes_prefix(self, _isolate_env, tmp_path):
        dg = _load_lib()
        assert dg.is_safe_path(Path("/tmp/hermes-abc/x.log")) is True

    def test_rejects_plain_tmp(self, _isolate_env):
        dg = _load_lib()
        assert dg.is_safe_path(Path("/tmp/other.log")) is False

    def test_rejects_tmp_hermes_symlink_escape_at_cleanup_boundary(
        self, _isolate_env
    ):
        dg = _load_lib()
        tmp_root = Path(tempfile.mkdtemp(prefix="hermes-scope-", dir="/tmp"))
        outside = _isolate_env.parent / "outside"
        outside.mkdir()
        victim = outside / "test_victim.py"
        victim.write_text("source\n")
        (tmp_root / "escape").symlink_to(outside, target_is_directory=True)
        escaped = tmp_root / "escape" / victim.name
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(escaped),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": victim.stat().st_size,
        }]))

        try:
            assert dg.is_safe_path(escaped) is False
            summary = dg.quick()
            assert summary["deleted"] == 0
            assert victim.exists()
            assert json.loads(tracked_file.read_text()) == []
        finally:
            shutil.rmtree(tmp_root)

    def test_rejects_windows_mount(self, _isolate_env):
        dg = _load_lib()
        assert dg.is_safe_path(Path("/mnt/c/Users/x/test.txt")) is False


class TestGuessCategory:
    def test_test_prefix(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "test_foo.py"
        p.write_text("x")
        assert dg.guess_category(p) == "test"

    def test_tmp_prefix(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "tmp_foo.log"
        p.write_text("x")
        assert dg.guess_category(p) == "test"

    def test_dot_test_suffix(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "mything.test.js"
        p.write_text("x")
        assert dg.guess_category(p) == "test"

    def test_skips_protected_top_level(self, _isolate_env):
        dg = _load_lib()
        logs_dir = _isolate_env / "logs"
        logs_dir.mkdir()
        p = logs_dir / "test_log.txt"
        p.write_text("x")
        # Even though it matches test_* pattern, logs/ is excluded.
        assert dg.guess_category(p) is None

    def test_cron_subtree_categorised(self, _isolate_env):
        dg = _load_lib()
        # Only files under ``cron/output/`` are disposable run artifacts.
        output_dir = _isolate_env / "cron" / "output" / "job_123"
        output_dir.mkdir(parents=True)
        p = output_dir / "run.md"
        p.write_text("x")
        assert dg.guess_category(p) == "cron-output"

    def test_cron_output_root_not_tracked(self, _isolate_env):
        """The cron/output root is durable container state, not an artifact."""
        dg = _load_lib()
        output_root = _isolate_env / "cron" / "output"
        output_root.mkdir(parents=True)
        assert dg.guess_category(output_root) is None

    def test_cron_jobs_json_not_tracked(self, _isolate_env):
        """Regression for #32164: the cron registry must never be tracked."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        p = cron_dir / "jobs.json"
        p.write_text("[]")
        assert dg.guess_category(p) is None

    def test_cron_tick_lock_not_tracked(self, _isolate_env):
        """Regression for #32164: cron tick-lock is control-plane state."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        p = cron_dir / ".tick.lock"
        p.write_text("")
        assert dg.guess_category(p) is None

    def test_cronjobs_top_level_not_tracked(self, _isolate_env):
        """The legacy ``cronjobs`` alias is also control-plane at the top."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cronjobs"
        cron_dir.mkdir()
        p = cron_dir / "jobs.json"
        p.write_text("[]")
        assert dg.guess_category(p) is None

    def test_ordinary_file_returns_none(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "notes.md"
        p.write_text("x")
        assert dg.guess_category(p) is None

    def test_git_worktree_test_file_returns_none(self, _isolate_env):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env / "scratch" / "repo")
        p = repo / "tests" / "test_feature.py"
        p.parent.mkdir()
        p.write_text("def test_feature(): pass\n")
        assert dg.guess_category(p) is None

    def test_linked_git_worktree_test_file_returns_none(self, _isolate_env):
        dg = _load_lib()
        linked = _make_linked_worktree(
            _isolate_env / "scratch" / "primary",
            _isolate_env / "scratch" / "linked",
        )
        p = linked / "test_feature.py"
        p.write_text("def test_feature(): pass\n")

        assert dg.guess_category(p) is None

    def test_ignored_test_file_in_hermes_home_repo_remains_disposable(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        (repo / ".gitignore").write_text("scratch/\n")
        p = repo / "scratch" / "test_ephemeral.py"
        p.parent.mkdir()
        p.write_text("x")
        assert dg.guess_category(p) == "test"

    def test_cron_output_in_hermes_home_repo_remains_disposable(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        p = repo / "cron" / "output" / "job-1" / "run.md"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        assert dg.guess_category(p) == "cron-output"

    def test_tracked_cron_output_in_hermes_home_repo_is_protected(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        p = repo / "cron" / "output" / "job-1" / "run.md"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        subprocess.run(
            ["git", "-C", str(repo), "add", str(p.relative_to(repo))],
            check=True,
        )
        assert dg.guess_category(p) is None


class TestStaleCronEntryMigration:
    """Regression tests for #37721 — stale cron-output entries in tracked.json."""

    def test_quick_skips_stale_cron_output_for_jobs_json(self, _isolate_env):
        """A stale tracked.json entry with category="cron-output" for
        cron/jobs.json must NOT be deleted by quick().

        This is the exact scenario from #37721: an old tracked.json has
        {"path": ".../cron/jobs.json", "category": "cron-output"} which
        would pass the delete filter but must be skipped because
        guess_category() now returns None for non-output cron paths.
        """
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        jobs_json = cron_dir / "jobs.json"
        jobs_json.write_text('{"jobs": []}')

        # Simulate a stale tracked.json entry from before #34840 by
        # directly writing the tracked file (track() would reject it).
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(jobs_json),
            "category": "cron-output",
            "timestamp": "2025-01-01T00:00:00+00:00",  # very old
            "size": 123,
        }]))

        summary = dg.quick()
        assert summary["deleted"] == 0, "cron/jobs.json must not be deleted"
        assert jobs_json.exists(), "jobs.json must still exist"
        # The stale entry should have been dropped from tracking.
        remaining = json.loads(tracked_file.read_text())
        assert len(remaining) == 0

    def test_quick_skips_stale_cron_output_for_cron_dir(self, _isolate_env):
        """Stale entry for the cron/ directory itself must not be deleted."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        output_dir = cron_dir / "output"
        output_dir.mkdir()
        (output_dir / "run.md").write_text("x")

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(cron_dir),
            "category": "cron-output",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.quick()
        assert summary["deleted"] == 0, "cron/ dir must not be deleted"
        assert cron_dir.exists()

    def test_quick_skips_stale_cron_output_for_output_root(self, _isolate_env):
        """Stale entry for cron/output itself must not delete all job output."""
        dg = _load_lib()
        output_root = _isolate_env / "cron" / "output"
        job_dir = output_root / "job_1"
        job_dir.mkdir(parents=True)
        run_md = job_dir / "run.md"
        run_md.write_text("x")

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(output_root),
            "category": "cron-output",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.quick()
        assert summary["deleted"] == 0, "cron/output root must not be deleted"
        assert output_root.exists()
        assert run_md.exists()

    def test_quick_skips_protected_cron_paths_defense_in_depth(self, _isolate_env):
        """Defense-in-depth: even if guess_category returned cron-output
        (hypothetically), protected cron paths are never deleted."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        tick_lock = cron_dir / ".tick.lock"
        tick_lock.write_text("")

        # Manually inject a stale entry with "test" category (would normally
        # be auto-deleted) — the protected path guard must still block it.
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(tick_lock),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.quick()
        assert summary["deleted"] == 0, ".tick.lock must not be deleted"
        assert tick_lock.exists()

    def test_dry_run_omits_stale_cron_output(self, _isolate_env):
        """dry_run() should also skip stale cron-output entries."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        jobs_json = cron_dir / "jobs.json"
        jobs_json.write_text("[]")

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(jobs_json),
            "category": "cron-output",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 123,
        }]))

        auto, prompt = dg.dry_run()
        assert len(auto) == 0, "stale cron-output for jobs.json must not appear"
        assert len(prompt) == 0

    def test_legitimate_cron_output_still_deleted(self, _isolate_env):
        """A valid cron-output entry under cron/output/ must still be deleted."""
        dg = _load_lib()
        output_dir = _isolate_env / "cron" / "output" / "job_1"
        output_dir.mkdir(parents=True)
        run_md = output_dir / "run.md"
        run_md.write_text("x")

        # Old enough to be deleted (>14 days)
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(run_md),
            "category": "cron-output",
            "timestamp": old_ts,
            "size": 10,
        }]))

        summary = dg.quick()
        assert summary["deleted"] == 1, "valid old cron-output should be deleted"
        assert not run_md.exists()


class TestTrackForgetQuick:
    def test_track_then_quick_deletes_test(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "test_a.py"
        p.write_text("x")
        assert dg.track(str(p), "test", silent=True) is True
        assert dg.load_tracked()[0]["explicit"] is True
        summary = dg.quick()
        assert summary["deleted"] == 1
        assert not p.exists()

    def test_track_dedup(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "test_a.py"
        p.write_text("x")
        assert dg.track(str(p), "test", silent=True) is True
        # Second call returns False (already tracked)
        assert dg.track(str(p), "test", silent=True) is False

    def test_track_rejects_outside_home(self, _isolate_env):
        dg = _load_lib()
        # /etc/hostname exists on most Linux boxes; fall back if not.
        outside = "/etc/hostname" if Path("/etc/hostname").exists() else "/etc/passwd"
        assert dg.track(outside, "test", silent=True) is False

    def test_manual_track_rejects_durable_state_path(self, _isolate_env):
        dg = _load_lib()
        protected = _isolate_env / "logs" / "test_agent.log"
        protected.parent.mkdir()
        protected.write_text("keep\n")

        assert dg.track(str(protected), "test", silent=True) is False
        assert dg.load_tracked() == []

    def test_manual_track_can_explicitly_delete_file_inside_git_worktree(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env / "worktrees" / "feature")
        p = repo / "test_feature.py"
        p.write_text("x")

        assert dg.track(str(p), "test", silent=True) is True
        assert dg.load_tracked()[0]["explicit"] is True
        summary = dg.quick()

        assert summary["deleted"] == 1
        assert not p.exists()

    def test_manual_track_upgrades_legacy_entry_to_explicit(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env / "worktrees" / "feature")
        p = repo / "test_feature.py"
        p.write_text("x")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "temp",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        assert dg.track(str(p), "test", silent=True) is True
        upgraded = dg.load_tracked()[0]
        assert upgraded["explicit"] is True
        assert upgraded["category"] == "test"
        assert upgraded["size"] == 1
        assert upgraded["timestamp"] != "2025-01-01T00:00:00+00:00"

    def test_track_skips_missing(self, _isolate_env):
        dg = _load_lib()
        assert dg.track(str(_isolate_env / "nope.txt"), "test", silent=True) is False

    def test_forget_removes_entry(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "keep.tmp"
        p.write_text("x")
        dg.track(str(p), "temp", silent=True)
        assert dg.forget(str(p)) == 1
        assert p.exists()  # forget does NOT delete the file

    def test_quick_preserves_unexpired_temp(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "fresh.tmp"
        p.write_text("x")
        dg.track(str(p), "temp", silent=True)
        summary = dg.quick()
        assert summary["deleted"] == 0
        assert p.exists()

    def test_quick_preserves_protected_top_level_dirs(self, _isolate_env):
        dg = _load_lib()
        for d in ("logs", "memories", "sessions", "cron", "cache"):
            (_isolate_env / d).mkdir()
        dg.quick()
        for d in ("logs", "memories", "sessions", "cron", "cache"):
            assert (_isolate_env / d).exists(), f"{d}/ should be preserved"

    def test_quick_drops_stale_durable_entry_without_deleting(self, _isolate_env):
        dg = _load_lib()
        protected = _isolate_env / "logs" / "test_agent.log"
        protected.parent.mkdir()
        protected.write_text("keep\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(protected),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": protected.stat().st_size,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert protected.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_delete_candidate_refuses_hermes_home_root_even_explicit(
        self, _isolate_env
    ):
        dg = _load_lib()
        victim = _isolate_env / "important.txt"
        victim.write_text("keep\n")

        removed, _reason = dg._delete_candidate(_isolate_env, explicit=True)

        assert removed is False
        assert victim.exists()

    def test_quick_preserves_mountpoint_candidate(self, _isolate_env, monkeypatch):
        dg = _load_lib()
        candidate = _isolate_env / "scratch" / "test_mount"
        candidate.mkdir(parents=True)
        source = candidate / "source.py"
        source.write_text("keep\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(candidate),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))
        real_ismount = dg.os.path.ismount
        monkeypatch.setattr(
            dg.os.path,
            "ismount",
            lambda value: Path(value) == candidate or real_ismount(value),
        )

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert source.exists()

    def test_quick_preserves_candidate_on_nested_device(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        candidate = _isolate_env / "scratch" / "test_mounted.py"
        candidate.parent.mkdir()
        candidate.write_text("keep\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(candidate),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": candidate.stat().st_size,
        }]))
        monkeypatch.setattr(dg, "_is_same_cleanup_device", lambda _path: False)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert candidate.exists()

    def test_quick_does_not_descend_into_protected_top_level_dirs(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        protected_empty = (
            _isolate_env / "hermes-agent" / "node_modules" / "pkg" / "empty"
        )
        protected_empty.mkdir(parents=True)

        original_iterdir = Path.iterdir

        def guarded_iterdir(path):
            if path == _isolate_env / "hermes-agent":
                raise AssertionError("quick() descended into protected hermes-agent/")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

        dg.quick()

        assert protected_empty.exists()

    def test_quick_removes_empty_dirs_in_managed_subtrees(self, _isolate_env):
        dg = _load_lib()
        managed_empty = _isolate_env / "scratch" / "nested" / "empty"
        managed_empty.mkdir(parents=True)

        dg.quick()

        assert not (_isolate_env / "scratch").exists()

    def test_empty_dir_rmdir_is_bound_to_validated_parent(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        parent = _isolate_env / "scratch" / "race"
        empty = parent / "empty"
        empty.mkdir(parents=True)
        moved_parent = parent.with_name("race-moved")
        outside = _isolate_env.parent / "outside-empty-race"
        outside_empty = outside / empty.name
        outside_empty.mkdir(parents=True)
        real_rmdir = dg._OS_RMDIR
        raced = False

        def replace_parent_at_rmdir(target, *args, dir_fd=None, **kwargs):
            nonlocal raced
            if not raced and Path(target).name == empty.name:
                parent.rename(moved_parent)
                parent.symlink_to(outside, target_is_directory=True)
                raced = True
            return real_rmdir(target, *args, dir_fd=dir_fd, **kwargs)

        monkeypatch.setattr(dg, "_OS_RMDIR", replace_parent_at_rmdir)

        dg.quick()

        assert raced is True
        assert outside_empty.exists()
        assert not (moved_parent / empty.name).exists()

    def test_quick_preserves_stale_test_entry_inside_git_worktree(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env / "worktrees" / "feature")
        p = repo / "tests" / "test_feature.py"
        p.parent.mkdir()
        p.write_text("x")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_quick_preserves_stale_test_entry_inside_linked_worktree(
        self, _isolate_env
    ):
        dg = _load_lib()
        linked = _make_linked_worktree(
            _isolate_env / "scratch" / "primary",
            _isolate_env / "worktrees" / "linked",
        )
        p = linked / "tests" / "test_feature.py"
        p.parent.mkdir()
        p.write_text("x")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_quick_drops_stale_external_entry_without_deleting_target(
        self, _isolate_env
    ):
        dg = _load_lib()
        outside = _isolate_env.parent / "outside" / "test_external.py"
        outside.parent.mkdir()
        outside.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(outside),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": outside.stat().st_size,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert outside.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_quick_rejects_noncanonical_parent_component_path(
        self, _isolate_env
    ):
        dg = _load_lib()
        bundle = _isolate_env / "scratch" / "bundle"
        child = bundle / "child"
        child.mkdir(parents=True)
        victim = bundle / "important.txt"
        victim.write_text("keep\n")
        dangerous = child / ".."
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(dangerous),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert victim.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_quick_rechecks_git_protection_at_delete_boundary(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        p = _isolate_env / "scratch" / "test_race.py"
        p.parent.mkdir()
        p.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": p.stat().st_size,
        }]))
        probes = 0

        def becomes_git_protected(path):
            nonlocal probes
            if path.resolve() == p.resolve():
                probes += 1
                return probes >= 2
            return False

        monkeypatch.setattr(dg, "_is_git_protected_path", becomes_git_protected)

        summary = dg.quick()

        assert probes >= 2
        assert summary["deleted"] == 0
        assert p.exists()

    def test_quick_unlink_is_bound_to_validated_parent(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        parent = _isolate_env / "scratch" / "race"
        parent.mkdir(parents=True)
        candidate = parent / "test_victim.py"
        candidate.write_text("inside\n")
        moved_parent = parent.with_name("race-moved")
        outside = _isolate_env.parent / "outside-race"
        outside.mkdir()
        outside_victim = outside / candidate.name
        outside_victim.write_text("outside\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(candidate),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": candidate.stat().st_size,
        }]))
        real_unlink = dg._OS_UNLINK
        raced = False

        def replace_parent_at_unlink(target, *args, dir_fd=None, **kwargs):
            nonlocal raced
            target_name = Path(target).name
            if not raced and target_name == candidate.name:
                parent.rename(moved_parent)
                parent.symlink_to(outside, target_is_directory=True)
                raced = True
            return real_unlink(target, *args, dir_fd=dir_fd, **kwargs)

        monkeypatch.setattr(dg, "_OS_UNLINK", replace_parent_at_unlink)

        summary = dg.quick()

        assert raced is True
        assert summary["deleted"] == 1
        assert outside_victim.exists()
        assert not (moved_parent / candidate.name).exists()

    def test_quick_rmtree_is_bound_to_validated_parent(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        parent = _isolate_env / "scratch" / "rmtree-race"
        candidate = parent / "test_directory"
        candidate.mkdir(parents=True)
        (candidate / "inside.py").write_text("inside\n")
        moved_parent = parent.with_name("rmtree-race-moved")
        outside = _isolate_env.parent / "outside-rmtree-race"
        outside_candidate = outside / candidate.name
        outside_candidate.mkdir(parents=True)
        outside_source = outside_candidate / "outside.py"
        outside_source.write_text("outside\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(candidate),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))
        real_rmtree = dg._SHUTIL_RMTREE
        raced = False

        def replace_parent_at_rmtree(target, *args, dir_fd=None, **kwargs):
            nonlocal raced
            if not raced and Path(target).name == candidate.name:
                parent.rename(moved_parent)
                parent.symlink_to(outside, target_is_directory=True)
                raced = True
            return real_rmtree(target, *args, dir_fd=dir_fd, **kwargs)

        monkeypatch.setattr(dg, "_SHUTIL_RMTREE", replace_parent_at_rmtree)

        summary = dg.quick()

        assert raced is True
        assert summary["deleted"] == 1
        assert outside_source.exists()
        assert not (moved_parent / candidate.name).exists()

    def test_quick_fails_closed_without_dir_fd_delete_support(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        candidate = _isolate_env / "scratch" / "test_unsupported.py"
        candidate.parent.mkdir()
        candidate.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(candidate),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": candidate.stat().st_size,
        }]))
        monkeypatch.setattr(dg, "_DIR_FD_DELETE_SUPPORTED", False)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert candidate.exists()

    def test_quick_fails_closed_without_symlink_safe_rmtree(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        candidate = _isolate_env / "scratch" / "test_directory"
        candidate.mkdir(parents=True)
        source = candidate / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(candidate),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))
        monkeypatch.setattr(dg, "_RMTREE_SYMLINK_SAFE", False)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert source.exists()

    def test_quick_rechecks_directory_after_final_nested_repo_scan(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        bundle = _isolate_env / "scratch" / "test_directory_race"
        bundle.mkdir(parents=True)
        source = bundle / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(bundle),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))
        protected = False
        marker_probes = 0

        def git_protection(_path):
            return protected

        def marker_scan(_path):
            nonlocal marker_probes, protected
            marker_probes += 1
            if marker_probes >= 2:
                protected = True
            return False

        monkeypatch.setattr(dg, "_is_git_protected_path", git_protection)
        monkeypatch.setattr(dg, "_directory_contains_git_marker", marker_scan)

        summary = dg.quick()

        assert marker_probes >= 2
        assert summary["deleted"] == 0
        assert source.exists()

    def test_quick_preserves_directory_containing_nested_git_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        bundle = _isolate_env / "scratch" / "test_bundle"
        repo = _make_git_repo(bundle / "project")
        source = repo / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(bundle),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert source.exists()

    def test_quick_refuses_explicit_ancestor_containing_nested_git_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        bundle = _isolate_env / "scratch" / "test_explicit_bundle"
        repo = _make_git_repo(bundle / "project")
        source = repo / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        entry = {
            "path": str(bundle),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
            "explicit": True,
        }
        tracked_file.write_text(json.dumps([entry]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert source.exists()
        assert json.loads(tracked_file.read_text()) == [entry]

    def test_quick_preserves_directory_containing_nested_bare_git_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        bundle = _isolate_env / "scratch" / "test_bare_bundle"
        bare = bundle / "archive.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(bare)],
            check=True,
        )
        head = bare / "HEAD"
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(bundle),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert head.exists()

    def test_quick_preserves_stale_file_inside_bare_git_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        bare = _isolate_env / "scratch" / "archive.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(bare)],
            check=True,
        )
        ref = bare / "refs" / "test_branch"
        ref.write_text("0" * 40 + "\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(ref),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": ref.stat().st_size,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert ref.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_quick_does_not_sweep_empty_directories_inside_bare_git_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        bare = _isolate_env / "scratch" / "archive.git"
        subprocess.run(
            ["git", "init", "--bare", "--quiet", str(bare)],
            check=True,
        )
        empty = bare / "refs" / "tags" / "nested-empty"
        empty.mkdir(parents=True)

        dg.quick()

        assert empty.exists()

    def test_quick_fails_closed_on_unexpected_git_ls_files_error(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        p = repo / "cache" / "test_cache.py"
        p.parent.mkdir()
        p.write_text("source\n")
        tracked_file = repo / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": p.stat().st_size,
        }]))
        real_run = dg.subprocess.run

        def fail_ls_files(args, *pargs, **kwargs):
            if "ls-files" in args:
                return subprocess.CompletedProcess(args, 128)
            return real_run(args, *pargs, **kwargs)

        monkeypatch.setattr(dg.subprocess, "run", fail_ls_files)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists()

    def test_quick_fails_closed_on_unexpected_git_check_ignore_error(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        p = repo / "scratch" / "test_cache.py"
        p.parent.mkdir()
        p.write_text("source\n")
        tracked_file = repo / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": p.stat().st_size,
        }]))
        real_run = dg.subprocess.run

        def fail_check_ignore(args, *pargs, **kwargs):
            if "check-ignore" in args:
                return subprocess.CompletedProcess(args, 128)
            return real_run(args, *pargs, **kwargs)

        monkeypatch.setattr(dg.subprocess, "run", fail_check_ignore)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists()

    def test_quick_fails_closed_when_git_marker_probe_errors(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        (_isolate_env / ".git").mkdir()
        p = _isolate_env / "scratch" / "test_source.py"
        p.parent.mkdir()
        p.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": p.stat().st_size,
        }]))
        real_is_dir = Path.is_dir

        def fail_git_marker(candidate):
            if candidate.name == ".git":
                raise OSError("marker unreadable")
            return real_is_dir(candidate)

        monkeypatch.setattr(Path, "is_dir", fail_git_marker)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists()

    def test_manual_cron_output_override_survives_revalidation(
        self, _isolate_env
    ):
        from datetime import datetime, timedelta, timezone

        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        p = repo / "cron" / "output" / "job-1" / "run.md"
        p.parent.mkdir(parents=True)
        p.write_text("output\n")
        subprocess.run(
            ["git", "-C", str(repo), "add", str(p.relative_to(repo))],
            check=True,
        )
        assert dg.track(str(p), "cron-output", silent=True, explicit=True) is True
        tracked = dg.load_tracked()
        tracked[0]["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(days=20)
        ).isoformat()
        dg.save_tracked(tracked)

        summary = dg.quick()

        assert summary["deleted"] == 1
        assert not p.exists()

    def test_quick_preserves_tracked_test_in_hermes_home_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        p = repo / "scripts" / "tests" / "test_source.py"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        subprocess.run(
            ["git", "-C", str(repo), "add", str(p.relative_to(repo))],
            check=True,
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert p.exists()
        assert json.loads(tracked_file.read_text()) == []

    def test_quick_deletes_ignored_test_in_hermes_home_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env)
        (repo / ".gitignore").write_text("scratch/\n")
        p = repo / "scratch" / "test_ephemeral.py"
        p.parent.mkdir()
        p.write_text("x")

        assert dg.track(str(p), "test", silent=True) is True
        summary = dg.quick()

        assert summary["deleted"] == 1
        assert not p.exists()

    def test_quick_does_not_sweep_empty_dirs_inside_git_worktree(
        self, _isolate_env
    ):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env / "worktrees" / "feature")
        empty = repo / "src" / "empty"
        empty.mkdir(parents=True)

        dg.quick()

        assert empty.exists()


class TestDeepCleanupSafety:
    def test_deep_rechecks_after_confirmation_before_recursive_delete(
        self, _isolate_env
    ):
        dg = _load_lib()
        profile = _isolate_env / "scratch" / "profile"
        profile.mkdir(parents=True)
        source = profile / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(profile),
            "category": "chrome-profile",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        def confirm_and_create_repo(_item):
            _make_git_repo(profile)
            return True

        summary = dg.deep(confirm=confirm_and_create_repo)

        assert summary["deep_deleted"] == 0
        assert source.exists()

    def test_deep_drops_stale_external_entry_without_deleting_target(
        self, _isolate_env
    ):
        dg = _load_lib()
        outside = _isolate_env.parent / "outside-profile"
        outside.mkdir()
        source = outside / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(outside),
            "category": "chrome-profile",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        summary = dg.deep(confirm=lambda _item: True)

        assert summary["quick"]["deleted"] == 0
        assert summary["deep_deleted"] == 0
        assert source.exists()
        assert json.loads(tracked_file.read_text()) == []


class TestStatus:
    def test_empty_status(self, _isolate_env):
        dg = _load_lib()
        s = dg.status()
        assert s["total_tracked"] == 0
        assert s["top10"] == []

    def test_status_with_entries(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "big.tmp"
        p.write_text("y" * 100)
        dg.track(str(p), "temp", silent=True)
        s = dg.status()
        assert s["total_tracked"] == 1
        assert len(s["top10"]) == 1
        rendered = dg.format_status(s)
        assert "temp" in rendered
        assert "big.tmp" in rendered


class TestDryRun:
    def test_classifies_by_category(self, _isolate_env):
        dg = _load_lib()
        test_f = _isolate_env / "test_x.py"
        test_f.write_text("x")
        big = _isolate_env / "big.bin"
        big.write_bytes(b"z" * 10)
        dg.track(str(test_f), "test", silent=True)
        dg.track(str(big), "other", silent=True)
        auto, prompt = dg.dry_run()
        # test → auto, other → neither (doesn't hit any rule)
        assert any(i["path"] == str(test_f) for i in auto)

    def test_omits_stale_git_worktree_entries(self, _isolate_env):
        dg = _load_lib()
        repo = _make_git_repo(_isolate_env / "worktrees" / "feature")
        p = repo / "test_feature.py"
        p.write_text("x")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        }]))

        auto, prompt = dg.dry_run()

        assert auto == []
        assert prompt == []

    def test_omits_tracked_ancestor_containing_nested_git_repo(
        self, _isolate_env
    ):
        dg = _load_lib()
        bundle = _isolate_env / "scratch" / "test_bundle"
        repo = _make_git_repo(bundle / "project")
        source = repo / "source.py"
        source.write_text("source\n")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(bundle),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]))

        auto, prompt = dg.dry_run()

        assert auto == []
        assert prompt == []


# ---------------------------------------------------------------------------
# Plugin hooks tests
# ---------------------------------------------------------------------------

class TestPostToolCallHook:
    def test_write_file_test_pattern_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "test_created.py"
        p.write_text("x")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="t1", session_id="s1",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text())
        assert len(data) == 1
        assert data[0]["category"] == "test"
        assert data[0]["explicit"] is False

    def test_write_file_git_worktree_test_is_not_auto_tracked(
        self, _isolate_env
    ):
        pi = _load_plugin_init()
        repo = _make_git_repo(_isolate_env / "worktrees" / "feature")
        p = repo / "test_created.py"
        p.write_text("x")

        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="tw", session_id="sw",
        )
        pi._on_session_end(session_id="sw", completed=True, interrupted=False)

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert p.exists()
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"

    def test_write_file_non_test_not_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "notes.md"
        p.write_text("x")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="t2", session_id="s2",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"

    def test_terminal_command_picks_up_paths(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "tmp_created.log"
        p.write_text("x")
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": f"touch {p}"},
            result=f"created {p}\n",
            task_id="t3", session_id="s3",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text())
        assert any(Path(i["path"]) == p.resolve() for i in data)

    def test_ignores_unrelated_tool(self, _isolate_env):
        pi = _load_plugin_init()
        pi._on_post_tool_call(
            tool_name="read_file",
            args={"path": str(_isolate_env / "test_x.py")},
            result="contents",
            task_id="t4", session_id="s4",
        )
        # read_file should never trigger tracking.
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"


class TestOnSessionEndHook:
    def test_runs_quick_when_test_files_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "test_cleanup.py"
        p.write_text("x")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="", session_id="s1",
        )
        assert p.exists()
        pi._on_session_end(session_id="s1", completed=True, interrupted=False)
        assert not p.exists(), "test file should be auto-deleted"

    def test_noop_when_no_test_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        # Nothing tracked → on_session_end should not raise.
        pi._on_session_end(session_id="empty", completed=True, interrupted=False)


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

class TestSlashCommand:
    def test_help(self, _isolate_env):
        pi = _load_plugin_init()
        out = pi._handle_slash("help")
        assert "disk-cleanup" in out
        assert "status" in out

    def test_status_empty(self, _isolate_env):
        pi = _load_plugin_init()
        out = pi._handle_slash("status")
        assert "nothing tracked" in out

    def test_track_rejects_missing(self, _isolate_env):
        pi = _load_plugin_init()
        out = pi._handle_slash(
            f"track {_isolate_env / 'nope.txt'} temp"
        )
        assert "Not tracked" in out

    def test_track_rejects_bad_category(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "a.tmp"
        p.write_text("x")
        out = pi._handle_slash(f"track {p} banana")
        assert "Unknown category" in out

    def test_track_and_forget(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "a.tmp"
        p.write_text("x")
        out = pi._handle_slash(f"track {p} temp")
        assert "Tracked" in out
        out = pi._handle_slash(f"forget {p}")
        assert "Removed 1" in out

    def test_unknown_subcommand(self, _isolate_env):
        pi = _load_plugin_init()
        out = pi._handle_slash("foobar")
        assert "Unknown subcommand" in out

    def test_quick_on_empty(self, _isolate_env):
        pi = _load_plugin_init()
        out = pi._handle_slash("quick")
        assert "Cleaned 0 files" in out


# ---------------------------------------------------------------------------
# Bundled-plugin discovery
# ---------------------------------------------------------------------------

class TestBundledDiscovery:
    def _write_enabled_config(self, hermes_home, names):
        """Write plugins.enabled allow-list to config.yaml."""
        import yaml
        cfg_path = hermes_home / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"plugins": {"enabled": list(names)}}))

    def test_disk_cleanup_discovered_but_not_loaded_by_default(self, _isolate_env):
        """Bundled plugins are discovered but NOT loaded without opt-in."""
        from hermes_cli import plugins as pmod
        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        # Discovered — appears in the registry
        assert "disk-cleanup" in mgr._plugins
        loaded = mgr._plugins["disk-cleanup"]
        assert loaded.manifest.source == "bundled"
        # But NOT enabled — no hooks or commands registered
        assert not loaded.enabled
        assert loaded.error and "not enabled" in loaded.error

    def test_disk_cleanup_loads_when_enabled(self, _isolate_env):
        """Adding to plugins.enabled activates the bundled plugin."""
        self._write_enabled_config(_isolate_env, ["disk-cleanup"])
        from hermes_cli import plugins as pmod
        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["disk-cleanup"]
        assert loaded.enabled
        assert "post_tool_call" in loaded.hooks_registered
        assert "on_session_end" in loaded.hooks_registered
        assert "disk-cleanup" in loaded.commands_registered

    def test_disabled_beats_enabled(self, _isolate_env):
        """plugins.disabled wins even if the plugin is also in plugins.enabled."""
        import yaml
        cfg_path = _isolate_env / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "plugins": {
                "enabled": ["disk-cleanup"],
                "disabled": ["disk-cleanup"],
            }
        }))
        from hermes_cli import plugins as pmod
        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["disk-cleanup"]
        assert not loaded.enabled
        assert loaded.error == "disabled via config"

    def test_memory_and_context_engine_subdirs_skipped(self, _isolate_env):
        """Bundled scan must NOT pick up plugins/memory or plugins/context_engine
        as top-level plugins — they have their own discovery paths."""
        self._write_enabled_config(
            _isolate_env, ["memory", "context_engine", "disk-cleanup"]
        )
        from hermes_cli import plugins as pmod
        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        assert "memory" not in mgr._plugins
        assert "context_engine" not in mgr._plugins
