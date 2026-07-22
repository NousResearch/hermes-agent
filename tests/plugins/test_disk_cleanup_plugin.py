"""Tests for the disk-cleanup plugin.

Covers the bundled plugin at ``plugins/disk-cleanup/``:

  * ``disk_cleanup`` library: track / forget / dry_run / quick / status,
    ``is_safe_path`` and ``guess_category`` filtering.
  * Plugin ``__init__``: ``post_tool_call`` hook auto-tracks files created
    by ``write_file``; ``on_session_end`` cleans only that session's paths.
  * Slash command handler: status / dry-run / quick / track / forget /
    unknown subcommand behaviours.
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
"""

import importlib
import concurrent.futures
import json
import os
import sys
import time
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
    # Ensure parent namespace package exists for the relative `. import disk_cleanup`
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


def _write_file_creation_result(plugin, path: Path) -> str:
    identity = plugin.dg._capture_identity(path)
    assert identity is not None
    return json.dumps({
        "created_paths": [str(path)],
        "created_path_identities": [
            {"path": str(path), "identity": identity},
        ],
    })


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
        # Legacy records without filesystem identity are preserved for manual
        # review rather than treated as deletion authority.
        remaining = json.loads(tracked_file.read_text())
        assert len(remaining) == 1

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

        assert dg.track(str(run_md), "cron-output", silent=True)
        tracked = dg.load_tracked()
        tracked[0]["timestamp"] = old_ts
        dg.save_tracked(tracked)

        summary = dg.quick()
        assert summary["deleted"] == 1, "valid old cron-output should be deleted"
        assert not run_md.exists()


class TestTrackForgetQuick:
    def test_track_then_quick_deletes_test(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "test_a.py"
        p.write_text("x")
        assert dg.track(str(p), "test", silent=True) is True
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

    def test_track_skips_missing(self, _isolate_env):
        dg = _load_lib()
        assert dg.track(str(_isolate_env / "nope.txt"), "test", silent=True) is False

    def test_track_rejects_directories(self, _isolate_env):
        dg = _load_lib()
        directory = _isolate_env / "test_tree"
        nested = directory / "existing"
        nested.mkdir(parents=True)
        (nested / "human.txt").write_text("keep")

        assert dg.track(str(directory), "test", silent=True) is False
        assert dg.quick()["deleted"] == 0
        assert (nested / "human.txt").exists()

    def test_quick_preserves_legacy_tracked_directory(self, _isolate_env):
        dg = _load_lib()
        directory = _isolate_env / "test_legacy_tree"
        nested = directory / "existing"
        nested.mkdir(parents=True)
        human = nested / "human.txt"
        human.write_text("keep")
        identity = dg._capture_identity(directory)
        assert identity is not None
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(directory),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
            "identity": identity,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert summary["errors"]
        assert human.read_text() == "keep"

    def test_track_rejects_protected_durable_file(self, _isolate_env):
        dg = _load_lib()
        memories = _isolate_env / "memories"
        memories.mkdir()
        memory = memories / "MEMORY.md"
        memory.write_text("human memory")

        assert dg.track(str(memory), "test", silent=True) is False
        identity = dg._capture_identity(memory)
        assert identity is not None
        dg.save_tracked([{
            "path": str(memory),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": memory.stat().st_size,
            "identity": identity,
        }])
        summary = dg.quick()
        assert summary["deleted"] == 0
        assert summary["errors"]
        assert memory.read_text() == "human memory"

    @pytest.mark.parametrize(
        "filename",
        [
            "state.db",
            "kanban.db",
            "projects.db",
            "response_store.db",
            "memory_store.db",
            "verification_evidence.db",
            "gateway_state.json",
            "channel_directory.json",
            "channel_aliases.json",
            "processes.json",
            "feishu_comment_pairing.json",
            "kanban.db-wal",
            "state.db-shm",
            "webhook_subscriptions.json",
            "auth.lock",
        ],
    )
    def test_canonical_top_level_state_is_never_deletable(
        self, _isolate_env, filename
    ):
        dg = _load_lib()
        durable = _isolate_env / filename
        durable.write_text("human state")
        identity = dg._capture_identity(durable)

        assert dg.track(str(durable), "test", silent=True) is False
        dg.save_tracked([{
            "path": str(durable),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": durable.stat().st_size,
            "identity": identity,
        }])

        assert dg.quick()["deleted"] == 0
        assert durable.read_text() == "human state"

    def test_quick_commit_preserves_concurrent_tracking_addition(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        first = _isolate_env / "test_first.py"
        concurrent = _isolate_env / "tmp_concurrent.log"
        first.write_text("delete")
        concurrent.write_text("preserve tracking")
        assert dg.track(str(first), "test", silent=True)
        original_remove = dg._remove_tracked_path

        def remove_while_another_hook_tracks(path, identity):
            assert dg.track(str(concurrent), "temp", silent=True)
            return original_remove(path, identity)

        monkeypatch.setattr(dg, "_remove_tracked_path", remove_while_another_hook_tracks)

        assert dg.quick()["deleted"] == 1
        assert [item["path"] for item in dg.load_tracked()] == [str(concurrent.resolve())]

    def test_parallel_tracking_keeps_every_registry_entry(self, _isolate_env):
        dg = _load_lib()
        paths = [_isolate_env / f"parallel-{index}.tmp" for index in range(20)]
        for path in paths:
            path.write_text(path.name)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(
                executor.map(
                    lambda path: dg.track(str(path), "temp", silent=True), paths
                )
            )

        assert all(results)
        assert {item["path"] for item in dg.load_tracked()} == {
            str(path.resolve()) for path in paths
        }

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
        (_isolate_env / "scratch" / ".hermes-managed").write_text("scratch\n")

        summary = dg.quick()

        assert summary["empty_dirs"] == 2
        assert not managed_empty.exists()
        assert (_isolate_env / "scratch").exists()

        # A second pass must be a no-op: the ownership marker keeps the
        # managed root, and no removed directory is rediscovered.
        second = dg.quick()
        assert second["deleted"] == 0
        assert second["empty_dirs"] == 0
        assert second["errors"] == []

    def test_quick_preserves_unmarked_empty_owned_name(self, _isolate_env):
        """A familiar root name is not ownership evidence by itself."""
        dg = _load_lib()
        unowned_empty = _isolate_env / "scratch" / "nested" / "empty"
        unowned_empty.mkdir(parents=True)

        summary = dg.quick()

        assert summary["empty_dirs"] == 0
        assert unowned_empty.exists()

    def test_quick_preserves_malformed_empty_root_marker(self, _isolate_env):
        """A foreign marker must not opt a root into recursive deletion."""
        dg = _load_lib()
        root = _isolate_env / "tmp"
        unowned_empty = root / "nested" / "empty"
        unowned_empty.mkdir(parents=True)
        (root / ".hermes-managed").write_text("not-tmp\n")

        summary = dg.quick()

        assert summary["empty_dirs"] == 0
        assert unowned_empty.exists()

    def test_quick_preserves_marked_root_with_live_lock(self, _isolate_env):
        """Ownership evidence does not override a live/ambiguous lock."""
        dg = _load_lib()
        root = _isolate_env / "scratch"
        nested = root / "nested" / "empty"
        nested.mkdir(parents=True)
        (root / ".hermes-managed").write_text("scratch\n")
        (root / ".lock").write_text("active\n")

        summary = dg.quick()

        assert summary["empty_dirs"] == 0
        assert nested.exists()

    def test_quick_preserves_malformed_record_and_continues(self, _isolate_env):
        """A malformed record cannot abort cleanup or authorize deletion."""
        dg = _load_lib()
        valid = _isolate_env / "test_valid.py"
        valid.write_text("keep only until the sweep")
        assert dg.track(str(valid), "test", silent=True)
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked = dg.load_tracked()
        tracked.insert(
            0,
            {"path": str(_isolate_env / "test_malformed.py"), "category": "test"},
        )
        dg.save_tracked(tracked)

        summary = dg.quick()

        assert summary["deleted"] == 1
        assert not valid.exists()
        saved = json.loads(tracked_file.read_text())
        assert len(saved) == 1
        assert saved[0]["path"].endswith("test_malformed.py")

    def test_quick_preserves_unverifiable_delete_and_reports_error(
        self, _isolate_env, monkeypatch
    ):
        """An inspection/delete error never counts as successful cleanup."""
        dg = _load_lib()
        locked = _isolate_env / "test_locked.py"
        locked.write_text("keep")
        dg.track(str(locked), "test", silent=True)

        original_is_file = Path.is_file

        def fail_for_locked(path):
            if path == locked:
                raise OSError("simulated live lock")
            return original_is_file(path)

        monkeypatch.setattr(Path, "is_file", fail_for_locked)

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert summary["errors"]
        assert locked.exists()
        assert any(item["path"] == str(locked) for item in dg.load_tracked())

    def test_quick_preserves_replacement_at_tracked_path(self, _isolate_env):
        """A path record cannot authorize deletion of a replacement object."""
        dg = _load_lib()
        path = _isolate_env / "test_agent_created.py"
        path.write_text("agent scratch")
        assert dg.track(str(path), "test", silent=True)

        path.unlink()
        path.write_text("human replacement")

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert summary["errors"]
        assert path.read_text() == "human replacement"
        assert any(item["path"] == str(path) for item in dg.load_tracked())

    def test_quick_prunes_only_marked_stale_hook_output(self, _isolate_env):
        """Hook spill files are owned only after the managed marker exists."""
        dg = _load_lib()
        session_dir = _isolate_env / "hook_outputs" / "session-1"
        session_dir.mkdir(parents=True)
        (session_dir / ".hermes-managed").write_text("hook_outputs\n")
        stale = session_dir / "old.txt"
        stale.write_text("stale")
        old = time.time() - (15 * 24 * 60 * 60)
        os.utime(stale, (old, old))

        summary = dg.quick()

        assert summary["artifacts"] == 1
        assert not stale.exists()

    def test_quick_preserves_unmarked_stale_hook_output(self, _isolate_env):
        """Unknown files below an artifact-looking root are never junk by name."""
        dg = _load_lib()
        session_dir = _isolate_env / "hook_outputs" / "session-unknown"
        session_dir.mkdir(parents=True)
        stale = session_dir / "old.txt"
        stale.write_text("keep")
        old = time.time() - (30 * 24 * 60 * 60)
        os.utime(stale, (old, old))

        summary = dg.quick()

        assert summary["artifacts"] == 0
        assert stale.exists()

    def test_quick_preserves_stale_artifact_when_session_is_locked(self, _isolate_env):
        """A managed session lock suppresses artifact retention cleanup."""
        dg = _load_lib()
        session_dir = _isolate_env / "hook_outputs" / "session-locked"
        session_dir.mkdir(parents=True)
        (session_dir / ".hermes-managed").write_text("hook_outputs\n")
        (session_dir / ".lock").write_text("active\n")
        stale = session_dir / "old.txt"
        stale.write_text("keep")
        old = time.time() - (30 * 24 * 60 * 60)
        os.utime(stale, (old, old))

        summary = dg.quick()

        assert summary["artifacts"] == 0
        assert stale.exists()

    def test_quick_prunes_stale_marked_spawn_tree_snapshots(self, _isolate_env):
        """Spawn-tree history is bounded without deleting unowned sessions."""
        dg = _load_lib()
        session_dir = _isolate_env / "spawn-trees" / "session-1"
        session_dir.mkdir(parents=True)
        (session_dir / ".hermes-managed").write_text("spawn-trees\n")
        stale = session_dir / "20240101T000000.json"
        stale.write_text("{}")
        old = time.time() - (31 * 24 * 60 * 60)
        os.utime(stale, (old, old))

        summary = dg.quick()

        assert summary["artifacts"] == 1
        assert not stale.exists()

    def test_quick_preserves_empty_unowned_top_level_dir(self, _isolate_env):
        """An empty directory is not deletion evidence merely because it is under HERMES_HOME."""
        dg = _load_lib()
        unowned = _isolate_env / "project-output"
        unowned.mkdir()

        dg.quick()

        assert unowned.exists()

    def test_quick_preserves_tracked_path_outside_hermes_home(self, _isolate_env, tmp_path):
        """A stale tracking record cannot authorize deleting an external path."""
        dg = _load_lib()
        outside = tmp_path / "external-test.py"
        outside.write_text("keep")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(outside),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": outside.stat().st_size,
        }]))

        summary = dg.quick()

        assert summary["deleted"] == 0
        assert outside.exists()

    def test_path_traversal_cannot_escape_tmp_hermes_scope(self, _isolate_env):
        dg = _load_lib()

        assert dg.is_safe_path(Path("/tmp/hermes-owned/../outside")) is False

    def test_explicit_windows_drive_tmp_is_not_approved_scope(self, _isolate_env):
        dg = _load_lib()
        if os.name == "nt":
            assert dg.is_safe_path(Path(r"C:\tmp\hermes-user\artifact")) is False


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


# ---------------------------------------------------------------------------
# Plugin hooks tests
# ---------------------------------------------------------------------------

class TestPostToolCallHook:
    def test_write_file_test_pattern_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "test_created.py"
        from tools.file_tools import write_file_tool

        result = write_file_tool(str(p), "x", task_id="t1")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result=result,
            task_id="t1", session_id="s1",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text())
        assert len(data) == 1
        assert data[0]["category"] == "test"

    def test_replacement_before_hook_is_not_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "test_replaced_before_hook.py"
        from tools.file_tools import write_file_tool

        result = write_file_tool(str(p), "agent scratch", task_id="race-task")
        assert p.read_text() == "agent scratch"
        p.unlink()
        p.write_text("human replacement")

        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "agent scratch"},
            result=result,
            task_id="race-task",
            session_id="race-session",
        )
        pi._on_session_end(session_id="race-session")

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"
        assert p.read_text() == "human replacement"

    def test_legacy_created_path_without_identity_is_not_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "test_legacy_metadata.py"
        p.write_text("human file")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "unknown"},
            result=json.dumps({"created_paths": [str(p)]}),
            task_id="legacy-task",
            session_id="legacy-session",
        )
        pi._on_session_end(session_id="legacy-session")
        assert p.read_text() == "human file"

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

    def test_terminal_metadata_cannot_authorize_cleanup(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "tmp_created.log"
        p.write_text("x")
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": f"touch {p}"},
            result=json.dumps({
                "output": f"created {p}\n",
                "created_paths": [str(p)],
            }),
            task_id="t3", session_id="s3",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"
        pi._on_session_end(session_id="s3", completed=True, interrupted=False)
        assert p.exists(), "terminal metadata must not authorize deletion"

    def test_patch_files_created_cannot_authorize_cleanup(self, _isolate_env):
        """V4A creation labels are ambiguous and never destructive evidence."""
        pi = _load_plugin_init()
        p = _isolate_env / "test_existing.py"
        p.write_text("human file")
        pi._on_post_tool_call(
            tool_name="patch",
            args={"mode": "patch", "patch": "*** Add File: test_existing.py"},
            result=json.dumps({"files_created": [str(p)]}),
            task_id="patch-task",
            session_id="patch-session",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"
        pi._on_session_end(session_id="patch-session")
        assert p.read_text() == "human file"

    def test_terminal_read_only_output_cannot_track_preexisting_test_file(
        self, _isolate_env
    ):
        """Mentioning a user file in terminal output is not creation evidence."""
        pi = _load_plugin_init()
        p = _isolate_env / "test_user_notes.py"
        p.write_text("keep me")
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": "python inspect_only.py"},
            result=json.dumps({
                "output": f"inspected {p}",
                "exit_code": 0,
                "error": None,
            }),
            task_id="readonly-task", session_id="readonly-session",
        )

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or tracked_file.read_text().strip() == "[]"
        pi._on_session_end(
            session_id="readonly-session", completed=True, interrupted=False
        )
        assert p.exists(), "pre-existing user file must survive session cleanup"

    def test_terminal_command_extracts_windows_drive_paths(self):
        pi = _load_plugin_init()

        paths = pi._extract_paths_from_terminal(
            {"command": r"type C:\Users\edson\test_created.py"},
            r"created C:\Users\edson\test_created.py",
        )

        assert any(path.replace("\\", "/") == "C:/Users/edson/test_created.py" for path in paths)

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
            result=_write_file_creation_result(pi, p),
            task_id="", session_id="s1",
        )
        assert p.exists()
        pi._on_session_end(session_id="s1", completed=True, interrupted=False)
        assert not p.exists(), "test file should be auto-deleted"

    def test_noop_when_no_test_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        # Nothing tracked → on_session_end should not raise.
        pi._on_session_end(session_id="empty", completed=True, interrupted=False)

    def test_cleanup_is_scoped_to_ending_session(self, _isolate_env):
        pi = _load_plugin_init()
        first = _isolate_env / "test_session_one.py"
        second = _isolate_env / "test_session_two.py"
        first.write_text("one")
        second.write_text("two")
        for path, session in ((first, "s1"), (second, "s2")):
            pi._on_post_tool_call(
                tool_name="write_file",
                args={"path": str(path), "content": path.read_text()},
                result=_write_file_creation_result(pi, path),
                task_id=f"task-{session}",
                session_id=session,
            )

        pi._on_session_end(session_id="s1")

        assert not first.exists()
        assert second.exists(), "ending session s1 must not clean active session s2"
        pi._on_session_end(session_id="s2")
        assert not second.exists()


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
