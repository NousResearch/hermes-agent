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
import sys
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

    def test_quick_does_not_descend_into_protected_top_level_dirs(self, _isolate_env):
        dg = _load_lib()
        protected_empty = (
            _isolate_env / "hermes-agent" / "node_modules" / "pkg" / "empty"
        )
        protected_empty.mkdir(parents=True)

        summary = dg.quick()

        assert summary["empty_dirs"] == 0
        assert protected_empty.exists()

    def test_quick_removes_empty_dirs_in_managed_subtrees(self, _isolate_env):
        dg = _load_lib()
        managed_empty = _isolate_env / "scratch" / "nested" / "empty"
        managed_empty.mkdir(parents=True)

        summary = dg.quick()

        assert summary["empty_dirs"] == 3
        assert not (_isolate_env / "scratch").exists()


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
