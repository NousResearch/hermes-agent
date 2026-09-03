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


    def test_forget_removes_entry(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "keep.tmp"
        p.write_text("x")
        dg.track(str(p), "temp", silent=True)
        assert dg.forget(str(p)) == 1
        assert p.exists()  # forget does NOT delete the file


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


    def test_unknown_subcommand(self, _isolate_env):
        pi = _load_plugin_init()
        out = pi._handle_slash("foobar")
        assert "Unknown subcommand" in out


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


# ---------------------------------------------------------------------------
# Regression tests for #75403 — durable test files must not be auto-deleted
# ---------------------------------------------------------------------------

class TestDurableTestFileProtection75403:
    """A file named ``test_*`` / ``tmp_*`` inside a durable project/config
    tree must never be treated as disposable, and a pre-existing file the
    agent merely edited must never be auto-tracked for deletion."""

    # --- Layer 1: guess_category must not infer disposability from a
    # basename inside durable project trees -------------------------------

    def test_guess_category_excludes_durable_project_trees(self, _isolate_env):
        dg = _load_lib()
        for tree in ("patches", "projects", "skins", "themes", "contributors"):
            p = _isolate_env / tree / "tests" / "test_manager.py"
            p.parent.mkdir(parents=True)
            p.write_text("x")
            assert dg.guess_category(p) is None, (
                f"{tree}/ test file must not be disposable (#75403)"
            )

    def test_guess_category_excludes_backups_profiles_optional_skills(self, _isolate_env):
        dg = _load_lib()
        for tree in ("backups", "profiles", "optional-skills"):
            p = _isolate_env / tree / "tmp_snapshot.json"
            p.parent.mkdir(parents=True)
            p.write_text("x")
            assert dg.guess_category(p) is None, (
                f"{tree}/ tmp file must not be disposable (#75403)"
            )

    # --- Layer 2: the empty-directory sweep must never prune durable
    # project trees --------------------------------------------------------

    def test_quick_does_not_sweep_empty_durable_project_dirs(self, _isolate_env):
        dg = _load_lib()
        durable_empty = _isolate_env / "patches" / "tests" / "empty"
        durable_empty.mkdir(parents=True)
        dg.quick()
        assert (_isolate_env / "patches").exists(), (
            "patches/ durable tree must not be swept (#75403)"
        )

    # --- Layer 3: pre-existing files must not be auto-tracked when a tool
    # merely edits them ---------------------------------------------------

    def test_preexisting_test_file_not_tracked_on_edit(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "scratch" / "test_existing.py"
        p.parent.mkdir(parents=True)
        p.write_text("original")  # exists BEFORE the tool call
        # pre_tool_call snapshots pre-existence
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "edited"},
            task_id="t75403a", session_id="s75403a",
        )
        # post_tool_call (file merely edited) must NOT track it
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "edited"},
            result="OK",
            task_id="t75403a", session_id="s75403a",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(tracked_file.read_text()) == []

    def test_preexisting_test_file_in_durable_tree_not_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "patches" / "tests" / "test_manager.py"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p)},
            task_id="t75403c", session_id="s75403c",
        )
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p)},
            result="OK",
            task_id="t75403c", session_id="s75403c",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(tracked_file.read_text()) == []

    def test_newly_created_test_file_still_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _isolate_env / "test_brand_new.py"
        # pre_tool_call fires while the file does not yet exist
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            task_id="t75403b", session_id="s75403b",
        )
        # the tool then creates the file
        p.write_text("x")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="t75403b", session_id="s75403b",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text())
        assert len(data) == 1
        assert data[0]["category"] == "test"

    # --- Layer 4: stale tracked entries must not be deleted ---------

    def test_quick_skips_stale_test_entry_for_durable_tree(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "patches" / "tests" / "test_manager.py"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        }]))
        summary = dg.quick()
        assert summary["deleted"] == 0, (
            "stale test entry in durable tree must not be deleted (#75403)"
        )
        assert p.exists()

    def test_dry_run_omits_stale_test_entry_for_durable_tree(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "patches" / "tests" / "test_manager.py"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        }]))
        auto, prompt = dg.dry_run()
        assert len(auto) == 0, "stale test entry must not appear in dry-run auto"
        assert len(prompt) == 0


# ---------------------------------------------------------------------------
# Regression tests for concurrency/data-loss blockers fixed in #75464
# ---------------------------------------------------------------------------

class TestTerminalResultOnlyPathProtection:
    """Terminal result-only paths must NEVER be auto-tracked because their
    pre-existence is unknowable at pre-call time (#75464 blocker)."""

    def test_result_only_path_not_tracked(self, _isolate_env):
        """A path that appears ONLY in terminal result (not in command args)
        must not be auto-tracked, even if it matches test patterns."""
        pi = _load_plugin_init()
        p = _isolate_env / "test_result_only.py"
        p.write_text("x")  # file exists, but path only in result, not command
        # pre_tool_call: command does NOT contain the path
        pi._on_pre_tool_call(
            tool_name="terminal",
            args={"command": "echo 'done'"},
            task_id="t_res", session_id="s_res",
        )
        # post_tool_call: result contains the path, but command didn't
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": "echo 'done'"},
            result=f"created {p}\n",
            task_id="t_res", session_id="s_res",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(tracked_file.read_text()) == []

    def test_command_arg_path_still_tracked(self, _isolate_env):
        """A path that appears in command args IS eligible for tracking
        (pre-existence was snapshotted)."""
        pi = _load_plugin_init()
        p = _isolate_env / "test_cmd_arg.py"
        # File does NOT pre-exist — it's being created
        pi._on_pre_tool_call(
            tool_name="terminal",
            args={"command": f"touch {p}"},
            task_id="t_cmd", session_id="s_cmd",
        )
        p.write_text("x")  # created by the command
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": f"touch {p}"},
            result="OK",
            task_id="t_cmd", session_id="s_cmd",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text())
        assert len(data) == 1
        assert data[0]["category"] == "test"

    def test_existing_command_arg_path_not_tracked(self, _isolate_env):
        """A pre-existing path in command args must NOT be auto-tracked
        (it was snapshotted pre-call and thus is merely being edited)."""
        pi = _load_plugin_init()
        p = _isolate_env / "test_existing_arg.py"
        p.write_text("original")  # exists before the tool call
        pi._on_pre_tool_call(
            tool_name="terminal",
            args={"command": f"echo foo >> {p}"},
            task_id="t_exist", session_id="s_exist",
        )
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": f"echo foo >> {p}"},
            result="OK",
            task_id="t_exist", session_id="s_exist",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(tracked_file.read_text()) == []


class TestTwoSessionInterleaving:
    """Two concurrent sessions must not interfere with each other's
    pre-existence snapshots (#75464 concurrency blocker)."""

    def test_session_a_end_does_not_erase_session_b_snapshot(self, _isolate_env):
        """Ending session A must not clear session B's pre-existing snapshots."""
        pi = _load_plugin_init()
        p_b = _isolate_env / "test_session_b.py"
        p_b.write_text("pre-existing for B")

        # Session B: snapshot pre-existence
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p_b), "content": "edited"},
            task_id="tb", session_id="session_B",
        )
        # Session A ends (should NOT affect session B's snapshot)
        pi._on_session_end(session_id="session_A", completed=True, interrupted=False)

        # Session B: post_tool_call should still see the pre-existing snapshot
        # and NOT track the file
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p_b), "content": "edited"},
            result="OK",
            task_id="tb", session_id="session_B",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(tracked_file.read_text()) == []

    def test_session_b_post_still_protects_preexisting_file(self, _isolate_env):
        """After session A ends, session B's post_tool_call still correctly
        protects its pre-existing file from auto-tracking."""
        pi = _load_plugin_init()
        p_b = _isolate_env / "test_session_b_protected.py"
        p_b.write_text("durable")

        # Session A: create and track a new file (unrelated)
        p_a = _isolate_env / "test_session_a_new.py"
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p_a), "content": "new"},
            task_id="ta", session_id="session_A",
        )
        p_a.write_text("new")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p_a), "content": "new"},
            result="OK",
            task_id="ta", session_id="session_A",
        )

        # Session B: snapshot pre-existing file
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p_b), "content": "modified"},
            task_id="tb", session_id="session_B",
        )

        # Session A ends
        pi._on_session_end(session_id="session_A", completed=True, interrupted=False)
        # Session A's file should be cleaned up
        assert not p_a.exists(), "session A's new test file should be cleaned"

        # Session B: post_tool_call must still protect p_b
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p_b), "content": "modified"},
            result="OK",
            task_id="tb", session_id="session_B",
        )
        # p_b must NOT be in tracked.json
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_data = json.loads(tracked_file.read_text()) if tracked_file.exists() else []
        p_b_resolved = str(p_b.resolve())
        assert not any(e["path"] == p_b_resolved for e in tracked_data), (
            "session B's pre-existing file must not be tracked"
        )
        assert p_b.exists(), "session B's durable file must survive"


    def test_explicit_tool_call_ids_isolate_snapshots(self, _isolate_env):
        """Each production-shaped tool call must consume only its own snapshot.

        A and B share task/session identity, so the explicit call IDs are the
        only discriminator. Ending an unrelated session must also leave both
        snapshots available.
        """
        pi = _load_plugin_init()
        p_a = _isolate_env / "test_call_a.py"
        p_b = _isolate_env / "test_call_b.py"
        p_a.write_text("durable A")
        p_b.write_text("durable B")

        hook_context = {"task_id": "shared-task", "session_id": "shared-session"}
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p_a), "content": "edited A"},
            tool_call_id="call-A",
            **hook_context,
        )
        pi._on_pre_tool_call(
            tool_name="write_file",
            args={"path": str(p_b), "content": "edited B"},
            tool_call_id="call-B",
            **hook_context,
        )

        # An unrelated session ending must not clear either in-flight call.
        pi._on_session_end(session_id="unrelated-session")

        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p_a), "content": "edited A"},
            result="OK",
            tool_call_id="call-A",
            **hook_context,
        )
        # If call IDs collide, call B's snapshot is consumed here and this
        # post hook will incorrectly auto-track B as a disposable new file.
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p_b), "content": "edited B"},
            result="OK",
            tool_call_id="call-B",
            **hook_context,
        )

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(tracked_file.read_text()) == []
        assert p_a.exists() and p_b.exists()


class TestDryRunNoDelete:
    """dry_run must never delete files — it's a read-only preview."""

    def test_dry_run_does_not_delete_tracked_files(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "test_dry.py"
        p.write_text("data")
        dg.track(str(p), "test", silent=True)
        assert p.exists()
        auto, prompt = dg.dry_run()
        # File should still exist — dry_run is non-destructive
        assert p.exists(), "dry_run must never delete files"
        # But it should show up in auto-delete preview
        assert any(i["path"] == str(p) for i in auto)

    def test_dry_run_no_side_effects(self, _isolate_env):
        """dry_run must not modify tracked.json or the filesystem."""
        dg = _load_lib()
        p = _isolate_env / "test_dry2.py"
        p.write_text("x")
        dg.track(str(p), "test", silent=True)
        tracked_before = json.loads(
            (_isolate_env / "disk-cleanup" / "tracked.json").read_text()
        )
        dg.dry_run()
        tracked_after = json.loads(
            (_isolate_env / "disk-cleanup" / "tracked.json").read_text()
        )
        assert tracked_before == tracked_after, "dry_run must not mutate tracked.json"
        assert p.exists(), "dry_run must not delete files"
