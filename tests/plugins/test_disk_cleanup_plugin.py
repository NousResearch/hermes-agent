"""Tests for the disk-cleanup plugin.

Covers the bundled plugin at ``plugins/disk-cleanup/``:

  * ``disk_cleanup`` library: track / forget / dry_run / quick / status,
    ``is_safe_path`` and ``guess_category`` filtering.
  * Plugin ``__init__``: ``post_tool_call`` auto-tracks files created in
    owned temporary roots; ``on_session_end`` cleans only the ending turn.
  * Slash command handler: status / dry-run / quick / track / forget /
    unknown subcommand behaviours.
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
"""

import importlib
import json
import multiprocessing
import os
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


@pytest.fixture
def _managed_tmp_root():
    """A real but unowned platform temp root for ownership-boundary tests."""
    root = Path(tempfile.mkdtemp(prefix="hermes-disk-cleanup-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _owned_test_root(module, hermes_home):
    """Add an isolated test-only immediate-cleanup root to one loaded module."""
    parts = ("cache", "disk-cleanup", "turn-files")
    module._MANAGED_HERMES_ROOTS[parts] = "test"
    root = hermes_home.joinpath(*parts)
    root.mkdir(parents=True, exist_ok=True)
    return root


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
    sys.modules.pop("hermes_plugins.disk_cleanup.disk_cleanup", None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.disk_cleanup"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.disk_cleanup"] = mod
    spec.loader.exec_module(mod)
    return mod


def _hold_state_transaction(home, entered, contender_started, release, results):
    os.environ["HERMES_HOME"] = home
    try:
        dg = _load_lib()
        with dg._state_transaction():
            entered.set()
            if not contender_started.wait(10):
                raise RuntimeError("contender did not start")
            if not release.wait(10):
                raise RuntimeError("holder was not released")
    except Exception as exc:
        results.put(repr(exc))
    else:
        results.put(None)


def _enter_state_transaction(home, started, entered, results):
    os.environ["HERMES_HOME"] = home
    try:
        dg = _load_lib()
        started.set()
        with dg._state_transaction():
            entered.set()
    except Exception as exc:
        results.put(repr(exc))
    else:
        results.put(None)


# ---------------------------------------------------------------------------
# Library tests
# ---------------------------------------------------------------------------

class TestIsSafePath:
    def test_accepts_path_under_hermes_home(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "subdir" / "file.txt"
        p.parent.mkdir()
        p.write_text("x", encoding="utf-8")
        assert dg.is_safe_path(p) is True

    def test_rejects_outside_hermes_home(self, _isolate_env):
        dg = _load_lib()
        assert dg.is_safe_path(Path("/etc/passwd")) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires privileges")
    def test_rejects_symlink_escape(self, _isolate_env):
        dg = _load_lib()
        link = _isolate_env / "escape"
        link.symlink_to(Path(__file__).resolve())
        assert dg.is_safe_path(link) is False


class TestGuessCategory:
    def test_filename_does_not_imply_ownership(self, _isolate_env):
        dg = _load_lib()
        sentinels = []
        for dirname in ("scripts", "projects", "node", "lsp", "browser-profile", "cache"):
            parent = _isolate_env / dirname
            parent.mkdir()
            p = parent / "test_durable.py"
            p.write_text("x", encoding="utf-8")
            sentinels.append(p)
            assert dg.guess_category(p) is None

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True)
        tracked_file.write_text(json.dumps([{
            "path": str(path),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 1,
        } for path in sentinels]), encoding="utf-8")
        auto, _prompt = dg.dry_run()
        summary = dg.quick()
        assert auto == []
        assert summary["deleted"] == 0
        assert all(path.exists() for path in sentinels)

    def test_system_temp_name_and_manual_category_do_not_establish_ownership(
        self, _isolate_env, _managed_tmp_root
    ):
        dg = _load_lib()
        p = _managed_tmp_root / "anything.log"
        p.write_text("x", encoding="utf-8")
        assert dg.guess_category(p) is None
        assert dg.is_safe_path(p) is False
        assert dg.track(str(p), "test", silent=True) is False

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(p),
            "category": "test",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": p.stat().st_size,
        }]), encoding="utf-8")
        auto, _prompt = dg.dry_run()
        summary = dg.quick()
        assert auto == []
        assert summary["deleted"] == 0
        assert p.read_text(encoding="utf-8") == "x"

    def test_owned_cache_file(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "cache" / "vision" / "temp_vision_images" / "image.png"
        p.parent.mkdir(parents=True)
        p.write_text("x", encoding="utf-8")
        assert dg.guess_category(p) == "temp"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires privileges")
    @pytest.mark.parametrize("escape_at_root", [True, False])
    def test_managed_cache_symlink_escape_is_never_deleted(
        self, _isolate_env, escape_at_root
    ):
        dg = _load_lib()
        managed = _isolate_env / "cache" / "vision" / "temp_vision_images"
        external = _isolate_env.parent / f"external-cache-{escape_at_root}"
        external.mkdir()
        victim = external / "victim.txt"
        victim.write_text("durable", encoding="utf-8")
        if escape_at_root:
            managed.parent.mkdir(parents=True)
            managed.symlink_to(external, target_is_directory=True)
            tracked_path = victim.resolve()
        else:
            managed.mkdir(parents=True)
            (managed / "escape").symlink_to(external, target_is_directory=True)
            tracked_path = managed / "escape" / victim.name

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(tracked_path),
            "category": "temp",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": victim.stat().st_size,
        }]), encoding="utf-8")

        auto, _prompt = dg.dry_run()
        summary = dg.quick()
        assert auto == []
        assert summary["deleted"] == 0
        assert victim.read_text(encoding="utf-8") == "durable"

    def test_skips_protected_top_level(self, _isolate_env):
        dg = _load_lib()
        logs_dir = _isolate_env / "logs"
        logs_dir.mkdir()
        p = logs_dir / "test_log.txt"
        p.write_text("x", encoding="utf-8")
        # Even though it matches test_* pattern, logs/ is excluded.
        assert dg.guess_category(p) is None

    def test_cron_subtree_categorised(self, _isolate_env):
        dg = _load_lib()
        # Only files under ``cron/output/`` are disposable run artifacts.
        output_dir = _isolate_env / "cron" / "output" / "job_123"
        output_dir.mkdir(parents=True)
        p = output_dir / "run.md"
        p.write_text("x", encoding="utf-8")
        assert dg.guess_category(p) == "cron-output"


    def test_cronjobs_top_level_not_tracked(self, _isolate_env):
        """The legacy ``cronjobs`` alias is also control-plane at the top."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cronjobs"
        cron_dir.mkdir()
        p = cron_dir / "jobs.json"
        p.write_text("[]", encoding="utf-8")
        assert dg.guess_category(p) is None

    def test_ordinary_file_returns_none(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "notes.md"
        p.write_text("x", encoding="utf-8")
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
        jobs_json.write_text('{"jobs": []}', encoding="utf-8")

        # Simulate a stale tracked.json entry from before #34840 by
        # directly writing the tracked file (track() would reject it).
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([
            {
                "path": str(jobs_json),
                "category": "cron-output",
                "timestamp": "2025-01-01T00:00:00+00:00",  # very old
                "size": 123,
            },
            {"damaged": True},
            "not-an-object",
        ]), encoding="utf-8")

        summary = dg.quick()
        assert summary["deleted"] == 0, "cron/jobs.json must not be deleted"
        assert jobs_json.exists(), "jobs.json must still exist"
        # The stale entry should have been dropped from tracking.
        remaining = json.loads(tracked_file.read_text(encoding="utf-8"))
        assert len(remaining) == 0


    def test_dry_run_omits_stale_cron_output(self, _isolate_env):
        """dry_run() should also skip stale cron-output entries."""
        dg = _load_lib()
        cron_dir = _isolate_env / "cron"
        cron_dir.mkdir()
        jobs_json = cron_dir / "jobs.json"
        jobs_json.write_text("[]", encoding="utf-8")

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(jobs_json),
            "category": "cron-output",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 123,
        }]), encoding="utf-8")

        auto, prompt = dg.dry_run()
        assert len(auto) == 0, "stale cron-output for jobs.json must not appear"
        assert len(prompt) == 0

    def test_legitimate_cron_output_still_deleted(self, _isolate_env):
        """A valid cron-output entry under cron/output/ must still be deleted."""
        dg = _load_lib()
        output_dir = _isolate_env / "cron" / "output" / "job_1"
        output_dir.mkdir(parents=True)
        run_md = output_dir / "run.md"
        run_md.write_text("x", encoding="utf-8")

        # Old enough to be deleted (>14 days)
        from datetime import datetime, timezone, timedelta
        old_ts = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()

        assert dg.track(str(run_md), "cron-output", silent=True)
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked = json.loads(tracked_file.read_text(encoding="utf-8"))
        tracked[0]["timestamp"] = old_ts
        tracked_file.write_text(json.dumps(tracked), encoding="utf-8")

        summary = dg.quick()
        if dg._secure_unlink_supported():
            assert summary["deleted"] == 1, "valid old cron-output should be deleted"
            assert not run_md.exists()
        else:
            assert summary["deleted"] == 0
            assert run_md.exists()


class TestTrackForgetQuick:
    def test_unsupported_secure_unlink_fails_closed_and_keeps_receipt(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        p = _owned_test_root(dg, _isolate_env) / "kept.txt"
        p.write_text("owned", encoding="utf-8")
        assert dg.track(str(p), "test", silent=True)
        monkeypatch.setattr(dg, "_secure_unlink_supported", lambda: False)

        summary = dg.quick()

        assert summary == {"deleted": 0, "empty_dirs": 0, "freed": 0, "errors": []}
        assert p.read_text(encoding="utf-8") == "owned"
        assert len(dg.load_tracked()) == 1

    def test_track_then_quick_deletes_test(self, _isolate_env):
        dg = _load_lib()
        p = _owned_test_root(dg, _isolate_env) / "test_a.py"
        p.write_text("x", encoding="utf-8")
        assert dg.track(str(p), "test", silent=True) is True
        summary = dg.quick()
        if dg._secure_unlink_supported():
            assert summary["deleted"] == 1
            assert not p.exists()
        else:
            assert summary["deleted"] == 0
            assert p.exists()

    def test_auto_cleanup_never_recursively_deletes_a_tracked_directory(
        self, _isolate_env
    ):
        dg = _load_lib()
        directory = (
            _isolate_env / "cache" / "vision" / "temp_vision_images" / "durable-dir"
        )
        directory.mkdir(parents=True)
        victim = directory / "victim.txt"
        victim.write_text("durable", encoding="utf-8")
        assert dg.track(str(directory), "temp", silent=True) is False

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text(json.dumps([{
            "path": str(directory),
            "category": "temp",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "size": 0,
        }]), encoding="utf-8")

        auto, _prompt = dg.dry_run()
        summary = dg.quick()
        assert auto == []
        assert summary["deleted"] == 0
        assert victim.read_text(encoding="utf-8") == "durable"

    def test_ancestor_swap_cannot_redirect_unlink(
        self, _isolate_env, monkeypatch
    ):
        dg = _load_lib()
        root = _owned_test_root(dg, _isolate_env)
        parent = root / "branch"
        parent.mkdir()
        owned = parent / "payload.txt"
        owned.write_text("owned", encoding="utf-8")
        outside = _isolate_env.parent / "outside-swap"
        outside.mkdir()
        victim = outside / owned.name
        victim.write_text("durable", encoding="utf-8")
        assert dg.track(str(owned), "test", silent=True)

        if not dg._secure_unlink_supported():
            summary = dg.quick()
            assert summary["deleted"] == 0
            assert owned.read_text(encoding="utf-8") == "owned"
            assert victim.read_text(encoding="utf-8") == "durable"
            return

        held = root / "held-branch"
        real_unlink = dg._unlink_at

        def swap_before_unlink(path, dir_fd):
            if path == owned.name:
                parent.rename(held)
                parent.symlink_to(outside, target_is_directory=True)
            return real_unlink(path, dir_fd)

        monkeypatch.setattr(dg, "_unlink_at", swap_before_unlink)
        summary = dg.quick()

        assert summary["deleted"] == 1
        assert not (held / owned.name).exists()
        assert victim.read_text(encoding="utf-8") == "durable"
        parent.unlink()


    def test_forget_removes_entry(self, _isolate_env):
        dg = _load_lib()
        p = _isolate_env / "keep.tmp"
        p.write_text("x", encoding="utf-8")
        dg.track(str(p), "other", silent=True)
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
        p.write_text("y" * 100, encoding="utf-8")
        dg.track(str(p), "other", silent=True)
        s = dg.status()
        assert s["total_tracked"] == 1
        assert len(s["top10"]) == 1
        rendered = dg.format_status(s)
        assert "other" in rendered
        assert "big.tmp" in rendered


class TestDryRun:
    def test_classifies_by_category(self, _isolate_env):
        dg = _load_lib()
        test_f = _owned_test_root(dg, _isolate_env) / "test_x.py"
        test_f.write_text("x", encoding="utf-8")
        big = _isolate_env / "big.bin"
        big.write_bytes(b"z" * 10)
        dg.track(str(test_f), "test", silent=True)
        dg.track(str(big), "other", silent=True)
        auto, prompt = dg.dry_run()
        # test → auto, other → neither (doesn't hit any rule)
        assert any(Path(i["path"]) == test_f.resolve() for i in auto)


# ---------------------------------------------------------------------------
# Plugin hooks tests
# ---------------------------------------------------------------------------

class TestPostToolCallHook:
    def test_preexisting_system_temp_file_is_never_claimed(
        self, _isolate_env, _managed_tmp_root
    ):
        pi = _load_plugin_init()
        victim = _managed_tmp_root / "victim.txt"
        victim.write_text("durable", encoding="utf-8")
        output_only = _managed_tmp_root / "output-only.txt"

        output_only.write_text("not-created-by-the-command", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": f"cat {victim}"},
            result=f"{victim}\n{output_only}\n",
            task_id="unowned", session_id="unowned", turn_id="unowned-turn",
            tool_call_id="unowned-call",
        )
        pi._on_session_end(
            session_id="unowned", task_id="unowned", turn_id="unowned-turn",
            completed=True, interrupted=False,
        )

        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert not tracked_file.exists() or json.loads(
            tracked_file.read_text(encoding="utf-8")
        ) == []
        assert victim.read_text(encoding="utf-8") == "durable"
        assert output_only.read_text(encoding="utf-8") == "not-created-by-the-command"

    def test_new_system_temp_and_git_source_are_never_claimed(
        self, _isolate_env, _managed_tmp_root
    ):
        pi = _load_plugin_init()
        repo = _managed_tmp_root / "project"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        paths = [_managed_tmp_root / "scratch.txt", repo / "src" / "service.py"]
        for index, path in enumerate(paths):
            args = {"path": str(path), "content": "created"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("created", encoding="utf-8")
            pi._on_post_tool_call(
                tool_name="write_file", args=args, result="OK", task_id="created",
                session_id="created", turn_id="created-turn",
                tool_call_id=f"created-call-{index}", status="ok",
            )

        pi._on_session_end(
            session_id="created", task_id="created", turn_id="created-turn",
            completed=True, interrupted=False,
        )

        assert [path.read_text(encoding="utf-8") for path in paths] == [
            "created", "created"
        ]
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        assert json.loads(tracked_file.read_text(encoding="utf-8")) == []

    def test_write_file_in_owned_temp_root_tracked(self, _isolate_env):
        pi = _load_plugin_init()
        p = _owned_test_root(pi.dg, _isolate_env) / "created.py"
        p.write_text("x", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="t1", session_id="s1", turn_id="turn-1",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["category"] == "test"

    def test_tracking_lock_timeout_remains_best_effort(
        self, _isolate_env, monkeypatch
    ):
        pi = _load_plugin_init()
        p = _owned_test_root(pi.dg, _isolate_env) / "created.py"
        p.write_text("x", encoding="utf-8")

        def _timeout(*_args, **_kwargs):
            raise RuntimeError("disk-cleanup state lock timed out")

        monkeypatch.setattr(pi.dg, "track", _timeout)
        pi._on_post_tool_call(
            tool_name="write_file",
            args={"path": str(p), "content": "x"},
            result="OK",
            task_id="t1", session_id="s1", turn_id="turn-1",
        )


    def test_terminal_command_picks_up_paths(self, _isolate_env):
        pi = _load_plugin_init()
        p = _owned_test_root(pi.dg, _isolate_env) / "created.log"
        p.write_text("x", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="terminal",
            args={"command": f"touch {p}"},
            result=f"created {p}\n",
            task_id="t3", session_id="s3", turn_id="turn-3",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        data = json.loads(tracked_file.read_text(encoding="utf-8"))
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
        assert not tracked_file.exists() or tracked_file.read_text(encoding="utf-8").strip() == "[]"


class TestOnSessionEndHook:
    def test_state_transaction_serializes_gateway_and_kanban_processes(
        self, _isolate_env
    ):
        ctx = multiprocessing.get_context("spawn")
        holder_entered = ctx.Event()
        contender_started = ctx.Event()
        contender_entered = ctx.Event()
        release = ctx.Event()
        results = ctx.Queue()
        home = str(_isolate_env)

        holder = ctx.Process(
            target=_hold_state_transaction,
            args=(home, holder_entered, contender_started, release, results),
        )
        contender = ctx.Process(
            target=_enter_state_transaction,
            args=(home, contender_started, contender_entered, results),
        )
        holder.start()
        assert holder_entered.wait(10)
        contender.start()
        assert contender_started.wait(10)
        assert not contender_entered.wait(2), (
            "a Kanban worker entered the profile-state transaction while the "
            "gateway process still owned it"
        )
        release.set()
        assert contender_entered.wait(10)
        holder.join(10)
        contender.join(10)
        assert holder.exitcode == 0
        assert contender.exitcode == 0
        assert [results.get(timeout=10) for _ in range(2)] == [None, None]

    def test_each_turn_runs_aged_owned_root_retention(self, _isolate_env):
        from datetime import datetime, timedelta, timezone

        pi = _load_plugin_init()
        path = (
            _isolate_env / "cache" / "vision" / "temp_vision_images" / "old.png"
        )
        path.parent.mkdir(parents=True)
        path.write_text("old", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path)}, result="OK",
            task_id="creator", session_id="session", turn_id="creator-turn",
        )
        tracked_file = _isolate_env / "disk-cleanup" / "tracked.json"
        tracked = json.loads(tracked_file.read_text(encoding="utf-8"))
        tracked[0]["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(days=8)
        ).isoformat()
        tracked_file.write_text(json.dumps(tracked), encoding="utf-8")

        pi._on_session_end(
            task_id="other", session_id="session", turn_id="ordinary-bot-turn"
        )
        assert path.exists() is (not pi.dg._secure_unlink_supported())

    def test_only_cleans_the_turn_that_ended(self, _isolate_env):
        pi = _load_plugin_init()
        root = _owned_test_root(pi.dg, _isolate_env)
        paths = [root / "turn-a.txt", root / "turn-b.txt"]
        for turn_id, p in zip(("turn-a", "turn-b"), paths):
            p.write_text("x", encoding="utf-8")
            pi._on_post_tool_call(
                tool_name="write_file",
                args={"path": str(p), "content": "x"},
                result="OK",
                task_id="task", session_id="session", turn_id=turn_id,
            )

        pi._on_session_end(
            session_id="session", task_id="task", turn_id="turn-a",
            completed=True, interrupted=False,
        )
        assert paths[0].exists() is (not pi.dg._secure_unlink_supported())
        assert paths[1].exists(), "ending one turn must not clean another turn's file"

        pi._on_session_end(
            session_id="session", task_id="task", turn_id="turn-b",
            completed=True, interrupted=False,
        )
        assert paths[1].exists() is (not pi.dg._secure_unlink_supported())

    def test_old_turn_cannot_delete_new_generation_at_same_path(self, _isolate_env):
        pi = _load_plugin_init()
        root = _owned_test_root(pi.dg, _isolate_env)
        path = root / "shared.txt"

        path.write_text("turn-a", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path)}, result="OK",
            task_id="task-a", session_id="session-a", turn_id="turn-a",
        )
        first_inode = path.stat().st_ino
        path.rename(root / "held-turn-a.txt")
        path.write_text("turn-b", encoding="utf-8")
        assert path.stat().st_ino != first_inode
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path)}, result="OK",
            task_id="task-b", session_id="session-b", turn_id="turn-b",
        )

        pi._on_session_end(task_id="task-a", session_id="session-a", turn_id="turn-a")
        assert path.read_text(encoding="utf-8") == "turn-b"

        pi._on_session_end(task_id="task-b", session_id="session-b", turn_id="turn-b")
        assert path.exists() is (not pi.dg._secure_unlink_supported())

    def test_old_turn_cannot_delete_in_place_rewrite_at_same_path(self, _isolate_env):
        pi = _load_plugin_init()
        root = _owned_test_root(pi.dg, _isolate_env)
        path = root / "shared-inode.txt"

        path.write_text("turn-a", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path)}, result="OK",
            task_id="task-a", session_id="session-a", turn_id="turn-a",
        )
        first_inode = path.stat().st_ino

        path.write_text("turn-b-has-different-size", encoding="utf-8")
        assert path.stat().st_ino == first_inode
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path)}, result="OK",
            task_id="task-b", session_id="session-b", turn_id="turn-b",
        )

        pi._on_session_end(task_id="task-a", session_id="session-a", turn_id="turn-a")
        assert path.read_text(encoding="utf-8") == "turn-b-has-different-size"

        pi._on_session_end(task_id="task-b", session_id="session-b", turn_id="turn-b")
        assert path.exists() is (not pi.dg._secure_unlink_supported())

    def test_same_turn_id_is_isolated_by_profile(
        self, _isolate_env, tmp_path, monkeypatch
    ):
        pi = _load_plugin_init()
        root_a = _owned_test_root(pi.dg, _isolate_env)
        path_a = root_a / "profile-a.txt"
        path_a.write_text("a", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path_a)}, result="OK",
            task_id="task", session_id="session", turn_id="shared-turn",
        )

        home_b = tmp_path / "profile-b"
        home_b.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        root_b = _owned_test_root(pi.dg, home_b)
        path_b = root_b / "profile-b.txt"
        path_b.write_text("b", encoding="utf-8")
        pi._on_post_tool_call(
            tool_name="write_file", args={"path": str(path_b)}, result="OK",
            task_id="task", session_id="session", turn_id="shared-turn",
        )
        pi._on_session_end(
            task_id="task", session_id="session", turn_id="shared-turn"
        )
        assert path_b.exists() is (not pi.dg._secure_unlink_supported())
        assert path_a.read_text(encoding="utf-8") == "a"

        monkeypatch.setenv("HERMES_HOME", str(_isolate_env))
        pi._on_session_end(
            task_id="task", session_id="session", turn_id="shared-turn"
        )
        assert path_a.exists() is (not pi.dg._secure_unlink_supported())

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
        cfg_path.write_text(
            yaml.safe_dump({"plugins": {"enabled": list(names)}}), encoding="utf-8")

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

    def test_enabled_plugin_registers_real_hooks_and_slash_command(self, _isolate_env):
        self._write_enabled_config(_isolate_env, ["disk-cleanup"])
        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()

        loaded = mgr._plugins["disk-cleanup"]
        assert loaded.enabled
        assert loaded.hooks_registered == ["post_tool_call", "on_session_end"]
        assert loaded.commands_registered == ["disk-cleanup"]
        assert "Dry-run preview" in mgr._plugin_commands["disk-cleanup"]["handler"](
            "dry-run"
        )

    def test_real_plugin_manager_dispatches_generation_bound_lifecycle(
        self, _isolate_env
    ):
        self._write_enabled_config(_isolate_env, ["disk-cleanup"])
        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        loaded = mgr._plugins["disk-cleanup"]
        assert loaded.module is not None
        path = _owned_test_root(loaded.module.dg, _isolate_env) / "dispatched.txt"
        path.write_text("owned", encoding="utf-8")

        mgr.invoke_hook(
            "post_tool_call", tool_name="write_file", args={"path": str(path)},
            result="OK", task_id="task", session_id="session", turn_id="turn",
            tool_call_id="call",
        )
        assert path.exists()
        mgr.invoke_hook(
            "on_session_end", task_id="task", session_id="session", turn_id="turn",
            completed=True, interrupted=False,
        )
        assert path.exists() is (not loaded.module.dg._secure_unlink_supported())


    def test_disabled_beats_enabled(self, _isolate_env):
        """plugins.disabled wins even if the plugin is also in plugins.enabled."""
        import yaml
        cfg_path = _isolate_env / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "plugins": {
                "enabled": ["disk-cleanup"],
                "disabled": ["disk-cleanup"],
            }
        }), encoding="utf-8")
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
