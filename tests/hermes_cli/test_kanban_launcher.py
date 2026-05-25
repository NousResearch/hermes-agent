"""Tests for hermes_cli.kanban_launcher — the dedicated worker spawn module.

These tests cover the preparation and validation layer independently from
the SQLite DB, verifying that:
- Environment validation catches missing assignees.
- build_worker_env builds the correct env dict from provided paths.
- build_worker_cmd constructs the correct argv.
- prepare_launch assembles a valid LaunchContext.
- execute_launch calls Popen with the right arguments.
- _default_spawn wires validation + preparation + execution end-to-end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_launcher as kl
from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Minimal Task fixture
# ---------------------------------------------------------------------------

def _make_task(**overrides) -> kb.Task:
    defaults = dict(
        id="t_test01",
        title="test task",
        body=None,
        assignee="test-profile",
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    defaults.update(overrides)
    return kb.Task(**defaults)


# ---------------------------------------------------------------------------
# SpawnValidationError
# ---------------------------------------------------------------------------

class TestSpawnValidationError:
    def test_carries_task_id(self):
        err = kl.SpawnValidationError("boom", task_id="t_abc")
        assert err.task_id == "t_abc"
        assert "boom" in str(err)

    def test_default_task_id_is_empty(self):
        err = kl.SpawnValidationError("oops")
        assert err.task_id == ""


# ---------------------------------------------------------------------------
# validate_spawn_env
# ---------------------------------------------------------------------------

class TestValidateSpawnEnv:
    def test_raises_for_missing_assignee(self):
        task = _make_task(assignee=None)
        with pytest.raises(kl.SpawnValidationError, match="no assignee"):
            kl.validate_spawn_env(task, "/tmp/ws")

    def test_raises_for_empty_task_id(self):
        task = _make_task(id="", assignee="someone")
        with pytest.raises(kl.SpawnValidationError, match="no id"):
            kl.validate_spawn_env(task, "/tmp/ws")

    def test_passes_for_scratch_task_with_nonexistent_workspace(self):
        task = _make_task()
        # scratch tasks may not have a real directory yet — no OSError
        kl.validate_spawn_env(task, "/tmp/does-not-exist-xyz")

    def test_passes_for_valid_task_and_existing_workspace(self, tmp_path):
        task = _make_task()
        kl.validate_spawn_env(task, str(tmp_path))


# ---------------------------------------------------------------------------
# build_worker_env
# ---------------------------------------------------------------------------

class TestBuildWorkerEnv:
    def test_sets_required_kanban_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        # Stub out profile resolution to avoid needing a real profile dir.
        monkeypatch.setattr(
            "hermes_cli.kanban_launcher.build_worker_env.__module__",
            kl.__name__,
        )
        # Patch the profile helpers so no filesystem access is needed.
        import hermes_cli.kanban_launcher as _kl  # re-import for monkeypatching
        monkeypatch.setattr(
            "hermes_cli.profiles.normalize_profile_name",
            lambda name: name,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.resolve_profile_env",
            lambda _name: str(tmp_path / ".hermes"),
        )

        task = _make_task(id="t_env1", assignee="builder")
        env = kl.build_worker_env(
            task,
            "/workspace/path",
            resolved_db_path="/path/to/kanban.db",
            resolved_workspaces_root="/path/to/workspaces",
            resolved_board="myboard",
            base_env={},
        )

        assert env["HERMES_KANBAN_TASK"] == "t_env1"
        assert env["HERMES_KANBAN_WORKSPACE"] == "/workspace/path"
        assert env["HERMES_KANBAN_DB"] == "/path/to/kanban.db"
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == "/path/to/workspaces"
        assert env["HERMES_KANBAN_BOARD"] == "myboard"
        assert env["HERMES_PROFILE"] == "builder"

    def test_sets_branch_name_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _n: str(tmp_path))
        task = _make_task(branch_name="feature/foo")
        env = kl.build_worker_env(
            task, "/ws",
            resolved_db_path="/db",
            resolved_workspaces_root="/wsr",
            resolved_board="default",
            base_env={},
        )
        assert env["HERMES_KANBAN_BRANCH"] == "feature/foo"

    def test_sets_tenant_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _n: str(tmp_path))
        task = _make_task(tenant="acme-corp")
        env = kl.build_worker_env(
            task, "/ws",
            resolved_db_path="/db",
            resolved_workspaces_root="/wsr",
            resolved_board="default",
            base_env={},
        )
        assert env["HERMES_TENANT"] == "acme-corp"

    def test_omits_optional_vars_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _n: str(tmp_path))
        task = _make_task()  # no branch_name, no tenant, no claim_lock
        env = kl.build_worker_env(
            task, "/ws",
            resolved_db_path="/db",
            resolved_workspaces_root="/wsr",
            resolved_board="default",
            base_env={},
        )
        assert "HERMES_KANBAN_BRANCH" not in env
        assert "HERMES_TENANT" not in env
        assert "HERMES_KANBAN_CLAIM_LOCK" not in env

    def test_raises_terminal_timeout_when_runtime_cap_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _n: str(tmp_path))
        task = _make_task(max_runtime_seconds=3600)
        env = kl.build_worker_env(
            task, "/ws",
            resolved_db_path="/db",
            resolved_workspaces_root="/wsr",
            resolved_board="default",
            base_env={},  # no TERMINAL_TIMEOUT present
        )
        # Should be set to 3600 - 30 (grace) = 3570
        assert "TERMINAL_TIMEOUT" in env
        assert int(env["TERMINAL_TIMEOUT"]) == 3570


# ---------------------------------------------------------------------------
# build_worker_cmd
# ---------------------------------------------------------------------------

class TestBuildWorkerCmd:
    def test_contains_profile_and_chat(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: False)
        task = _make_task(id="t_cmd1", assignee="myprofile")
        cmd = kl.build_worker_cmd(task)
        assert "-p" in cmd
        assert "myprofile" in cmd
        assert "chat" in cmd
        assert "-q" in cmd
        assert "work kanban task t_cmd1" in cmd

    def test_includes_accept_hooks(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: False)
        task = _make_task()
        cmd = kl.build_worker_cmd(task)
        assert "--accept-hooks" in cmd

    def test_injects_kanban_worker_skill_when_available(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: True)
        task = _make_task()
        cmd = kl.build_worker_cmd(task)
        idx = cmd.index("--skills")
        assert cmd[idx + 1] == "kanban-worker"

    def test_skips_kanban_worker_when_not_available(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: False)
        task = _make_task()
        cmd = kl.build_worker_cmd(task)
        assert "--skills" not in cmd

    def test_injects_model_override(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: False)
        task = _make_task(model_override="claude-opus-4-7")
        cmd = kl.build_worker_cmd(task)
        assert "-m" in cmd
        assert "claude-opus-4-7" in cmd

    def test_dedupes_kanban_worker_from_task_skills(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: True)
        task = _make_task(skills=["kanban-worker", "my-skill"])
        cmd = kl.build_worker_cmd(task)
        # kanban-worker appears exactly once
        assert cmd.count("kanban-worker") == 1
        assert "my-skill" in cmd


# ---------------------------------------------------------------------------
# LaunchContext
# ---------------------------------------------------------------------------

class TestLaunchContext:
    def test_default_fields(self):
        ctx = kl.LaunchContext(task_id="t_1", profile="coder", workspace="/ws")
        assert ctx.env == {}
        assert ctx.cmd == []
        assert ctx.log_path == ""
        assert ctx.board == "default"

    def test_dataclass_stores_all_fields(self):
        ctx = kl.LaunchContext(
            task_id="t_2",
            profile="p",
            workspace="/w",
            env={"A": "1"},
            cmd=["hermes", "chat"],
            log_path="/tmp/t_2.log",
            board="myboard",
        )
        assert ctx.task_id == "t_2"
        assert ctx.env["A"] == "1"
        assert ctx.board == "myboard"


# ---------------------------------------------------------------------------
# prepare_launch
# ---------------------------------------------------------------------------

class TestPrepareLaunch:
    def test_returns_launch_context(self, tmp_path, monkeypatch):
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _n: str(tmp_path))
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: False)
        monkeypatch.setattr("hermes_cli.kanban_launcher.worker_log_rotation_config", lambda: (0, 0))

        log_dir = tmp_path / "logs"
        task = _make_task(id="t_prep1", assignee="builder")
        ctx = kl.prepare_launch(
            task, "",
            resolved_db_path=str(tmp_path / "kanban.db"),
            resolved_workspaces_root=str(tmp_path / "workspaces"),
            resolved_board="default",
            resolved_log_dir=log_dir,
        )

        assert isinstance(ctx, kl.LaunchContext)
        assert ctx.task_id == "t_prep1"
        assert ctx.profile == "builder"
        assert ctx.board == "default"
        assert ctx.env["HERMES_KANBAN_TASK"] == "t_prep1"
        assert "chat" in ctx.cmd
        assert log_dir.exists()
        assert ctx.log_path.endswith("t_prep1.log")

    def test_raises_validation_error_for_missing_assignee(self, tmp_path):
        task = _make_task(assignee=None)
        with pytest.raises(kl.SpawnValidationError):
            kl.prepare_launch(
                task, "",
                resolved_db_path="/db",
                resolved_workspaces_root="/wsr",
                resolved_board="default",
                resolved_log_dir=tmp_path / "logs",
            )


# ---------------------------------------------------------------------------
# execute_launch
# ---------------------------------------------------------------------------

class TestExecuteLaunch:
    def test_calls_popen_and_returns_pid(self, tmp_path, monkeypatch):
        captured = {}

        class FakeProc:
            pid = 55555

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return FakeProc()

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        log_path = tmp_path / "t_exec.log"
        ctx = kl.LaunchContext(
            task_id="t_exec",
            profile="p",
            workspace=str(tmp_path),
            env={"A": "1"},
            cmd=["/usr/bin/hermes", "chat", "-q", "work kanban task t_exec"],
            log_path=str(log_path),
            board="default",
        )
        pid = kl.execute_launch(ctx)
        assert pid == 55555
        assert captured["cmd"] == ctx.cmd
        assert captured["kwargs"]["env"] == {"A": "1"}

    def test_raises_on_missing_executable(self, tmp_path, monkeypatch):
        import subprocess as _subprocess

        def bad_popen(*_args, **_kwargs):
            raise FileNotFoundError("hermes not found")

        monkeypatch.setattr("subprocess.Popen", bad_popen)
        log_path = tmp_path / "t_bad.log"
        ctx = kl.LaunchContext(
            task_id="t_bad",
            profile="p",
            workspace="",
            env={},
            cmd=["hermes-does-not-exist"],
            log_path=str(log_path),
            board="default",
        )
        with pytest.raises(RuntimeError, match="executable not found"):
            kl.execute_launch(ctx)


# ---------------------------------------------------------------------------
# _default_spawn integration (end-to-end without real Popen)
# ---------------------------------------------------------------------------

class TestDefaultSpawnIntegration:
    """Smoke-tests for the full _default_spawn path with Popen stubbed out."""

    def test_returns_pid(self, tmp_path, monkeypatch):
        self._patch_deps(tmp_path, monkeypatch)
        captured = {}

        class FakeProc:
            pid = 12345

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return FakeProc()

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="integration test", assignee="worker-profile")
            task = kb.get_task(conn, tid)
            workspace = kb.resolve_workspace(task)

        pid = kl._default_spawn(task, str(workspace))
        assert pid == 12345
        assert captured["env"]["HERMES_KANBAN_TASK"] == tid
        assert captured["env"]["HERMES_PROFILE"] == "worker-profile"

    def test_raises_for_missing_assignee(self, tmp_path, monkeypatch):
        self._patch_deps(tmp_path, monkeypatch)
        task = _make_task(id="t_noass", assignee=None)
        with pytest.raises(ValueError, match="no assignee"):
            kl._default_spawn(task, str(tmp_path))

    @staticmethod
    def _patch_deps(tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda n: n)
        monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda _n: str(hermes_home))
        monkeypatch.setattr("hermes_cli.kanban_launcher._resolve_hermes_argv", lambda: ["/usr/bin/hermes"])
        monkeypatch.setattr("hermes_cli.kanban_launcher._kanban_worker_skill_available", lambda _h: False)
