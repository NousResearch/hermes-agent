"""Every gateway respawn boundary must strip an inherited delegate-child marker: a gateway
is never a delegate child, and a marker inherited from the fenced shell that (re)started it
fences the gateway's embedded dispatcher — every board write fails with PermissionError.
See #136081."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER


class TestStartupScrubHelper:
    def test_strips_both_markers_and_warns(self, monkeypatch, caplog):
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "/tmp/board")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
        with caplog.at_level("WARNING", logger="hermes_cli.gateway_restart_env"):
            assert scrub_delegate_child_env_markers(os.environ) is True
        assert DELEGATED_CHILD_ENV_MARKER not in os.environ
        assert "HERMES_KANBAN_TASK" not in os.environ
        assert any("delegate child" in r.getMessage() for r in caplog.records)

    def test_clean_environment_is_a_silent_noop(self, monkeypatch, caplog):
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        with caplog.at_level("WARNING", logger="hermes_cli.gateway_restart_env"):
            assert scrub_delegate_child_env_markers(os.environ) is False
        assert not caplog.records

    def test_empty_marker_value_is_left_alone(self, monkeypatch):
        # An empty value never fences (bool(os.environ.get(...)) is False); the scrub
        # keeps the same semantics instead of popping empty-string noise.
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "")
        assert scrub_delegate_child_env_markers(os.environ) is False
        assert os.environ.get(DELEGATED_CHILD_ENV_MARKER) == ""


class TestRestartWatcherEnv:
    @pytest.fixture(autouse=True)
    def _standalone_profile_home(self, tmp_path, monkeypatch):
        """A standalone named-profile home: the served_profile_child_env branch, whose
        returned env inherits the (possibly fenced) process environment."""
        from agent import secret_scope as ss

        ss.set_multiplex_active(False)
        home = tmp_path / "profile"
        home.mkdir()
        (home / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        yield
        ss.set_multiplex_active(False)

    def test_watcher_env_strips_inherited_delegate_markers(self, monkeypatch):
        """Markers present in the serving environment must not reach the watcher/respawn."""
        from gateway.run_shutdown import GatewayShutdownMixin

        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "/tmp/board")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
        env = GatewayShutdownMixin._restart_watcher_env()
        assert DELEGATED_CHILD_ENV_MARKER not in env
        assert "HERMES_KANBAN_TASK" not in env
        assert "_HERMES_GATEWAY" not in env

    def test_watcher_env_scrubs_markers_from_the_served_env(self, monkeypatch):
        """Even a served env that itself carries the markers is scrubbed at the boundary."""
        from gateway.run_shutdown import GatewayShutdownMixin

        monkeypatch.setattr(
            "tools.environments.local.served_profile_child_env",
            lambda **kw: {DELEGATED_CHILD_ENV_MARKER: "/b", "HERMES_KANBAN_TASK": "t", "_HERMES_GATEWAY": "1"},
        )
        monkeypatch.setattr("gateway.config_loader.drop_bridged_env", lambda e: e)
        env = GatewayShutdownMixin._restart_watcher_env()
        assert DELEGATED_CHILD_ENV_MARKER not in env
        assert "HERMES_KANBAN_TASK" not in env


class TestWindowsDetachedSpawn:
    def test_spawn_detached_child_env_strips_delegate_markers(self, monkeypatch, tmp_path):
        from hermes_cli import gateway_windows as gw

        captured = {}

        class _FakeProc:
            pid = 4242

        def _fake_popen(argv, **kwargs):
            captured["env"] = kwargs["env"]
            return _FakeProc()

        monkeypatch.setattr(gw, "_assert_windows", lambda: None)
        monkeypatch.setattr(gw, "_build_gateway_argv", lambda home: (["hermes"], str(tmp_path), {}))
        monkeypatch.setattr(gw, "_hermes_home", lambda: tmp_path)
        monkeypatch.setattr(gw, "windows_detach_flags", lambda: 0)
        monkeypatch.setattr(
            "tools.environments.local.served_profile_child_env",
            lambda **kw: {DELEGATED_CHILD_ENV_MARKER: "/b", "HERMES_KANBAN_TASK": "t"},
        )
        monkeypatch.setattr(gw.subprocess, "Popen", _fake_popen)

        assert gw._spawn_detached(home=tmp_path) == 4242
        assert DELEGATED_CHILD_ENV_MARKER not in captured["env"]
        assert "HERMES_KANBAN_TASK" not in captured["env"]


class TestPosixWatcherSource:
    def test_watcher_pops_inherited_markers_before_respawn(self):
        """The POSIX watcher respawns with its own os.environ when the overlay is empty,
        so it must pop the inherited markers up front (static contract lock)."""
        import inspect

        from hermes_cli import gateway as gw

        src = inspect.getsource(gw._spawn_gateway_restart_watcher)
        assert "os.environ.pop" in src
        assert "DELEGATED_CHILD_ENV_MARKER" in src
