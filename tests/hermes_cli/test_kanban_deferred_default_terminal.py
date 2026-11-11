"""Deferred target resolution: boundary scrub, no runtime pin, then default startup."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolated_managed_loader(monkeypatch):
    from hermes_cli import managed_scope, env_loader
    monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", set())
    managed_scope.invalidate_managed_cache()
    yield
    managed_scope.invalidate_managed_cache()


@pytest.mark.parametrize("target_mode,expected,managed_mode", [
    ("real", "real", None), (None, "auto", None), (None, "real", "real")])
@pytest.mark.parametrize("runtime", [None, 900])
def test_deferred_default_boundary_and_later_startup(tmp_path, monkeypatch, target_mode, expected, runtime, managed_mode):
    from hermes_cli import config, env_loader, kanban_db as kb, kanban_db_dispatch as dispatch, profiles
    from tools import process_registry
    import hermes_constants

    root = tmp_path / ".hermes"
    launch = root / "profiles" / "secondary"
    launch.mkdir(parents=True)
    (root / "config.yaml").write_text("terminal:\n  backend: local\n" + (
        f"  home_mode: {target_mode}\n" if target_mode else ""))
    (root / ".env").write_text("PROFILE_MARKER=default\n")
    # Explicit profile YAML wins over managed dotenv. Leave the launch mode
    # unpinned for this control so the real loader can install administrator policy.
    (launch / "config.yaml").write_text("terminal:\n  backend: docker\n" + (
        "  home_mode: profile\n" if not managed_mode else ""))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", str(launch))
    for key, value in {"HOME": str(tmp_path), "HERMES_HOME": str(launch), "HERMES_KANBAN_HOME": str(root),
                       "TERMINAL_ENV": "docker", "TERMINAL_DOCKER_IMAGE": "stale", "terminal_home_mode": "profile",
                       "TERMINAL_HOME_MODE": "profile", "TERMINAL_DOCKER_FORWARD_ENV": '["SECRET"]',
                       "terminal_docker_image": "stale-lower", "TERMINAL_TIMEOUT": "17",
                       "TERMINAL_MAX_FOREGROUND_TIMEOUT": "19", "TERMINAL_UNRELATED": "keep"}.items():
        monkeypatch.setenv(key, value)
    if managed_mode:
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / ".env").write_text(f"TERMINAL_HOME_MODE={managed_mode}\n")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        env_loader.load_hermes_dotenv(hermes_home=launch, load_external_secrets=False)
    before = dict(os.environ)
    captured = {}
    def popen(argv, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pid=424242)
    real_resolver = profiles.resolve_profile_env
    def defer(name):
        assert name == "default"
        raise FileNotFoundError("default destination is deferred at spawn")
    monkeypatch.setattr(profiles, "resolve_profile_env", defer)
    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, argv: argv)
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda env: env)
    monkeypatch.setattr(dispatch, "_open_worker_log", lambda task, board: open(os.devnull, "wb"))
    monkeypatch.setattr(dispatch.subprocess, "Popen", popen)
    task = kb.Task(id="deferred", title="deferred", body=None, assignee="default", status="claimed", priority=0,
                   created_by=None, created_at=0, started_at=None, completed_at=None, workspace_kind="dir",
                   workspace_path=None, claim_lock="claim", claim_expires=None, tenant=None,
                   max_runtime_seconds=runtime)
    dispatch._default_spawn(task, str(workspace))
    env = captured["env"]
    canonical = {"TERMINAL_ENV", "TERMINAL_DOCKER_IMAGE", "TERMINAL_DOCKER_FORWARD_ENV", "TERMINAL_HOME_MODE"}
    if managed_mode:
        # A newer upstream guard exempts administrator policy from launch residue.
        canonical.remove("TERMINAL_HOME_MODE")
        assert env["TERMINAL_HOME_MODE"] == managed_mode
    assert not {key for key in env if key.upper() in canonical}
    assert env["TERMINAL_CWD"] == captured["cwd"] == str(workspace)
    assert env["HERMES_PROFILE"] == "default" and env["HERMES_KANBAN_TASK"] == "deferred"
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == "claim" and env["HERMES_SESSION_SOURCE"] == "kanban"
    assert env["TERMINAL_UNRELATED"] == "keep"
    if runtime is None:
        assert "TERMINAL_TIMEOUT" not in env  # canonical launch policy, not a task pin
        assert env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] == "19"  # noncanonical: no broadened scrub
    else:
        assert env["TERMINAL_TIMEOUT"] == env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] == "870"
    monkeypatch.setattr(profiles, "resolve_profile_env", real_resolver)
    # HERMES_PROFILE / CLI resolution is deliberately delayed, not mocked startup success.
    with patch.dict(os.environ, env, clear=True):
        home = real_resolver(env["HERMES_PROFILE"])
        assert Path(home) == root
        os.environ["HERMES_HOME"] = str(home)
        monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", str(home))
        env_loader.load_hermes_dotenv(hermes_home=Path(home), load_external_secrets=False)
        config.apply_terminal_config_to_env()
        assert os.environ["TERMINAL_ENV"] == "local"
        assert os.environ["TERMINAL_HOME_MODE"] == expected
        assert os.environ["PROFILE_MARKER"] == "default"
        assert json.loads(os.environ["TERMINAL_DOCKER_FORWARD_ENV"]) == []
        assert os.environ["TERMINAL_DOCKER_IMAGE"] != "stale"
    assert dict(os.environ) == before
