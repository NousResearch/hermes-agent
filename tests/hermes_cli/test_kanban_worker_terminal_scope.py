"""A kanban worker spawned for profile B must not inherit the DISPATCHER's credentials.

``_default_spawn`` gated the credential scrub on ``is_multiplex_active()``, so on a single-profile
host — the common Kanban deployment — B's worker env was byte-identical to the dispatcher's: its
``OPENAI_API_KEY`` and anything systemd injected crossed straight into another profile's worker.
The authority test is "does this worker act for a ROUTED home", exactly as ``served_profile_child_env``
decides it. The secret scope bound around the env build is what supplies B's OWN values for the
dispatcher's declared ``terminal.env_passthrough`` names.
"""
import pytest

from hermes_cli import kanban_db_dispatch
from tools.terminal_scope import get_terminal_scope


class _StopSpawn(Exception):
    """Abort ``_default_spawn`` after the env is built so no worker process is created."""


@pytest.fixture
def profile_b(tmp_path, monkeypatch):
    """A fake HOME so ``profiles/`` never resolves to the live install (see hermes-agent-dev)."""
    launch = tmp_path / "fakehome" / ".hermes"
    served = launch / "profiles" / "b"
    served.mkdir(parents=True)
    (served / "config.yaml").write_text(
        "terminal:\n  backend: docker\n  docker_image: b-image\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("TERMINAL_ENV", "local")  # ambient launch-profile policy
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "launch-image")
    return served


def test_worker_profile_scope_installs_the_assigned_profiles_terminal_policy(profile_b):
    """The toolset-resolution seam (``bind_home=True``) reads config under B's own policy."""
    with kanban_db_dispatch._worker_profile_scope(str(profile_b)):
        scope = get_terminal_scope() or {}
    assert scope.get("TERMINAL_ENV") == "docker"
    assert scope.get("TERMINAL_DOCKER_IMAGE") == "b-image", (
        f"worker inherited the launch profile's terminal policy: {scope}")


def _spawn_env_for_profile_b(monkeypatch, tmp_path):
    """Run ``_default_spawn`` far enough to capture the worker env, never spawning anything."""
    from hermes_cli.kanban_db import Task
    from tools import process_registry

    captured: list[dict] = []

    def _capture(env):
        captured.append(dict(env))
        raise _StopSpawn

    monkeypatch.setattr(process_registry, "systemd_user_bus_env", _capture)

    task = Task(
        id="t1", title="t", body=None, assignee="b", status="claimed", priority=0,
        created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="dir", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None)
    with pytest.raises(_StopSpawn):
        kanban_db_dispatch._default_spawn(task, str(tmp_path / "ws"))
    assert captured, "_default_spawn never built a worker env"
    return captured[0]


def test_worker_for_another_profile_never_inherits_the_dispatchers_credentials(
        profile_b, tmp_path, monkeypatch):
    """Single-profile host (multiplex OFF) — the case the gateway-wide flag left unprotected."""
    monkeypatch.setenv("OPENAI_API_KEY", "dispatcher-launch-key")
    env = _spawn_env_for_profile_b(monkeypatch, tmp_path)
    assert "OPENAI_API_KEY" not in env, (
        "B's worker inherited the dispatcher's provider credential")


def test_launch_profiles_own_worker_keeps_its_credentials(tmp_path, monkeypatch):
    """Control: a worker for the LAUNCH profile is not acting for another tenant."""
    launch = tmp_path / "fakehome" / ".hermes"
    (launch / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("OPENAI_API_KEY", "dispatcher-launch-key")

    from hermes_cli.kanban_db import Task
    from tools import process_registry

    captured: list[dict] = []

    def _capture(env):
        captured.append(dict(env))
        raise _StopSpawn

    monkeypatch.setattr(process_registry, "systemd_user_bus_env", _capture)
    task = Task(
        id="t1", title="t", body=None, assignee="default", status="claimed", priority=0,
        created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="dir", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None)
    with pytest.raises(_StopSpawn):
        kanban_db_dispatch._default_spawn(task, str(tmp_path / "ws"))

    assert captured and captured[0].get("OPENAI_API_KEY") == "dispatcher-launch-key"
