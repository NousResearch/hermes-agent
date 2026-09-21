"""A kanban worker spawned for profile B builds its env under B's terminal scope, not the ambient one.

``_default_spawn`` bound only a secret scope, and only when ``is_multiplex_active()``. The dispatcher
runs detached from any turn, so ``build_subprocess_env`` (terminal ``env_passthrough`` resolution)
and ``_resolve_worker_cli_toolsets`` read whatever ``TERMINAL_*`` the host process happened to carry
— the LAUNCH profile's policy applied to another tenant's worker.
"""
import subprocess  # noqa: F401 — imported so a stray real spawn is obvious in a traceback

import pytest

from hermes_cli import kanban_db_dispatch
from tools.terminal_scope import get_terminal_scope


class _StopSpawn(Exception):
    """Abort ``_default_spawn`` at the env-build seam so no worker process is created."""


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
    """The seam both dispatch-side readers use: toolset resolution and the spawn-env build."""
    with kanban_db_dispatch._worker_profile_scope(str(profile_b)):
        scope = get_terminal_scope() or {}
    assert scope.get("TERMINAL_ENV") == "docker"
    assert scope.get("TERMINAL_DOCKER_IMAGE") == "b-image", (
        f"worker inherited the launch profile's terminal policy: {scope}")


def test_default_spawn_builds_the_worker_env_under_the_assigned_profiles_scope(
        profile_b, tmp_path, monkeypatch):
    from hermes_cli.kanban_db import Task

    seen: list[dict] = []

    import tools.environments.local as local_env

    def _capture(*_args, **_kwargs):
        seen.append(dict(get_terminal_scope() or {}))
        raise _StopSpawn  # the invariant is observed; nothing must actually spawn

    monkeypatch.setattr(local_env, "build_subprocess_env", _capture)

    task = Task(
        id="t1", title="t", body=None, assignee="b", status="claimed", priority=0,
        created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="dir", workspace_path=None, claim_lock=None, claim_expires=None,
        tenant=None)
    with pytest.raises(_StopSpawn):
        kanban_db_dispatch._default_spawn(task, str(tmp_path / "ws"))

    assert seen, "_default_spawn never reached build_subprocess_env"
    assert seen[0].get("TERMINAL_ENV") == "docker", (
        f"spawn env built under the launch profile's terminal policy: {seen[0]}")
