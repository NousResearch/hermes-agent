"""End-to-end: a real ``_default_spawn`` invocation produces a sandboxed argv.

This is the dispatcher-integration half of the t_ff7d30cc acceptance.
The unit tests in ``test_worker_isolate.py`` exercise the pure argv
builder, the live synthetic-repo harness exercises the launcher, and
the smoke test exercises the launcher against the real production
tree -- but only THIS test asserts that ``hermes_cli.kanban_db_dispatch.
_default_spawn`` actually injects the wrap.

Specifically:

1. With a worktree task and a real git checkout as the agent home,
   ``_default_spawn`` must wrap the worker argv with ``unshare -Urm``
   + the launcher module + ``--``.
2. With a scratch task, the wrap must NOT happen (we never carve out
   an arbitrary cwd).
3. With the kill switch set, the wrap must NOT happen even for a
   worktree task.

The test is auto-skipped when unprivileged userns is unavailable so
it can run on CI hosts that don't ship the kernel feature.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(shutil.which("unshare") is None, reason="unshare not on PATH"),
    pytest.mark.skipif(
        subprocess.run(
            ["unshare", "--user", "--map-root-user", "true"],
            capture_output=True, timeout=5,
        ).returncode != 0,
        reason="unprivileged user namespace unavailable on this host",
    ),
]


def _seed_repo(path: Path) -> None:
    """Make ``path`` a git repo on ``canonical-deploy`` with one commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(path), "init", "-q", "-b", "canonical-deploy", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--allow-empty", "-q", "-m", "initial"],
        check=True,
    )


def _make_task(kb, *, workspace_kind: str, workspace_path: str | None, task_id: str = "t_iso_e2e"):
    return kb.Task(
        id=task_id,
        title="iso e2e",
        body=None,
        assignee="friday",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        branch_name=f"wt/{task_id}",
        current_run_id=11,
    )


def _drive_spawn(kb, kbd, monkeypatch, task, workspace, agent_home: str, popen_seen: dict) -> int:
    """Run ``_default_spawn`` with Popen monkeypatched to a recorder.

    Returns the recorded subprocess pid (always 4242 in tests). Does NOT
    actually exec the worker -- the worker argv is what we are asserting on.
    """
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setenv("HERMES_AGENT_HOME", agent_home)

    class _Fake:
        pid = 4242

    def _fake_popen(cmd, *args, **kwargs):
        popen_seen["cmd"] = list(cmd)
        popen_seen["cwd"] = kwargs.get("cwd")
        popen_seen["env"] = dict(kwargs.get("env") or {})
        return _Fake()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return kbd._default_spawn(task, workspace)


def test_default_spawn_wraps_worktree_task_in_unshare(tmp_path, monkeypatch):
    """A worktree task under ``<agent_home>/.worktrees/<id>`` must be wrapped."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    agent_home = tmp_path / "agent"
    _seed_repo(agent_home)
    worktree = agent_home / ".worktrees" / "t_iso_e2e"
    worktree.mkdir(parents=True, exist_ok=True)

    task = _make_task(kb, workspace_kind="worktree", workspace_path=str(worktree))
    seen: dict = {}
    _drive_spawn(kb, kbd, monkeypatch, task, str(worktree), str(agent_home), seen)

    cmd = seen["cmd"]
    assert cmd[0] == "unshare", f"expected unshare prefix; full cmd={cmd!r}"
    assert cmd[1] == "-Urm"
    # The launcher module is the next thing; it takes agent_home + --worktree.
    assert "hermes_cli.worker_isolate" in cmd
    assert "_launcher" in cmd
    assert str(agent_home) in cmd
    assert "--worktree" in cmd
    assert str(worktree) in cmd


def test_default_spawn_does_not_wrap_scratch_task(tmp_path, monkeypatch):
    """A scratch task must NOT be wrapped -- the carve-out set is unsafe on arbitrary cwds."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    agent_home = tmp_path / "agent"
    agent_home.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    task = _make_task(kb, workspace_kind="scratch", workspace_path=str(scratch))
    seen: dict = {}
    _drive_spawn(kb, kbd, monkeypatch, task, str(scratch), str(agent_home), seen)

    cmd = seen["cmd"]
    # No unshare prefix when the task is not worktree-anchored.
    assert cmd[0] != "unshare"


def test_default_spawn_does_not_wrap_when_kill_switch_set(tmp_path, monkeypatch):
    """HERMES_WORKER_ISOLATION=off short-circuits the wrap."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    agent_home = tmp_path / "agent"
    _seed_repo(agent_home)
    worktree = agent_home / ".worktrees" / "t_iso_e2e"
    worktree.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_WORKER_ISOLATION", "off")

    task = _make_task(kb, workspace_kind="worktree", workspace_path=str(worktree))
    seen: dict = {}
    _drive_spawn(kb, kbd, monkeypatch, task, str(worktree), str(agent_home), seen)

    cmd = seen["cmd"]
    assert cmd[0] != "unshare"
