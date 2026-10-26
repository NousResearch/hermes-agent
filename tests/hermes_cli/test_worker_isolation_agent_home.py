"""Regression: the agent home must resolve to the PRODUCTION CODE TREE.

The mount-namespace sandbox protects the tree the agent's source lives in.
On this fleet that is ``~/.hermes/hermes-agent`` (the ``hermes-agent`` repo),
which is a DIFFERENT git checkout from ``~/.hermes`` itself (the
``hermes-config`` repo that holds the profiles and the kanban workspaces).

Before this fix, ``_resolve_worker_agent_home`` derived the agent home from
``HERMES_HOME`` -- which is the config repo. Two things went wrong at once:

1. The launcher flipped the CONFIG repo read-only, leaving the code tree
   (the thing a worker could actually rewrite) writable.
2. Every scratch workspace lives at
   ``<config-repo>/kanban/boards/<board>/workspaces/<id>``, i.e. INSIDE the
   wrongly-resolved agent home, so ``_is_outside_agent_home`` refused the
   wrap outright. The dispatcher logged "isolation skipped" and spawned the
   worker UNWRAPPED.

Net effect on 2026-10-26: ZERO workers were isolated, and a probe confirmed a
worker could write to the production tree. These tests pin the resolution
order and the containment predicates against that exact topology.

The synthetic layout stands in for the fleet's two-repo shape; the code-repo
resolver is stubbed to point at the synthetic code repo so the containment
predicates and the dispatch decision are exercised against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _seed_repo(path: Path) -> None:
    """Make ``path`` a git repo with one commit."""
    import subprocess

    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(path), "init", "-q", "-b", "main", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--allow-empty", "-q", "-m", "initial"],
        check=True,
    )


def _prod_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build the fleet's two-repo layout under ``tmp_path``.

    Returns ``(config_repo, code_repo, scratch_ws)``:

    * ``config_repo`` = ``<home>``   -- the hermes-config checkout
    * ``code_repo``   = ``<home>/hermes-agent`` -- the hermes-agent checkout
    * ``scratch_ws``  = ``<home>/kanban/boards/homelab/workspaces/t_x``
    """
    config_repo = tmp_path / "home"
    code_repo = config_repo / "hermes-agent"
    scratch_ws = config_repo / "kanban" / "boards" / "homelab" / "workspaces" / "t_x"
    _seed_repo(config_repo)
    _seed_repo(code_repo)
    scratch_ws.mkdir(parents=True, exist_ok=True)
    return config_repo, code_repo, scratch_ws


@pytest.fixture
def fleet_layout(tmp_path, monkeypatch):
    """Synthetic fleet layout + the resolver pointed at the synthetic code repo."""
    from hermes_cli import kanban_db_dispatch as kbd

    config_repo, code_repo, scratch_ws = _prod_layout(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(config_repo))
    monkeypatch.delenv("HERMES_AGENT_HOME", raising=False)
    monkeypatch.delenv("HERMES_AGENT", raising=False)
    monkeypatch.setattr(
        kbd, "_production_code_repo_root", lambda: str(code_repo.resolve()),
    )
    return config_repo, code_repo, scratch_ws


def test_agent_home_is_the_code_repo_not_the_config_repo(fleet_layout):
    """``HERMES_HOME`` pointing at the config repo must NOT win over the code tree."""
    from hermes_cli import kanban_db_dispatch as kbd

    config_repo, code_repo, _ = fleet_layout
    resolved = kbd._resolve_worker_agent_home()
    assert resolved is not None
    assert Path(resolved) == code_repo.resolve()
    assert Path(resolved) != config_repo.resolve(), (
        "the config repo was resolved as the agent home -- the sandbox would "
        "protect the wrong tree"
    )


def test_explicit_agent_home_override_wins(tmp_path, monkeypatch):
    """``HERMES_AGENT_HOME`` is an operator override and still wins."""
    from hermes_cli import kanban_db_dispatch as kbd

    config_repo, code_repo, _ = _prod_layout(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(config_repo))
    monkeypatch.setenv("HERMES_AGENT_HOME", str(code_repo))
    # Point the code-repo resolver at the CONFIG repo: the explicit
    # override must still win, so ordering cannot silently demote it.
    monkeypatch.setattr(
        kbd, "_production_code_repo_root", lambda: str(config_repo.resolve()),
    )

    resolved = kbd._resolve_worker_agent_home()
    assert resolved == str(code_repo.resolve())


def test_scratch_workspace_is_outside_the_code_repo(fleet_layout):
    """A fleet-layout scratch workspace is OUTSIDE the code repo (so it may be wrapped)."""
    from hermes_cli import kanban_db_dispatch as kbd

    _, _, scratch_ws = fleet_layout
    resolved = kbd._resolve_worker_agent_home()
    assert resolved is not None
    # The predicate that decides whether a scratch task can be wrapped.
    assert kbd._is_outside_agent_home(str(scratch_ws), resolved) is True


def test_scratch_workspace_inside_code_repo_is_refused(fleet_layout):
    """A scratch path INSIDE the code repo is refused (would re-grant prod write)."""
    from hermes_cli import kanban_db_dispatch as kbd

    _, code_repo, _ = fleet_layout
    inside = code_repo / "scratch-inside"
    inside.mkdir(parents=True, exist_ok=True)
    resolved = kbd._resolve_worker_agent_home()
    assert resolved is not None
    assert kbd._is_outside_agent_home(str(inside), resolved) is False


def test_worktree_under_code_repo_is_recognised(fleet_layout):
    """``<code_repo>/.worktrees/<id>`` is the dispatcher-provisioned worktree shape."""
    from hermes_cli import kanban_db_dispatch as kbd

    _, code_repo, _ = fleet_layout
    wt = code_repo / ".worktrees" / "t_x"
    wt.mkdir(parents=True, exist_ok=True)
    resolved = kbd._resolve_worker_agent_home()
    assert resolved is not None
    assert kbd._is_worktree_under_agent_home(str(wt), resolved) is True


def test_worktree_under_config_repo_is_not_a_code_worktree(fleet_layout):
    """A worktree under the CONFIG repo is not the code repo's worktree."""
    from hermes_cli import kanban_db_dispatch as kbd

    config_repo, _, _ = fleet_layout
    wt = config_repo / ".worktrees" / "t_x"
    wt.mkdir(parents=True, exist_ok=True)
    resolved = kbd._resolve_worker_agent_home()
    assert resolved is not None
    assert kbd._is_worktree_under_agent_home(str(wt), resolved) is False


def test_dispatcher_wraps_scratch_task_in_fleet_layout(fleet_layout, monkeypatch):
    """End-to-end: a fleet-layout SCRATCH task must be wrapped in ``unshare``.

    This is the assertion that would have caught the 2026-10-26 regression:
    the dispatcher skipped the wrap (and logged "isolation skipped") for
    every scratch task because the scratch path looked like it lived inside
    the wrongly-resolved agent home.
    """
    import shutil
    import subprocess

    if shutil.which("unshare") is None:
        pytest.skip("unshare not on PATH")
    probe = subprocess.run(
        ["unshare", "--user", "--map-root-user", "true"], capture_output=True, timeout=5,
    )
    if probe.returncode != 0:
        pytest.skip("unprivileged user namespace unavailable on this host")

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import worker_isolate as wi

    config_repo, code_repo, scratch_ws = fleet_layout
    monkeypatch.setenv("HERMES_WORKER_ISOLATION", "enforce")
    monkeypatch.setattr(wi, "_userns_available", lambda: True)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    seen: dict = {}

    class _Fake:
        pid = 4242
        returncode = 0

        def __init__(self, cmd, *args, **kwargs):
            seen["cmd"] = list(cmd)
            self.cmd = seen["cmd"]

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _Fake)

    task = kb.Task(
        id="t_x",
        title="scratch wrap",
        body=None,
        assignee="friday",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=str(scratch_ws),
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        branch_name="wt/t_x",
        current_run_id=1,
    )
    kbd._default_spawn(task, str(scratch_ws))

    cmd = seen["cmd"]
    assert cmd[0] == "unshare", f"scratch task was NOT wrapped: {cmd!r}"
    assert cmd[1] == "-Urm"
    assert "--scratch" in cmd
    assert str(scratch_ws) in cmd
    # The protected tree is the CODE repo, not the config repo.
    assert str(code_repo.resolve()) in cmd
    assert str(config_repo.resolve()) not in cmd
