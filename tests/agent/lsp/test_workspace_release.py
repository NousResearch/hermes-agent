"""Workspace-scoped teardown: removed worktrees and deleted roots must not keep language servers alive.

A gateway outlives the coding sessions it runs, so a ``(server, root)`` client that survives its
worktree is an unbounded leak (multi-GiB tsserver heaps for trees that no longer exist).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import agent.lsp
from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec
from agent.lsp.workspace import clear_cache

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _git(*args: str, cwd: Path) -> str:
    res = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=60)
    assert res.returncode == 0, f"git {' '.join(args)} failed: {res.stderr}"
    return res.stdout.strip()


def _seed_project(root: Path) -> Path:
    """A Python project the mock ``pyright`` will attach to."""
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    return root


def _make_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / ".git").mkdir()
    return _seed_project(repo)


@pytest.fixture
def mock_pyright(request):
    """Install the mock as ``pyright``; ``request.param`` selects the multi-root shape."""
    idx = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[idx]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(command=[sys.executable, MOCK_SERVER], workspace_root=root, cwd=root,
                         env={"MOCK_LSP_SCRIPT": "errors"}, initialization_options={})

    SERVERS[idx] = ServerDef(
        server_id="pyright", extensions=original.extensions, resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn, seed_first_push=False, description="mock pyright",
        multi_root=getattr(request, "param", False),
    )
    clear_cache()
    yield
    SERVERS[idx] = original
    clear_cache()


def _service() -> LSPService:
    return LSPService(enabled=True, wait_mode="document", wait_timeout=3.0, install_strategy="manual",
                      idle_timeout=600.0)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A pushed-clean project repo: every worktree-removal path agrees it holds no work."""
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", str(origin), cwd=tmp_path)
    repo = tmp_path / "project"
    _git("clone", "-q", str(origin), str(repo), cwd=tmp_path)
    _git("config", "user.email", "t@example.com", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _seed_project(repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    _git("push", "-q", "origin", "HEAD", cwd=repo)
    return repo


def _linked_worktree(project: Path, tmp_path: Path) -> Path:
    wt = tmp_path / "wt"
    _git("worktree", "add", "-q", "-b", "wt/t1", str(wt), "HEAD", cwd=project)
    return wt


# Production removal paths; each returns (root that goes away, sibling root that stays, remove()).
def _kanban_worktree(project, tmp_path, monkeypatch):
    from hermes_cli import kanban_db_workspace as kbw
    wt = _linked_worktree(project, tmp_path)
    return wt, project, lambda: kbw._cleanup_worktree_workspace("t1", str(wt), "wt/t1")


def _cli_worktree(project, tmp_path, monkeypatch):
    import cli
    wt = _linked_worktree(project, tmp_path)
    info = {"path": str(wt), "branch": "wt/t1", "repo_root": str(project)}
    return wt, project, lambda: cli._cleanup_worktree(info)


def _subagent_worktree(project, tmp_path, monkeypatch):
    from tools.subagent_worktree import finalize_subagent_worktree
    wt = _linked_worktree(project, tmp_path)
    info = {"path": str(wt), "branch": "wt/t1", "repo_root": str(project),
            "base_commit": _git("rev-parse", "HEAD", cwd=project)}

    def remove():
        assert finalize_subagent_worktree(info)["pruned"] is True
    return wt, project, remove


def _web_git_worktree_remove(project, tmp_path, monkeypatch):
    from hermes_cli import web_git
    wt = _linked_worktree(project, tmp_path)
    return wt, project, lambda: web_git.worktree_remove(str(project), str(wt), True)


def _worktree_gc_reclaim(project, tmp_path, monkeypatch):
    from hermes_cli import worktree_gc
    wt = _linked_worktree(project, tmp_path)
    record = worktree_gc.TreeRecord(
        name="wt", path=str(wt), branch="wt/t1", age_days=0.0, size_mb=None,
        verdict="reap", reason="test-forced")

    def remove():
        actions = worktree_gc.reclaim_worktrees(str(project), records=[record])
        assert actions and actions[0].startswith("removed")
    return wt, project, remove


def _kanban_scratch(project, tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_workspace as kbw
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="scratch")
        ws = kbw.resolve_workspace(kb.get_task(conn, tid))
        kbw.set_workspace_path(conn, tid, ws)
    (ws / ".git").mkdir()  # the worker cloned a project into its scratch dir
    _seed_project(ws)

    def remove():
        with kbc.connect() as conn:
            assert kb.complete_task(conn, tid, result="ok")
    return ws, project, remove


@pytest.mark.parametrize("mock_pyright", [False, True], ids=["single-root", "multi-root"], indirect=True)
@pytest.mark.parametrize(
    "entry",
    [_kanban_worktree, _cli_worktree, _subagent_worktree, _kanban_scratch,
     _web_git_worktree_remove, _worktree_gc_reclaim],
    ids=lambda f: f.__name__.strip("_"))
def test_workspace_removal_releases_only_its_language_servers(entry, mock_pyright, project, tmp_path,
                                                                 monkeypatch):
    """Every in-process workspace-removal path (kanban worktree/scratch cleanup, ``hermes -w`` exit, the
    delegate_task worktree prune) shuts down (single-root) or detaches (multi-root) exactly the removed
    root's servers BEFORE the tree goes: the sibling keeps serving diagnostics, the released state is gone
    everywhere, and a repeat release is a no-op."""
    gone, kept, remove = entry(project, tmp_path, monkeypatch)
    gone_s, kept_s = str(gone.resolve()), str(kept.resolve())
    svc = _service()
    monkeypatch.setattr(agent.lsp, "_service", svc)  # the process-wide singleton the removal paths reach
    try:
        svc.snapshot_baseline(str(gone / "x.py"))
        assert svc.get_diagnostics_sync(str(gone / "x.py"), delta=False)
        assert svc.get_diagnostics_sync(str(kept / "x.py"), delta=False)
        clients = dict(svc._clients)
        procs = {key: c._proc for key, c in clients.items()}  # asyncio subprocess handles

        remove()

        assert not gone.exists(), "the production removal path must have removed the tree"
        assert all(gone_s not in f for c in svc._clients.values() for f in c.workspace_folders)
        assert all(not p.startswith(gone_s) for p in svc._delta_baseline)
        if len(clients) == 2:  # single-root: the released process is gone, the sibling's is not
            (gone_key,) = [k for k in clients if k[1] == gone_s]
            assert gone_key not in svc._clients and gone_key not in svc._last_used
            assert procs[gone_key].returncode is not None, "released server process must have exited"
        else:  # multi-root: one shared process, only the folder was dropped
            (client,) = svc._clients.values()
            assert client.workspace_folders == [kept_s] and client.is_running
        assert svc.release_workspace(gone_s) == 0
        assert svc.get_diagnostics_sync(str(kept / "x.py"), delta=False)
    finally:
        svc.shutdown()


def test_reaper_shuts_down_client_whose_root_was_deleted(mock_pyright, tmp_path):
    """A root deleted outside Hermes is reaped on the next sweep even though the client is not idle."""
    repo = _make_repo(tmp_path, "repo")
    svc = _service()
    try:
        assert svc.get_diagnostics_sync(str(repo / "x.py"), delta=False)
        (key,) = svc._clients
        proc = svc._clients[key]._proc

        svc._loop.run(svc._reap_idle_once(), timeout=10.0)
        assert key in svc._clients, "an existing, recently used root must survive the sweep"

        shutil.rmtree(repo)
        svc._loop.run(svc._reap_idle_once(), timeout=10.0)
        assert key not in svc._clients and key not in svc._last_used
        assert proc.returncode is not None, "reaped server process must have exited"
    finally:
        svc.shutdown()
