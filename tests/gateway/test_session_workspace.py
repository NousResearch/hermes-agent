"""Messaging-chat workspaces: ``/workspace DIR`` binds a chat, and opt-in per-thread forks give every
thread its own git worktree of each repository in it (real git repositories under tmp_path)."""
import asyncio
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore


def _git(cwd, *args):
    out = subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout.strip()


def _clone(tmp_path, name):
    origin = tmp_path / "remotes" / f"{name}.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed" / name
    subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
    (seed / "README.md").write_text(name, encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    _git(seed, "push", "-q", str(origin), "main")
    return origin


def _runner(tmp_path, monkeypatch):
    import hermes_state

    def no_sqlite():
        raise RuntimeError("SQLite disabled in test")

    monkeypatch.setattr(hermes_state, "SessionDB", no_sqlite)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._loaded = True
    evicted = []
    runner = SimpleNamespace(session_store=store, async_session_store=AsyncSessionStore(store),
                             _evict_cached_agent=evicted.append)
    runner._session_key_for_source = store._generate_session_key
    return runner


def _turn(runner, entry, config):
    """Run one turn's workspace binding the way the turn runner does; the bound root."""
    from gateway.session_workspace import workspace_turn
    with workspace_turn(runner, entry.session_key, entry.session_id, config) as root:
        return root


def _thread(thread_id):
    return SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", thread_id=thread_id,
                         user_id="U1", user_name="tester")


@pytest.mark.asyncio
async def test_workspace_command_binds_the_chat_and_its_turns_run_there(tmp_path, monkeypatch):
    from agent.runtime_cwd import resolve_agent_cwd
    from gateway.session_workspace import bound_root, handle_workspace_command
    from tools.terminal_tool import resolve_task_overrides
    project = tmp_path / "project"
    project.mkdir()
    runner = _runner(tmp_path, monkeypatch)
    source = _thread("1.0")
    reply = await handle_workspace_command(runner, MessageEvent(text=f"/workspace {project}", source=source))
    entry = runner.session_store.get_or_create_session(source)
    assert str(project) in reply and bound_root(runner, entry.session_key) == str(project)

    assert _turn(runner, entry, {}) == str(project)
    assert resolve_agent_cwd() == project
    assert resolve_task_overrides(entry.session_id)["cwd"] == str(project)

    # /new starts a new conversation in the same project; /workspace clear returns to terminal.cwd.
    fresh = runner.session_store.reset_session(entry.session_key)
    assert bound_root(runner, fresh.session_key) == str(project)
    await handle_workspace_command(runner, MessageEvent(text="/workspace clear", source=source))
    assert bound_root(runner, entry.session_key) == ""
    # A configured default directory does not come back after an explicit clear, nor after /new.
    other = tmp_path / "default"
    other.mkdir()
    assert _turn(runner, entry, {"gateway": {"workspaces": {"dir": str(other)}}}) == ""
    runner.session_store.reset_session(entry.session_key)
    assert _turn(runner, entry, {"gateway": {"workspaces": {"dir": str(other)}}}) == ""


@pytest.mark.asyncio
async def test_two_threads_get_their_own_worktrees_and_a_repo_index(tmp_path, monkeypatch):
    from gateway.session_workspace import handle_workspace_command
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _git(tmp_path, "clone", "-q", str(_clone(tmp_path, "api")), str(workspace / "api"))
    web_url = str(_clone(tmp_path, "web"))
    runner = _runner(tmp_path, monkeypatch)
    config = {"gateway": {"workspaces": {"per_thread": "worktree", "fetch_fresh_seconds": 3600,
                                         "repos": [{"path": "web", "url": web_url}]}}}
    roots = []
    for thread in ("1.0", "2.0"):
        source = _thread(thread)
        await handle_workspace_command(runner, MessageEvent(text=f"/workspace {workspace}", source=source))
        entry = runner.session_store.get_or_create_session(source)
        roots.append(_turn(runner, entry, config))

    assert roots[0] != roots[1]
    for root in roots:
        tree = os.path.realpath(os.path.join(root, "api"))
        assert tree.startswith(str(workspace / "api" / ".worktrees"))
        assert _git(tree, "branch", "--show-current").startswith("hermes/thread-")
        index = Path(root, "AGENTS.md").read_text(encoding="utf-8-sig")
        assert "Repositories in this workspace" in index
        assert f"not cloned: `git clone --filter=blob:none {web_url} {workspace / 'web'} && " in index
    assert not (workspace / "web").exists()  # never cloned synchronously
    # The second fork reused the first one's fetch (within fetch_fresh_seconds).
    assert "s ago)" in Path(roots[1], "AGENTS.md").read_text(encoding="utf-8-sig")
    assert _git(workspace / "api", "worktree", "list").count(".worktrees/thread-") == 2


@pytest.mark.asyncio
async def test_a_thread_fork_heals_and_its_repos_answer_references_and_diff(tmp_path, monkeypatch):
    """A tree the sweep removed is rebuilt, ``@file:`` reads a repository's worktree through the fork's
    link, ``/diff`` reports each repository, and an idle fork holds no live lock."""
    from gateway.session_context_refs import expand_references
    from gateway.session_workspace import binding, handle_workspace_command, reference_root
    from hermes_cli.worktree_ops import _worktree_lock_is_live
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("api", "web"):
        _git(tmp_path, "clone", "-q", str(_clone(tmp_path, name)), str(workspace / name))
    runner = _runner(tmp_path, monkeypatch)
    config = {"gateway": {"workspaces": {"per_thread": "worktree", "fetch_fresh_seconds": 3600}}}
    source = _thread("1.0")
    await handle_workspace_command(runner, MessageEvent(text=f"/workspace {workspace}", source=source))
    entry = runner.session_store.get_or_create_session(source)
    root = _turn(runner, entry, config)
    api_tree = os.path.realpath(os.path.join(root, "api"))
    assert _worktree_lock_is_live(str(workspace / "api"), api_tree) is None  # unlocked between turns

    _git(workspace / "api", "worktree", "remove", "--force", api_tree)
    assert _turn(runner, entry, config) == root and os.path.isdir(api_tree)  # rebuilt at its saved path

    stored = binding(runner, entry.session_key)
    text, refusal = await expand_references("see @file:api/README.md", cwd=root, context_length=100_000,
                                            plugins=False, allowed_root=reference_root(stored))
    assert refusal is None and "api" in text.split("see", 1)[1]

    Path(api_tree, "README.md").write_text("changed", encoding="utf-8")
    from gateway.slash_commands import GatewaySlashCommandsMixin
    runner._checkouts = lambda src: GatewaySlashCommandsMixin._checkouts(runner, src)
    runner._terminal_cwd = lambda src=None: GatewaySlashCommandsMixin._terminal_cwd(runner, src)
    runner._render_diff_result = lambda result, stat: GatewaySlashCommandsMixin._render_diff_result(runner, result, stat)
    runner._fenced_truncated_diff = GatewaySlashCommandsMixin._fenced_truncated_diff
    reply = await GatewaySlashCommandsMixin._handle_diff_command(runner, MessageEvent(text="/diff", source=source))
    assert "**api**" in reply and "README.md" in reply and "**web**" in reply and "Not a git repository" not in reply


def test_a_thread_whose_tree_the_sweep_archived_gets_its_work_back(tmp_path, monkeypatch):
    """The sweep archives an idle thread's tree and deletes its unpublished branch; the thread's next
    turn restores the archive (commits and uncommitted edits), not a fresh checkout of the remote."""
    from gateway.session_workspace import handle_workspace_command
    from hermes_cli.worktree_sweep import archive_then_remove
    project = tmp_path / "project"
    _git(tmp_path, "clone", "-q", str(_clone(tmp_path, "api")), str(project))
    runner = _runner(tmp_path, monkeypatch)
    config = {"gateway": {"workspaces": {"per_thread": "worktree", "fetch_fresh_seconds": 3600}}}
    source = _thread("1.0")
    asyncio.run(handle_workspace_command(runner, MessageEvent(text=f"/workspace {project}", source=source)))
    entry = runner.session_store.get_or_create_session(source)
    root = _turn(runner, entry, config)
    Path(root, "work.txt").write_text("committed", encoding="utf-8")
    _git(root, "add", "work.txt")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "thread work")
    Path(root, "README.md").write_text("uncommitted", encoding="utf-8")
    branch = _git(root, "branch", "--show-current")

    ref, kept, removed, _ = archive_then_remove(str(project), root, reason="test", max_untracked_mb=10,
                                                unlock=True, remote_heads_fn=dict)
    assert removed and kept is None and ref  # the branch is gone; only the archive holds the work

    assert _turn(runner, entry, config) == root
    assert Path(root, "work.txt").read_text(encoding="utf-8-sig") == "committed"
    assert Path(root, "README.md").read_text(encoding="utf-8-sig") == "uncommitted"
    assert _git(root, "branch", "--show-current") == branch


def test_gateway_workspaces_settings_load_from_config_yaml(tmp_path):
    """``gateway.workspaces`` read through the gateway's config loader."""
    from gateway.run import _load_gateway_config
    from gateway.session_workspace_fork import WorkspaceSettings
    path = tmp_path / "config.yaml"
    path.write_text("gateway:\n  workspaces:\n    per_thread: worktree\n    fetch_fresh_seconds: 5\n"
                    f"    dir: {tmp_path}\n    repos:\n      - path: web\n        url: https://example.invalid/web.git\n",
                    encoding="utf-8")
    settings = WorkspaceSettings.from_config(_load_gateway_config(path))
    assert (settings.per_thread, settings.fetch_fresh_seconds, settings.dir) == ("worktree", 5.0, str(tmp_path))
    assert [(r.path, r.url) for r in settings.repos] == [("web", "https://example.invalid/web.git")]
