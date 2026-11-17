"""Messaging-chat workspaces: ``/workspace DIR`` binds a chat, and opt-in per-thread forks give every
thread its own git worktree of each repository in it (real git repositories under tmp_path)."""
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


def _thread(thread_id):
    return SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", thread_id=thread_id,
                         user_id="U1", user_name="tester")


@pytest.mark.asyncio
async def test_workspace_command_binds_the_chat_and_its_turns_run_there(tmp_path, monkeypatch):
    from agent.runtime_cwd import resolve_agent_cwd
    from gateway.session_workspace import bind_turn, bound_root, handle_workspace_command
    from tools.terminal_tool import resolve_task_overrides
    project = tmp_path / "project"
    project.mkdir()
    runner = _runner(tmp_path, monkeypatch)
    source = _thread("1.0")
    reply = await handle_workspace_command(runner, MessageEvent(text=f"/workspace {project}", source=source))
    entry = runner.session_store.get_or_create_session(source)
    assert str(project) in reply and bound_root(runner, entry.session_key) == str(project)

    assert bind_turn(runner, entry.session_key, entry.session_id, {}) == str(project)
    assert resolve_agent_cwd() == project
    assert resolve_task_overrides(entry.session_id)["cwd"] == str(project)

    # /new starts a new conversation in the same project; /workspace clear returns to terminal.cwd.
    fresh = runner.session_store.reset_session(entry.session_key)
    assert bound_root(runner, fresh.session_key) == str(project)
    await handle_workspace_command(runner, MessageEvent(text="/workspace clear", source=source))
    assert bound_root(runner, entry.session_key) == ""


@pytest.mark.asyncio
async def test_two_threads_get_their_own_worktrees_and_a_repo_index(tmp_path, monkeypatch):
    from gateway.session_workspace import bind_turn, handle_workspace_command
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
        roots.append(bind_turn(runner, entry.session_key, entry.session_id, config))

    assert roots[0] != roots[1]
    for root in roots:
        tree = os.path.realpath(os.path.join(root, "api"))
        assert tree.startswith(str(workspace / "api" / ".worktrees"))
        assert _git(tree, "branch", "--show-current").startswith("hermes/thread-")
        index = Path(root, "AGENTS.md").read_text(encoding="utf-8-sig")
        assert "Repositories in this workspace" in index
        assert f"not cloned: `git clone --filter=blob:none {web_url} {workspace / 'web'}`" in index
    assert not (workspace / "web").exists()  # never cloned synchronously
    # The second fork reused the first one's fetch (within fetch_fresh_seconds).
    assert "s ago)" in Path(roots[1], "AGENTS.md").read_text(encoding="utf-8-sig")
    assert _git(workspace / "api", "worktree", "list").count(".worktrees/thread-") == 2
