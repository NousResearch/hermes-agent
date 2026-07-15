import asyncio
from pathlib import Path
import subprocess

from gateway.session import SessionContext, SessionEntry, SessionSource
from gateway.config import Platform
from gateway.run import (
    GatewayRunner,
    _cleanup_gateway_session_worktrees,
    _ensure_gateway_session_worktree,
    _ensure_gateway_session_worktree_map,
    _prepare_gateway_session_worktrees_async,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_disabled_auto_worktree_returns_original_cwd(tmp_path):
    repo = _repo(tmp_path)

    assert _ensure_gateway_session_worktree(
        base_cwd=str(repo),
        session_key="agent:main:slack:dm:U1",
        session_id="20260623_120000_abcd",
        user_config={"gateway": {"auto_worktrees": {"enabled": False}}},
    ) == str(repo)


def test_enabled_auto_worktree_creates_per_session_git_worktree(tmp_path):
    repo = _repo(tmp_path)

    cwd = _ensure_gateway_session_worktree(
        base_cwd=str(repo),
        session_key="agent:main:slack:dm:U1",
        session_id="20260623_120000_abcd",
        user_config={"gateway": {"auto_worktrees": {"enabled": True, "sync_base": False}}},
    )

    worktree = Path(cwd)
    assert worktree != repo
    assert worktree.parent == repo / ".worktrees"
    assert (worktree / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert _git(worktree, "rev-parse", "--is-inside-work-tree") == "true"
    assert _git(worktree, "branch", "--show-current").startswith("hermes/gateway/")
    assert ".worktrees/" in (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8")


def test_enabled_auto_worktree_reuses_existing_session_worktree(tmp_path):
    repo = _repo(tmp_path)
    cfg = {"gateway": {"auto_worktrees": {"enabled": True, "sync_base": False}}}

    first = _ensure_gateway_session_worktree(
        base_cwd=str(repo),
        session_key="agent:main:slack:dm:U1",
        session_id="20260623_120000_abcd",
        user_config=cfg,
    )
    second = _ensure_gateway_session_worktree(
        base_cwd=str(repo),
        session_key="agent:main:slack:dm:U1",
        session_id="20260623_120000_abcd",
        user_config=cfg,
    )

    assert second == first


def test_cleanup_removes_session_worktree_and_branch(tmp_path):
    repo = _repo(tmp_path)
    cfg = {"gateway": {"auto_worktrees": {"enabled": True, "sync_base": False}}}
    session_key = "agent:main:slack:dm:U1"
    session_id = "20260623_120000_abcd"
    cwd = _ensure_gateway_session_worktree(
        base_cwd=str(repo),
        session_key=session_key,
        session_id=session_id,
        user_config=cfg,
    )
    branch = _git(Path(cwd), "branch", "--show-current")

    assert _cleanup_gateway_session_worktrees(
        base_cwd=str(repo),
        user_config=cfg,
        session_key=session_key,
        session_id=session_id,
    ) == 1

    assert not Path(cwd).exists()
    assert branch not in _git(repo, "branch", "--format=%(refname:short)").splitlines()


def test_worktree_map_materializes_configured_repo_even_when_base_cwd_elsewhere(tmp_path):
    repo = _repo(tmp_path)
    cfg = {
        "gateway": {
            "auto_worktrees": {
                "enabled": True,
                "sync_base": False,
                "repos": [str(repo)],
            }
        }
    }

    mapping = _ensure_gateway_session_worktree_map(
        user_config=cfg,
        session_key="agent:main:slack:dm:U1",
        session_id="20260623_120000_abcd",
    )

    worktree = Path(mapping[str(repo.resolve())])
    assert worktree != repo
    assert worktree.parent == repo / ".worktrees"
    assert _git(worktree, "branch", "--show-current").startswith("hermes/gateway/")


def test_concurrent_session_setup_keeps_event_loop_responsive_and_isolates_worktrees(
    tmp_path, monkeypatch
):
    import time

    import gateway.run as gateway_run

    repo = _repo(tmp_path)
    cfg = {"gateway": {"auto_worktrees": {"enabled": True, "sync_base": False}}}
    real_subprocess_run = subprocess.run
    loop_ticks = [0]
    delay_observations = []

    def delayed_subprocess_run(command, *args, **kwargs):
        if "worktree" in command and "add" in command:
            ticks_before_delay = loop_ticks[0]
            time.sleep(0.1)
            delay_observations.append(loop_ticks[0] > ticks_before_delay)
        return real_subprocess_run(command, *args, **kwargs)

    monkeypatch.setattr(gateway_run.subprocess, "run", delayed_subprocess_run)

    async def run_setups():
        stop_ticker = asyncio.Event()

        async def tick_while_setup_runs():
            while not stop_ticker.is_set():
                loop_ticks[0] += 1
                await asyncio.sleep(0)

        ticker = asyncio.create_task(tick_while_setup_runs())
        setup_a, setup_b = await asyncio.gather(
            _prepare_gateway_session_worktrees_async(
                base_cwd=str(repo),
                user_config=cfg,
                session_key="agent:main:slack:dm:U1",
                session_id="session-a",
            ),
            _prepare_gateway_session_worktrees_async(
                base_cwd=str(repo),
                user_config=cfg,
                session_key="agent:main:slack:dm:U2",
                session_id="session-b",
            ),
        )
        stop_ticker.set()
        await ticker
        return setup_a, setup_b

    (cwd_a, map_a), (cwd_b, map_b) = asyncio.run(run_setups())

    assert delay_observations
    assert all(delay_observations)
    assert cwd_a != cwd_b
    assert Path(cwd_a).parent == repo / ".worktrees"
    assert Path(cwd_b).parent == repo / ".worktrees"
    assert map_a[str(repo.resolve())] == cwd_a
    assert map_b[str(repo.resolve())] == cwd_b


def test_set_session_env_pins_gateway_session_cwd(monkeypatch, tmp_path):
    from agent.runtime_cwd import resolve_agent_cwd

    base = tmp_path / "base"
    worktree = tmp_path / "worktree"
    base.mkdir()
    worktree.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(base))

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1")
    entry = SessionEntry(
        session_key="agent:main:slack:dm:U1",
        session_id="sid",
        created_at=__import__("datetime").datetime.now(),
        updated_at=__import__("datetime").datetime.now(),
        origin=source,
        platform=Platform.SLACK,
    )
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.SLACK],
        home_channels={},
        session_key=entry.session_key,
        session_id=entry.session_id,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )

    tokens = runner._set_session_env(
        context,
        cwd=str(worktree),
        worktree_map={str(base): str(worktree)},
    )
    try:
        assert resolve_agent_cwd() == worktree
        from agent.runtime_cwd import map_session_path_to_worktree

        assert map_session_path_to_worktree(str(base / "x.py")) == worktree / "x.py"
    finally:
        runner._clear_session_env(tokens)
