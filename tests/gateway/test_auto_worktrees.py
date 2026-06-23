from pathlib import Path
import subprocess

from gateway.session import SessionContext, SessionEntry, SessionSource
from gateway.config import Platform
from gateway.run import GatewayRunner, _ensure_gateway_session_worktree


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
    repo.mkdir()
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

    tokens = runner._set_session_env(context, cwd=str(worktree))
    try:
        assert resolve_agent_cwd() == worktree
    finally:
        runner._clear_session_env(tokens)
