"""Multi-root servers share ONE process across project roots.

A profiled session with subagents editing across ~30 git worktrees ran
30-60 pyright processes.  Pyright supports multi-root workspaces, so
the service keys such clients by ``server_id`` alone and attaches each
new root via ``workspace/didChangeWorkspaceFolders``.  Single-root
servers keep the one-client-per-root behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec
from agent.lsp.workspace import clear_cache

MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


@pytest.fixture(autouse=True)
def _clear_workspace_cache():
    clear_cache()
    yield
    clear_cache()


def _make_repo(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    (repo / "x.py").write_text("print('hi')\n", encoding="utf-8")
    return repo


@pytest.fixture
def two_repos(tmp_path):
    return _make_repo(tmp_path, "repo-a"), _make_repo(tmp_path, "repo-b")


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright``; yield (spawn_count, folders_log, set_multi_root)."""
    idx = next(i for i, s in enumerate(SERVERS) if s.server_id == "pyright")
    original = SERVERS[idx]
    spawns = {"value": 0}
    folders_log = tmp_path / "folders.jsonl"

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        spawns["value"] += 1
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={"MOCK_LSP_SCRIPT": "errors", "MOCK_LSP_FOLDERS_LOG": str(folders_log)},
        )

    def _install(multi_root: bool) -> None:
        SERVERS[idx] = ServerDef(
            server_id="pyright",
            extensions=original.extensions,
            resolve_root=lambda fp, ws: ws,
            build_spawn=_spawn,
            multi_root=multi_root,
            description="mock pyright",
        )

    yield spawns, folders_log, _install
    SERVERS[idx] = original


def _service() -> LSPService:
    return LSPService(
        enabled=True, wait_mode="document", wait_timeout=3.0, install_strategy="manual"
    )


def test_multi_root_server_shares_one_client_across_roots(two_repos, mock_pyright, monkeypatch):
    repo_a, repo_b = two_repos
    spawns, folders_log, install = mock_pyright
    install(multi_root=True)
    svc = _service()
    try:
        monkeypatch.chdir(str(repo_a))
        diags_a = svc.get_diagnostics_sync(str(repo_a / "x.py"))
        monkeypatch.chdir(str(repo_b))
        diags_b = svc.get_diagnostics_sync(str(repo_b / "x.py"))

        # Exactly one process; the second root arrived as a folder change.
        assert spawns["value"] == 1
        assert len(svc._clients) == 1
        client = next(iter(svc._clients.values()))
        assert client.workspace_folders == [str(repo_a), str(repo_b)]
        events = [json.loads(line) for line in folders_log.read_text(encoding="utf-8").splitlines()]
        assert [f["uri"] for e in events for f in e["event"]["added"]] == [
            Path(repo_b).as_uri()
        ]
        # Diagnostics still resolve per file in both folders.
        assert len(diags_a) == 1 and len(diags_b) == 1
        status = svc.get_status()["clients"][0]
        assert status["workspace_root"] == str(repo_a)
        assert status["workspace_folders"] == [str(repo_a), str(repo_b)]
    finally:
        svc.shutdown()


def test_single_root_server_still_spawns_per_root(two_repos, mock_pyright, monkeypatch):
    repo_a, repo_b = two_repos
    spawns, folders_log, install = mock_pyright
    install(multi_root=False)
    svc = _service()
    try:
        monkeypatch.chdir(str(repo_a))
        svc.get_diagnostics_sync(str(repo_a / "x.py"))
        monkeypatch.chdir(str(repo_b))
        svc.get_diagnostics_sync(str(repo_b / "x.py"))
        assert spawns["value"] == 2
        assert set(svc._clients) == {("pyright", str(repo_a)), ("pyright", str(repo_b))}
        assert not folders_log.exists()
    finally:
        svc.shutdown()
