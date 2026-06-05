"""MCP self_improvement_snapshot integration tests."""

import json

import pytest


@pytest.fixture
def snapshot_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    repo.mkdir()
    home.mkdir()
    monkeypatch.setenv("HERMES_REPO", str(repo))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_AGENTS_DIR", raising=False)

    import hermes_skills_mcp as mcp

    monkeypatch.setattr(mcp, "_gateway_reachable", lambda: False)
    return repo, home, mcp


def _registered_tools(mcp_module):
    class _FakeMcp:
        def tool(self):
            def _decorator(fn):
                setattr(self, fn.__name__, fn)
                return fn

            return _decorator

    fake = _FakeMcp()
    mcp_module.register_skills_tools(fake)
    return fake


def test_self_improvement_snapshot_summary_empty_repo(snapshot_env):
    repo, _home, mcp = snapshot_env
    learnings = repo / ".learnings"
    learnings.mkdir()
    (learnings / "memory.md").write_text("# Memory\n", encoding="utf-8")

    result = mcp.build_self_improvement_snapshot(summary=True)

    assert result["format"] == "summary"
    assert result["writes_allowed"] is False
    assert result["status"] in ("ok", "attention")
    assert "recursive_loop" in result
    assert result["hot_tier"]["present"] is True


def test_self_improvement_snapshot_mcp_tool_registered(snapshot_env):
    _repo, _home, mcp = snapshot_env
    fake = _registered_tools(mcp)
    assert hasattr(fake, "self_improvement_snapshot")
    payload = json.loads(fake.self_improvement_snapshot(summary=True))
    assert payload["writes_allowed"] is False
