import json
import os
import time

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


def _write_registry(agents_dir, payload):
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "AGENT_REGISTRY.json").write_text(json.dumps(payload), encoding="utf-8")


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


def test_fleet_context_snapshot_reports_missing_agents_dir(snapshot_env):
    _repo, _home, mcp = snapshot_env

    result = mcp.build_fleet_context_snapshot()

    assert result["mode"] == "skills_only"
    assert result["writes_allowed"] is False
    assert result["source_of_truth"] is None
    assert result["authority_boundary"] == {
        "mode": "skills_only",
        "writes_allowed": False,
        "source_of_truth": None,
        "gateway_reachable": False,
    }
    assert result["gateway_reachable"] is False
    assert result["agents_dir"] is None
    assert result["registry_present"] is False
    assert result["agent_count"] == 0
    assert "agents_dir" in result["missing_layers"]
    assert "registry" in result["missing_layers"]
    assert result["warnings"]


def test_fleet_context_snapshot_registry_only(snapshot_env):
    repo, _home, mcp = snapshot_env
    agents_dir = repo / "agents"
    _write_registry(
        agents_dir,
        {
            "alpha": {"lane": "A", "status": "active", "authority": "observe_only"},
            "beta": {"lane": "B", "status": "paused", "authority": "write_artifacts"},
        },
    )

    result = mcp.build_fleet_context_snapshot()

    assert result["agents_dir"] == str(agents_dir)
    assert result["source_of_truth"] == "HERMES_REPO/agents"
    assert result["writes_allowed"] is False
    assert result["registry_present"] is True
    assert result["agent_count"] == 2
    assert result["registry_summary"]["by_lane"] == {"A": 1, "B": 1}
    assert result["registry_summary"]["by_status"] == {"active": 1, "paused": 1}
    assert {item["agent"] for item in result["stale_heartbeats"]} == {"alpha", "beta"}
    assert {item["status"] for item in result["stale_heartbeats"]} == {"missing"}


def test_fleet_context_snapshot_full_fleet_is_bounded(snapshot_env):
    repo, _home, mcp = snapshot_env
    agents_dir = repo / "agents"
    _write_registry(
        agents_dir,
        {
            "alpha": {"lane": "A", "status": "active", "authority": "observe_only"},
            "beta": {"lane": "B", "status": "active", "authority": "write_artifacts"},
        },
    )
    for name in ("alpha", "beta"):
        agent_dir = agents_dir / name
        agent_dir.mkdir()
        (agent_dir / "SOUL.md").write_text(f"# {name}\n", encoding="utf-8")
        heartbeat = agent_dir / "HEARTBEAT.md"
        heartbeat.write_text("ok\n", encoding="utf-8")
    stale_ts = time.time() - (mcp._HEARTBEAT_STALE_SECONDS + 3600)
    os.utime(agents_dir / "beta" / "HEARTBEAT.md", (stale_ts, stale_ts))

    learnings = repo / ".learnings"
    learnings.mkdir()
    (learnings / "memory.md").write_text("hot\n" + ("x" * 10_000), encoding="utf-8")

    knowledge = repo / "artifacts" / "ops" / "knowledge_layer"
    knowledge.mkdir(parents=True)
    (knowledge / "latest_state.md").write_text("state\n" + ("y" * 10_000), encoding="utf-8")

    held = repo / "artifacts" / "ops" / "held_spec_ledger"
    held.mkdir(parents=True)
    (held / "latest.md").write_text("HELD: no ranking changes\nordinary note\n", encoding="utf-8")

    result = mcp.build_fleet_context_snapshot()

    assert result["agent_count"] == 2
    assert result["stale_heartbeats"][0]["agent"] == "beta"
    assert result["stale_heartbeats"][0]["status"] == "stale"
    assert result["hot_learnings_excerpt"]["truncated"] is True
    assert len(result["hot_learnings_excerpt"]["content"]) == mcp._SNAPSHOT_TEXT_CAP
    assert result["latest_state_digest"]["truncated"] is True
    assert len(result["latest_state_digest"]["content"]) == mcp._SNAPSHOT_TEXT_CAP
    assert result["held_spec_flags"] == ["HELD: no ranking changes"]
    assert result["source_of_truth_hierarchy"]["version"]


def test_fleet_context_snapshot_gateway_unavailable_keeps_skills_mode(snapshot_env, monkeypatch):
    repo, _home, mcp = snapshot_env
    _write_registry(repo / "agents", {"alpha": {"status": "active"}})
    monkeypatch.setattr(mcp, "_gateway_reachable", lambda: False)

    result = mcp.build_fleet_context_snapshot()

    assert result["gateway_reachable"] is False
    assert result["mode"] == "skills_only"
    assert result["authority_boundary"]["writes_allowed"] is False


def test_skills_list_supports_external_hermes_agents_dir(tmp_path, monkeypatch):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    agents_dir = tmp_path / "product" / "agents"
    agent_dir = agents_dir / "alpha"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("# alpha\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_REPO", str(repo))
    monkeypatch.setenv("HERMES_AGENTS_DIR", str(agents_dir))

    import hermes_skills_mcp as mcp
    tools = _registered_tools(mcp)

    result = json.loads(tools.skills_list("agents"))

    assert result["paths"]["agents_dir"] == str(agents_dir)
    assert result["agents_skills"][0]["agent"] == "alpha"
    assert result["agents_skills"][0]["path"] == str(agent_dir)


def test_skills_context_tools_reject_path_escape(snapshot_env):
    repo, _home, mcp = snapshot_env
    agents_dir = repo / "agents"
    agent_dir = agents_dir / "alpha"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("# alpha\n", encoding="utf-8")
    (repo / ".learnings").mkdir()
    (repo / "artifacts").mkdir()
    tools = _registered_tools(mcp)

    assert "Invalid agent path" in tools.skills_read("../..", "etc/passwd")
    assert "Invalid learnings path" in tools.learnings_read("../config.yaml")
    assert "Invalid artifacts path" in tools.artifacts_list("../")


def test_agents_tools_support_nested_registry_shape(snapshot_env):
    repo, _home, mcp = snapshot_env
    agents_dir = repo / "agents"
    _write_registry(
        agents_dir,
        {"agents": {"alpha": {"lane": "A", "status": "active"}}},
    )
    (agents_dir / "alpha").mkdir()
    (agents_dir / "alpha" / "SOUL.md").write_text("# alpha\n", encoding="utf-8")
    tools = _registered_tools(mcp)

    listed = json.loads(tools.agents_list())
    detail = json.loads(tools.agents_get("alpha"))

    assert listed["count"] == 1
    assert listed["agents"][0]["name"] == "alpha"
    assert detail["registry"] == {"lane": "A", "status": "active"}


def test_fleet_context_snapshot_explicit_agents_dir_is_source_of_truth(snapshot_env, monkeypatch):
    _repo, _home, mcp = snapshot_env
    runtime_agents = _repo / "runtime-agents"
    _write_registry(runtime_agents, {"alpha": {"status": "active"}})
    monkeypatch.setenv("HERMES_AGENTS_DIR", str(runtime_agents))

    result = mcp.build_fleet_context_snapshot()

    assert result["agents_dir"] == str(runtime_agents)
    assert result["source_of_truth"] == "HERMES_AGENTS_DIR"
    assert result["authority_boundary"] == {
        "mode": "skills_only",
        "writes_allowed": False,
        "source_of_truth": "HERMES_AGENTS_DIR",
        "gateway_reachable": False,
    }


def test_agent_health_summary_reports_actionable_anomalies(snapshot_env):
    repo, _home, mcp = snapshot_env
    agents_dir = repo / "agents"
    _write_registry(
        agents_dir,
        {
            "alpha": {"status": "active"},
            "beta": {"status": "active"},
        },
    )
    beta = agents_dir / "beta"
    beta.mkdir()
    (beta / "HEARTBEAT.md").write_text("ok\n", encoding="utf-8")
    stale_ts = time.time() - (mcp._HEARTBEAT_STALE_SECONDS + 3600)
    os.utime(beta / "HEARTBEAT.md", (stale_ts, stale_ts))

    summary = mcp.build_agent_health_summary()

    assert summary["status"] == "attention"
    assert summary["mode"] == "skills_only"
    assert summary["writes_allowed"] is False
    assert summary["registry_present"] is True
    assert summary["agent_count"] == 2
    assert {item["agent"] for item in summary["stale_heartbeats"]} == {"alpha", "beta"}
    assert "latest_state" in summary["missing_layers"]
    assert summary["next_action"]


def test_agent_health_summary_registered_tool(snapshot_env):
    repo, _home, mcp = snapshot_env
    _write_registry(repo / "agents", {"alpha": {"status": "active"}})
    tools = _registered_tools(mcp)

    result = json.loads(tools.agent_health_summary())

    assert result["mode"] == "skills_only"
    assert result["agent_count"] == 1
    assert result["writes_allowed"] is False
