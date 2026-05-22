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


def test_fleet_context_snapshot_reports_missing_agents_dir(snapshot_env):
    _repo, _home, mcp = snapshot_env

    result = mcp.build_fleet_context_snapshot()

    assert result["mode"] == "skills_only"
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
