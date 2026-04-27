"""Tests for the local Brain memory provider plugin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    brain_repo = tmp_path / "brain"
    (brain_repo / "src").mkdir(parents=True)
    (brain_repo / "src" / "index.js").write_text("export {};\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BRAIN_REPO", str(brain_repo))
    return hermes_home, brain_repo


def test_provider_is_discoverable_and_available(_isolate_env):
    from plugins.memory import discover_memory_providers, load_memory_provider
    from plugins.memory.brain import BrainMemoryProvider

    discovered = {name: available for name, _desc, available in discover_memory_providers()}
    assert discovered["brain"] is True
    provider = load_memory_provider("brain")
    assert isinstance(provider, BrainMemoryProvider)
    assert provider.is_available() is True


def test_sync_turn_persists_events_and_runs_brain_experiment(_isolate_env):
    hermes_home, _brain_repo = _isolate_env
    calls = []

    def fake_runner(payload):
        calls.append(payload)
        return {
            "status": "completed",
            "longTermCandidates": [
                {"memoryId": "turn-1-user", "content": "User wants Hermes Brain connected.", "score": 0.9}
            ],
            "pageRank": {"maxIterations": 90, "iterationsCompleted": 3},
        }

    from plugins.memory.brain import BrainMemoryProvider

    provider = BrainMemoryProvider(config={"auto_consolidate": True}, experiment_runner=fake_runner)
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="discord")
    provider.sync_turn("User wants Hermes Brain connected.", "I will connect it.", session_id="session-1")

    assert calls
    assert calls[-1]["agentId"] == "hermes"
    assert calls[-1]["iterations"] == 90
    assert calls[-1]["runtime"] == {"phase": "idle", "authority": "caller"}
    assert calls[-1]["hippocampus"] == {"enabled": True}
    assert len(calls[-1]["events"]) == 2

    state_path = hermes_home / "brain" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_experiment"]["status"] == "completed"
    assert state["long_term_candidates"][0]["content"] == "User wants Hermes Brain connected."


def test_provider_filters_polluted_events_before_persisting_or_running(_isolate_env):
    hermes_home, _brain_repo = _isolate_env
    calls = []

    def fake_runner(payload):
        calls.append(payload)
        return {"status": "completed", "longTermCandidates": []}

    from plugins.memory.brain import BrainMemoryProvider

    provider = BrainMemoryProvider(config={"auto_consolidate": False}, experiment_runner=fake_runner)
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="discord")
    provider.sync_turn(
        "Review the conversation above and consider whether a skill should be saved.",
        "Skill update summary: nothing changed.",
        session_id="session-1",
    )
    provider.sync_turn(
        "<memory-context>old recalled answer</memory-context>\n[JQ] this should not feed back",
        "Brain provider connected as local Hermes memory sidecar.",
        session_id="session-1",
    )
    provider.on_memory_write("add", "memory", "User prefers project-aware Gmail labels.")
    provider.on_session_end([
        {"role": "tool", "content": '{"success": true, "name": "hermes-agent", "content": "raw tool dump"}'},
    ])
    provider._run_and_store(iterations=90, top_k=5)

    state = json.loads((hermes_home / "brain" / "state.json").read_text(encoding="utf-8"))
    persisted = [event["content"] for event in state["events"]]
    assert persisted == ["[JQ] this should not feed back", "User prefers project-aware Gmail labels."]
    assert [event["content"] for event in calls[-1]["events"]] == persisted


def test_prefetch_returns_brain_ranked_context_from_persisted_events(_isolate_env):
    hermes_home, _brain_repo = _isolate_env

    def fake_runner(payload):
        return {
            "status": "completed",
            "longTermCandidates": [
                {"memoryId": "mem-1", "content": "SNUTI should be remembered via Calendar context.", "score": 0.8},
                {"memoryId": "mem-2", "content": "Gmail labels should be project-aware.", "score": 0.7},
            ],
            "pageRank": {"maxIterations": payload["iterations"], "iterationsCompleted": 2},
        }

    from plugins.memory.brain import BrainMemoryProvider

    provider = BrainMemoryProvider(config={"top_k": 2}, experiment_runner=fake_runner)
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="discord")
    provider.on_memory_write("add", "memory", "SNUTI should be remembered via Calendar context.")

    context = provider.prefetch("What do we know about SNUTI?", session_id="session-1")

    assert "## Brain Memory" in context
    assert "SNUTI should be remembered" in context
    assert "90-iteration" in context


def test_brain_tool_status_and_search_use_provider_state(_isolate_env):
    hermes_home, _brain_repo = _isolate_env

    def fake_runner(_payload):
        return {
            "status": "completed",
            "hippocampus": {"enabled": True},
            "graph": {
                "nodes": [{"memoryId": "mem-1"}],
                "edges": [{"from": "mem-1", "to": "mem-2", "relation": "shared-concept"}],
            },
            "longTermCandidates": [
                {"memoryId": "mem-1", "content": "Hermes Brain adapter is active.", "score": 0.9}
            ],
            "pageRank": {"maxIterations": 90, "iterationsCompleted": 1},
        }

    from plugins.memory.brain import BrainMemoryProvider

    provider = BrainMemoryProvider(experiment_runner=fake_runner)
    provider.initialize("session-1", hermes_home=str(hermes_home), platform="discord")
    provider.on_memory_write("add", "memory", "Hermes Brain adapter is active.")

    status = json.loads(provider.handle_tool_call("brain_status", {}))
    assert status["provider"] == "brain"
    assert status["events"] == 1
    assert status["brain_repo_available"] is True
    assert status["hippocampus_enabled"] is True
    assert status["last_graph_nodes"] == 1
    assert status["last_graph_edges"] == 1

    search = json.loads(provider.handle_tool_call("brain_search", {"query": "adapter", "top_k": 1}))
    assert search["count"] == 1
    assert search["results"][0]["content"] == "Hermes Brain adapter is active."
