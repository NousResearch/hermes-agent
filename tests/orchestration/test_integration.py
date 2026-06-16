"""Integration test: end-to-end agent routing and context lifecycle.

Simulates the complete message-dispatch chain without touching the
Gateway core — proves that AgentRoutingTable + AgentContextPool
correctly route incoming messages to the right agent and persist
pseudo-conversational state at turn boundaries.
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def orchestration_home(tmp_path, monkeypatch):
    """Redirect ~/.hermes/ to a temp directory with pre-seeded agents."""
    home = tmp_path / ".hermes"
    home.mkdir()

    # patch the home path for all three modules
    monkeypatch.setattr(
        "src.orchestration.agent_context.get_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "src.orchestration.agent_pool.get_hermes_home", lambda: home
    )

    # seed two agents on disk
    agents_dir = home / "agents"
    for agent_id in ("coding-agent", "cs-agent"):
        ws = agents_dir / agent_id
        ws.mkdir(parents=True)
        (ws / "MEMORY.md").write_text(f"role: {agent_id}\n")
        (ws / "state.json").write_text("{}")

    # routing config
    routing_yaml = {
        "routing": [
            {"topic": "telegram:-100X:101", "agent": "coding-agent"},
            {"topic": "telegram:-100X:102", "agent": "coding-agent"},
            {"topic": "telegram:-100X:201", "agent": "cs-agent"},
            {"dm": "telegram:user123", "agent": "cs-agent"},
        ],
        "default_agent": "coding-agent",
    }
    (home / "agent_routing.yaml").write_text(yaml.dump(routing_yaml))

    return home


@pytest.fixture
def routing_table(orchestration_home):
    from src.orchestration.agent_routing import AgentRoutingTable

    return AgentRoutingTable(
        config_path=orchestration_home / "agent_routing.yaml",
        default_agent="coding-agent",
    )


@pytest.fixture
def pool(orchestration_home):
    from src.orchestration.agent_pool import AgentContextPool

    return AgentContextPool(agents_dir=orchestration_home / "agents")


# ── simulate a message dispatch ──────────────────────────────────────


def _dispatch(
    platform: str,
    chat_id: str,
    topic_id: str | None,
    routing_table,
    pool,
    response_content: str,
):
    """Simulate one complete message turn.

    Returns the resolved agent id and the updated AgentContext.
    """
    agent_id = routing_table.resolve(platform, chat_id, topic_id)
    ctx = pool.get_or_create(agent_id)

    # Simulate: the agent processes the message and updates state
    turn = ctx.state.get("turn_count", 0) + 1
    ctx.state["turn_count"] = turn
    ctx.state["last_response"] = response_content
    ctx.state.setdefault("history", []).append(
        {"topic_id": topic_id, "turn": turn, "response": response_content}
    )

    # Turn boundary: snapshot state
    pool.snapshot(agent_id)
    return agent_id, ctx


# ── tests ────────────────────────────────────────────────────────────


class TestOrchestrationIntegration:

    # ── routing correctness ───────────────────────────────────────

    def test_topic_routes_to_coding_agent(self, routing_table, pool):
        """Message in coding topic → coding-agent."""
        agent_id, ctx = _dispatch(
            "telegram", "-100X", "101", routing_table, pool,
            "[coding] fixed the bug",
        )
        assert agent_id == "coding-agent"
        assert ctx.memory.get("role") == "coding-agent"
        assert ctx.state["turn_count"] == 1

    def test_topic_routes_to_cs_agent(self, routing_table, pool):
        """Message in support topic → cs-agent."""
        agent_id, ctx = _dispatch(
            "telegram", "-100X", "201", routing_table, pool,
            "[cs] your ticket has been resolved",
        )
        assert agent_id == "cs-agent"
        assert ctx.memory.get("role") == "cs-agent"
        assert ctx.state["turn_count"] == 1

    def test_unknown_topic_falls_back_to_default(self, routing_table, pool):
        """Topic not in the table → default_agent."""
        agent_id, ctx = _dispatch(
            "telegram", "-100X", "999", routing_table, pool,
            "i don't know where this goes",
        )
        assert agent_id == "coding-agent"  # default

    # ── multi-turn state ──────────────────────────────────────────

    def test_multi_turn_conversation_in_same_topic(self, routing_table, pool):
        """Multiple messages in the same topic accumulate state."""
        for i in range(5):
            _dispatch(
                "telegram", "-100X", "101", routing_table, pool,
                f"message {i}",
            )

        ctx = pool.get_or_create("coding-agent")
        assert ctx.state["turn_count"] == 5
        assert len(ctx.state["history"]) == 5

    # ── cross-agent isolation ─────────────────────────────────────

    def test_coding_agent_does_not_see_cs_state(self, routing_table, pool):
        """Agent isolation: coding agent's state is separate from cs agent's."""
        _dispatch("telegram", "-100X", "101", routing_table, pool, "code")
        _dispatch("telegram", "-100X", "101", routing_table, pool, "code2")
        _dispatch("telegram", "-100X", "201", routing_table, pool, "support")

        coding = pool.get_or_create("coding-agent")
        cs = pool.get_or_create("cs-agent")

        assert coding.state["turn_count"] == 2
        assert cs.state["turn_count"] == 1
        # coding should NOT see cs-agent's messages
        coding_topics = {h["topic_id"] for h in coding.state["history"]}
        assert coding_topics == {"101"}

    # ── state persistence across pool reload ──────────────────────

    def test_state_survives_pool_reload(self, routing_table, pool):
        """Agent state persists on disk and survives pool recreation."""
        _dispatch("telegram", "-100X", "101", routing_table, pool, "stuff")

        from src.orchestration.agent_pool import AgentContextPool

        pool2 = AgentContextPool(agents_dir=pool._agents_dir)
        ctx2 = pool2.get_or_create("coding-agent")
        assert ctx2.state["turn_count"] == 1
        assert ctx2.state["last_response"] == "stuff"

    # ── hot reload routing ────────────────────────────────────────

    def test_routing_hot_reload(self, routing_table, pool):
        """Changing the YAML mid-flight redirects new messages."""
        assert routing_table.resolve("telegram", "-100X", "899") == "coding-agent"

        # update routing table on disk
        new_config = {
            "routing": [
                {"topic": "telegram:-100X:899", "agent": "cs-agent"},
            ],
            "default_agent": "coding-agent",
        }
        routing_table.config_path.write_text(yaml.dump(new_config))

        assert routing_table.resolve("telegram", "-100X", "899") == "cs-agent"
