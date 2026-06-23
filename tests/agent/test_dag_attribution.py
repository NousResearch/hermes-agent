"""
Tests for hermes_agent.prompts.dag_attribution
"""

from __future__ import annotations

import json
from typing import List

import pytest

from hermes_agent.prompts.dag_attribution import (
    DagNode,
    Observation,
    build_attribution_prompt,
    build_system_message,
    parse_attribution_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_nodes() -> List[DagNode]:
    return [
        DagNode(
            node_id="n1",
            label="Fetch user profile",
            inputs={"user_id": 42},
            output={"name": "Alice", "plan": "pro"},
        ),
        DagNode(
            node_id="n2",
            label="Call billing API",
            inputs={"plan": "pro", "action": "charge"},
            output={"status": "ok", "charge_id": "ch_001"},
            parent_ids=["n1"],
        ),
        DagNode(
            node_id="n3",
            label="Send confirmation email",
            inputs={"email": "alice@example.com", "charge_id": "ch_001"},
            output=None,
            parent_ids=["n1", "n2"],
        ),
    ]


@pytest.fixture()
def simple_observations() -> List[Observation]:
    return [
        Observation(
            observation_id="obs1",
            description="User reported they never received a confirmation email.",
            sentiment="negative",
        ),
        Observation(
            observation_id="obs2",
            description="Charge appeared correctly on the user's statement.",
            sentiment="positive",
            raw_value=True,
        ),
    ]


# ---------------------------------------------------------------------------
# DagNode tests
# ---------------------------------------------------------------------------


class TestDagNode:
    def test_defaults(self):
        node = DagNode(node_id="x", label="do something")
        assert node.inputs == {}
        assert node.output is None
        assert node.parent_ids == []
        assert node.metadata == {}

    def test_with_values(self):
        node = DagNode(
            node_id="y",
            label="compute",
            inputs={"a": 1},
            output=42,
            parent_ids=["x"],
            metadata={"model": "gpt-4o"},
        )
        assert node.inputs == {"a": 1}
        assert node.output == 42
        assert node.parent_ids == ["x"]
        assert node.metadata["model"] == "gpt-4o"

    def test_mutable_defaults_are_independent(self):
        n1 = DagNode(node_id="a", label="a")
        n2 = DagNode(node_id="b", label="b")
        n1.parent_ids.append("root")
        assert n2.parent_ids == [], "mutable defaults must not be shared"


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------


class TestObservation:
    def test_defaults(self):
        obs = Observation(observation_id="o1", description="something happened")
        assert obs.sentiment is None
        assert obs.raw_value is None

    def test_with_values(self):
        obs = Observation(
            observation_id="o2",
            description="error occurred",
            sentiment="negative",
            raw_value={"code": 500},
        )
        assert obs.sentiment == "negative"
        assert obs.raw_value == {"code": 500}


# ---------------------------------------------------------------------------
# build_attribution_prompt tests
# ---------------------------------------------------------------------------


class TestBuildAttributionPrompt:
    def test_returns_string(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_node_ids(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        for node in simple_nodes:
            assert node.node_id in prompt, f"node_id {node.node_id!r} missing from prompt"

    def test_contains_node_labels(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        for node in simple_nodes:
            assert node.label in prompt, f"label {node.label!r} missing from prompt"

    def test_contains_observation_ids(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        for obs in simple_observations:
            assert obs.observation_id in prompt

    def test_contains_observation_descriptions(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        for obs in simple_observations:
            assert obs.description in prompt

    def test_system_preamble_included_by_default(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        # The preamble mentions "Directed Acyclic Graph"
        assert "Directed Acyclic Graph" in prompt

    def test_system_preamble_excluded(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(
            simple_nodes, simple_observations, include_system_preamble=False
        )
        assert "Directed Acyclic Graph" not in prompt

    def test_extra_context_included(self, simple_nodes, simple_observations):
        ctx = "The user asked to upgrade their subscription."
        prompt = build_attribution_prompt(
            simple_nodes, simple_observations, extra_context=ctx
        )
        assert ctx in prompt

    def test_extra_context_absent_when_not_provided(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        assert "Additional Context" not in prompt

    def test_empty_nodes(self, simple_observations):
        prompt = build_attribution_prompt([], simple_observations)
        assert "no nodes provided" in prompt

    def test_empty_observations(self, simple_nodes):
        prompt = build_attribution_prompt(simple_nodes, [])
        assert "no observations provided" in prompt

    def test_parent_ids_rendered(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        # n2 has parent n1
        assert "n1" in prompt

    def test_root_node_label(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        assert "root node" in prompt

    def test_sentiment_rendered(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        assert "negative" in prompt
        assert "positive" in prompt

    def test_instructions_present(self, simple_nodes, simple_observations):
        prompt = build_attribution_prompt(simple_nodes, simple_observations)
        assert "attributed_node_ids" in prompt
        assert "reasoning" in prompt

    def test_node_with_metadata(self, simple_observations):
        node = DagNode(
            node_id="m1",
            label="meta node",
            metadata={"model": "gpt-4o", "latency_ms": 320},
        )
        prompt = build_attribution_prompt([node], simple_observations)
        assert "gpt-4o" in prompt
        assert "latency_ms" in prompt

    def test_node_output_rendered(self, simple_observations):
        node = DagNode(node_id="o1", label="output node", output={"key": "value"})
        prompt = build_attribution_prompt([node], simple_observations)
        assert "key" in prompt

    def test_observation_raw_value_rendered(self, simple_nodes):
        obs = Observation(
            observation_id="rv1",
            description="raw value test",
            raw_value={"score": 0.95},
        )
        prompt = build_attribution_prompt(simple_nodes, [obs])
        assert "0.95" in prompt


# ---------------------------------------------------------------------------
# build_system_message tests
# ---------------------------------------------------------------------------


class TestBuildSystemMessage:
    def test_returns_string(self):
        msg = build_system_message()
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_mentions_dag(self):
        msg = build_system_message()
        assert "DAG" in msg or "Directed Acyclic Graph" in msg

    def test_no_trailing_whitespace(self):
        msg = build_system_message()
        assert msg == msg.strip()


# ---------------------------------------------------------------------------
# parse_attribution_response tests
# ---------------------------------------------------------------------------


class TestParseAttributionResponse:
    def _make_response(self, items):
        return json.dumps(items)

    def test_parses_valid_response(self):
        items = [
            {
                "observation_id": "obs1",
                "attributed_node_ids": ["n3"],
                "reasoning": "Node n3 is responsible for sending the email.",
            }
        ]
        result = parse_attribution_response(self._make_response(items))
        assert len(result) == 1
        assert result[0]["observation_id"] == "obs1"
        assert result[0]["attributed_node_ids"] == ["n3"]

    def test_parses_multiple_items(self):
        items = [
            {
                "observation_id": "obs1",
                "attributed_node_ids": ["n3"],
                "reasoning": "Email node failed.",
            },
            {
                "observation_id": "obs2",
                "attributed_node_ids": ["n2"],
                "reasoning": "Billing node succeeded.",
            },
        ]
        result = parse_attribution_response(self._make_response(items))
        assert len(result) == 2

    def test_parses_response_with_markdown_fences(self):
        items = [{"observation_id": "o1", "attributed_node_ids": [], "reasoning": "none"}]
        wrapped = f"```json\n{json.dumps(items)}\n```"
        result = parse_attribution_response(wrapped)
        assert len(result) == 1

    def test_parses_response_with_plain_code_fence(self):
        items = [{"observation_id": "o1", "attributed_node_ids": ["n1"], "reasoning": "ok"}]
        wrapped = f"```\n{json.dumps(items)}\n```"
        result = parse_attribution_response(wrapped)
        assert len(result) == 1
        assert result[0]["attributed_node_ids"] == ["n1"]

    def test_returns_empty_list_on_invalid_json(self):
        result = parse_attribution_response("this is not json at all")
        assert result == []

    def test_returns_empty_list_on_empty_string(self):
        result = parse_attribution_response("")
        assert result == []

    def test_returns_empty_list_when_no_array_found(self):
        result = parse_attribution_response('{"key": "value"}')
        assert result == []

    def test_empty_array_response(self):
        result = parse_attribution_response("[]")
        assert result == []

    def test_empty_attributed_node_ids(self):
        items = [
            {
                "observation_id": "obs_x",
                "attributed_node_ids": [],
                "reasoning": "No node is responsible.",
            }
        ]
        result = parse_attribution_response(self._make_response(items))
        assert result[0]["attributed_node_ids"] == []

    def test_multiple_attributed_nodes(self):
        items = [
            {
                "observation_id": "obs_y",
                "attributed_node_ids": ["n1", "n2", "n3"],
                "reasoning": "All three nodes contributed.",
            }
        ]
        result = parse_attribution_response(self._make_response(items))
        assert result[0]["attributed_node_ids"] == ["n1", "n2", "n3"]

    def test_response_with_preamble_text(self):
        """LLM sometimes adds a sentence before the JSON array."""
        items = [{"observation_id": "o1", "attributed_node_ids": ["n1"], "reasoning": "yes"}]
        text = f"Here is my analysis:\n{json.dumps(items)}\nHope that helps!"
        result = parse_attribution_response(text)
        assert len(result) == 1
