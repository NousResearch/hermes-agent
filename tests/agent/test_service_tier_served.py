"""Requested vs SERVED service tier: the turn result and ``--usage-file`` carry both, and a paid
tier the endpoint did not serve is logged. The requested tier is read where ``/fast`` puts it
(top-level ``service_tier``), not only from ``extra_body``."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import fast_mode

OPENAI = dict(provider="openai", base_url="https://api.openai.com/v1", api_mode="codex_responses")
CODEX = dict(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses")


def _agent(request_overrides, route=OPENAI, tier=None):
    return SimpleNamespace(model="gpt-6-astra", service_tier=tier, request_overrides=request_overrides, **route)


@pytest.mark.parametrize("overrides,expected", [
    ({"service_tier": "ultrafast"}, "ultrafast"),                 # what /fast sends on OpenAI
    ({"extra_body": {"service_tier": "flex"}}, "flex"),           # custom provider body
    ({"extra_body": {"keep": 1}}, None),
    ({}, None),
])
def test_requested_tier_reads_the_wire_shape(overrides, expected):
    assert fast_mode.requested_service_tier(_agent(overrides)) == expected


def test_downgrade_is_recorded_and_warned(caplog):
    agent = _agent({"service_tier": "ultrafast"})
    with caplog.at_level(logging.DEBUG, logger="agent.fast_mode"):
        fast_mode.record_served_service_tier(agent, SimpleNamespace(service_tier="default"))
    assert (agent._last_requested_service_tier, agent._last_served_service_tier) == ("ultrafast", "default")
    assert [r.levelname for r in caplog.records] == ["WARNING"]


def test_codex_echo_is_recorded_but_not_judged(caplog):
    agent = _agent({"service_tier": "ultrafast"}, route=CODEX)
    with caplog.at_level(logging.DEBUG, logger="agent.fast_mode"):
        fast_mode.record_served_service_tier(agent, SimpleNamespace(service_tier="default"))
    assert agent._last_served_service_tier == "default"
    assert all(r.levelno < logging.WARNING for r in caplog.records)


def test_served_tier_is_per_call_and_a_match_is_silent(caplog):
    agent = _agent({"service_tier": "priority"})
    with caplog.at_level(logging.DEBUG, logger="agent.fast_mode"):
        fast_mode.record_served_service_tier(agent, SimpleNamespace(service_tier="Priority"))
        assert agent._last_served_service_tier == "priority"
        fast_mode.record_served_service_tier(agent, SimpleNamespace())  # later tier-less response
    assert agent._last_served_service_tier is None
    assert not caplog.records


class _Stream:
    def __init__(self, served):
        msg = SimpleNamespace(type="message", id="msg_1", role="assistant", status="completed",
                              content=[SimpleNamespace(type="output_text", text="hello", annotations=[])])
        usage = SimpleNamespace(input_tokens=10, output_tokens=2, total_tokens=12,
                                input_tokens_details=SimpleNamespace(cached_tokens=0),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=0))
        self._events = [
            SimpleNamespace(type="response.output_item.added", output_index=0, item=msg),
            SimpleNamespace(type="response.output_text.delta", item_id="msg_1", delta="hello"),
            SimpleNamespace(type="response.output_item.done", output_index=0, item=msg),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                id="resp_1", status="completed", usage=usage, service_tier=served, output=[msg])),
        ]

    def __iter__(self):
        return iter(self._events)

    def close(self):
        pass


def test_turn_result_carries_requested_and_served_tier(tmp_path):
    """Real turn on a Responses route: /fast's top-level tier is requested, the backend serves less."""
    import run_agent
    from hermes_cli.oneshot import _write_usage_file

    agent = run_agent.AIAgent(model="gpt-6-astra", api_key="test-key", quiet_mode=True, skip_context_files=True,
                              skip_memory=True, session_db=MagicMock(), enabled_toolsets=["terminal"],
                              service_tier="ultrafast", request_overrides={"service_tier": "ultrafast"}, **OPENAI)
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return _Stream("default")

    agent.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    agent._create_request_openai_client = lambda *a, **k: agent.client
    result = agent.run_conversation("hi")

    assert sent["service_tier"] == result["service_tier"] == "ultrafast"
    assert result["service_tier_served"] == "default"
    path = tmp_path / "usage.json"
    _write_usage_file(str(path), result)
    report = json.loads(path.read_text())
    assert (report["service_tier"], report["service_tier_served"]) == ("ultrafast", "default")
