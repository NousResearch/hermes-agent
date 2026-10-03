"""Every billed provider response is counted, including truncated and refused ones.

``finish_reason == "length"`` (continuation fragments, thinking-budget exhaustion) and
``"content_filter"`` responses are completed, billed provider calls; the session's token,
cost and API-call counters must include them like any other response.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _response(content, finish_reason, prompt, completion):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model",
        usage=None if prompt is None else SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
    )


def _make_agent(monkeypatch, **kwargs):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, **kwargs)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    return a


@pytest.fixture()
def agent(monkeypatch):
    return _make_agent(monkeypatch)


@pytest.mark.parametrize("responses", [
    pytest.param([_response("Part 1 ", "length", 5000, 4096), _response("Part 2", "stop", 9200, 300)],
                 id="length-continuation"),
    pytest.param([_response("<think>" + "x" * 200 + "</think>", "length", 20000, 32000)],
                 id="thinking-budget-exhausted"),
    pytest.param([_response("", "content_filter", 7000, 5)], id="content-filter-refusal"),
])
def test_session_counters_match_every_billed_response(agent, responses):
    agent.client.chat.completions.create.side_effect = list(responses)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation("hello")

    assert agent.client.chat.completions.create.call_count == len(responses)
    assert agent.session_api_calls == len(responses)
    assert agent.session_prompt_tokens == sum(r.usage.prompt_tokens for r in responses)
    assert agent.session_completion_tokens == sum(r.usage.completion_tokens for r in responses)


@pytest.mark.parametrize("responses", [
    pytest.param([_response("Part 1 ", "length", None, None), _response("Part 2", "stop", 9200, 300)],
                 id="usage-less-length-fragment"),
    pytest.param([_response("", "content_filter", None, None)], id="usage-less-refusal"),
])
def test_state_db_counts_every_call_including_usage_less_ones(monkeypatch, tmp_path, responses):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    agent = _make_agent(monkeypatch, session_db=db, session_id="usage-less-calls")
    agent.client.chat.completions.create.side_effect = list(responses)
    with patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"):
        agent.run_conversation("hello")
    assert db.flush_token_counts()

    row = db.get_session("usage-less-calls")
    assert row["api_call_count"] == agent.session_api_calls == len(responses)
