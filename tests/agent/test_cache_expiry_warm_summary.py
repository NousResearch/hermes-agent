"""The cache-expiry compaction's summary call reads the conversation's warm prompt cache: it goes
to the conversation's own model with the last request's system prompt, ``tools[]`` and messages
verbatim, and the summarize instruction appended as the final user message (only).

A real ``AIAgent`` runs turns against a capturing fake client; the compaction then runs through
``compress_now`` with the ``cache_expiry`` trigger, like every surface owner does."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression_manual import CompressRequest, compress_now
from run_agent import AIAgent

MODEL = "anthropic/claude-sonnet-4"
SUMMARY = "## Historical Task\nNone\n\n## Completed Actions\n1. Discussed fruit.\n"


def _usage():
    return SimpleNamespace(prompt_tokens=1200, completion_tokens=20, total_tokens=1220,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=1000))


def _response(content):
    msg = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model=MODEL, usage=_usage())


@pytest.fixture
def provider(monkeypatch):
    requests: list[dict] = []
    client = MagicMock()

    def _create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _response(SUMMARY if len(requests) > 6 else f"Fruit paragraph {len(requests)}. " + "detail " * 300)

    client.chat.completions.create.side_effect = _create
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    tools = [{"type": "function", "function": {"name": "terminal", "description": "terminal tool",
                                               "parameters": {"type": "object", "properties": {}}}}]
    patches = [
        patch("model_tools.get_tool_definitions", return_value=tools),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        # The old aux summary route, answered by the same fake so it is visible as a request too.
        patch("agent.context_compressor.call_llm", side_effect=lambda **kw: _create(messages=kw["messages"])),
    ]
    for p in patches:
        p.start()
    yield requests
    for p in reversed(patches):
        p.stop()


def _canon(value):
    """Collapse the text-block form a cache marker forces back to the string it wraps, and drop
    message-level markers (the rolling breakpoints legitimately move to the newest rows)."""
    if isinstance(value, list):
        items = [_canon(v) for v in value]
        if items and all(isinstance(i, dict) and set(i) == {"type", "text"} and i["type"] == "text" for i in items):
            return "".join(i["text"] for i in items)
        return items
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items() if k != "cache_control"}
    return value


def test_cache_expiry_summary_request_extends_the_last_conversation_request(provider, monkeypatch):
    monkeypatch.setattr("agent.cache_expiry_compaction.window_still_warm", lambda agent, **k: True, raising=False)
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model=MODEL, quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    agent._cached_system_prompt = "You are helpful."
    agent.save_trajectories, agent.skip_background_review = False, True
    compressor = agent.context_compressor
    compressor.tail_mode, compressor.protect_first_n, compressor.protect_last_n = "legacy", 1, 2
    compressor._tail_token_budget = 600
    assert agent._use_prompt_caching
    history: list = []
    for i in range(6):
        history = agent.run_conversation(f"Describe fruit #{i + 1}.", conversation_history=history)["messages"]
    last_turn = provider[-1]

    result = compress_now(agent, history, CompressRequest(), trigger="cache_expiry")

    assert result.status == "compressed" and len(provider) == 7
    summary_call = provider[-1]
    assert summary_call.get("model") == last_turn["model"]  # the conversation's own model and route
    # System prompt and tools[] (with their cache breakpoints and TTL) are byte-identical.
    assert summary_call["messages"][0] == last_turn["messages"][0]
    assert summary_call.get("tools") == last_turn["tools"]
    turn_msgs, summary_msgs = _canon(last_turn["messages"]), _canon(summary_call["messages"])
    assert summary_msgs[:len(turn_msgs)] == turn_msgs
    # Then only the last reply and the summarize instruction, as the final user message.
    assert [m["role"] for m in summary_msgs[len(turn_msgs):]] == ["assistant", "user"]
    assert "context checkpoint" in summary_msgs[-1]["content"]
    assert summary_call.get("max_tokens")  # output bounded by the compressor's summary budget


def test_an_explicitly_refused_warm_summary_never_replaces_history(monkeypatch):
    """A reply carrying both text and an explicit provider refusal is unusable on the warm path too."""
    from agent.context_compressor_summary import SummaryDispatchMixin
    from agent.transports.types import NormalizedResponse

    refused = NormalizedResponse(content=SUMMARY, tool_calls=None, finish_reason="stop",
                                 provider_data={"refusal": "I can't help with that."})
    monkeypatch.setattr("agent.side_question_request.ask_on_live_prefix", lambda *a, **k: (refused, {}))
    compressor = SimpleNamespace(_build_summary_prompt=lambda *a: [{"content": "i"}, {"content": "s"}],
                                 _compression_cancelled=lambda: False, tail_mode="legacy")
    warm = SimpleNamespace(agent=SimpleNamespace(model=MODEL), history=[])
    with pytest.raises(RuntimeError, match="refusal"):
        SummaryDispatchMixin._warm_prefix_summary(compressor, warm, [{"role": "user", "content": "x"}],
                                                  (2000, None, "", True))
