"""Bedrock dynamic hooks must not turn otherwise hidden deltas into spinner events."""
import pytest

from agent import chat_completion_helpers as cch
from run_agent import AIAgent


def test_bedrock_dynamic_delta_spinner(monkeypatch):
    from agent import relay_llm
    from agent import plugin_stream_hooks

    agent = AIAgent(api_key="test", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                    model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._interrupt_requested = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    delivered = []
    monkeypatch.setattr(agent, "_enqueue_stream_hook", lambda event, **kw: delivered.append((event, kw)))
    active = {"text": False, "reasoning": False}
    monkeypatch.setattr(agent, "_has_stream_consumers", lambda: active["text"] or agent.stream_delta_callback is not None)
    monkeypatch.setattr(plugin_stream_hooks, "has_reasoning_stream_observer_hooks", lambda: active["reasoning"])
    monkeypatch.setattr(plugin_stream_hooks, "stream_reasoning_deltas_enabled", lambda: True)
    stopped = []

    def event(delta):
        return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": delta}}

    class Stream:
        final_response = None
        def __iter__(self):
            yield event({"text": "hidden"})
            yield event({"reasoningContent": {"text": "thought"}})
            assert not stopped
            active["text"] = True
            yield event({"text": "observed"})
            assert not stopped  # Upstream snapshots spinner eligibility at stream start.
            yield {"messageStop": {"stopReason": "end_turn"}}
        def close(self):
            pass

    monkeypatch.setattr(relay_llm, "stream", lambda *args, **kwargs: Stream())
    call = cch._BedrockStream(agent, {}, lambda: stopped.append(True))
    call._worker()
    assert call.result["error"] is None
    assert [kw["delta"] for event, kw in delivered if event == "on_stream_delta"] == ["hidden", "thought", "observed"]


@pytest.mark.parametrize("delta", [
    {"reasoningContent": {"text": "thought"}},
    {"text": "<think>thought</think>"},
    {"text": "\n"},
])
def test_bedrock_reasoning_and_inline_keep_upstream_spinner_timing(monkeypatch, delta):
    from agent import relay_llm
    from agent import plugin_stream_hooks

    agent = AIAgent(api_key="test", base_url="https://openrouter.ai/api/v1", provider="openrouter",
                    model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent._interrupt_requested = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = lambda text: None
    monkeypatch.setattr(agent, "_enqueue_stream_hook", lambda *args, **kwargs: None)
    monkeypatch.setattr(plugin_stream_hooks, "has_reasoning_stream_observer_hooks", lambda: False)
    stopped = []

    class Stream:
        final_response = None
        def __iter__(self):
            yield {"contentBlockDelta": {"delta": delta}}
            assert stopped == [True]
            yield {"messageStop": {"stopReason": "end_turn"}}
        def close(self):
            pass

    monkeypatch.setattr(relay_llm, "stream", lambda *args, **kwargs: Stream())
    call = cch._BedrockStream(agent, {}, lambda: stopped.append(True))
    call._worker()
    assert call.result["error"] is None
