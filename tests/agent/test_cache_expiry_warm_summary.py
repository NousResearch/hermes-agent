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
    # No new cache entry: the breakpoints sit exactly where the last conversation request had them
    # (same rows, same TTL), and the appended reply + instruction carry none, so they are read, never written.
    assert _markers(summary_call["messages"][:len(last_turn["messages"])]) == _markers(last_turn["messages"])
    assert _markers(summary_call["messages"][len(last_turn["messages"]):]) == []


def _markers(messages):
    found = []
    for i, m in enumerate(messages):
        if "cache_control" in m:
            found.append((i, None, m["cache_control"]))
        for j, part in enumerate(m.get("content") if isinstance(m.get("content"), list) else ()):
            if isinstance(part, dict) and "cache_control" in part:
                found.append((i, j, part["cache_control"]))
    return found


def _read_rounds(n):
    rows = [{"role": "user", "content": "Please inspect the files"}]
    for i in range(n):  # the same call each round: identical openings, bodies the prune rewrites
        rows += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function",
                  "function": {"name": "read_file", "arguments": '{"path": "file.py"}'}}]},
                 {"role": "tool", "tool_call_id": f"call_{i}", "content": f"1|unique file {i}\n" + "many distinct lines\n" * 1000}]
    return rows + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "continue"},
                   {"role": "assistant", "content": "OK"}]


def test_warm_summary_names_its_window_by_rows_the_replayed_history_contains(monkeypatch):
    """Repeated tool reads, pruned in the compressor's copy: the instruction names the window's first and
    last rows by their position and call ids in the replayed (unpruned) history, unambiguously."""
    from agent.context_compressor import ContextCompressor
    from agent.context_compressor_summary import SummaryDispatchMixin

    compressor = ContextCompressor(model="claude-sonnet-4-6", quiet_mode=True, protect_first_n=1, protect_last_n=3)
    compressor.tail_mode, compressor._tail_token_budget = "legacy", 800
    history = [{"role": "system", "content": "sys"}] + _read_rounds(15)
    work, pruned, start, end = compressor._plan_compaction_window(history[1:])
    assert pruned and end - start >= 2
    sent = {}
    monkeypatch.setattr("agent.side_question_request.ask_on_live_prefix",
                        lambda agent, hist, prompt, **k: (sent.update(prompt=prompt) or None, {}))
    warm = SimpleNamespace(agent=SimpleNamespace(model=MODEL), history=history)
    SummaryDispatchMixin._warm_prefix_summary(compressor, warm, work[start:end], (2000, None, "", True))
    first, last = history[1:][start], history[1:][end - 1]
    for row in (first, last):
        call_id = row.get("tool_call_id") or row["tool_calls"][0]["id"]
        assert call_id in sent["prompt"]
    assert "read file.py from line 1" not in sent["prompt"]  # never a pruned stub's text
    assert "#" not in sent["prompt"].split("[The turns", 1)[1]  # prefill rows / merged results renumber the wire


def test_a_repeated_text_boundary_is_named_by_its_occurrence_or_not_at_all():
    from agent.context_compressor_summary import warm_window_pointer

    go, done = {"role": "user", "content": "go"}, {"role": "assistant", "content": "Done."}
    history = [go, done, {"role": "user", "content": "x"}, dict(go), dict(done), {"role": "user", "content": "y"}]
    pointer = warm_window_pointer(history, [dict(m) for m in history[3:6]])
    assert 'begins "go" (occurrence 2 of 2' in pointer and 'begins "y"' in pointer
    # A window that fits two places equally well is not named at all: the normal summary call runs.
    assert warm_window_pointer([go, done, dict(go), dict(done)], [dict(go), dict(done)]) is None


def test_warm_summary_is_skipped_when_the_summary_model_is_cheaper(monkeypatch):
    """A huge cached prefix on an expensive model vs a small window on a cheap summary model: the normal
    summary call wins, and no warm request is sent."""
    from agent.context_compressor_summary import SummaryDispatchMixin
    import agent.cache_expiry_compaction as cec

    calls = []
    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model",
                        lambda task: ("anthropic", "claude-haiku-4-5", None, None, None))
    live = SimpleNamespace(model="claude-sonnet-4-6", provider="anthropic", base_url="https://api.anthropic.com",
                           context_compressor=SimpleNamespace(last_prompt_tokens=900_000))
    compressor = SimpleNamespace(_compression_cancelled=lambda: False,
                                 _call_summary_llm=lambda messages, started: calls.append("aux") or SUMMARY,
                                 _warm_prefix_summary=lambda warm, turns, args: calls.append("warm") or SUMMARY)
    compressor._warm_prefix_is_cheaper = lambda warm, msgs: SummaryDispatchMixin._warm_prefix_is_cheaper(compressor, warm, msgs)
    messages = [{"role": "system", "content": "summarize"}, {"role": "user", "content": "turns " * 2000}]
    with cec.warm_prefix_summary(live, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]):
        out = SummaryDispatchMixin._summary_content(compressor, messages, 0.0, [{"role": "user", "content": "hi"}],
                                                    (2000, None, "", True))
    assert (out, calls) == (SUMMARY, ["aux"])


def test_an_explicitly_refused_warm_summary_never_replaces_history(monkeypatch):
    """A reply carrying both text and an explicit provider refusal is unusable on the warm path too."""
    from agent.context_compressor_summary import SummaryDispatchMixin
    from agent.transports.types import NormalizedResponse

    refused = NormalizedResponse(content=SUMMARY, tool_calls=None, finish_reason="stop",
                                 provider_data={"refusal": "I can't help with that."})
    monkeypatch.setattr("agent.side_question_request.ask_on_live_prefix", lambda *a, **k: (refused, {}))
    compressor = SimpleNamespace(_build_summary_prompt=lambda *a: [{"content": "i"}, {"content": "s"}],
                                 _compression_cancelled=lambda: False, tail_mode="legacy")
    warm = SimpleNamespace(agent=SimpleNamespace(model=MODEL), history=[{"role": "user", "content": "x"},
                                                                      {"role": "assistant", "content": "ok"}])
    with pytest.raises(RuntimeError, match="refusal"):
        SummaryDispatchMixin._warm_prefix_summary(compressor, warm, [{"role": "user", "content": "x"}],
                                                  (2000, None, "", True))
