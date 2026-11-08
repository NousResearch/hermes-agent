"""Provider redelivery must not duplicate reasoning (#59009)."""
import json
import threading
from types import SimpleNamespace

import pytest

from agent.reasoning_summaries import extract_reasoning


@pytest.fixture
def agent(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    instance = AIAgent(api_key="test-key", base_url="https://example.test/v1",
        provider="custom", model="test/model", api_mode="chat_completions",
        enabled_toolsets=[], quiet_mode=True, skip_memory=True,
        skip_context_files=True, skip_background_review=True)
    yield instance
    instance.client.close()


def _stream(agent, monkeypatch, deltas):
    """Real SDK and production request/collector path; only HTTP is replaced."""
    import httpx
    from openai import OpenAI

    chunks = [{"id": "local-fixture", "created": 1, "model": "test/model",
        "object": "chat.completion.chunk", "choices": [{"index": 0,
        "delta": delta, "finish_reason": None}]} for delta in deltas]
    chunks.append({"id": "local-fixture", "created": 1, "model": "test/model",
        "object": "chat.completion.chunk", "choices": [{"index": 0,
        "delta": {}, "finish_reason": "stop"}]})
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
            content=(body + "data: [DONE]\n\n").encode(), request=request)

    with OpenAI(api_key="test-key", base_url="https://example.test/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
        result = agent._interruptible_streaming_api_call({"model": "test/model",
            "messages": [{"role": "user", "content": "fixture"}]})
    assert requests and all(request["stream"] for request in requests)
    return result.choices[0].message


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_detail_redelivery_is_suppressed_after_typed_text_flattening(field):
    first, second, new = "Alpha", "Beta", "Gamma"
    message = SimpleNamespace(**{field: [{"type": "text", "text": first},
                                        {"type": "text", "text": second}]},
        reasoning_details=[{"thinking": [{"text": first}]}, {"text": second},
                           {"summary": {"text": new}}], content="Answer")
    assert extract_reasoning(None, message) == first + second + "\n\n" + new
    # Only details are redelivery fragments: independent top-level channels keep
    # the target's exact-dedup contract rather than losing legitimate short text.
    assert extract_reasoning(None, SimpleNamespace(reasoning=first + second,
        reasoning_content=first)) == first + second + "\n\n" + first


_PREFIX = "abcdefghijklmnopqrstuvwxyz0123456789"
_FIRST = "**Inspecting the first summary part**"
_SECOND = "**Checking the second summary part**"


@pytest.mark.parametrize("surface", ["main", "relay"])
@pytest.mark.parametrize("deltas,expected", [
    ([_PREFIX, _PREFIX + " NEW", _PREFIX + " NEW"], _PREFIX + " NEW"),
    (["intro " + _PREFIX, _PREFIX + " NEW"], "intro " + _PREFIX + " NEW"),
    (["the", "the", " so ", " so "], "thethe so  so "),
    ([_PREFIX[:23], _PREFIX[:23]], _PREFIX[:23] * 2),
    ([_PREFIX[:24], _PREFIX[:24]], _PREFIX[:24]),
    ([_PREFIX + " middle ", _PREFIX], _PREFIX + " middle " + _PREFIX),
    ([_FIRST, _FIRST + _SECOND, _FIRST + _SECOND], _FIRST + "\n\n" + _SECOND),
    ([_FIRST, _SECOND, _FIRST + _SECOND + " tail"], _FIRST + "\n\n" + _SECOND + " tail"),
])
def test_stream_redelivery_keeps_new_text_and_summary_boundaries(
        agent, monkeypatch, surface, deltas, expected):
    from agent.chat_completion_helpers_relay import RelayChatAccumulator

    # Exercise list/dict flattening before dedup, at BOTH independent collectors.
    chunks = [{"reasoning_content": [{"type": "text", "text": delta}]} for delta in deltas]
    chunks.append({"content": [{"text": "Answer"}]})
    if surface == "main":
        callbacks = []
        agent.reasoning_callback = callbacks.append
        message = _stream(agent, monkeypatch, chunks)
        actual = extract_reasoning(agent, message)
        assert "".join(callbacks) == expected
        assert all(callbacks), "duplicate/empty reasoning must not fire callbacks"
        assert message.content == "Answer"
    else:
        accumulator = RelayChatAccumulator()
        for delta in chunks:
            accumulator.observe({"choices": [{"delta": delta}]})
        message = accumulator.finalize()["choices"][0]["message"]
        actual = message["reasoning_content"]
        assert message["content"] == "Answer"
    assert actual == expected


@pytest.mark.parametrize("callback_raises", [False, True])
def test_materialization_never_refires_streamed_reasoning(agent, monkeypatch, callback_raises):
    from agent.chat_completion_helpers import build_assistant_message

    delivered = []

    def receive(text):
        delivered.append(text)
        if callback_raises:
            raise RuntimeError("display failed after accepting text")

    agent.reasoning_callback = receive
    message = _stream(agent, monkeypatch, [{"reasoning": _PREFIX},
        {"reasoning": " tail"}, {"content": "Answer"}])
    assert delivered == [_PREFIX, " tail"]
    for _ in range(2):
        stored = build_assistant_message(agent, message, "stop")
        assert stored["reasoning"] == _PREFIX + " tail"
    assert delivered == [_PREFIX, " tail"]


@pytest.mark.parametrize("route", ["nonstream-worker", "nonstream-inline", "stream-chat",
    "stream-codex", "stream-bedrock", "stream-interrupted"])
def test_api_entry_resets_aborted_response_latch_before_dispatch(agent, monkeypatch, route):
    from agent import chat_completion_helpers as h
    from agent.chat_completion_nonstream import _NonStreamRequest

    delivered = []
    agent.reasoning_callback = delivered.append
    agent._reasoning_streamed_this_response = True  # prior response never materialized
    message = SimpleNamespace(reasoning="Next response", content="Answer")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    dispatched = []

    def no_reasoning_deltas(*args, **kwargs):
        assert agent._reasoning_streamed_this_response is False
        dispatched.append(route)
        return response

    def interrupt_before_dispatch():
        agent._interrupt_requested = True
        with pytest.raises(InterruptedError):
            h.interruptible_streaming_api_call(agent, {})
        assert agent._reasoning_streamed_this_response is False

    def configure_nonstream_inline():
        agent.platform = "cron"
        monkeypatch.setattr(h, "direct_api_call", no_reasoning_deltas)

    def configure_nonstream_worker():
        monkeypatch.setattr(_NonStreamRequest, "run", no_reasoning_deltas)

    def configure_stream_codex():
        agent.api_mode = "codex_responses"
        monkeypatch.setattr(agent, "_interruptible_api_call", no_reasoning_deltas)

    def configure_stream_bedrock():
        agent.api_mode = "bedrock_converse"
        monkeypatch.setattr(h._BedrockStream, "run", no_reasoning_deltas)

    def configure_stream_chat():
        monkeypatch.setattr(h._StreamingCall, "run", no_reasoning_deltas)

    setup = {
        "stream-interrupted": interrupt_before_dispatch,
        "nonstream-inline": configure_nonstream_inline,
        "nonstream-worker": configure_nonstream_worker,
        "stream-codex": configure_stream_codex,
        "stream-bedrock": configure_stream_bedrock,
        "stream-chat": configure_stream_chat,
    }[route]
    setup()
    if route == "stream-interrupted":
        return
    call = h.interruptible_api_call if route.startswith("nonstream") else h.interruptible_streaming_api_call
    assert call(agent, {}) is response
    assert dispatched == [route]
    # Completed-only reasoning still fires once, including repeated materialization.
    for _ in range(2):
        assert h.build_assistant_message(agent, message, "stop")["reasoning"] == message.reasoning
    assert delivered == [message.reasoning]


@pytest.mark.parametrize("platform", ["cli", "cron"])
def test_cached_agent_stream_to_completed_response_delivers_each_once(agent, monkeypatch, platform):
    import httpx
    from openai import OpenAI
    from agent.chat_completion_helpers import build_assistant_message

    delivered = []
    agent.reasoning_callback = delivered.append
    agent.platform = platform
    streamed = _stream(agent, monkeypatch, [{"reasoning": "Streamed thought"}, {"content": "Answer"}])
    assert build_assistant_message(agent, streamed, "stop")["reasoning"] == "Streamed thought"
    body = {"id": "local-completion", "created": 1, "model": "test/model", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer",
        "reasoning": "Completed thought"}, "finish_reason": "stop"}]}
    with OpenAI(api_key="test-key", base_url="https://example.test/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body, request=request)))) as client:
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
        response = agent._interruptible_api_call({"model": "test/model",
            "messages": [{"role": "user", "content": "fixture"}]})
    for _ in range(2):
        stored = build_assistant_message(agent, response.choices[0].message, "stop")
        assert stored["reasoning"] == "Completed thought"
    assert delivered == ["Streamed thought", "Completed thought"]


def test_plugin_only_and_superseded_writers_do_not_poison_callback_latch(agent, monkeypatch):
    from agent import plugin_stream_hooks
    from agent.chat_completion_helpers import build_assistant_message

    hooks, delivered = [], []
    monkeypatch.setattr(plugin_stream_hooks, "stream_reasoning_deltas_enabled", lambda: True)
    monkeypatch.setattr(plugin_stream_hooks, "enqueue_plugin_stream_hook",
        lambda event, **payload: hooks.append((event, payload)))
    agent.reasoning_callback = None
    agent._fire_reasoning_delta("plugin only")
    assert not getattr(agent, "_reasoning_streamed_this_response", False)
    assert [(event, payload["kind"], payload["delta"]) for event, payload in hooks] == [
        ("on_stream_delta", "reasoning", "plugin only")]
    agent.reasoning_callback = delivered.append
    agent._claim_stream_writer()
    # Supersede the current thread with a real claim, not a mocked guard.
    thread = threading.Thread(target=agent._claim_stream_writer)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    agent._fire_reasoning_delta("stale")
    assert delivered == [] and len(hooks) == 1
    assert not getattr(agent, "_reasoning_streamed_this_response", False)
    agent._claim_stream_writer()
    message = SimpleNamespace(reasoning="completed fallback", content="Answer")
    assert build_assistant_message(agent, message, "stop")["reasoning"] == message.reasoning
    assert delivered == [message.reasoning]
    agent._fire_reasoning_delta("current delta")
    assert delivered[-1] == "current delta"
    assert hooks[-1][1]["delta"] == "current delta"


@pytest.mark.parametrize("text_callback", ["stream_delta_callback", "_stream_callback"])
def test_text_stream_think_extraction_guard_still_suppresses_full_replay(agent, text_callback):
    from agent.chat_completion_helpers import build_assistant_message

    delivered = []
    agent.reasoning_callback = delivered.append
    setattr(agent, text_callback, lambda text: None)
    message = SimpleNamespace(content="<think>Inline thought</think>Answer")
    assert build_assistant_message(agent, message, "stop")["reasoning"] == "Inline thought"
    assert delivered == []


def test_display_dedup_preserves_structured_replay_fragments(agent, monkeypatch):
    """Display dedup must leave the structured replay fragments (0ddb62bce6) untouched."""
    from agent.chat_completion_helpers import build_assistant_message

    delivered = []
    agent.reasoning_callback = delivered.append
    encrypted = {"type": "reasoning.encrypted", "data": "opaque"}
    message = _stream(agent, monkeypatch, [
        {"reasoning": _PREFIX, "reasoning_details": [
            {"type": "reasoning.text", "text": _PREFIX[:12]}]},
        {"reasoning": _PREFIX + " tail", "reasoning_details": [
            {"type": "reasoning.text", "text": _PREFIX[12:] + " tail",
             "signature": "sig-final", "id": "block-id", "format": "fixture", "index": 0}]},
        {"reasoning": _PREFIX + " tail", "reasoning_details": [encrypted, encrypted]},
        {"content": "Answer"},
    ])
    expected_details = [
        {"type": "reasoning.text", "text": _PREFIX + " tail", "signature": "sig-final",
         "id": "block-id", "format": "fixture", "index": 0},
        encrypted, encrypted,
    ]
    assert message.reasoning_details == expected_details
    assert delivered == [_PREFIX, " tail"]
    for _ in range(2):
        stored = build_assistant_message(agent, message, "stop")
        assert stored["reasoning_details"] == expected_details
        assert stored["reasoning"] == _PREFIX + " tail"
    assert delivered == [_PREFIX, " tail"]
