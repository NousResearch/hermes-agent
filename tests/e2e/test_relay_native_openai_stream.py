"""Native OpenAI SDK streaming through Relay's managed execution path.

Relay runs its finalizer as soon as the provider stream ends — concurrently with Hermes'
consumer thread, which may not have processed the last chunk yet. Each test forces that
ordering deterministically (finalizer runs BEFORE the consumer sees a chosen chunk) and
asserts Relay's LLM end event still records the full response.
"""

from __future__ import annotations

import threading

import pytest

_CHUNK_PREFIX = b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"test/model",'


def _sse(*chunk_bodies: bytes) -> bytes:
    return b"".join(_CHUNK_PREFIX + body + b"}\n\n" for body in chunk_bodies) + b"data: [DONE]\n\n"


def _stream_through_relay(tmp_path, monkeypatch, response_body: bytes, *, finalize_before):
    """Stream ``response_body`` through Relay; Relay's finalizer is forced to complete before
    the consumer thread processes the first chunk matching ``finalize_before(chunk)``.
    Returns ``(hermes_result, relay_llm_end_event)``."""
    httpx = pytest.importorskip("httpx")
    nemo_relay = pytest.importorskip("nemo_relay")
    openai = pytest.importorskip("openai")

    from agent import chat_completion_helpers, relay_llm, relay_runtime
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    def respond(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=response_body, request=request)

    client = openai.OpenAI(api_key="test-key", base_url="https://example.com/v1",
                           http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    relay_runtime._reset_for_tests()
    agent = AIAgent(api_key="test-key", base_url="https://example.com/v1", provider="test-provider",
                    model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent.session_id = "openai-relay-session"
    agent._interrupt_requested = False
    agent._create_request_openai_client = lambda *args, **kwargs: client
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id=agent.session_id, platform="cli")
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease, turn_id="openai-relay-turn", task_id="openai-relay-task")
    consumer = "test.openai_relay"
    subscriber_name = "test.openai_stream"
    events = []
    relay_finalizer_started = threading.Event()
    allow_relay_finalizer = threading.Event()
    relay_finalizer_finished = threading.Event()
    run_relay_finalizer = relay_llm.ManagedLlmStream._relay_finalizer

    def run_synchronized_relay_finalizer(managed_stream, attempt):
        relay_finalizer_started.set()
        assert allow_relay_finalizer.wait(30), "consumer did not release Relay's finalizer"
        try:
            return run_relay_finalizer(managed_stream, attempt)
        finally:
            relay_finalizer_finished.set()

    monkeypatch.setattr(relay_llm.ManagedLlmStream, "_relay_finalizer", run_synchronized_relay_finalizer)

    count_chunk = chat_completion_helpers._StreamingCall._count_chunk

    def count_chunk_after_relay_finalizes(self, diag, chunk):
        # Relay's producer is pumped by the consumer thread's OWN event loop (``ManagedLlmStream.
        # __next__`` -> ``run_until_complete``), so the finalizer can only start while that loop runs.
        # Blocking the consumer thread here and waiting for it therefore deadlocked whenever the loop
        # had not reached EOF yet (~1 run in 6). Instead: release the finalizer and PUMP THE STREAM'S
        # LOOP until it has completed, then hand the chunk to the consumer. That is a real
        # happens-before (finalizer done -> consumer sees chunk) on every schedule, not a retry lottery.
        if finalize_before(chunk):
            import asyncio

            stream = self.managed_stream_holder["stream"]
            allow_relay_finalizer.set()
            # A schedule probe may hold the provider generator at this chunk until the consumer has
            # taken it (adverse consumer-first ordering); release it so the pump below can reach EOF.
            gate = getattr(stream, "_review_gate", None)
            if gate is not None:
                gate.set()

            async def finalizer_done():
                return await asyncio.to_thread(relay_finalizer_finished.wait, 30)

            assert stream._loop.run_until_complete(finalizer_done()), "Relay's finalizer did not finish"
            assert relay_finalizer_started.is_set()
        return count_chunk(self, diag, chunk)

    monkeypatch.setattr(chat_completion_helpers._StreamingCall, "_count_chunk", count_chunk_after_relay_finalizes)
    lease.host.retain_managed_execution(consumer)
    lease.host.relay.subscribers.register(subscriber_name, events.append)
    try:
        result = agent._interruptible_streaming_api_call({
            "model": "test/model", "messages": [{"role": "user", "content": "hi"}]})
        lease.host.relay.subscribers.flush()
    finally:
        lease.host.relay.subscribers.deregister(subscriber_name)
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()
        client.close()

    llm_end_events = [
        event for event in events
        if isinstance(event, nemo_relay.ScopeEvent) and event.name == "openai.chat_completions"
        and event.category == "llm" and event.scope_category == "end"
    ]
    assert len(llm_end_events) == 1
    assert llm_end_events[0].annotated_response is not None
    return result, llm_end_events[0]


def test_openai_stream_usage_reaches_relay_parent_event(tmp_path, monkeypatch):
    """A trailing usage-only chunk is retained on Relay's parent LLM event."""
    body = _sse(
        b'"choices":[{"index":0,"delta":{"role":"assistant","content":"done"},"finish_reason":null}]',
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]',
        b'"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}',
    )
    result, llm_end = _stream_through_relay(
        tmp_path, monkeypatch, body,
        finalize_before=lambda chunk: not chunk.choices and getattr(chunk, "usage", None) is not None)
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (100, 10, 110)
    assert llm_end.annotated_response.usage == {
        "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    assert llm_end.annotated_response.message == "done"


def test_openai_stream_final_tool_call_delta_reaches_relay_parent_event(tmp_path, monkeypatch):
    """The last chunk's tool-call arguments and finish_reason are retained on Relay's parent
    LLM event — the same finalizer-before-consumer race as the usage frame, without one."""
    body = _sse(
        b'"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,'
        b'"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\": "}}]},'
        b'"finish_reason":null}]',
        b'"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"/tmp/x\\"}"}}]},'
        b'"finish_reason":"tool_calls"}]',
    )
    result, llm_end = _stream_through_relay(
        tmp_path, monkeypatch, body,
        finalize_before=lambda chunk: bool(chunk.choices) and chunk.choices[0].finish_reason == "tool_calls")
    hermes_call = result.choices[0].message.tool_calls[0]
    assert (hermes_call.function.name, hermes_call.function.arguments) == ("read_file", '{"path": "/tmp/x"}')
    assert result.choices[0].finish_reason == "tool_calls"
    assert llm_end.annotated_response.message is None
    (relay_call,) = llm_end.annotated_response.tool_calls
    assert (relay_call["name"], relay_call["arguments"]) == ("read_file", {"path": "/tmp/x"})
    assert llm_end.annotated_response.finish_reason == "tool_use"
