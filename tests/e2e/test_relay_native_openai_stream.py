"""Native OpenAI SDK streaming through Relay's managed execution path."""

from __future__ import annotations

import threading

import pytest


def test_openai_stream_usage_reaches_relay_parent_event(tmp_path, monkeypatch):
    """A trailing usage-only chunk is retained on Relay's parent LLM event."""
    httpx = pytest.importorskip("httpx")
    nemo_relay = pytest.importorskip("nemo_relay")
    openai = pytest.importorskip("openai")

    from agent import chat_completion_helpers, relay_llm, relay_runtime
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    response_body = b"""data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"test/model","choices":[{"index":0,"delta":{"role":"assistant","content":"done"},"finish_reason":null}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"test/model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"test/model","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}

data: [DONE]

"""

    def respond(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_body,
            request=request,
        )

    client = openai.OpenAI(
        api_key="test-key",
        base_url="https://example.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    relay_runtime._reset_for_tests()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        provider="test-provider",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent.session_id = "openai-relay-session"
    agent._interrupt_requested = False
    agent._create_request_openai_client = lambda *args, **kwargs: client
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id=agent.session_id,
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="openai-relay-turn",
        task_id="openai-relay-task",
    )
    consumer = "test.openai_relay"
    subscriber_name = "test.openai_stream_usage"
    events = []
    relay_finalizer_started = threading.Event()
    allow_relay_finalizer = threading.Event()
    relay_finalizer_finished = threading.Event()
    run_relay_finalizer = relay_llm.ManagedLlmStream._relay_finalizer

    def run_synchronized_relay_finalizer(managed_stream, attempt):
        relay_finalizer_started.set()
        assert allow_relay_finalizer.wait(5), (
            "consumer did not release Relay's finalizer"
        )
        try:
            return run_relay_finalizer(managed_stream, attempt)
        finally:
            relay_finalizer_finished.set()

    monkeypatch.setattr(
        relay_llm.ManagedLlmStream,
        "_relay_finalizer",
        run_synchronized_relay_finalizer,
    )

    parse_choiceless_chunk = chat_completion_helpers._StreamingCall._choiceless_chunk

    def parse_usage_after_relay_finalizes(chunk, finish_reason):
        if not chunk.choices and getattr(chunk, "usage", None) is not None:
            # Force Relay to finalize before the consumer copies the usage frame.
            assert relay_finalizer_started.wait(5), "Relay's finalizer did not start"
            allow_relay_finalizer.set()
            assert relay_finalizer_finished.wait(5), "Relay's finalizer did not finish"
        return parse_choiceless_chunk(chunk, finish_reason)

    monkeypatch.setattr(
        chat_completion_helpers._StreamingCall,
        "_choiceless_chunk",
        staticmethod(parse_usage_after_relay_finalizes),
    )
    lease.host.retain_managed_execution(consumer)
    lease.host.relay.subscribers.register(subscriber_name, events.append)

    try:
        result = agent._interruptible_streaming_api_call({
            "model": "test/model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        lease.host.relay.subscribers.flush()
    finally:
        lease.host.relay.subscribers.deregister(subscriber_name)
        lease.host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()
        client.close()

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 110
    llm_end_events = [
        event
        for event in events
        if isinstance(event, nemo_relay.ScopeEvent)
        and event.name == "openai.chat_completions"
        and event.category == "llm"
        and event.scope_category == "end"
    ]
    assert len(llm_end_events) == 1
    llm_end = llm_end_events[0]
    assert llm_end.annotated_response is not None
    assert llm_end.annotated_response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
    }
