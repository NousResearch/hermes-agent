"""Native Anthropic SDK streaming through Relay's managed execution path."""

import pytest


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_anthropic_sdk_stream_runs_through_relay_managed_execution(
    tmp_path,
    monkeypatch,
):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("nemo_relay")
    from agent import relay_runtime
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    response_body = b"""event: message_start
data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

"""

    def respond(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_body,
            request=request,
        )

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    relay_runtime._reset_for_tests()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        provider="anthropic",
        model="claude-test",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent.session_id = "anthropic-relay-session"
    agent._interrupt_requested = False
    agent._create_request_anthropic_client = lambda *args, **kwargs: client
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id=agent.session_id,
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="anthropic-relay-turn",
        task_id="anthropic-relay-task",
    )
    lease.host.retain_managed_execution("test.anthropic_relay")

    try:
        result = agent._interruptible_streaming_api_call(
            {
                "model": "claude-test",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    finally:
        lease.host.release_managed_execution("test.anthropic_relay")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()
        client.close()

    assert result.content[0].text == "hello"
    assert result.stop_reason == "end_turn"
