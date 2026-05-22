import asyncio
from types import SimpleNamespace

import pytest


class _FakeHooks:
    loaded_hooks = False

    async def emit(self, *_args, **_kwargs):
        return None


class _FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = kwargs.get("session_id")
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        self.is_interrupted = False
        self.interim_assistant_callback = None

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.interim_assistant_callback:
            self.interim_assistant_callback("I am checking the gateway seam.")
        return {
            "final_response": "I finished the gateway voice regression and verified the seam.",
            "messages": [
                {"role": "assistant", "content": "I am checking the gateway seam."},
                {
                    "role": "assistant",
                    "content": "I finished the gateway voice regression and verified the seam.",
                },
            ],
            "api_calls": 1,
            "completed": True,
        }

    def get_activity_summary(self):
        return {"seconds_since_activity": 0.0, "api_call_count": 1, "max_iterations": 90}

    def interrupt(self, _message=None):
        self.is_interrupted = True


@pytest.mark.asyncio
async def test_gateway_success_turn_publishes_only_final_voice_event(monkeypatch):
    from gateway.config import GatewayConfig, Platform
    import gateway.pulse_voice_events as pulse_voice_events
    import gateway.run as gateway_run
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    import run_agent

    voice_event_calls = []
    voice_out_calls = []
    completion_calls = []

    def record_voice_event(kind, text, **metadata):
        voice_event_calls.append((kind, text, metadata))

    def record_voice_out(kind, text, **metadata):
        voice_out_calls.append((kind, text, metadata))

    def record_completion(final_response, **metadata):
        completion_calls.append((final_response, metadata))

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)
    monkeypatch.setattr(pulse_voice_events, "publish_voice_event", record_voice_event)
    monkeypatch.setattr(pulse_voice_events, "publish_voice_out", record_voice_out)
    monkeypatch.setattr(pulse_voice_events, "publish_completion_voice_out", record_completion)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "display": {
                "tool_progress": "off",
                "interim_assistant_messages": True,
                "streaming": False,
            },
            "agent": {"disabled_toolsets": []},
        },
    )
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0")

    runner = GatewayRunner.__new__(GatewayRunner)
    object.__setattr__(runner, "config", GatewayConfig())
    object.__setattr__(runner, "adapters", {})
    object.__setattr__(runner, "hooks", _FakeHooks())
    object.__setattr__(runner, "_ephemeral_system_prompt", "")
    object.__setattr__(runner, "_provider_routing", {})
    object.__setattr__(runner, "_prefill_messages", [])
    object.__setattr__(runner, "_fallback_model", None)
    object.__setattr__(runner, "_session_db", None)
    object.__setattr__(runner, "session_store", SimpleNamespace(_entries={}, _save=lambda: None))
    object.__setattr__(runner, "_running_agents", {})
    object.__setattr__(runner, "_draining", False)
    object.__setattr__(runner, "_get_proxy_url", lambda: None)
    object.__setattr__(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_kwargs: ("fake-model", {"provider": "fake-provider"}),
    )
    object.__setattr__(runner, "_resolve_session_reasoning_config", lambda **_kwargs: None)
    object.__setattr__(runner, "_load_service_tier", lambda: None)
    object.__setattr__(
        runner,
        "_resolve_turn_agent_config",
        lambda user_message, model, runtime_kwargs: {
            "model": model,
            "runtime": runtime_kwargs,
            "request_overrides": None,
        },
    )
    object.__setattr__(runner, "_agent_config_signature", lambda *_args, **_kwargs: "sig")
    object.__setattr__(runner, "_extract_cache_busting_config", lambda user_config: {})
    object.__setattr__(runner, "_enforce_agent_cache_cap", lambda: None)
    object.__setattr__(runner, "_consume_pending_native_image_paths", lambda session_key: [])
    object.__setattr__(
        runner,
        "_thread_metadata_for_source",
        lambda source, reply_to_message_id=None: None,
    )
    object.__setattr__(runner, "_is_session_run_current", lambda session_key, generation: True)
    object.__setattr__(
        runner,
        "_run_in_executor_with_context",
        lambda func, *args: asyncio.to_thread(func, *args),
    )

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-123",
        thread_id="thread-456",
        user_id="user-789",
        user_name="Brenno",
        chat_type="thread",
    )

    result = await runner._run_agent(
        "please do the work",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-abc",
        session_key="discord:chat-123:thread-456",
        run_generation=1,
        event_message_id="message-source-001",
    )

    assert result["final_response"] == "I finished the gateway voice regression and verified the seam."
    assert ("commentary", "I am checking the gateway seam.") in [
        (kind, text) for kind, text, _metadata in voice_event_calls
    ]
    assert all(kind != "ack" for kind, _text, _metadata in voice_event_calls)
    assert all(kind != "ack" for kind, _text, _metadata in voice_out_calls)
    assert completion_calls == [
        (
            "I finished the gateway voice regression and verified the seam.",
            {
                "session_id": "session-abc",
                "platform": "discord",
                "chat_id": "chat-123",
                "thread_id": "thread-456",
                "source_message_id": "message-source-001",
            },
        )
    ]
