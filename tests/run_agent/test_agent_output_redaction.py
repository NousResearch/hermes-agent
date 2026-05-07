from types import SimpleNamespace

import run_agent


def _agent():
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent.interim_assistant_callback = None
    agent._stream_callback = None
    agent._stream_needs_break = False
    agent._current_streamed_assistant_text = ""
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent.verbose_logging = False
    return agent


def test_stream_delta_force_redacts_secret_before_callbacks():
    agent = _agent()
    observed = []
    agent.stream_delta_callback = observed.append

    agent._fire_stream_delta("Use OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012 now")

    assert observed == ["Use OPENAI_API_KEY=*** now"]
    assert "abc123def456ghi789jk" not in observed[0]


def test_reasoning_delta_force_redacts_secret_before_callback():
    agent = _agent()
    observed = []
    agent.reasoning_callback = observed.append

    agent._fire_reasoning_delta("I saw Authorization: Bearer sk-proj-abc123def456ghi789jkl012")

    assert observed == ["I saw Authorization: Bearer ***"]
    assert "abc123def456ghi789jk" not in observed[0]


def test_interim_assistant_callback_force_redacts_secret():
    agent = _agent()
    observed = {}
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: observed.update(
        {"text": text, "already_streamed": already_streamed}
    )

    agent._emit_interim_assistant_message(
        {
            "role": "assistant",
            "content": "Found api_key='sk-proj-abc123def456ghi789jkl012' in config.",
        }
    )

    assert observed == {
        "text": "Found api_key='sk-pro...l012' in config.",
        "already_streamed": False,
    }
    assert "abc123def456ghi789jk" not in observed["text"]


def test_build_assistant_message_force_redacts_stored_content_and_reasoning():
    agent = _agent()
    observed_reasoning = []
    agent.reasoning_callback = observed_reasoning.append

    msg = SimpleNamespace(
        content="The key is OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012",
        tool_calls=None,
        reasoning_content="Need to avoid leaking sk-proj-reason123456789abcdefghijkl",
        reasoning=None,
        reasoning_details=None,
    )

    built = agent._build_assistant_message(msg, "stop")

    assert built["content"] == "The key is OPENAI_API_KEY=***"
    assert built["reasoning"] == "Need to avoid leaking sk-pro...ijkl"
    assert built["reasoning_content"] == "Need to avoid leaking sk-pro...ijkl"
    assert observed_reasoning == ["Need to avoid leaking sk-pro...ijkl"]
    assert "abc123def456ghi789jk" not in built["content"]
    assert "reason123456789abcde" not in built["reasoning"]
