"""Interim assistant delivery hides the /loop completion marker on every surface.

All interim text (commentary beside tool calls, Codex completed agentMessage items) funnels
through ``_deliver_interim``; the gateway, TUI and API callbacks hang off it.
"""


def _agent(sink):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.interim_assistant_callback = lambda text, *, already_streamed=False: sink.append(text)
    return agent


def test_interim_message_strips_trailing_loop_complete():
    sent = []
    agent = _agent(sent)
    agent._emit_interim_assistant_message({"content": "CI is green.\nLOOP_COMPLETE"})
    assert sent == ["CI is green."]


def test_interim_bare_marker_is_not_sent_and_not_redelivered():
    sent = []
    agent = _agent(sent)
    agent._emit_interim_assistant_message({"content": "LOOP_COMPLETE"})
    agent._emit_interim_assistant_message({"content": "LOOP_COMPLETE"})
    assert sent == []


def test_interim_marker_inside_fence_is_content():
    sent = []
    agent = _agent(sent)
    text = "Example:\n```text\nLOOP_COMPLETE\n```"
    agent._emit_interim_assistant_message({"content": text})
    assert sent == [text]


def test_streamed_codex_commentary_strips_trailing_loop_complete():
    sent = []
    agent = _agent(sent)
    agent._fire_streamed_codex_commentary("Checked the queue.\nLOOP_COMPLETE")
    assert sent == ["Checked the queue."]
