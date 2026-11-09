"""A ``hold_stream`` plugin holds the agent's user-facing text until its rewrite lands.

Every surface (messaging, TUI/Desktop/CLI viewers, API server, ACP, managed workers) receives the
reply through the agent's display callbacks, so the agent is the one place the hold must apply: no
text delta and no interim commentary leaves it while a loaded ``transform_llm_output`` plugin in
the session's profile declared ``hold_stream=True``. The final reply is the rewrite. A plugin that
did not declare it keeps the normal stream (control).
"""

import json
import os
from pathlib import Path

import pytest

from tests.fakes.fake_llm_provider import FakeLLMServer, Raw, Text, ToolCall, _chunk
from tests.fakes.redactor_plugin import COMMENTARY, RAW, REWRITE, raw_leaks, redactor_home, redactor_turn


def _text_after_tool_call():
    """A tool-call response whose text arrives after the call began streaming: the agent routes
    that text through its suppressed-content path rather than the ordinary delta path."""
    call = {"index": 0, "id": "call_late_text", "type": "function",
            "function": {"name": "todo_list", "arguments": "{}"}}
    finish = {**_chunk({}, "tool_calls"), "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
    frames = [_chunk({"role": "assistant", "content": ""}), _chunk({"tool_calls": [call]}),
              _chunk({"content": COMMENTARY}), finish]
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    return [Raw(body=body, content_type="text/event-stream"), Text(RAW, chunk_chars=6)]


TURNS = {"commentary_then_tool": redactor_turn, "text_after_tool_call": _text_after_tool_call}


@pytest.mark.parametrize("turn", TURNS)
@pytest.mark.parametrize("hold", [True, False], ids=["hold_stream", "plain_transform"])
def test_held_turns_show_only_the_rewritten_final(hold, turn):
    from hermes_cli import plugins
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer(TURNS[turn]()) as srv:
        redactor_home(home, srv.base_url, hold=hold)
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = AIAgent(base_url=srv.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"])
            shown: list = []
            agent.stream_delta_callback = shown.append
            agent.interim_assistant_callback = lambda text, **_: shown.append(text)
            result = agent.run_conversation("show me the config")
        finally:
            plugins._reset_plugin_managers_for_tests()

    assert result["final_response"] == REWRITE
    assert len(srv.main_requests()) == 2  # the tool round ran, then the answer
    if hold:
        assert [text for text in shown if text] == []
    else:  # the plain transform rewrites the final only; the stream and commentary showed raw text
        assert COMMENTARY in shown and raw_leaks(shown)


def test_a_held_guardrail_halt_reply_is_not_streamed_ahead_of_the_rewrite():
    """A turn the tool-loop guardrail halts ends with Hermes' halt explanation, pushed straight to
    the display callback so SSE/TUI clients see why the agent stopped. Held, it arrives only as the
    final (after ``transform_llm_output``), never as a stream delta first."""
    from hermes_cli import plugins
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    failing = ToolCall("read_file", {"path": str(home / "missing" / "SECRET-9.txt")})
    with FakeLLMServer([failing] * 6 + [Text(RAW)]) as srv:
        redactor_home(home, srv.base_url, hold=True)
        with (home / "config.yaml").open("a", encoding="utf-8") as config:
            config.write("tool_loop_guardrails:\n  hard_stop_enabled: true\n  hard_stop_after:\n    exact_failure: 2\n")
        plugins._reset_plugin_managers_for_tests()
        try:
            agent = AIAgent(base_url=srv.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["file"])
            shown: list = []
            agent.stream_delta_callback = shown.append
            result = agent.run_conversation("read the secret file")
        finally:
            plugins._reset_plugin_managers_for_tests()

    assert result["turn_exit_reason"] == "guardrail_halt" and result["final_response"]
    assert [text for text in shown if text] == []
