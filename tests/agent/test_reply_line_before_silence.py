"""A bare silence marker after the model already wrote its reply ends the turn on that reply.

When the follow-up after a housekeeping round is EMPTY the loop already ends the turn on the
line written beside that tool (``fallback_prior_turn_content``). A follow-up that is only a
silence marker says the same thing ("nothing to add"). On a turn whose host says a marker may not
stay silent (``_silence_marker_needs_reply``, set by the gateway from ``silence_allowed``), the
loop takes the same path, so the reply runs through the normal output transform, persistence and
receipt. Without that flag (cron, CLI, machinery or unaddressed turns) the marker is kept.
"""

from unittest.mock import patch

from agent.persisted_turn_receipt import persisted_turn_receipt
from tests.agent import test_empty_response_recovery_persistence as _recovery
from tests.agent import test_hold_stream_exits as _held_exits

real_loop = _recovery.real_loop
held = _held_exits.held
_call, _response = _recovery._call, _recovery._response
ANSWER = "Your meeting is at 3pm."


def _register_todo(agent):
    agent.valid_tool_names.add("todo_list")
    agent.tools.append({"type": "function", "function": {"name": "todo_list", "parameters": {}}})


def _reply_then_marker(answer=ANSWER):
    return [
        _response(answer, finish_reason="tool_calls", tool_calls=[_call("todo_list", {"todos": []}, "call_todo")]),
        _response("NO_REPLY"),
    ]


def _stored_final(real_loop):
    rows = real_loop.db.get_messages_as_conversation(real_loop.sid)
    return [m for m in rows if m.get("role") == "assistant" and not m.get("tool_calls")][-1]["content"]


def _receipt(agent, result):
    return persisted_turn_receipt(
        result["messages"], agent._persist_user_message_idx, [], result["final_response"], "complete",
        compression_unchanged=True, prefix_row_matches=lambda before, after: before is after)


def test_marker_after_a_written_reply_ends_the_turn_on_that_reply(real_loop):
    agent = real_loop.agent
    _register_todo(agent)
    agent._silence_marker_needs_reply = True

    result = real_loop.run(_reply_then_marker(), "when is my meeting?")

    assert result["final_response"] == ANSWER
    assert result["completed"] is True
    # Same identity as the empty-follow-up fallback: a client settles the text it already shows.
    assert result["response_reused"] is True
    assert _stored_final(real_loop) == ANSWER
    assert "final_assistant_row_id" in _receipt(agent, result)


def test_a_crash_left_turn_recovers_the_reply_not_the_warning(real_loop):
    # Crash recovery judges the persisted final row as live delivery would have.
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    _register_todo(real_loop.agent)
    real_loop.agent._silence_marker_needs_reply = True
    real_loop.run(_reply_then_marker(), "when is my meeting?")
    history = real_loop.db.get_messages_as_conversation(real_loop.sid)
    origin = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm", user_id="u1")

    assert GatewayRunner._crash_left_reply(object(), history, 0, origin) == ANSWER


def test_only_a_complete_uninterrupted_marker_turn_takes_the_reply(real_loop):
    # The exact gate finish_text_response applies before transform and persistence.
    from agent.turn_final_response import _bare_marker_after_reply

    agent = real_loop.agent
    agent._last_content_with_tools, agent._last_content_tools_all_housekeeping = ANSWER, True
    agent._silence_marker_needs_reply = True

    assert _bare_marker_after_reply(agent, "NO_REPLY") is True
    assert _bare_marker_after_reply(agent, "Something else.") is False
    assert _bare_marker_after_reply(agent, "NO_REPLY", truncated_parts=[("NO", True)]) is False
    assert _bare_marker_after_reply(agent, "NO_REPLY", promoted="NO_REPLY") is False
    agent._interrupt_requested = True
    assert _bare_marker_after_reply(agent, "NO_REPLY") is False
    agent._interrupt_requested = False
    agent._silence_marker_needs_reply = False
    assert _bare_marker_after_reply(agent, "NO_REPLY") is False


def test_the_chosen_reply_runs_through_the_output_transform(real_loop):
    agent = real_loop.agent
    _register_todo(agent)
    agent._silence_marker_needs_reply = True
    hook_inputs = []

    def hook(name, **kwargs):
        if name == "transform_llm_output":
            hook_inputs.append(kwargs["response_text"])
            return ["Answer [REDACTED]." if "SECRET_TOKEN" in kwargs["response_text"] else kwargs["response_text"]]
        return []

    with patch("hermes_cli.lifecycle.invoke_hook", hook):
        result = real_loop.run(_reply_then_marker("Answer SECRET_TOKEN"), "token?")

    assert hook_inputs == ["Answer SECRET_TOKEN"]
    assert result["final_response"] == "Answer [REDACTED]."
    assert result["response_transformed"] is True
    assert _stored_final(real_loop) == "Answer [REDACTED]."
    assert "final_assistant_row_id" in _receipt(agent, result)


def test_without_the_host_flag_the_marker_is_kept(real_loop):
    # Cron, CLI, machinery and unaddressed turns: the marker is the deliberate result.
    _register_todo(real_loop.agent)

    result = real_loop.run(_reply_then_marker(), "when is my meeting?")

    assert result["final_response"] == "NO_REPLY"
    assert _stored_final(real_loop) == "NO_REPLY"


def test_narration_beside_a_substantive_tool_is_not_a_reply(real_loop):
    real_loop.agent._silence_marker_needs_reply = True

    result = real_loop.run(
        [
            _response("Let me check the file.", finish_reason="tool_calls",
                      tool_calls=[_call("read_file", {"path": "missing.txt"}, "call_read")]),
            _response("NO_REPLY"),
        ],
        "is the file there?",
    )

    assert result["final_response"] == "NO_REPLY"


def test_a_later_substantive_round_or_a_new_turn_drops_the_earlier_line(real_loop):
    agent = real_loop.agent
    _register_todo(agent)
    agent._silence_marker_needs_reply = True

    result = real_loop.run(
        [
            _response("Answer A", finish_reason="tool_calls", tool_calls=[_call("todo_list", {"todos": []}, "t1")]),
            _response("", finish_reason="tool_calls", tool_calls=[_call("read_file", {"path": "missing"}, "r1")]),
            _response("NO_REPLY"),
        ],
        "question",
    )
    second = real_loop.run([_response("NO_REPLY")], "second question")

    assert result["final_response"] == "NO_REPLY"
    assert second["final_response"] == "NO_REPLY"


def test_held_turn_rewrites_the_recovered_reply_as_a_new_final(held, monkeypatch):
    # A hold_stream transform withheld the reply line, so it was never shown: the recovered reply
    # goes out as the rewrite and must not be flagged as previewed/reused (same rule as the
    # empty-follow-up path on main).
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import Text, ToolCall
    from tests.fakes.redactor_plugin import RAW, REWRITE

    monkeypatch.setattr(AIAgent, "_silence_marker_needs_reply", True, raising=False)
    result, shown, displayed, _, _ = held([ToolCall("todo_list", {}, text=RAW), Text("NO_REPLY")])

    assert result["final_response"] == REWRITE and shown == []
    assert not result["response_previewed"] and not result["response_reused"]
    assert "SECRET-" not in str(displayed)
    assert REWRITE in [row["text"] for row in displayed if row["role"] == "assistant"]
