"""Prior-answer replay recovery (regression for #7619).

A long conversation on one topic, then a switch: the model does the new turn's tool work and
then answers with a verbatim copy of an earlier turn's answer. The loop accepted that copy as
the turn's final response, so the new question went unanswered while the transcript looked
healthy. No guard covered it — ``agent.repetition_guard`` only inspects repetition INSIDE one
fragment, and the final-text guards (stall intent, degenerate fragment, dropped tool call) look
at shape, not at what the model already said earlier in the same conversation.

The guard rides the existing ack-continuation path: same scope knob (``agent.stall_guards``),
same bounded per-turn counter, same durable interim + nudge rows. A verbatim copy of an earlier
answer is never a valid answer to a different question, so the nudge asks for the new one; a
user who genuinely asked for a repeat pays one extra call and then gets the text delivered.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _PRIOR_ANSWER_REPLAY_NUDGE
from agent.repetition_guard import replays_prior_answer


PRIOR_ANSWER = (
    "I searched your memory for 5523 and found three matching records. The first is a note from "
    "March about the 5523 relay board, the second is a purchase order, and the third is a test "
    "log that references the same part number twice."
)
NEW_ANSWER = (
    "Draft notice: the cleaning crew will remove willow catkins from the two WeChat groups' "
    "shared drives on Thursday morning, before the group members are notified."
)


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _final(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop")


def _tool_round(call_id):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id=call_id)],
    )


def _call(agent, stages, prompt, history):
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value="ok"),
    ):
        return agent.run_conversation(prompt, conversation_history=list(history))


def test_topic_switch_after_a_long_session_is_not_answered_with_a_prior_answer(loop_agent):
    """The reported shape: a long run of turns on one topic, then a switch."""
    history = []
    for i in range(12):
        answer = PRIOR_ANSWER if i == 0 else (
            f"Topic A turn {i}: relay board 5523 note {i} and its purchase order {i}."
        )
        result = _call(loop_agent, [_final(answer)], f"topic A question {i}", history)
        history.extend(result["messages"])
    assert PRIOR_ANSWER in [m.get("content") for m in history]
    # The topic switch does the new turn's tool work, then answers with the old topic's answer.
    result = _call(
        loop_agent,
        [_tool_round("call_1"), _final(PRIOR_ANSWER), _final(NEW_ANSWER)],
        "completely different topic B question", history,
    )

    assert result["final_response"] == NEW_ANSWER
    assert PRIOR_ANSWER not in (result["final_response"] or "")
    # Both halves of the replay re-prompt are ephemeral scaffolding, exactly like the dropped
    # tool-call pair: the finalization pop strips the hidden placeholder and the nudge, so
    # neither survives as a durable row and the copied text is nowhere in the turn's rows.
    rows = result["messages"]
    assert all(m.get("content") != _PRIOR_ANSWER_REPLAY_NUDGE for m in rows)
    assert not any(m.get("_prior_answer_replay_nudge") for m in rows)
    assert not any(m.get("display_kind") == "hidden" and not m.get("content") for m in rows)
    assert any(m.get("content") == NEW_ANSWER for m in rows)


def test_persistent_prior_answer_replay_is_bounded(loop_agent):
    """Two re-prompts per turn, then the turn ends on whatever the model said — never a loop."""
    seed = _call(loop_agent, [_final(PRIOR_ANSWER)], "search memory for 5523", [])
    before = loop_agent.client.chat.completions.create.call_count
    result = _call(
        loop_agent,
        [_final(PRIOR_ANSWER), _final(PRIOR_ANSWER), _final(PRIOR_ANSWER)],
        "draft a cleaning notice", seed["messages"],
    )

    assert result["final_response"] == PRIOR_ANSWER
    assert loop_agent.client.chat.completions.create.call_count - before == 3


class TestReplaysPriorAnswer:
    def test_a_verbatim_copy_of_an_earlier_answer_is_a_replay(self):
        history = [
            {"role": "user", "content": "search memory for 5523"},
            {"role": "assistant", "content": PRIOR_ANSWER},
            {"role": "user", "content": "draft a cleaning notice"},
        ]
        assert replays_prior_answer(PRIOR_ANSWER, history)
        # Whitespace-only drift from the wire is still the same answer.
        assert replays_prior_answer(PRIOR_ANSWER.replace(". The", ".\n\n  The"), history)
        # A model that re-emitted the answer with a lead-in is still replaying it.
        assert replays_prior_answer("Sure, here it is again:\n\n" + PRIOR_ANSWER, history)

    def test_a_new_answer_for_a_new_question_is_not_a_replay(self):
        history = [
            {"role": "user", "content": "search memory for 5523"},
            {"role": "assistant", "content": PRIOR_ANSWER},
            {"role": "user", "content": "draft a cleaning notice"},
        ]
        assert not replays_prior_answer(NEW_ANSWER, history)
        # Reusing one paragraph of an earlier answer is not replaying that answer.
        partial = "Here is the notice you asked for.\n\n" + NEW_ANSWER + "\n\n" + PRIOR_ANSWER[:80]
        assert not replays_prior_answer(partial, history)

    def test_short_answers_are_never_a_replay(self):
        history = [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "next"}]
        assert not replays_prior_answer("Done.", history)

    def test_narration_and_hidden_rows_are_not_compared(self):
        history = [
            {"role": "assistant", "content": PRIOR_ANSWER, "tool_calls": [{"id": "c1"}]},
            {"role": "assistant", "content": PRIOR_ANSWER, "display_kind": "hidden"},
            {"role": "user", "content": "next"},
        ]
        assert not replays_prior_answer(PRIOR_ANSWER, history)


def test_replay_scaffolding_is_registered_in_every_scaffolding_table():
    """The flag must be known to all three consumers: the pop, the persistence skip and the
    synthetic-user classifier. A table that misses it keeps the synthetic nudge as a real turn."""
    from agent.conversation_compression import _SYNTHETIC_USER_FLAGS, _is_real_user_message
    from agent.session_persistence import _EPHEMERAL_SCAFFOLDING_FLAGS, _is_ephemeral_scaffolding
    from agent.turn_final_response import _EPHEMERAL_SCAFFOLDING_FLAGS as _POPPED_FLAGS

    row = {"role": "user", "content": _PRIOR_ANSWER_REPLAY_NUDGE, "_prior_answer_replay_nudge": True}
    assert "_prior_answer_replay_nudge" in _POPPED_FLAGS
    assert "_prior_answer_replay_nudge" in _EPHEMERAL_SCAFFOLDING_FLAGS
    assert "_prior_answer_replay_nudge" in _SYNTHETIC_USER_FLAGS
    assert _is_ephemeral_scaffolding(row)
    assert not _is_real_user_message(row)
