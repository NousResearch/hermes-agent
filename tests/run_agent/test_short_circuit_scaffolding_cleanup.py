"""Regression test for FIX A — short-circuit early-return scaffolding cleanup.

Why: The ``pre_llm_call`` short-circuit early-return in
``agent.conversation_loop.run_conversation`` appends a terminal assistant
message and returns *without* going through ``finalize_turn`` /
``_persist_session``. When the incoming history still carries leftover
synthetic recovery scaffolding (``_empty_recovery_synthetic`` /
``_empty_terminal_sentinel``) from a prior empty-response nudge cycle, that
scaffolding survives into the returned/persisted history. On every following
turn ``repair_message_sequence`` then logs "Repaired N message-alternation
violations" because the stale synthetic ``user(nudge)`` collides with the next
real user turn.

What: Drives the short-circuit branch with a history whose tail is leftover
synthetic scaffolding and asserts the returned ``messages`` carries NO
synthetic-flagged messages and contains no consecutive-user alternation
violation.

Test: RED before the fix (scaffolding persists, repair count > 0); GREEN after
the fix pops trailing synthetic scaffolding before appending the short-circuit
assistant turn.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.conversation_loop import run_conversation
from agent.turn_context import TurnContext
from run_agent import AIAgent


def _messages_with_leftover_scaffolding():
    """History tail = prior empty-response nudge scaffolding + this turn's user.

    Shape:
        user(real, prior turn)
        assistant("(empty)", _empty_recovery_synthetic)   <- leftover
        user(nudge, _empty_recovery_synthetic)             <- leftover
        user(this turn's real message)                     <- appended by prologue
    """
    return [
        {"role": "user", "content": "do the prior task"},
        {"role": "assistant", "content": "Prior task done."},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": (
                "You just executed tool calls but returned an empty response. "
                "Please process the tool results above and continue with the task."
            ),
            "_empty_recovery_synthetic": True,
        },
        {"role": "user", "content": "what's the weather?"},
    ]


def _count_consecutive_user_violations(messages):
    violations = 0
    prev_role = None
    for m in messages:
        role = m.get("role")
        if role == "user" and prev_role == "user":
            violations += 1
        prev_role = role
    return violations


def test_short_circuit_strips_leftover_synthetic_scaffolding():
    messages = _messages_with_leftover_scaffolding()

    ctx = TurnContext(
        user_message="what's the weather?",
        original_user_message="what's the weather?",
        messages=messages,
        conversation_history=[],
        active_system_prompt=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        current_turn_user_idx=len(messages) - 1,
        short_circuit_response="It is sunny.",
    )

    agent = SimpleNamespace(api_mode="standard")

    with patch(
        "agent.conversation_loop.build_turn_context", return_value=ctx
    ):
        result = run_conversation(agent, "what's the weather?")

    out = result["messages"]

    # No synthetic scaffolding may survive into the persisted/returned history.
    assert all(
        not m.get("_empty_recovery_synthetic")
        and not m.get("_empty_terminal_sentinel")
        for m in out
    ), f"synthetic scaffolding leaked into returned history: {out}"

    # And the tail must be a well-formed assistant turn with no consecutive-user
    # alternation violation (the source of the "Repaired N" log spam).
    assert _count_consecutive_user_violations(out) == 0, out
    assert out[-1] == {"role": "assistant", "content": "It is sunny."}
    assert result["final_response"] == "It is sunny."
    assert result["completed"] is True


class _RecordingSessionDB:
    """Captures append_message calls so the flush path can be asserted on."""

    def __init__(self):
        self.appended = []

    def append_message(self, **kwargs):
        self.appended.append(kwargs)


def test_session_flush_drops_nontrailing_synthetic_scaffolding():
    """The DB flush path must never persist synthetic scaffolding, even when a
    synthetic message is followed by real content (non-trailing) — the trailing-
    only stripper would miss it and the next turn would replay it.
    """
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = _RecordingSessionDB()
    agent._session_db_created = True
    agent.session_id = "sess-1"
    agent._last_flushed_db_idx = 0
    agent._apply_persist_user_message_override = lambda messages: None
    agent._ensure_db_session = lambda: None

    messages = [
        {"role": "user", "content": "hi"},
        # Non-trailing synthetic scaffolding: real assistant content follows it.
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {"role": "assistant", "content": "Hello!"},
    ]

    AIAgent._flush_messages_to_session_db(agent, messages, conversation_history=[])

    persisted_contents = [row["content"] for row in agent._session_db.appended]
    assert "(empty)" not in persisted_contents, persisted_contents
    assert "Hello!" in persisted_contents
    assert "hi" in persisted_contents
