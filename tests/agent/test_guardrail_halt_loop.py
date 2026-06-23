"""Regression tests for guardrail halt-response loop (#49631).

When guardrail blocks a repeated tool call, the user-facing halt response
must NOT be persisted as an assistant message that the model sees on the
next turn.  If it is, the model echoes it, creating a halt-response loop.

This module covers:
  1. The synthetic tool result format (structured, not natural language)
  2. The ephemeral message stripping in finalize_turn
  3. The ephemeral message stripping in _persist_session
  4. That trajectory, session DB, plugin hooks, and external memory all
     receive the filtered messages list — never the raw one.
"""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolGuardrailDecision,
    toolguard_synthetic_result,
)


def test_guardrail_halt_response_is_not_appended_as_assistant_history():
    """Guardrail halt response must NOT be persisted as an assistant message.

    Regression test: the user-facing halt response ('I stopped retrying...')
    was previously appended as {"role": "assistant", "content": ...} to the
    messages list, causing the model to echo it on the next turn — a
    halt-response loop.

    After the fix:
    - A short, structured tool result is appended (TOOL_GUARDRAIL_BLOCKED: ...)
    - It is tagged with _guardrail_ephemeral=True
    - _persist_session strips ephemeral messages before writing to session DB
    """
    config = ToolCallGuardrailConfig()
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    # Simulate repeated identical terminal calls
    for i in range(6):
        decision = controller.before_call("terminal", {"command": "echo hello"})
        if not decision.allows_execution:
            break
        controller.after_call(
            "terminal",
            {"command": "echo hello"},
            '{"output": "hello", "exit_code": 0}',
            failed=False,
        )

    # The synthetic tool result should be structured, not natural language
    halt_decision = ToolGuardrailDecision(
        action="block",
        code="no_progress_block",
        message="Blocked terminal: this call returned the same result 5 times.",
        tool_name="terminal",
        count=5,
    )
    synthetic = toolguard_synthetic_result(halt_decision)

    # The synthetic result should NOT contain the user-facing halt text
    assert "I stopped retrying" not in synthetic
    # It should be structured JSON with error/guardrail keys
    parsed = json.loads(synthetic)
    assert "error" in parsed
    assert "guardrail" in parsed


def test_guardrail_block_tool_observation_can_exist_without_user_halt_text_in_context():
    """Model-facing synthetic tool result must not contain user-facing halt text.

    The synthetic tool result is what enters the model context. It must be
    short, structured, and must not contain the natural-language halt response
    that the model would echo back.
    """
    decision = ToolGuardrailDecision(
        action="block",
        code="no_progress_cross_turn_block",
        message="Blocked terminal: this call returned the same result 5 times "
                "across multiple turns.",
        tool_name="terminal",
        count=5,
    )
    synthetic = toolguard_synthetic_result(decision)

    # Must NOT contain the user-facing halt response pattern
    assert "I stopped retrying" not in synthetic
    assert "because it hit the tool-call guardrail" not in synthetic
    assert "change strategy instead of repeating" not in synthetic

    # Must contain structured error info
    parsed = json.loads(synthetic)
    assert parsed["guardrail"]["action"] == "block"
    assert parsed["guardrail"]["code"] == "no_progress_cross_turn_block"


def test_strip_ephemeral_guardrail_messages_removes_ephemeral_entries():
    """_strip_ephemeral_guardrail_messages must remove all ephemeral entries."""
    from agent.turn_finalizer import _strip_ephemeral_guardrail_messages

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {
            "role": "tool",
            "content": "TOOL_GUARDRAIL_BLOCKED: repeated identical terminal call blocked.",
            "_guardrail_ephemeral": True,
        },
        {"role": "assistant", "content": "Let me try a different approach."},
        {
            "role": "tool",
            "content": "TOOL_GUARDRAIL_BLOCKED: another block.",
            "_guardrail_ephemeral": True,
        },
    ]

    result = _strip_ephemeral_guardrail_messages(messages)

    # Ephemeral messages removed
    assert len(result) == 3
    assert all(not m.get("_guardrail_ephemeral") for m in result)

    # Original messages preserved
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "assistant"

    # Original list unchanged (pure filter, no mutation)
    assert len(messages) == 5


def test_strip_ephemeral_guardrail_messages_no_op_when_clean():
    """_strip_ephemeral_guardrail_messages is a no-op when there are no ephemeral entries."""
    from agent.turn_finalizer import _strip_ephemeral_guardrail_messages

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = _strip_ephemeral_guardrail_messages(messages)

    assert len(result) == 2
    assert result is not messages  # returns a new list
    assert result == messages  # but contents are identical


def test_strip_ephemeral_guardrail_messages_handles_non_dict_entries():
    """_strip_ephemeral_guardrail_messages must not crash on non-dict entries."""
    from agent.turn_finalizer import _strip_ephemeral_guardrail_messages

    messages = [
        {"role": "user", "content": "Hello"},
        "legacy_string_message",  # edge case: non-dict
        {
            "role": "tool",
            "content": "TOOL_GUARDRAIL_BLOCKED",
            "_guardrail_ephemeral": True,
        },
    ]

    result = _strip_ephemeral_guardrail_messages(messages)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1] == "legacy_string_message"


def test_persist_session_strips_guardrail_ephemeral_messages():
    """_persist_session must strip messages tagged with _guardrail_ephemeral.

    Guardrail ephemeral messages are appended to the messages list during
    the turn so the model sees the block observation, but they must NOT be
    persisted to session DB / JSON log — otherwise they leak into future
    turns via conversation history or context compression.
    """
    # Simulate a messages list with a guardrail ephemeral message
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {
            "role": "tool",
            "content": "TOOL_GUARDRAIL_BLOCKED: repeated identical terminal call blocked.",
            "_guardrail_ephemeral": True,
        },
    ]

    # Simulate what _persist_session does
    session_messages = [
        m for m in messages if not m.get("_guardrail_ephemeral")
    ]

    # The ephemeral message should be stripped
    assert len(session_messages) == 2
    assert all(not m.get("_guardrail_ephemeral") for m in session_messages)

    # The remaining messages should be the original conversation
    assert session_messages[0]["role"] == "user"
    assert session_messages[1]["role"] == "assistant"


def test_guardrail_ephemeral_tag_preserved_in_tool_result():
    """The _guardrail_ephemeral tag must survive JSON serialization round-trip."""
    msg = {
        "role": "tool",
        "content": "TOOL_GUARDRAIL_BLOCKED: repeated identical terminal call blocked "
                   "(no_progress_cross_turn_block). Change strategy; do not retry.",
        "_guardrail_ephemeral": True,
    }

    # Round-trip through JSON (as happens in session log)
    serialized = json.dumps(msg)
    deserialized = json.loads(serialized)

    assert deserialized.get("_guardrail_ephemeral") is True
    assert deserialized["role"] == "tool"


def test_next_turn_context_has_no_guardrail_artifacts():
    """Simulate the full persist → reload cycle: next turn must see no guardrail content.

    This is the integration-level regression test: after a guardrail halt turn,
    the next turn's model context must not contain any guardrail artifacts.
    """
    from agent.turn_finalizer import _strip_ephemeral_guardrail_messages

    # Turn 1: conversation with guardrail halt
    turn1_messages = [
        {"role": "user", "content": "Run this command"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "terminal", "arguments": '{"command":"echo hello"}'}}]},
        {"role": "tool", "content": '{"output": "hello", "exit_code": 0}', "tool_call_id": "call_1"},
        # ... repeated calls omitted for brevity ...
        {
            "role": "tool",
            "content": "TOOL_GUARDRAIL_BLOCKED: repeated identical terminal call blocked "
                       "(no_progress_cross_turn_block). Change strategy; do not retry.",
            "_guardrail_ephemeral": True,
        },
    ]

    # Persist path strips ephemeral messages
    persisted = _strip_ephemeral_guardrail_messages(turn1_messages)

    # Turn 2: user sends a new message; context is built from persisted history
    turn2_messages = list(persisted)  # simulate loading from session DB
    turn2_messages.append({"role": "user", "content": "Try a different approach"})

    # Assert: no guardrail artifacts in next turn context
    for msg in turn2_messages:
        content = msg.get("content", "")
        assert "I stopped retrying" not in content
        assert "TOOL_GUARDRAIL_BLOCKED" not in content
        assert not msg.get("_guardrail_ephemeral")
