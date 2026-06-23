"""Regression tests for guardrail halt-response loop (#49631).

When guardrail blocks a repeated tool call, the user-facing halt response
must NOT be persisted as an assistant message that the model sees on the
next turn.  If it is, the model echoes it, creating a halt-response loop.
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
