"""Recovery mode tests for tool-call guardrails (#49631).

When guardrail blocks a repeated tool call, the agent should enter recovery
mode instead of immediately halting. The model gets a limited number of
recovery attempts (default: 2) to re-plan with a different strategy.

This module covers:
  1. Recovery budget tracking (attempts, exhaustion)
  2. Blocked signature tracking (prevents retrying same call during recovery)
  3. Recovery observation format (structured, not natural language)
  4. Final halt after recovery budget exhausted
  5. Different tool/args allowed during recovery
"""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolGuardrailDecision,
    toolguard_synthetic_result,
    ToolCallSignature,
)


def test_recovery_budget_starts_at_zero():
    """Recovery budget starts at 0 for a fresh turn."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    assert controller.recovery_attempts == 0
    assert not controller.recovery_exhausted


def test_recovery_budget_resets_each_turn():
    """Recovery budget resets when a new turn starts."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    # Use up recovery budget
    sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})
    controller.record_recovery(sig)
    controller.record_recovery(sig)
    assert controller.recovery_attempts == 2
    assert controller.recovery_exhausted

    # New turn resets budget
    controller.reset_for_turn()
    assert controller.recovery_attempts == 0
    assert not controller.recovery_exhausted


def test_recovery_exhausted_after_max_attempts():
    """Recovery is exhausted after max_recovery_attempts (default: 2)."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})

    # First recovery attempt
    controller.record_recovery(sig)
    assert controller.recovery_attempts == 1
    assert not controller.recovery_exhausted

    # Second recovery attempt (max)
    controller.record_recovery(sig)
    assert controller.recovery_attempts == 2
    assert controller.recovery_exhausted


def test_blocked_signature_prevents_retry():
    """A blocked signature cannot be retried during recovery."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})

    # Record this signature as blocked
    controller.record_recovery(sig)

    # Check that it's blocked
    assert controller.is_signature_blocked(sig)

    # before_call should block this signature
    decision = controller.before_call("terminal", {"command": "echo hello"})
    assert decision.action == "block"
    assert decision.code == "recovery_retry_block"
    assert "already blocked this turn" in decision.message


def test_different_tool_allowed_during_recovery():
    """A different tool call is allowed during recovery."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    # Block terminal
    terminal_sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})
    controller.record_recovery(terminal_sig)

    # Different tool should be allowed
    decision = controller.before_call("web_search", {"query": "test"})
    assert decision.allows_execution

    # Same tool with different args should be allowed
    decision2 = controller.before_call("terminal", {"command": "ls -la"})
    assert decision2.allows_execution


def test_same_tool_different_args_allowed():
    """Same tool with different arguments is allowed during recovery."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    # Block one command
    blocked_sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})
    controller.record_recovery(blocked_sig)

    # Different command with same tool should be allowed
    decision = controller.before_call("terminal", {"command": "echo world"})
    assert decision.allows_execution


def test_recovery_retry_block_message_is_structured():
    """Recovery retry block message is actionable, not natural language."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})
    controller.record_recovery(sig)

    decision = controller.before_call("terminal", {"command": "echo hello"})

    # Must be a block action
    assert decision.action == "block"
    assert decision.code == "recovery_retry_block"

    # Must contain actionable guidance
    assert "already blocked" in decision.message
    assert "materially different strategy" in decision.message

    # Must NOT contain user-facing halt text that model would echo
    assert "I stopped retrying" not in decision.message


def test_synthetic_result_does_not_contain_user_halt_text():
    """Synthetic tool result must not contain user-facing halt text."""
    decision = ToolGuardrailDecision(
        action="block",
        code="no_progress_cross_turn_block",
        message="Blocked terminal: this call returned the same result 5 times.",
        tool_name="terminal",
        count=5,
    )
    synthetic = toolguard_synthetic_result(decision)

    # Must NOT contain user-facing halt text
    assert "I stopped retrying" not in synthetic
    assert "because it hit the tool-call guardrail" not in synthetic

    # Must be structured JSON
    parsed = json.loads(synthetic)
    assert "error" in parsed
    assert "guardrail" in parsed


def test_recovery_observation_format():
    """Recovery observation uses TOOL_GUARDRAIL_RECOVERY_REQUIRED prefix."""
    # Simulate what conversation_loop.py appends
    observation = {
        "role": "tool",
        "content": (
            "TOOL_GUARDRAIL_RECOVERY_REQUIRED: blocked_tool=terminal; "
            "blocked_code=no_progress_cross_turn_block; "
            "recovery_attempt=1/2; "
            "The previous tool call made no progress repeatedly. "
            "Do not retry the same tool with the same arguments. "
            "Re-plan from the original user goal."
        ),
        "_guardrail_ephemeral": True,
    }

    # Must have the right prefix
    assert observation["content"].startswith("TOOL_GUARDRAIL_RECOVERY_REQUIRED:")

    # Must contain key info
    assert "blocked_tool=terminal" in observation["content"]
    assert "blocked_code=no_progress_cross_turn_block" in observation["content"]
    assert "recovery_attempt=1/2" in observation["content"]

    # Must NOT contain user-facing halt text
    assert "I stopped retrying" not in observation["content"]

    # Must be ephemeral
    assert observation["_guardrail_ephemeral"] is True


def test_full_recovery_then_halt_cycle():
    """Simulate the full cycle: block → recovery → block again → final halt.

    1. Model calls terminal repeatedly → guardrail blocks
    2. Recovery attempt 1: model tries same call → blocked again
    3. Recovery attempt 2: model tries same call → blocked again
    4. Recovery exhausted → final halt
    """
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})

    # First block → record recovery
    controller.record_recovery(sig)
    assert controller.recovery_attempts == 1
    assert not controller.recovery_exhausted

    # Recovery attempt 1: model retries same call → blocked
    decision1 = controller.before_call("terminal", {"command": "echo hello"})
    assert decision1.action == "block"
    assert decision1.code == "recovery_retry_block"

    # Record second recovery
    controller.record_recovery(sig)
    assert controller.recovery_attempts == 2
    assert controller.recovery_exhausted

    # Recovery exhausted → should final halt
    # (In conversation_loop.py, this triggers the break path)
    assert controller.recovery_exhausted is True


def test_recovery_allows_different_strategy():
    """After a block, model can use a different tool/args and succeed."""
    config = ToolCallGuardrailConfig(hard_stop_enabled=True)
    controller = ToolCallGuardrailController(config)
    controller.reset_for_turn()

    # Block terminal with one command
    terminal_sig = ToolCallSignature.from_call("terminal", {"command": "echo hello"})
    controller.record_recovery(terminal_sig)

    # Model chooses web_search instead → should be allowed
    decision = controller.before_call("web_search", {"query": "alternative approach"})
    assert decision.allows_execution

    # Model chooses terminal with different command → should be allowed
    decision2 = controller.before_call("terminal", {"command": "ls -la /tmp"})
    assert decision2.allows_execution
