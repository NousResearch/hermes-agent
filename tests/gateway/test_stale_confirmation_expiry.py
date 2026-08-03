"""Tests for stale confirmation text expiry in gateway history.

Reproduces the failure mode in #59607: when a host-restart-causing tool
runs, the user confirmation text remains in conversation history. On
resume, the LLM sees the confirmation at the tail and may re-execute the
destructive action if the user sends any follow-up.

The fix: high-risk confirmation messages (user messages containing a
known dangerous confirmation pattern) are tagged with a creation
timestamp when they enter conversation history. On history replay, if
the confirmation is older than EXPIRY (60s by default), the confirmation
text is stripped from agent history before the model sees it.

This complements 75ed07ace (which handles dangling assistant tool_calls
tail) by handling the user-side confirmation tail.
"""

import time
import pytest
from typing import Dict, List

from gateway.run import (
    _build_gateway_agent_history,
    _is_dangerous_confirmation,
    _strip_stale_dangerous_confirmations,
)


# High-risk confirmation patterns. A user message matching one of these
# (case-insensitive) is considered a "confirmation text" and is subject
# to the expiry rule. Add new patterns here as new high-risk side effects
# are introduced.
def test_dangerous_confirmation_helper():
    """The pattern matcher is case-insensitive and substring-based."""
    assert _is_dangerous_confirmation("confirm forced restart")
    assert _is_dangerous_confirmation("CONFIRM FORCED RESTART")
    assert _is_dangerous_confirmation("  confirm forced restart please  ")
    assert _is_dangerous_confirmation("I want to confirm forced restart the server")

    # i18n
    assert _is_dangerous_confirmation("確認強制重開機")

    # Not a confirmation
    assert not _is_dangerous_confirmation("can you restart the docker container?")
    assert not _is_dangerous_confirmation("hello world")
    assert not _is_dangerous_confirmation("")
    assert not _is_dangerous_confirmation(None)
    assert not _is_dangerous_confirmation(123)


def _make_history_with_confirmation(
    *,
    user_message_at: float,
    assistant_warning_at: float,
    confirmation_message: str,
    confirmation_at: float,
    assistant_action_at: float,
) -> List[Dict]:
    """Build a synthetic conversation history with a confirmation text.

    Uses the real gateway's "timestamp" field (epoch seconds, as set in
    gateway/run.py:11616, 11650, 11692, etc).
    """
    return [
        {"role": "user", "content": "can you force a restart?", "timestamp": user_message_at},
        {"role": "assistant", "content": "Rebooting the host is dangerous. To confirm, please type: 'confirm forced restart'", "timestamp": assistant_warning_at},
        {"role": "user", "content": confirmation_message, "timestamp": confirmation_at},
        {"role": "assistant", "content": "OK, restarting now.", "timestamp": assistant_action_at},
    ]


def test_stale_confirmation_text_is_stripped_on_resume():
    """A confirmation text older than EXPIRY (60s by default) is stripped.

    This is the core fix for #59607.
    """
    current_time = time.time()
    # User confirmation was 5 minutes ago — well past the 60s expiry
    user_message_at = current_time - 1000
    assistant_warning_at = current_time - 999
    confirmation_at = current_time - 300  # 5 minutes ago
    assistant_action_at = current_time - 299

    history = _make_history_with_confirmation(
        user_message_at=user_message_at,
        assistant_warning_at=assistant_warning_at,
        confirmation_message="confirm forced restart",
        confirmation_at=confirmation_at,
        assistant_action_at=assistant_action_at,
    )

    agent_history, _ = _build_gateway_agent_history(history)

    # The stale confirmation text should be gone
    confirmation_present = any(
        m.get("role") == "user" and "confirm forced restart" in (m.get("content") or "")
        for m in agent_history
    )
    assert not confirmation_present, (
        f"Stale confirmation text should be stripped on resume. "
        f"Got agent_history: {agent_history}"
    )


def test_fresh_confirmation_text_is_preserved():
    """A confirmation text within EXPIRY is kept (not yet expired)."""
    current_time = time.time()
    user_message_at = current_time - 30
    assistant_warning_at = current_time - 29
    confirmation_at = current_time - 5  # 5 seconds ago — fresh
    assistant_action_at = current_time - 4

    history = _make_history_with_confirmation(
        user_message_at=user_message_at,
        assistant_warning_at=assistant_warning_at,
        confirmation_message="confirm forced restart",
        confirmation_at=confirmation_at,
        assistant_action_at=assistant_action_at,
    )

    agent_history, _ = _build_gateway_agent_history(history)

    # Fresh confirmation should still be there
    confirmation_present = any(
        m.get("role") == "user" and "confirm forced restart" in (m.get("content") or "")
        for m in agent_history
    )
    assert confirmation_present, (
        f"Fresh confirmation (5s old) should NOT be stripped. "
        f"Got agent_history: {agent_history}"
    )


def test_non_confirmation_text_is_preserved():
    """A regular user message is never treated as a confirmation."""
    current_time = time.time()
    user_message_at = current_time - 1000  # 17 min ago
    confirmation_at = current_time - 300   # 5 min ago

    history = [
        {"role": "user", "content": "can you help me with the docs?", "timestamp": user_message_at},
        {"role": "assistant", "content": "Sure, what do you need?", "timestamp": confirmation_at},
    ]

    agent_history, _ = _build_gateway_agent_history(history)

    # Both messages should still be there
    user_msgs = [m for m in agent_history if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert "help me with the docs" in user_msgs[0].get("content", "")


def test_no_dangerous_pattern_at_all_preserves_everything():
    """If the conversation has no dangerous confirmation, nothing is stripped."""
    current_time = time.time()
    user_message_at = current_time - 1000

    history = [
        {"role": "user", "content": "tell me a joke", "timestamp": user_message_at},
        {"role": "assistant", "content": "Why did the chicken cross the road?", "timestamp": user_message_at + 1},
        {"role": "user", "content": "haha", "timestamp": user_message_at + 2},
    ]

    agent_history, _ = _build_gateway_agent_history(history)

    assert len(agent_history) == 3


def test_strip_stale_dangerous_confirmations_directly():
    """Unit test the strip helper in isolation."""
    current_time = time.time()
    history = _make_history_with_confirmation(
        user_message_at=current_time - 1000,
        assistant_warning_at=current_time - 999,
        confirmation_message="confirm forced restart",
        confirmation_at=current_time - 300,
        assistant_action_at=current_time - 299,
    )

    cleaned = _strip_stale_dangerous_confirmations(history, now=current_time)

    # The dangerous confirmation should be gone
    assert not any(
        "confirm forced restart" in (m.get("content") or "")
        for m in cleaned
        if m.get("role") == "user"
    )
    # The original user question and the assistant responses stay
    assert any("can you force a restart" in (m.get("content") or "") for m in cleaned)
    assert any("Rebooting the host is dangerous" in (m.get("content") or "") for m in cleaned)
    assert any("OK, restarting now" in (m.get("content") or "") for m in cleaned)

def test_redaction_preserves_role_alternation():
    """Expiry must redact in place, never delete the user message.

    The incident tail is ``user(confirm) → assistant("OK, restarting")``.
    Deleting the user row would leave two consecutive assistant messages,
    violating the strict role-alternation invariant providers enforce.
    """
    current_time = time.time()
    history = _make_history_with_confirmation(
        user_message_at=current_time - 1000,
        assistant_warning_at=current_time - 999,
        confirmation_message="confirm forced restart",
        confirmation_at=current_time - 300,
        assistant_action_at=current_time - 299,
    )

    cleaned = _strip_stale_dangerous_confirmations(history, now=current_time)

    # Same message count — nothing deleted.
    assert len(cleaned) == len(history)
    # The user slot survives with an expiry sentinel instead of the phrase.
    redacted = cleaned[2]
    assert redacted["role"] == "user"
    assert "confirm forced restart" not in redacted["content"]
    assert "EXPIRED" in redacted["content"]
    # No two consecutive same-role messages anywhere.
    roles = [m["role"] for m in cleaned]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    # Original history object is not mutated.
    assert history[2]["content"] == "confirm forced restart"
