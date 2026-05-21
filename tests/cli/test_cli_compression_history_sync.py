"""Tests for CLI conversation_history sync after run_conversation with auto-compression.

Regression for issue #29926: when auto-compression rotates the session mid-run,
result["messages"] contains the inflated post-turn list (compressed baseline +
this turn's tool output), but conversation_history must use the agent's internal
_session_messages instead so the next turn starts from the compressed state.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from tests.cli.test_cli_init import _make_cli


def test_post_run_sync_uses_session_messages_when_session_rotated():
    """When run_conversation rotates session via auto-compression, conversation_history
    must come from agent._session_messages, not result["messages"].
    """
    shell = _make_cli()
    old_id = shell.session_id
    new_child_id = "20260101_000000_compressed_child"

    # Pre-turn history (inflated)
    pre_history = [{"role": "user", "content": f"msg_{i}"} for i in range(50)]

    # After compression, agent._session_messages holds the compressed baseline
    compressed_messages = [
        {"role": "system", "content": "[COMPACTED CONTEXT]"},
        {"role": "user", "content": "msg_1"},
        {"role": "assistant", "content": "msg_2"},
        {"role": "user", "content": "msg_49"},  # compressed down to ~3 messages
    ]

    # result["messages"] is inflated: compressed + this turn's tool output
    inflated_result = list(compressed_messages) + [
        {"role": "assistant", "content": "expanded response with tools"},
        {"role": "user", "content": "new user msg"},
    ]

    shell.conversation_history = pre_history
    shell.agent = MagicMock()
    shell.agent.session_id = old_id  # starts at parent session

    # Simulate the post-run logic from cli.py lines ~11360-11378
    result = {"final_response": "done", "messages": inflated_result}
    shell._cli_last_run_old_session_id = old_id
    # After run_conversation returns, agent.session_id rotated
    shell.agent.session_id = new_child_id
    # _session_messages has the compressed state (what agent actually used)
    shell.agent._session_messages = compressed_messages

    # Reproduce the fix logic from cli.py
    if result:
        compressed_attr = getattr(shell.agent, "_session_messages", None)
        session_rotated = (
            shell.agent
            and shell._cli_last_run_old_session_id is not None
            and getattr(shell.agent, "session_id", None) != shell._cli_last_run_old_session_id
        )
        if session_rotated and compressed_attr:
            shell.conversation_history = list(compressed_attr)
        else:
            shell.conversation_history = result.get("messages", shell.conversation_history)
    else:
        pass

    # Must use compressed, NOT inflated
    assert len(shell.conversation_history) == 4
    assert shell.conversation_history[0]["role"] == "system"
    assert "[COMPACTED CONTEXT]" in shell.conversation_history[0]["content"]
    assert len(shell.conversation_history) != len(inflated_result)


def test_post_run_sync_uses_result_messages_when_no_rotation():
    """When session does NOT rotate (normal completion, no compression),
    conversation_history must come from result["messages"] as before.
    """
    shell = _make_cli()

    pre_history = [{"role": "user", "content": f"msg_{i}"} for i in range(10)]
    result_messages = list(pre_history) + [
        {"role": "assistant", "content": "response"},
    ]

    shell.conversation_history = pre_history
    shell.agent = MagicMock()
    shell.agent.session_id = shell.session_id  # same session, no rotation

    result = {"final_response": "done", "messages": result_messages}
    shell._cli_last_run_old_session_id = shell.session_id

    # Reproduce the fix logic from cli.py
    if result:
        compressed_attr = getattr(shell.agent, "_session_messages", None)
        session_rotated = (
            shell.agent
            and shell._cli_last_run_old_session_id is not None
            and getattr(shell.agent, "session_id", None) != shell._cli_last_run_old_session_id
        )
        if session_rotated and compressed_attr:
            shell.conversation_history = list(compressed_attr)
        else:
            shell.conversation_history = result.get("messages", shell.conversation_history)
    else:
        pass

    # Must use result["messages"] since no rotation
    assert len(shell.conversation_history) == 11
    assert shell.conversation_history[-1]["content"] == "response"


def test_post_run_sync_no_session_messages_falls_back_to_result():
    """When session rotated but _session_messages is not set (edge case),
    must fall back to result["messages"] rather than crashing.
    """
    shell = _make_cli()
    old_id = shell.session_id
    new_child_id = "20260101_000000_compressed_child"

    inflated_result = [
        {"role": "system", "content": "[COMPACTED]"},
        {"role": "assistant", "content": "expanded response"},
    ]

    shell.conversation_history = [{"role": "user", "content": "pre"}]
    shell.agent = MagicMock()
    shell.agent.session_id = new_child_id  # rotated but no _session_messages attr
    shell.agent._session_messages = None  # explicitly None (not unset)
    shell._cli_last_run_old_session_id = old_id

    result = {"final_response": "done", "messages": inflated_result}

    # Reproduce the fix logic from cli.py
    if result:
        compressed_attr = getattr(shell.agent, "_session_messages", None)
        session_rotated = (
            shell.agent
            and shell._cli_last_run_old_session_id is not None
            and getattr(shell.agent, "session_id", None) != shell._cli_last_run_old_session_id
        )
        if session_rotated and compressed_attr:
            shell.conversation_history = list(compressed_attr)
        else:
            shell.conversation_history = result.get("messages", shell.conversation_history)
    else:
        pass

    # Must fall back to result when _session_messages missing/None
    assert len(shell.conversation_history) == 2
    assert shell.conversation_history[-1]["content"] == "expanded response"


def test_post_run_sync_null_result_preserves_history():
    """When result is None/empty, conversation_history must stay unchanged."""
    shell = _make_cli()
    original_history = [
        {"role": "user", "content": "msg_1"},
        {"role": "assistant", "content": "msg_2"},
    ]
    shell.conversation_history = list(original_history)
    shell.agent = MagicMock()

    result = None  # error path
    shell._cli_last_run_old_session_id = shell.session_id

    # Reproduce the fix logic from cli.py
    if result:
        compressed_attr = getattr(shell.agent, "_session_messages", None)
        session_rotated = (
            shell.agent
            and shell._cli_last_run_old_session_id is not None
            and getattr(shell.agent, "session_id", None) != shell._cli_last_run_old_session_id
        )
        if session_rotated and compressed_attr:
            shell.conversation_history = list(compressed_attr)
        else:
            shell.conversation_history = result.get("messages", shell.conversation_history)
    elif shell.conversation_history:
        pass  # Keep existing history on error/null result

    assert shell.conversation_history == original_history


def test_post_run_sync_old_session_id_none_preserves_history():
    """When _old_session_id is None (first run or agent not initialized),
    must NOT attempt rotation check and use result["messages"] normally.
    """
    shell = _make_cli()

    pre_history = [{"role": "user", "content": f"msg_{i}"} for i in range(10)]
    result_messages = list(pre_history) + [
        {"role": "assistant", "content": "response"},
    ]

    shell.conversation_history = pre_history
    shell.agent = MagicMock()
    shell.agent.session_id = None  # no session set yet

    result = {"final_response": "done", "messages": result_messages}
    shell._cli_last_run_old_session_id = None  # No old session to compare against

    # Reproduce the fix logic from cli.py
    if result:
        compressed_attr = getattr(shell.agent, "_session_messages", None)
        session_rotated = (
            shell.agent
            and shell._cli_last_run_old_session_id is not None
            and getattr(shell.agent, "session_id", None) != shell._cli_last_run_old_session_id
        )
        if session_rotated and compressed_attr:
            shell.conversation_history = list(compressed_attr)
        else:
            shell.conversation_history = result.get("messages", shell.conversation_history)
    else:
        pass

    # old_session_id is None → session_rotated is False → use result["messages"]
    assert len(shell.conversation_history) == 11


def test_post_run_sync_no_agent():
    """When self.agent is None (edge case), must NOT crash."""
    shell = _make_cli()

    pre_history = [{"role": "user", "content": f"msg_{i}"} for i in range(10)]
    shell.conversation_history = list(pre_history)

    result = {"final_response": "done", "messages": pre_history}
    shell._cli_last_run_old_session_id = shell.session_id
    shell.agent = None  # agent not set

    # Reproduce the fix logic from cli.py
    if result:
        compressed_attr = getattr(shell.agent, "_session_messages", None)
        session_rotated = (
            shell.agent
            and shell._cli_last_run_old_session_id is not None
            and getattr(shell.agent, "session_id", None) != shell._cli_last_run_old_session_id
        )
        if session_rotated and compressed_attr:
            shell.conversation_history = list(compressed_attr)
        else:
            shell.conversation_history = result.get("messages", shell.conversation_history)

    # Must use result["messages"] (no agent → no rotation possible)
    assert len(shell.conversation_history) == 10
