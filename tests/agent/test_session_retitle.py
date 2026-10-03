from unittest.mock import MagicMock, patch

import pytest

from agent.session_retitle import (
    RECENT_RETITLE_MESSAGES,
    RetitleGenerationError,
    RetitlePersistenceError,
    recent_retitle_context,
    retitle_session,
)
from agent.title_generator import MAX_TITLE_INPUT_CHARS


def test_recent_context_keeps_newest_real_exchange_and_its_followup_context():
    history = [
        {"role": "user", "content": "Fix the login regression in auth.py"},
        {"role": "assistant", "content": "The bug is in token refresh."},
        {"role": "tool", "content": "large tool output"},
        {"role": "assistant", "content": "calling another tool", "tool_calls": [{"id": "1"}]},
        {"role": "user", "content": "compressed machinery", "_compressed_summary": True},
        {"role": "user", "content": "background completion", "display_kind": "internal_notification"},
        {"role": "user", "content": "[System: The active model for this chat has changed to foo]"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "I am updating the refresh guard now."},
    ]

    context = recent_retitle_context(history)

    assert RECENT_RETITLE_MESSAGES == 4
    assert context.splitlines() == [
        "User: Fix the login regression in auth.py",
        "Assistant: The bug is in token refresh.",
        "User: continue",
        "Assistant: I am updating the refresh guard now.",
    ]
    assert len(context) <= MAX_TITLE_INPUT_CHARS


def test_retitle_distinguishes_context_generation_and_persistence_outcomes():
    db = MagicMock()
    db.get_conversation_root.return_value = "root-1"
    db.set_session_title.return_value = True
    db.get_session_title.return_value = "Persisted exact title"

    with patch("agent.session_retitle.generate_retitle") as generate:
        assert retitle_session(db, "session-1", [{"role": "tool", "content": "noise"}]) is None
        generate.assert_not_called()

        generate.side_effect = RuntimeError("provider down")
        with pytest.raises(RetitleGenerationError, match="provider down"):
            retitle_session(db, "session-1", [{"role": "user", "content": "Fix auth"}])

        generate.side_effect = None
        generate.return_value = "Generated title"
        db.set_session_title.return_value = False
        with pytest.raises(RetitlePersistenceError, match="not found"):
            retitle_session(db, "session-1", [{"role": "user", "content": "Fix auth"}])

        db.set_session_title.return_value = True
        assert retitle_session(
            db, "session-1", [{"role": "user", "content": "Fix auth"}]
        ) == "Persisted exact title"

    db.set_session_title.assert_called_with("session-1", "Generated title")
