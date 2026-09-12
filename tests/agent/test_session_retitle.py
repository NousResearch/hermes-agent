from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.session_retitle import (
    RECENT_RETITLE_MESSAGES,
    generate_retitle,
    recent_retitle_context,
    retitle_session,
)
from agent.title_generator import MAX_TITLE_INPUT_CHARS


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_recent_context_keeps_four_real_conversation_messages():
    history = [
        {"role": "user", "content": "Fix the login regression in auth.py"},
        {"role": "assistant", "content": "The regression comes from the token refresh path."},
        {"role": "tool", "content": "large tool output that should not title the session"},
        {"role": "user", "content": "[System: The active model for this chat has changed to foo]"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "I am updating the refresh guard now."},
    ]

    context = recent_retitle_context(history)

    assert RECENT_RETITLE_MESSAGES == 4
    assert context.splitlines() == [
        "User: Fix the login regression in auth.py",
        "Assistant: The regression comes from the token refresh path.",
        "User: continue",
        "Assistant: I am updating the refresh guard now.",
    ]


def test_recent_context_reuses_existing_total_input_budget():
    history = [
        {"role": "user", "content": "old " + "x" * 1500},
        {"role": "assistant", "content": "latest " + "y" * 1500},
    ]

    context = recent_retitle_context(history)

    assert len(context) <= MAX_TITLE_INPUT_CHARS
    assert context.endswith("y" * 100)


def test_generate_retitle_accepts_cjk_title_without_word_count_gate():
    with patch("agent.session_retitle.call_llm", return_value=_response('{"title":"修复会话标题生成"}')):
        title = generate_retitle("User: 继续修复标题生成\nAssistant: 已定位到上下文选择逻辑")

    assert title == "修复会话标题生成"


def test_explicit_retitle_replaces_current_title_through_user_write():
    db = MagicMock()
    db.get_conversation_root.return_value = "root-1"
    db.set_session_title.return_value = True
    history = [{"role": "user", "content": "Fix title regeneration"}]

    with patch("agent.session_retitle.generate_retitle", return_value="Fix title regeneration"):
        title = retitle_session(db, "session-1", history)

    assert title == "Fix title regeneration"
    db.set_session_title.assert_called_once_with("session-1", "Fix title regeneration")
    db.set_auto_title.assert_not_called()


def test_retitle_does_not_call_model_without_real_conversation_context():
    db = MagicMock()
    history = [
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "[System: The active model for this chat has changed to foo]"},
    ]

    with patch("agent.session_retitle.generate_retitle") as generate:
        assert retitle_session(db, "session-1", history) is None

    generate.assert_not_called()
    db.set_session_title.assert_not_called()


def test_retitle_raises_when_title_generation_returns_nothing():
    db = MagicMock()
    db.get_conversation_root.return_value = "session-1"

    with patch("agent.session_retitle.generate_retitle", return_value=None):
        with pytest.raises(RuntimeError, match="title generation returned no title"):
            retitle_session(
                db,
                "session-1",
                [{"role": "user", "content": "Rename this conversation"}],
            )

    db.set_session_title.assert_not_called()


def test_retitle_raises_when_session_write_does_not_land():
    db = MagicMock()
    db.get_conversation_root.return_value = "session-1"
    db.set_session_title.return_value = False

    with patch("agent.session_retitle.generate_retitle", return_value="New title"):
        with pytest.raises(RuntimeError, match="not found while storing title"):
            retitle_session(
                db,
                "session-1",
                [{"role": "user", "content": "Rename this conversation"}],
            )
