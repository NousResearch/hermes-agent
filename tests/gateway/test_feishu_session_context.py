"""Tests for Feishu chat_id and chat_type injection in session context prompts."""

from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context_prompt,
)
from gateway.config import Platform


def _make_feishu_context(chat_id="oc_abc123", chat_type="group", **kwargs):
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=kwargs.pop("user_id", "user-1"),
        user_name=kwargs.pop("user_name", None),
    )
    return SessionContext(
        source=source,
        connected_platforms=[Platform.FEISHU],
        home_channels=kwargs.pop("home_channels", {}),
    )


class TestFeishuChatIdInjection:
    def test_chat_id_in_prompt(self):
        ctx = _make_feishu_context(chat_id="oc_abc123")
        prompt = build_session_context_prompt(ctx)
        assert "Feishu Chat ID" in prompt
        assert "oc_abc123" in prompt

    def test_chat_type_in_prompt(self):
        ctx = _make_feishu_context(chat_type="group")
        prompt = build_session_context_prompt(ctx)
        assert "Chat Type" in prompt
        assert "group" in prompt

    def test_chat_type_present_without_chat_id(self):
        """chat_type should appear even when chat_id is empty."""
        ctx = _make_feishu_context(chat_id="", chat_type="dm")
        prompt = build_session_context_prompt(ctx)
        assert "Chat Type" in prompt
        assert "dm" in prompt

    def test_chat_id_present_without_chat_type(self):
        ctx = _make_feishu_context(chat_id="oc_xyz", chat_type="")
        prompt = build_session_context_prompt(ctx)
        assert "Feishu Chat ID" in prompt
        assert "oc_xyz" in prompt
        assert "Chat Type" not in prompt

    def test_both_empty(self):
        ctx = _make_feishu_context(chat_id="", chat_type="")
        prompt = build_session_context_prompt(ctx)
        assert "Feishu Chat ID" not in prompt
        assert "Chat Type" not in prompt


class TestFeishuPromptInjectionSafety:
    def test_backticks_stripped_from_chat_id(self):
        ctx = _make_feishu_context(chat_id="oc_`injected`_abc")
        prompt = build_session_context_prompt(ctx)
        # Backticks should be stripped from the value
        assert "injected" in prompt
        # The value inside the backtick-wrapped code span should not contain backticks
        assert "`oc_injected_abc`" in prompt

    def test_backticks_stripped_from_chat_type(self):
        ctx = _make_feishu_context(chat_type="group`malicious`")
        prompt = build_session_context_prompt(ctx)
        assert "groupmalicious" in prompt

    def test_only_backticks_results_in_skip(self):
        """If value is only backticks, it should be skipped entirely."""
        ctx = _make_feishu_context(chat_id="```")
        prompt = build_session_context_prompt(ctx)
        assert "Feishu Chat ID" not in prompt

    def test_control_chars_preserved_in_id(self):
        """Newlines/tabs are stripped by strip(), but normal text is kept."""
        ctx = _make_feishu_context(chat_id="  oc_abc  ")
        prompt = build_session_context_prompt(ctx)
        assert "oc_abc" in prompt
