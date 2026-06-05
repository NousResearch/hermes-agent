"""Tests for Slack channel_skill_bindings auto-skill resolution."""
from unittest.mock import MagicMock


def _make_adapter(extra=None):
    """Create a minimal SlackAdapter stub with the given ``config.extra``."""
    from gateway.platforms.slack import SlackAdapter
    adapter = object.__new__(SlackAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra or {}
    return adapter


def _resolve(adapter, channel_id, parent_id=None):
    from gateway.platforms.base import resolve_channel_skills
    return resolve_channel_skills(adapter.config.extra, channel_id, parent_id)


class TestSlackResolveChannelSkills:
    def test_no_bindings_returns_none(self):
        adapter = _make_adapter()
        assert _resolve(adapter, "D0ABC") is None

    def test_match_by_dm_channel_id(self):
        """The primary use case: binding a skill to a Slack DM channel."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]

    def test_match_by_parent_id_for_thread(self):
        """Slack threads inherit the parent channel's binding."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "C0PARENT", "skills": ["parent-skill"]},
            ]
        })
        assert _resolve(adapter, "thread-ts-123", parent_id="C0PARENT") == ["parent-skill"]

    def test_no_match_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0AAA", "skills": ["skill-a"]},
            ]
        })
        assert _resolve(adapter, "D0BBB") is None

    def test_single_skill_string(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skill": "german-flashcards"},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["german-flashcards"]

    def test_dedup_preserves_order(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["a", "b", "a", "c", "b"]},
            ]
        })
        assert _resolve(adapter, "D0ATH9TQ0G6") == ["a", "b", "c"]

    def test_multiple_bindings_pick_correct(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0AAA", "skills": ["skill-a"]},
                {"id": "D0BBB", "skills": ["skill-b"]},
                {"id": "D0CCC", "skills": ["skill-c"]},
            ]
        })
        assert _resolve(adapter, "D0BBB") == ["skill-b"]

    def test_malformed_entry_skipped(self):
        """Non-dict entries should be ignored, not raise."""
        adapter = _make_adapter({
            "channel_skill_bindings": [
                "not-a-dict",
                {"id": "D0ABC", "skills": ["good"]},
            ]
        })
        assert _resolve(adapter, "D0ABC") == ["good"]

    def test_empty_skills_list_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ABC", "skills": []},
            ]
        })
        assert _resolve(adapter, "D0ABC") is None

    def test_empty_skill_string_returns_none(self):
        adapter = _make_adapter({
            "channel_skill_bindings": [
                {"id": "D0ABC", "skill": ""},
            ]
        })
        assert _resolve(adapter, "D0ABC") is None


class TestSlackMessageEventAutoSkill:
    """Integration-style test: verify auto_skill propagates to MessageEvent."""

    def test_message_event_carries_auto_skill(self):
        """Simulate the handler wiring: resolve + attach to MessageEvent."""
        from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource, resolve_channel_skills

        config_extra = {
            "channel_skill_bindings": [
                {"id": "D0ATH9TQ0G6", "skills": ["german-flashcards"]},
            ]
        }
        auto_skill = resolve_channel_skills(config_extra, "D0ATH9TQ0G6", None)

        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="D0ATH9TQ0G6",
            chat_name="Mats",
            chat_type="dm",
            user_id="U0ABC",
            user_name="Mats",
        )
        event = MessageEvent(
            text="work",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="123.456",
            auto_skill=auto_skill,
        )
        assert event.auto_skill == ["german-flashcards"]
