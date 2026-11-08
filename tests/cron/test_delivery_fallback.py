"""Unit tests for cron.delivery_fallback (pure helpers).

Covers the definitive-vs-uncertain failure decision, the parent/home fallback
target resolution across both thread models, and the redirect notice text.
"""

import pytest

from cron.delivery_fallback import (
    build_fallback_targets,
    format_fallback_notice,
    is_definitive_delivery_failure,
)


class TestIsDefinitiveDeliveryFailure:
    @pytest.mark.parametrize(
        "error",
        [
            "404 Not Found (error code: 10003): Unknown Channel",  # Discord
            "403 Forbidden (error code: 50001): Missing Access",   # Discord
            "channel_not_found",                                    # Slack
            "is_archived",                                          # Slack
            "Bad Request: thread not found",                        # Telegram
            "Bad Request: topic_closed",                            # Telegram
            "This thread is locked",                                # generic
        ],
    )
    def test_definitive_failures(self, error):
        assert is_definitive_delivery_failure(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            None,
            "",
            "Timeout sending message",
            "503 Service Unavailable",
            "Too Many Requests: retry after 30",
            "some entirely novel provider message",
            "media file not found on disk",
        ],
    )
    def test_uncertain_failures_are_not_definitive(self, error):
        assert is_definitive_delivery_failure(error) is False

    def test_accepts_exception_objects(self):
        assert is_definitive_delivery_failure(Exception("Unknown Channel")) is True
        assert is_definitive_delivery_failure(TimeoutError("timed out")) is False


class TestBuildFallbackTargets:
    def test_discord_model_a_parent_then_home(self):
        """Discord: chat_id IS the thread; redirect to the separate parent id, then home."""
        target = {"platform": "discord", "chat_id": "thread-1", "thread_id": "thread-1"}
        out = build_fallback_targets(
            target,
            parent_chat_id="parent-chan",
            home_chat_id="home-chan",
            home_thread_id=None,
        )
        assert out == [
            {"platform": "discord", "chat_id": "parent-chan", "thread_id": None, "fallback_kind": "parent"},
            {"platform": "discord", "chat_id": "home-chan", "thread_id": None, "fallback_kind": "home"},
        ]

    def test_telegram_model_b_drops_topic_to_channel(self):
        """Telegram: chat_id is the channel, thread_id the topic; drop the topic."""
        target = {"platform": "telegram", "chat_id": "-100chan", "thread_id": "topic-9"}
        out = build_fallback_targets(target, home_chat_id="home-chan")
        assert out[0] == {
            "platform": "telegram",
            "chat_id": "-100chan",
            "thread_id": None,
            "fallback_kind": "parent",
        }
        assert out[1]["fallback_kind"] == "home"
        assert out[1]["chat_id"] == "home-chan"

    def test_home_thread_id_is_preserved(self):
        target = {"platform": "telegram", "chat_id": "c1", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="home", home_thread_id="home-topic")
        assert out == [
            {"platform": "telegram", "chat_id": "home", "thread_id": "home-topic", "fallback_kind": "home"},
        ]

    def test_flat_dm_with_no_home_has_no_fallbacks(self):
        target = {"platform": "sms", "chat_id": "+15550001111", "thread_id": None}
        assert build_fallback_targets(target) == []

    def test_parent_equal_to_failed_target_is_skipped(self):
        target = {"platform": "discord", "chat_id": "same", "thread_id": "same"}
        out = build_fallback_targets(target, parent_chat_id="same", home_chat_id="home")
        # parent == current chat -> skipped; only home remains.
        assert [t["fallback_kind"] for t in out] == ["home"]

    def test_home_equal_to_failed_target_is_skipped(self):
        target = {"platform": "telegram", "chat_id": "home", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="home")
        assert out == []

    def test_home_deduped_against_parent(self):
        """If the parent channel and home channel are the same place, list it once."""
        target = {"platform": "discord", "chat_id": "thread-1", "thread_id": "thread-1"}
        out = build_fallback_targets(target, parent_chat_id="shared", home_chat_id="shared")
        assert [t["fallback_kind"] for t in out] == ["parent"]

    def test_model_a_takes_precedence_over_model_b(self):
        """An explicit parent id wins over dropping a differing thread id."""
        target = {"platform": "matrix", "chat_id": "room-thread", "thread_id": "tid"}
        out = build_fallback_targets(target, parent_chat_id="real-parent")
        assert out == [
            {"platform": "matrix", "chat_id": "real-parent", "thread_id": None, "fallback_kind": "parent"},
        ]

    def test_dm_suppresses_home_fallback(self):
        """A 1:1 DM that fails must not escalate to a (shared) home channel."""
        target = {"platform": "telegram", "chat_id": "user-123", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="group-home", is_direct_message=True)
        assert out == []  # no leak to the shared home channel

    def test_dm_still_drops_topic_to_dm_root(self):
        """A deleted DM *topic* still falls back to the DM root (stays private)."""
        target = {"platform": "telegram", "chat_id": "user-123", "thread_id": "topic-5"}
        out = build_fallback_targets(target, home_chat_id="group-home", is_direct_message=True)
        assert out == [
            {"platform": "telegram", "chat_id": "user-123", "thread_id": None, "fallback_kind": "parent"},
        ]

    def test_non_dm_still_uses_home(self):
        target = {"platform": "slack", "chat_id": "C-archived", "thread_id": None}
        out = build_fallback_targets(target, home_chat_id="slack-home", is_direct_message=False)
        assert [t["fallback_kind"] for t in out] == ["home"]


class TestFormatFallbackNotice:
    def test_parent_notice_prepended(self):
        out = format_fallback_notice("Daily report ready.", "parent")
        assert out.startswith("⚠️")
        assert "parent channel" in out
        assert out.endswith("Daily report ready.")

    def test_home_notice_prepended(self):
        out = format_fallback_notice("Daily report ready.", "home")
        assert out.startswith("⚠️")
        assert "home channel" in out
        assert "Daily report ready." in out

    def test_unknown_kind_returns_content_unchanged(self):
        assert format_fallback_notice("body", "mystery") == "body"

    def test_empty_content_returns_bare_notice(self):
        out = format_fallback_notice("", "parent")
        assert out.startswith("⚠️")
        assert "\n\n" not in out
