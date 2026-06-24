"""
Tests for EmojiAnnotationHandler
=================================

Coverage targets
----------------
* Default emoji map is applied correctly for every built-in category.
* Unknown / missing category leaves the message text unchanged.
* ``handle()`` mutates the mapping in-place **and** returns it.
* ``handle()`` raises ``KeyError`` when ``"text"`` is absent.
* Custom emoji map replaces the default map entirely.
* Custom separator is honoured.
* ``annotate()`` convenience helper works on plain strings.
* ``emoji_map`` property returns a *copy* (mutations don't affect handler).
"""

from __future__ import annotations

import pytest

from hermes_agent.handlers.emoji_annotation_handler import (
    DEFAULT_EMOJI_MAP,
    EmojiAnnotationHandler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def handler() -> EmojiAnnotationHandler:
    """A default-configured handler instance."""
    return EmojiAnnotationHandler()


# ---------------------------------------------------------------------------
# Default emoji map — one test per built-in category
# ---------------------------------------------------------------------------


class TestDefaultEmojiMap:
    @pytest.mark.parametrize(
        "category, expected_emoji",
        list(DEFAULT_EMOJI_MAP.items()),
    )
    def test_known_category_prepends_emoji(
        self,
        handler: EmojiAnnotationHandler,
        category: str,
        expected_emoji: str,
    ) -> None:
        msg: dict = {"text": "hello", "category": category}
        result = handler.handle(msg)
        assert result["text"].startswith(expected_emoji), (
            f"Expected text to start with {expected_emoji!r} for category {category!r}, "
            f"got {result['text']!r}"
        )

    @pytest.mark.parametrize(
        "category, expected_emoji",
        list(DEFAULT_EMOJI_MAP.items()),
    )
    def test_known_category_includes_original_text(
        self,
        handler: EmojiAnnotationHandler,
        category: str,
        expected_emoji: str,
    ) -> None:
        original = "original text"
        msg: dict = {"text": original, "category": category}
        handler.handle(msg)
        assert original in msg["text"]

    def test_default_separator_is_space(self, handler: EmojiAnnotationHandler) -> None:
        msg: dict = {"text": "hi", "category": "positive"}
        handler.handle(msg)
        # "😊 hi"
        assert msg["text"] == f"😊 hi"


# ---------------------------------------------------------------------------
# Unknown / missing category
# ---------------------------------------------------------------------------


class TestUnknownCategory:
    def test_unknown_category_leaves_text_unchanged(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "unchanged", "category": "nonexistent"}
        handler.handle(msg)
        assert msg["text"] == "unchanged"

    def test_missing_category_key_leaves_text_unchanged(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "unchanged"}
        handler.handle(msg)
        assert msg["text"] == "unchanged"

    def test_empty_string_category_leaves_text_unchanged(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "unchanged", "category": ""}
        handler.handle(msg)
        assert msg["text"] == "unchanged"

    def test_whitespace_only_category_leaves_text_unchanged(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "unchanged", "category": "   "}
        handler.handle(msg)
        assert msg["text"] == "unchanged"


# ---------------------------------------------------------------------------
# handle() contract
# ---------------------------------------------------------------------------


class TestHandleContract:
    def test_handle_returns_same_mapping_object(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "hi", "category": "info"}
        returned = handler.handle(msg)
        assert returned is msg

    def test_handle_mutates_in_place(self, handler: EmojiAnnotationHandler) -> None:
        msg: dict = {"text": "hi", "category": "success"}
        handler.handle(msg)
        assert msg["text"].startswith("✅")

    def test_handle_raises_key_error_when_text_missing(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        with pytest.raises(KeyError, match="text"):
            handler.handle({"category": "info"})  # type: ignore[arg-type]

    def test_handle_preserves_extra_keys(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "hi", "category": "error", "sender": "alice"}
        handler.handle(msg)
        assert msg["sender"] == "alice"

    def test_category_matching_is_case_insensitive(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "hi", "category": "POSITIVE"}
        handler.handle(msg)
        assert msg["text"].startswith("😊")

    def test_category_matching_strips_whitespace(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        msg: dict = {"text": "hi", "category": "  warning  "}
        handler.handle(msg)
        assert msg["text"].startswith("⚠️")


# ---------------------------------------------------------------------------
# Custom emoji map
# ---------------------------------------------------------------------------


class TestCustomEmojiMap:
    def test_custom_map_replaces_default(self) -> None:
        custom = {"happy": "🎉"}
        h = EmojiAnnotationHandler(emoji_map=custom)
        msg: dict = {"text": "yay", "category": "happy"}
        h.handle(msg)
        assert msg["text"] == "🎉 yay"

    def test_custom_map_does_not_include_defaults(self) -> None:
        custom = {"happy": "🎉"}
        h = EmojiAnnotationHandler(emoji_map=custom)
        msg: dict = {"text": "hi", "category": "positive"}
        h.handle(msg)
        # "positive" is not in the custom map → text unchanged
        assert msg["text"] == "hi"

    def test_empty_custom_map_never_annotates(self) -> None:
        h = EmojiAnnotationHandler(emoji_map={})
        for category in DEFAULT_EMOJI_MAP:
            msg: dict = {"text": "hi", "category": category}
            h.handle(msg)
            assert msg["text"] == "hi", f"Expected no annotation for {category!r}"


# ---------------------------------------------------------------------------
# Custom separator
# ---------------------------------------------------------------------------


class TestCustomSeparator:
    def test_custom_separator_is_used(self) -> None:
        h = EmojiAnnotationHandler(separator=" | ")
        msg: dict = {"text": "hello", "category": "info"}
        h.handle(msg)
        assert msg["text"] == "ℹ️ | hello"

    def test_empty_separator(self) -> None:
        h = EmojiAnnotationHandler(separator="")
        msg: dict = {"text": "world", "category": "success"}
        h.handle(msg)
        assert msg["text"] == "✅world"


# ---------------------------------------------------------------------------
# annotate() helper
# ---------------------------------------------------------------------------


class TestAnnotateHelper:
    def test_annotate_returns_annotated_string(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        result = handler.annotate("test", "question")
        assert result == "❓ test"

    def test_annotate_unknown_category_returns_original(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        result = handler.annotate("test", "unknown")
        assert result == "test"

    def test_annotate_does_not_mutate_handler_state(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        before = handler.emoji_map
        handler.annotate("x", "positive")
        assert handler.emoji_map == before


# ---------------------------------------------------------------------------
# emoji_map property
# ---------------------------------------------------------------------------


class TestEmojiMapProperty:
    def test_emoji_map_returns_copy(self, handler: EmojiAnnotationHandler) -> None:
        copy = handler.emoji_map
        copy["positive"] = "💥"
        # Original handler map should be unaffected
        assert handler.emoji_map["positive"] == "😊"

    def test_emoji_map_contains_all_defaults(
        self, handler: EmojiAnnotationHandler
    ) -> None:
        for category in DEFAULT_EMOJI_MAP:
            assert category in handler.emoji_map
