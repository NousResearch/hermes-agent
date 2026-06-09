"""Tests for Telegram model picker label shortening.

Verifies that `_build_model_keyboard` produces button labels that survive
Telegram's 2-column inline-keyboard rendering: the longest hyphen-aligned
common prefix shared by the page is stripped, and any leftover that exceeds
the cap is truncated from the *front* with a leading ellipsis so the unique
suffix is preserved.
"""

import sys
from unittest.mock import MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class TestShortenModelLabels:
    def test_strips_publisher_prefix(self):
        labels = TelegramAdapter._shorten_model_labels(
            ["google/gemini-2.5-pro", "google/gemini-2.5-flash"]
        )
        # publisher prefix gone; common "gemini-2.5-" stripped
        assert labels == ["pro", "flash"]

    def test_vertex_preview_set_distinguishable(self):
        """The original bug: 4 preview models all rendered as 'gemini-...-preview'."""
        labels = TelegramAdapter._shorten_model_labels(
            [
                "gemini-3.1-pro-preview",
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "gemini-3.1-flash-lite-preview",
            ]
        )
        # No two labels should be identical, and each must be obviously different.
        assert len(set(labels)) == len(labels)
        assert labels == [
            "3.1-pro-preview",
            "3-pro-preview",
            "3-flash-preview",
            "3.1-flash-lite-preview",
        ]

    def test_no_common_prefix_leaves_names_alone(self):
        labels = TelegramAdapter._shorten_model_labels(
            ["claude-sonnet-4-6", "gpt-5.5", "qwen3-coder"]
        )
        assert labels == ["claude-sonnet-4-6", "gpt-5.5", "qwen3-coder"]

    def test_single_model_keeps_full_name(self):
        # commonprefix of one string is the string itself; ensure we don't strip it all.
        labels = TelegramAdapter._shorten_model_labels(["gemini-2.5-pro"])
        assert labels == ["gemini-2.5-pro"]

    def test_long_label_truncates_from_front_preserving_suffix(self):
        # Two unrelated long names: no common prefix to strip, must truncate.
        long_name = "a-really-long-model-id-with-no-shared-prefix-preview"
        labels = TelegramAdapter._shorten_model_labels([long_name, "claude-haiku-4-5"])
        # The truncated label must end in "-preview" (the distinguishing part)
        # and start with the ellipsis.
        assert labels[0].startswith("…")
        assert labels[0].endswith("-preview")
        assert len(labels[0]) <= TelegramAdapter._MODEL_LABEL_MAX
        assert labels[1] == "claude-haiku-4-5"

    def test_common_prefix_alignment_at_hyphen_only(self):
        # Without hyphen alignment, "gemini-2.5-pro" and "gemini-2.5-pre-x" would
        # share "gemini-2.5-pr" — we don't want to strip mid-token.
        labels = TelegramAdapter._shorten_model_labels(
            ["gemini-2.5-pro", "gemini-2.5-preview"]
        )
        # Both share "gemini-2.5-" (hyphen-aligned) — that's the right cut.
        assert labels == ["pro", "preview"]

    def test_empty_label_falls_back_to_full_name(self):
        # If a model id is exactly the common prefix, the stripped label would be
        # empty — we must fall back to the original short name so the button
        # isn't blank.
        labels = TelegramAdapter._shorten_model_labels(
            ["gemini-2.5-flash-", "gemini-2.5-flash-lite", "gemini-2.5-flash-001"]
        )
        # Common prefix is "gemini-2.5-flash-".
        # First model strips to empty -> falls back to short name.
        assert labels[0] == "gemini-2.5-flash-"
        assert labels[1] == "lite"
        assert labels[2] == "001"

    def test_label_cap_is_respected(self):
        labels = TelegramAdapter._shorten_model_labels(
            ["x" * 50, "y" * 50]
        )
        for label in labels:
            assert len(label) <= TelegramAdapter._MODEL_LABEL_MAX
