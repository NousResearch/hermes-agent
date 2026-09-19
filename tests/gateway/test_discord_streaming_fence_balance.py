"""Streaming previews must be fence-balanced (Discord).

A streaming edit renders a *prefix* of the reply. When the model has opened a ```
fence but not yet closed it, the preview arrives unterminated and Discord renders
everything after the opener as plain text: a table being drawn live collapses and
only snaps back on the final edit.

``truncate_message`` balances fences *between* chunks, but the streaming path takes
``[0]`` of the split (and short previews skip splitting entirely), so nothing closed
the trailing fence.
"""
import pytest

from plugins.platforms.discord.adapter import DiscordAdapter
from gateway.platforms.helpers import fence_state_after


def _balance(text):
    return DiscordAdapter._balance_stream_preview(text)


def _is_balanced(text):
    return not fence_state_after(text)[0]


class TestBalanceStreamPreview:
    def test_open_fence_is_closed(self):
        assert _balance("```\nrow one") == "```\nrow one\n```"

    def test_language_tagged_fence_is_closed(self):
        out = _balance("```python\nx = 1")
        assert out.endswith("\n```")
        assert _is_balanced(out)

    def test_already_closed_fence_untouched(self):
        text = "```\nrow\n```"
        assert _balance(text) == text

    def test_prose_without_fence_untouched(self):
        text = "just a sentence with no fence"
        assert _balance(text) == text

    def test_empty_string_untouched(self):
        assert _balance("") == ""

    def test_none_like_empty_is_safe(self):
        assert _balance("") == ""

    def test_inline_code_span_not_treated_as_fence(self):
        text = "use `foo` and `bar` here"
        assert _balance(text) == text

    def test_multiple_blocks_last_one_open(self):
        text = "```\na\n```\nprose\n```\nb"
        out = _balance(text)
        assert out == text + "\n```"
        assert _is_balanced(out)

    def test_multiple_blocks_all_closed(self):
        text = "```\na\n```\nprose\n```\nb\n```"
        assert _balance(text) == text

    @pytest.mark.parametrize("pct", [10, 25, 40, 55, 70, 85, 99])
    def test_every_stream_prefix_renders_balanced(self, pct):
        """Walk a realistic reply and assert every prefix is balanced after the fix."""
        message = (
            "**Results table**\n```\n"
            + "\n".join(f"item-{i:03d}   value {i * 7 % 100:02d}" for i in range(60))
            + "\n```\nClosing prose."
        )
        prefix = message[: max(1, len(message) * pct // 100)]
        assert _is_balanced(_balance(prefix)), f"prefix at {pct}% still unbalanced"

    def test_regression_unbalanced_before_fix(self):
        """The exact shape that broke: prefix cut inside an open fence."""
        prefix = "**Results table**\n```\nitem-001   value 07\nitem-002   val"
        assert not _is_balanced(prefix), "fixture must be unbalanced to be meaningful"
        assert _is_balanced(_balance(prefix))

    def test_closer_is_exactly_four_chars(self):
        """Cheap enough to apply on every edit without risking the 2000-char cap."""
        assert len(_balance("```\nx")) - len("```\nx") == len("\n```")
