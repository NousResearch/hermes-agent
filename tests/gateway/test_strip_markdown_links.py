"""``strip_markdown`` link handling.

The default drops a link target (SMS, IRC, Feishu, QQ all rely on that);
``keep_link_targets=True`` leaves the bare URL behind for platforms whose own
data detection is the only thing that can re-linkify it.
"""
from __future__ import annotations

from gateway.platforms.helpers import strip_markdown

_LINK = "[Open the itinerary](https://example.com/i?tfs=CBwQ&hl=en)"


def test_link_target_is_dropped_by_default() -> None:
    assert strip_markdown(f"see {_LINK} now") == "see Open the itinerary now"


def test_keep_link_targets_puts_the_url_on_its_own_line() -> None:
    assert strip_markdown(_LINK, keep_link_targets=True) == (
        "Open the itinerary\nhttps://example.com/i?tfs=CBwQ&hl=en"
    )


def test_keep_link_targets_does_not_duplicate_a_self_labelled_link() -> None:
    url = "https://example.com/x"
    assert strip_markdown(f"[{url}]({url})", keep_link_targets=True) == url


def test_keep_link_targets_ignores_non_http_targets() -> None:
    """``mailto:``/relative targets are not auto-linked, so emitting them is noise."""
    text = "[mail](mailto:a@example.com) and [rel](/docs/page)"
    assert strip_markdown(text, keep_link_targets=True) == "mail and rel"


def test_keep_link_targets_still_strips_inline_formatting() -> None:
    text = f"**Release 1.2.0**\n- `code` here\n{_LINK}"
    out = strip_markdown(text, keep_link_targets=True)
    assert "**" not in out and "`" not in out
    assert out.endswith("https://example.com/i?tfs=CBwQ&hl=en")
