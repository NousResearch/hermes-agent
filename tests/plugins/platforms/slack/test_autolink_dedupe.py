"""Flat Slack autolinks may mirror ordinary text without losing real destinations."""

import pytest

from plugins.platforms.slack.adapter import SlackAdapter


def _blocks(*elements):
    return [{"type": "rich_text", "elements": list(elements)}]


def _section(*elements):
    return {"type": "rich_text_section", "elements": list(elements)}


@pytest.mark.parametrize(
    "links,display",
    [
        ("<http://example.com|example.com>", "example.com"),
        ("<https://example.com|example.com>", "example.com"),
        ("<http://example.com/|example.com>", "example.com"),
        ("<https://example.com/|example.com/>", "example.com/"),
        ("<https://example.com/path?q=1|example.com/path?q=1>", "example.com/path?q=1"),
        ("<https://example.com/path?a=1&amp;b=2|example.com/path?a=1&amp;b=2>",
         "example.com/path?a=1&b=2"),
        ("<http://example.com|example.com> and <http://example.com|example.com>",
         "example.com and example.com"),
    ],
)
@pytest.mark.parametrize("style", [{}, None])
def test_plain_autolink_mirror_is_not_appended(links, display, style):
    original = f"<@U_BOT> check {links} now"
    blocks = _blocks(_section(
        {"type": "user", "user_id": "U_BOT"},
        {"type": "text", "text": f" check {display} now", "style": style},
    ))
    live = SlackAdapter._append_block_text(original, blocks, "U_BOT")
    hydrated = SlackAdapter._render_message_text({"text": original, "blocks": blocks}, "U_BOT")
    assert (live, hydrated) == (original, f"check {links} now")


@pytest.mark.parametrize(
    "flat,element,extra",
    [
        ("<https://good.example|good.example>",
         _section({"type": "link", "url": "https://evil.example", "text": "good.example"}),
         "good.example (https://evil.example)"),
        ("<http://example.com|example.com>",
         _section({"type": "link", "url": "https://example.com", "text": "example.com"}),
         "example.com (https://example.com)"),
        ("<https://example.com|documentation>",
         _section({"type": "link", "url": "https://other.example", "text": "documentation"}),
         "documentation (https://other.example)"),
        ("<https://example.com|example.com>",
         {"type": "rich_text_quote", "elements": [_section(
             {"type": "text", "text": "example.com"})]}, "> example.com"),
        ("<https://example.com|example.com>",
         {"type": "rich_text_list", "style": "bullet", "elements": [_section(
             {"type": "text", "text": "example.com"})]}, "• example.com"),
        ("<https://example.com|example.com>",
         {"type": "rich_text_preformatted", "elements": [
             {"type": "text", "text": "example.com"}]}, "```\nexample.com\n```"),
        ("check <https://example.com|example.com> now",
         _section({"type": "text", "text": "check example.com now", "style": {"code": True}}),
         "check example.com now"),
        ("check `<https://example.com|example.com>` now",
         _section({"type": "text", "text": "check example.com now"}), "check example.com now"),
        ("check ```\n<https://example.com|example.com>\n``` now",
         _section({"type": "text", "text": "check example.com now"}), "check example.com now"),
        ("AT&amp;amp;T", _section({"type": "text", "text": "AT&T"}), "AT&T"),
        ("check <https://example.com|example.com> now",
         _section({"type": "text", "text": "check example.com now", "style": "unknown"}),
         "check example.com now"),
        ("Summary", _section({"type": "text", "text": "New context"}), "New context"),
        pytest.param("Summary", _section(
            {"type": "text", "text": "New context"}, {"type": "text", "text": None}),
            "New context", id="null-leaf-keeps-valid-prose"),
        pytest.param("Summary", _section(
            {"type": "text", "text": "New context"}, {"type": "text", "text": 7}),
            "New context", id="integer-leaf-keeps-valid-prose"),
        pytest.param("Summary", _section(
            {"type": "text", "text": "New context"}, {"type": "text", "text": True}),
            "New context", id="boolean-leaf-keeps-valid-prose"),
        pytest.param("check <https://example.com|example.com> now", _section(
            {"type": "text", "text": "check example.com now"},
            {"type": "text", "text": ["ignored"]}),
            "check example.com now", id="list-leaf-cannot-qualify-as-mirror"),
        pytest.param("check <https://example.com|example.com> now", _section(
            {"type": "text", "text": "check example.com now"},
            {"type": "text", "text": {"text": "ignored"}}),
            "check example.com now", id="dict-leaf-cannot-qualify-as-mirror"),
        ("check <https://example.com|example.com> now and read",
         _section({"type": "text", "text": "check example.com now"}), "check example.com now"),
        ("check <https://example.com|documentation> now",
         _section({"type": "text", "text": "check documentation now"}), "check documentation now"),
        ("check <https://example.com/path/|example.com/path> now",
         _section({"type": "text", "text": "check example.com/path now"}), "check example.com/path now"),
    ],
)
def test_autolink_comparison_keeps_additional_context(flat, element, extra):
    blocks = _blocks(element)
    assert SlackAdapter._append_block_text(flat, blocks, "") == f"{flat}\n{extra}"
    assert SlackAdapter._render_message_text({"text": flat, "blocks": blocks}) == f"{flat}\n{extra}"
