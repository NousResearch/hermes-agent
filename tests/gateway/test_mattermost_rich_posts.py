"""Tests for the pure Mattermost rich-post renderer."""

from plugins.platforms.mattermost.rich_posts import (
    build_attachment_action,
    render_rich_post,
)


def test_empty_markdown_has_no_rich_post():
    assert render_rich_post("") is None


def test_oversized_or_non_string_markdown_has_no_rich_post():
    assert render_rich_post("x" * 16_384) is None
    assert render_rich_post({"text": "not Markdown"}) is None


def test_markdown_is_preserved_in_fallback_and_card_text():
    markdown = "- first\n- second\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n---\n\n```py\nprint('ok')\n```"

    assert render_rich_post(markdown) == {
        "props": {
            "attachments": [
                {
                    "fallback": markdown,
                    "text": markdown,
                }
            ]
        }
    }


def test_first_atx_heading_is_promoted_without_changing_remaining_body():
    markdown = "# Release Notes\n\nIntro\n\n## Details\n- one\n- two"

    payload = render_rich_post(markdown)
    assert payload is not None
    attachment = payload["props"]["attachments"][0]

    assert attachment["fallback"] == markdown
    assert attachment["title"] == "Release Notes"
    assert attachment["text"] == "\nIntro\n\n## Details\n- one\n- two"


def test_heading_inside_code_fence_is_not_promoted():
    markdown = "```md\n# Example only\n```\n\n## Actual Title\nBody"

    payload = render_rich_post(markdown)
    assert payload is not None
    attachment = payload["props"]["attachments"][0]

    assert attachment["fallback"] == markdown
    assert attachment["title"] == "Actual Title"
    assert attachment["text"] == "```md\n# Example only\n```\n\nBody"


def test_build_attachment_action_uses_mattermost_button_shape():
    assert build_attachment_action(
        "approve",
        "Approve",
        "https://bot.example/actions/approve",
        style="success",
    ) == {
        "id": "approve",
        "name": "Approve",
        "type": "button",
        "style": "success",
        "integration": {"url": "https://bot.example/actions/approve"},
    }


def test_attachment_action_sanitizes_unsafe_style_to_default():
    action = build_attachment_action("go", "Go", "https://example.com/go", style="neon")

    assert action is not None
    assert action["style"] == "default"


def test_attachment_action_omits_missing_required_fields():
    malformed = (
        ("", "Go", "https://example.com/go"),
        ("go", " ", "https://example.com/go"),
        ("go", "Go", None),
        ({"not": "an id"}, "Go", "https://example.com/go"),
    )

    assert [build_attachment_action(*values) for values in malformed] == [None] * 4


def test_rich_post_includes_optional_color_and_actions():
    action = build_attachment_action("open", "Open", "https://example.com/open")
    assert action is not None

    payload = render_rich_post("Status: ready", color="#22aa66", actions=[action])

    assert payload == {
        "props": {
            "attachments": [
                {
                    "fallback": "Status: ready",
                    "text": "Status: ready",
                    "color": "#22aa66",
                    "actions": [action],
                }
            ]
        }
    }


def test_rich_post_omits_invalid_actions_and_keeps_supported_ones():
    valid = build_attachment_action("open", "Open", "https://example.com/open")
    assert valid is not None

    payload = render_rich_post(
        "Ready",
        actions=[None, "button", {}, {"id": "missing-fields"}, valid],
    )

    assert payload is not None
    assert payload["props"]["attachments"][0]["actions"] == [valid]


def test_attachment_action_preserves_json_object_context():
    context = {"nonce": "abc", "choice": "helpful"}
    action = build_attachment_action(
        "feedback",
        "Helpful",
        "https://example.com/actions",
        context=context,
    )

    assert action is not None
    assert action["integration"]["context"] == context
    payload = render_rich_post("Done", actions=[action])
    assert payload is not None
    rendered = payload["props"]["attachments"][0]["actions"][0]
    assert rendered["integration"]["context"] == context
