"""Tests for inbound Teams card text extraction (no SDK required)."""
import json

from plugins.platforms.teams.card_text import attachments_to_text

# Synthetic card — shape mirrors a Teams "share location" adaptive card.
LOCATION_CARD = {
    "type": "AdaptiveCard",
    "version": "1.0",
    "body": [
        {
            "type": "RichTextBlock",
            "inlines": [{"type": "TextRun", "text": "1 Example Street"}],
        },
        {"type": "TextBlock", "text": "Sample District"},
        {
            "type": "Image",
            "url": "https://graph.microsoft.com/v1.0/chats/x/messages/y/hostedContents/z/$value",
            "selectAction": {
                "type": "Action.OpenUrl",
                "title": "Directions",
                "url": "https://www.bing.com/maps?rtp=adr.~pos.12.3456_78.9012",
            },
        },
    ],
}


def _att(content, content_type="application/vnd.microsoft.card.adaptive"):
    return {"content_type": content_type, "content": content}


def test_location_card_yields_text_and_coordinates():
    text = attachments_to_text([_att(json.dumps(LOCATION_CARD))])
    assert text.startswith("[card] ")
    assert "1 Example Street" in text
    assert "Sample District" in text
    assert "location: 12.3456, 78.9012" in text


def test_card_content_may_arrive_as_dict():
    assert "1 Example Street" in attachments_to_text([_att(LOCATION_CARD)])


def test_text_only_card_is_not_dropped():
    card = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "Approve this request"}]}
    assert attachments_to_text([_att(json.dumps(card))]) == "[card] Approve this request"


def test_duplicate_labels_collapse():
    card = {"body": [{"text": "Approve"}, {"text": "Approve"}]}
    assert attachments_to_text([_att(json.dumps(card))]) == "[card] Approve"


def test_non_card_attachments_are_ignored():
    assert attachments_to_text([_att(json.dumps(LOCATION_CARD), content_type="image/png")]) == ""
    assert attachments_to_text([_att("plain text", content_type="text/plain")]) == ""


def test_malformed_or_empty_payloads_return_empty_string():
    assert attachments_to_text([_att("{not json")]) == ""
    assert attachments_to_text([_att(json.dumps({"body": []}))]) == ""
    assert attachments_to_text([]) == ""
    assert attachments_to_text(None) == ""


def test_card_without_recognisable_text_returns_empty_string():
    card = {"type": "AdaptiveCard", "body": [{"type": "Image", "url": "https://example.invalid/a.png"}]}
    assert attachments_to_text([_att(json.dumps(card))]) == ""
