"""Display-field contracts from Telegram Bot API RichText/RichBlock schemas.

Reference: https://core.telegram.org/bots/api (schema snapshot retained with the review).
No rendering of callback payloads or fetching media is implied.
"""

from types import SimpleNamespace

import pytest

from plugins.platforms.telegram.telegram_rich_inbound import project_rich_message


def project(*blocks):
    return project_rich_message({"blocks": list(blocks)})


@pytest.mark.parametrize("kind", ["date_time", "subscript", "superscript", "marked", "email_address",
                                 "phone_number", "bank_card_number", "hashtag", "cashtag",
                                 "anchor_link", "reference", "reference_link"])
def test_documented_inline_display_fields_survive(kind):
    assert project({"type": "paragraph", "text": {"type": kind, "text": "displayed"}}) == "displayed"


def test_invisible_table_cell_does_not_invalidate_other_content():
    assert project({"type": "table", "cells": [[{"text": "left"}, {}, {"text": "right"}]]}) == "| left |  | right |"


@pytest.mark.parametrize("cell", [SimpleNamespace(text=None), {"text": None}], ids=["typed", "mapping"])
def test_null_table_cell_text_does_not_invalidate_neighbors(cell):
    assert project({"type": "table", "cells": [[{"text": "left"}, cell, {"text": "right"}]]}) == "| left |  | right |"


def test_expandable_quote_preserves_text_and_multiline_credit():
    result = project({"type": "expandable_blockquote", "text": "quoted", "credit": "author\ncredit"})
    assert result == "[Quote]\nquoted\nauthor\ncredit\n[/Quote]"


@pytest.mark.parametrize("kind", ["collage", "slideshow"])
def test_grouped_media_keeps_children_captions(kind):
    result = project({"type": kind, "caption": {"text": "group"}, "blocks": [
        {"type": "photo", "caption": {"text": "child", "credit": "photographer"}}]})
    assert "group" in result and "child" in result and "photographer" in result


def test_table_caption_is_richtext_not_media_caption():
    assert project({"type": "table", "cells": [], "caption": {
        "type": "url", "text": "Source", "url": "https://example.com"}}) == "Source (https://example.com)"
