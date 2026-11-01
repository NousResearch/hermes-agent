"""Preserve nodeXbit/toffguy77 content contracts in the bounded projection.

Cases adapted from toffguy77's f385b2eb32252012b4c7a25b9457c5a24eddd2e8.
Rendering delimiters are not the contract: labels, states and display content are.
"""

import pytest

from plugins.platforms.telegram.telegram_rich_inbound import project_rich_message


def project(*blocks):
    return project_rich_message({"blocks": list(blocks)})


def test_list_keeps_labels_checkbox_state_own_text_and_siblings():
    result = project({"type": "list", "text": "list title", "blocks": [
        {"type": "paragraph", "text": "sibling block"}], "items": [
        {"label": "1.", "has_checkbox": True, "is_checked": True, "blocks": [
            {"type": "paragraph", "text": "done"}]},
        {"label": "2.", "has_checkbox": True, "blocks": [
            {"type": "paragraph", "text": "todo"}]}]})
    assert result.splitlines() == ["1. [x] done", "2. [ ] todo", "list title", "sibling block"]


@pytest.mark.parametrize("prefix, expected_prefix", [
    ({"label": "1."}, "1. "),
    ({"has_checkbox": True}, "[ ] "),
    ({"label": "1.", "has_checkbox": True, "is_checked": True}, "1. [x] "),
])
@pytest.mark.parametrize("empty_blocks", [
    [{"type": "paragraph", "text": ""}],
    [{"type": "paragraph", "text": []},
     {"type": "paragraph", "text": {"type": "bold", "text": ""}}],
    [{"type": "list", "items": []}],
])
@pytest.mark.parametrize("nested", [False, True])
def test_list_prefix_continues_past_empty_blocks(prefix, expected_prefix, empty_blocks, nested):
    block = {"type": "list", "items": [{**prefix, "blocks": [
        *empty_blocks,
        {"type": "paragraph", "text": "Contenido"},
        {"type": "paragraph", "text": "Siguiente"},
    ]}]}
    expected = expected_prefix + "Contenido\nSiguiente"
    if nested:
        block = {"type": "list", "items": [{"label": "A.", "blocks": [
            {"type": "paragraph", "text": "Parent"}, block,
        ]}]}
        expected = "A. Parent\n" + expected
    assert project(block) == expected


def test_empty_nested_list_item_does_not_break_label_continuation():
    assert project({"type": "list", "items": [{"label": "1.", "blocks": [
        {"type": "list", "items": [{"label": "", "blocks": []}]},
        {"type": "paragraph", "text": "Contenido"},
    ]}]}) == "1. Contenido"


@pytest.mark.parametrize("blocks", [[], [{"type": "paragraph", "text": ""}]])
def test_empty_list_item_retains_prefix_without_joining_next_item(blocks):
    assert project({"type": "list", "items": [
        {"label": "1.", "has_checkbox": True, "blocks": blocks},
        {"label": "2.", "blocks": [{"type": "paragraph", "text": "Next"}]},
    ]}) == "1. [ ] \n2. Next"


def test_nested_heading_and_link_list_keep_content_and_destinations():
    result = project(
        {"type": "heading", "size": 2, "text": ["🔥 ", {"type": "bold", "text": "Main"}]},
        {"type": "list", "items": [{"label": "1.", "blocks": [{"type": "paragraph", "text": [
            {"type": "url", "text": "Six questions", "url": "https://example.com/a1"},
            " — ", {"type": "italic", "text": "CNews.Ru"}]}]}]})
    assert "🔥 Main" in result
    assert "1. Six questions (https://example.com/a1) — CNews.Ru" in result


def test_inline_and_block_math_are_preserved():
    result = project({"type": "paragraph", "text": [
        {"type": "code", "text": "x = 1"}, " ",
        {"type": "custom_emoji", "alternative_text": "🙂"}, " ",
        {"type": "mathematical_expression", "expression": "x^2"}]},
        {"type": "mathematical_expression", "expression": "y^2"})
    assert "x = 1 🙂 $x^2$" in result
    assert "$$y^2$$" in result


@pytest.mark.parametrize("kind", ["photo", "video", "animation", "audio", "voice_note", "document", "map", "collage", "slideshow"])
def test_media_display_caption_and_credit_are_kept_without_file_metadata(kind):
    result = project({"type": kind, "file_id": "DO_NOT_LEAK", "caption": {
        "text": ["shot ", {"type": "italic", "text": "here"}], "credit": "Author"}})
    assert "shot here" in result and "Author" in result
    assert "DO_NOT_LEAK" not in result


def test_table_caption_preserves_link_and_cell_boundaries():
    result = project({"type": "table", "caption": {"type": "url", "text": "Source", "url": "https://example.com"},
                      "cells": [[{"text": "A|B\nC"}, {"text": "D"}]]})
    assert "Source (https://example.com)" in result
    assert "| A\\|B\\nC | D |" in result


def test_blockquote_credit_and_divider_survive():
    result = project({"type": "blockquote", "blocks": [{"type": "paragraph", "text": "quoted"}],
                      "credit": "author\nsecond line"}, {"type": "divider"})
    assert "quoted\nauthor\nsecond line\n[/Quote]" in result
    assert result.endswith("---")


def test_new_display_fields_share_limits_without_silent_loss():
    for block in (
        {"type": "list", "items": [{"label": "x" * 40000, "blocks": []}]},
        {"type": "photo", "caption": {"text": "x" * 40000}},
        {"type": "mathematical_expression", "expression": "x" * 40000},
    ):
        result = project(block)
        assert len(result) <= 32768
        assert result.endswith("[Telegram rich message truncated]")
