"""Pure Telegram rich-content projection: readable text, fail-visible, bounded work."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from plugins.platforms.telegram.telegram_rich_inbound import (
    has_rich_message,
    project_rich_message,
    rich_message_text,
)


UNSUPPORTED = "[Unsupported Telegram rich content]"
INVALID = "[Invalid Telegram rich content]"
EMPTY = "[Empty Telegram rich message]"
TRUNCATED = "[Telegram rich message truncated]"


def paragraph(text):
    return {"type": "paragraph", "text": text}


def project(*blocks):
    return project_rich_message({"blocks": list(blocks)})


class ReadTrap:
    """Unknown branches and branches after a limit must never be inspected."""

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected field read: {name}")

    def __str__(self):
        raise AssertionError("Never stringify arbitrary rich-content objects")


def test_absent_rich_content_is_the_only_none_result():
    for message in (None, SimpleNamespace(), SimpleNamespace(rich_message=None),
                    SimpleNamespace(api_kwargs={}), {"rich_message": None}):
        assert not has_rich_message(message)
        assert rich_message_text(message) is None


@pytest.mark.parametrize("payload", [None, {}, [], "", False, 0])
def test_explicit_raw_key_is_present_even_if_null_or_malformed(payload):
    message = SimpleNamespace(rich_message=None, api_kwargs={"rich_message": payload})
    assert has_rich_message(message)
    assert rich_message_text(message) == INVALID


@pytest.mark.parametrize("payload", [{}, [], "", False, 0])
def test_non_null_typed_field_is_present_even_if_falsey(payload):
    message = SimpleNamespace(rich_message=payload, api_kwargs=ReadTrap())
    assert has_rich_message(message)
    assert rich_message_text(message) == INVALID


def test_typed_field_has_priority_and_presence_check_does_not_walk_payload():
    payload = SimpleNamespace(blocks=(SimpleNamespace(type="paragraph", text="typed"),))
    message = SimpleNamespace(rich_message=payload, api_kwargs=ReadTrap())
    assert has_rich_message(message)
    assert rich_message_text(message) == "typed"
    assert has_rich_message(SimpleNamespace(rich_message=ReadTrap()))
    assert has_rich_message(SimpleNamespace(api_kwargs={"rich_message": ReadTrap()}))


def test_dynamic_proxy_cannot_invent_a_rich_carrier_on_plain_media():
    message = MagicMock(text=None, caption=None)
    assert not has_rich_message(message)
    assert rich_message_text(message) is None


def test_raw_fallback_and_mapping_message_are_supported():
    message = {"api_kwargs": {"rich_message": {"blocks": [paragraph("raw")]}}}
    assert has_rich_message(message)
    assert rich_message_text(message) == "raw"


@pytest.mark.parametrize("kind", ["paragraph", "heading", "footer"])
def test_text_blocks_preserve_display_text(kind):
    assert project({"type": kind, "text": "  displayed\ntext  ", "level": 3}) == "  displayed\ntext  "


@pytest.mark.parametrize("kind", [
    "bold", "italic", "underline", "strikethrough", "spoiler", "code", "url",
    "mention", "text_mention", "bot_command",
])
def test_inline_objects_use_only_display_text(kind):
    token = {"type": kind, "text": "shown", "url": "hidden-url", "command": "/hidden",
             "user": ReadTrap(), "file_id": "secret-id"}
    assert project(paragraph(["before ", token, " after"])) == "before shown after"


def test_nested_inline_lists_custom_emoji_and_typed_objects_without_to_dict():
    class TelegramLike(SimpleNamespace):
        def to_dict(self):
            raise AssertionError("No whole-object conversion")

    token = TelegramLike(type="bold", text=("nested ", TelegramLike(
        type="custom_emoji", alternative_text="🙂", custom_emoji_id="secret")))
    payload = TelegramLike(blocks=(TelegramLike(type="paragraph", text=(token, "!")),))
    assert project_rich_message(payload) == "nested 🙂!"


def test_structured_blocks_keep_order_and_boundaries():
    payload = {"blocks": [
        {"type": "heading", "text": "Title"},
        {"type": "blockquote", "blocks": [paragraph("Quote"), paragraph("More quote")]},
        {"type": "pullquote", "text": [{"type": "italic", "text": "Pull"}], "credit": "Author"},
        {"type": "details", "summary": "Summary", "blocks": [paragraph("Details")]},
        {"type": "list", "items": [{"blocks": [paragraph("First")]},
                                    {"blocks": [paragraph("Second"), paragraph("Continued")]}]},
        {"type": "pre", "text": "a = 1\n  b = 2"},
        {"type": "footer", "text": "Footer"},
    ]}
    before = deepcopy(payload)
    assert project_rich_message(payload) == (
        "Title\n[Quote]\nQuote\nMore quote\n[/Quote]\n[Quote]\nPull\nAuthor\n[/Quote]\n"
        "[Details]\nSummary\nDetails\n[/Details]\nFirst\nSecond\nContinued\n"
        "[Code]\na = 1\n  b = 2\n[/Code]\nFooter"
    )
    assert payload == before


def test_rich_urls_are_kept_without_opening_or_executing_them():
    assert project(paragraph({"type": "url", "text": "site", "url": "https://example.org"})) == "site (https://example.org)"
    assert project(paragraph({"type": "url", "text": "site", "url": "javascript:secret"})) == "site"
    assert project(paragraph({"type": "url", "text": "site", "url": "https://" + "x" * 40000})).endswith(TRUNCATED)


def test_code_boundaries_preserve_literal_content():
    assert project({"type": "pre", "text": "/stop\n  literal"}) == "[Code]\n/stop\n  literal\n[/Code]"


def test_optional_credit_caption_and_empty_structured_blocks():
    assert project({"type": "pullquote", "text": "Quote"}) == "[Quote]\nQuote\n[/Quote]"
    assert project({"type": "blockquote", "blocks": []}, paragraph("Kept")) == "[Quote]\n[/Quote]\nKept"
    assert project({"type": "details", "summary": "Summary", "blocks": []}) == "[Details]\nSummary\n[/Details]"
    assert project({"type": "list", "items": []}) == EMPTY


def test_table_escapes_boundaries_preserves_empty_cells_and_span_markers():
    table = {"type": "table", "caption": [{"type": "bold", "text": "Caption"}], "cells": [
        [{"text": "a|b\nc\rd\\e", "colspan": 2}, {"text": "", "rowspan": 3},
         {"text": [{"type": "code", "text": "last|cell"}]}],
        [{"text": ""}, {"text": "middle"}, {"text": ""}],
    ]}
    assert project(table) == (
        "Caption\n| a\\|b\\nc\\rd\\\\e [colspan=2] |  [rowspan=3] | last\\|cell |\n"
        "|  | middle |  |"
    )


def test_typed_table_rows_and_cells():
    table = SimpleNamespace(type="table", caption=None, cells=(
        (SimpleNamespace(text="left", colspan=1, rowspan=None), SimpleNamespace(text="")),
        (SimpleNamespace(text="right", rowspan=2),),
    ))
    assert project(table) == "| left |  |\n| right [rowspan=2] |"


@pytest.mark.parametrize("kind", ["photo", "video", "document", "future_block"])
def test_unknown_media_never_inspects_or_leaks_payload(kind):
    class Media:
        type = kind
        blocks = ReadTrap()
        text = ReadTrap()
        file_id = "DO_NOT_LEAK"

        def to_dict(self):
            raise AssertionError("No media serialization")

    assert project(Media()) == UNSUPPORTED
    assert project(paragraph({"type": kind, "text": ReadTrap(), "file_id": "secret"})) == UNSUPPORTED


@pytest.mark.parametrize("payload", [
    None, "secret", [], 9, {"blocks": None}, {"blocks": "secret"},
    {"blocks": [None]}, {"blocks": [{}]}, {"blocks": [{"type": []}]},
    {"blocks": [paragraph(None)]}, {"blocks": [paragraph(42)]},
    {"blocks": [{"type": "paragraph"}]},
    {"blocks": [paragraph({"text": "secret"})]},
    {"blocks": [paragraph({"type": "custom_emoji", "emoji": "secret"})]},
    {"blocks": [{"type": "blockquote", "blocks": {}}]},
    {"blocks": [{"type": "details", "summary": 3, "blocks": []}]},
    {"blocks": [{"type": "list", "items": "secret"}]},
    {"blocks": [{"type": "list", "items": [None]}]},
    {"blocks": [{"type": "table", "cells": None}]},
    {"blocks": [{"type": "table", "cells": [None]}]},
    {"blocks": [{"type": "table", "cells": [[None]]}]},
])
def test_malformed_content_is_fixed_and_visible(payload):
    assert project_rich_message(payload) == INVALID


@pytest.mark.parametrize("span", [0, -1, True, "2", [], {}])
def test_malformed_spans_are_visible_without_raw_value_dump(span):
    assert INVALID in project({"type": "table", "cells": [[{"text": "x", "colspan": span}]]})


def test_malformed_content_fails_visible_without_returning_a_partial_message():
    assert project(paragraph("before"), {"type": "photo", "file_id": "secret"},
                   paragraph(None), paragraph("after")) == INVALID


def test_unsupported_blocks_preserve_supported_siblings():
    assert project(paragraph("before"), {"type": "photo", "file_id": "secret"},
                   paragraph("after")) == f"before\n{UNSUPPORTED}\nafter"


@pytest.mark.parametrize("blocks", [[], [paragraph("")], [paragraph(" \n\t ")],
                                   [{"type": "table", "cells": []}]])
def test_empty_content_has_fixed_fallback(blocks):
    assert project_rich_message({"blocks": blocks}) == EMPTY


def test_depth_limit_is_inclusive_and_aborts_remaining_branches():
    text = "deep"
    # Root=0, blocks=1, paragraph=2, inline=3; lists each add one edge.
    for _ in range(16 - 3):
        text = [text]
    assert project(paragraph(text)) == "deep"
    assert project(paragraph([text]), ReadTrap()) == TRUNCATED


@pytest.mark.parametrize("typed", [False, True])
def test_cycles_are_bounded_and_fail_visible(typed):
    block: Any
    if typed:
        block = SimpleNamespace(type="blockquote")
        block.blocks = [block]
    else:
        block = {"type": "blockquote"}
        block["blocks"] = [block]
    assert project(block, ReadTrap()).endswith(TRUNCATED)
    inline = []
    inline.append(inline)
    assert project(paragraph(inline), ReadTrap()) == TRUNCATED


def test_shared_node_limit_is_inclusive_across_inline_and_block_walks():
    # Root + blocks + paragraph + inline list consume four of the 2048 nodes.
    tokens = ["x"] * (2048 - 4)
    assert project(paragraph(tokens)) == "x" * len(tokens)
    assert project(paragraph(tokens + [ReadTrap()]), ReadTrap()) == "x" * len(tokens) + "\n" + TRUNCATED
    # Exhausting nodes in one block also stops the following block.
    assert project(paragraph(tokens), ReadTrap()).endswith(TRUNCATED)


def test_block_limit_is_inclusive_and_shared_across_nested_blocks():
    blocks = [paragraph("x")] * 256
    assert project(*blocks) == "\n".join(["x"] * 256)
    result = project(*blocks, ReadTrap())
    assert result.endswith(TRUNCATED)
    assert result.count("x") == 256
    nested = [{"type": "blockquote", "blocks": [paragraph("x")]}] * (256 // 2)
    assert project(*nested).count("[Quote]") == 128
    assert project(*nested, ReadTrap()).endswith(TRUNCATED)


def test_cell_limit_is_inclusive_and_shared_across_tables():
    table = {"type": "table", "cells": [[{"text": "x"}] * (512 // 2)]}
    assert project(table, table).count("x") == 512
    overflow = {"type": "table", "cells": [[ReadTrap()]]}
    result = project(table, table, overflow, ReadTrap())
    assert result.endswith(TRUNCATED)
    assert result.count("x") == 512


def test_empty_table_rows_cannot_bypass_shared_node_budget():
    result = project({"type": "table", "cells": [[]] * 5000}, ReadTrap())
    assert result.endswith(TRUNCATED)
    assert len(result) <= 32768


@pytest.mark.parametrize("size", [32767, 32768, 32769, 1000000])
def test_output_limit_includes_the_final_truncation_marker(size):
    result = project(paragraph("x" * size))
    if size <= 32768:
        assert result == "x" * size
    else:
        assert len(result) == 32768
        assert result.endswith("\n" + TRUNCATED)


def test_character_limit_stops_before_any_remaining_branch():
    assert project(paragraph("x" * 32769), ReadTrap()).endswith(TRUNCATED)
    result = project(paragraph("x" * 32768), ReadTrap())
    assert len(result) == 32768
    assert result.endswith(TRUNCATED)


def test_large_text_is_sliced_before_escaping_or_other_string_operations():
    class BoundedString(str):
        def __getitem__(self, key):
            assert isinstance(key, slice)
            assert key.stop <= 32769
            return super().__getitem__(key)

        def replace(self, *args, **kwargs):
            raise AssertionError("Do not replace on the whole input")

        def encode(self, *args, **kwargs):
            raise AssertionError("Do not encode the whole input")

    result = project({"type": "table", "cells": [[{"text": BoundedString("|\n" * 500000)}]]})
    assert len(result) == 32768
    assert result.endswith(TRUNCATED)


def test_huge_span_integer_is_bounded_without_decimal_conversion():
    result = project({"type": "table", "cells": [[{"text": "x", "colspan": 1 << 1000000}]]})
    assert "[colspan]" in result
    assert len(result) < 100


@pytest.mark.parametrize("kind,field", [
    ("pullquote", "text"), ("details", "summary"), ("table", "caption"),
])
def test_full_output_never_reads_a_remaining_child_field(kind, field):
    class LimitedBlock(ReadTrap):
        pass

    block = LimitedBlock()
    block.type = kind
    setattr(block, field, "x" * 32768)
    result = project(block)
    assert len(result) == 32768
    assert result.endswith(TRUNCATED)


def test_full_cell_output_never_reads_span_fields():
    cell = ReadTrap()
    cell.text = "x" * (32768 - len("| "))
    result = project({"type": "table", "cells": [[cell]]})
    assert len(result) == 32768
    assert result.endswith(TRUNCATED)


def test_unknown_long_type_never_hashes_or_walks_its_payload():
    class LongType(str):
        def __hash__(self):
            raise AssertionError("Do not hash an unbounded discriminator")

    token = {"type": LongType("unknown" * 100000), "text": ReadTrap(), "blocks": ReadTrap()}
    assert project(token) == UNSUPPORTED
    assert project(paragraph(token)) == UNSUPPORTED
