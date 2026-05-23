"""Markdown table → Feishu Card JSON 2.0 outbound conversion tests.

Validates the new dispatch in ``_build_outbound_payload``: when the LLM
emits a markdown table, the adapter must send a native ``tag: "table"``
interactive card (Lark V7.4+) instead of the prior plain-text fallback
that lost all alignment.

Also doubles as a self-check script — when run directly via
``python tests/gateway/platforms/test_feishu_card_table.py`` it prints
the full card JSON produced from a real-world Scout / Feishu transcript
fragment so a human can eyeball the structure before shipping.
"""
from __future__ import annotations

import json

from gateway.platforms._feishu_card_table import (
    _build_card_with_table_payload,
    _parse_markdown_table_to_card_element,
    _split_content_at_tables,
    _split_md_table_row,
)


# ---------------------------------------------------------------------------
# Helper-level unit tests
# ---------------------------------------------------------------------------


def test_split_md_table_row_strips_pipes_and_whitespace():
    row = "| office desk | 2961 | **123,664** | ❌ |"
    assert _split_md_table_row(row) == ["office desk", "2961", "**123,664**", "❌"]


def test_split_md_table_row_keeps_empty_cells():
    row = "| a |   | c |"
    assert _split_md_table_row(row) == ["a", "", "c"]


def test_split_content_at_tables_pure_text_yields_single_text_segment():
    content = "Hello world\nLine two"
    segs = _split_content_at_tables(content)
    assert len(segs) == 1
    assert segs[0][0] == "text"
    assert "Hello world" in segs[0][1]


def test_split_content_at_tables_isolates_table_block():
    content = (
        "Intro paragraph before the table.\n"
        "\n"
        "| col1 | col2 |\n"
        "|------|------|\n"
        "| a    | b    |\n"
        "| c    | d    |\n"
        "\n"
        "Footer line after the table.\n"
    )
    segs = _split_content_at_tables(content)
    kinds = [k for k, _ in segs]
    assert kinds == ["text", "table", "text"]
    assert "Intro paragraph" in segs[0][1]
    assert segs[1][1].count("\n") >= 3  # header + sep + 2 data rows
    assert "Footer line" in segs[2][1]


def test_parse_markdown_table_to_card_element_builds_columns_and_rows():
    table = (
        "| 关键词 | SERP 数 |\n"
        "|---|---|\n"
        "| office desk | 2961 |\n"
        "| office chair | 201 |\n"
    )
    elem = _parse_markdown_table_to_card_element(table)
    assert elem["tag"] == "table"
    assert elem["columns"] == [
        {"name": "col1", "display_name": "关键词", "data_type": "markdown"},
        {"name": "col2", "display_name": "SERP 数", "data_type": "markdown"},
    ]
    assert elem["rows"] == [
        {"col1": "office desk", "col2": "2961"},
        {"col1": "office chair", "col2": "201"},
    ]
    assert elem["page_size"] == 10


def test_parse_markdown_table_handles_ragged_rows_by_padding():
    table = (
        "| a | b | c |\n"
        "|---|---|---|\n"
        "| 1 | 2 |\n"  # missing third cell
    )
    elem = _parse_markdown_table_to_card_element(table)
    assert elem["rows"] == [{"col1": "1", "col2": "2", "col3": ""}]


def test_parse_markdown_table_degrades_when_no_data_rows():
    table = "| only header |\n|------|\n"
    elem = _parse_markdown_table_to_card_element(table)
    assert elem["tag"] == "table"
    assert elem["rows"] == []


# ---------------------------------------------------------------------------
# Top-level payload build tests
# ---------------------------------------------------------------------------


def test_build_card_with_table_payload_emits_schema_2_envelope():
    content = "| h1 | h2 |\n|---|---|\n| a | b |\n"
    raw = _build_card_with_table_payload(content)
    card = json.loads(raw)
    assert card["schema"] == "2.0"
    assert card["config"] == {"wide_screen_mode": True}
    assert isinstance(card["body"]["elements"], list)
    assert card["body"]["elements"][0]["tag"] == "table"


def test_build_card_with_table_payload_mixes_markdown_and_table_in_order():
    content = (
        "**主要发现:** 办公家具大词都 >1万\n"
        "\n"
        "| 关键词 | SERP | 状态 |\n"
        "|---|---|---|\n"
        "| desk | 41284 | ❌ |\n"
        "| chair | 980 | ✅ |\n"
        "\n"
        "Next: 用 ✅ 词跑下一阶段\n"
    )
    raw = _build_card_with_table_payload(content)
    card = json.loads(raw)
    elements = card["body"]["elements"]
    tags = [e["tag"] for e in elements]
    # Order: intro / table / hint / raw-source / footer
    assert tags == ["markdown", "table", "markdown", "markdown", "markdown"]
    assert "主要发现" in elements[0]["content"]
    assert elements[1]["rows"][1] == {"col1": "chair", "col2": "980", "col3": "✅"}
    # Hint and raw-source disclosure live at indices 2 and 3.
    assert "长按 cell 复制单元格" in elements[2]["content"]
    assert elements[3]["content"].startswith("```markdown")
    assert "Next" in elements[4]["content"]


def test_build_card_with_table_payload_pure_table_no_prose():
    content = "| h1 | h2 |\n|---|---|\n| a | b |\n"
    raw = _build_card_with_table_payload(content)
    card = json.loads(raw)
    tags = [e["tag"] for e in card["body"]["elements"]]
    # A bare table still emits the hint and raw-source disclosure after it.
    assert tags == ["table", "markdown", "markdown"]


# ---------------------------------------------------------------------------
# Real-world regression: the 2026-05-21 Scout / Feishu office-furniture
# incident transcript fragment. This must produce a card whose table rows
# preserve the ❌ / ✅ status markers and the SERP integers as cell strings.
# ---------------------------------------------------------------------------


_SCOUT_FEISHU_FRAGMENT = (
    "结果出来了,先做个初步筛选:\n"
    "\n"
    "| 关键词 | AMZ123排名 | 亚马逊竞品数 | 竞品<2000? |\n"
    "|---|---|---|---|\n"
    "| office desk | 2961 | **123,664** | ❌ |\n"
    "| office chair | 201 | **62,794** | ❌ |\n"
    "| ergonomic office chair | 3546 | **16,873** | ❌ |\n"
    "| filing cabinet | 9120 | **18,600** | ❌ |\n"
    "| office desk accessories | 3043 | **111,356** | ❌ |\n"
    "| ✅ **criss cross office chair** | **7256** | **1,410** | ✅ |\n"
    "\n"
    "**主要发现:** 办公家具类目主词竞品数都 >1 万,符合 <2000 的就一个\n"
)


def test_scout_feishu_office_furniture_fragment_round_trip():
    raw = _build_card_with_table_payload(_SCOUT_FEISHU_FRAGMENT)
    card = json.loads(raw)
    elements = card["body"]["elements"]
    tags = [e["tag"] for e in elements]
    # Order: intro / table / hint / raw-source / footer
    assert tags == ["markdown", "table", "markdown", "markdown", "markdown"]
    table = elements[1]
    assert len(table["columns"]) == 4
    assert table["columns"][0]["display_name"] == "关键词"
    assert len(table["rows"]) == 6
    # The last row must preserve ✅ marker and bold markdown:
    last = table["rows"][5]
    assert "criss cross office chair" in last["col1"]
    assert last["col4"] == "✅"
    # Hint educates users; raw-source preserves the full markdown table.
    assert "长按 cell 复制单元格" in elements[2]["content"]
    assert "criss cross office chair" in elements[3]["content"]


# ---------------------------------------------------------------------------
# Self-check entry point — `python test_feishu_card_table.py`
# Prints the full card JSON produced from the real Scout transcript so a
# human can eyeball the structure (columns, rows, ✅/❌, markdown spans)
# before merging the PR.
# ---------------------------------------------------------------------------


def _self_check() -> None:
    raw = _build_card_with_table_payload(_SCOUT_FEISHU_FRAGMENT)
    card = json.loads(raw)
    print("=" * 72)
    print("SELF-CHECK: Scout / Feishu 2026-05-21 office-furniture transcript")
    print("=" * 72)
    print("Input markdown content:")
    print("-" * 72)
    print(_SCOUT_FEISHU_FRAGMENT)
    print("-" * 72)
    print("Output card JSON (msg_type=interactive payload):")
    print("-" * 72)
    print(json.dumps(card, ensure_ascii=False, indent=2))
    print("-" * 72)
    elements = card["body"]["elements"]
    print(f"Element count: {len(elements)}")
    for i, e in enumerate(elements):
        if e["tag"] == "table":
            print(
                f"  [{i}] table: {len(e['columns'])} cols, {len(e['rows'])} rows, "
                f"page_size={e['page_size']}"
            )
        else:
            content_preview = e.get("content", "")[:60].replace("\n", " ")
            print(f"  [{i}] {e['tag']}: {content_preview!r}")
    print("=" * 72)
    print("OK")


if __name__ == "__main__":
    _self_check()
