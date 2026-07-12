"""Tests for the gateway Markdown-to-document parser."""

import pytest

from gateway.rendering.document import DividerBlock, ParagraphBlock, TableBlock
from gateway.rendering.markdown_parser import parse_markdown_document


def test_pipe_table_separator_row_is_parsed_as_table_not_divider():
    doc = parse_markdown_document(
        "| 项目 | 结果 |\n"
        "| --- | --- |\n"
        "| smoke | pass |\n"
    )

    assert len(doc.blocks) == 1
    table = doc.blocks[0]
    assert isinstance(table, TableBlock)
    assert table.headers == ["项目", "结果"]
    assert table.rows == [["smoke", "pass"]]
    assert not any(isinstance(block, DividerBlock) for block in doc.blocks)


def test_escaped_pipe_stays_inside_cell():
    """\\| 不应被当作列分隔符"""
    doc = parse_markdown_document(
        "| A | B |\n| --- | --- |\n| one \\| two | ok |"
    )
    table = doc.blocks[0]
    assert table.rows[0] == ["one \\| two", "ok"], f"got {table.rows[0]}"


def test_inline_code_pipe_does_not_split_cell():
    """inline code 内的 | 不应被当作列分隔符"""
    doc = parse_markdown_document(
        "| A | B |\n| --- | --- |\n| `foo|bar` | ok |"
    )
    table = doc.blocks[0]
    assert table.rows[0] == ["`foo|bar`", "ok"], f"got {table.rows[0]}"


def test_unbalanced_backticks_degrade_to_raw_markdown():
    """未闭合的 backtick 导致不确定时，保持原始 markdown 不分割"""
    doc = parse_markdown_document(
        "| A | B |\n| --- | --- |\n| `foo | bar |"
    )
    # 如果无法确定，要么保留原始文本要么降级为段落
    # 至少不错误地分成 3 个单元格
    assert len(doc.blocks) >= 1
    if hasattr(doc.blocks[0], 'rows'):
        assert len(doc.blocks[0].rows[0]) <= 2  # 最多 2 列


def test_normal_table_parsing_unchanged():
    """普通表格不应受影响"""
    doc = parse_markdown_document(
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
    )
    table = doc.blocks[0]
    assert table.rows[0] == ["1", "2"]


def test_escaped_pipe_at_line_start():
    doc = parse_markdown_document(
        "| A |\n| --- |\n| \\|content |"
    )
    table = doc.blocks[0]
    assert table.rows[0] == ["\\|content"]


def test_double_backtick_inline_code_pipe_stays_inside_cell():
    doc = parse_markdown_document(
        "| A | B |\n| --- | --- |\n| ``foo|bar`` | ok |"
    )

    assert len(doc.blocks) == 1
    table = doc.blocks[0]
    assert isinstance(table, TableBlock)
    assert table.rows == [["``foo|bar``", "ok"]]


def test_matching_backtick_run_length_controls_inline_span():
    doc = parse_markdown_document(
        "| A | B |\n"
        "| --- | --- |\n"
        "| ``foo`|bar`` | ok |\n"
        "| ```one``|two``` | kept |\n"
        "| ````left```|right```` | preserved |"
    )

    assert len(doc.blocks) == 1
    table = doc.blocks[0]
    assert isinstance(table, TableBlock)
    assert table.rows == [
        ["``foo`|bar``", "ok"],
        ["```one``|two```", "kept"],
        ["````left```|right````", "preserved"],
    ]


def test_later_uncertain_row_degrades_entire_table_without_loss():
    markdown = (
        "| A | B |\n"
        "| --- | --- |\n"
        "| first | ok |\n"
        "| `uncertain | still here |\n"
        "| after | survives |"
    )

    doc = parse_markdown_document(markdown)

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert isinstance(block, ParagraphBlock)
    assert block.text == markdown
    assert "| first | ok |" in block.text
    assert "| `uncertain | still here |" in block.text
    assert "| after | survives |" in block.text


@pytest.mark.parametrize(
    "mismatched_row",
    [
        "| too | many | SECRET |",
        "| too-few |",
    ],
    ids=["wider-than-header", "narrower-than-header"],
)
def test_row_width_mismatch_degrades_entire_table_without_loss(mismatched_row):
    markdown = (
        "| A | B |\n"
        "| --- | --- |\n"
        "| before | survives |\n"
        f"{mismatched_row}\n"
        "| after | also-survives |"
    )

    doc = parse_markdown_document(markdown)

    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert isinstance(block, ParagraphBlock)
    assert block.text == markdown
    assert mismatched_row in block.text
