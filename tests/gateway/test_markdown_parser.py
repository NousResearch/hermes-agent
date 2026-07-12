"""Tests for the gateway Markdown-to-document parser."""

from gateway.rendering.document import DividerBlock, TableBlock
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
