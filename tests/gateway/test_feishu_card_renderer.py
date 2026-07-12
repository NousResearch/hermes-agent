import json

import pytest

from gateway.rendering.document import (
    CodeBlock,
    DividerBlock,
    HeadingBlock,
    ImageBlock,
    ListBlock,
    MessageDocument,
    ParagraphBlock,
    TableBlock,
)
from gateway.platforms.feishu_card_renderer import (
    FeishuCardRenderingError,
    render_document_to_feishu_card_v2,
)


def test_card_v2_contains_schema_body_and_summary():
    card = render_document_to_feishu_card_v2(
        MessageDocument([ParagraphBlock("Hello **Feishu**")]),
        title="Hermes",
    )

    assert card["schema"] == "2.0"
    assert card["config"]["width_mode"] == "fill"
    assert card["config"]["summary"]["content"] == "Hello Feishu"
    assert card["header"]["title"]["content"] == "Hermes"
    assert card["body"]["elements"] == [
        {"tag": "markdown", "content": "Hello **Feishu**", "text_size": "normal"}
    ]


def test_table_block_renders_native_table_component():
    card = render_document_to_feishu_card_v2(
        MessageDocument([
            HeadingBlock(level=2, text="状态"),
            TableBlock(
                headers=["名称", "状态"],
                rows=[["A", "**完成**"], ["B", "[详情](https://example.com)"]],
                raw_markdown="| 名称 | 状态 |\n|---|---|\n| A | **完成** |\n| B | [详情](https://example.com) |",
            ),
        ])
    )

    elements = card["body"]["elements"]
    assert elements[0] == {"tag": "markdown", "content": "状态", "text_size": "heading"}
    assert elements[1]["tag"] == "table"
    assert elements[1]["row_height"] == "auto"
    assert elements[1]["columns"] == [
        {"name": "col_0", "display_name": "名称", "data_type": "markdown"},
        {"name": "col_1", "display_name": "状态", "data_type": "markdown"},
    ]
    assert elements[1]["rows"] == [
        {"col_0": "A", "col_1": "**完成**"},
        {"col_0": "B", "col_1": "[详情](https://example.com)"},
    ]


def test_code_block_renders_markdown_fence():
    card = render_document_to_feishu_card_v2(
        MessageDocument([CodeBlock(language="python", code="print('hi')")])
    )

    assert card["body"]["elements"] == [
        {"tag": "markdown", "content": "```python\nprint('hi')\n```", "text_size": "normal"}
    ]


def test_list_and_divider_blocks_render_losslessly():
    card = render_document_to_feishu_card_v2(
        MessageDocument(
            [
                ListBlock(ordered=False, items=["alpha", "beta"]),
                DividerBlock(),
                ListBlock(ordered=True, items=["first", "second"]),
            ]
        )
    )

    assert card["body"]["elements"] == [
        {"tag": "markdown", "content": "- alpha\n- beta", "text_size": "normal"},
        {"tag": "hr"},
        {"tag": "markdown", "content": "1. first\n2. second", "text_size": "normal"},
    ]


def test_unknown_document_block_fails_instead_of_silently_dropping_content():
    with pytest.raises(FeishuCardRenderingError, match="unsupported message document block"):
        render_document_to_feishu_card_v2(MessageDocument([object()]))  # type: ignore[list-item]


def test_too_wide_table_falls_back_to_markdown_code_block():
    raw = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
    card = render_document_to_feishu_card_v2(
        MessageDocument([TableBlock(headers=["A", "B", "C"], rows=[["1", "2", "3"]], raw_markdown=raw)]),
        max_columns=2,
    )

    assert card["body"]["elements"] == [
        {"tag": "markdown", "content": f"```markdown\n{raw}\n```", "text_size": "normal"}
    ]


def test_mismatched_direct_table_rows_degrade_losslessly_to_markdown():
    table = TableBlock(
        headers=["A", "B"],
        rows=[
            ["kept", "also-kept", "EXTRA_SENTINEL"],
            ["short-row"],
        ],
    )

    card = render_document_to_feishu_card_v2(MessageDocument([table]))

    assert card["body"]["elements"] == [
        {
            "tag": "markdown",
            "content": (
                "```markdown\n"
                "| A | B |\n"
                "| --- | --- |\n"
                "| kept | also-kept | EXTRA_SENTINEL |\n"
                "| short-row |\n"
                "```"
            ),
            "text_size": "normal",
        }
    ]


def test_card_payload_is_json_serializable():
    card = render_document_to_feishu_card_v2(MessageDocument([ParagraphBlock("Hello")]))

    assert json.loads(json.dumps(card, ensure_ascii=False))["schema"] == "2.0"


def test_image_block_renders_card_v2_img_element():
    card = render_document_to_feishu_card_v2(
        MessageDocument([
            ParagraphBlock("图片前"),
            ImageBlock(source="/tmp/a.png", alt="测试图"),
            ParagraphBlock("图片后"),
        ]),
        image_key_by_source={"/tmp/a.png": "img_test"},
    )

    elements = card["body"]["elements"]
    assert elements[0] == {"tag": "markdown", "content": "图片前", "text_size": "normal"}
    assert elements[1] == {
        "tag": "img",
        "img_key": "img_test",
        "alt": {"tag": "plain_text", "content": "测试图"},
    }
    assert elements[2] == {"tag": "markdown", "content": "图片后", "text_size": "normal"}
