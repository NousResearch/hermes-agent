"""Tests for serialized JSON byte hard-limit and table pagination in Feishu Card renderer.

Task 6:
- Card-level byte hard-limit (28000 bytes serialized JSON)
- Table row pagination (max 10 rows per sub-table)
- Degradation to markdown when pagination insufficient
"""

import json

import pytest

from gateway.platforms.feishu_card_renderer import (
    _DEFAULT_MAX_CARD_BYTES,
    FeishuCardRenderingError,
    _check_serialized_card_size,
    _paginate_table_element,
    build_feishu_card_v2_payload,
    build_feishu_card_v2_payload_from_document,
    build_feishu_card_v2_payloads,
    build_feishu_card_v2_payloads_from_document,
    render_document_to_feishu_card_v2,
    render_document_to_feishu_card_v2_parts,
)
from gateway.rendering.document import (
    ImageBlock,
    MessageDocument,
    ParagraphBlock,
    TableBlock,
    HeadingBlock,
)
from plugins.platforms.feishu.adapter import (
    _serialized_sdk_request_body_size,
    _validate_interactive_request_body_size,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_table_element(rows: int, cols: int = 3) -> dict:
    """Build a table element dict directly (bypassing the document pipeline)."""
    columns = [
        {"name": f"col_{i}", "display_name": f"H{i}", "data_type": "markdown"}
        for i in range(cols)
    ]
    row_data = []
    for r in range(rows):
        row = {f"col_{i}": f"cell_{r}_{i}" for i in range(cols)}
        row_data.append(row)
    return {
        "tag": "table",
        "page_size": min(rows, 10),
        "row_height": "auto",
        "header_style": {"text_align": "left", "text_size": "normal", "bold": True},
        "columns": columns,
        "rows": row_data,
    }


def _card_bytes(card_dict: dict) -> int:
    """Compute serialized JSON byte size (same method the implementation should use)."""
    return len(json.dumps(card_dict, ensure_ascii=False).encode("utf-8"))


def _all_elements(cards: list[dict]) -> list[dict]:
    return [element for card in cards for element in card["body"]["elements"]]


def _sdk_body_bytes(body) -> int:
    import lark_oapi as lark

    serialized = lark.JSON.marshal(body)
    assert isinstance(serialized, str)
    return len(serialized.encode("utf-8"))


# ---------------------------------------------------------------------------
# 6a: Serialized byte size hard-limit
# ---------------------------------------------------------------------------

class TestCardByteHardLimit:
    """Tests for the serialized JSON byte hard-limit."""

    def test_card_under_limit_passes_through(self):
        """Small card under byte limit is not modified."""
        card = {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {"title": {"tag": "plain_text", "content": "Test"}},
            "body": {"elements": [{"tag": "markdown", "content": "hello"}]},
        }
        size = _check_serialized_card_size(card)
        assert size == _card_bytes(card)
        assert size < 28000

    def test_default_card_limit_is_exactly_28_kib(self):
        assert _DEFAULT_MAX_CARD_BYTES == 28 * 1024

    def test_serialized_size_uses_utf8_bytes(self):
        """Size check uses len(json.dumps(card, ensure_ascii=False).encode('utf-8'))."""
        card = {
            "tag": "markdown",
            "content": "中文测试" * 100,  # 4 chars * 100 = 400 chars = 1200 bytes
        }
        size = _check_serialized_card_size(card)
        expected = len(json.dumps(card, ensure_ascii=False).encode("utf-8"))
        assert size == expected

    def test_card_with_huge_markdown_is_split(self):
        """Single markdown element exceeding byte limit is split into multiple cards."""
        # Create a markdown element so large it exceeds 28000 bytes when serialized
        huge_content = "x" * 30000
        doc = MessageDocument([ParagraphBlock(text=huge_content)])
        cards = render_document_to_feishu_card_v2_parts(doc)
        for card in cards:
            size = _card_bytes(card)
            assert size <= _DEFAULT_MAX_CARD_BYTES, (
                f"Card {size} bytes exceeds limit {_DEFAULT_MAX_CARD_BYTES}"
            )

    def test_card_with_many_elements_is_regrouped(self):
        """Multiple elements exceeding byte limit are regrouped into smaller cards."""
        # Each element is ~2000 bytes; 20 of them = ~40000 bytes (exceeds 28000)
        elements = []
        for i in range(20):
            elements.append({"tag": "markdown", "content": f"x" * 2000, "text_size": "normal"})
        # Build a card with all elements
        big_card = {
            "schema": "2.0",
            "config": {"update_multi": True},
            "header": {"title": {"tag": "plain_text", "content": "Test"}},
            "body": {"direction": "vertical", "elements": elements},
        }
        assert _card_bytes(big_card) > _DEFAULT_MAX_CARD_BYTES

        # When rendered via the pipeline, each card should fit
        doc = MessageDocument([ParagraphBlock(text="x" * 2000) for _ in range(20)])
        cards = render_document_to_feishu_card_v2_parts(doc)
        for card in cards:
            size = _card_bytes(card)
            assert size <= _DEFAULT_MAX_CARD_BYTES, (
                f"Card {size} bytes exceeds limit {_DEFAULT_MAX_CARD_BYTES}"
            )


# ---------------------------------------------------------------------------
# 6b: Table pagination
# ---------------------------------------------------------------------------

class TestTablePagination:
    """Tests for table row pagination."""

    def test_card_with_large_table_is_paginated(self):
        """Table with 25 rows is split into 3 sub-tables (10+10+5)."""
        table_el = _make_table_element(rows=25, cols=3)
        parts = _paginate_table_element(table_el, max_rows=10)
        assert len(parts) == 3
        assert len(parts[0]["rows"]) == 10
        assert len(parts[1]["rows"]) == 10
        assert len(parts[2]["rows"]) == 5

    def test_full_pipeline_keeps_25_rows_as_native_tables_10_10_5(self):
        rows = [[f"row-{index}", f"value-{index}"] for index in range(25)]
        doc = MessageDocument([TableBlock(headers=["name", "value"], rows=rows)])

        cards = render_document_to_feishu_card_v2_parts(doc)
        tables = [element for element in _all_elements(cards) if element["tag"] == "table"]

        assert [len(table["rows"]) for table in tables] == [10, 10, 5]
        assert [table["page_size"] for table in tables] == [10, 10, 5]
        assert all(table["columns"][0]["display_name"] == "name" for table in tables)
        assert [row["col_0"] for table in tables for row in table["rows"]] == [
            f"row-{index}" for index in range(25)
        ]

    def test_table_pagination_preserves_header(self):
        """Each paginated sub-table includes the original header columns."""
        table_el = _make_table_element(rows=25, cols=4)
        parts = _paginate_table_element(table_el, max_rows=10)
        for part in parts:
            assert "columns" in part
            assert len(part["columns"]) == 4
            # Each column should have the same display_name as original
            for i, col in enumerate(part["columns"]):
                assert col["display_name"] == f"H{i}"

    def test_table_page_size_does_not_exceed_10(self):
        """Each paginated sub-table's page_size should not exceed 10."""
        table_el = _make_table_element(rows=35, cols=2)
        parts = _paginate_table_element(table_el, max_rows=10)
        for part in parts:
            assert part["page_size"] <= 10
            assert part["page_size"] == len(part["rows"])

    def test_small_table_not_paginated(self):
        """A table with <= 10 rows should return a single part."""
        table_el = _make_table_element(rows=5, cols=3)
        parts = _paginate_table_element(table_el, max_rows=10)
        assert len(parts) == 1
        assert len(parts[0]["rows"]) == 5

    def test_paginated_tables_never_exceed_five_per_card(self):
        """Row pagination must not bypass the five-table-per-card guard."""
        doc = MessageDocument(
            [
                TableBlock(
                    headers=["H"],
                    rows=[[str(row)] for row in range(20)],
                )
                for _ in range(3)
            ]
        )

        cards = render_document_to_feishu_card_v2_parts(doc)

        assert len(cards) == 2
        for card in cards:
            tables = [
                element
                for element in card["body"]["elements"]
                if element.get("tag") == "table"
            ]
            assert len(tables) <= 5

    def test_six_source_tables_are_partitioned_across_cards_without_degrading(self):
        doc = MessageDocument(
            [
                TableBlock(headers=["H"], rows=[[f"table-{index}"]])
                for index in range(6)
            ]
        )

        cards = render_document_to_feishu_card_v2_parts(doc)

        assert len(cards) == 2
        assert [
            len([element for element in card["body"]["elements"] if element["tag"] == "table"])
            for card in cards
        ] == [5, 1]
        assert all(element["tag"] == "table" for element in _all_elements(cards))
        assert [
            row["col_0"]
            for table in _all_elements(cards)
            for row in table["rows"]
        ] == [f"table-{index}" for index in range(6)]

    def test_table_degradation_to_markdown_when_still_oversized(self):
        """If a single-row table still exceeds byte limit, degrade to markdown.

        This tests the fallback path: when even a paginated single-row table
        exceeds the byte limit, the card pipeline should degrade it to a
        markdown element.
        """
        # Build a table element with huge cell content that even one row
        # exceeds the byte limit when serialized inside a card
        huge_cell = "x" * 28000
        columns = [
            {"name": "col_0", "display_name": "H0", "data_type": "markdown"},
            {"name": "col_1", "display_name": "H1", "data_type": "markdown"},
        ]
        rows = [{f"col_0": huge_cell, "col_1": huge_cell}]
        table_el = {
            "tag": "table",
            "page_size": 1,
            "row_height": "auto",
            "header_style": {"text_align": "left"},
            "columns": columns,
            "rows": rows,
        }
        # Even paginated to 1 row, this table element alone exceeds the limit
        parts = _paginate_table_element(table_el, max_rows=10)
        single_table_json = json.dumps(parts[0], ensure_ascii=False).encode("utf-8")
        # The table element itself is huge
        assert len(single_table_json) > _DEFAULT_MAX_CARD_BYTES

        # When passed through the full pipeline via a document, it should be
        # degraded (table_policy="markdown" or byte-limit fallback)
        doc = MessageDocument([
            TableBlock(
                headers=["H0", "H1"],
                rows=[[huge_cell, huge_cell]],
                raw_markdown="| H0 | H1 |\n|---|---|\n| huge | huge |",
            )
        ])
        cards = render_document_to_feishu_card_v2_parts(doc, table_policy="table")
        elements = _all_elements(cards)
        assert elements
        assert all(element["tag"] == "markdown" for element in elements)
        combined = "".join(element["content"] for element in elements)
        assert combined.count("x") >= len(huge_cell) * 2
        for card in cards:
            size = _card_bytes(card)
            assert size <= _DEFAULT_MAX_CARD_BYTES, (
                f"Degraded card {size} bytes still exceeds limit"
            )


# ---------------------------------------------------------------------------
# Integration: full pipeline byte check
# ---------------------------------------------------------------------------

class TestFullPipelineByteLimit:
    """Integration tests: all rendered cards must be within byte limit."""

    def test_huge_text_produces_compliant_cards(self):
        """Rendering very large text produces only byte-compliant cards."""
        text = "A" * 50000
        payloads = build_feishu_card_v2_payloads(text)
        for payload in payloads:
            assert len(payload.encode("utf-8")) <= _DEFAULT_MAX_CARD_BYTES, (
                f"Payload {len(payload.encode('utf-8'))} exceeds {_DEFAULT_MAX_CARD_BYTES}"
            )

    def test_huge_table_produces_compliant_cards(self):
        """Rendering a document with a large table produces byte-compliant cards."""
        # Create a table with many rows and wide columns
        headers = [f"列{i}" for i in range(5)]
        rows = [[f"数据_{r}_{c}" for c in range(5)] for r in range(30)]
        doc = MessageDocument([
            HeadingBlock(level=2, text="大表格测试"),
            TableBlock(headers=headers, rows=rows),
        ])
        payloads = build_feishu_card_v2_payloads_from_document(doc)
        for payload in payloads:
            assert len(payload.encode("utf-8")) <= _DEFAULT_MAX_CARD_BYTES, (
                f"Payload {len(payload.encode('utf-8'))} exceeds {_DEFAULT_MAX_CARD_BYTES}"
            )

    def test_outer_create_request_stays_under_30kb(self):
        """Real lark SDK request bodies stay below Feishu's 30 KiB limit."""
        lark_im = pytest.importorskip("lark_oapi.api.im.v1")
        create_body_type = lark_im.CreateMessageRequestBody
        escape_dense_cell = '"\\中文' * 30
        documents = {
            "large_chinese": MessageDocument(
                [ParagraphBlock(text="大段中文内容。" * 6000)]
            ),
            "escape_dense": MessageDocument(
                [ParagraphBlock(text='引号"与反斜杠\\' * 6000)]
            ),
            "large_table": MessageDocument(
                [
                    TableBlock(
                        headers=[f"列{i}" for i in range(8)],
                        rows=[
                            [escape_dense_cell for _ in range(8)]
                            for _ in range(20)
                        ],
                    )
                ]
            ),
        }

        for case, doc in documents.items():
            cards = render_document_to_feishu_card_v2_parts(doc)
            assert cards, case
            for card in cards:
                content = json.dumps(card, ensure_ascii=False)
                assert len(content.encode("utf-8")) <= 28 * 1024, case
                body = (
                    create_body_type.builder()
                    .receive_id("oc_真实接收者_1234567890")
                    .msg_type("interactive")
                    .content(content)
                    .uuid("u" * 50)
                    .build()
                )
                outer_size = len(
                    json.dumps(body.__dict__, ensure_ascii=False).encode("utf-8")
                )
                assert outer_size < 30 * 1024, (
                    f"{case} SDK request body is {outer_size} bytes"
                )

    def test_real_sdk_create_and_reply_bodies_stay_under_30_kib(self):
        lark_im = pytest.importorskip("lark_oapi.api.im.v1")
        doc = MessageDocument([ParagraphBlock(text=('"\\中文' * 10000))])

        for card in render_document_to_feishu_card_v2_parts(doc):
            content = json.dumps(card, ensure_ascii=False)
            create = (
                lark_im.CreateMessageRequestBody.builder()
                .receive_id("oc_真实接收者_1234567890")
                .msg_type("interactive")
                .content(content)
                .uuid("u" * 50)
                .build()
            )
            reply = (
                lark_im.ReplyMessageRequestBody.builder()
                .content(content)
                .msg_type("interactive")
                .reply_in_thread(True)
                .uuid("u" * 50)
                .build()
            )
            assert _sdk_body_bytes(create) < 30 * 1024
            assert _sdk_body_bytes(reply) < 30 * 1024
            assert _serialized_sdk_request_body_size(create) == _sdk_body_bytes(create)
            assert _serialized_sdk_request_body_size(reply) == _sdk_body_bytes(reply)
            _validate_interactive_request_body_size(create)
            _validate_interactive_request_body_size(reply)

    def test_actual_sdk_body_guard_rejects_oversize_before_transport(self):
        lark_im = pytest.importorskip("lark_oapi.api.im.v1")
        body = (
            lark_im.ReplyMessageRequestBody.builder()
            .content("x" * (30 * 1024))
            .msg_type("interactive")
            .reply_in_thread(True)
            .uuid("u" * 50)
            .build()
        )

        with pytest.raises(ValueError, match="30 KiB hard limit"):
            _validate_interactive_request_body_size(body)

    def test_huge_title_is_moved_losslessly_into_size_compliant_body(self):
        title = "标题" * 12000

        cards = render_document_to_feishu_card_v2_parts(MessageDocument([]), title=title)

        assert cards
        assert all(_card_bytes(card) <= _DEFAULT_MAX_CARD_BYTES for card in cards)
        combined = "".join(
            element["content"]
            for element in _all_elements(cards)
            if element["tag"] == "markdown"
        )
        assert combined == title

    def test_huge_image_alt_is_preserved_outside_image_element(self):
        alt = "图片说明" * 9000
        doc = MessageDocument([ImageBlock(source="/tmp/a.png", alt=alt)])

        cards = render_document_to_feishu_card_v2_parts(
            doc,
            image_key_by_source={"/tmp/a.png": "img_key"},
        )

        assert cards
        assert all(_card_bytes(card) <= _DEFAULT_MAX_CARD_BYTES for card in cards)
        elements = _all_elements(cards)
        assert any(element["tag"] == "img" for element in elements)
        assert "".join(
            element["content"] for element in elements if element["tag"] == "markdown"
        ) == alt

    def test_numbered_final_cards_are_rechecked_against_both_limits(self):
        title = "T" * 200
        doc = MessageDocument([ParagraphBlock(text=('"\\中文' * 12000))])

        cards = render_document_to_feishu_card_v2_parts(doc, title=title)

        assert len(cards) > 1
        assert all(_card_bytes(card) <= _DEFAULT_MAX_CARD_BYTES for card in cards)
        assert all(
            card["header"]["title"]["content"].endswith(f"/{len(cards)}")
            for card in cards
        )

    def test_1024_byte_title_moves_to_body_before_multicard_numbering(self):
        title = "title:" + ("T" * (1024 - len("title:")))
        doc = MessageDocument([ParagraphBlock(text="正文" * 12000)])

        cards = render_document_to_feishu_card_v2_parts(doc, title=title)

        assert len(cards) > 1
        assert all(
            len(card["header"]["title"]["content"].encode("utf-8")) <= 1024
            for card in cards
        )
        assert all(
            card["header"]["title"]["content"].startswith("Hermes ")
            for card in cards
        )
        body_markdown = "".join(
            element["content"]
            for element in _all_elements(cards)
            if element.get("tag") == "markdown"
        )
        assert body_markdown.count(title) == 1

    def test_custom_table_limits_are_honored_with_hard_limit_clamping(self):
        table = TableBlock(
            headers=["A", "B"],
            rows=[[str(index), "value"] for index in range(7)],
        )
        cards = render_document_to_feishu_card_v2_parts(
            MessageDocument([table, table, table]),
            max_rows=3,
            max_tables=2,
        )
        table_counts = [
            sum(element.get("tag") == "table" for element in card["body"]["elements"])
            for card in cards
        ]
        assert all(count <= 2 for count in table_counts)
        assert [
            len(element["rows"])
            for element in _all_elements(cards)
            if element.get("tag") == "table"
        ] == [3, 3, 1] * 3

        too_wide = TableBlock(
            headers=[f"c{index}" for index in range(51)],
            rows=[["v"] * 51],
        )
        wide_cards = render_document_to_feishu_card_v2_parts(
            MessageDocument([too_wide]),
            max_columns=100,
        )
        assert all(
            element.get("tag") != "table" for element in _all_elements(wide_cards)
        )

    @pytest.mark.parametrize("name", ["max_tables", "max_columns", "max_rows"])
    def test_invalid_table_limits_are_rejected(self, name):
        kwargs = {name: 0}
        with pytest.raises(FeishuCardRenderingError, match=name):
            render_document_to_feishu_card_v2_parts(
                MessageDocument([ParagraphBlock(text="x")]),
                **kwargs,
            )

    @pytest.mark.parametrize("name", ["max_elements_per_card", "max_card_chars"])
    def test_invalid_partition_limits_are_rejected(self, name):
        with pytest.raises(FeishuCardRenderingError, match=name):
            render_document_to_feishu_card_v2_parts(
                MessageDocument([ParagraphBlock(text="x")]),
                **{name: 0},
            )

    def test_element_limit_override_is_clamped_to_feishu_hard_limit(self):
        doc = MessageDocument([ParagraphBlock(text="x") for _ in range(201)])

        cards = render_document_to_feishu_card_v2_parts(
            doc,
            max_elements_per_card=1000,
            max_card_chars=100000,
        )

        assert [len(card["body"]["elements"]) for card in cards] == [200, 1]

    def test_singular_public_builders_reject_content_that_requires_multiple_cards(self):
        text = "x" * 50000
        doc = MessageDocument([ParagraphBlock(text=text)])

        with pytest.raises(FeishuCardRenderingError, match="requires multiple cards"):
            build_feishu_card_v2_payload(text)
        with pytest.raises(FeishuCardRenderingError, match="requires multiple cards"):
            build_feishu_card_v2_payload_from_document(doc)
        with pytest.raises(FeishuCardRenderingError, match="requires multiple cards"):
            render_document_to_feishu_card_v2(doc)
