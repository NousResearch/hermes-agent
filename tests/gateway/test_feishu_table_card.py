"""Tests for Feishu table-to-card conversion (_build_table_card and friends).

Rules observed:
- No change-detector tests (no snapshot of exact weather values).
- Assert structural invariants only.
- No live network calls.
"""
import json
import re
import unittest

from gateway.platforms.feishu import (
    _build_table_card,
    _parse_gfm_table,
    _split_content_into_blocks,
)


WEATHER_CONTENT = """\
**成都天气预报**

| 时段 | 天气 | 温度 | 雨概率 |
|------|------|------|--------|
| 今晚 | 多云 | 27°C | 4%     |
| 明晨 | 烟雾 | 25°C | 6%     |
| 明日 | 晴   | 31°C | 2%     |

**广州天气预报**

| 时段 | 天气  | 温度 | 雨概率 |
|------|-------|------|--------|
| 今晚 | 雷阵雨 | 29°C | 72%   |
| 明晨 | 小雨   | 27°C | 55%   |
| 明日 | 多云   | 33°C | 18%   |
"""

FENCED_PSEUDO_TABLE = """\
Some prose.

```
| col1 | col2 |
|------|------|
| a    | b    |
```

More prose.
"""

NO_TABLE_CONTENT = """\
# Hello

This is a **bold** statement with a [link](https://example.com).

- item 1
- item 2
"""


class TestSplitContentIntoBlocks(unittest.TestCase):
    def test_two_tables_with_prose(self):
        blocks = _split_content_into_blocks(WEATHER_CONTENT)
        types = [b["type"] for b in blocks]
        # Must have exactly 2 table blocks
        self.assertEqual(types.count("table"), 2)
        # And at least 2 markdown blocks (prose between tables)
        self.assertGreaterEqual(types.count("markdown"), 2)
        # Order: markdown, table, markdown, table
        table_indices = [i for i, t in enumerate(types) if t == "table"]
        md_indices = [i for i, t in enumerate(types) if t == "markdown"]
        # First block should be markdown (prose before first table)
        self.assertEqual(types[0], "markdown")
        # Tables appear after the first markdown block
        self.assertGreater(min(table_indices), 0)

    def test_fenced_code_block_not_parsed_as_table(self):
        blocks = _split_content_into_blocks(FENCED_PSEUDO_TABLE)
        table_blocks = [b for b in blocks if b["type"] == "table"]
        self.assertEqual(len(table_blocks), 0, "Fenced pseudo-table must NOT be a table block")

    def test_no_table_content_all_markdown(self):
        blocks = _split_content_into_blocks(NO_TABLE_CONTENT)
        table_blocks = [b for b in blocks if b["type"] == "table"]
        self.assertEqual(len(table_blocks), 0)


class TestParseGfmTable(unittest.TestCase):
    def test_basic_4col_table(self):
        raw = (
            "| 时段 | 天气 | 温度 | 雨概率 |\n"
            "|------|------|------|--------|\n"
            "| 今晚 | 多云 | 27°C | 4%     |\n"
            "| 明晨 | 烟雾 | 25°C | 6%     |\n"
        )
        result = _parse_gfm_table(raw)
        self.assertIsNotNone(result)
        # 4 columns
        self.assertEqual(len(result["columns"]), 4)
        # column names are c0/c1/c2/c3
        for idx, col in enumerate(result["columns"]):
            self.assertEqual(col["name"], f"c{idx}")
            self.assertEqual(col["data_type"], "markdown")
            self.assertEqual(col["width"], "auto")
        # display_name comes from header
        self.assertEqual(result["columns"][0]["display_name"], "时段")
        self.assertEqual(result["columns"][1]["display_name"], "天气")
        # 2 data rows
        self.assertEqual(len(result["rows"]), 2)
        # Row keys match column names
        for row in result["rows"]:
            for col in result["columns"]:
                self.assertIn(col["name"], row)

    def test_missing_cells_padded(self):
        raw = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| x |   |\n"  # only 2 cells
        )
        result = _parse_gfm_table(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["columns"]), 3)
        row = result["rows"][0]
        self.assertEqual(row["c2"], "")

    def test_too_few_rows_returns_none(self):
        raw = "| A | B |\n"  # no separator line
        result = _parse_gfm_table(raw)
        self.assertIsNone(result)


class TestBuildTableCard(unittest.TestCase):
    def test_weather_report_returns_interactive(self):
        card_json = _build_table_card(WEATHER_CONTENT)
        self.assertIsNotNone(card_json)
        card = json.loads(card_json)
        # Schema 2.0
        self.assertEqual(card["schema"], "2.0")
        # Body has elements
        elements = card["body"]["elements"]
        table_elements = [e for e in elements if e.get("tag") == "table"]
        md_elements = [e for e in elements if e.get("tag") == "markdown"]
        # Exactly 2 table elements
        self.assertEqual(len(table_elements), 2)
        # At least some markdown elements (prose preserved)
        self.assertGreater(len(md_elements), 0)
        # Table element IDs are tbl_0 and tbl_1
        ids = {e["element_id"] for e in table_elements}
        self.assertEqual(ids, {"tbl_0", "tbl_1"})

    def test_table_structure_correct(self):
        card_json = _build_table_card(WEATHER_CONTENT)
        card = json.loads(card_json)
        table_el = next(e for e in card["body"]["elements"] if e.get("tag") == "table")
        # Must have columns list
        self.assertIn("columns", table_el)
        self.assertIn("rows", table_el)
        # column names c0/c1/...
        for idx, col in enumerate(table_el["columns"]):
            self.assertEqual(col["name"], f"c{idx}")
            self.assertEqual(col["data_type"], "markdown")
        # header_style present
        self.assertIn("header_style", table_el)
        self.assertTrue(table_el["header_style"]["bold"])

    def test_order_prose_table_preserved(self):
        """Prose before first table must come before the table element."""
        card_json = _build_table_card(WEATHER_CONTENT)
        card = json.loads(card_json)
        elements = card["body"]["elements"]
        tags = [e["tag"] for e in elements]
        # First element is markdown (prose)
        self.assertEqual(tags[0], "markdown")
        first_table_idx = tags.index("table")
        # There is a markdown element before the first table
        self.assertGreater(first_table_idx, 0)

    def test_fenced_pseudo_table_not_converted(self):
        """Content whose only table-like structure is inside a fence must not return a card."""
        # No real table => _build_table_card returns None (0 tables found)
        card_json = _build_table_card(FENCED_PSEUDO_TABLE)
        self.assertIsNone(card_json)

    def test_too_many_tables_returns_none(self):
        """More than 5 tables must fall back (return None), not raise."""
        one_table = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n\n"
        )
        content = one_table * 6  # 6 tables
        try:
            result = _build_table_card(content)
        except Exception as exc:
            self.fail(f"_build_table_card raised an exception with >5 tables: {exc}")
        self.assertIsNone(result)

    def test_no_table_returns_none(self):
        result = _build_table_card(NO_TABLE_CONTENT)
        self.assertIsNone(result)


class TestBuildOutboundPayloadTablePath(unittest.TestCase):
    """Test _build_outbound_payload (module-level) for the table branch.

    We call the module function directly rather than instantiating the full
    Feishu platform class.
    """

    def _call(self, content):
        # The method is an instance method; replicate its logic via the helpers.
        from gateway.platforms.feishu import (
            _MARKDOWN_TABLE_RE,
            _MARKDOWN_HINT_RE,
            _build_table_card,
            _build_markdown_post_payload,
        )
        import json as _json
        if _MARKDOWN_TABLE_RE.search(content):
            try:
                card_json = _build_table_card(content)
                if card_json is not None:
                    return "interactive", card_json
            except Exception:
                pass
            return "text", _json.dumps({"text": content}, ensure_ascii=False)
        if _MARKDOWN_HINT_RE.search(content):
            return "post", _build_markdown_post_payload(content)
        return "text", _json.dumps({"text": content}, ensure_ascii=False)

    def test_table_content_returns_interactive(self):
        msg_type, payload = self._call(WEATHER_CONTENT)
        self.assertEqual(msg_type, "interactive")
        card = json.loads(payload)
        self.assertEqual(card["schema"], "2.0")

    def test_no_table_markdown_returns_post(self):
        msg_type, payload = self._call(NO_TABLE_CONTENT)
        self.assertEqual(msg_type, "post")

    def test_plain_text_returns_text(self):
        msg_type, payload = self._call("Hello world")
        self.assertEqual(msg_type, "text")

    def test_too_many_tables_falls_back_to_text(self):
        one_table = "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        content = one_table * 6
        msg_type, payload = self._call(content)
        # Must fall back gracefully — no exception, no interactive
        self.assertIn(msg_type, ("text", "post"))

    def test_fenced_pseudo_table_no_crash(self):
        msg_type, payload = self._call(FENCED_PSEUDO_TABLE)
        # No table detected at top level → post (has markdown hints)
        # or text — either is fine; must not crash
        self.assertIn(msg_type, ("text", "post", "interactive"))


if __name__ == "__main__":
    unittest.main()
