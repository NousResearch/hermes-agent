"""Markdown → Feishu Card JSON 2.0 table conversion (pure-stdlib helpers).

Isolated module so the conversion logic can be unit-tested without loading
the full ``gateway.platforms.feishu`` stack (which pulls in ``hermes_cli``,
``lark_oapi``, websockets, aiohttp, etc.). The dispatch is wired up in
``gateway.platforms.feishu._build_outbound_payload``.

Why a dedicated module:

- Feishu Card JSON 1.0 (used by ``send_exec_approval`` / ``send_update_prompt``)
  has no ``tag: "table"`` component. Card JSON 2.0 introduces it (Lark V7.4+),
  and the two schemas may be mixed across messages within the same chat —
  each ``send_message`` payload is independent.
- We invoke 2.0 only when the outbound content actually contains a markdown
  table; everything else stays on the existing text / post / 1.0 card paths.
- Keeping the helpers in their own module makes them trivially unit-testable
  with the stdlib alone (``re``, ``json``, ``typing``).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


_MD_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[-|: ]+\|\s*$")


def _split_md_table_row(row_line: str) -> List[str]:
    """Split one markdown table row line into cell strings.

    Strips the leading / trailing ``|`` and trims surrounding whitespace
    on each cell.
    """
    stripped = row_line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _split_content_at_tables(content: str) -> List[Tuple[str, str]]:
    """Split content into alternating ('text', str) and ('table', str) segments.

    A table starts at a pipe-row whose next line matches the separator
    pattern ``^\\s*\\|[-|: ]+\\|\\s*$``. The table extends through every
    following line that begins with ``|``. Non-table chunks (including
    blank lines) are returned verbatim.
    """
    segments: List[Tuple[str, str]] = []
    lines = content.splitlines(keepends=True)
    n = len(lines)
    i = 0
    while i < n:
        is_table_start = (
            i + 1 < n
            and lines[i].lstrip().startswith("|")
            and lines[i].rstrip().endswith("|")
            and _MD_TABLE_SEPARATOR_RE.match(lines[i + 1] or "")
        )
        if is_table_start:
            j = i + 2
            while j < n and lines[j].lstrip().startswith("|"):
                j += 1
            segments.append(("table", "".join(lines[i:j])))
            i = j
            continue
        # Accumulate non-table lines until next table start or EOF.
        j = i
        while j < n:
            if (
                j + 1 < n
                and lines[j].lstrip().startswith("|")
                and lines[j].rstrip().endswith("|")
                and _MD_TABLE_SEPARATOR_RE.match(lines[j + 1] or "")
            ):
                break
            j += 1
        segments.append(("text", "".join(lines[i:j])))
        i = j
    return segments


def _parse_markdown_table_to_card_element(table_text: str) -> Dict[str, Any]:
    """Parse a markdown table block into a Feishu Card JSON 2.0 table element.

    First content row is treated as the header. Subsequent rows are data.
    Columns use ``data_type: "markdown"`` so cell content can carry inline
    formatting (bold / links / emoji ✅/❌ etc.) that scout-mcp commonly
    emits. ``page_size: 10`` keeps long shortlists scrollable without
    making the card huge.
    """
    rows_raw = [ln for ln in table_text.splitlines() if ln.strip().startswith("|")]
    if len(rows_raw) < 2:
        return {"tag": "markdown", "content": table_text}
    header_cells = _split_md_table_row(rows_raw[0])
    data_rows_raw = rows_raw[2:]  # rows_raw[1] is the |---| separator
    columns: List[Dict[str, Any]] = []
    for idx, cell in enumerate(header_cells):
        col_name = f"col{idx + 1}"
        columns.append(
            {
                "name": col_name,
                "display_name": cell or col_name,
                "data_type": "markdown",
            }
        )
    rows: List[Dict[str, Any]] = []
    for row_line in data_rows_raw:
        cells = _split_md_table_row(row_line)
        while len(cells) < len(columns):
            cells.append("")
        rows.append({columns[i]["name"]: cells[i] for i in range(len(columns))})
    return {
        "tag": "table",
        "page_size": 10,
        "columns": columns,
        "rows": rows,
    }


def _build_card_with_table_payload(content: str) -> str:
    """Build a Feishu interactive card (JSON 2.0) preserving markdown tables.

    Splits ``content`` at each markdown table block: prose around the table
    becomes ``tag: "markdown"`` element(s); each table becomes a native
    ``tag: "table"`` element. Renders as sortable / paginated table UI on
    Lark V7.4+ clients instead of the previously-broken md-table fallback
    that produced blank messages.
    """
    segments = _split_content_at_tables(content)
    elements: List[Dict[str, Any]] = []
    for seg_kind, seg_text in segments:
        if seg_kind == "table":
            elements.append(_parse_markdown_table_to_card_element(seg_text))
            # Native client UX hint: educate users about Feishu's built-in
            # cell-copy gestures, plus the raw-markdown fallback they can
            # long-press select. Lightweight zero-dependency enhancement —
            # later iterations can replace this with a "Save to Sheet"
            # button once OAuth scopes (sheets:spreadsheet) are granted.
            elements.append(_build_table_hint_element())
            # Append the raw markdown source itself so long-press select +
            # copy works for the entire table at once on mobile clients
            # that don't expose per-cell copy.
            elements.append(_build_raw_source_disclosure(seg_text))
        elif seg_text.strip():
            elements.append({"tag": "markdown", "content": seg_text})
    if not elements:
        elements = [{"tag": "markdown", "content": content}]
    card: Dict[str, Any] = {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "body": {"elements": elements},
    }
    return json.dumps(card, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Self-check entry point — `python3 gateway/platforms/_feishu_card_table.py`
# ---------------------------------------------------------------------------
#
# Runs a minimal end-to-end check using a real Scout / Feishu transcript
# fragment (the 2026-05-21 office-furniture incident). Does NOT import the
# full gateway package, so it works in a sparse checkout without hermes_cli
# or lark_oapi installed.
#
# Run via:
#     python3 gateway/platforms/_feishu_card_table.py
#
# Exits non-zero on any assertion failure; on success, dumps the produced
# card JSON so a human can eyeball the structure before shipping.


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


_TABLE_HINT_CONTENT = (
    "💡 **复制 / 下载提示** · 长按 cell 复制单元格 · "
    "长按下方原始数据可全选复制并粘贴到飞书表格 / Excel · "
    "「保存到飞书表格」按钮 v3 上线中"
)


def _build_table_hint_element() -> Dict[str, Any]:
    """Inline markdown hint educating users about native Feishu copy gestures.

    Lightweight zero-dependency enhancement: explains that Feishu desktop
    + mobile clients already support long-press copy on table cells, and
    points to the raw-markdown disclosure block that follows for full
    table copy / paste workflows.
    """
    return {"tag": "markdown", "content": _TABLE_HINT_CONTENT}


def _build_raw_source_disclosure(table_text: str) -> Dict[str, Any]:
    """Re-emit the original markdown table inside a fenced code block.

    Feishu cards render fenced ``markdown`` content as plain monospace,
    so long-press select-all on this block lets users copy the table as
    valid markdown to paste into Lark Sheets, Excel, or any md-capable
    surface. This is the closest we can ship without a backend Sheets
    API integration.
    """
    stripped = table_text.rstrip("\n")
    return {
        "tag": "markdown",
        "content": f"```markdown\n{stripped}\n```",
    }


def _run_selfcheck() -> int:
    raw = _build_card_with_table_payload(_SCOUT_FEISHU_FRAGMENT)
    card = json.loads(raw)

    # Structural assertions
    assert card["schema"] == "2.0", f"expected schema 2.0, got {card['schema']!r}"
    elements = card["body"]["elements"]
    tags = [e["tag"] for e in elements]
    # Order: intro-markdown / table / hint-markdown / raw-source-markdown / footer-markdown
    assert tags == [
        "markdown", "table", "markdown", "markdown", "markdown"
    ], f"unexpected element order: {tags}"

    # Hint element must contain the copy-affordance educational text.
    assert "长按 cell 复制单元格" in elements[2]["content"], "hint element missing copy text"
    # Raw-source disclosure must be a fenced markdown block preserving pipes.
    assert elements[3]["content"].startswith("```markdown"), "raw-source missing fence"
    assert "| 关键词 |" in elements[3]["content"], "raw-source missing original table"

    table = elements[1]
    assert len(table["columns"]) == 4, f"expected 4 columns, got {len(table['columns'])}"
    assert table["columns"][0]["display_name"] == "关键词"
    assert len(table["rows"]) == 6, f"expected 6 rows, got {len(table['rows'])}"

    last = table["rows"][5]
    assert "criss cross office chair" in last["col1"], f"row 5 col1 wrong: {last['col1']!r}"
    assert last["col4"] == "✅", f"row 5 col4 wrong: {last['col4']!r}"

    # Dump for visual eyeballing
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
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_selfcheck())
