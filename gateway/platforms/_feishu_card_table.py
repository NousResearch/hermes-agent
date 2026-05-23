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
from typing import Any, Dict, List, Optional, Tuple


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


def _build_card_with_table_payload(
    content: str,
    sheet_url: Optional[str] = None,
    bitable_url: Optional[str] = None,
) -> str:
    """Build a Feishu interactive card (JSON 2.0) preserving markdown tables.

    Splits ``content`` at each markdown table block: prose around the table
    becomes ``tag: "markdown"`` element(s); each table becomes a native
    ``tag: "table"`` element. Renders as sortable / paginated table UI on
    Lark V7.4+ clients instead of the previously-broken md-table fallback
    that produced blank messages.

    Right after the **first** table, a single inline element renders the
    storage CTAs as bold markdown links (schema-portable across Card JSON
    v1 / v2). Two channels are supported:

    - ``sheet_url`` — populated Lark Sheet for download / edit / share
      (Lark Sheet has built-in CSV export = "本地下载")
    - ``bitable_url`` — blank Lark Bitable app for downstream automation
      / view / dashboard workflows ("保存到飞书多维表格")

    Both ``None`` ⇒ no storage CTA element is emitted; the user still
    gets the native table UI (long-press copy on cells works natively).
    """
    segments = _split_content_at_tables(content)
    elements: List[Dict[str, Any]] = []
    table_index = 0
    for seg_kind, seg_text in segments:
        if seg_kind == "table":
            elements.append(_parse_markdown_table_to_card_element(seg_text))
            if table_index == 0:
                storage_el = _build_storage_links_element(sheet_url, bitable_url)
                if storage_el is not None:
                    elements.append(storage_el)
            table_index += 1
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


def _build_storage_links_element(
    sheet_url: Optional[str],
    bitable_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Render the inline storage CTAs as bold markdown links.

    Two complementary export channels:

    - **Lark Sheet** (``sheet_url``): data is pre-populated; clients use
      the in-sheet menu to export CSV / Excel locally ("本地下载") or
      copy / share / edit collaboratively.
    - **Lark Bitable** (``bitable_url``): blank base ready for views,
      automations, and downstream pipelines ("保存到飞书多维表格 — 自己
      组织的多维表格").

    Markdown links are deliberately chosen over ``tag: "button"`` because
    they are schema-portable across Card JSON v1 / v2 and every Lark
    client build. Returns ``None`` when neither URL is available so the
    caller can skip emitting an empty element.
    """
    parts: List[str] = []
    if sheet_url:
        parts.append(f"[**📥 下载 CSV · 编辑(飞书表格)**]({sheet_url})")
    if bitable_url:
        parts.append(f"[**📊 保存到多维表格**]({bitable_url})")
    if not parts:
        return None
    return {"tag": "markdown", "content": "  ·  ".join(parts)}


def _run_selfcheck() -> int:
    # === Case A: no storage URLs — minimal card (intro / table / footer) ===
    raw = _build_card_with_table_payload(_SCOUT_FEISHU_FRAGMENT)
    card = json.loads(raw)
    assert card["schema"] == "2.0", f"expected schema 2.0, got {card['schema']!r}"
    elements = card["body"]["elements"]
    tags = [e["tag"] for e in elements]
    assert tags == [
        "markdown", "table", "markdown"
    ], f"case A: unexpected element order: {tags}"
    # No raw-markdown disclosure block in the output anymore.
    for el in elements:
        assert not el.get("content", "").startswith("```markdown"), (
            "case A: raw-source disclosure must be gone in v4"
        )

    # === Case B: sheet_url only — single CTA link ===
    mock_sheet = "https://feishu.cn/sheets/shtcnMOCKTOKEN12345"
    raw_b = _build_card_with_table_payload(_SCOUT_FEISHU_FRAGMENT, sheet_url=mock_sheet)
    card_b = json.loads(raw_b)
    elements_b = card_b["body"]["elements"]
    tags_b = [e["tag"] for e in elements_b]
    # Order: intro / table / storage-links / footer
    assert tags_b == [
        "markdown", "table", "markdown", "markdown"
    ], f"case B: unexpected element order: {tags_b}"
    assert mock_sheet in elements_b[2]["content"], "case B: sheet URL missing"
    assert "📥 下载 CSV" in elements_b[2]["content"], "case B: missing CSV button label"
    assert "多维表格" not in elements_b[2]["content"], "case B: bitable label should be absent"

    # === Case C: both sheet_url and bitable_url — twin CTA links ===
    mock_bitable = "https://feishu.cn/base/bascnMOCKTOKEN67890"
    raw_c = _build_card_with_table_payload(
        _SCOUT_FEISHU_FRAGMENT, sheet_url=mock_sheet, bitable_url=mock_bitable
    )
    card_c = json.loads(raw_c)
    elements_c = card_c["body"]["elements"]
    storage = elements_c[2]["content"]
    assert mock_sheet in storage and mock_bitable in storage, (
        "case C: both URLs must appear in the storage CTA element"
    )
    assert "📥 下载 CSV" in storage and "📊 保存到多维表格" in storage, (
        "case C: both button labels must appear"
    )

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
