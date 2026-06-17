"""Feishu Interactive Card builder.

Constructs card JSON for FeishuAdapter — handles content → card element
conversion, markdown table parsing, tool semantic mapping, and footer
field formatting. Only imported by feishu.py; no upstream dependencies.
"""
from __future__ import annotations


def format_token_count(value: int) -> str:
    if value < 0:
        value = 0
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


TOOL_SEMANTICS: dict[str, tuple[str, str]] = {
    "Read": ("Read", "阅读文件"),
    "read_file": ("Read", "阅读文件"),
    "Bash": ("Bash", "执行命令"),
    "terminal": ("Bash", "执行命令"),
    "Edit": ("Edit", "改代码"),
    "edit_file": ("Edit", "改代码"),
    "Write": ("Write", "写文件"),
    "write_file": ("Write", "写文件"),
    "MultiEdit": ("MultiEdit", "批量改代码"),
    "Grep": ("Grep", "搜索代码"),
    "Glob": ("Glob", "查找文件"),
    "WebFetch": ("WebFetch", "抓取网页"),
    "web_fetch": ("WebFetch", "抓取网页"),
    "WebSearch": ("WebSearch", "搜索网络"),
    "web_search": ("WebSearch", "搜索网络"),
    "Task": ("Agent", "派出子任务"),
    "Agent": ("Agent", "派出子任务"),
    "TodoWrite": ("TodoWrite", "更新任务"),
}


def get_tool_display(tool_name: str) -> str:
    entry = TOOL_SEMANTICS.get(tool_name)
    if entry:
        return f"{entry[0]} · {entry[1]}"
    return tool_name


import re

_TABLE_RE = re.compile(
    r"(?:^|\n)"
    r"(\|[^\n]+\|\n)"        # header row
    r"(\|[\s:|-]+\|\n)"      # separator row
    r"((?:\|[^\n]+\|\n?)+)", # data rows
    re.MULTILINE,
)


def _parse_table_block(match: re.Match) -> dict:
    header_line = match.group(1).strip()
    data_lines = match.group(3).strip().splitlines()

    headers = [h.strip() for h in header_line.strip("|").split("|")]
    columns = [
        {"name": f"col_{i}", "display_name": h}
        for i, h in enumerate(headers)
    ]
    rows = []
    for line in data_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        row = {f"col_{i}": cells[i] if i < len(cells) else "" for i in range(len(headers))}
        rows.append(row)

    return {"columns": columns, "rows": rows}


def parse_markdown_tables(text: str) -> list[tuple[str, ...]]:
    if not text:
        return [("text", "")]

    result: list[tuple] = []
    last_end = 0

    for match in _TABLE_RE.finditer(text):
        start = match.start()
        if text[start] == "\n":
            start += 1
        before = text[last_end:match.start()].strip()
        if before:
            result.append(("text", before))
        result.append(("table", _parse_table_block(match)))
        last_end = match.end()

    after = text[last_end:].strip()
    if after:
        result.append(("text", after))

    if not result:
        result.append(("text", text))

    return result


from typing import Optional


def build_card_json(
    *,
    content: str,
    footer_line: Optional[str] = None,
    status_text: Optional[str] = None,
    tool_status: Optional[str] = None,
) -> dict:
    card = {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "elements": [],
    }

    if tool_status and content:
        display = f"{content}\n\n{tool_status}"
        card["elements"].append({"tag": "markdown", "content": display})
    elif tool_status:
        card["elements"].append({"tag": "markdown", "content": tool_status})
    else:
        segments = parse_markdown_tables(content)
        for seg_type, seg_data in segments:
            if seg_type == "table":
                card["elements"].append({"tag": "table", **seg_data})
            else:
                card["elements"].append({"tag": "markdown", "content": seg_data})

    if footer_line:
        card["elements"].append({"tag": "hr"})
        card["elements"].append({
            "tag": "note",
            "elements": [{"tag": "plain_text", "content": footer_line}],
        })

    if status_text:
        card["elements"].append({
            "tag": "note",
            "elements": [{"tag": "plain_text", "content": status_text}],
        })

    return card
