"""Feishu Interactive Card builder.

Constructs card JSON for FeishuAdapter — handles content → card element
conversion, markdown table parsing, tool semantic mapping, and footer
field formatting. Only imported by feishu.py; no upstream dependencies.
"""
from __future__ import annotations

import os
import subprocess


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

MAX_MARKDOWN_CHARS = 4000
MAX_TABLES = 5
MAX_ELEMENTS = 30
MAX_CARD_BYTES = 24000

_TRUNCATION_NOTICE = "...(内容过长已截断)"


class CardElementValidator:
    """Pre-flight validation for Feishu card elements."""

    @staticmethod
    def validate(elements: list[dict]) -> list[dict]:
        result = _enforce_table_limit(elements)
        result = _split_long_markdown(result)
        result = _merge_adjacent_markdown(result)
        result = _enforce_byte_limit(result)
        return result


def _enforce_table_limit(elements: list[dict]) -> list[dict]:
    table_count = 0
    result = []
    for el in elements:
        if el.get("tag") == "table":
            table_count += 1
            if table_count > MAX_TABLES:
                result.append({"tag": "markdown", "content": _table_to_markdown(el)})
                continue
        result.append(el)
    return result


def _table_to_markdown(table_el: dict) -> str:
    cols = table_el.get("columns", [])
    rows = table_el.get("rows", [])
    if not cols:
        return ""
    headers = [c.get("display_name", c.get("name", "")) for c in cols]
    col_keys = [c.get("name", "") for c in cols]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(row.get(k, "")) for k in col_keys]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _split_long_markdown(elements: list[dict]) -> list[dict]:
    result = []
    for el in elements:
        if el.get("tag") != "markdown":
            result.append(el)
            continue
        content = el["content"]
        if len(content) <= MAX_MARKDOWN_CHARS:
            result.append(el)
            continue
        chunks = _split_by_paragraphs(content, MAX_MARKDOWN_CHARS)
        for chunk in chunks:
            result.append({"tag": "markdown", "content": chunk})
    return result


def _split_by_paragraphs(text: str, max_chars: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.append(para[:max_chars - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE)
            continue
        added_len = len(para) + (2 if current else 0)
        if current and current_len + added_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += added_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text[:max_chars - len(_TRUNCATION_NOTICE)] + _TRUNCATION_NOTICE]


def _merge_adjacent_markdown(elements: list[dict]) -> list[dict]:
    if len(elements) <= MAX_ELEMENTS:
        return elements
    result: list[dict] = []
    for el in elements:
        if (
            el.get("tag") == "markdown"
            and result
            and result[-1].get("tag") == "markdown"
            and len(result) >= MAX_ELEMENTS
        ):
            merged = result[-1]["content"] + "\n\n" + el["content"]
            if len(merged) <= MAX_MARKDOWN_CHARS:
                result[-1] = {"tag": "markdown", "content": merged}
                continue
        result.append(el)
    if len(result) > MAX_ELEMENTS:
        result = result[:MAX_ELEMENTS]
    return result


def _enforce_byte_limit(elements: list[dict]) -> list[dict]:
    import json as _json
    encoded = _json.dumps(elements, ensure_ascii=False).encode()
    if len(encoded) <= MAX_CARD_BYTES:
        return elements
    while len(elements) > 1:
        elements = elements[:-1]
        encoded = _json.dumps(elements, ensure_ascii=False).encode()
        if len(encoded) <= MAX_CARD_BYTES:
            break
    if elements and elements[-1].get("tag") == "markdown":
        content = elements[-1]["content"]
        while len(_json.dumps(elements, ensure_ascii=False).encode()) > MAX_CARD_BYTES and len(content) > 100:
            content = content[:len(content) // 2]
            elements[-1] = {"tag": "markdown", "content": content + _TRUNCATION_NOTICE}
    return elements


def build_progress_card_json(
    *,
    accumulated_text: str,
    tool_lines: list[str],
    status_line: str,
) -> dict:
    card = {
        "config": {"wide_screen_mode": True, "update_multi": True},
        "elements": [],
    }
    parts: list[str] = []
    if accumulated_text:
        parts.append(accumulated_text)
    if tool_lines:
        parts.append(" → ".join(tool_lines))
    if status_line:
        parts.append(f"*{status_line}*")
    body = "\n\n".join(parts) if parts else status_line
    card["elements"].append({"tag": "markdown", "content": body})
    card["elements"] = CardElementValidator.validate(card["elements"])
    return card


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


def detect_git_context(cwd: str) -> str:
    if not cwd:
        return ""
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        if toplevel.returncode != 0:
            return ""

        repo_name = ""
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            url = remote.stdout.strip()
            repo_name = url.rstrip("/").rsplit("/", 1)[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
        if not repo_name:
            repo_name = os.path.basename(toplevel.stdout.strip())

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd, capture_output=True, text=True, timeout=1,
        )
        branch_name = branch.stdout.strip() if branch.returncode == 0 else ""
        if not branch_name:
            return ""

        return f"{repo_name}:{branch_name}"
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _model_short(model: str) -> str:
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def build_card_footer_line(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_tokens: int = 0,
    cost_usd: float = 0.0,
    git_context: str = "",
    elapsed_seconds: float = 0.0,
    model: str = "",
    context_tokens: int = 0,
    context_length: int = 0,
) -> str:
    parts: list[str] = []
    parts.append(f"↑{format_token_count(input_tokens)}")
    parts.append(f"↓{format_token_count(output_tokens)}")
    if cache_tokens > 0:
        parts.append(f"cache:{format_token_count(cache_tokens)}")
    if context_length and context_length > 0 and context_tokens >= 0:
        pct = max(0, min(100, round((context_tokens / context_length) * 100)))
        parts.append(f"ctx:{pct}%")
    parts.append(f"${cost_usd:.4f}")
    if git_context:
        parts.append(f"@{git_context}")
    _secs = int(elapsed_seconds)
    parts.append(f"⏳{_secs // 60}m{_secs % 60}s" if _secs >= 60 else f"⏳{_secs}s")
    m = _model_short(model)
    if m:
        parts.append(f"🧠{m}")
    return "📊 " + " | ".join(parts)
