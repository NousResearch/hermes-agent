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
