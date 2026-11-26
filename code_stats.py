#!/usr/bin/env python3
"""代码统计分析工具 - 按目录/类型分组，区分代码行/注释行/空行，ASCII柱状图"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# 文件类型 → (单行注释, 多行开始, 多行结束)
LANG_MAP = {
    ".py":   ("#", '"""', '"""'),
    ".js":   ("//", "/*", "*/"),
    ".ts":   ("//", "/*", "*/"),
    ".tsx":  ("//", "/*", "*/"),
    ".jsx":  ("//", "/*", "*/"),
    ".java": ("//", "/*", "*/"),
    ".c":    ("//", "/*", "*/"),
    ".cpp":  ("//", "/*", "*/"),
    ".h":    ("//", "/*", "*/"),
    ".go":   ("//", "/*", "*/"),
    ".rs":   ("//", "/*", "*/"),
    ".rb":   ("#", "=begin", "=end"),
    ".sh":   ("#", None, None),
    ".bash": ("#", None, None),
    ".yml":  ("#", None, None),
    ".yaml": ("#", None, None),
    ".toml": ("#", None, None),
    ".sql":  ("--", "/*", "*/"),
    ".html": (None, "<!--", "-->"),
    ".css":  (None, "/*", "*/"),
    ".vue":  ("//", "/*", "*/"),
    ".swift": ("//", "/*", "*/"),
    ".kt":   ("//", "/*", "*/"),
    ".lua":  ("--", "--[[", "]]"),
    ".r":    ("#", None, None),
    ".php":  ("//", "/*", "*/"),
}

SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__",
    ".venv", "venv", "env", ".env", ".tox", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".eggs", "target", "vendor",
    ".next", ".nuxt", "coverage",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def analyze_file(filepath: Path) -> dict:
    ext = filepath.suffix.lower()
    if ext not in LANG_MAP:
        return None

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except (OSError, PermissionError):
        return None

    line_comment, block_start, block_end = LANG_MAP[ext]
    total = len(lines)
    blank = 0
    comment = 0
    in_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue
        if in_block:
            comment += 1
            if block_end and block_end in stripped:
                in_block = False
            continue
        if block_start and stripped.startswith(block_start):
            comment += 1
            if block_end and block_end not in stripped[len(block_start):]:
                in_block = True
            continue
        if line_comment and stripped.startswith(line_comment):
            comment += 1

    return {"ext": ext, "total": total, "code": total - blank - comment, "comment": comment, "blank": blank}


def bar(value, max_val, width=25):
    if max_val == 0:
        return ""
    return "█" * int(width * value / max_val) + "░" * (width - int(width * value / max_val))


def stacked_bar(code, comment, blank, total, width=30):
    """三段式柱状图：█代码 ▓注释 ░空行"""
    if total == 0:
        return "░" * width
    cw = int(width * code / total)
    mw = int(width * comment / total)
    bw = width - cw - mw
    return "█" * cw + "▓" * mw + "░" * bw


def fmt(n):
    """数字格式化"""
    return f"{n:,}"


def print_table(title, rows, headers, col_widths, grand_row=None):
    """通用表格打印"""
    print(f"\n📁 {title}")
    header_line = "".join(h.rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("─" * len(header_line))

    for row in rows:
        print("".join(str(v).rjust(w) for v, w in zip(row, col_widths)))

    if grand_row:
        print("─" * len(header_line))
        print("".join(str(v).rjust(w) for v, w in zip(grand_row, col_widths)))


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not target.is_dir():
        print(f"❌ 不是有效目录: {target}")
        sys.exit(1)

    # 收集所有文件的分析结果
    file_results = []
    for filepath in target.rglob("*"):
        if not filepath.is_file() or should_skip(filepath):
            continue
        result = analyze_file(filepath)
        if result:
            rel = filepath.relative_to(target)
            file_results.append({"path": rel, **result})

    if not file_results:
        print("未找到可分析的代码文件。")
        return

    # 汇总
    grand = {"files": len(file_results), "total": 0, "code": 0, "comment": 0, "blank": 0}
    for r in file_results:
        for k in ("total", "code", "comment", "blank"):
            grand[k] += r[k]

    comment_rate = grand["comment"] / max(grand["code"] + grand["comment"], 1) * 100

    print(f"\n📊 代码统计 — {target.resolve()}")
    print(f"   {grand['files']} 个文件 | {fmt(grand['total'])} 行 (代码 {fmt(grand['code'])} / 注释 {fmt(grand['comment'])} / 空行 {fmt(grand['blank'])})")
    print(f"   代码占比 {grand['code']/max(grand['total'],1)*100:.1f}% | 注释率 {comment_rate:.1f}%")

    # ── 按目录分组 ──
    dir_stats = defaultdict(lambda: {"files": 0, "total": 0, "code": 0, "comment": 0, "blank": 0})
    for r in file_results:
        top_dir = r["path"].parts[0] if len(r["path"].parts) > 1 else "[root]"
        bucket = dir_stats[top_dir]
        bucket["files"] += 1
        for k in ("total", "code", "comment", "blank"):
            bucket[k] += r[k]

    headers = ["目录", "文件数", "代码行", "注释行", "空行", "注释率"]
    col_widths = [16, 8, 10, 10, 10, 9]
    rows = []
    for d, b in sorted(dir_stats.items(), key=lambda x: x[1]["code"], reverse=True):
        cr = b["comment"] / max(b["code"] + b["comment"], 1) * 100
        rows.append([d, str(b["files"]), fmt(b["code"]), fmt(b["comment"]), fmt(b["blank"]), f"{cr:.1f}%"])

    print_table("按目录分组", rows, headers, col_widths,
                grand_row=["合计", str(grand["files"]), fmt(grand["code"]), fmt(grand["comment"]), fmt(grand["blank"]), f"{comment_rate:.1f}%"])

    # 按目录柱状图
    max_code = max(b["code"] for b in dir_stats.values())
    print(f"\n📊 目录代码量分布")
    for d, b in sorted(dir_stats.items(), key=lambda x: x[1]["code"], reverse=True):
        pct = b["code"] / max(grand["code"], 1) * 100
        print(f"  {d:<14} {bar(b['code'], max_code, 30)} {pct:5.1f}%")

    # ── 按文件类型 ──
    ext_stats = defaultdict(lambda: {"files": 0, "total": 0, "code": 0, "comment": 0, "blank": 0})
    for r in file_results:
        bucket = ext_stats[r["ext"]]
        bucket["files"] += 1
        for k in ("total", "code", "comment", "blank"):
            bucket[k] += r[k]

    headers2 = ["类型", "文件数", "代码行", "注释行", "空行", "注释率"]
    col_widths2 = [10, 8, 10, 10, 10, 9]
    rows2 = []
    for ext, b in sorted(ext_stats.items(), key=lambda x: x[1]["code"], reverse=True):
        cr = b["comment"] / max(b["code"] + b["comment"], 1) * 100
        rows2.append([ext, str(b["files"]), fmt(b["code"]), fmt(b["comment"]), fmt(b["blank"]), f"{cr:.1f}%"])

    print_table("按文件类型", rows2, headers2, col_widths2,
                grand_row=["合计", str(grand["files"]), fmt(grand["code"]), fmt(grand["comment"]), fmt(grand["blank"]), f"{comment_rate:.1f}%"])

    # 按类型柱状图
    max_code_ext = max(b["code"] for b in ext_stats.values())
    print(f"\n📊 类型代码量分布")
    for ext, b in sorted(ext_stats.items(), key=lambda x: x[1]["code"], reverse=True):
        pct = b["code"] / max(grand["code"], 1) * 100
        print(f"  {ext:<8} {bar(b['code'], max_code_ext, 30)} {pct:5.1f}%")

    # ── 综合堆叠柱状图（按目录） ──
    print(f"\n📊 目录代码结构（█代码 ▓注释 ░空行）")
    for d, b in sorted(dir_stats.items(), key=lambda x: x[1]["code"], reverse=True):
        print(f"  {d:<14} {stacked_bar(b['code'], b['comment'], b['blank'], b['total'], 40)}")

    print()


if __name__ == "__main__":
    main()
