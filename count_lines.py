#!/usr/bin/env python3
"""统计当前目录下各类型文件的代码行数"""

import os
from pathlib import Path
from collections import defaultdict

# 常见代码文件扩展名
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".sql", ".html", ".css", ".scss", ".less", ".vue", ".svelte",
    ".yaml", ".yml", ".json", ".xml", ".toml", ".ini", ".cfg",
    ".md", ".rst", ".txt",
}

# 跳过的目录
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    ".idea", ".vscode", "target", "vendor", ".next", ".nuxt",
}


def count_lines(path: Path) -> tuple[int, int, int]:
    """返回 (总行数, 非空行数, 注释行数)"""
    total = blank = comment = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                stripped = line.strip()
                if not stripped:
                    blank += 1
                elif stripped.startswith("#") or stripped.startswith("//"):
                    comment += 1
    except (PermissionError, OSError):
        pass
    return total, total - blank, comment


def main():
    base = Path.cwd()
    stats = defaultdict(lambda: {"files": 0, "total": 0, "code": 0, "comment": 0})

    for root, dirs, files in os.walk(base):
        # 原地删除跳过目录
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in files:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            if ext in CODE_EXTENSIONS:
                total, code, comment = count_lines(fpath)
                if total > 0:
                    lang = ext.lstrip(".")
                    stats[lang]["files"] += 1
                    stats[lang]["total"] += total
                    stats[lang]["code"] += code
                    stats[lang]["comment"] += comment

    if not stats:
        print("未找到代码文件")
        return

    # 按代码行数降序排列
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["code"], reverse=True)

    # 表头
    print(f"\n{'语言':<12} {'文件数':>6} {'总行数':>8} {'代码行':>8} {'注释行':>8}")
    print("-" * 50)

    sum_files = sum_total = sum_code = sum_comment = 0
    for lang, s in sorted_stats:
        print(f".{lang:<11} {s['files']:>6} {s['total']:>8} {s['code']:>8} {s['comment']:>8}")
        sum_files += s["files"]
        sum_total += s["total"]
        sum_code += s["code"]
        sum_comment += s["comment"]

    print("-" * 50)
    print(f"{'合计':<12} {sum_files:>6} {sum_total:>8} {sum_code:>8} {sum_comment:>8}\n")


if __name__ == "__main__":
    main()
