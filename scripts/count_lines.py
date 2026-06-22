#!/usr/bin/env python3
import os, pathlib
from collections import defaultdict

# Map file extensions to language names
EXT_MAP = {
    '.py': 'Python',
    '.ts': 'TypeScript',
    '.tsx': 'TSX',
    '.js': 'JavaScript',
    '.json': 'JSON',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.sh': 'Shell',
    '.md': 'Markdown',
    '.html': 'HTML',
    '.css': 'CSS',
    '.rs': 'Rust',
    '.toml': 'TOML',
}

ROOT = pathlib.Path(os.path.expanduser('~/hermes-source'))
EXCLUDE_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv'}

stats = defaultdict(lambda: {'files': 0, 'lines': 0})

total_lines = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
    for f in filenames:
        ext = pathlib.Path(f).suffix.lower()
        if ext in EXT_MAP:
            lang = EXT_MAP[ext]
            file_path = pathlib.Path(dirpath) / f
            try:
                with open(file_path, 'rb') as fp:
                    line_count = sum(1 for _ in fp)
            except Exception:
                continue
            stats[lang]['files'] += 1
            stats[lang]['lines'] += line_count
            total_lines += line_count

out_path = pathlib.Path(os.path.expanduser('~/Documents/HermesVault/02_知识/2026-06-22_codex测试_代码行数统计.md'))
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as out:
    out.write('| Language | Files | Lines | Percent |\n')
    out.write('|---|---|---|---|\n')
    for lang, data in sorted(stats.items(), key=lambda x: x[1]['lines'], reverse=True):
        percent = (data['lines'] / total_lines * 100) if total_lines else 0
        out.write(f'| {lang} | {data["files"]} | {data["lines"]} | {percent:.2f}% |\n')
    out.write(f'| **Total** | {sum(d["files"] for d in stats.values())} | {total_lines} | 100% |\n')

print('Report written to', out_path)
