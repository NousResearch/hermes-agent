"""Speculative read-only repo map prefetcher (05-final-plan.md §9, §10).

Scans Python and JS/TS files in a project directory in <50ms using ``ast`` and regex,
stores the symbol map in the ``repo_map`` table of ``.samagent/ledger.db``, and formats
a compact index so workers don't waste turns running ``search_files`` to locate symbols.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import time
from typing import Dict, List

from samagent.ledger.store import ProjectLedger
from tools.threat_patterns import first_threat_message

_IGNORE_DIRS = frozenset({
    ".git",
    ".samagent",
    ".worktrees",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
    ".pytest_cache",
})

_JS_EXPORT_RE = re.compile(
    r"\bexport\s+(?:async\s+)?(?:function|class|const|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def extract_file_symbols(file_path: Path) -> List[str]:
    """Extract top-level classes/functions from .py or exported identifiers from .js/.ts."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    # Untrusted-content rule: skip files that trigger strict threat-pattern scans
    if first_threat_message(text, scope="strict") is not None:
        return ["[skipped: threat_pattern_detected]"]

    symbols: List[str] = []
    if file_path.suffix == ".py":
        try:
            tree = ast.parse(text, filename=str(file_path))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")
                    ]
                    m_str = f"({', '.join(methods[:5])})" if methods else ""
                    symbols.append(f"class {node.name}{m_str}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        symbols.append(f"def {node.name}()")
        except SyntaxError:
            pass
    elif file_path.suffix in (".js", ".ts", ".tsx", ".jsx"):
        for m in _JS_EXPORT_RE.finditer(text):
            symbols.append(f"export {m.group(1)}")
    return symbols[:12]


def prefetch_repo_map(project_dir: Path, ledger: ProjectLedger) -> Dict[str, str]:
    """Scan *project_dir*, populate the ``repo_map`` table in *ledger*, and return ``{rel_path: summary}``."""
    root = Path(project_dir)
    entries: Dict[str, str] = {}
    now = time.time()

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in _IGNORE_DIRS for part in rel_parts):
            continue
        if p.suffix not in (".py", ".js", ".ts", ".tsx", ".html", ".sql"):
            continue
        rel = p.relative_to(root).as_posix()
        syms = extract_file_symbols(p)
        summary = ", ".join(syms) if syms else f"{p.suffix[1:] or 'file'} ({p.stat().st_size}B)"
        entries[rel] = summary

    with ledger._connect() as conn:
        for rel, summary in entries.items():
            conn.execute(
                "INSERT OR REPLACE INTO repo_map (path, symbols_summary, updated_at) VALUES (?, ?, ?)",
                (rel, summary, now),
            )
    return entries
