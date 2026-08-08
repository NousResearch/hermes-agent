#!/usr/bin/env python3
"""Audit Hermes/agent context files for prompt-budget hygiene.

This helper is intentionally local-only: it reports file sizes and stable
relative paths without printing context file contents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "SOUL.md")
CONTEXT_DIRS = (".", ".hermes")
DEFAULT_MAX_CHARS = 14_000
DEFAULT_WARN_RATIO = 0.75
SILENT_MARKER = "[SILENT]"


@dataclass(frozen=True)
class ContextFile:
    path: Path
    relative_path: str


@dataclass(frozen=True)
class ContextReport:
    path: str
    char_count: int
    line_count: int
    max_chars: int
    budget_ratio: float
    status: str
    suggestion: str


def _stable_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_context_files(root: Path) -> list[ContextFile]:
    """Return known context files in deterministic root-then-.hermes order."""
    root = root.resolve()
    found: list[ContextFile] = []
    for directory in CONTEXT_DIRS:
        base = root if directory == "." else root / directory
        for filename in CONTEXT_FILENAMES:
            path = base / filename
            if path.is_file():
                found.append(ContextFile(path=path, relative_path=_stable_relative(path, root)))
    return found


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _status_for(char_count: int, max_chars: int, warn_ratio: float) -> str:
    if char_count > max_chars:
        return "over"
    if char_count >= max_chars * warn_ratio:
        return "warn"
    return "ok"


def _suggestion_for(status: str) -> str:
    if status == "ok":
        return "No action needed."
    if status == "warn":
        return "Review soon: split durable procedures into skills/references before this context file becomes a prompt-budget sink."
    return "Action recommended: split durable procedures into skills/references and keep only routing-critical guidance in the context file."


def analyze_file(path: Path, root: Path, *, max_chars: int, warn_ratio: float = DEFAULT_WARN_RATIO) -> ContextReport:
    """Measure one context file without exposing its contents."""
    text = path.read_text(encoding="utf-8", errors="replace")
    char_count = len(text)
    status = _status_for(char_count, max_chars, warn_ratio)
    ratio = 0.0 if max_chars <= 0 else char_count / max_chars
    return ContextReport(
        path=_stable_relative(path.resolve(), root.resolve()),
        char_count=char_count,
        line_count=_line_count(text),
        max_chars=max_chars,
        budget_ratio=round(ratio, 3),
        status=status,
        suggestion=_suggestion_for(status),
    )


def analyze_root(root: Path, *, max_chars: int, warn_ratio: float = DEFAULT_WARN_RATIO) -> list[ContextReport]:
    return [
        analyze_file(item.path, root, max_chars=max_chars, warn_ratio=warn_ratio)
        for item in discover_context_files(root)
    ]


def has_attention_items(reports: Iterable[ContextReport]) -> bool:
    return any(report.status != "ok" for report in reports)


def render_human(reports: list[ContextReport], root: Path) -> str:
    if not reports:
        return f"No known context files found under {root}."

    lines = [f"Context file prompt-budget brief for {root}:"]
    for report in reports:
        percent = report.budget_ratio * 100
        lines.append(
            f"- {report.path}: {report.status} — {report.char_count:,} chars, "
            f"{report.line_count:,} lines ({percent:.1f}% of {report.max_chars:,})"
        )
        if report.status != "ok":
            lines.append(f"  ↳ {report.suggestion}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit agent context files for prompt-budget hygiene.")
    parser.add_argument("--root", default=".", help="Repository/workspace root to scan (default: current directory).")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help=f"Per-file character budget (default: {DEFAULT_MAX_CHARS}).")
    parser.add_argument("--warn-ratio", type=float, default=DEFAULT_WARN_RATIO, help=f"Warn at this fraction of --max-chars (default: {DEFAULT_WARN_RATIO}).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--silent-ok", action="store_true", help=f"Print exact {SILENT_MARKER} when every discovered file is OK.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    reports = analyze_root(root, max_chars=args.max_chars, warn_ratio=args.warn_ratio)

    attention = has_attention_items(reports)
    if args.silent_ok and not attention:
        print(SILENT_MARKER)
        return 0

    if args.json:
        payload = {"root": str(root), "files": [asdict(report) for report in reports]}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_human(reports, root))

    return 1 if attention else 0


if __name__ == "__main__":
    raise SystemExit(main())
