#!/usr/bin/env python3
"""Summarize prompt-budget hygiene for repository context files.

This helper is intentionally local-only and stdlib-only. It scans a repo for
common agent context files and reports files that are near or over a configured
character budget so maintainers can split stable procedures into skills/docs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "SOUL.md")
CONTEXT_DIRS = (Path("."), Path(".hermes"))
DEFAULT_MAX_CHARS = 14_000
DEFAULT_WARN_RATIO = 0.8


@dataclass(frozen=True)
class ContextFileResult:
    """Analysis result for a single context file."""

    path: str
    chars: int
    lines: int
    status: str
    budget_ratio: float
    suggestion: str


def discover_context_files(root: str | Path) -> list[Path]:
    """Return known context files under ``root`` in deterministic order."""

    root_path = Path(root)
    paths: list[Path] = []
    for directory in CONTEXT_DIRS:
        for name in CONTEXT_FILE_NAMES:
            candidate = root_path / directory / name
            if candidate.is_file():
                paths.append(candidate)
    return paths


def _relative_display_path(path: Path, root: str | Path | None = None) -> str:
    """Return a stable display path relative to root or cwd when possible."""

    resolved = path.resolve()
    bases = [Path(root).resolve()] if root is not None else []
    bases.append(Path.cwd().resolve())
    for base in bases:
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def _status(chars: int, max_chars: int, warn_ratio: float) -> str:
    if chars > max_chars:
        return "over"
    if chars >= int(max_chars * warn_ratio):
        return "warn"
    return "ok"


def _suggestion(status: str) -> str:
    if status == "over":
        return "Over budget: split stable procedures into skills or reference docs; keep only durable routing rules here."
    if status == "warn":
        return "Near budget: trim repeated examples and move reusable procedures into skills before the file starts truncating."
    return "OK: no action needed."


def analyze_file(
    path: str | Path,
    max_chars: int = DEFAULT_MAX_CHARS,
    warn_ratio: float = DEFAULT_WARN_RATIO,
    root: str | Path | None = None,
) -> ContextFileResult:
    """Analyze one context file's size against the configured budget."""

    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    chars = len(text)
    lines = 0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1)
    status = _status(chars, max_chars, warn_ratio)
    return ContextFileResult(
        path=_relative_display_path(file_path, root=root),
        chars=chars,
        lines=lines,
        status=status,
        budget_ratio=round(chars / max_chars, 3) if max_chars else 0.0,
        suggestion=_suggestion(status),
    )


def summarize(results: Iterable[ContextFileResult]) -> dict[str, int]:
    """Count results by status."""

    summary = {"ok": 0, "warn": 0, "over": 0}
    for result in results:
        summary[result.status] += 1
    return summary


def render_brief(results: Sequence[ContextFileResult], silent_ok: bool = False) -> str:
    """Render a concise human-readable context-file brief."""

    attention = [result for result in results if result.status != "ok"]
    if silent_ok and not attention:
        return "[SILENT]"

    counts = summarize(results)
    if not results:
        return "Context file brief: no known context files found."

    lines = [
        "Context file brief",
        f"Summary: {counts['ok']} ok, {counts['warn']} warn, {counts['over']} over",
    ]

    shown = attention or list(results)
    for result in shown:
        percent = round(result.budget_ratio * 100)
        lines.append(
            f"- {result.path}: {result.status} — {result.chars} chars, {result.lines} lines ({percent}% of budget). {result.suggestion}"
        )

    return "\n".join(lines)


def results_to_json(results: Sequence[ContextFileResult]) -> str:
    """Serialize results and summary as stable JSON."""

    payload = {
        "summary": summarize(results),
        "files": [asdict(result) for result in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan (default: current directory)")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help=f"Character budget per file (default: {DEFAULT_MAX_CHARS})")
    parser.add_argument("--warn-ratio", type=float, default=DEFAULT_WARN_RATIO, help=f"Warn when chars >= max-chars * ratio (default: {DEFAULT_WARN_RATIO})")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--silent-ok", action="store_true", help="Emit exact [SILENT] when all discovered files are OK")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = discover_context_files(args.root)
    results = [
        analyze_file(path, max_chars=args.max_chars, warn_ratio=args.warn_ratio, root=args.root)
        for path in paths
    ]

    if args.json:
        print(results_to_json(results))
    else:
        print(render_brief(results, silent_ok=args.silent_ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
