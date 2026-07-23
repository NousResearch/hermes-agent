#!/usr/bin/env python3
"""Scan agent context files for prompt-budget bloat.

This helper is intentionally read-only and local-first. It is useful from cron
jobs that should warn when AGENTS.md / CLAUDE.md / SOUL.md-style files grow
large enough to waste context or hide stale instructions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_CONTEXT_NAMES = (
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "SOUL.md",
    ".cursorrules",
)
DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}
DEFAULT_WARN_BYTES = 20_000
DEFAULT_OVER_BYTES = 50_000


@dataclass(frozen=True)
class ContextFileReport:
    path: Path
    size_bytes: int
    status: str

    def as_dict(self, root: Path) -> dict:
        return {
            "path": _display_path(self.path, root),
            "size_bytes": self.size_bytes,
            "status": self.status,
        }


def classify_size(size_bytes: int, *, warn_bytes: int, over_bytes: int) -> str:
    """Classify a context file size against inclusive thresholds."""
    if over_bytes <= warn_bytes:
        raise ValueError("over_bytes must be greater than warn_bytes")
    if size_bytes >= over_bytes:
        return "over"
    if size_bytes >= warn_bytes:
        return "warn"
    return "ok"


def scan_context_files(
    root: Path | str,
    *,
    names: Sequence[str] = DEFAULT_CONTEXT_NAMES,
    skip_dirs: Iterable[str] = DEFAULT_SKIP_DIRS,
    warn_bytes: int = DEFAULT_WARN_BYTES,
    over_bytes: int = DEFAULT_OVER_BYTES,
) -> list[ContextFileReport]:
    """Return reports for matching context files under ``root``.

    The walk prunes common generated/vendor directories so a repository's
    vendored dependencies do not dominate the report.
    """
    root_path = Path(root).expanduser().resolve()
    wanted = set(names)
    skipped = set(skip_dirs)
    reports: list[ContextFileReport] = []

    if not root_path.exists():
        raise FileNotFoundError(f"root does not exist: {root_path}")
    if root_path.is_file():
        candidates = [root_path] if root_path.name in wanted else []
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [name for name in dirnames if name not in skipped]
            for filename in filenames:
                if filename in wanted:
                    candidates.append(Path(dirpath) / filename)

    for path in sorted(candidates, key=lambda p: str(p.relative_to(root_path) if p != root_path else p)):
        size = path.stat().st_size
        reports.append(
            ContextFileReport(
                path=path,
                size_bytes=size,
                status=classify_size(size, warn_bytes=warn_bytes, over_bytes=over_bytes),
            )
        )
    return reports


def render_json(reports: Sequence[ContextFileReport], root: Path | str) -> str:
    root_path = Path(root).expanduser().resolve()
    counts = Counter(report.status for report in reports)
    payload = {
        "root": str(root_path),
        "counts": {"ok": counts.get("ok", 0), "warn": counts.get("warn", 0), "over": counts.get("over", 0)},
        "files": [report.as_dict(root_path) for report in reports],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def render_markdown(
    reports: Sequence[ContextFileReport],
    root: Path | str,
    *,
    include_ok: bool = False,
    silent_if_ok: bool = False,
) -> str:
    root_path = Path(root).expanduser().resolve()
    attention = [report for report in reports if include_ok or report.status != "ok"]
    if silent_if_ok and not any(report.status != "ok" for report in reports):
        return "[SILENT]"

    counts = Counter(report.status for report in reports)
    if not reports:
        return f"# Context file brief\n\nNo context files found under `{root_path}`."

    lines = [
        "# Context file brief",
        "",
        f"Root: `{root_path}`",
        f"Summary: {counts.get('over', 0)} over / {counts.get('warn', 0)} warn / {counts.get('ok', 0)} ok",
        "",
    ]

    if attention:
        lines.extend([
            "## Files needing attention",
            "",
            "| Status | Size | Path |",
            "| --- | ---: | --- |",
        ])
        for report in attention:
            lines.append(
                f"| {report.status} | {_format_bytes(report.size_bytes)} | `{_display_path(report.path, root_path)}` |"
            )
        lines.extend([
            "",
            "Next action: trim duplicated rules, move procedural detail into skills/references, or split repo-specific context into narrower subdirectory files.",
        ])
    else:
        lines.append("All discovered context files are within the configured budget.")

    return "\n".join(lines)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Directory or file to scan (default: current directory)")
    parser.add_argument("--names", default=",".join(DEFAULT_CONTEXT_NAMES), help="Comma-separated file names to scan")
    parser.add_argument("--warn-bytes", type=int, default=DEFAULT_WARN_BYTES, help="Inclusive warning threshold")
    parser.add_argument("--over-bytes", type=int, default=DEFAULT_OVER_BYTES, help="Inclusive over-budget threshold")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--include-ok", action="store_true", help="Include ok files in Markdown output")
    parser.add_argument("--silent-if-ok", action="store_true", help="Print exactly [SILENT] when every discovered file is ok")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    reports = scan_context_files(
        root,
        names=_parse_names(args.names),
        warn_bytes=args.warn_bytes,
        over_bytes=args.over_bytes,
    )
    if args.format == "json":
        print(render_json(reports, root))
    else:
        print(render_markdown(reports, root, include_ok=args.include_ok, silent_if_ok=args.silent_if_ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
