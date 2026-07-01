#!/usr/bin/env python3
"""
claude-history-scan.py — scan ~/.claude/projects/ and emit session metadata.

M1 of the Claude Code history sidebar feature (see
~/.hermes/plans/2026-07-01-claude-code-history-sidebar.md).

Strictly READ-ONLY on the Claude home directory. Emits metadata only; never
persists message content (raw data, tool_result, attachment, etc.).

Output schema (one entry per session .jsonl):
{
  "session_id":      "<uuid>",
  "cwd":             "C:\\...",
  "first_user":      "first line of first user text block, truncated to 100 chars",
  "first_timestamp": "ISO-8601 string or null",
  "last_timestamp":  "ISO-8601 string or null",
  "message_count":   <int, user+assistant rows that have a text block>,
  "workspace_group": "<immediate subdir under projects/>",
  "file_size_bytes": <int>,
  "file_path":       "<absolute path to the .jsonl>"
}

Usage:
  python claude-history-scan.py
  python claude-history-scan.py --claude-home C:\\Users\\hf\\.claude
  python claude-history-scan.py --output metadata.json
  python claude-history-scan.py --pretty
  python claude-history-scan.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SKIP_TYPES = frozenset({"queue-operation", "attachment"})
COUNT_TYPES = frozenset({"user", "assistant"})
PREVIEW_MAX_CHARS = 100


@dataclass(frozen=True)
class ScanResult:
    session_id: str
    cwd: str | None
    first_user: str | None
    first_timestamp: str | None
    last_timestamp: str | None
    message_count: int
    workspace_group: str
    file_size_bytes: int
    file_path: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "first_user": self.first_user,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "message_count": self.message_count,
            "workspace_group": self.workspace_group,
            "file_size_bytes": self.file_size_bytes,
            "file_path": self.file_path,
        }
        if self.error is not None:
            out["error"] = self.error
        return out


def _extract_text(content: Any) -> str | None:
    """Pull the first text block out of a message.content value.

    Per the verified jsonl schema, content is either:
      - a list of blocks like [{"type": "text", "text": "..."}, ...], or
      - a plain string (rare, e.g. some system rows).
    Returns the first non-empty text, or None if no text block is present.
    """
    if isinstance(content, str):
        return content.strip() or None
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


def _scan_one(jsonl_path: Path, workspace_group: str) -> ScanResult:
    """Read a single .jsonl, walk it once, return metadata.

    Designed to be called from a thread pool: NO shared state, NO mutation of
    the input file. Malformed lines are silently skipped (jsonl allows this),
    which keeps a single corrupt entry from poisoning the whole scan.
    """
    session_id = jsonl_path.stem
    file_size = jsonl_path.stat().st_size

    cwd: str | None = None
    first_user: str | None = None
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    message_count = 0
    error: str | None = None

    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                # Cheap prefix check before paying for json.loads.
                stripped = raw.lstrip()
                if not stripped or stripped[0] != "{":
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                row_type = row.get("type")
                if row_type in SKIP_TYPES:
                    continue

                ts = row.get("timestamp")
                if isinstance(ts, str):
                    if first_timestamp is None:
                        first_timestamp = ts
                    last_timestamp = ts

                if row_type in COUNT_TYPES:
                    if cwd is None:
                        row_cwd = row.get("cwd")
                        if isinstance(row_cwd, str):
                            cwd = row_cwd
                    message = row.get("message")
                    if not isinstance(message, dict):
                        continue
                    text = _extract_text(message.get("content"))
                    if text is None:
                        continue
                    message_count += 1
                    if row_type == "user" and first_user is None:
                        # First line only, trimmed to PREVIEW_MAX_CHARS.
                        first_line = text.splitlines()[0] if text else ""
                        first_user = first_line[:PREVIEW_MAX_CHARS] or None
    except OSError as exc:
        error = f"read failed: {exc}"

    return ScanResult(
        session_id=session_id,
        cwd=cwd,
        first_user=first_user,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        message_count=message_count,
        workspace_group=workspace_group,
        file_size_bytes=file_size,
        file_path=str(jsonl_path),
        error=error,
    )


def _iter_session_files(projects_dir: Path) -> Iterable[tuple[Path, str]]:
    """Yield (jsonl_path, workspace_group) under projects/.

    Workspace group = immediate subdirectory name (e.g. "C--Claude"). Skips
    subdirectories named "tool-results" because Claude Code stores raw tool
    output there, not session transcripts.
    """
    if not projects_dir.is_dir():
        return
    for entry in projects_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "tool-results":
            continue
        for jsonl in entry.glob("*.jsonl"):
            yield jsonl, entry.name


def scan(claude_home: Path, workers: int) -> list[dict[str, Any]]:
    projects_dir = claude_home / "projects"
    files = list(_iter_session_files(projects_dir))
    if not files:
        return []

    results: list[ScanResult] = []
    # Thread pool is fine: pure file I/O + json.loads (GIL released during I/O
    # and during json decode for large strings). Asyncio + aiofiles would be
    # heavier and pulls in deps we don't have.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_scan_one, path, group): path
            for path, group in files
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # pragma: no cover — defensive
                path = futures[fut]
                results.append(
                    ScanResult(
                        session_id=path.stem,
                        cwd=None,
                        first_user=None,
                        first_timestamp=None,
                        last_timestamp=None,
                        message_count=0,
                        workspace_group=path.parent.name,
                        file_size_bytes=path.stat().st_size,
                        file_path=str(path),
                        error=f"worker failed: {exc}",
                    )
                )

    # Stable, helpful order: workspace_group asc, then last_timestamp desc
    # (newest first within each group). None timestamps sort last.
    grouped: dict[str, list[ScanResult]] = {}
    for r in results:
        grouped.setdefault(r.workspace_group, []).append(r)
    ordered: list[ScanResult] = []
    for group in sorted(grouped):
        ordered.extend(
            sorted(
                grouped[group],
                key=lambda r: r.last_timestamp or r.first_timestamp or "",
                reverse=True,
            )
        )
    return [r.to_dict() for r in ordered]


def _default_claude_home() -> Path:
    # Honor $CLAUDE_HOME if set (handy for tests + cross-platform CI), then
    # fall back to ~/.claude. On Windows this is %USERPROFILE%\.claude.
    env = os.environ.get("CLAUDE_HOME")
    if env:
        return Path(env)
    return Path.home() / ".claude"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan Claude Code project history and emit session metadata."
    )
    parser.add_argument(
        "--claude-home",
        type=Path,
        default=_default_claude_home(),
        help="Path to Claude home (default: $CLAUDE_HOME or ~/.claude).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (indent=2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, max(2, (os.cpu_count() or 4))),
        help="Thread-pool size for parallel file reads (default: ~cpu_count).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any per-file read failure was logged.",
    )
    args = parser.parse_args(argv)

    if not args.claude_home.exists():
        print(
            f"error: claude home does not exist: {args.claude_home}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not (args.claude_home / "projects").is_dir():
        print(
            f"error: no projects/ under {args.claude_home}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    entries = scan(args.claude_home, args.workers)

    payload = json.dumps(
        entries,
        indent=2 if args.pretty else None,
        ensure_ascii=False,
    )

    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            f"wrote {len(entries)} session(s) to {args.output}",
            file=sys.stderr,
        )

    # Non-fatal warnings: per-file read failures, surfaced so the user can see
    # why a session is missing from the list.
    bad = [e for e in entries if e.get("error")]
    for e in bad:
        print(
            f"warn: {e['file_path']}: {e['error']}",
            file=sys.stderr,
        )
    if args.strict and bad:
        raise SystemExit(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
