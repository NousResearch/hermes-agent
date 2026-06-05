#!/usr/bin/env python3
"""Audit recalled memory-context text for likely non-durable fragments.

The audit is deliberately non-destructive. It reports candidates and writes
structured JSONL, but it does not invalidate memories itself. When a memory id
is present, ``--commands`` prints suggested ``mnemosyne_validate`` calls for a
human or agent to review.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from hermes_constants import get_hermes_home

DEFAULT_ROOT = get_hermes_home() / "ops" / "self-improvement-log"
DEFAULT_OUTPUT = DEFAULT_ROOT / "memory_context_audit.jsonl"

_ENTRY_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\[[^\]]+\]\s*)?"
    r"(?:\(importance\s+(?P<importance>[0-9.]+)\)\s*)?"
    r"(?:(?:id|memory_id)=(?P<memory_id>[A-Za-z0-9_-]+)\s*)?"
    r"(?P<content>.*\S)\s*$"
)
_COMMAND_FRAGMENT_RE = re.compile(
    r"^\[USER\]\s*(?:proceed|commit|what next|make it happen|go ahead|ok start(?: working)?(?: on the plan)?\.?|review now i ran another task|you decide.*proceed)\s*$",
    re.IGNORECASE,
)
_ONE_OFF_PROMPT_RE = re.compile(
    r"^\[USER\].*(?:let me know where|start working on that plan|final video|ran another task)",
    re.IGNORECASE,
)
_BACKGROUND_PROCESS_RE = re.compile(
    r"background process|matched watch pattern|startup done|matched output|^command:",
    re.IGNORECASE,
)
_RAW_PREFIXES = ("[USER]", "[conversation]")


@dataclass(frozen=True)
class MemoryContextEntry:
    content: str
    importance: float | None = None
    memory_id: str = ""


@dataclass(frozen=True)
class AuditCandidate:
    content: str
    reasons: list[str]
    importance: float | None = None
    memory_id: str = ""


@dataclass(frozen=True)
class AuditReport:
    schema_version: int
    kind: str
    captured_at: str
    entry_count: int
    candidate_count: int
    candidates: list[AuditCandidate]


def parse_memory_context(text: str) -> list[MemoryContextEntry]:
    """Parse recalled memory-context text into coarse line-based entries."""
    entries: list[MemoryContextEntry] = []
    current: MemoryContextEntry | None = None
    continuation_lines: list[str] = []

    def flush() -> None:
        nonlocal current, continuation_lines
        if current is None:
            return
        if continuation_lines:
            joined = "\n".join([current.content, *continuation_lines]).strip()
            entries.append(
                MemoryContextEntry(
                    content=joined,
                    importance=current.importance,
                    memory_id=current.memory_id,
                )
            )
        else:
            entries.append(current)
        current = None
        continuation_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("##") or line.startswith("<"):
            continue
        match = _ENTRY_RE.match(raw_line)
        if not match:
            continue
        content = match.group("content").strip()
        if not content:
            continue
        looks_like_new_entry = content.startswith(_RAW_PREFIXES) or "(importance " in raw_line or "id=" in raw_line
        if looks_like_new_entry:
            flush()
            importance_raw = match.group("importance")
            current = MemoryContextEntry(
                content=content,
                importance=float(importance_raw) if importance_raw else None,
                memory_id=match.group("memory_id") or "",
            )
        elif current is not None:
            continuation_lines.append(content)
        else:
            current = MemoryContextEntry(content=content)
    flush()
    return entries


def classify_entry(entry: MemoryContextEntry) -> list[str]:
    content = entry.content.strip()
    lower = content.lower()
    reasons: list[str] = []

    if content.startswith("[USER]"):
        reasons.append("raw_user_fragment")
    if content.startswith("[conversation]"):
        reasons.append("raw_conversation_fragment")
    if _COMMAND_FRAGMENT_RE.search(content):
        reasons.append("standalone_command_fragment")
    if _ONE_OFF_PROMPT_RE.search(content):
        reasons.append("one_off_task_prompt")
    if _BACKGROUND_PROCESS_RE.search(content):
        reasons.append("background_process_fragment")
    if "sleep_consolidation" in lower:
        reasons.append("sleep_consolidation_fragment")
    if entry.importance is not None and entry.importance <= 0.3 and content.startswith(_RAW_PREFIXES):
        reasons.append("low_importance_raw_fragment")

    # Stable distilled facts/preferences should not be candidates solely because
    # they mention noisy words like "proceed" inside the rule itself.
    if not content.startswith(_RAW_PREFIXES):
        return []
    return reasons


def audit_entries(entries: list[MemoryContextEntry]) -> AuditReport:
    candidates = [
        AuditCandidate(
            content=entry.content,
            reasons=classify_entry(entry),
            importance=entry.importance,
            memory_id=entry.memory_id,
        )
        for entry in entries
    ]
    candidates = [candidate for candidate in candidates if candidate.reasons]
    return AuditReport(
        schema_version=1,
        kind="memory_context_audit",
        captured_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        entry_count=len(entries),
        candidate_count=len(candidates),
        candidates=candidates,
    )


def audit_text(text: str) -> AuditReport:
    return audit_entries(parse_memory_context(text))


def append_report(path: Path, report: AuditReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True) + "\n")


def suggested_commands(report: AuditReport) -> str:
    lines: list[str] = []
    for candidate in report.candidates:
        if not candidate.memory_id:
            continue
        reason = ",".join(candidate.reasons)
        lines.append(
            "mnemosyne_validate("
            f'action="invalidate", bank="private", memory_id="{candidate.memory_id}", '
            f'note="Memory-context audit candidate: {reason}")'
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="-", help="Input text file, or '-' for stdin")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSONL audit output path")
    parser.add_argument("--commands", action="store_true", help="Print suggested invalidation commands instead of JSON summary")
    parser.add_argument("--no-write", action="store_true", help="Do not append the JSONL audit report")
    return parser.parse_args(argv)


def read_input(path: str) -> str:
    if path == "-":
        import sys

        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_text(read_input(args.input))
    if not args.no_write:
        append_report(Path(args.output), report)
    if args.commands:
        print(suggested_commands(report))
    else:
        print(
            json.dumps(
                {
                    "kind": report.kind,
                    "entry_count": report.entry_count,
                    "candidate_count": report.candidate_count,
                    "applied": False,
                    "output": "" if args.no_write else str(args.output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
