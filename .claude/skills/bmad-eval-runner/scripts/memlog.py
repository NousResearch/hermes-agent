#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""memlog -- an append-only memory log: LLM-optimal working memory for a skill.

A memlog is the dense, chronological record of everything that mattered in a piece of
work -- every decision, direction, assumption, gap, note, and event as it happened --
kept minimal like human memory: only what is important, never bloated. It persists
ACROSS sessions, so a fresh session can load it once and continue. It is NOT a
deliverable; downstream artifacts (a brief, a PRD, a report) are derived from it on
demand.

It is a FLAT log: there are no sections or grouping. Every entry is one line, recorded
at the END in the order it happened. The chronology itself is the structure.

Two invariants make it trustworthy:

  1. Append-only, chronological. Entries land at the end, in the order they happen.
     Nothing is ever inserted backward, reordered, edited, or removed. There is no
     edit or delete subcommand by design; history is never rewritten.
  2. Write-only / blind. Every command is an atomic, context-free write and echoes the
     new state as one line of JSON, so the caller never re-reads the file mid-session.
     The one time the file is read is on resume, and the caller reads it itself, not
     via this script.

Atomicity: every write goes to a temp file, is flushed and fsync'd, then atomically
renamed over the target, so a crash never leaves a half-written entry.

The file shape (.memlog.md):

    ---
    subject: Onboarding flow for a budgeting app
    status: active
    updated: 2026-06-06T14:22
    ---

    - (note) user picked the lean draft path
    - (decision) lead with one pre-categorized account; defer multi-account import
    - (direction) optimize for the anxious first-timer, not the power user
    - (assumption) open-banking consent is available in the target market
    - (gap) no data yet on week-1 retention baseline
    - (event) ran baseline eval mode

Each entry carries a typed tag drawn from a fixed vocabulary so the chronology stays
machine-scannable: decision, direction, assumption, gap, note, event.

Commands:
  init         --path FILE [--field k=v ...]                create the memlog (errors if it exists)
  append       --path FILE --type T --text STR             append one typed entry at the end
  set-complete --path FILE                                 flip frontmatter status to complete

The path is the memlog file itself (conventionally {run-folder}/.memlog.md).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ENTRY_TYPES = ("decision", "direction", "assumption", "gap", "note", "event")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M")


def split(text: str) -> tuple[dict, str]:
    """Return (frontmatter dict in source order, body str). Frontmatter is plain key: value.

    The closing fence is the first line that is *exactly* `---`, so a `---` inside a
    field value (subject is free user text) never truncates the frontmatter.
    """
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(".memlog.md has no frontmatter")
    end = next((i for i in range(1, len(lines)) if lines[i] == "---"), None)
    if end is None:
        raise ValueError(".memlog.md frontmatter is not terminated")
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def render(meta: dict, body: str) -> str:
    # Neutralize newlines in values so a multi-line field can't break the fence on re-read.
    fm = "\n".join(f"{k}: {' '.join(str(v).splitlines())}" for k, v in meta.items())
    return "---\n" + fm + "\n---\n\n" + body.rstrip("\n") + "\n"


def touch(meta: dict) -> None:
    """Stamp `updated` and keep it last so the field order stays predictable."""
    meta.pop("updated", None)
    meta["updated"] = now()


def write_atomic(path: Path, text: str) -> None:
    """Temp + flush + fsync + atomic rename, so a crash never half-writes an entry."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def entry_count(body: str) -> int:
    return sum(1 for ln in body.splitlines() if ln.startswith("- "))


def ack(path: Path, meta: dict, body: str, entry_type: str = "") -> None:
    """Echo new state so the caller never re-reads the file to know where it stands."""
    out = {
        "ok": True,
        "memlog": str(path),
        "status": meta.get("status", ""),
        "n": entry_count(body),
    }
    if entry_type:
        out["type"] = entry_type
    print(json.dumps(out))


def cmd_init(args) -> int:
    path = Path(args.path)
    if path.exists():
        print(f"error: {path} already exists; use append/set-complete to update it", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, str] = {}
    for pair in args.field or []:
        if "=" not in pair:
            print(f"error: --field expects key=value, got {pair!r}", file=sys.stderr)
            return 2
        k, v = pair.split("=", 1)
        meta[k.strip()] = v.strip()
    meta.setdefault("status", "active")
    touch(meta)
    write_atomic(path, render(meta, ""))
    ack(path, meta, "")
    return 0


def cmd_append(args) -> int:
    path = Path(args.path)
    if args.type not in ENTRY_TYPES:
        print(f"error: --type must be one of {', '.join(ENTRY_TYPES)}; got {args.type!r}", file=sys.stderr)
        return 2
    meta, body = split(path.read_text(encoding="utf-8"))
    text = " ".join(args.text.split())  # collapse newlines/runs -> one-line entry
    entry = f"- ({args.type}) {text}"
    body = (body.rstrip("\n") + "\n" + entry) if body.strip() else entry  # always at the end
    touch(meta)
    write_atomic(path, render(meta, body))
    ack(path, meta, body, args.type)
    return 0


def cmd_set_complete(args) -> int:
    path = Path(args.path)
    meta, body = split(path.read_text(encoding="utf-8"))
    meta["status"] = "complete"
    touch(meta)
    write_atomic(path, render(meta, body))
    ack(path, meta, body)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="create the memlog")
    pi.add_argument("--path", required=True, help="memlog file path (e.g. {run-folder}/.memlog.md)")
    pi.add_argument("--field", action="append", metavar="KEY=VALUE", help="frontmatter field (repeatable)")
    pi.set_defaults(func=cmd_init)

    pa = sub.add_parser("append", help="append one typed entry at the end")
    pa.add_argument("--path", required=True)
    pa.add_argument("--type", required=True, choices=ENTRY_TYPES, help="entry kind")
    pa.add_argument("--text", required=True)
    pa.set_defaults(func=cmd_append)

    pc = sub.add_parser("set-complete", help="flip frontmatter status to complete")
    pc.add_argument("--path", required=True)
    pc.set_defaults(func=cmd_set_complete)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
