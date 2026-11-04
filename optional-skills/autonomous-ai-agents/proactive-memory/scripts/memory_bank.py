#!/usr/bin/env python3
"""A compact status/knowledge/procedural memory bank with a deterministic relevance gate.

    python memory_bank.py record BANK.jsonl --kind procedural --text "..." --trigger "npm install,proxy" [--supersedes ID]
    python memory_bank.py check  BANK.jsonl --action "the next command or step I am about to take"
    python memory_bank.py list   BANK.jsonl [--kind status|knowledge|procedural]

The bank fights *behavioral state decay*: it stays small (capped per kind, newest wins) and every
entry carries `triggers` — the tokens that make it decision-relevant. `check` returns only the
entries whose trigger is present in the proposed action, so a reminder is surfaced when it would
change the next action and stays silent otherwise. That selective gate is the whole point: it is
deterministic and testable, and it beats blindly re-injecting the entire bank ("always inject").

An entry with no trigger is rejected — a reminder must be grounded in something concrete, never
generic advice. Nothing here touches a live prompt: the agent calls `check` at a boundary it
already controls (before a committing action, at a cron-run start, at session resume) and acts on
the result. Exit codes: 0 ok, 2 invalid input.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("status", "knowledge", "procedural")
# Reminders lead in the order most likely to change the next action: procedural
# ("don't repeat this failure") first, then current status, then background knowledge.
# Independent of KINDS' declaration order, which is only the validation set.
KIND_ORDER = {"procedural": 0, "status": 1, "knowledge": 2}
DEFAULT_CAP = 8  # active entries kept per kind; a bank that grows unbounded has already decayed.
_WS = re.compile(r"\s+")


class BankError(ValueError):
    pass


def _normalize(text):
    return _WS.sub(" ", str(text).strip().lower())


def _new_id(existing):
    n = len(existing) + 1
    while any(e["id"] == f"m{n}" for e in existing):
        n += 1
    return f"m{n}"


def load(path):
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def save(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_triggers(raw):
    triggers = [_normalize(t) for t in (raw or "").split(",")]
    triggers = [t for t in triggers if t]
    # Preserve order, drop duplicates.
    seen, unique = set(), []
    for trigger in triggers:
        if trigger not in seen:
            seen.add(trigger)
            unique.append(trigger)
    return unique


def record(entries, *, kind, text, triggers, supersedes=None, cap=DEFAULT_CAP, now=None):
    if kind not in KINDS:
        raise BankError(f"kind must be one of {KINDS}")
    if not isinstance(text, str) or not text.strip():
        raise BankError("text must be a non-empty string")
    if not triggers:
        raise BankError("an entry needs at least one --trigger: a reminder must be grounded, not generic")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise BankError("cap must be a positive integer")

    if supersedes is not None:
        target = next((e for e in entries if e["id"] == supersedes), None)
        if target is None:
            raise BankError(f"--supersedes names an unknown entry id: {supersedes!r}")
        target["active"] = False

    normalized = _normalize(text)
    existing = next((e for e in entries if e["active"] and e["kind"] == kind
                     and _normalize(e["text"]) == normalized), None)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    if existing is not None:
        # Same fact restated: refresh it and merge triggers rather than duplicate.
        existing["at"] = stamp
        for trigger in triggers:
            if trigger not in existing["triggers"]:
                existing["triggers"].append(trigger)
        entry = existing
    else:
        entry = {"id": _new_id(entries), "kind": kind, "text": text.strip(),
                 "triggers": triggers, "at": stamp, "active": True}
        entries.append(entry)

    # Keep the bank compact: retire the oldest active entries of this kind beyond the cap.
    active = [e for e in entries if e["active"] and e["kind"] == kind]
    for stale in active[:-cap]:
        stale["active"] = False
    return entry


def check(entries, action):
    if not isinstance(action, str) or not action.strip():
        raise BankError("--action must be a non-empty string")
    haystack = _normalize(action)
    matched = []
    for entry in entries:
        if not entry["active"]:
            continue
        hits = [t for t in entry["triggers"] if t in haystack]
        if hits:
            matched.append({"id": entry["id"], "kind": entry["kind"], "text": entry["text"],
                            "matched_triggers": hits})
    # Deterministic: procedural first, then by the order entries were added (id number).
    matched.sort(key=lambda m: (KIND_ORDER[m["kind"]], int(m["id"][1:])))
    return matched


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact proactive memory bank with a relevance gate.")
    sub = parser.add_subparsers(dest="mode", required=True)

    up = sub.add_parser("record", help="add or refresh an entry (newest wins, bank stays capped)")
    up.add_argument("bank", type=Path)
    up.add_argument("--kind", required=True, choices=KINDS)
    up.add_argument("--text", required=True)
    up.add_argument("--trigger", required=True, help="comma-separated decision-relevant keys")
    up.add_argument("--supersedes", help="id of an entry this one replaces")
    up.add_argument("--cap", type=int, default=DEFAULT_CAP)

    ck = sub.add_parser("check", help="return only the reminders relevant to a proposed action")
    ck.add_argument("bank", type=Path)
    ck.add_argument("--action", required=True)

    ls = sub.add_parser("list", help="list active entries")
    ls.add_argument("bank", type=Path)
    ls.add_argument("--kind", choices=KINDS)

    args = parser.parse_args(argv)
    try:
        entries = load(args.bank)
        if args.mode == "record":
            triggers = _parse_triggers(args.trigger)
            entry = record(entries, kind=args.kind, text=args.text, triggers=triggers,
                           supersedes=args.supersedes, cap=args.cap)
            save(args.bank, entries)
            active = sum(1 for e in entries if e["active"])
            print(json.dumps({"updated": entry["id"], "kind": entry["kind"],
                              "active_entries": active}, ensure_ascii=False))
        elif args.mode == "check":
            matched = check(entries, args.action)
            print(json.dumps({"action": args.action, "reminders": matched,
                              "count": len(matched)}, ensure_ascii=False, indent=2))
        else:
            active = [e for e in entries if e["active"] and (not args.kind or e["kind"] == args.kind)]
            active.sort(key=lambda e: (KIND_ORDER[e["kind"]], int(e["id"][1:])))
            print(json.dumps({"entries": active, "count": len(active)},
                             ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError, BankError) as exc:
        print(f"memory_bank: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
