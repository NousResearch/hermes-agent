"""``hermes approvals audit`` — read the append-only approval audit sink.

The sink itself lives in ``tools/approval_audit.py`` (one SQLite partition per UTC day
under the store root's ``audit/`` directory). This module is only the query surface.

Querying never fails the caller's tool call: a missing, empty or unreadable store
renders as an empty table (with the reason on stderr), so a console wrapper can always
print something rather than raising out of a read.
"""

from __future__ import annotations

import json
import sys
from typing import Any

# Substrings identifying a classification key that describes an attempt to reach a
# protected file (Hermes home/config/env, SSH + shell-rc, cloud instance credentials).
# Deliberately over-inclusive: for a security query a false positive is a row you skim
# past, a false negative is an attempt nobody ever saw.
_PROTECTED_CLASS_PATTERNS = (
    "%secret%",
    "%credential%",
    "%protected file%",
    "%Hermes config%",
    "%system config%",
    "%shell-rc%",
    "%metadata endpoint%",
    "%.ssh%",
    "%vault%",
    "%instruction file%",
)

_DECISION_ALIASES = {"ask": "prompt", "human": "prompt", "yes": "allow", "no": "deny"}
_VALID_DECISIONS = ("allow", "deny", "prompt")

_COLUMNS = (
    ("time (UTC)", lambda r: str(r["ts"])[:19].replace("T", " ")),
    ("decision", lambda r: str(r["decision"])),
    ("outcome", lambda r: str(r["outcome"])),
    ("surface", lambda r: str(r["surface"])),
    ("classification", lambda r: str(r["class_key"])),
    ("target", lambda r: str(r["target_digest"])[:12]),
    ("trace", lambda r: str(r["trace_id"])),
)


def approvals_audit_command(args: Any) -> int:
    """Dispatch ``hermes approvals audit``. Returns a shell exit code."""
    from tools.approval_audit import audit_summary, iter_events, list_partitions, verify_partitions

    if getattr(args, "verify", False):
        return _render_verify(verify_partitions())

    decision = getattr(args, "decision", "") or ""
    if decision:
        decision = _DECISION_ALIASES.get(decision, decision)
        if decision not in _VALID_DECISIONS:
            print(f"error: --decision must be one of {', '.join(_VALID_DECISIONS)} "
                  f"(got {decision!r})", file=sys.stderr)
            return 2

    class_like: list[str] = []
    if getattr(args, "protected", False):
        class_like.extend(_PROTECTED_CLASS_PATTERNS)
    if getattr(args, "class_like", ""):
        class_like.append(args.class_like)

    try:
        rows = list(iter_events(
            decision=decision,
            surface=getattr(args, "surface", "") or "",
            trace_id=getattr(args, "trace_id", "") or "",
            since_days=getattr(args, "days", 0) or 0,
            class_like=class_like or "",
            limit=max(int(getattr(args, "limit", 0) or 0), 0),
        ))
    except OSError as exc:  # unreadable store must not blow up a console wrapper
        print(f"error: approval audit store unreadable: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    if not rows:
        print("No approval audit events match.")
        if not list_partitions():
            print("(the store is empty: no audit partition has been written yet)")
        return 0

    _render_table(rows)
    print(f"\n{len(rows)} event(s)")
    if getattr(args, "summary", False):
        _render_summary(audit_summary(since_days=getattr(args, "days", 0) or 0))
    return 0


def _render_table(rows: list[dict]) -> None:
    # class_key carries a full sentence; untruncated it wraps every row past the
    # terminal width. --json is the escape hatch for the whole value.
    def trimmed(getter):
        return lambda row: getter(row)[:70]

    columns = [(header, trimmed(getter)) if header == "classification" else (header, getter)
               for header, getter in _COLUMNS]
    widths = [
        max([len(header)] + [len(getter(row)) for row in rows])
        for header, getter in columns
    ]

    def line(values: list[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths)).rstrip()

    print(line([header for header, _ in columns]))
    print(line(["-" * width for width in widths]))
    for row in rows:
        print(line([getter(row) for _, getter in columns]))


def _render_summary(summary: dict) -> None:
    print("\nsummary:")
    print(f"  store: {summary.get('store', '')}")
    print(f"  rows: {summary.get('rows', 0)}")
    print(f"  by decision: {_fmt_counts(summary.get('by_decision', {}))}")
    print(f"  by surface: {_fmt_counts(summary.get('by_surface', {}))}")
    print(f"  by class: {_fmt_counts(summary.get('by_class_key', {}))}")
    print(f"  enabled: {summary.get('enabled')}  retention_days: {summary.get('retention_days')}")


def _fmt_counts(counts: dict) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _render_verify(report: dict) -> int:
    bad = 0
    for partition in sorted(report):
        status = report[partition]
        if status != "ok":
            bad += 1
        print(f"{partition}: {status}")
    if not report:
        print("no audit partitions found")
    print(f"\n{len(report)} partition(s), {bad} not ok")
    return 0 if bad == 0 else 1
