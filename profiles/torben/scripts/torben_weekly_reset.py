#!/usr/bin/env python3
"""Torben weekly reset packet over open loops and pending decisions."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from torben_open_loops import ACTIVE_STATES, VALID_STATES, LoopRow, load_loops, overdue_loops

SECTION_ORDER = [
    "MUST NOT DROP",
    "SCHEDULE FLAGS",
    "PAPERWORK & ADMIN",
    "MESSAGES TO DRAFT",
    "WAITING ON",
    "PENDING DECISIONS",
    "ONE THING FOR THE WEEKEND",
    "STATE FLAGS",
]


def _today(value: date | None = None) -> date:
    return value or datetime.now(timezone.utc).date()


def load_pending_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8") or "[]")
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def load_pattern_proposals(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(payload, dict):
        return []
    try:
        from torben_pattern_miner import proposals_for_weekly_reset
    except Exception:
        return []
    return proposals_for_weekly_reset(payload)


def _loop_item(row: LoopRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "item": row.item,
        "state": row.state,
        "owner": row.owner,
        "due": row.due,
        "domain": row.domain,
        "note": row.note,
    }


def age_loops(rows: list[LoopRow]) -> dict[str, Any]:
    stateless = [_loop_item(row) for row in rows if row.state not in VALID_STATES]
    active = [_loop_item(row) for row in rows if row.state in ACTIVE_STATES]
    terminal = [_loop_item(row) for row in rows if row.state in {"done", "dropped"}]
    return {
        "active_count": len(active),
        "terminal_count": len(terminal),
        "stateless_flags": stateless,
    }


def build_weekly_packet(
    *,
    loops: list[LoopRow],
    pending_decisions: list[dict[str, Any]],
    pattern_proposals: list[dict[str, Any]] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    current = _today(today)
    aging = age_loops(loops)
    overdue = overdue_loops(loops, today=current)
    active = [row for row in loops if row.state in ACTIVE_STATES]
    waiting = [row for row in active if row.state == "waiting-on"]
    admin = [row for row in active if row.domain in {"admin", "money", "health"}]
    draftable = [row for row in active if row.domain in {"gtm", "harness"} or "draft" in row.item.lower()]
    weekend = [row for row in active if row.domain in {"home", "health"}]

    sections: dict[str, list[dict[str, Any]]] = {
        "MUST NOT DROP": [_loop_item(row) for row in (overdue + active)[:5]],
        "SCHEDULE FLAGS": [_loop_item(row) for row in overdue],
        "PAPERWORK & ADMIN": [_loop_item(row) for row in admin],
        "MESSAGES TO DRAFT": [_loop_item(row) for row in draftable],
        "WAITING ON": [_loop_item(row) for row in waiting],
        "PENDING DECISIONS": pending_decisions + list(pattern_proposals or []),
        "ONE THING FOR THE WEEKEND": [_loop_item(row) for row in weekend[:1]],
        "STATE FLAGS": aging["stateless_flags"],
    }
    return {
        "schema": "torben.weekly-reset.v1",
        "generated_for": current.isoformat(),
        "sections": sections,
        "aging": aging,
    }


def render_packet(packet: dict[str, Any]) -> str:
    lines = [f"Torben weekly reset - {packet['generated_for']}"]
    sections = packet.get("sections") or {}
    for section in SECTION_ORDER:
        lines.append("")
        lines.append(section)
        items = sections.get(section) or []
        if not items:
            lines.append("- none")
            continue
        for item in items:
            label = item.get("item") or item.get("summary") or item.get("handle") or "item"
            suffix = f" [{item.get('state')}]" if item.get("state") else ""
            lines.append(f"- {label}{suffix}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops", required=True)
    parser.add_argument("--pending-decisions", required=True)
    parser.add_argument("--pattern-proposals")
    parser.add_argument("--output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    packet = build_weekly_packet(
        loops=load_loops(Path(args.loops)),
        pending_decisions=load_pending_decisions(Path(args.pending_decisions)),
        pattern_proposals=load_pattern_proposals(Path(args.pattern_proposals)) if args.pattern_proposals else [],
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(render_packet(packet), encoding="utf-8")
    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    else:
        print(render_packet(packet), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
