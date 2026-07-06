#!/usr/bin/env python3
"""Torben open-loop tracker."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HEADER = ["id", "item", "state", "owner", "due", "domain", "note", "created", "updated"]
VALID_STATES = {"next-action", "waiting-on", "deferred-until", "dropped", "done"}
ACTIVE_STATES = {"next-action", "waiting-on", "deferred-until"}
DEFAULT_DOMAINS = {"home", "health", "money", "admin", "gtm", "harness"}


@dataclass
class LoopRow:
    id: int
    item: str
    state: str
    owner: str
    due: str
    domain: str
    note: str
    created: str
    updated: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoopRow":
        return cls(
            id=int(payload.get("id") or 0),
            item=str(payload.get("item") or ""),
            state=str(payload.get("state") or ""),
            owner=str(payload.get("owner") or "eric"),
            due=str(payload.get("due") or ""),
            domain=str(payload.get("domain") or "admin"),
            note=str(payload.get("note") or ""),
            created=str(payload.get("created") or ""),
            updated=str(payload.get("updated") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": str(self.id),
            "item": self.item,
            "state": self.state,
            "owner": self.owner,
            "due": self.due,
            "domain": self.domain,
            "note": self.note,
            "created": self.created,
            "updated": self.updated,
        }


def _today(value: date | None = None) -> str:
    return (value or datetime.now(timezone.utc).date()).isoformat()


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def ensure_tracker(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
    os.replace(tmp, path)


def load_loops(path: Path) -> list[LoopRow]:
    ensure_tracker(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Open-loop tracker header must be {','.join(HEADER)}: {path}")
        return [LoopRow.from_dict(row) for row in reader]


def write_loops(path: Path, rows: list[LoopRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    os.replace(tmp, path)


def validate_loops(rows: list[LoopRow]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in rows:
        if row.id in seen_ids:
            invalid.append({"id": row.id, "field": "id", "reason": "duplicate_id"})
        seen_ids.add(row.id)
        if not row.item.strip():
            invalid.append({"id": row.id, "field": "item", "reason": "missing_item"})
        if row.state not in VALID_STATES:
            invalid.append({"id": row.id, "field": "state", "reason": "invalid_or_missing_state"})
        if row.due:
            try:
                _parse_date(row.due)
            except ValueError:
                invalid.append({"id": row.id, "field": "due", "reason": "invalid_date"})
        for field_name in ("created", "updated"):
            try:
                _parse_date(getattr(row, field_name))
            except ValueError:
                invalid.append({"id": row.id, "field": field_name, "reason": "invalid_date"})
    return invalid


def add_loop(
    *,
    path: Path,
    item: str,
    state: str = "next-action",
    owner: str = "eric",
    due: str = "",
    domain: str = "admin",
    note: str = "",
    today: date | None = None,
) -> LoopRow:
    rows = load_loops(path)
    if state not in VALID_STATES:
        raise ValueError(f"Invalid loop state: {state}")
    if due:
        _parse_date(due)
    day = _today(today)
    row = LoopRow(
        id=max([existing.id for existing in rows] or [0]) + 1,
        item=item,
        state=state,
        owner=owner,
        due=due,
        domain=domain,
        note=note,
        created=day,
        updated=day,
    )
    rows.append(row)
    write_loops(path, rows)
    return row


def set_loop_state(*, path: Path, loop_id: int, state: str, today: date | None = None) -> LoopRow:
    if state not in VALID_STATES:
        raise ValueError(f"Invalid loop state: {state}")
    rows = load_loops(path)
    for row in rows:
        if row.id == loop_id:
            row.state = state
            row.updated = _today(today)
            write_loops(path, rows)
            return row
    raise KeyError(f"Loop id not found: {loop_id}")


def overdue_loops(rows: list[LoopRow], *, today: date | None = None) -> list[LoopRow]:
    current = today or datetime.now(timezone.utc).date()
    result: list[LoopRow] = []
    for row in rows:
        due = _parse_date(row.due)
        if due and due <= current and row.state in ACTIVE_STATES:
            result.append(row)
    return result


def stale_waiting_loops(rows: list[LoopRow], *, today: date | None = None, stale_days: int = 7) -> list[dict[str, Any]]:
    current = today or datetime.now(timezone.utc).date()
    threshold = current - timedelta(days=stale_days)
    stale: list[dict[str, Any]] = []
    for row in rows:
        updated = _parse_date(row.updated)
        if row.state == "waiting-on" and updated and updated <= threshold:
            stale.append(
                {
                    **row.to_dict(),
                    "drafted_nudge": f"Nudge {row.owner}: still waiting on {row.item}",
                    "stale_days": stale_days,
                }
            )
    return stale


def _default_tracker_path() -> Path:
    home = os.getenv("HERMES_HOME")
    return Path(home).expanduser() / "state" / "torben-open-loops.csv" if home else Path("state/torben-open-loops.csv")


def _rows_payload(rows: list[LoopRow]) -> list[dict[str, str]]:
    return [row.to_dict() for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=None, help="Tracker CSV path")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create tracker header if needed")

    add = subparsers.add_parser("add", help="Add a loop")
    add.add_argument("item")
    add.add_argument("--state", default="next-action")
    add.add_argument("--owner", default="eric")
    add.add_argument("--due", default="")
    add.add_argument("--domain", default="admin")
    add.add_argument("--note", default="")

    set_state = subparsers.add_parser("set-state", help="Set loop state")
    set_state.add_argument("id", type=int)
    set_state.add_argument("state")

    subparsers.add_parser("list", help="List loops")
    subparsers.add_parser("validate", help="Validate loops")
    subparsers.add_parser("overdue", help="List overdue active loops")
    stale = subparsers.add_parser("stale", help="List stale waiting-on loops")
    stale.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else _default_tracker_path()
    if args.command == "init":
        ensure_tracker(path)
        payload = {"schema": "torben.open-loops.v1", "path": str(path), "header": HEADER}
    elif args.command == "add":
        row = add_loop(
            path=path,
            item=args.item,
            state=args.state,
            owner=args.owner,
            due=args.due,
            domain=args.domain,
            note=args.note,
        )
        payload = {"schema": "torben.open-loops.v1", "loop": row.to_dict()}
    elif args.command == "set-state":
        row = set_loop_state(path=path, loop_id=args.id, state=args.state)
        payload = {"schema": "torben.open-loops.v1", "loop": row.to_dict()}
    elif args.command == "list":
        payload = {"schema": "torben.open-loops.v1", "loops": _rows_payload(load_loops(path))}
    elif args.command == "validate":
        invalid = validate_loops(load_loops(path))
        payload = {"schema": "torben.open-loops.v1", "valid": not invalid, "invalid": invalid}
    elif args.command == "overdue":
        payload = {"schema": "torben.open-loops.v1", "loops": _rows_payload(overdue_loops(load_loops(path)))}
    elif args.command == "stale":
        payload = {"schema": "torben.open-loops.v1", "loops": stale_waiting_loops(load_loops(path), stale_days=args.days)}
    else:
        raise ValueError(f"Unhandled command: {args.command}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
