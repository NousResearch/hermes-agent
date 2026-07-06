#!/usr/bin/env python3
"""Torben receipts, refunds, and subscription-hunting task layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from torben_open_loops import LoopRow, add_loop, load_loops, write_loops


SCHEMA = "torben.subscription-hunt.v1"
STATE_SCHEMA = "torben.subscription-hunt-state.v1"


def _today(value: date | None = None) -> date:
    return value or datetime.now(timezone.utc).date()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value))


def _key(*parts: Any) -> str:
    material = "|".join(_clean_text(part).lower() for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "recurring": {}, "refunds": {}}
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(payload, dict):
        return {"schema": STATE_SCHEMA, "recurring": {}, "refunds": {}}
    payload.setdefault("schema", STATE_SCHEMA)
    payload.setdefault("recurring", {})
    payload.setdefault("refunds", {})
    return payload


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _update_loop_note(*, tracker_path: Path, loop_id: int, note: str, today: date) -> bool:
    rows = load_loops(tracker_path)
    for row in rows:
        if row.id == loop_id:
            row.note = note
            row.updated = today.isoformat()
            write_loops(tracker_path, rows)
            return True
    return False


def _recurring_charge_key(charge: dict[str, Any]) -> str:
    return _key(charge.get("merchant"), charge.get("amount"), charge.get("cadence") or "recurring")


def _refund_key(refund: dict[str, Any]) -> str:
    return _key(refund.get("merchant"), refund.get("amount"), refund.get("expected_by"), refund.get("source_id"))


def cancellation_packet(subscription: dict[str, Any]) -> dict[str, Any]:
    merchant = _clean_text(subscription.get("merchant") or subscription.get("name") or "subscription")
    amount = _clean_text(subscription.get("amount") or "unknown")
    cadence = _clean_text(subscription.get("cadence") or "unknown")
    return {
        "schema": SCHEMA,
        "type": "cancellation_packet",
        "category": "payment_adjacent",
        "status": "packet_only",
        "merchant": merchant,
        "amount": amount,
        "cadence": cadence,
        "reason": _clean_text(subscription.get("reason") or "review cancellable subscription"),
        "cancel_pointer": _clean_text(subscription.get("cancel_pointer") or subscription.get("url") or ""),
        "approval_request": f"Approve cancellation packet for {merchant}, or defer/drop.",
        "blocked_actions": ["no autonomous cancellation", "no payment/account secret stored"],
        "external_actions_taken": [],
    }


def _process_recurring_charges(
    *,
    charges: list[dict[str, Any]],
    tracker_path: Path,
    state: dict[str, Any],
    today: date,
    ttl_days: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    recurring_state = state.setdefault("recurring", {})
    for charge in charges:
        if not (charge.get("recurring") is True or charge.get("cadence")):
            continue
        key = _recurring_charge_key(charge)
        existing = recurring_state.get(key)
        merchant = _clean_text(charge.get("merchant") or "unknown merchant")
        amount = _clean_text(charge.get("amount") or "unknown amount")
        cadence = _clean_text(charge.get("cadence") or "recurring")
        note = f"subscription_key={key};merchant={merchant};amount={amount};cadence={cadence};last_seen={today.isoformat()}"
        if existing:
            last_alerted = _parse_day(existing.get("last_alerted"))
            loop_id = int(existing.get("loop_id") or 0)
            _update_loop_note(tracker_path=tracker_path, loop_id=loop_id, note=note, today=today)
            existing["last_seen"] = today.isoformat()
            if last_alerted and today < last_alerted + timedelta(days=ttl_days):
                events.append({"status": "deduped", "kind": "recurring_charge", "key": key, "loop_id": loop_id})
            else:
                existing["last_alerted"] = today.isoformat()
                events.append({"status": "updated", "kind": "recurring_charge", "key": key, "loop_id": loop_id})
            continue
        loop = add_loop(
            path=tracker_path,
            item=f"Review recurring charge: {merchant} {amount} ({cadence})",
            state="next-action",
            owner="eric",
            due="",
            domain="money",
            note=note,
            today=today,
        )
        recurring_state[key] = {"loop_id": loop.id, "last_alerted": today.isoformat(), "last_seen": today.isoformat()}
        events.append({"status": "created", "kind": "recurring_charge", "key": key, "loop_id": loop.id})
    return events


def _process_refunds(
    *,
    refunds: list[dict[str, Any]],
    tracker_path: Path,
    state: dict[str, Any],
    today: date,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    refund_state = state.setdefault("refunds", {})
    for refund in refunds:
        if refund.get("arrived") is True:
            continue
        expected_by = _parse_day(refund.get("expected_by"))
        if not expected_by or expected_by >= today:
            continue
        key = _refund_key(refund)
        existing = refund_state.get(key)
        if existing and existing.get("followed_up"):
            events.append({"status": "deduped", "kind": "refund_followup", "key": key, "loop_id": existing.get("loop_id")})
            continue
        merchant = _clean_text(refund.get("merchant") or "unknown merchant")
        amount = _clean_text(refund.get("amount") or "unknown amount")
        loop = add_loop(
            path=tracker_path,
            item=f"Follow up on missing refund: {merchant} {amount}",
            state="next-action",
            owner="eric",
            due=today.isoformat(),
            domain="money",
            note=f"refund_key={key};expected_by={expected_by.isoformat()}",
            today=today,
        )
        refund_state[key] = {"loop_id": loop.id, "expected_by": expected_by.isoformat(), "followed_up": True}
        events.append({"status": "created", "kind": "refund_followup", "key": key, "loop_id": loop.id})
    return events


def run_subscription_hunt(
    *,
    monarch_status: str,
    charges: list[dict[str, Any]],
    refunds: list[dict[str, Any]],
    subscriptions: list[dict[str, Any]],
    tracker_path: Path,
    state_path: Path,
    today: date | None = None,
    ttl_days: int = 30,
) -> dict[str, Any]:
    current = _today(today)
    if monarch_status != "ok":
        return {
            "schema": SCHEMA,
            "status": "source_failure",
            "reason": "p0_7_monarch_source_unavailable_or_empty",
            "monarch_status": monarch_status,
            "events": [],
            "packets": [],
        }
    state = load_state(state_path)
    events = []
    events.extend(
        _process_recurring_charges(
            charges=charges,
            tracker_path=tracker_path,
            state=state,
            today=current,
            ttl_days=ttl_days,
        )
    )
    events.extend(_process_refunds(refunds=refunds, tracker_path=tracker_path, state=state, today=current))
    packets = [cancellation_packet(item) for item in subscriptions if item.get("cancel_recommended") is True]
    write_state(state_path, state)
    return {
        "schema": SCHEMA,
        "status": "ok",
        "events": events,
        "packets": packets,
        "event_count": len(events),
        "packet_count": len(packets),
    }


def _default_state_path(name: str) -> Path:
    home = os.getenv("HERMES_HOME")
    root = Path(home).expanduser() if home else Path(".")
    return root / "state" / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--ttl-days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.input_json)
    result = run_subscription_hunt(
        monarch_status=str(payload.get("monarch_status") or "empty"),
        charges=list(payload.get("charges") or []),
        refunds=list(payload.get("refunds") or []),
        subscriptions=list(payload.get("subscriptions") or []),
        tracker_path=Path(args.tracker) if args.tracker else _default_state_path("torben-open-loops.csv"),
        state_path=Path(args.state) if args.state else _default_state_path("torben-subscription-hunt-state.json"),
        ttl_days=args.ttl_days,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
