#!/usr/bin/env python3
"""Evaluate Torben staged items under the go-dark contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from torben_autonomy_ladder import evaluate_dispatch  # noqa: E402


def _repo_root() -> Path:
    current = SCRIPT_PATH
    for parent in current.parents:
        if (parent / "hermes_cli").exists():
            return parent
        if parent.name == ".hermes" and (parent / "hermes-agent" / "hermes_cli").exists():
            return parent / "hermes-agent"
    fallback = os.getenv("HERMES_REPO_ROOT")
    if fallback:
        return Path(fallback)
    return Path("/Users/ericfreeman/.hermes/hermes-agent")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home  # noqa: E402
from hermes_cli.signal_coo.action_ledger import ActionLedger, ActionRecord, parse_time  # noqa: E402


OPEN_STATUSES = {"drafted", "staged", "approval_required", "approved", "executing"}
NEVER_GO_DARK_AUTO_CATEGORIES = {"gmail_trash", "booking", "payment_adjacent"}
DEFAULT_REPING_AFTER_HOURS = 24
DEFAULT_ACT_AFTER_HOURS = 48
DEFAULT_EXPIRES_AFTER_HOURS = 168


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _raw_records_by_handle(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    if path.suffix == ".jsonl":
        records: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("handle"):
                records[str(payload["handle"])] = payload
        return records
    payload = json.loads(path.read_text(encoding="utf-8") or "[]")
    if not isinstance(payload, list):
        return {}
    return {str(item["handle"]): item for item in payload if isinstance(item, dict) and item.get("handle")}


def _record_risk_class(record: ActionRecord, raw: dict[str, Any] | None) -> str:
    if not raw or not str(raw.get("risk_class") or "").strip():
        return "high"
    return str(raw.get("risk_class")).strip().lower()


def _record_category(record: ActionRecord) -> str:
    state = record.executor_state or {}
    return str(state.get("category") or state.get("ladder_category") or "").strip()


def _go_dark_state(record: ActionRecord) -> dict[str, Any]:
    state = record.executor_state.setdefault("go_dark", {})
    if not isinstance(state, dict):
        state = {}
        record.executor_state["go_dark"] = state
    return state


def _hours(go_dark: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(go_dark.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(value, 0)


def _created_at(record: ActionRecord) -> datetime:
    return record.created_at.astimezone(timezone.utc)


def _due_at(record: ActionRecord, *, hours: int) -> datetime:
    return _created_at(record) + timedelta(hours=hours)


def _append_history(record: ActionRecord, *, event: str, at: datetime, **extra: Any) -> None:
    payload = {"at": _iso(at), "status": event, **extra}
    record.resolution_history.append(payload)


def _write_pending_decisions(path: Path, decisions: list[dict[str, Any]]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8") or "[]")
        if isinstance(payload, list):
            existing = [item for item in payload if isinstance(item, dict)]
    by_handle = {str(item.get("handle")): item for item in existing if item.get("handle")}
    for decision in decisions:
        by_handle[str(decision["handle"])] = decision
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(list(by_handle.values()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _pending_decision(record: ActionRecord, *, risk_class: str, category: str, now: datetime, reason: str) -> dict[str, Any]:
    return {
        "schema": "torben.pending-decision.v1",
        "handle": record.handle,
        "summary": record.summary,
        "risk_class": risk_class,
        "category": category,
        "reason": reason,
        "created_at": _iso(now),
    }


def _undo_pointer(record: ActionRecord) -> str | None:
    state = record.executor_state or {}
    value = state.get("undo_pointer") or state.get("undo")
    return str(value).strip() if value else None


def _is_low_risk_go_dark_actionable(record: ActionRecord, *, risk_class: str, category: str) -> bool:
    state = record.executor_state or {}
    return (
        risk_class == "low"
        and category
        and category not in NEVER_GO_DARK_AUTO_CATEGORIES
        and state.get("go_dark_actionable") is True
    )


def _protective_action(record: ActionRecord) -> bool:
    state = record.executor_state or {}
    return state.get("protective_action") is True or str(state.get("operation") or "").lower() in {"rollback", "undo"}


def evaluate_go_dark(
    *,
    ledger_path: Path,
    pending_decisions_path: Path,
    ladder_config_path: Path | None = None,
    ladder_state_path: Path | None = None,
    scope_inventory_path: Path | None = None,
    now: datetime | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    ledger = ActionLedger(ledger_path)
    records = ledger.load()
    raw_by_handle = _raw_records_by_handle(ledger.path)
    repings: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    pending_decisions: list[dict[str, Any]] = []
    expirations: list[dict[str, Any]] = []
    protective_actions: list[dict[str, Any]] = []
    changed = False

    for record in records:
        if record.status not in OPEN_STATUSES:
            continue
        raw = raw_by_handle.get(record.handle)
        risk_class = _record_risk_class(record, raw)
        category = _record_category(record)
        go_dark = _go_dark_state(record)
        reping_after = _hours(go_dark, "reping_after_hours", DEFAULT_REPING_AFTER_HOURS)
        act_after = _hours(go_dark, "act_after_hours", DEFAULT_ACT_AFTER_HOURS)
        expires_after = _hours(go_dark, "expires_after_hours", DEFAULT_EXPIRES_AFTER_HOURS)

        if _protective_action(record) and now >= _due_at(record, hours=reping_after):
            record.status = "executed"
            _append_history(record, event="protective_action_executed", at=now, reason="protective_action_not_blocked_by_pending_input")
            protective_actions.append({"handle": record.handle, "category": category, "status": "executed"})
            changed = True
            continue

        if now >= _due_at(record, hours=expires_after):
            record.status = "expired"
            go_dark["expired_at"] = _iso(now)
            _append_history(record, event="go_dark_expired", at=now, outcome="drop_not_act")
            expirations.append({"handle": record.handle, "outcome": "drop_not_act"})
            changed = True
            continue

        if not go_dark.get("reping_sent_at") and now >= _due_at(record, hours=reping_after):
            go_dark["reping_sent_at"] = _iso(now)
            _append_history(record, event="go_dark_reping_sent", at=now)
            repings.append({"handle": record.handle, "summary": record.summary})
            changed = True
            continue

        if not go_dark.get("reping_sent_at") or now < _due_at(record, hours=act_after):
            continue

        if _is_low_risk_go_dark_actionable(record, risk_class=risk_class, category=category):
            undo_pointer = _undo_pointer(record)
            if not undo_pointer:
                pending_decisions.append(
                    _pending_decision(record, risk_class=risk_class, category=category, now=now, reason="missing_undo_pointer")
                )
                go_dark["pending_decision_recorded_at"] = _iso(now)
                changed = True
                continue
            dispatch = evaluate_dispatch(
                category=category,
                item_status="approved",
                config_path=ladder_config_path,
                state_path=ladder_state_path,
                scope_inventory_path=scope_inventory_path,
                ledger_path=ledger.path,
                now=now,
            )
            if not dispatch.get("technical_auto_execution"):
                pending_decisions.append(
                    _pending_decision(record, risk_class=risk_class, category=category, now=now, reason="ladder_does_not_permit_auto")
                )
                go_dark["pending_decision_recorded_at"] = _iso(now)
                changed = True
                continue
            record.status = "executed"
            go_dark["acted_at"] = _iso(now)
            _append_history(record, event="go_dark_policy_approved", at=now, category=category)
            _append_history(record, event="go_dark_executed", at=now, undo_pointer=undo_pointer)
            actions.append({"handle": record.handle, "category": category, "undo_pointer": undo_pointer})
            changed = True
            continue

        if not go_dark.get("pending_decision_recorded_at"):
            pending_decisions.append(
                _pending_decision(record, risk_class=risk_class, category=category, now=now, reason="high_risk_or_not_go_dark_actionable")
            )
            go_dark["pending_decision_recorded_at"] = _iso(now)
            changed = True

    if apply and changed:
        ledger.save(records)
    if apply and pending_decisions:
        _write_pending_decisions(pending_decisions_path, pending_decisions)

    return {
        "schema": "torben.go-dark.v1",
        "generated_at": _iso(now),
        "apply": apply,
        "repings": repings,
        "actions": actions,
        "pending_decisions": pending_decisions,
        "expirations": expirations,
        "protective_actions": protective_actions,
        "changed": changed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", help="Action ledger path")
    parser.add_argument("--pending-decisions", help="Pending decisions output path")
    parser.add_argument("--ladder-config", help="Autonomy ladder config path")
    parser.add_argument("--ladder-state", help="Autonomy ladder state path")
    parser.add_argument("--scope-inventory", help="P0-9 scope inventory path")
    parser.add_argument("--apply", action="store_true", help="Persist ledger/pending-decision changes")
    parser.add_argument("--now", help="Optional ISO timestamp")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    home = get_hermes_home()
    state_dir = home / "state"
    payload = evaluate_go_dark(
        ledger_path=Path(args.ledger) if args.ledger else state_dir / "torben-action-ledger.jsonl",
        pending_decisions_path=Path(args.pending_decisions) if args.pending_decisions else state_dir / "torben-pending-decisions.json",
        ladder_config_path=Path(args.ladder_config) if args.ladder_config else home / "config" / "torben-autonomy-ladder.yaml",
        ladder_state_path=Path(args.ladder_state) if args.ladder_state else state_dir / "torben-autonomy-ladder.json",
        scope_inventory_path=Path(args.scope_inventory) if args.scope_inventory else state_dir / "torben-oauth-scope-inventory.json",
        now=parse_time(args.now) if args.now else None,
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "go-dark: "
            f"repings={len(payload['repings'])} "
            f"actions={len(payload['actions'])} "
            f"pending={len(payload['pending_decisions'])} "
            f"expired={len(payload['expirations'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
