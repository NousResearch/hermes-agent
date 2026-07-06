#!/usr/bin/env python3
"""Torben v2 autonomy ladder config, state, and dispatch gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA = "torben.autonomy-ladder.v1"
CONFIG_SCHEMA = "torben.autonomy-ladder-config.v1"
RUNGS = ("packet_only", "approve_each", "auto_within_caps")
CATEGORIES = (
    "gmail_archive",
    "gmail_trash",
    "calendar_edit",
    "booking",
    "form_filing",
    "gtm_post",
    "payment_adjacent",
)
P0_9_GATED_CATEGORIES = {"gmail_archive", "gmail_trash", "calendar_edit"}
DEFAULT_EVENT_LOG = "state/torben-autonomy-ladder-events.jsonl"
DEFAULT_STATE_PATH = "state/torben-autonomy-ladder.json"
DEFAULT_CONFIG_PATH = "config/torben-autonomy-ladder.yaml"
DEFAULT_LEDGER_PATH = "state/torben-action-ledger.jsonl"
DEFAULT_SCOPE_INVENTORY_PATH = "state/torben-oauth-scope-inventory.json"


class LadderError(RuntimeError):
    """Raised when a ladder operation is invalid."""


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _day_key(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _profile_home() -> Path:
    explicit = os.getenv("HERMES_HOME")
    if explicit:
        return Path(explicit)
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return get_hermes_home()
    except Exception:
        return Path.cwd()


def resolve_profile_path(path: str | os.PathLike[str], *, home: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (home or _profile_home()) / candidate


def default_config() -> dict[str, Any]:
    categories = {
        category: {
            "initial_rung": "packet_only",
            "N_clean_required": 10,
            "max_per_run": 1,
            "max_per_day": 3,
        }
        for category in CATEGORIES
    }
    categories["gmail_trash"].update(
        {
            "N_clean_required": 50,
            "max_per_run": 5,
            "max_per_day": 25,
            "auto_allowlist_classes": [
                "expired_mfa_code",
                "expired_security_code",
                "disposable_security_notification",
            ],
        }
    )
    for category in ("booking", "form_filing", "gtm_post", "payment_adjacent"):
        categories[category]["max_per_day"] = 1
    return {
        "schema": CONFIG_SCHEMA,
        "autonomy_kill_switch": False,
        "scope_gate_decisions": {},
        "event_log_path": DEFAULT_EVENT_LOG,
        "state_path": DEFAULT_STATE_PATH,
        "categories": categories,
    }


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_config()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise LadderError(f"Ladder config must be a mapping: {path}")
    merged = default_config()
    merged.update({key: value for key, value in payload.items() if key != "categories"})
    categories = dict(merged["categories"])
    for category, category_config in (payload.get("categories") or {}).items():
        if category not in CATEGORIES:
            raise LadderError(f"Unknown ladder category in config: {category}")
        if not isinstance(category_config, dict):
            raise LadderError(f"Ladder category config must be a mapping: {category}")
        categories[category] = {**categories[category], **category_config}
    merged["categories"] = categories
    return merged


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(payload, dict):
        raise LadderError(f"JSON payload must be an object: {path}")
    return payload


def _category_config(config: Mapping[str, Any], category: str) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise LadderError(f"Unknown ladder category: {category}")
    raw = (config.get("categories") or {}).get(category) or {}
    if not isinstance(raw, dict):
        raise LadderError(f"Ladder config for {category} must be a mapping")
    return raw


def _int_config(category_config: Mapping[str, Any], key: str, default: int) -> int:
    raw = category_config.get(key)
    if raw is None and key == "N_clean_required":
        raw = category_config.get("n_clean_required")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _initial_category_state(category: str, config: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    category_config = _category_config(config, category)
    rung = str(category_config.get("initial_rung") or "packet_only")
    if rung not in RUNGS:
        raise LadderError(f"Invalid initial rung for {category}: {rung}")
    return {
        "rung": rung,
        "clean_approved_executions": 0,
        "daily_auto_counts": {},
        "promotion": {
            "status": "not_eligible",
            "manual_signal_required": True,
            "eligible_input_needed": False,
        },
        "updated_at": _iso(now),
    }


def initial_state(config: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": _iso(now),
        "categories": {category: _initial_category_state(category, config, now=now) for category in CATEGORIES},
    }


def load_state(path: Path, config: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    if path.exists():
        state = load_json(path)
    else:
        state = initial_state(config, now=now)
    categories = state.setdefault("categories", {})
    if not isinstance(categories, dict):
        raise LadderError(f"Ladder state categories must be a mapping: {path}")
    changed = False
    for category in CATEGORIES:
        if category not in categories:
            categories[category] = _initial_category_state(category, config, now=now)
            changed = True
        category_state = categories[category]
        if not isinstance(category_state, dict):
            raise LadderError(f"Ladder state for {category} must be a mapping")
        if category_state.get("rung") not in RUNGS:
            raise LadderError(f"Invalid live rung for {category}: {category_state.get('rung')}")
        category_state.setdefault("clean_approved_executions", 0)
        category_state.setdefault("daily_auto_counts", {})
        category_state.setdefault(
            "promotion",
            {
                "status": "not_eligible",
                "manual_signal_required": True,
                "eligible_input_needed": False,
            },
        )
    if changed and path.exists():
        write_json_atomic(path, state)
    return state


def is_kill_switch_enabled(config: Mapping[str, Any] | None = None, *, env: Mapping[str, str] | None = None) -> bool:
    env_map = env if env is not None else os.environ
    if _truthy(env_map.get("TORBEN_AUTONOMY_KILL")):
        return True
    return bool(config and config.get("autonomy_kill_switch") is True)


def _has_eric_scope_decision(config: Mapping[str, Any], category: str) -> bool:
    decisions = config.get("scope_gate_decisions") or {}
    if not isinstance(decisions, dict):
        return False
    decision = decisions.get(category)
    if not isinstance(decision, dict):
        return False
    if str(decision.get("decided_by") or "").strip().lower() != "eric":
        return False
    return str(decision.get("status") or "") in {
        "accepted_with_compensating_controls",
        "revoked_read_only",
        "read_only_reconsented",
    }


def scope_gate_for_category(
    category: str,
    *,
    config: Mapping[str, Any],
    scope_inventory_path: Path,
) -> dict[str, Any]:
    if category not in P0_9_GATED_CATEGORIES:
        return {"status": "clear", "reason": "category_not_p0_9_gated"}
    if _has_eric_scope_decision(config, category):
        return {"status": "clear", "reason": "eric_scope_gate_decision_recorded"}
    if not scope_inventory_path.exists():
        return {"status": "blocked_unverified", "floor": "packet_only", "reason": "scope_inventory_missing"}
    inventory = load_json(scope_inventory_path)
    gates = inventory.get("category_gates") or {}
    gate = gates.get(category) if isinstance(gates, dict) else None
    if gate is None:
        return {"status": "blocked_unverified", "floor": "packet_only", "reason": "scope_category_gate_missing"}
    if isinstance(gate, dict) and gate.get("status") == "blocked_type_1":
        return {
            "status": "blocked_type_1",
            "floor": "packet_only",
            "reason": gate.get("reason") or "p0_9_type_1_scope_gate",
        }
    return {"status": "clear", "reason": "no_blocking_type_1_scope_gate"}


def migration_safety_gate(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.exists():
        return {"status": "pass", "approved_migrated_count": 0, "reason": "ledger_missing"}
    approved_migrated: list[str] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("status") == "approved" and record.get("migrated") is True:
                approved_migrated.append(str(record.get("handle") or f"line:{line_number}"))
    return {
        "status": "pass" if not approved_migrated else "blocked",
        "approved_migrated_count": len(approved_migrated),
        "handles": approved_migrated[:25],
    }


def _state_paths(
    *,
    home: Path,
    config: Mapping[str, Any],
    state_path: Path | None = None,
    event_log_path: Path | None = None,
    scope_inventory_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Path]:
    return {
        "state": state_path or resolve_profile_path(str(config.get("state_path") or DEFAULT_STATE_PATH), home=home),
        "event_log": event_log_path
        or resolve_profile_path(str(config.get("event_log_path") or DEFAULT_EVENT_LOG), home=home),
        "scope_inventory": scope_inventory_path or resolve_profile_path(DEFAULT_SCOPE_INVENTORY_PATH, home=home),
        "ledger": ledger_path or resolve_profile_path(DEFAULT_LEDGER_PATH, home=home),
    }


def _packet_only_result(category: str, requested_count: int, *, reason: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "torben.autonomy-dispatch.v1",
        "category": category,
        "configured_rung": "unknown",
        "effective_rung": "packet_only",
        "decision": "packet_only",
        "technical_auto_execution": False,
        "allowed_auto_count": 0,
        "approval_required_count": requested_count,
        "overflow_count": requested_count,
        "reasons": [reason],
    }
    if extra:
        result.update(extra)
    return result


def evaluate_dispatch(
    *,
    category: str,
    item_status: str,
    operation_class: str | None = None,
    requested_count: int = 1,
    config_path: Path | None = None,
    state_path: Path | None = None,
    scope_inventory_path: Path | None = None,
    ledger_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    requested = max(int(requested_count), 0)
    home = _profile_home()
    config_file = config_path or resolve_profile_path(DEFAULT_CONFIG_PATH, home=home)
    try:
        config = load_config(config_file)
    except Exception as exc:
        return _packet_only_result(category, requested, reason="config_unavailable", extra={"error": str(exc)})

    if is_kill_switch_enabled(config, env=env):
        return _packet_only_result(category, requested, reason="global_kill_switch")

    try:
        paths = _state_paths(
            home=home,
            config=config,
            state_path=state_path,
            scope_inventory_path=scope_inventory_path,
            ledger_path=ledger_path,
        )
        scope_gate = scope_gate_for_category(category, config=config, scope_inventory_path=paths["scope_inventory"])
        if scope_gate.get("status") != "clear":
            return _packet_only_result(
                category,
                requested,
                reason="p0_9_type_1_scope_gate" if scope_gate.get("status") == "blocked_type_1" else "p0_9_scope_gate_unverified",
                extra={"scope_gate": scope_gate},
            )
        migration_gate = migration_safety_gate(paths["ledger"])
        if migration_gate.get("status") != "pass":
            return _packet_only_result(
                category,
                requested,
                reason="p0_4_migration_safety_gate",
                extra={"migration_gate": migration_gate},
            )
        state = load_state(paths["state"], config, now=now)
        category_config = _category_config(config, category)
        category_state = state["categories"][category]
        rung = str(category_state["rung"])
    except Exception as exc:
        return _packet_only_result(category, requested, reason="state_unavailable", extra={"error": str(exc)})

    reasons: list[str] = []
    if rung != "auto_within_caps":
        reasons.append(f"rung_{rung}")
        return {
            "schema": "torben.autonomy-dispatch.v1",
            "category": category,
            "configured_rung": rung,
            "effective_rung": rung,
            "decision": rung,
            "technical_auto_execution": False,
            "allowed_auto_count": 0,
            "approval_required_count": requested,
            "overflow_count": requested,
            "reasons": reasons,
            "scope_gate": scope_gate,
            "migration_gate": migration_gate,
        }

    if item_status != "approved":
        reasons.append(f"status_{item_status or 'missing'}")
    if category == "gmail_trash":
        allowlist = {str(item) for item in category_config.get("auto_allowlist_classes") or []}
        if operation_class not in allowlist:
            reasons.append(f"gmail_trash_class_{operation_class or 'missing'}")
    if reasons:
        return {
            "schema": "torben.autonomy-dispatch.v1",
            "category": category,
            "configured_rung": rung,
            "effective_rung": rung,
            "decision": "approve_each",
            "technical_auto_execution": False,
            "allowed_auto_count": 0,
            "approval_required_count": requested,
            "overflow_count": requested,
            "reasons": reasons,
            "scope_gate": scope_gate,
            "migration_gate": migration_gate,
        }

    max_per_run = _int_config(category_config, "max_per_run", 1)
    max_per_day = _int_config(category_config, "max_per_day", 1)
    day = _day_key(now)
    daily_counts = category_state.get("daily_auto_counts") or {}
    used_today = int(daily_counts.get(day) or 0)
    remaining_today = max(max_per_day - used_today, 0)
    allowed = min(requested, max_per_run, remaining_today)
    overflow = max(requested - allowed, 0)
    if allowed == 0:
        reasons.append("cap_exhausted")
    elif overflow:
        reasons.append("cap_overflow")
    return {
        "schema": "torben.autonomy-dispatch.v1",
        "category": category,
        "configured_rung": rung,
        "effective_rung": rung,
        "decision": "auto_within_caps" if allowed else "approve_each",
        "technical_auto_execution": allowed > 0,
        "allowed_auto_count": allowed,
        "approval_required_count": overflow,
        "overflow_count": overflow,
        "reasons": reasons,
        "caps": {
            "max_per_run": max_per_run,
            "max_per_day": max_per_day,
            "used_today": used_today,
            "remaining_today": remaining_today,
        },
        "scope_gate": scope_gate,
        "migration_gate": migration_gate,
    }


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _persist_state_and_event(
    *,
    state_path: Path,
    event_log_path: Path,
    state: Mapping[str, Any],
    event: Mapping[str, Any],
) -> None:
    write_json_atomic(state_path, state)
    _append_event(event_log_path, event)


def record_clean_execution(
    *,
    category: str,
    config_path: Path | None = None,
    state_path: Path | None = None,
    event_log_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    home = _profile_home()
    config = load_config(config_path or resolve_profile_path(DEFAULT_CONFIG_PATH, home=home))
    paths = _state_paths(home=home, config=config, state_path=state_path, event_log_path=event_log_path)
    state = load_state(paths["state"], config, now=now)
    category_config = _category_config(config, category)
    category_state = state["categories"][category]
    category_state["clean_approved_executions"] = int(category_state.get("clean_approved_executions") or 0) + 1
    needed = _int_config(category_config, "N_clean_required", 10)
    eligible = category_state["clean_approved_executions"] >= needed and category_state["rung"] != "auto_within_caps"
    category_state["promotion"] = {
        "status": "eligible_input_needed" if eligible else "not_eligible",
        "manual_signal_required": True,
        "eligible_input_needed": eligible,
        "N_clean_required": needed,
    }
    category_state["updated_at"] = _iso(now)
    event = {
        "schema": "torben.autonomy-ladder-event.v1",
        "event": "clean_approved_execution",
        "category": category,
        "count": category_state["clean_approved_executions"],
        "promotion": category_state["promotion"],
        "created_at": _iso(now),
    }
    _persist_state_and_event(state_path=paths["state"], event_log_path=paths["event_log"], state=state, event=event)
    return {"state_path": str(paths["state"]), "event_log_path": str(paths["event_log"]), **event}


def _demoted_rung(current: str, *, restore_from_trash: bool = False) -> str:
    if restore_from_trash:
        if current == "auto_within_caps":
            return "approve_each"
        return current
    index = RUNGS.index(current)
    return RUNGS[max(index - 1, 0)]


def record_error(
    *,
    category: str,
    error: str,
    config_path: Path | None = None,
    state_path: Path | None = None,
    event_log_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    home = _profile_home()
    config = load_config(config_path or resolve_profile_path(DEFAULT_CONFIG_PATH, home=home))
    paths = _state_paths(home=home, config=config, state_path=state_path, event_log_path=event_log_path)
    state = load_state(paths["state"], config, now=now)
    category_state = state["categories"][category]
    before = str(category_state["rung"])
    after = _demoted_rung(before)
    category_state["rung"] = after
    category_state["promotion"] = {
        "status": "demoted_on_error",
        "manual_signal_required": True,
        "eligible_input_needed": False,
    }
    category_state["updated_at"] = _iso(now)
    event = {
        "schema": "torben.autonomy-ladder-event.v1",
        "event": "demotion_on_error",
        "category": category,
        "from_rung": before,
        "to_rung": after,
        "error": error,
        "created_at": _iso(now),
    }
    _persist_state_and_event(state_path=paths["state"], event_log_path=paths["event_log"], state=state, event=event)
    return {"state_path": str(paths["state"]), "event_log_path": str(paths["event_log"]), **event}


def record_auto_execution(
    *,
    category: str,
    count: int,
    config_path: Path | None = None,
    state_path: Path | None = None,
    event_log_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    home = _profile_home()
    config = load_config(config_path or resolve_profile_path(DEFAULT_CONFIG_PATH, home=home))
    paths = _state_paths(home=home, config=config, state_path=state_path, event_log_path=event_log_path)
    state = load_state(paths["state"], config, now=now)
    category_state = state["categories"][category]
    day = _day_key(now)
    daily_counts = dict(category_state.get("daily_auto_counts") or {})
    daily_counts[day] = int(daily_counts.get(day) or 0) + max(int(count), 0)
    category_state["daily_auto_counts"] = daily_counts
    category_state["updated_at"] = _iso(now)
    event = {
        "schema": "torben.autonomy-ladder-event.v1",
        "event": "auto_execution_recorded",
        "category": category,
        "count": max(int(count), 0),
        "day": day,
        "day_count": daily_counts[day],
        "created_at": _iso(now),
    }
    _persist_state_and_event(state_path=paths["state"], event_log_path=paths["event_log"], state=state, event=event)
    return {"state_path": str(paths["state"]), "event_log_path": str(paths["event_log"]), **event}


def record_gmail_trash_restore(
    *,
    config_path: Path | None = None,
    state_path: Path | None = None,
    event_log_path: Path | None = None,
    source: str = "eric_restore_from_trash",
    now: datetime | None = None,
) -> dict[str, Any]:
    home = _profile_home()
    config = load_config(config_path or resolve_profile_path(DEFAULT_CONFIG_PATH, home=home))
    paths = _state_paths(home=home, config=config, state_path=state_path, event_log_path=event_log_path)
    state = load_state(paths["state"], config, now=now)
    category_state = state["categories"]["gmail_trash"]
    before = str(category_state["rung"])
    after = _demoted_rung(before, restore_from_trash=True)
    category_state["rung"] = after
    category_state["promotion"] = {
        "status": "demoted_on_restore" if before != after else "restore_seen_no_rung_change",
        "manual_signal_required": True,
        "eligible_input_needed": False,
    }
    category_state["updated_at"] = _iso(now)
    event = {
        "schema": "torben.autonomy-ladder-event.v1",
        "event": "gmail_trash_restore_demote",
        "category": "gmail_trash",
        "from_rung": before,
        "to_rung": after,
        "source": source,
        "created_at": _iso(now),
    }
    _persist_state_and_event(state_path=paths["state"], event_log_path=paths["event_log"], state=state, event=event)
    return {"state_path": str(paths["state"]), "event_log_path": str(paths["event_log"]), **event}


def init_state(
    *,
    config_path: Path | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    home = _profile_home()
    config = load_config(config_path or resolve_profile_path(DEFAULT_CONFIG_PATH, home=home))
    paths = _state_paths(home=home, config=config, state_path=state_path)
    state = load_state(paths["state"], config, now=now)
    write_json_atomic(paths["state"], state)
    return {"state_path": str(paths["state"]), **state}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Ladder config path")
    parser.add_argument("--state", help="Ladder state path")
    parser.add_argument("--events", help="Ladder event log path")
    parser.add_argument("--scope-inventory", help="P0-9 scope inventory path")
    parser.add_argument("--ledger", help="Action ledger path")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize or reconcile ladder state")
    dispatch = subparsers.add_parser("dispatch", help="Evaluate a category dispatch")
    dispatch.add_argument("category", choices=CATEGORIES)
    dispatch.add_argument("--status", default="approval_required")
    dispatch.add_argument("--operation-class")
    dispatch.add_argument("--count", type=int, default=1)

    clean = subparsers.add_parser("record-clean", help="Record one clean approved execution")
    clean.add_argument("category", choices=CATEGORIES)
    error = subparsers.add_parser("record-error", help="Record an execution error and demote")
    error.add_argument("category", choices=CATEGORIES)
    error.add_argument("--error", default="unspecified_error")
    auto = subparsers.add_parser("record-auto", help="Record auto-execution usage")
    auto.add_argument("category", choices=CATEGORIES)
    auto.add_argument("--count", type=int, default=1)
    restore = subparsers.add_parser("record-trash-restore", help="Record Eric restoring a trashed Gmail item")
    restore.add_argument("--source", default="eric_restore_from_trash")
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else None
    state_path = Path(args.state) if args.state else None
    event_log_path = Path(args.events) if args.events else None
    scope_inventory_path = Path(args.scope_inventory) if args.scope_inventory else None
    ledger_path = Path(args.ledger) if args.ledger else None

    if args.command == "init":
        payload = init_state(config_path=config_path, state_path=state_path)
    elif args.command == "dispatch":
        payload = evaluate_dispatch(
            category=args.category,
            item_status=args.status,
            operation_class=args.operation_class,
            requested_count=args.count,
            config_path=config_path,
            state_path=state_path,
            scope_inventory_path=scope_inventory_path,
            ledger_path=ledger_path,
        )
    elif args.command == "record-clean":
        payload = record_clean_execution(
            category=args.category,
            config_path=config_path,
            state_path=state_path,
            event_log_path=event_log_path,
        )
    elif args.command == "record-error":
        payload = record_error(
            category=args.category,
            error=args.error,
            config_path=config_path,
            state_path=state_path,
            event_log_path=event_log_path,
        )
    elif args.command == "record-auto":
        payload = record_auto_execution(
            category=args.category,
            count=args.count,
            config_path=config_path,
            state_path=state_path,
            event_log_path=event_log_path,
        )
    elif args.command == "record-trash-restore":
        payload = record_gmail_trash_restore(
            config_path=config_path,
            state_path=state_path,
            event_log_path=event_log_path,
            source=args.source,
        )
    else:
        raise LadderError(f"Unhandled command: {args.command}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
