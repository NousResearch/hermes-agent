#!/usr/bin/env python3
"""Manual dry-run runner for Hermes anticipation loops."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# Let the script run directly via `python scripts/anticipation_run.py ...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.anticipation import AnticipationRuntimeConfig, parse_bool, parse_loop_config
from agent.anticipation_log import append_decision_log, read_recent_decision_logs
from agent.anticipation_loops import build_router_monitor_candidates, build_stale_task_candidates
from agent.anticipation_policy import AnticipationDecision, AnticipationDecisionHistory, decide_anticipation_action
from hermes_cli.config import load_config
from hermes_state import SessionDB


def format_dry_run_decisions(
    decisions: Sequence[AnticipationDecision],
    *,
    empty_message: str = "No stale-task candidates found.",
) -> str:
    """Format dry-run decisions for terminal output without delivering anything."""

    lines = ["# Anticipation DRY RUN", "", "No messages were delivered.", ""]
    if not decisions:
        lines.append(empty_message)
        return "\n".join(lines)

    for index, decision in enumerate(decisions, start=1):
        candidate = decision.candidate
        body = (
            candidate.body
            if decision.action != "skip"
            else "[candidate body hidden because decision was skipped]"
        )
        lines.extend(
            [
                f"## {index}. {candidate.title}",
                f"Action: {decision.action}",
                f"Reason: {decision.reason}",
                f"Confidence: {candidate.confidence:.2f}",
                "",
                body,
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def history_from_decision_logs(
    logs: Sequence[dict],
    candidates: Sequence,
    *,
    now: datetime | None = None,
) -> AnticipationDecisionHistory:
    """Hydrate policy history from sanitized decision logs for current candidates.

    Logs intentionally store only dedupe hashes, not raw dedupe keys. To make
    dedupe work without weakening log privacy, map hashes back only for the
    candidates generated in this dry-run.
    """

    now = now or datetime.now(timezone.utc)
    hash_to_key = {
        hashlib.sha256(candidate.dedupe_key.encode("utf-8")).hexdigest(): candidate.dedupe_key
        for candidate in candidates
    }
    recent_dedupe_keys: dict[str, datetime] = {}
    notifications_today = 0
    last_notification_at: datetime | None = None

    for record in logs:
        ts = _parse_log_timestamp(record.get("ts"))
        if ts is None:
            continue
        dedupe_key = hash_to_key.get(str(record.get("dedupe_key_hash") or ""))
        if dedupe_key:
            prior = recent_dedupe_keys.get(dedupe_key)
            if prior is None or ts > prior:
                recent_dedupe_keys[dedupe_key] = ts
        if _is_notification_action(str(record.get("action") or "")):
            if ts.date() == now.date():
                notifications_today += 1
            if last_notification_at is None or ts > last_notification_at:
                last_notification_at = ts

    return AnticipationDecisionHistory(
        recent_dedupe_keys=recent_dedupe_keys,
        notifications_today=notifications_today,
        last_notification_at=last_notification_at,
    )


def _runtime_config_for_loop(loop_name: str) -> AnticipationRuntimeConfig:
    config = load_config()
    anticipation = config.get("anticipation", {}) if isinstance(config, dict) else {}
    loops = anticipation.get("loops", {}) if isinstance(anticipation, dict) else {}
    loop_config = parse_loop_config(loop_name, loops.get(loop_name, {}))
    quiet_hours = anticipation.get("quiet_hours", {}) if isinstance(anticipation, dict) else {}
    budget = anticipation.get("notification_budget", {}) if isinstance(anticipation, dict) else {}

    return AnticipationRuntimeConfig(
        enabled=parse_bool(anticipation.get("enabled", False)),
        loop_enabled=loop_config.enabled,
        loop_permission=loop_config.permission,
        min_confidence=loop_config.min_confidence,
        quiet_hours_enabled=parse_bool(quiet_hours.get("enabled", False)),
        quiet_hours_start=str(quiet_hours.get("start", "22:00")),
        quiet_hours_end=str(quiet_hours.get("end", "08:00")),
        max_per_day=int(budget.get("max_per_day", 3)),
        min_minutes_between=int(budget.get("min_minutes_between", 120)),
    )


def run_stale_task_resurfacer_dry_run(*, db=None, limit: int = 5, lookback_days: int | None = None) -> list[AnticipationDecision]:
    """Generate, gate, and audit stale-task decisions without delivery."""

    config = load_config()
    anticipation = config.get("anticipation", {}) if isinstance(config, dict) else {}
    loops = anticipation.get("loops", {}) if isinstance(anticipation, dict) else {}
    loop_config = parse_loop_config("stale_task_resurfacer", loops.get("stale_task_resurfacer", {}))
    runtime = _runtime_config_for_loop("stale_task_resurfacer")

    db = db or SessionDB()
    candidates = build_stale_task_candidates(
        db,
        lookback_days=lookback_days or loop_config.lookback_days,
        limit=limit,
    )
    history = history_from_decision_logs(read_recent_decision_logs(limit=200), candidates)
    decisions = [
        decide_anticipation_action(candidate, runtime, history, now=candidate.created_at)
        for candidate in candidates
    ]
    for decision in decisions:
        append_decision_log(decision)
    return decisions


def run_router_monitor_dry_run(
    *,
    snapshot: dict | None = None,
    snapshot_path: str | Path | None = None,
    limit: int = 5,
    now: datetime | None = None,
) -> list[AnticipationDecision]:
    """Generate, gate, and audit router-monitor decisions without delivery."""

    if snapshot is None:
        snapshot = _load_router_snapshot(snapshot_path)
    now = now or datetime.now(timezone.utc)
    runtime = _runtime_config_for_loop("router_monitor")
    candidates = build_router_monitor_candidates(snapshot, now=now, limit=limit)
    history = history_from_decision_logs(read_recent_decision_logs(limit=200), candidates, now=now)
    decisions = [
        decide_anticipation_action(candidate, runtime, history, now=now)
        for candidate in candidates
    ]
    for decision in decisions:
        append_decision_log(decision)
    return decisions


def _load_router_snapshot(snapshot_path: str | Path | None) -> dict:
    if snapshot_path is None:
        return {"monitoring": {}, "unknown_devices": []}
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Router snapshot JSON must be an object")
    return loaded


def _emit_decisions(decisions: Sequence[AnticipationDecision], *, json_output: bool, empty_message: str) -> None:
    if json_output:
        payload = [
            {
                "action": decision.action,
                "reason": decision.reason,
                "title": decision.candidate.title,
                "confidence": decision.candidate.confidence,
            }
            for decision in decisions
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(format_dry_run_decisions(decisions, empty_message=empty_message))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Hermes anticipation loops manually.")
    subparsers = parser.add_subparsers(dest="loop")

    stale = subparsers.add_parser("stale_task_resurfacer", help="Dry-run stale task resurfacing.")
    stale.add_argument("--dry-run", action="store_true", help="Required for V0; no delivery is supported.")
    stale.add_argument("--limit", type=int, default=5)
    stale.add_argument("--lookback-days", type=int, default=None)
    stale.add_argument("--json", action="store_true", help="Emit machine-readable decisions.")

    router = subparsers.add_parser("router_monitor", help="Dry-run router monitoring anticipation.")
    router.add_argument("--dry-run", action="store_true", help="Required for V0; no delivery is supported.")
    router.add_argument("--limit", type=int, default=5)
    router.add_argument("--snapshot", type=str, default=None, help="Optional sanitized router snapshot JSON file.")
    router.add_argument("--json", action="store_true", help="Emit machine-readable decisions.")

    logs = subparsers.add_parser("log", help="Show recent anticipation decision logs.")
    logs.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(argv)
    if args.loop == "log":
        print(json.dumps(read_recent_decision_logs(limit=args.limit), indent=2))
        return 0

    if args.loop == "router_monitor":
        if not args.dry_run:
            print("V0 only supports --dry-run; no delivery was attempted.", file=sys.stderr)
            return 2
        try:
            decisions = run_router_monitor_dry_run(limit=args.limit, snapshot_path=args.snapshot)
        except (OSError, ValueError) as exc:
            print(f"Unable to read router snapshot: {exc}", file=sys.stderr)
            return 2
        _emit_decisions(decisions, json_output=args.json, empty_message="No router-monitor candidates found.")
        return 0

    if args.loop != "stale_task_resurfacer":
        parser.print_help()
        return 2
    if not args.dry_run:
        print("V0 only supports --dry-run; no delivery was attempted.", file=sys.stderr)
        return 2

    decisions = run_stale_task_resurfacer_dry_run(limit=args.limit, lookback_days=args.lookback_days)
    _emit_decisions(decisions, json_output=args.json, empty_message="No stale-task candidates found.")
    return 0


def _parse_log_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_notification_action(action: str) -> bool:
    return action not in {"", "skip", "silent_log"}


if __name__ == "__main__":
    raise SystemExit(main())
