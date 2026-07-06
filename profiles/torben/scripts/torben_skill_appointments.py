#!/usr/bin/env python3
"""Torben appointments and service-call packet skill."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "torben.skill-appointments.v1"
SECRET_FIELD_RE = re.compile(r"(ssn|social|card|cvv|member\s*id|account|password|token|secret)", re.I)
REQUIRED_CONSTRAINTS = ("who", "where", "earliest", "latest", "duration")


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _redact_secret_fields(values: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    safe: dict[str, str] = {}
    hand_to_eric: list[str] = []
    for key, value in values.items():
        normalized_key = _clean_text(key)
        if SECRET_FIELD_RE.search(normalized_key):
            hand_to_eric.append(normalized_key)
            continue
        safe[normalized_key] = _clean_text(value)
    return safe, hand_to_eric


def rank_candidate_windows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        label = _clean_text(candidate.get("label") or candidate.get("window") or f"Option {index + 1}")
        score = float(candidate.get("score", 0))
        ranked.append(
            {
                "label": label,
                "window": _clean_text(candidate.get("window") or label),
                "provider_link": _clean_text(candidate.get("provider_link")),
                "phone": _clean_text(candidate.get("phone")),
                "score": score,
                "notes": _clean_text(candidate.get("notes")),
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def build_appointment_packet(
    *,
    goal: str,
    constraints: dict[str, Any],
    candidates: list[dict[str, Any]],
    provider: dict[str, Any] | None = None,
    forms: list[str] | None = None,
    questions: list[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    safe_constraints, hand_to_eric = _redact_secret_fields(constraints)
    ranked = rank_candidate_windows(candidates)
    missing = [field for field in REQUIRED_CONSTRAINTS if not safe_constraints.get(field)]
    if not ranked:
        missing.append("candidate windows")
    needed_before_booking = sorted(set(missing + hand_to_eric))
    best = ranked[0] if ranked else None
    backup = ranked[1] if len(ranked) > 1 else None
    provider_safe, provider_secret_fields = _redact_secret_fields(provider or {})
    needed_before_booking.extend(provider_secret_fields)
    needed_before_booking = sorted(set(needed_before_booking))
    packet = {
        "schema": SCHEMA,
        "created_at": _iso(now),
        "category": "booking",
        "status": "packet_only",
        "goal": _clean_text(goal),
        "constraints": safe_constraints,
        "provider": provider_safe,
        "best_option": best,
        "backup_option": backup,
        "needed_before_booking": needed_before_booking,
        "forms_doc_pointers": [_clean_text(item) for item in (forms or []) if _clean_text(item)],
        "questions": [_clean_text(item) for item in (questions or []) if _clean_text(item)],
        "blocked_actions": [
            "no calls placed",
            "no forms submitted",
            "no appointment booked",
            "no payment/account/medical secrets stored",
        ],
        "external_actions_taken": [],
    }
    return packet


def packet_to_decision_options(packet: dict[str, Any]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for label, option in (("Book best option", packet.get("best_option")), ("Use backup option", packet.get("backup_option"))):
        if not isinstance(option, dict):
            continue
        options.append(
            {
                "label": f"{label}: {option.get('label') or option.get('window')}",
                "upside": "Moves appointment/service call forward",
                "downside": "May need reschedule if provider availability changes",
                "cost_time": option.get("notes") or "provider-dependent",
                "risk": "medium",
            }
        )
    if not options:
        options.append(
            {
                "label": "Ask provider for viable windows",
                "upside": "Gets missing scheduling data",
                "downside": "Adds one follow-up step",
                "cost_time": "5-10m",
                "risk": "low",
            }
        )
    return options


def stage_appointment_decision(
    *,
    ledger_path: Path,
    loop_id: int,
    packet: dict[str, Any],
) -> dict[str, Any]:
    from torben_decision_packet import stage_decision_packet

    options = packet_to_decision_options(packet)
    context_parts = [
        f"Goal: {packet['goal']}",
        f"Needed before booking: {', '.join(packet['needed_before_booking']) or 'none'}",
        f"Provider: {packet.get('provider') or {}}",
    ]
    return stage_decision_packet(
        ledger_path=ledger_path,
        loop_id=loop_id,
        item=f"Approve appointment/service-call booking: {packet['goal']}",
        context="\n".join(context_parts),
        options=options,
        recommendation=options[0]["label"],
        category="booking",
        risk_class="medium",
    )


def execute_approved_booking(
    *,
    ledger_path: Path,
    handle: str,
    confirmation_pointer: str,
    summary: str | None = None,
) -> dict[str, Any]:
    from hermes_cli.signal_coo.action_ledger import ActionLedger
    from torben_autonomy_ladder import evaluate_dispatch
    from torben_mutation_spine import record_mutation

    ledger = ActionLedger(ledger_path)
    record = ledger.get(handle)
    if record is None:
        return {"schema": SCHEMA, "status": "refused", "reason": "record_not_found", "handle": handle}
    state = record.executor_state or {}
    if record.status != "approved":
        return {"schema": SCHEMA, "status": "refused", "reason": "explicit_approval_required", "handle": handle}
    if state.get("category") != "booking":
        return {"schema": SCHEMA, "status": "refused", "reason": "not_booking_packet", "handle": handle}
    dispatch = evaluate_dispatch(category="booking", item_status=record.status, requested_count=1, ledger_path=ledger_path)
    mutation = record_mutation(
        ledger_path=ledger_path,
        category="booking",
        executor="agent",
        risk_class=record.risk_class or "medium",
        summary=summary or record.summary,
        undo_pointer=None,
        surface="appointment-or-service-provider",
        metadata={"approved_handle": handle, "confirmation_pointer": confirmation_pointer, "dispatch": dispatch},
    )
    return {
        "schema": SCHEMA,
        "status": "executed",
        "handle": handle,
        "confirmation_pointer": confirmation_pointer,
        "dispatch": dispatch,
        "mutation": mutation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packet = subparsers.add_parser("packet")
    packet.add_argument("--goal", required=True)
    packet.add_argument("--constraints-json", default="{}")
    packet.add_argument("--candidates-json", default="[]")
    packet.add_argument("--provider-json", default="{}")
    packet.add_argument("--forms-json", default="[]")
    packet.add_argument("--questions-json", default="[]")

    stage = subparsers.add_parser("stage")
    stage.add_argument("--ledger", required=True)
    stage.add_argument("--loop-id", type=int, required=True)
    stage.add_argument("--packet-json", required=True)

    execute = subparsers.add_parser("execute-approved")
    execute.add_argument("--ledger", required=True)
    execute.add_argument("--handle", required=True)
    execute.add_argument("--confirmation-pointer", required=True)
    execute.add_argument("--summary")
    args = parser.parse_args(argv)

    if args.command == "packet":
        payload = build_appointment_packet(
            goal=args.goal,
            constraints=json.loads(args.constraints_json),
            candidates=json.loads(args.candidates_json),
            provider=json.loads(args.provider_json),
            forms=json.loads(args.forms_json),
            questions=json.loads(args.questions_json),
        )
    elif args.command == "stage":
        payload = stage_appointment_decision(
            ledger_path=Path(args.ledger),
            loop_id=args.loop_id,
            packet=json.loads(args.packet_json),
        )
    elif args.command == "execute-approved":
        payload = execute_approved_booking(
            ledger_path=Path(args.ledger),
            handle=args.handle,
            confirmation_pointer=args.confirmation_pointer,
            summary=args.summary,
        )
    else:
        raise ValueError(f"Unhandled command: {args.command}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
