#!/usr/bin/env python3
"""Torben forms, paperwork, and reimbursements packet skill."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCHEMA = "torben.skill-forms.v1"
SECRET_FIELD_RE = re.compile(r"(ssn|social|card|cvv|member\s*id|account|routing|password|token|secret)", re.I)
FIXED_PACKET_FIELDS = (
    "what",
    "deadline",
    "consequence",
    "cost",
    "refundability",
    "required_docs",
    "channel",
)


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _domain(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"mailto://{value}")
    if parsed.hostname:
        return parsed.hostname.lower().removeprefix("www.")
    if "@" in value:
        return value.rsplit("@", 1)[-1].lower()
    return value.lower().removeprefix("www.")


def _is_secret_field(field_name: str) -> bool:
    return bool(SECRET_FIELD_RE.search(field_name))


def _redact_answers(answers: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    safe: dict[str, str] = {}
    hand_to_eric: list[str] = []
    for key, value in answers.items():
        normalized_key = _clean_text(key)
        if _is_secret_field(normalized_key):
            hand_to_eric.append(normalized_key)
            continue
        safe[normalized_key] = _clean_text(value)
    return safe, hand_to_eric


def verify_source(
    *,
    sender: str | None,
    source_url: str | None = None,
    payment_links: list[str] | None = None,
    trusted_domains: list[str] | None = None,
) -> dict[str, Any]:
    trusted = {_domain(domain) for domain in (trusted_domains or []) if _clean_text(domain)}
    sender_domain = _domain(sender or "") if sender else ""
    source_domain = _domain(source_url or "") if source_url else ""
    payment_domains = [_domain(link) for link in (payment_links or []) if _clean_text(link)]
    observed = [domain for domain in [sender_domain, source_domain, *payment_domains] if domain]
    def is_trusted(domain: str) -> bool:
        return any(domain == trusted_domain or domain.endswith(f".{trusted_domain}") for trusted_domain in trusted)

    unknown = [domain for domain in observed if not is_trusted(domain)]
    if not observed:
        return {"status": "unknown", "trusted": False, "reason": "source_missing", "domains": []}
    if unknown:
        return {
            "status": "flagged",
            "trusted": False,
            "reason": "untrusted_or_unknown_domain",
            "domains": observed,
            "unknown_domains": unknown,
        }
    return {"status": "trusted", "trusted": True, "reason": "trusted_domain_match", "domains": observed}


def build_form_packet(
    *,
    action: dict[str, Any],
    source: dict[str, Any],
    answers: dict[str, Any],
    required_fields: list[str] | None = None,
    docs_to_attach: list[str] | None = None,
    trusted_domains: list[str] | None = None,
    category: str = "form_filing",
    now: datetime | None = None,
) -> dict[str, Any]:
    if category not in {"form_filing", "payment_adjacent"}:
        raise ValueError(f"Unsupported form skill category: {category}")
    source_check = verify_source(
        sender=source.get("sender"),
        source_url=source.get("url"),
        payment_links=list(source.get("payment_links") or []),
        trusted_domains=trusted_domains,
    )
    if not source_check["trusted"]:
        safe_answers: dict[str, str] = {}
        hand_to_eric: list[str] = []
    else:
        safe_answers, hand_to_eric = _redact_answers(answers)
    required = [_clean_text(item) for item in (required_fields or []) if _clean_text(item)]
    missing = [field for field in required if field not in safe_answers]
    missing.extend(hand_to_eric)
    if not source_check["trusted"]:
        missing.append("source verification")
    fixed = {field: _clean_text(action.get(field)) for field in FIXED_PACKET_FIELDS}
    fixed["required_docs"] = list(action.get("required_docs") or docs_to_attach or [])
    packet = {
        "schema": SCHEMA,
        "created_at": _iso(now),
        "category": category,
        "status": "packet_only" if source_check["trusted"] else "flagged_unknown_source",
        "source_check": source_check,
        "action": fixed,
        "prepared_answers": safe_answers,
        "missing_info": sorted(set(missing)),
        "hand_to_eric": sorted(set(hand_to_eric)),
        "docs_to_attach": [_clean_text(item) for item in (docs_to_attach or []) if _clean_text(item)],
        "approval_request": "Approve submission through the verified channel, or revise/drop/defer.",
        "blocked_actions": [
            "no form submitted",
            "no payment submitted",
            "no medical/payment/account secret stored",
            "unknown source routes to inbox-safety semantics",
        ],
        "external_actions_taken": [],
    }
    return packet


def packet_to_decision_options(packet: dict[str, Any]) -> list[dict[str, str]]:
    if packet.get("status") == "flagged_unknown_source":
        return [
            {
                "label": "Verify source before filling",
                "upside": "Avoids unsafe form/payment submission",
                "downside": "Adds verification step",
                "cost_time": "5-15m",
                "risk": "low",
            }
        ]
    action = packet.get("action") or {}
    return [
        {
            "label": f"Submit via {action.get('channel') or 'approved channel'}",
            "upside": "Completes paperwork/reimbursement",
            "downside": "Requires Eric approval and any hand-to-Eric secret fields",
            "cost_time": action.get("cost") or "unknown",
            "risk": "high" if packet.get("category") == "payment_adjacent" else "medium",
        },
        {
            "label": "Draft questions instead of submitting",
            "upside": "Clears missing info safely",
            "downside": "Does not complete the filing",
            "cost_time": "5-10m",
            "risk": "low",
        },
    ]


def stage_form_decision(
    *,
    ledger_path: Path,
    loop_id: int,
    packet: dict[str, Any],
) -> dict[str, Any]:
    from torben_decision_packet import stage_decision_packet

    action = packet.get("action") or {}
    options = packet_to_decision_options(packet)
    return stage_decision_packet(
        ledger_path=ledger_path,
        loop_id=loop_id,
        item=f"Approve form/reimbursement action: {action.get('what') or 'paperwork'}",
        context=(
            f"Deadline: {action.get('deadline') or 'unknown'}\n"
            f"Missing info: {', '.join(packet.get('missing_info') or []) or 'none'}\n"
            f"Source status: {(packet.get('source_check') or {}).get('status')}"
        ),
        options=options,
        recommendation=options[0]["label"],
        category=packet.get("category") or "form_filing",
        risk_class="high" if packet.get("category") == "payment_adjacent" else "medium",
    )


def submit_approved_form(
    *,
    ledger_path: Path,
    handle: str,
    confirmation_pointer: str,
    approved_channel: str,
) -> dict[str, Any]:
    from hermes_cli.signal_coo.action_ledger import ActionLedger
    from torben_autonomy_ladder import evaluate_dispatch
    from torben_mutation_spine import record_mutation

    ledger = ActionLedger(ledger_path)
    record = ledger.get(handle)
    if record is None:
        return {"schema": SCHEMA, "status": "refused", "reason": "record_not_found", "handle": handle}
    state = record.executor_state or {}
    category = state.get("category")
    if record.status != "approved":
        return {"schema": SCHEMA, "status": "refused", "reason": "explicit_approval_required", "handle": handle}
    if category not in {"form_filing", "payment_adjacent"}:
        return {"schema": SCHEMA, "status": "refused", "reason": "not_form_packet", "handle": handle}
    dispatch = evaluate_dispatch(category=category, item_status=record.status, requested_count=1, ledger_path=ledger_path)
    mutation = record_mutation(
        ledger_path=ledger_path,
        category=category,
        executor="agent",
        risk_class=record.risk_class or "medium",
        summary=record.summary,
        undo_pointer=None,
        surface=approved_channel,
        metadata={"approved_handle": handle, "confirmation_pointer": confirmation_pointer, "dispatch": dispatch},
    )
    return {
        "schema": SCHEMA,
        "status": "submitted",
        "handle": handle,
        "approved_channel": approved_channel,
        "confirmation_pointer": confirmation_pointer,
        "dispatch": dispatch,
        "mutation": mutation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    packet = subparsers.add_parser("packet")
    packet.add_argument("--action-json", required=True)
    packet.add_argument("--source-json", required=True)
    packet.add_argument("--answers-json", default="{}")
    packet.add_argument("--required-fields-json", default="[]")
    packet.add_argument("--docs-json", default="[]")
    packet.add_argument("--trusted-domains-json", default="[]")
    packet.add_argument("--category", default="form_filing")

    stage = subparsers.add_parser("stage")
    stage.add_argument("--ledger", required=True)
    stage.add_argument("--loop-id", type=int, required=True)
    stage.add_argument("--packet-json", required=True)

    submit = subparsers.add_parser("submit-approved")
    submit.add_argument("--ledger", required=True)
    submit.add_argument("--handle", required=True)
    submit.add_argument("--confirmation-pointer", required=True)
    submit.add_argument("--approved-channel", required=True)
    args = parser.parse_args(argv)

    if args.command == "packet":
        payload = build_form_packet(
            action=json.loads(args.action_json),
            source=json.loads(args.source_json),
            answers=json.loads(args.answers_json),
            required_fields=json.loads(args.required_fields_json),
            docs_to_attach=json.loads(args.docs_json),
            trusted_domains=json.loads(args.trusted_domains_json),
            category=args.category,
        )
    elif args.command == "stage":
        payload = stage_form_decision(ledger_path=Path(args.ledger), loop_id=args.loop_id, packet=json.loads(args.packet_json))
    elif args.command == "submit-approved":
        payload = submit_approved_form(
            ledger_path=Path(args.ledger),
            handle=args.handle,
            confirmation_pointer=args.confirmation_pointer,
            approved_channel=args.approved_channel,
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
