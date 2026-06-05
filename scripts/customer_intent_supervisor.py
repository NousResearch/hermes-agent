#!/usr/bin/env python3
"""Collect pending customer-service intents for a Zeus supervisor cron.

This script is intentionally deterministic and safe: it does not send emails,
create quotes, mutate agenda, or complete work. It only prints compact context
when pending intents exist. Hermes cron injects that output into an LLM supervisor
prompt that can decide and execute privileged actions with verification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import agent_core_sql as sql  # noqa: E402


def _q(value: Any) -> str:
    return sql.quote_literal(value)


def fetch_pending(limit: int) -> list[dict[str, Any]]:
    return sql.rows(f"""
      SELECT ci.intent_id, ci.status, ci.priority, ci.intent_type, ci.channel,
             ci.conversation_ref, ci.source_ref, ci.customer_request_raw,
             ci.summary, ci.required_action, ci.assigned_to, ci.due_at,
             ci.created_at, ci.metadata,
             ci.organization_id, o.name AS organization_name,
             ci.contact_id, c.full_name AS contact_name, c.email AS contact_email, c.phone AS contact_phone,
             ci.opportunity_id, opp.title AS opportunity_title, opp.stage AS opportunity_stage,
             ci.interaction_id
      FROM crm.customer_intents ci
      LEFT JOIN crm.contacts c ON c.contact_id = ci.contact_id
      LEFT JOIN crm.organizations o ON o.organization_id = ci.organization_id
      LEFT JOIN crm.opportunities opp ON opp.opportunity_id = ci.opportunity_id
      WHERE ci.status = 'pending'
      ORDER BY
        CASE ci.priority WHEN 'urgent' THEN 0 WHEN 'high' THEN 1 WHEN 'normal' THEN 2 ELSE 3 END,
        ci.created_at ASC
      LIMIT {max(1, min(50, int(limit)))}
    """, user=sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime"))


def fetch_recent_context(intent: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    clauses: list[str] = []
    if intent.get("contact_id"):
        clauses.append(f"i.contact_id = {_q(intent['contact_id'])}")
    if intent.get("organization_id"):
        clauses.append(f"i.organization_id = {_q(intent['organization_id'])}")
    if intent.get("opportunity_id"):
        clauses.append(f"i.opportunity_id = {_q(intent['opportunity_id'])}")
    if not clauses:
        return []
    where = " OR ".join(clauses)
    return sql.rows(f"""
      SELECT i.interaction_id, i.channel, i.direction, i.summary, i.occurred_at, i.actor, i.metadata
      FROM crm.interactions i
      WHERE {where}
      ORDER BY i.occurred_at DESC
      LIMIT {max(1, min(10, int(limit)))}
    """, user=sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime"))


def build_payload(limit: int) -> dict[str, Any] | None:
    intents = fetch_pending(limit)
    if not intents:
        return None
    for intent in intents:
        intent["recent_interactions"] = fetch_recent_context(intent)
    return {
        "kind": "pending_customer_intents",
        "count": len(intents),
        "instructions": [
            "Analyze each pending intent with CRM context before acting.",
            "Do not mark completed until the requested action is actually executed and provider delivery/CRM evidence is verified.",
            "If action is ambiguous or unsafe, mark blocked with a concise reason and notify Jean/owner if appropriate.",
            "After processing, update the intent via customer_intent_update and record CRM interaction evidence.",
        ],
        "intents": intents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print pending customer intents for Zeus supervisor cron.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = build_payload(args.limit)
    if not payload:
        return
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
