#!/usr/bin/env python3
"""Ingest public delivery-sandbox events into Agent Core and visible logs.

The public delivery sandbox intentionally stores only append-only JSONL events.
This trusted-side worker promotes those events into canonical local modules:

- sales.customer_workspace_events for quote/catalog/invoice workspaces
- accounting.receipt_events for payment receipt workspaces
- crm.follow_ups so the agent/owner has an action to process
- per-workspace comments.json files shown on the public template

It is idempotent by storing the public delivery event id in metadata.delivery_event_id.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import agent_core_sql as sql  # noqa: E402
from scripts.runtime.delivery_document_actions import (  # noqa: E402
    DOCUMENT_ACTION_EVENT_TYPES,
    OTP_REQUIRED_DOCUMENT_EVENT_TYPES,
)
from tools import sales_tool  # noqa: E402

EVENT_TYPES_WITH_OWNER_ACTION = DOCUMENT_ACTION_EVENT_TYPES | {"change_requested", "payment_failed"}
SALES_EVENT_TYPES = {"opened", "paid", "payment_started", "cancelled", "expired"} | DOCUMENT_ACTION_EVENT_TYPES
RECEIPT_EVENT_TYPES = {"opened"} | DOCUMENT_ACTION_EVENT_TYPES
FINAL_SALES_STATUS = {"paid", "cancelled", "expired"} | OTP_REQUIRED_DOCUMENT_EVENT_TYPES
FINAL_RECEIPT_STATUS = {"paid", "cancelled"} | OTP_REQUIRED_DOCUMENT_EVENT_TYPES


def _q(value: Any) -> str:
    return sql.quote_literal(value)


def _j(value: Any) -> str:
    return sql.quote_jsonb(value)


def _ts(value: Any) -> str:
    if not value:
        return "now()"
    try:
        # Validate rough ISO value; Postgres performs final parsing.
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return f"{_q(value)}::timestamptz"
    except Exception:
        return "now()"


def _sales_user() -> str:
    return sql.runtime_env().get("SALES_DB_RUNTIME_USER", "sales_runtime")


def _accounting_user() -> str:
    return sql.runtime_env().get("ACCOUNTING_DB_RUNTIME_USER", "accounting_runtime")


def _crm_user() -> str:
    return os.getenv("CRM_DB_RUNTIME_USER") or sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime")


def _event_id(event: dict[str, Any]) -> str:
    return str(event.get("event_id") or "").strip()


def _metadata(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _canonical_metadata(event: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(_metadata(event))
    metadata.update({
        "delivery_event_id": _event_id(event),
        "delivery_source": event.get("source") or "delivery-sandbox",
        "deliverable_id": event.get("deliverable_id"),
        "ip_address": event.get("ip_address"),
        "user_agent": event.get("user_agent"),
    })
    if event.get("token_ref"):
        metadata["token_ref"] = event.get("token_ref")
    if extra:
        metadata.update(extra)
    return metadata


def _already_ingested_sales(event_id: str) -> bool:
    if not event_id:
        return False
    row = sql.one(
        f"SELECT 1 FROM sales.customer_workspace_events WHERE metadata->>'delivery_event_id'={_q(event_id)}",
        user=_sales_user(),
    )
    return bool(row)


def _already_ingested_receipt(event_id: str) -> bool:
    if not event_id:
        return False
    row = sql.one(
        f"SELECT 1 FROM accounting.receipt_events WHERE metadata->>'delivery_event_id'={_q(event_id)}",
        user=_accounting_user(),
    )
    return bool(row)


def _sales_workspace(event: dict[str, Any]) -> dict[str, Any] | None:
    metadata = _metadata(event)
    workspace_id = str(metadata.get("workspace_id") or "").strip()
    deliverable_id = str(event.get("deliverable_id") or "").strip()
    if workspace_id:
        row = sql.one(
            f"SELECT * FROM sales.customer_workspaces WHERE workspace_id={_q(workspace_id)}",
            user=_sales_user(),
        )
        if row:
            return row
    if deliverable_id:
        row = sql.one(
            f"SELECT * FROM sales.customer_workspaces WHERE document_id={_q(deliverable_id)} ORDER BY updated_at DESC",
            user=_sales_user(),
        )
        if row:
            return row
    return None


def _receipt_row(event: dict[str, Any]) -> dict[str, Any] | None:
    metadata = _metadata(event)
    receipt_id = str(metadata.get("receipt_id") or event.get("deliverable_id") or "").strip()
    if not receipt_id:
        return None
    return sql.one(f"SELECT * FROM accounting.receipts WHERE receipt_id={_q(receipt_id)}", user=_accounting_user())


def _status_for_event(event_type: str, current: str | None) -> str | None:
    if current in {"approved", "rejected", "signed", "paid", "cancelled", "expired"} and event_type not in FINAL_SALES_STATUS | FINAL_RECEIPT_STATUS:
        return None
    if event_type == "opened" and current in {None, "pending", "sent", "draft"}:
        return "viewed"
    if event_type == "commented" and current not in {"approved", "rejected", "signed", "paid", "cancelled", "expired"}:
        return "commented"
    if event_type in FINAL_SALES_STATUS or event_type in FINAL_RECEIPT_STATUS:
        return event_type
    return None


def _record_follow_up(kind: str, subject: str, event: dict[str, Any], metadata: dict[str, Any]) -> None:
    event_type = str(event.get("event_type") or "")
    if event_type not in EVENT_TYPES_WITH_OWNER_ACTION:
        return
    event_id = _event_id(event)
    if event_id:
        exists = sql.one(
            f"SELECT 1 FROM crm.follow_ups WHERE metadata->>'delivery_event_id'={_q(event_id)}",
            user=_crm_user(),
        )
        if exists:
            return
    actor = event.get("actor_ref") or event.get("actor_type") or "cliente"
    comment = str(event.get("comment") or "").strip()
    action_labels = {
        "commented": "comentó",
        "approved": "aprobó",
        "rejected": "rechazó",
        "signed": "firmó",
        "change_requested": "pidió cambios en",
        "payment_failed": "reportó fallo de pago en",
    }
    summary = f"{actor} {action_labels.get(event_type, event_type)} {kind} {subject}".strip()
    if comment:
        summary = f"{summary}: {comment}"
    priority = "high" if event_type in {"approved", "rejected", "signed", "change_requested", "payment_failed"} else "normal"
    sql.statement_one(
        f"""
        INSERT INTO crm.follow_ups (organization_id, contact_id, opportunity_id, due_at, summary, status, priority, assignee, metadata, created_at, updated_at)
        VALUES (NULL, NULL, NULL, now(), {_q(summary)}, 'open', {_q(priority)}, 'agent', {_j(metadata)}, now(), now())
        RETURNING *
        """,
        user=_crm_user(),
    )


def _refresh_sales_comments(workspace: dict[str, Any], public_root: Path) -> None:
    token = workspace.get("public_token")
    if not token:
        return
    path = public_root / "w" / str(token) / "comments.json"
    if not path.parent.exists():
        return
    rows = sql.rows(
        f"""
        SELECT workspace_event_id AS id, event_type, actor_type, actor_ref, comment, metadata,
               to_char(occurred_at AT TIME ZONE 'America/New_York', 'YYYY-MM-DD HH24:MI') AS occurred_at
        FROM sales.customer_workspace_events
        WHERE workspace_id={_q(workspace['workspace_id'])}
        ORDER BY occurred_at, workspace_event_id
        """,
        user=_sales_user(),
    )
    path.write_text(json.dumps({"events": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _refresh_receipt_comments(receipt: dict[str, Any], public_root: Path) -> None:
    token = receipt.get("public_token")
    if not token:
        return
    path = public_root / "w" / str(token) / "comments.json"
    if not path.parent.exists():
        return
    rows = sql.rows(
        f"""
        SELECT receipt_event_id AS id, event_type, actor_type, actor_ref, comment, metadata,
               to_char(occurred_at AT TIME ZONE 'America/New_York', 'YYYY-MM-DD HH24:MI') AS occurred_at
        FROM accounting.receipt_events
        WHERE receipt_id={_q(receipt['receipt_id'])}
        ORDER BY occurred_at, receipt_event_id
        """,
        user=_accounting_user(),
    )
    path.write_text(json.dumps({"events": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _convert_quote_on_approval(workspace: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    if workspace.get("document_type") != "quote" or event.get("event_type") != "approved":
        return {}
    # sales_tool is idempotent on deterministic IDs derived from quote/title.
    order_payload = json.loads(sales_tool._handle_order_create({
        "quote_id": workspace["document_id"],
        "metadata": {
            "source": "delivery_sandbox_ingest",
            "workspace_id": workspace["workspace_id"],
            "delivery_event_id": _event_id(event),
        },
    }))
    if not order_payload.get("ok"):
        return {"order_error": order_payload.get("error") or order_payload}
    order_id = order_payload.get("order", {}).get("order_id")
    invoice_payload = json.loads(sales_tool._handle_invoice_create({
        "order_id": order_id,
        "metadata": {
            "source": "delivery_sandbox_ingest",
            "workspace_id": workspace["workspace_id"],
            "delivery_event_id": _event_id(event),
        },
    }))
    if not invoice_payload.get("ok"):
        return {"order_id": order_id, "invoice_error": invoice_payload.get("error") or invoice_payload}
    return {"order_id": order_id, "invoice_id": invoice_payload.get("invoice", {}).get("invoice_id")}


def ingest_sales_event(event: dict[str, Any], public_root: Path) -> str | None:
    event_type = str(event.get("event_type") or "").strip()
    if event_type not in SALES_EVENT_TYPES:
        return None
    event_id = _event_id(event)
    if _already_ingested_sales(event_id):
        return "sales:duplicate"
    workspace = _sales_workspace(event)
    if not workspace:
        return None
    extra = _convert_quote_on_approval(workspace, event)
    metadata = _canonical_metadata(event, extra)
    sql.statement_one(
        f"""
        INSERT INTO sales.customer_workspace_events (workspace_id, event_type, actor_type, actor_ref, comment, metadata, occurred_at)
        VALUES ({_q(workspace['workspace_id'])}, {_q(event_type)}, {_q(event.get('actor_type') or 'customer')}, {_q(event.get('actor_ref'))}, {_q(event.get('comment'))}, {_j(metadata)}, {_ts(event.get('occurred_at'))})
        RETURNING *
        """,
        user=_sales_user(),
    )
    new_status = _status_for_event(event_type, workspace.get("status"))
    if new_status:
        sql.statement_one(
            f"UPDATE sales.customer_workspaces SET status={_q(new_status)}, updated_at=now(), metadata=metadata || {_j(extra)} WHERE workspace_id={_q(workspace['workspace_id'])} RETURNING *",
            user=_sales_user(),
        )
    _record_follow_up(workspace.get("document_type") or "documento", workspace.get("document_id") or "", event, metadata)
    _refresh_sales_comments({**workspace, "status": new_status or workspace.get("status")}, public_root)
    return "sales:ingested"


def ingest_receipt_event(event: dict[str, Any], public_root: Path) -> str | None:
    event_type = str(event.get("event_type") or "").strip()
    if event_type not in RECEIPT_EVENT_TYPES:
        return None
    event_id = _event_id(event)
    if _already_ingested_receipt(event_id):
        return "receipt:duplicate"
    receipt = _receipt_row(event)
    if not receipt:
        return None
    metadata = _canonical_metadata(event)
    sql.statement_one(
        f"""
        INSERT INTO accounting.receipt_events (receipt_id, event_type, actor_type, actor_ref, comment, metadata, occurred_at)
        VALUES ({_q(receipt['receipt_id'])}, {_q(event_type)}, {_q(event.get('actor_type') or 'counterparty')}, {_q(event.get('actor_ref'))}, {_q(event.get('comment'))}, {_j(metadata)}, {_ts(event.get('occurred_at'))})
        RETURNING *
        """,
        user=_accounting_user(),
    )
    new_status = _status_for_event(event_type, receipt.get("status"))
    if new_status:
        updates = ["status=" + _q(new_status), "updated_at=now()"]
        if event_type == "rejected" and event.get("comment"):
            updates.append("rejection_reason=" + _q(event.get("comment")))
        if event_type == "approved" and _metadata(event).get("approval_hash"):
            updates.append("approval_hash=" + _q(_metadata(event).get("approval_hash")))
        sql.statement_one(
            f"UPDATE accounting.receipts SET {', '.join(updates)} WHERE receipt_id={_q(receipt['receipt_id'])} RETURNING *",
            user=_accounting_user(),
        )
    _record_follow_up("recibo/nota de pago", receipt.get("receipt_id") or "", event, metadata)
    _refresh_receipt_comments({**receipt, "status": new_status or receipt.get("status")}, public_root)
    return "receipt:ingested"


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("status") in {None, "pending_agent_ingest", "queued_for_agent_ingest"}:
            events.append(obj)
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest delivery-sandbox events into Agent Core")
    parser.add_argument("--events", default="/home/jean/zeus-runtime/delivery-sandbox/events/events.jsonl")
    parser.add_argument("--public-root", default="/home/jean/zeus-runtime/delivery-sandbox/public")
    parser.add_argument("--deliverable-id", help="Limit to one public deliverable/document id")
    parser.add_argument("--since", help="Only process events at/after this ISO timestamp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Print only when new canonical events were ingested")
    args = parser.parse_args(argv)

    events_path = Path(args.events)
    public_root = Path(args.public_root)
    since_dt = None
    if args.since:
        since_dt = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)

    counts: dict[str, int] = {"seen": 0, "sales_ingested": 0, "receipt_ingested": 0, "duplicates": 0, "unmatched": 0, "dry_run": 0}
    for event in load_events(events_path):
        if args.deliverable_id and str(event.get("deliverable_id")) != args.deliverable_id:
            continue
        if since_dt and event.get("occurred_at"):
            try:
                occurred = datetime.fromisoformat(str(event["occurred_at"]).replace("Z", "+00:00"))
                if occurred < since_dt:
                    continue
            except Exception:
                pass
        counts["seen"] += 1
        if args.dry_run:
            print(json.dumps({"would_process": event.get("event_id"), "event_type": event.get("event_type"), "deliverable_id": event.get("deliverable_id")}, ensure_ascii=False))
            counts["dry_run"] += 1
            continue
        result = ingest_sales_event(event, public_root)
        if result is None:
            result = ingest_receipt_event(event, public_root)
        if result == "sales:ingested":
            counts["sales_ingested"] += 1
        elif result == "receipt:ingested":
            counts["receipt_ingested"] += 1
        elif result and "duplicate" in result:
            counts["duplicates"] += 1
        else:
            counts["unmatched"] += 1
    if not args.quiet or counts["sales_ingested"] or counts["receipt_ingested"]:
        print(json.dumps({"ok": True, "events_path": str(events_path), "public_root": str(public_root), "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
