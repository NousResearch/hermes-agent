"""Canonical payments-review adapter for dashboard + Slack operator surfaces."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import types
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Dict, List

from hermes_constants import get_hermes_home
from hermes_cli.config import load_env

PAYMENT_STATUSES = {
    "new",
    "needs_review",
    "ready_to_pay",
    "paid",
    "ignored",
}

OPERATOR_STATUSES = (
    "needs_review",
    "ready_to_pay",
    "paid",
    "ignored",
)

INBOX_ITEM_STATUSES = PAYMENT_STATUSES

PAYMENT_SOURCES = (
    ("gmail", "Gmail"),
    ("email", "Email"),
    ("uploads", "Uploads"),
    ("slack", "Slack"),
)

PAYMENTS_GMAIL_SYNC_CRON_SCRIPT = "payments_sync_gmail.py"
PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME = "Payments Gmail sync"
DEFAULT_GMAIL_SYNC_SCHEDULE = "every 1h"
PAYMENTS_GMAIL_SHADOW_SYNC_CRON_SCRIPT = "payments_shadow_sync_gmail.py"
PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME = "Payments Gmail shadow sync"
DEFAULT_GMAIL_SHADOW_SYNC_SCHEDULE = "every 6h"

DEFAULT_GMAIL_QUERY = (
    '((subject:invoice OR subject:receipt OR subject:"payment due" OR '
    'subject:remittance OR subject:"remittance advice" OR '
    'subject:"booking confirmation" OR "amount due" OR "view invoice" OR '
    '"payment reference" OR "invoice #") -category:promotions '
    '-category:social -label:spam) newer_than:90d'
)
DEFAULT_GMAIL_MAX_RESULTS = 20

_PAID_TITLES_RE = re.compile(
    r"\b(receipt|booking confirmation|thanks for riding|amount paid|paid)\b",
    re.IGNORECASE,
)
_PAID_WARNING_RE = re.compile(r"already been made", re.IGNORECASE)
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PAYMENTS_QUEUE = "payments"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_runtime_env() -> dict[str, Any]:
    try:
        return load_env()
    except Exception:
        return {}


def _require_env_value(name: str) -> str:
    env = _load_runtime_env()
    value = str(os.environ.get(name) or env.get(name) or "").strip()
    return value


def _canonical_db_path() -> Path | None:
    value = _require_env_value("PAYMENTS_REVIEW_DB_PATH")
    return Path(value) if value else None


def _canonical_asset_root() -> Path | None:
    value = _require_env_value("PAYMENTS_REVIEW_ASSET_ROOT")
    return Path(value) if value else None


def _canonical_materialized_root() -> Path | None:
    value = _require_env_value("PAYMENTS_REVIEW_MATERIALIZED_ROOT")
    return Path(value) if value else None


def _google_token_path() -> Path:
    env = _load_runtime_env()
    raw = str(
        os.environ.get("GOOGLE_WORKSPACE_TOKEN_PATH")
        or env.get("GOOGLE_WORKSPACE_TOKEN_PATH")
        or get_hermes_home() / "google_token.json"
    ).strip()
    return Path(raw)


def _slack_token_present() -> bool:
    env = _load_runtime_env()
    return bool(
        str(os.environ.get("SLACK_BOT_TOKEN") or env.get("SLACK_BOT_TOKEN") or "").strip()
    )


def _storage_display_path() -> str:
    db_path = _canonical_db_path()
    if db_path is not None:
        return str(db_path)
    return str(get_hermes_home() / "payments" / "requests.json")


def _canonical_enabled() -> bool:
    return _canonical_db_path() is not None


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _load_runtime_from_path(module_name: str, runtime_path: Path):
    plugin_parents = {str(runtime_path.parent.parent)}
    for env_name in ("PAYMENTS_REVIEW_PLUGIN_ROOT", "GOOGLE_WORKSPACE_PLUGIN_ROOT"):
        value = _require_env_value(env_name)
        if value:
            plugin_parents.add(str(Path(value).expanduser().resolve().parent))
    for plugin_parent in sorted(parent for parent in plugin_parents if parent):
        if plugin_parent not in sys.path:
            sys.path.insert(0, plugin_parent)
    if "." in module_name:
        package_name = module_name.rsplit(".", 1)[0]
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = [str(runtime_path.parent)]  # type: ignore[attr-defined]
            sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_runtime_paths() -> list[Path]:
    candidates = [
        Path("/opt/data/plugins/payments_review/runtime.py"),
        get_hermes_home() / "plugins" / "payments_review" / "runtime.py",
        Path.cwd() / "plugins" / "payments_review" / "runtime.py",
    ]
    value = _require_env_value("PAYMENTS_REVIEW_PLUGIN_ROOT")
    if value:
        candidates.insert(0, Path(value) / "runtime.py")
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _payments_review_runtime():
    try:
        from plugins.payments_review import runtime as payments_runtime

        return payments_runtime
    except Exception:
        try:
            import payments_review.runtime as payments_runtime

            return payments_runtime
        except Exception:
            for runtime_path in _candidate_runtime_paths():
                if runtime_path.is_file():
                    module = _load_runtime_from_path("payments_review.runtime", runtime_path)
                    if module is not None:
                        return module
    raise RuntimeError("payments_review runtime is unavailable")


def _connect() -> sqlite3.Connection:
    db_path = _canonical_db_path()
    if db_path is None:
        raise RuntimeError("PAYMENTS_REVIEW_DB_PATH is not configured")
    attempts: list[tuple[Any, dict[str, Any]]] = [
        (str(db_path), {}),
        (f"file:{db_path}?mode=ro&immutable=1", {"uri": True}),
    ]
    last_error: sqlite3.OperationalError | None = None
    for target, kwargs in attempts:
        try:
            conn = sqlite3.connect(target, **kwargs)
            # Force SQLite to open the schema now so read-only mounts fail over
            # before later queries reach the dashboard surface.
            conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to connect to canonical payments storage")


def _canonical_storage_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('inbox_items', 'payment_requests')"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _canonical_storage_kind(conn: sqlite3.Connection, *, prefer: str = "payment_requests") -> str:
    names = _canonical_storage_tables(conn)
    if prefer == "payment_requests":
        if "payment_requests" in names:
            return "payment_requests"
        if "inbox_items" in names:
            return "inbox_items"
    else:
        if "inbox_items" in names:
            return "inbox_items"
        if "payment_requests" in names:
            return "payment_requests"
    raise RuntimeError("Canonical payments storage is not initialized")


def _ensure_inbox_items_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inbox_items (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_account TEXT,
            source_thread_id TEXT,
            source_message_id TEXT,
            source_url TEXT,
            captured_at TEXT,
            updated_at TEXT,
            last_classified_at TEXT,
            workflow_type TEXT NOT NULL,
            queue TEXT NOT NULL,
            status TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            counterparty TEXT,
            action_required INTEGER,
            confidence REAL,
            amount_value REAL,
            amount_currency TEXT,
            amount_display TEXT,
            due_date TEXT,
            meeting_dates_json TEXT,
            meeting_timezone TEXT,
            reference_id TEXT,
            receipt_like INTEGER,
            invoice_like INTEGER,
            calendar_like INTEGER,
            warning_flags_json TEXT,
            operator_notes TEXT,
            raw_payload_path TEXT,
            materialized_path TEXT,
            manual_status INTEGER,
            manual_notes INTEGER,
            reviewed_at TEXT,
            reviewed_by TEXT,
            entities_json TEXT,
            artifacts_json TEXT
        )
        """
    )


def _format_due_date(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if _DATE_ONLY_RE.match(text):
        return text
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return text


def _looks_paid(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") == "paid":
        return True
    title = str(row.get("title") or "")
    if _PAID_TITLES_RE.search(title):
        return True
    warnings = row.get("warnings") or []
    return any(_PAID_WARNING_RE.search(str(item)) for item in warnings)


def _operator_status(status: str) -> str:
    return "needs_review" if status == "new" else status


def _status_label(status: str) -> str:
    return {
        "new": "New",
        "needs_review": "Needs review",
        "ready_to_pay": "Ready to pay",
        "paid": "Paid",
        "ignored": "Ignored",
    }.get(status, status.replace("_", " ").title())


def _source_label(source: str) -> str:
    for source_id, label in PAYMENT_SOURCES:
        if source_id == source:
            return label
    return source.title() or "Unknown"


def _row_to_request(row: sqlite3.Row) -> Dict[str, Any]:
    request = {
        "id": row["payment_id"],
        "payment_id": row["payment_id"],
        "source": row["source"],
        "source_label": _source_label(row["source"]),
        "status": row["status"],
        "operator_status": _operator_status(row["status"]),
        "confidence": row["confidence"],
        "received_at": row["captured_at"],
        "updated_at": row["updated_at"],
        "vendor": row["vendor"] or "",
        "title": row["title"] or "",
        "amount": {
            "value": row["amount_value"],
            "currency": row["amount_currency"] or "",
            "display": row["amount_display"] or "",
        },
        "due_date": _format_due_date(row["due_date"]),
        "payee_name": row["payee_name"] or "",
        "account_holder": row["account_holder"] or "",
        "account_number": row["account_number"] or "",
        "sort_code": row["sort_code"] or "",
        "iban": row["iban"] or "",
        "swift": row["swift"] or "",
        "routing_number": row["routing_number"] or "",
        "payment_reference": row["payment_reference"] or "",
        "invoice_number": row["invoice_number"] or "",
        "billing_address": row["billing_address"] or "",
        "tax_details": row["tax_details"] or "",
        "preview_text": row["preview_text"] or "",
        "warnings": json.loads(row["warnings_json"] or "[]"),
        "attachments": json.loads(row["attachments_json"] or "[]"),
        "original": json.loads(row["original_json"] or "{}"),
        "raw_text_path": row["raw_text_path"],
        "materialized_path": row["materialized_path"],
        "review_note": row["review_note"] or "",
        "looks_paid": False,
    }
    request["looks_paid"] = _looks_paid(request)
    return request


def _payment_workflow_type(request: Dict[str, Any]) -> str:
    return "payment_receipt" if request.get("looks_paid") or request.get("status") == "paid" else "payment_request"


def _request_to_inbox_item(request: Dict[str, Any]) -> Dict[str, Any]:
    from plugins.inbox_ops_payments import PaymentCandidateClassifier, PaymentCandidateExtractor

    classification = PaymentCandidateClassifier().classify(request)
    return PaymentCandidateExtractor().extract(request, classification)


def _inbox_item_to_request(row: sqlite3.Row) -> Dict[str, Any]:
    entities = _json_loads(_row_value(row, "entities_json", "{}"), {})
    artifacts = _json_loads(_row_value(row, "artifacts_json", "{}"), {})
    warnings = [str(item) for item in _json_loads(_row_value(row, "warning_flags_json", "[]"), [])]
    attachments = [str(item) for item in artifacts.get("attachments", []) if str(item).strip()]
    original = {
        "label": str(artifacts.get("label") or entities.get("sender") or ""),
        "url": str(_row_value(row, "source_url", "") or artifacts.get("original_thread_url") or ""),
        "message_id": str(_row_value(row, "source_message_id", "") or artifacts.get("message_id") or ""),
        "thread_id": str(_row_value(row, "source_thread_id", "") or artifacts.get("thread_id") or ""),
    }
    request = {
        "id": str(_row_value(row, "id", "")),
        "payment_id": str(_row_value(row, "id", "")),
        "source": str(_row_value(row, "source", "gmail") or "gmail"),
        "source_label": _source_label(str(_row_value(row, "source", "gmail") or "gmail")),
        "status": str(_row_value(row, "status", "needs_review") or "needs_review"),
        "operator_status": _operator_status(str(_row_value(row, "status", "needs_review") or "needs_review")),
        "confidence": _row_value(row, "confidence", ""),
        "received_at": _row_value(row, "captured_at"),
        "updated_at": _row_value(row, "updated_at", _row_value(row, "last_classified_at")),
        "vendor": str(_row_value(row, "counterparty", "") or entities.get("organization") or ""),
        "title": str(_row_value(row, "title", "") or ""),
        "amount": {
            "value": _row_value(row, "amount_value"),
            "currency": str(_row_value(row, "amount_currency", "") or ""),
            "display": str(_row_value(row, "amount_display", "") or ""),
        },
        "due_date": _format_due_date(_row_value(row, "due_date")),
        "payee_name": str(entities.get("payee_name") or ""),
        "account_holder": str(entities.get("account_holder") or ""),
        "account_number": str(entities.get("account_number") or ""),
        "sort_code": str(entities.get("sort_code") or ""),
        "iban": str(entities.get("iban") or ""),
        "swift": str(entities.get("swift") or ""),
        "routing_number": str(entities.get("routing_number") or ""),
        "payment_reference": str(entities.get("payment_reference") or _row_value(row, "reference_id", "") or ""),
        "invoice_number": str(entities.get("invoice_number") or ""),
        "billing_address": str(entities.get("billing_address") or ""),
        "tax_details": str(entities.get("tax_details") or ""),
        "preview_text": str(_row_value(row, "summary", "") or ""),
        "warnings": warnings,
        "attachments": attachments,
        "original": original,
        "raw_text_path": _row_value(row, "raw_payload_path"),
        "materialized_path": _row_value(row, "materialized_path"),
        "review_note": str(_row_value(row, "operator_notes", "") or ""),
        "looks_paid": bool(_row_value(row, "receipt_like", 0)),
    }
    request["looks_paid"] = request["looks_paid"] or _looks_paid(request)
    return request


def _upsert_inbox_item(conn: sqlite3.Connection, item: Dict[str, Any]) -> None:
    _ensure_inbox_items_schema(conn)
    amount = item.get("amount") or {}
    entities = item.get("entities") if isinstance(item.get("entities"), dict) else {}
    artifacts = item.get("artifacts") if isinstance(item.get("artifacts"), dict) else {}
    payload = {
        "id": str(item.get("id") or ""),
        "source": str(item.get("source") or "gmail"),
        "source_account": str(item.get("source_account") or ""),
        "source_thread_id": str(item.get("source_thread_id") or ""),
        "source_message_id": str(item.get("source_message_id") or ""),
        "source_url": str(item.get("source_url") or ""),
        "captured_at": item.get("captured_at"),
        "updated_at": item.get("updated_at"),
        "last_classified_at": item.get("last_classified_at") or item.get("updated_at"),
        "workflow_type": str(item.get("workflow_type") or "payment_request"),
        "queue": str(item.get("queue") or _PAYMENTS_QUEUE),
        "status": str(item.get("status") or "needs_review"),
        "title": str(item.get("title") or ""),
        "summary": str(item.get("summary") or ""),
        "counterparty": str(item.get("counterparty") or ""),
        "action_required": 1 if bool(item.get("action_required")) else 0,
        "confidence": item.get("confidence"),
        "amount_value": amount.get("value"),
        "amount_currency": str(amount.get("currency") or ""),
        "amount_display": str(amount.get("display") or ""),
        "due_date": item.get("due_date"),
        "meeting_dates_json": json.dumps(item.get("meeting_dates") or []),
        "meeting_timezone": str(item.get("meeting_timezone") or ""),
        "reference_id": str(item.get("reference_id") or ""),
        "receipt_like": 1 if bool(item.get("receipt_like")) else 0,
        "invoice_like": 1 if bool(item.get("invoice_like")) else 0,
        "calendar_like": 1 if bool(item.get("calendar_like")) else 0,
        "warning_flags_json": json.dumps(item.get("warning_flags") or []),
        "operator_notes": str(item.get("operator_notes") or ""),
        "raw_payload_path": item.get("raw_payload_path"),
        "materialized_path": item.get("materialized_path"),
        "manual_status": 1 if bool(item.get("manual_status")) else 0,
        "manual_notes": 1 if bool(item.get("manual_notes")) else 0,
        "reviewed_at": item.get("reviewed_at"),
        "reviewed_by": item.get("reviewed_by"),
        "entities_json": json.dumps(entities, sort_keys=True),
        "artifacts_json": json.dumps(artifacts, sort_keys=True),
    }
    columns = list(payload.keys())
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "id")
    conn.execute(
        f"""
        INSERT INTO inbox_items ({", ".join(columns)}) VALUES ({placeholders})
        ON CONFLICT(id) DO UPDATE SET {updates}
        """,
        tuple(payload[column] for column in columns),
    )


def _row_to_inbox_item(row: sqlite3.Row) -> Dict[str, Any]:
    entities = _json_loads(_row_value(row, "entities_json", "{}"), {})
    artifacts = _json_loads(_row_value(row, "artifacts_json", "{}"), {})
    warning_flags = [str(item) for item in _json_loads(_row_value(row, "warning_flags_json", "[]"), [])]
    return {
        "id": str(_row_value(row, "id", "")),
        "source": str(_row_value(row, "source", "gmail") or "gmail"),
        "source_account": str(_row_value(row, "source_account", "") or ""),
        "source_thread_id": str(_row_value(row, "source_thread_id", "") or ""),
        "source_message_id": str(_row_value(row, "source_message_id", "") or ""),
        "source_url": str(_row_value(row, "source_url", "") or ""),
        "captured_at": _row_value(row, "captured_at"),
        "updated_at": _row_value(row, "updated_at"),
        "last_classified_at": _row_value(row, "last_classified_at"),
        "workflow_type": str(_row_value(row, "workflow_type", "") or ""),
        "queue": str(_row_value(row, "queue", "") or ""),
        "status": str(_row_value(row, "status", "needs_review") or "needs_review"),
        "title": str(_row_value(row, "title", "") or ""),
        "summary": str(_row_value(row, "summary", "") or ""),
        "counterparty": str(_row_value(row, "counterparty", "") or ""),
        "action_required": bool(_row_value(row, "action_required", 0)),
        "confidence": _row_value(row, "confidence"),
        "amount": {
            "value": _row_value(row, "amount_value"),
            "currency": str(_row_value(row, "amount_currency", "") or ""),
            "display": str(_row_value(row, "amount_display", "") or ""),
        },
        "due_date": _format_due_date(_row_value(row, "due_date")),
        "meeting_dates": _json_loads(_row_value(row, "meeting_dates_json", "[]"), []),
        "meeting_timezone": str(_row_value(row, "meeting_timezone", "") or ""),
        "reference_id": str(_row_value(row, "reference_id", "") or ""),
        "receipt_like": bool(_row_value(row, "receipt_like", 0)),
        "invoice_like": bool(_row_value(row, "invoice_like", 0)),
        "calendar_like": bool(_row_value(row, "calendar_like", 0)),
        "warning_flags": warning_flags,
        "operator_notes": str(_row_value(row, "operator_notes", "") or ""),
        "raw_payload_path": _row_value(row, "raw_payload_path"),
        "materialized_path": _row_value(row, "materialized_path"),
        "manual_status": bool(_row_value(row, "manual_status", 0)),
        "manual_notes": bool(_row_value(row, "manual_notes", 0)),
        "reviewed_at": _row_value(row, "reviewed_at"),
        "reviewed_by": _row_value(row, "reviewed_by"),
        "entities": entities,
        "artifacts": artifacts,
    }


def _fetch_canonical_request(payment_id: str) -> Dict[str, Any]:
    with _connect() as conn:
        storage_kind = _canonical_storage_kind(conn)
        if storage_kind == "inbox_items":
            row = conn.execute(
                "SELECT * FROM inbox_items WHERE id = ? AND queue = ?",
                (payment_id, _PAYMENTS_QUEUE),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM payment_requests WHERE payment_id = ?",
                (payment_id,),
            ).fetchone()
    if row is None:
        raise KeyError(payment_id)
    return _inbox_item_to_request(row) if storage_kind == "inbox_items" else _row_to_request(row)


def _update_canonical_status(payment_id: str, status: str) -> Dict[str, Any]:
    with _connect() as conn:
        storage_kind = _canonical_storage_kind(conn)
        if storage_kind == "inbox_items":
            existing = conn.execute(
                "SELECT id FROM inbox_items WHERE id = ? AND queue = ?",
                (payment_id, _PAYMENTS_QUEUE),
            ).fetchone()
            if existing is None:
                raise KeyError(payment_id)
            conn.execute(
                """
                UPDATE inbox_items
                SET status = ?, updated_at = ?, manual_status = 1, reviewed_at = COALESCE(reviewed_at, ?)
                WHERE id = ? AND queue = ?
                """,
                (status, _now_iso(), _now_iso(), payment_id, _PAYMENTS_QUEUE),
            )
        else:
            existing = conn.execute(
                "SELECT payment_id FROM payment_requests WHERE payment_id = ?",
                (payment_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(payment_id)
            conn.execute(
                "UPDATE payment_requests SET status = ?, updated_at = ? WHERE payment_id = ?",
                (status, _now_iso(), payment_id),
            )
        conn.commit()
    return _fetch_canonical_request(payment_id)


def _source_statuses() -> List[Dict[str, Any]]:
    env = _load_runtime_env()
    email_address = str(env.get("EMAIL_ADDRESS") or "").strip()
    statuses = []
    for source_id, label in PAYMENT_SOURCES:
        if source_id == "gmail":
            connected = _google_token_path().exists()
            statuses.append(
                {
                    "id": source_id,
                    "label": label,
                    "connected": connected,
                    "detail": (
                        "Google Workspace token is present."
                        if connected
                        else "Connect Google Workspace to scan Gmail invoices."
                    ),
                }
            )
        elif source_id == "email":
            connected = bool(email_address)
            statuses.append(
                {
                    "id": source_id,
                    "label": label,
                    "connected": connected,
                    "detail": email_address if connected else "Configure the Email channel to ingest invoices.",
                }
            )
        elif source_id == "uploads":
            statuses.append(
                {
                    "id": source_id,
                    "label": label,
                    "connected": True,
                    "detail": "Use the Files page to stage invoice PDFs and screenshots.",
                }
            )
        elif source_id == "slack":
            connected = _slack_token_present()
            statuses.append(
                {
                    "id": source_id,
                    "label": label,
                    "connected": connected,
                    "detail": (
                        "Slack bot token is configured."
                        if connected
                        else "Slack mobile inbox requires the Slack gateway bot token."
                    ),
                }
            )
    return statuses


def _legacy_requests_path() -> Path:
    return get_hermes_home() / "payments" / "requests.json"


def _shadow_report_path() -> Path:
    return get_hermes_home() / "payments" / "shadow-report.json"


def _normalize_legacy_request(raw: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    source = str(raw.get("source") or "uploads").strip() or "uploads"
    amount = raw.get("amount") if isinstance(raw.get("amount"), dict) else {}
    request = {
        "id": str(raw.get("id") or f"payment-{index + 1}"),
        "payment_id": str(raw.get("id") or f"payment-{index + 1}"),
        "source": source,
        "source_label": _source_label(source),
        "status": str(raw.get("status") or "needs_review"),
        "operator_status": _operator_status(str(raw.get("status") or "needs_review")),
        "confidence": str(raw.get("confidence") or "low"),
        "received_at": raw.get("received_at"),
        "updated_at": raw.get("updated_at"),
        "vendor": str(raw.get("vendor") or ""),
        "title": str(raw.get("title") or ""),
        "amount": {
            "value": amount.get("value"),
            "currency": str(amount.get("currency") or ""),
            "display": str(amount.get("display") or ""),
        },
        "due_date": _format_due_date(raw.get("due_date")),
        "payee_name": str(raw.get("payee_name") or ""),
        "account_holder": str(raw.get("account_holder") or ""),
        "account_number": str(raw.get("account_number") or ""),
        "sort_code": str(raw.get("sort_code") or ""),
        "iban": str(raw.get("iban") or ""),
        "swift": str(raw.get("swift") or ""),
        "routing_number": str(raw.get("routing_number") or ""),
        "payment_reference": str(raw.get("payment_reference") or ""),
        "invoice_number": str(raw.get("invoice_number") or ""),
        "billing_address": str(raw.get("billing_address") or ""),
        "tax_details": str(raw.get("tax_details") or ""),
        "preview_text": str(raw.get("preview_text") or ""),
        "warnings": [str(item) for item in (raw.get("warnings") or [])],
        "attachments": [str(item) for item in (raw.get("attachments") or [])],
        "original": raw.get("original") if isinstance(raw.get("original"), dict) else {},
        "raw_text_path": str(raw.get("raw_text_path") or ""),
        "materialized_path": str(raw.get("materialized_path") or ""),
        "review_note": str(raw.get("review_note") or ""),
        "looks_paid": False,
    }
    request["looks_paid"] = _looks_paid(request)
    return request


def _load_legacy_requests() -> list[Dict[str, Any]]:
    path = _legacy_requests_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("requests") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [
        _normalize_legacy_request(item, index=i)
        for i, item in enumerate(items)
        if isinstance(item, dict)
    ]


def _save_legacy_requests(requests: list[Dict[str, Any]]) -> None:
    path = _legacy_requests_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"requests": requests, "updated_at": _now_iso()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_payment_requests() -> Dict[str, Any]:
    if _canonical_enabled():
        with _connect() as conn:
            storage_kind = _canonical_storage_kind(conn, prefer="payment_requests")
            if storage_kind == "inbox_items":
                rows = conn.execute(
                    """
                    SELECT * FROM inbox_items
                    WHERE queue = ?
                    ORDER BY captured_at DESC, updated_at DESC, id DESC
                    """,
                    (_PAYMENTS_QUEUE,),
                ).fetchall()
                requests = [_inbox_item_to_request(row) for row in rows]
            else:
                rows = conn.execute(
                    "SELECT * FROM payment_requests ORDER BY captured_at DESC, updated_at DESC, payment_id DESC"
                ).fetchall()
                requests = [_row_to_request(row) for row in rows]
    else:
        requests = _load_legacy_requests()
    return {
        "sources": _source_statuses(),
        "requests": requests,
        "storage_path": _storage_display_path(),
    }


def list_inbox_items(
    *,
    queue: str | None = None,
    workflow_type: str | None = None,
) -> Dict[str, Any]:
    normalized_queue = str(queue or "").strip().lower()
    normalized_workflow = str(workflow_type or "").strip().lower()
    if _canonical_enabled():
        with _connect() as conn:
            storage_kind = _canonical_storage_kind(conn, prefer="inbox_items")
            if storage_kind == "inbox_items":
                clauses: list[str] = []
                params: list[Any] = []
                if normalized_queue:
                    clauses.append("queue = ?")
                    params.append(normalized_queue)
                if normalized_workflow:
                    clauses.append("workflow_type = ?")
                    params.append(normalized_workflow)
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                rows = conn.execute(
                    f"SELECT * FROM inbox_items {where} ORDER BY captured_at DESC, updated_at DESC, id DESC",
                    tuple(params),
                ).fetchall()
                items = [_row_to_inbox_item(row) for row in rows]
            else:
                rows = conn.execute(
                    "SELECT * FROM payment_requests ORDER BY captured_at DESC, updated_at DESC, payment_id DESC"
                ).fetchall()
                items = [_request_to_inbox_item(_row_to_request(row)) for row in rows]
        if normalized_queue:
            items = [item for item in items if item["queue"] == normalized_queue]
        if normalized_workflow:
            items = [item for item in items if item["workflow_type"] == normalized_workflow]
    else:
        items = [_request_to_inbox_item(request) for request in _load_legacy_requests()]
        if normalized_queue:
            items = [item for item in items if item["queue"] == normalized_queue]
        if normalized_workflow:
            items = [item for item in items if item["workflow_type"] == normalized_workflow]
    return {
        "items": items,
        "storage_path": _storage_display_path(),
    }


def sync_gmail_payment_requests(
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> Dict[str, Any]:
    if _canonical_enabled():
        runtime = _payments_review_runtime()
        payload = json.loads(
            runtime.ingest_payments_from_gmail(
                {"query_override": query, "max_threads": int(max_results)}
            )
        )
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "payments ingest failed"))
        requests = []
        imported = 0
        updated = 0
        for item in payload.get("payments", []):
            request = dict(item)
            request["id"] = request.get("payment_id")
            request["source_label"] = _source_label(str(request.get("source") or "gmail"))
            request["operator_status"] = _operator_status(str(request.get("status") or "needs_review"))
            request["looks_paid"] = _looks_paid(request)
            requests.append(request)
            if request.get("duplicate"):
                updated += 1
            else:
                imported += 1
        return {
            "source": "gmail",
            "query": str(payload.get("query") or query),
            "fetched": int(payload.get("thread_count") or 0),
            "imported": imported,
            "updated": updated,
            "requests": requests,
        }

    raise RuntimeError("Canonical payments_review storage is not configured")


def sync_gmail_payment_requests_shadow(
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> Dict[str, Any]:
    result = sync_gmail_payment_requests(query=query, max_results=max_results)
    if not _canonical_enabled():
        raise RuntimeError("Canonical payments_review storage is not configured")

    mirrored = 0
    with _connect() as conn:
        _ensure_inbox_items_schema(conn)
        for request in list_payment_requests()["requests"]:
            item = _request_to_inbox_item(request)
            _upsert_inbox_item(conn, item)
            mirrored += 1
        conn.commit()

    report = generate_payments_shadow_report()
    response = {
        **result,
        "shadow": {
            "mirrored": mirrored,
            "storage_path": _storage_display_path(),
            "parity": report,
        },
    }
    _save_shadow_report_snapshot(response)
    return response


def generate_payments_shadow_report() -> Dict[str, Any]:
    payments = {item["id"]: item for item in list_payment_requests()["requests"]}
    inbox_items = {
        item["id"]: item for item in list_inbox_items(queue=_PAYMENTS_QUEUE)["items"]
    }
    all_ids = sorted(set(payments) | set(inbox_items))
    status_mismatches: list[str] = []
    amount_mismatches: list[str] = []
    due_date_mismatches: list[str] = []
    reference_mismatches: list[str] = []
    missing_in_shadow: list[str] = []
    shadow_only: list[str] = []
    for item_id in all_ids:
        payment = payments.get(item_id)
        shadow = inbox_items.get(item_id)
        if payment is None:
            shadow_only.append(item_id)
            continue
        if shadow is None:
            missing_in_shadow.append(item_id)
            continue
        if str(payment.get("status") or "") != str(shadow.get("status") or ""):
            status_mismatches.append(item_id)
        if str((payment.get("amount") or {}).get("display") or "") != str((shadow.get("amount") or {}).get("display") or ""):
            amount_mismatches.append(item_id)
        if str(payment.get("due_date") or "") != str(shadow.get("due_date") or ""):
            due_date_mismatches.append(item_id)
        if str(payment.get("payment_reference") or "") != str(shadow.get("reference_id") or ""):
            reference_mismatches.append(item_id)

    compared = len([item_id for item_id in all_ids if item_id in payments and item_id in inbox_items])
    parity_ok = not any(
        [
            status_mismatches,
            amount_mismatches,
            due_date_mismatches,
            reference_mismatches,
            missing_in_shadow,
        ]
    )
    return {
        "generated_at": _now_iso(),
        "payments_count": len(payments),
        "shadow_count": len(inbox_items),
        "compared_count": compared,
        "parity_ok": parity_ok,
        "status_mismatches": status_mismatches,
        "amount_mismatches": amount_mismatches,
        "due_date_mismatches": due_date_mismatches,
        "reference_mismatches": reference_mismatches,
        "missing_in_shadow": missing_in_shadow,
        "shadow_only": shadow_only,
    }


def _save_shadow_report_snapshot(result: Dict[str, Any]) -> None:
    path = _shadow_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": _now_iso(),
        "result": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_shadow_report_snapshot() -> Dict[str, Any]:
    path = _shadow_report_path()
    if not path.exists():
        return {
            "updated_at": None,
            "result": None,
            "path": str(path),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return {
        "updated_at": payload.get("updated_at"),
        "result": payload.get("result"),
        "path": str(path),
    }


def format_sync_summary(result: Dict[str, Any]) -> str:
    return (
        "synced gmail payments"
        f" fetched={int(result.get('fetched') or 0)}"
        f" imported={int(result.get('imported') or 0)}"
        f" updated={int(result.get('updated') or 0)}"
    )


def format_shadow_summary(result: Dict[str, Any]) -> str:
    shadow = result.get("shadow") or {}
    parity = shadow.get("parity") or {}
    return (
        f"{format_sync_summary(result)}"
        f" shadow_mirrored={int(shadow.get('mirrored') or 0)}"
        f" parity_ok={bool(parity.get('parity_ok'))}"
        f" compared={int(parity.get('compared_count') or 0)}"
        f" payments={int(parity.get('payments_count') or 0)}"
        f" shadow={int(parity.get('shadow_count') or 0)}"
    )


def render_gmail_sync_cron_script(
    *,
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> str:
    return (
        "#!/usr/bin/env python3\n"
        "\"\"\"Managed Hermes cron script: sync Gmail payments into the canonical store.\"\"\"\n"
        "\n"
        "from hermes_cli.payments import format_sync_summary, sync_gmail_payment_requests\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        f"    result = sync_gmail_payment_requests(query={query!r}, max_results={int(max_results)})\n"
        "    print(format_sync_summary(result))\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main())\n"
    )


def render_gmail_shadow_sync_cron_script(
    *,
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> str:
    return (
        "#!/usr/bin/env python3\n"
        "\"\"\"Managed Hermes cron script: shadow-sync Gmail payments into inbox_items.\"\"\"\n"
        "\n"
        "from hermes_cli.payments import format_shadow_summary, sync_gmail_payment_requests_shadow\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        f"    result = sync_gmail_payment_requests_shadow(query={query!r}, max_results={int(max_results)})\n"
        "    print(format_shadow_summary(result))\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    raise SystemExit(main())\n"
    )


def install_gmail_sync_cron_script(
    *,
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> Path:
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / PAYMENTS_GMAIL_SYNC_CRON_SCRIPT
    content = render_gmail_sync_cron_script(query=query, max_results=max_results)
    script_path.write_text(content, encoding="utf-8")
    try:
        script_path.chmod(0o700)
    except OSError:
        pass
    return script_path


def install_gmail_shadow_sync_cron_script(
    *,
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> Path:
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / PAYMENTS_GMAIL_SHADOW_SYNC_CRON_SCRIPT
    content = render_gmail_shadow_sync_cron_script(query=query, max_results=max_results)
    script_path.write_text(content, encoding="utf-8")
    try:
        script_path.chmod(0o700)
    except OSError:
        pass
    return script_path


def ensure_gmail_sync_cron_job(
    *,
    schedule: str = DEFAULT_GMAIL_SYNC_SCHEDULE,
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
    name: str = PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME,
) -> Dict[str, Any]:
    from cron.jobs import create_job, resolve_job_ref, update_job

    install_gmail_sync_cron_script(query=query, max_results=max_results)
    spec = {
        "prompt": "Sync Gmail payment requests into the canonical payments review store.",
        "schedule": schedule,
        "name": str(name or PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME).strip() or PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME,
        "deliver": "local",
        "script": PAYMENTS_GMAIL_SYNC_CRON_SCRIPT,
        "no_agent": True,
    }
    existing = resolve_job_ref(spec["name"])
    if existing is not None:
        updated = update_job(existing["id"], spec)
        if updated is None:
            raise RuntimeError(f"Failed to update cron job: {existing['id']}")
        return updated
    return create_job(**spec)


def ensure_gmail_shadow_sync_cron_job(
    *,
    schedule: str = DEFAULT_GMAIL_SHADOW_SYNC_SCHEDULE,
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
    name: str = PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME,
) -> Dict[str, Any]:
    from cron.jobs import create_job, resolve_job_ref, update_job

    install_gmail_shadow_sync_cron_script(query=query, max_results=max_results)
    spec = {
        "prompt": "Shadow-sync Gmail payment requests into inbox_items and capture parity metrics.",
        "schedule": schedule,
        "name": str(name or PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME).strip()
        or PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME,
        "deliver": "local",
        "script": PAYMENTS_GMAIL_SHADOW_SYNC_CRON_SCRIPT,
        "no_agent": True,
    }
    existing = resolve_job_ref(spec["name"])
    if existing is not None:
        updated = update_job(existing["id"], spec)
        if updated is None:
            raise RuntimeError(f"Failed to update cron job: {existing['id']}")
        return updated
    return create_job(**spec)


def payments_command(args: Any) -> int:
    subcommand = str(getattr(args, "payments_command", "") or "").strip()
    if not subcommand:
        print(
            "usage: hermes payments <subcommand>\n"
            "\n"
            "subcommands:\n"
            "  sync-gmail             Run the Gmail -> canonical payments sync now\n"
            "  schedule-gmail-sync    Install/update a recurring Hermes cron job\n",
            file=sys.stderr,
        )
        return 1

    if subcommand == "sync-gmail":
        result = sync_gmail_payment_requests(
            query=str(getattr(args, "query", DEFAULT_GMAIL_QUERY) or DEFAULT_GMAIL_QUERY),
            max_results=int(getattr(args, "max_results", DEFAULT_GMAIL_MAX_RESULTS) or DEFAULT_GMAIL_MAX_RESULTS),
        )
        print(format_sync_summary(result))
        return 0

    if subcommand == "shadow-sync-gmail":
        result = sync_gmail_payment_requests_shadow(
            query=str(getattr(args, "query", DEFAULT_GMAIL_QUERY) or DEFAULT_GMAIL_QUERY),
            max_results=int(getattr(args, "max_results", DEFAULT_GMAIL_MAX_RESULTS) or DEFAULT_GMAIL_MAX_RESULTS),
        )
        print(format_shadow_summary(result))
        return 0

    if subcommand == "shadow-report":
        print(json.dumps(generate_payments_shadow_report(), indent=2, sort_keys=True))
        return 0

    if subcommand == "schedule-shadow-sync":
        job = ensure_gmail_shadow_sync_cron_job(
            schedule=str(
                getattr(args, "schedule", DEFAULT_GMAIL_SHADOW_SYNC_SCHEDULE)
                or DEFAULT_GMAIL_SHADOW_SYNC_SCHEDULE
            ),
            query=str(getattr(args, "query", DEFAULT_GMAIL_QUERY) or DEFAULT_GMAIL_QUERY),
            max_results=int(
                getattr(args, "max_results", DEFAULT_GMAIL_MAX_RESULTS)
                or DEFAULT_GMAIL_MAX_RESULTS
            ),
            name=str(
                getattr(args, "name", PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME)
                or PAYMENTS_GMAIL_SHADOW_SYNC_CRON_JOB_NAME
            ),
        )
        print(
            f"scheduled payments gmail shadow sync job id={job['id']} "
            f"schedule={job.get('schedule_display') or job.get('schedule')}"
        )
        if bool(getattr(args, "run_now", False)):
            result = sync_gmail_payment_requests_shadow(
                query=str(getattr(args, "query", DEFAULT_GMAIL_QUERY) or DEFAULT_GMAIL_QUERY),
                max_results=int(
                    getattr(args, "max_results", DEFAULT_GMAIL_MAX_RESULTS)
                    or DEFAULT_GMAIL_MAX_RESULTS
                ),
            )
            print(format_shadow_summary(result))
        return 0

    if subcommand == "schedule-gmail-sync":
        job = ensure_gmail_sync_cron_job(
            schedule=str(getattr(args, "schedule", DEFAULT_GMAIL_SYNC_SCHEDULE) or DEFAULT_GMAIL_SYNC_SCHEDULE),
            query=str(getattr(args, "query", DEFAULT_GMAIL_QUERY) or DEFAULT_GMAIL_QUERY),
            max_results=int(getattr(args, "max_results", DEFAULT_GMAIL_MAX_RESULTS) or DEFAULT_GMAIL_MAX_RESULTS),
            name=str(getattr(args, "name", PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME) or PAYMENTS_GMAIL_SYNC_CRON_JOB_NAME),
        )
        print(
            f"scheduled payments gmail sync job id={job['id']} "
            f"schedule={job.get('schedule_display') or job.get('schedule')}"
        )
        if bool(getattr(args, "run_now", False)):
            result = sync_gmail_payment_requests(
                query=str(getattr(args, "query", DEFAULT_GMAIL_QUERY) or DEFAULT_GMAIL_QUERY),
                max_results=int(getattr(args, "max_results", DEFAULT_GMAIL_MAX_RESULTS) or DEFAULT_GMAIL_MAX_RESULTS),
            )
            print(format_sync_summary(result))
        return 0

    print(f"Unknown payments subcommand: {subcommand}", file=sys.stderr)
    return 1


def update_payment_status(payment_id: str, status: str) -> Dict[str, Any]:
    normalized_status = str(status or "").strip()
    if normalized_status not in PAYMENT_STATUSES:
        raise ValueError(f"Unknown payment status: {status}")

    if _canonical_enabled():
        return _update_canonical_status(payment_id, normalized_status)

    requests = _load_legacy_requests()
    for record in requests:
        if str(record.get("id")) == payment_id:
            record["status"] = normalized_status
            record["operator_status"] = _operator_status(normalized_status)
            record["updated_at"] = _now_iso()
            _save_legacy_requests(requests)
            return record
    raise KeyError(payment_id)


def update_inbox_item_status(item_id: str, status: str) -> Dict[str, Any]:
    normalized_status = str(status or "").strip()
    if normalized_status not in INBOX_ITEM_STATUSES:
        raise ValueError(f"Unknown inbox item status: {status}")

    if _canonical_enabled():
        with _connect() as conn:
            storage_kind = _canonical_storage_kind(conn, prefer="inbox_items")
            if storage_kind == "inbox_items":
                existing = conn.execute(
                    "SELECT id FROM inbox_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
                if existing is None:
                    raise KeyError(item_id)
                now = _now_iso()
                conn.execute(
                    """
                    UPDATE inbox_items
                    SET status = ?, updated_at = ?, manual_status = 1, reviewed_at = COALESCE(reviewed_at, ?)
                    WHERE id = ?
                    """,
                    (normalized_status, now, now, item_id),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM inbox_items WHERE id = ?", (item_id,)).fetchone()
                if row is None:
                    raise KeyError(item_id)
                return _row_to_inbox_item(row)

    updated = update_payment_status(item_id, normalized_status)
    return _request_to_inbox_item(updated)


def _summarize_payment_line(payment: Dict[str, Any]) -> str:
    amount = payment["amount"]["display"] or "Amount missing"
    due = payment.get("due_date") or "No due date"
    warning = " receipt-like" if payment.get("looks_paid") else ""
    status = _status_label(str(payment.get("operator_status") or payment.get("status") or "needs_review"))
    return (
        f"- **{payment.get('vendor') or payment.get('title') or payment.get('id')}**"
        f" · {amount} · due {due} · {status}{warning}"
    )


def render_slack_canvas_markdown(
    *,
    dashboard_url: str = "",
    per_status_limit: int = 8,
) -> str:
    payload = list_payment_requests()
    requests = payload["requests"]
    sections = [
        ("Needs review", [item for item in requests if item["operator_status"] == "needs_review"]),
        ("Ready to pay", [item for item in requests if item["operator_status"] == "ready_to_pay"]),
        ("Recently paid", [item for item in requests if item["status"] == "paid"]),
    ]
    lines = [
        "# Payments Ops",
        "",
        f"_Refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "This canvas is a compact summary. Use the dashboard or Slack action inbox to make updates.",
    ]
    if dashboard_url.strip():
        lines.extend(["", f"[Open dashboard payments board]({dashboard_url.strip()})"])
    for title, items in sections:
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append("- None")
            continue
        for payment in items[:per_status_limit]:
            lines.append(_summarize_payment_line(payment))
    return "\n".join(lines).strip() + "\n"


def render_slack_canvas_spec(
    *,
    channel_key: str,
    channel_name: str,
    dashboard_url: str = "",
    canvas_title: str = "Payments Ops",
    per_status_limit: int = 8,
) -> str:
    markdown = render_slack_canvas_markdown(
        dashboard_url=dashboard_url,
        per_status_limit=per_status_limit,
    ).rstrip("\n")
    indented_markdown = "\n".join(f"          {line}" if line else "" for line in markdown.splitlines())
    return (
        "channels:\n"
        f"  - key: {channel_key}\n"
        f"    name: {json.dumps(channel_name)}\n"
        "    canvases:\n"
        f"      - key: payments-ops\n"
        f"        title: {json.dumps(canvas_title)}\n"
        "        channel_canvas: true\n"
        "        markdown: |\n"
        f"{indented_markdown}\n"
    )


def _slack_payment_summary(payment: Dict[str, Any]) -> str:
    title = payment.get("vendor") or payment.get("title") or payment.get("id")
    amount = payment["amount"]["display"] or "Amount missing"
    due = payment.get("due_date") or "No due date"
    status = _status_label(str(payment.get("operator_status") or payment.get("status") or "needs_review"))
    suffix = " · receipt-like" if payment.get("looks_paid") else ""
    return f"*{title}*\n{amount} · due {due} · {status}{suffix}"


def build_slack_status_modal_view(
    payment: Dict[str, Any],
    *,
    private_metadata: str = "",
) -> Dict[str, Any]:
    payment_id = str(payment["id"])
    return {
        "type": "modal",
        "callback_id": "payments_status_modal",
        "title": {"type": "plain_text", "text": "Update payment", "emoji": True},
        "close": {"type": "plain_text", "text": "Close", "emoji": True},
        "private_metadata": private_metadata,
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": _slack_payment_summary(payment)},
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "This updates the canonical payments review store. No payment is executed automatically.",
                    }
                ],
            },
            {
                "type": "actions",
                "block_id": f"payment-status:{payment_id}",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "payments_mark_review",
                        "text": {"type": "plain_text", "text": "Needs review"},
                        "value": payment_id,
                    },
                    {
                        "type": "button",
                        "action_id": "payments_mark_ready",
                        "text": {"type": "plain_text", "text": "Ready to pay"},
                        "value": payment_id,
                    },
                    {
                        "type": "button",
                        "action_id": "payments_mark_paid",
                        "text": {"type": "plain_text", "text": "Paid"},
                        "style": "primary",
                        "value": payment_id,
                    },
                    {
                        "type": "button",
                        "action_id": "payments_mark_ignored",
                        "text": {"type": "plain_text", "text": "Ignored"},
                        "style": "danger",
                        "value": payment_id,
                    },
                ],
            },
        ],
    }


def build_slack_mobile_blocks(
    *,
    statuses: tuple[str, ...] = ("needs_review", "ready_to_pay"),
    per_status_limit: int = 5,
) -> list[dict[str, Any]]:
    payload = list_payment_requests()
    requests = payload["requests"]
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Payments mobile inbox", "emoji": True},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Refreshed {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · status updates are local only",
                }
            ],
        },
    ]
    for status in statuses:
        label = {
            "needs_review": "Needs review",
            "ready_to_pay": "Ready to pay",
            "paid": "Recently paid",
            "ignored": "Ignored",
        }.get(status, _status_label(status))
        items = [item for item in requests if item["operator_status"] == status or item["status"] == status]
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{label}*"},
            }
        )
        if not items:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "_None_"}],
                }
            )
            continue
        for payment in items[:per_status_limit]:
            title = payment.get("vendor") or payment.get("title") or payment.get("id")
            amount = payment["amount"]["display"] or "Amount missing"
            due = payment.get("due_date") or "No due date"
            blocks.append(
                {
                    "type": "section",
                    "block_id": f"payment:{payment['id']}",
                    "text": {
                        "type": "mrkdwn",
                        "text": _slack_payment_summary(payment),
                    },
                    "accessory": {
                        "type": "button",
                        "action_id": "payments_open_status_modal",
                        "text": {"type": "plain_text", "text": "Update status"},
                        "value": payment["id"],
                    },
                }
            )
    return blocks
