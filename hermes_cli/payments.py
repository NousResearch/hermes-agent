"""Profile-scoped payment-request storage for the dashboard Payments page.

The first slice is intentionally narrow:
  * keep a durable, profile-local review queue of extracted payment requests
  * expose source status (Gmail / Email / Uploads / Slack) for the dashboard
  * support manual status transitions only; no payment initiation

Actual ingestion/extraction can be layered on top of this store later without
changing the dashboard contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List

from hermes_constants import get_hermes_home
from hermes_cli.config import load_env

PAYMENTS_DIR = "payments"
REQUESTS_FILE = "requests.json"

PAYMENT_STATUSES = {
    "new",
    "needs_review",
    "ready_to_pay",
    "paid",
    "ignored",
}

PAYMENT_SOURCES = (
    ("gmail", "Gmail"),
    ("email", "Email"),
    ("uploads", "Uploads"),
    ("slack", "Slack"),
)

DEFAULT_GMAIL_QUERY = (
    'newer_than:120d (invoice OR "payment due" OR "request for payment" OR '
    '"bank transfer" OR remittance OR billing OR statement)'
)
DEFAULT_GMAIL_MAX_RESULTS = 25

_AMOUNT_PATTERNS = [
    re.compile(r"\b(GBP|USD|EUR|AUD|CAD|NZD|CHF|SEK|NOK|DKK)\s?([0-9][0-9,]*(?:\.[0-9]{2})?)\b", re.I),
    re.compile(r"([£$€])\s?([0-9][0-9,]*(?:\.[0-9]{2})?)"),
]
_SORT_CODE_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
_ACCOUNT_NUMBER_RE = re.compile(r"\b\d{8}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b")
_SWIFT_RE = re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
_ROUTING_RE = re.compile(r"\brouting number\b[:\s]*([0-9]{9})", re.I)
_REFERENCE_RE = re.compile(
    r"\b(?:reference|payment reference|remittance reference)\b[:\s#-]*([A-Z0-9-]{4,})",
    re.I,
)
_INVOICE_RE = re.compile(r"\b(?:invoice|inv(?:oice)? number|invoice #)\b[:\s#-]*([A-Z0-9-]{3,})", re.I)
_DUE_RE = re.compile(
    r"\b(?:due date|payment due|due)\b[:\s-]*("
    r"[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r")",
    re.I,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payments_dir() -> Path:
    return get_hermes_home() / PAYMENTS_DIR


def _requests_path() -> Path:
    return _payments_dir() / REQUESTS_FILE


def _google_api_script() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "productivity"
        / "google-workspace"
        / "scripts"
        / "google_api.py"
    )


def _empty_record(payment_id: str) -> Dict[str, Any]:
    return {
        "id": payment_id,
        "source": "uploads",
        "source_label": "Uploads",
        "status": "new",
        "confidence": "low",
        "received_at": None,
        "updated_at": None,
        "vendor": "",
        "title": "",
        "amount": {"value": None, "currency": "", "display": ""},
        "due_date": None,
        "payee_name": "",
        "account_holder": "",
        "account_number": "",
        "sort_code": "",
        "iban": "",
        "swift": "",
        "routing_number": "",
        "payment_reference": "",
        "invoice_number": "",
        "billing_address": "",
        "tax_details": "",
        "preview_text": "",
        "warnings": [],
        "attachments": [],
        "original": {"label": "", "url": "", "message_id": "", "thread_id": ""},
    }


def _normalize_record(raw: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    record = _empty_record(str(raw.get("id") or f"payment-{index + 1}"))
    record.update({k: v for k, v in raw.items() if k in record})

    amount = raw.get("amount")
    if isinstance(amount, dict):
        record["amount"] = {
            "value": amount.get("value"),
            "currency": str(amount.get("currency") or ""),
            "display": str(amount.get("display") or ""),
        }

    original = raw.get("original")
    if isinstance(original, dict):
        record["original"] = {
            "label": str(original.get("label") or ""),
            "url": str(original.get("url") or ""),
            "message_id": str(original.get("message_id") or ""),
            "thread_id": str(original.get("thread_id") or ""),
        }

    status = str(record.get("status") or "new").strip()
    record["status"] = status if status in PAYMENT_STATUSES else "needs_review"

    source = str(record.get("source") or "uploads").strip()
    record["source"] = source or "uploads"
    record["source_label"] = str(record.get("source_label") or source.title())
    record["confidence"] = str(record.get("confidence") or "low")
    record["warnings"] = [
        str(item).strip()
        for item in (record.get("warnings") or [])
        if str(item).strip()
    ]
    record["attachments"] = [
        str(item).strip()
        for item in (record.get("attachments") or [])
        if str(item).strip()
    ]
    return record


def _run_google_api(args: List[str]) -> Any:
    script = _google_api_script()
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip() or "google_api.py failed"
        raise RuntimeError(err)
    stdout = result.stdout.strip()
    if not stdout:
        return []
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected google_api.py output: {stdout[:200]}") from exc


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _format_vendor(sender: str) -> str:
    name, addr = parseaddr(sender or "")
    if name.strip():
        return name.strip().strip('"')
    if addr:
        return addr.split("@", 1)[0]
    return sender.strip()


def _parse_amount(text: str) -> Dict[str, Any]:
    for pattern in _AMOUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if len(match.groups()) == 2:
            g1, g2 = match.groups()
            symbol_currency = {"£": "GBP", "$": "USD", "€": "EUR"}
            currency = symbol_currency.get(g1, g1.upper())
            display = f"{currency} {g2}"
            try:
                value = float(g2.replace(",", ""))
            except ValueError:
                value = None
            return {"value": value, "currency": currency, "display": display}
    return {"value": None, "currency": "", "display": ""}


def _extract_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    if match.lastindex:
        return str(match.group(1) or "").strip()
    return str(match.group(0) or "").strip()


def _parse_received_at(value: str) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).astimezone(timezone.utc).isoformat()
    except Exception:
        return text


def _confidence_for(record: Dict[str, Any]) -> str:
    score = 0
    if record["amount"]["display"]:
        score += 1
    if record.get("invoice_number"):
        score += 1
    if record.get("payment_reference"):
        score += 1
    if any(
        record.get(key)
        for key in ("account_number", "sort_code", "iban", "swift", "routing_number")
    ):
        score += 2
    if record.get("due_date"):
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _warnings_for(record: Dict[str, Any]) -> List[str]:
    warnings = []
    if not record["amount"]["display"]:
        warnings.append("No amount detected.")
    if not any(
        record.get(key)
        for key in ("account_number", "sort_code", "iban", "swift", "routing_number")
    ):
        warnings.append("No bank account details detected.")
    if not record.get("payment_reference") and not record.get("invoice_number"):
        warnings.append("No invoice or payment reference detected.")
    return warnings


def _build_gmail_payment_record(summary: Dict[str, Any], detail: Dict[str, Any], existing: Dict[str, Any] | None) -> Dict[str, Any]:
    subject = str(detail.get("subject") or summary.get("subject") or "").strip()
    body = str(detail.get("body") or "").strip()
    snippet = str(summary.get("snippet") or "").strip()
    combined = "\n".join(part for part in (subject, snippet, body) if part).strip()
    vendor = _format_vendor(str(detail.get("from") or summary.get("from") or ""))
    amount = _parse_amount(combined)
    record = _empty_record(f"gmail:{detail['id']}")
    record.update(
        {
            "source": "gmail",
            "source_label": "Gmail",
            "status": existing.get("status") if existing else "needs_review",
            "received_at": _parse_received_at(str(detail.get("date") or summary.get("date") or "")),
            "updated_at": _now_iso(),
            "vendor": vendor,
            "title": subject,
            "amount": amount,
            "due_date": _extract_match(_DUE_RE, combined) or None,
            "payee_name": vendor,
            "account_holder": vendor,
            "account_number": _extract_match(_ACCOUNT_NUMBER_RE, combined),
            "sort_code": _extract_match(_SORT_CODE_RE, combined),
            "iban": _extract_match(_IBAN_RE, combined),
            "swift": _extract_match(_SWIFT_RE, combined),
            "routing_number": _extract_match(_ROUTING_RE, combined),
            "payment_reference": _extract_match(_REFERENCE_RE, combined),
            "invoice_number": _extract_match(_INVOICE_RE, combined),
            "preview_text": _clean_text(body or snippet)[:1500],
            "attachments": existing.get("attachments", []) if existing else [],
            "original": {
                "label": str(detail.get("from") or summary.get("from") or ""),
                "url": f"https://mail.google.com/mail/u/0/#all/{detail['id']}",
                "message_id": str(detail.get("id") or ""),
                "thread_id": str(detail.get("threadId") or summary.get("threadId") or ""),
            },
        }
    )
    record["confidence"] = _confidence_for(record)
    record["warnings"] = _warnings_for(record)
    return record


def _load_requests() -> List[Dict[str, Any]]:
    path = _requests_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("requests") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    return [_normalize_record(item, index=i) for i, item in enumerate(items) if isinstance(item, dict)]


def _save_requests(requests: List[Dict[str, Any]]) -> None:
    path = _requests_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requests": requests,
        "updated_at": _now_iso(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_statuses() -> List[Dict[str, Any]]:
    env = load_env()
    home = get_hermes_home()

    gmail_token = home / "google_token.json"
    email_address = str(env.get("EMAIL_ADDRESS") or "").strip()
    slack_token = str(env.get("SLACK_BOT_TOKEN") or "").strip()

    statuses = []
    for source_id, label in PAYMENT_SOURCES:
        if source_id == "gmail":
            connected = gmail_token.exists()
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
            connected = bool(slack_token)
            statuses.append(
                {
                    "id": source_id,
                    "label": label,
                    "connected": connected,
                    "detail": (
                        "Slack bot token is configured."
                        if connected
                        else "Slack capture can be added once the Slack channel is configured."
                    ),
                }
            )
    return statuses


def list_payment_requests() -> Dict[str, Any]:
    requests = _load_requests()
    requests.sort(
        key=lambda item: (
            str(item.get("received_at") or ""),
            str(item.get("updated_at") or ""),
            str(item.get("id") or ""),
        ),
        reverse=True,
    )
    return {
        "sources": _source_statuses(),
        "requests": requests,
        "storage_path": str(_requests_path()),
    }


def sync_gmail_payment_requests(
    query: str = DEFAULT_GMAIL_QUERY,
    max_results: int = DEFAULT_GMAIL_MAX_RESULTS,
) -> Dict[str, Any]:
    summaries = _run_google_api(["gmail", "search", query, "--max", str(max_results)])
    if not isinstance(summaries, list):
        raise RuntimeError("Gmail search returned an unexpected payload")

    requests = _load_requests()
    existing_by_id = {str(record.get("id")): record for record in requests}
    imported = 0
    updated = 0

    for summary in summaries:
        if not isinstance(summary, dict) or not summary.get("id"):
            continue
        detail = _run_google_api(["gmail", "get", str(summary["id"])])
        if not isinstance(detail, dict) or not detail.get("id"):
            continue
        record_id = f"gmail:{detail['id']}"
        existing = existing_by_id.get(record_id)
        record = _build_gmail_payment_record(summary, detail, existing)
        if existing is None:
            requests.append(record)
            existing_by_id[record_id] = record
            imported += 1
        else:
            existing.update(record)
            updated += 1

    _save_requests(requests)
    return {
        "source": "gmail",
        "query": query,
        "fetched": len(summaries),
        "imported": imported,
        "updated": updated,
        "requests": requests,
    }


def update_payment_status(payment_id: str, status: str) -> Dict[str, Any]:
    normalized_status = str(status or "").strip()
    if normalized_status not in PAYMENT_STATUSES:
        raise ValueError(f"Unknown payment status: {status}")

    requests = _load_requests()
    for record in requests:
        if str(record.get("id")) == payment_id:
            record["status"] = normalized_status
            record["updated_at"] = _now_iso()
            _save_requests(requests)
            return record
    raise KeyError(payment_id)
