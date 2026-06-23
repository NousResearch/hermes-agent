"""Safe Codex service-tier diagnostics.

This module writes a tiny JSONL audit trail for the ChatGPT Codex OAuth
Responses path so users can verify whether Hermes requested Priority
Processing and what tier the backend reported in the response.

Privacy boundary: never persist prompts, message bodies, authorization headers,
raw session ids, raw request/response ids, account ids, cookies, or tokens.
Correlators are short SHA-256 hashes only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

LOG_FILENAME = "codex-service-tier.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hash_identifier(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"sha256:{digest[:16]}"


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _log_path() -> Path:
    return get_hermes_home() / "logs" / LOG_FILENAME


def _append_record(record: dict[str, Any]) -> None:
    """Best-effort append; diagnostics must never break model calls."""
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        sanitized = {k: v for k, v in record.items() if v is not None}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def record_codex_service_tier_request(
    *,
    model: Any = None,
    issuer_kind: Any = None,
    requested_service_tier: Any = None,
    session_id: Any = None,
    request_id: Any = None,
    source: str = "transport.build_kwargs",
) -> None:
    """Record the safe shape of a Codex service-tier request."""
    requested = _clean_str(requested_service_tier)
    if not requested:
        return
    _append_record(
        {
            "ts": _now_iso(),
            "event": "request",
            "source": source,
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "model": _clean_str(model),
            "issuer_kind": _clean_str(issuer_kind),
            "requested_service_tier": requested,
            "session_hash": _hash_identifier(session_id),
            "request_hash": _hash_identifier(request_id),
        }
    )


def record_codex_service_tier_response(
    *,
    model: Any = None,
    issuer_kind: Any = None,
    requested_service_tier: Any = None,
    effective_service_tier: Any = None,
    session_id: Any = None,
    response_id: Any = None,
    status: Any = None,
    source: str = "transport.normalize_response",
) -> None:
    """Record the backend-reported service tier for a Codex response."""
    requested = _clean_str(requested_service_tier)
    effective = _clean_str(effective_service_tier)
    if not requested and not effective:
        return
    _append_record(
        {
            "ts": _now_iso(),
            "event": "response",
            "source": source,
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "model": _clean_str(model),
            "issuer_kind": _clean_str(issuer_kind),
            "requested_service_tier": requested,
            "effective_service_tier": effective,
            "status": _clean_str(status),
            "session_hash": _hash_identifier(session_id),
            "response_hash": _hash_identifier(response_id),
        }
    )
