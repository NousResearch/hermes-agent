"""Explicit HTTP delivery for Cron jobs created by stateless API sessions."""

from __future__ import annotations

import json
import os
import re
import socket
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_MAX_SUMMARY_CHARS = 12_000
_MAX_RESPONSE_BYTES = 16_384
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)[^\s]+"),
    re.compile(
        r"(?i)\b((?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[=:]\s*)"
        r"([^\s,;]+)"
    ),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])_[A-Za-z0-9_-]{12,}\b"),
)


def _setting(config: dict, key: str, default: Any = None) -> Any:
    cron = config.get("cron") if isinstance(config, dict) else None
    delivery = cron.get("api_origin_delivery") if isinstance(cron, dict) else None
    return delivery.get(key, default) if isinstance(delivery, dict) else default


def _is_api_origin(job: dict, *, for_failure: bool) -> bool:
    origin = job.get("origin")
    lane = job.get("failure_deliver") if for_failure and job.get("failure_deliver") else job.get(
        "deliver", "local"
    )
    deliver = str(lane or "local").strip().lower()
    return (
        deliver == "origin"
        and isinstance(origin, dict)
        and str(origin.get("platform") or "").strip().lower() == "api_server"
    )


def _redact_and_bound(value: str) -> str:
    cleaned = str(value or "").strip()
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(
            lambda match: f"{match.group(1)}[REDACTED]" if match.lastindex else "[REDACTED]",
            cleaned,
        )
    if len(cleaned) > _MAX_SUMMARY_CHARS:
        cleaned = cleaned[: _MAX_SUMMARY_CHARS - 1].rstrip() + "…"
    return cleaned


def _execution_timestamp(job: dict) -> str:
    execution_id = str(job.get("execution_id") or "")
    if execution_id:
        try:
            from cron.executions import get_execution

            execution = get_execution(execution_id) or {}
            value = execution.get("started_at") or execution.get("claimed_at")
            if isinstance(value, str) and value:
                return value
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


def _response_error(response: object, run_id: str) -> Optional[str]:
    if not isinstance(response, dict):
        return "API-origin Cron delivery endpoint returned a malformed response (expected JSON)"
    if response.get("ok") is not True:
        return "API-origin Cron delivery endpoint returned a malformed response (ok was not true)"
    if response.get("run_id") != run_id:
        return "API-origin Cron delivery endpoint returned a malformed response (run_id mismatch)"
    if not isinstance(response.get("entry_id"), int) or not isinstance(response.get("duplicate"), bool):
        return "API-origin Cron delivery endpoint returned a malformed acknowledgement"
    return None


def deliver_api_origin(
    job: dict, content: str, *, execution_success: bool, for_failure: bool, config: dict
) -> tuple[bool, Optional[str]]:
    """Return ``(handled, error)`` for an API-origin job's explicit HTTP delivery."""
    if not _is_api_origin(job, for_failure=for_failure):
        return False, None

    url = str(_setting(config, "url", "") or "").strip()
    token_env = str(
        _setting(config, "token_env", "CRON_API_ORIGIN_DELIVERY_TOKEN") or ""
    ).strip()
    timeout_raw = _setting(config, "timeout_seconds", 10)
    try:
        timeout = min(60.0, max(0.1, float(timeout_raw)))
    except (TypeError, ValueError):
        return True, "API-origin Cron delivery timeout_seconds is invalid"
    if not url.startswith(("http://", "https://")):
        return True, "API-origin Cron delivery URL is not configured"
    if not token_env:
        return True, "API-origin Cron delivery token_env is not configured"
    token = os.getenv(token_env, "").strip()
    if not token:
        return True, f"API-origin Cron delivery credential {token_env} is unavailable"

    run_id = str(job.get("execution_id") or "").strip()
    if not run_id:
        return True, "API-origin Cron delivery has no execution identifier"
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "job_id": str(job.get("id") or ""),
        "job_name": str(job.get("name") or job.get("id") or ""),
        "run_id": run_id,
        "executed_at": _execution_timestamp(job),
        "delivered_at": now,
        "execution_status": "success" if execution_success else "failure",
        "summary": _redact_and_bound(content),
        "delivery_status": "success",
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 — configured operator URL
            status = int(getattr(response, "status", 0) or 0)
            response_body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        return True, f"API-origin Cron delivery endpoint rejected the request (HTTP {exc.code})"
    except (TimeoutError, socket.timeout):
        return True, f"API-origin Cron delivery timed out after {timeout:g}s"
    except URLError as exc:
        reason = "connection failed"
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            reason = f"timed out after {timeout:g}s"
        return True, f"API-origin Cron delivery {reason}"
    except OSError:
        return True, "API-origin Cron delivery connection failed"

    if not 200 <= status < 300:
        return True, f"API-origin Cron delivery endpoint rejected the request (HTTP {status})"
    if len(response_body) > _MAX_RESPONSE_BYTES:
        return True, "API-origin Cron delivery endpoint returned an oversized response"
    try:
        response_payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True, "API-origin Cron delivery endpoint returned invalid JSON"
    return True, _response_error(response_payload, run_id)
