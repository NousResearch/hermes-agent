"""Google quota scope and retry hints, shared by native errors and classification."""
from __future__ import annotations

import math
import re
from typing import Any

from agent.retry_utils import parse_retry_after_seconds

_RETRY_IN = re.compile(r"\bretry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?\b", re.IGNORECASE)
_DURATION = re.compile(r"(\d+(?:\.\d+)?)s")
_QUOTA_LINE = re.compile(
    r"Quota exceeded for metric:\s*[^,\n]+,\s*limit:\s*(\d+(?:\.\d+)?),\s*model:\s*([^\s,;]+)",
    re.IGNORECASE,
)


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, str):
        match = _DURATION.fullmatch(value.strip())
        return _nonnegative_number(match.group(1)) if match else None
    if isinstance(value, dict):
        seconds = _nonnegative_number(value.get("seconds", 0))
        nanos = _nonnegative_number(value.get("nanos", 0))
        if seconds is not None and nanos is not None and seconds.is_integer() and nanos.is_integer() and nanos < 1_000_000_000:
            return seconds + nanos / 1_000_000_000
    return None


def gemini_error_payload(body: dict) -> dict:
    nested = body.get("error")
    return nested if isinstance(nested, dict) else body


def gemini_retry_after_seconds(body: dict, headers: Any = None) -> float | None:
    """Keep the largest valid minimum: Google's prose may be more precise than RetryInfo."""
    payload = gemini_error_payload(body)
    delays = [parse_retry_after_seconds(headers)]
    details = payload.get("details")
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
            delays.append(_duration_seconds(detail.get("retryDelay")))
    message = payload.get("message")
    if isinstance(message, str):
        delays.extend(float(match.group(1)) for match in _RETRY_IN.finditer(message))
    valid = [delay for delay in delays if delay is not None and math.isfinite(delay) and delay > 0]
    return max(valid) if valid else None


def gemini_quota_context(body: dict) -> dict:
    """Only exempt the credential when EVERY reported quota violation names a model.

    Missing/malformed/mixed dimensions stay on the existing account-level path.
    A retry hint is not proof that zero allowance will reopen after that delay.
    """
    payload = gemini_error_payload(body)
    message = payload.get("message")
    text_rows = list(_QUOTA_LINE.finditer(message)) if isinstance(message, str) else []
    details = payload.get("details")
    quota_details = [detail for detail in details if isinstance(detail, dict) and detail.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure"] if isinstance(details, list) else []
    models, zero = set(), False
    if quota_details:
        for detail in quota_details:
            violations = detail.get("violations")
            if not isinstance(violations, list) or not violations:
                return {}
            for violation in violations:
                if not isinstance(violation, dict):
                    return {}
                dimensions = violation.get("quotaDimensions")
                model = dimensions.get("model") if isinstance(dimensions, dict) else None
                if not isinstance(model, str) or not model.strip():
                    return {}
                models.add(model.strip())
                zero |= _nonnegative_number(violation.get("quotaValue")) == 0
    elif text_rows and len(text_rows) == len(re.findall(r"Quota exceeded for metric:", message, re.IGNORECASE)):
        models.update(match.group(2) for match in text_rows)
    else:
        return {}
    zero |= any(float(match.group(1)) == 0 for match in text_rows)
    return {"quota_scope": "model", "quota_models": sorted(models), "quota_zero": zero}
