"""Sanitized Codex quota data, independent of credentials and presentation.

Only allowlisted fields/labels cross this boundary. Missing numeric values are
None, never zero. Unknown provider windows are not supported or copied through.
No cache here: the current /usage resolver cannot bind explicit/pool credentials
to a reliable account identity. Account-bound single-flight, cache and backoff
belong with the integration that owns that identity and its invalidation.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

WindowId = Literal["primary_window", "secondary_window"]
WindowLabel = Literal["Session", "Weekly"]
_WINDOWS: tuple[tuple[WindowId, WindowLabel], ...] = (
    ("primary_window", "Session"), ("secondary_window", "Weekly"),
)
# Never title-case arbitrary provider text: it could contain identity/credentials.
_PLANS = {name: name.title() for name in (
    "free", "go", "plus", "pro", "team", "business", "enterprise", "edu",
)}


@dataclass(frozen=True)
class CodexQuotaWindow:
    id: WindowId
    label: WindowLabel
    used_percent: float | None
    reset_at: int | None


@dataclass(frozen=True)
class CodexQuotaSuccess:
    fetched_at: int
    plan: str | None
    banked_resets: int | None
    windows: tuple[CodexQuotaWindow, ...]
    credits_balance: float | None = None
    credits_unlimited: bool | None = None
    status: Literal["ok"] = field(default="ok", init=False)
    supported: Literal[True] = field(default=True, init=False)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["windows"] = [asdict(window) for window in self.windows]
        return result


QuotaError = Literal["auth", "rate_limit", "network", "unsupported"]


@dataclass(frozen=True)
class CodexQuotaFailure:
    fetched_at: int
    error: QuotaError
    status: Literal["error"] = field(default="error", init=False)

    @property
    def supported(self) -> bool:
        return self.error != "unsupported"

    def to_dict(self) -> dict:
        return {**asdict(self), "supported": self.supported}


CodexQuotaResult = CodexQuotaSuccess | CodexQuotaFailure


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _epoch(value: object) -> int | None:
    number = _integer(value)
    # Seconds, not milliseconds; also safe for the presentation's datetime.
    return number if number is not None and number <= 253402300799 else None


def _reset_epoch(value: object) -> int | None:
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            value = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
        except (ValueError, OverflowError, OSError):
            return None
    return _epoch(value)


def parse_codex_quota(
    payload: object, *, fetched_at: int, http_status: int | None = 200,
) -> CodexQuotaResult:
    """Parse a decoded response; never accept exception text, headers or URLs.

    Integration passes None for a transport failure (timeout/connection error),
    or the HTTP status. Auth failures before HTTP can use CodexQuotaFailure.
    ``supported`` distinguishes unsupported endpoints/schema from temporarily
    unavailable quota. ``fetched_at`` is the observation/attempt epoch in seconds.
    """
    stamp = _epoch(fetched_at)
    if stamp is None:
        raise ValueError("fetched_at must be epoch seconds")
    fetched_at = stamp
    if http_status is None or 500 <= http_status <= 599:
        return CodexQuotaFailure(fetched_at, "network")
    if not 200 <= http_status <= 299:
        errors: dict[int, QuotaError] = {401: "auth", 403: "auth", 429: "rate_limit"}
        return CodexQuotaFailure(fetched_at, errors.get(http_status, "unsupported"))
    if not isinstance(payload, dict):
        return CodexQuotaFailure(fetched_at, "unsupported")
    body = payload
    if body.get("rate_limit") is not None and not isinstance(body["rate_limit"], dict):
        return CodexQuotaFailure(fetched_at, "unsupported")
    rate_limit = _object(body.get("rate_limit"))
    windows = tuple(
        CodexQuotaWindow(
            id=key, label=label,
            used_percent=_number(_object(rate_limit.get(key)).get("used_percent")),
            reset_at=_reset_epoch(_object(rate_limit.get(key)).get("reset_at")),
        ) for key, label in _WINDOWS
    )
    plan = body.get("plan_type")
    credits = _object(body.get("credits"))
    has_credits = credits.get("has_credits") is True
    return CodexQuotaSuccess(
        fetched_at=fetched_at,
        plan=_PLANS.get(plan) if isinstance(plan, str) else None,
        banked_resets=_integer(_object(body.get("rate_limit_reset_credits")).get("available_count")),
        windows=windows,
        credits_balance=_number(credits.get("balance")) if has_credits else None,
        credits_unlimited=(credits["unlimited"] if has_credits and isinstance(credits.get("unlimited"), bool)
                           else False if credits.get("has_credits") is False else None),
    )
