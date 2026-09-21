"""Bounded capacity retries for idempotent Hindsight reads."""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable


def supports_keyword(callable_obj: Callable[..., Any], name: str) -> bool:
    """Return whether a public callable signature declares *name*."""
    try:
        parameter = inspect.signature(callable_obj).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


DEFAULT_TOTAL_BUDGET_SECONDS = 10.0
_FALLBACK_BASE_DELAY_SECONDS = 0.5
_FALLBACK_MAX_DELAY_SECONDS = 2.0


async def retry_capacity(
    call: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = 3,
    total_budget: float = DEFAULT_TOTAL_BUDGET_SECONDS,
) -> Any:
    """Retry an idempotent call on 429/503 within one bounded wait budget."""
    attempts = max(1, int(max_attempts))
    deadline = time.monotonic() + max(0.0, float(total_budget))
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not _is_capacity_error(exc) or attempt == attempts:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            retry_after = _retry_after_seconds(exc)
            if retry_after is None:
                ceiling = min(
                    _FALLBACK_MAX_DELAY_SECONDS,
                    _FALLBACK_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                )
                delay = max(0.0, min(float(random.uniform(0.0, ceiling)), ceiling))
            else:
                delay = retry_after
            if delay > remaining:
                raise
            await asyncio.sleep(delay)
            if time.monotonic() >= deadline:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _is_capacity_error(exc: Exception) -> bool:
    try:
        from hindsight_client_api.exceptions import ApiException
    except ImportError:
        return False
    return isinstance(exc, ApiException) and getattr(exc, "status", None) in (429, 503)


def _header_value(headers: Any, name: str) -> Any:
    target = name.casefold()
    if hasattr(headers, "items"):
        try:
            for key, value in headers.items():
                if str(key).casefold() == target:
                    return value
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        for key, value in headers:
            if str(key).casefold() == target:
                return value
    except (TypeError, ValueError):
        pass
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    raw = _header_value(headers, "Retry-After")
    if raw is None:
        return None
    text = str(raw).strip()
    if text.isascii() and text.isdigit():
        return float(text)
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
