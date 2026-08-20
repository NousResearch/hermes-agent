"""Search strategies: serial, hedge, hybrid.

Each strategy returns (winner_name, result_dict, latency_ms, attempts).
attempts is a list of Attempt dicts for all tried backends, so the caller
can record per-backend stats with error classification.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

SearchCall = Callable[[str, int], Dict[str, Any]]
BackendEntry = Tuple[str, Any]  # (name, provider)

# Error categories
TRANSIENT = "transient"      # timeout, 503, 502, connection reset
RATE_LIMIT = "rate_limit"    # 429
QUOTA = "quota"              # 402, quota/credit exhausted
AUTH_ERR = "auth"            # 401, 403
UNKNOWN = "unknown"

# Retry config
MAX_RETRIES = 2
RETRY_BACKOFF = 0.5  # seconds between retries


def classify_error(exc: Exception) -> str:
    """Classify an exception into an error category."""
    # TimeoutError has empty str() — must check by type first
    if isinstance(exc, TimeoutError):
        return TRANSIENT
    msg = str(exc).lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)

    # Try to extract HTTP status from message
    if code is None:
        for pattern, val in [("401", 401), ("403", 403), ("402", 402), ("429", 429),
                              ("502", 502), ("503", 503), ("504", 504)]:
            if pattern in msg:
                code = val
                break

    # Auth
    if code in (401, 403):
        if any(kw in msg for kw in ("quota", "credit", "exhaust", "insufficien")):
            return QUOTA
        return AUTH_ERR

    # Payment required = quota
    if code == 402:
        return QUOTA

    # Rate limit — distinguish from quota
    if code == 429:
        if any(kw in msg for kw in ("quota", "credit", "monthly", "exhaust")):
            return QUOTA
        return RATE_LIMIT

    # Server errors
    if code in (502, 503, 504):
        return TRANSIENT

    # Text-based fallback
    if any(kw in msg for kw in ("quota", "credit", "exhaust", "insufficien", "payment")):
        return QUOTA
    if any(kw in msg for kw in ("unauthorized", "invalid api key", "invalid key", "forbidden")):
        return AUTH_ERR
    if any(kw in msg for kw in ("timeout", "timed out", "connection reset", "connection refused",
                                  "connection aborted", "name resolution", "temporary")):
        return TRANSIENT

    return UNKNOWN


def _should_retry(error_type: str) -> bool:
    """Only transient and rate_limit errors are worth retrying."""
    return error_type in (TRANSIENT, RATE_LIMIT)


def _timed_search(
    name: str,
    provider: Any,
    query: str,
    limit: int,
    timeout: float,
) -> Tuple[str, Dict[str, Any], float, str]:
    """Run provider.search() with a wall-clock timeout.

    Uses ThreadPoolExecutor WITHOUT context manager so we can abandon
    the thread on timeout instead of blocking on shutdown(wait=True).

    Returns (name, result, elapsed_ms, error_type).
    """
    start = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(provider.search, query, limit)
        result = future.result(timeout=timeout)
        elapsed = (time.monotonic() - start) * 1000
        return name, result, elapsed, ""
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        error_type = classify_error(exc)
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None) or "?"
        error_msg = str(exc)[:200]
        logger.warning(
            "\u26a0 %s failed [%s] HTTP %s in %.0fms: %s",
            name, error_type, status_code, elapsed, error_msg,
        )
        return (
            name,
            {"success": False, "error": error_msg, "status_code": status_code, "error_type": error_type},
            elapsed,
            error_type,
        )
    finally:
        # Don't wait — abandon thread if still running (it will die on its own
        # when the HTTP request times out internally)
        pool.shutdown(wait=False)


def _search_with_retry(
    name: str,
    provider: Any,
    query: str,
    limit: int,
    per_timeout: float,
    deadline: float = 0.0,
) -> Tuple[str, Dict[str, Any], float, str, int]:
    """Search with up to MAX_RETRIES for transient/rate_limit errors.

    Respects deadline: won't start a new attempt if deadline is exceeded.

    Returns (name, result, elapsed_ms, error_type, attempt_count).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        # Check deadline before each attempt
        if deadline and time.monotonic() >= deadline:
            logger.info("⏰ %s: deadline reached before attempt %d", name, attempt)
            return name, {"success": False, "error": "deadline exceeded", "error_type": TRANSIENT}, 0.0, TRANSIENT, attempt - 1

        # Cap timeout to remaining deadline
        timeout = per_timeout
        if deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return name, {"success": False, "error": "deadline exceeded", "error_type": TRANSIENT}, 0.0, TRANSIENT, attempt - 1
            timeout = min(per_timeout, remaining)

        name, result, elapsed, error_type = _timed_search(name, provider, query, limit, timeout)

        if result.get("success"):
            return name, result, elapsed, "", attempt

        if not _should_retry(error_type) or attempt >= MAX_RETRIES:
            return name, result, elapsed, error_type, attempt

        # Brief backoff before retry (capped to deadline)
        backoff = RETRY_BACKOFF
        if deadline:
            remaining = deadline - time.monotonic() - backoff
            if remaining <= 0:
                return name, result, elapsed, error_type, attempt
        logger.info("\u21bb %s retry %d/%d after %.1fs (error: %s)",
                    name, attempt + 1, MAX_RETRIES, backoff, error_type)
        time.sleep(backoff)

    return name, result, elapsed, error_type, MAX_RETRIES


# Attempt record: (name, success, latency_ms, error_type, attempts_count)
Attempt = Tuple[str, bool, float, str, int]


def serial(
    backends: List[BackendEntry],
    query: str,
    limit: int,
    per_timeout: float,
    total_timeout: float,
) -> Tuple[str, Dict[str, Any], float, List[Attempt]]:
    """Try backends sequentially. First success wins."""
    deadline = time.monotonic() + total_timeout
    attempts: List[Attempt] = []
    last_result = None
    last_name = ""
    last_latency = 0.0

    for name, provider in backends:
        if time.monotonic() >= deadline:
            logger.info("Cascade serial: total_timeout reached")
            break

        name, result, latency, error_type, count = _search_with_retry(
            name, provider, query, limit, per_timeout, deadline=deadline,
        )
        success = result.get("success", False)
        attempts.append((name, success, latency, error_type, count))

        if success:
            return name, result, latency, attempts

        last_name, last_result, last_latency = name, result, latency

    return (
        last_name or "none",
        last_result or {"success": False, "error": "No backends available"},
        last_latency,
        attempts,
    )


def hedge(
    backends: List[BackendEntry],
    query: str,
    limit: int,
    per_timeout: float,
    total_timeout: float,
) -> Tuple[str, Dict[str, Any], float, List[Attempt]]:
    """Launch all backends in parallel. First success wins."""
    attempts: List[Attempt] = []
    if not backends:
        return "none", {"success": False, "error": "No backends available"}, 0.0, attempts

    best = None
    deadline = time.monotonic() + total_timeout

    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        futures = {
            pool.submit(_search_with_retry, name, prov, query, limit, per_timeout, deadline): name
            for name, prov in backends
        }
        try:
            for future in as_completed(futures, timeout=total_timeout):
                name, result, latency, error_type, count = future.result()
                success = result.get("success", False)
                attempts.append((name, success, latency, error_type, count))
                if success and best is None:
                    best = (name, result, latency)
                    break
        except TimeoutError:
            logger.warning("Cascade hedge: total_timeout reached")

    if best:
        return best[0], best[1], best[2], attempts

    last = list(futures.values())[-1]
    return last, {"success": False, "error": "All backends failed"}, 0.0, attempts


def hybrid(
    backends: List[BackendEntry],
    query: str,
    limit: int,
    per_timeout: float,
    total_timeout: float,
    trigger_after: float = 3.0,
) -> Tuple[str, Dict[str, Any], float, List[Attempt]]:
    """Start primary; if no response in trigger_after seconds, hedge remaining."""
    attempts: List[Attempt] = []
    if not backends:
        return "none", {"success": False, "error": "No backends available"}, 0.0, attempts

    if len(backends) == 1:
        return serial(backends, query, limit, per_timeout, total_timeout)

    primary_name, primary_prov = backends[0]
    rest = backends[1:]

    deadline = time.monotonic() + total_timeout

    # Try primary (with retries, capped to trigger_after)
    primary_deadline = min(time.monotonic() + trigger_after, deadline)
    name, result, latency, error_type, count = _search_with_retry(
        primary_name, primary_prov, query, limit, trigger_after, deadline=primary_deadline,
    )
    success = result.get("success", False)
    attempts.append((name, success, latency, error_type, count))

    if success:
        return name, result, latency, attempts

    # Hedge with remaining
    remaining_names = [n for n, _ in rest]
    logger.info("Cascade hybrid: primary %s slow/failed, hedging with %s", name, remaining_names)

    remaining_time = deadline - time.monotonic()
    if remaining_time <= 0:
        return name, result, latency, attempts

    # Launch remaining in parallel
    best = None
    with ThreadPoolExecutor(max_workers=len(rest)) as pool:
        futures = {
            pool.submit(_search_with_retry, n, p, query, limit, min(per_timeout, remaining_time), deadline): n
            for n, p in rest
        }
        try:
            for future in as_completed(futures, timeout=remaining_time):
                n, r, lat, et, c = future.result()
                ok = r.get("success", False)
                attempts.append((n, ok, lat, et, c))
                if ok and best is None:
                    best = (n, r, lat)
                    break
        except TimeoutError:
            logger.warning("Cascade hybrid: total_timeout reached during hedge")

    if best:
        return best[0], best[1], best[2], attempts

    return name, result, latency, attempts
