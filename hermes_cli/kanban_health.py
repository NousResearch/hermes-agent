"""Health gates for optional external Kanban worker profiles.

The decomposer normally exposes every installed profile in its routing roster.
An explicitly configured health policy can make one or more external profiles
optional: the profile is advertised only when its OpenAI-compatible ``/models``
endpoint responds successfully and contains the required model.

This module deliberately owns no task or dispatcher state. It performs a
bounded, cached read-only probe and returns a filtered roster to the
decomposer. No existing task is reassigned when a probe fails.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 2.0
_DEFAULT_CACHE_TTL_SECONDS = 30.0
_MAX_TIMEOUT_SECONDS = 10.0
_MAX_CACHE_TTL_SECONDS = 3600.0
_MAX_RESPONSE_BYTES = 1_048_576

# ``external_worker_health`` is the documented key. The aliases keep the
# resolver tolerant of early/private config names without changing the default
# (disabled) behavior for existing installations.
_POLICY_KEYS = (
    "external_worker_health",
    "decomposition_health",
    "worker_health",
    "roster_health",
)


@dataclass(frozen=True)
class ExternalWorkerHealthPolicy:
    """Configuration for one optional profile in the decomposer roster."""

    profile: str
    enabled: bool
    base_url: str
    required_model: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS
    fail_closed: bool = True
    capacity_multiplier: int = 1


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_bounded_float(value: Any, *, default: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    if parsed <= 0:
        return 0.0
    return min(parsed, maximum)


def _policy_entries(raw: Any) -> list[Mapping[str, Any]]:
    """Normalize the singular and list forms accepted by the config loader."""
    if isinstance(raw, Mapping):
        # The documented form is one dict. ``profiles``/``workers`` are useful
        # for installations with more than one optional external worker.
        for key in ("profiles", "workers", "policies"):
            nested = raw.get(key)
            if isinstance(nested, list):
                return [entry for entry in nested if isinstance(entry, Mapping)]
        return [raw]
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, Mapping)]
    return []


def load_health_policies(config: Mapping[str, Any] | None) -> tuple[ExternalWorkerHealthPolicy, ...]:
    """Return enabled external-worker policies from a resolved config.

    An omitted or disabled policy returns an empty tuple. This is important:
    callers can return their historical roster object unchanged in that case.
    Malformed enabled policies are retained and fail closed by default, so a
    typo cannot silently advertise an optional remote worker.
    """
    if not isinstance(config, Mapping):
        return ()
    kanban_cfg = config.get("kanban")
    if not isinstance(kanban_cfg, Mapping):
        return ()
    raw = None
    for key in _POLICY_KEYS:
        if key in kanban_cfg:
            raw = kanban_cfg.get(key)
            break
    if raw is None:
        return ()

    policies: list[ExternalWorkerHealthPolicy] = []
    for entry in _policy_entries(raw):
        enabled = _coerce_bool(entry.get("enabled"), default=False)
        if not enabled:
            continue
        profile = str(
            entry.get("profile")
            or entry.get("profile_name")
            or entry.get("name")
            or ""
        ).strip()
        if not profile:
            # There is no profile to remove; skip without exposing config
            # values in logs.
            continue
        base_url = str(entry.get("base_url") or "").strip()
        required_model = str(
            entry.get("required_model") or entry.get("model") or ""
        ).strip()
        timeout_seconds = _coerce_bounded_float(
            entry.get("timeout_seconds", entry.get("timeout", _DEFAULT_TIMEOUT_SECONDS)),
            default=_DEFAULT_TIMEOUT_SECONDS,
            maximum=_MAX_TIMEOUT_SECONDS,
        )
        cache_ttl_seconds = _coerce_bounded_float(
            entry.get(
                "cache_ttl_seconds",
                entry.get("cache_ttl", entry.get("ttl_seconds", _DEFAULT_CACHE_TTL_SECONDS)),
            ),
            default=_DEFAULT_CACHE_TTL_SECONDS,
            maximum=_MAX_CACHE_TTL_SECONDS,
        )
        fail_closed = _coerce_bool(
            entry.get("fail_closed", entry.get("fail_closed_optional", True)),
            default=True,
        )
        raw_capacity_multiplier = entry.get("capacity_multiplier", 1)
        if (
            type(raw_capacity_multiplier) is not int
            or raw_capacity_multiplier < 1
            or raw_capacity_multiplier > 16
        ):
            capacity_multiplier = 1
        else:
            capacity_multiplier = raw_capacity_multiplier
        policies.append(
            ExternalWorkerHealthPolicy(
                profile=profile,
                enabled=True,
                base_url=base_url,
                required_model=required_model,
                timeout_seconds=timeout_seconds,
                cache_ttl_seconds=cache_ttl_seconds,
                fail_closed=fail_closed,
                capacity_multiplier=capacity_multiplier,
            )
        )
    return tuple(policies)


def _models_url(base_url: str) -> str | None:
    """Resolve an OpenAI-compatible base URL to its model-list endpoint."""
    candidate = base_url.strip().rstrip("/")
    if not candidate:
        return None
    try:
        parsed = urllib.parse.urlparse(candidate)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.path.rstrip("/").endswith("/models"):
        return candidate
    # OpenAI-compatible base URLs conventionally end in /v1. If the caller
    # gives only an origin, add /v1 so both forms remain useful.
    if not parsed.path or parsed.path == "/":
        return candidate + "/v1/models"
    return candidate + "/models"


def _model_ids(payload: Any) -> set[str]:
    if isinstance(payload, Mapping):
        items = payload.get("data")
        if items is None:
            items = payload.get("models")
    else:
        items = payload
    if not isinstance(items, list):
        return set()

    ids: set[str] = set()
    for item in items:
        if isinstance(item, str) and item.strip():
            ids.add(item.strip())
        elif isinstance(item, Mapping):
            model_id = item.get("id") or item.get("name")
            if isinstance(model_id, str) and model_id.strip():
                ids.add(model_id.strip())
    return ids


def _probe_models(
    policy: ExternalWorkerHealthPolicy,
    *,
    opener: Callable[..., Any] | None = None,
) -> bool:
    """Perform one bounded model-list probe without logging secrets or URLs."""
    url = _models_url(policy.base_url)
    if not url or not policy.required_model:
        return False
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "hermes-agent-kanban-health/1",
        },
    )
    open_fn = opener or urllib.request.urlopen
    response = None
    try:
        response = open_fn(request, timeout=policy.timeout_seconds)
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "status_code", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else 200
        if int(status) != 200:
            return False
        try:
            raw = response.read(_MAX_RESPONSE_BYTES)
        except TypeError:
            # Keep lightweight test/double response objects compatible with
            # urllib responses while real responses remain size-bounded.
            raw = response.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        payload = json.loads(raw)
        return policy.required_model in _model_ids(payload)
    except Exception:
        # Do not include the exception or URL: either may contain credentials
        # supplied by a custom urllib opener or a query-bearing endpoint.
        logger.debug("kanban health probe failed for optional profile %r", policy.profile)
        return False
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


_HEALTH_CACHE: dict[tuple[Any, ...], tuple[float, bool]] = {}
_HEALTH_CACHE_LOCK = threading.RLock()


def clear_health_cache() -> None:
    """Clear in-process probe results (primarily useful for tests and reloads)."""
    with _HEALTH_CACHE_LOCK:
        _HEALTH_CACHE.clear()


def check_health(
    policy: ExternalWorkerHealthPolicy,
    *,
    now: Callable[[], float] = time.monotonic,
    opener: Callable[..., Any] | None = None,
) -> bool:
    """Return the cached or freshly probed health result for *policy*."""
    cache_key = (
        policy.profile,
        policy.base_url,
        policy.required_model,
        policy.timeout_seconds,
        policy.cache_ttl_seconds,
    )
    with _HEALTH_CACHE_LOCK:
        # Sample after acquiring the lock so a waiter evaluates expiry against
        # the current time rather than a timestamp captured before another
        # probe completed.
        current = now()
        cached = _HEALTH_CACHE.get(cache_key)
        if cached is not None:
            cached_at, result = cached
            if policy.cache_ttl_seconds > 0 and current - cached_at < policy.cache_ttl_seconds:
                return result
        result = _probe_models(policy, opener=opener)
        # Probe duration must not consume the newly produced result's TTL.
        _HEALTH_CACHE[cache_key] = (now(), result)
        return result


def filter_roster(
    roster: list[dict],
    valid_names: set[str],
    policies: Sequence[ExternalWorkerHealthPolicy],
    *,
    protected_names: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[dict], set[str]]:
    """Remove unhealthy optional profiles while preserving fallback safety."""
    if not policies:
        # Preserve the historical result exactly when the opt-in gate is not
        # enabled; callers rely on this for all existing configurations.
        return roster, valid_names

    filtered_roster = roster
    filtered_names = valid_names
    for policy in policies:
        if policy.profile not in filtered_names:
            continue
        if policy.profile in protected_names:
            # A configured default/orchestrator is a safety anchor, not an
            # optional worker. Never strand root/fallback work by filtering it.
            logger.warning(
                "kanban health gate will not filter protected profile %r",
                policy.profile,
            )
            continue
        if check_health(policy):
            continue
        if not policy.fail_closed:
            continue
        if filtered_roster is roster:
            filtered_roster = list(roster)
        filtered_roster = [
            entry for entry in filtered_roster if entry.get("name") != policy.profile
        ]
        if filtered_names is valid_names:
            filtered_names = set(valid_names)
        filtered_names.discard(policy.profile)
    return filtered_roster, filtered_names


def resolve_capacity_limits(
    config: Mapping[str, Any] | None,
    *,
    max_in_progress: int | None,
    max_in_progress_per_profile: int | None,
) -> tuple[int | None, int | None]:
    """Apply the largest healthy optional-worker capacity multiplier."""
    multiplier = 1
    for policy in load_health_policies(config):
        if policy.capacity_multiplier > 1 and check_health(policy):
            multiplier = max(multiplier, policy.capacity_multiplier)

    def _scaled(value: int | None) -> int | None:
        return None if value is None else value * multiplier

    return _scaled(max_in_progress), _scaled(max_in_progress_per_profile)


__all__ = [
    "ExternalWorkerHealthPolicy",
    "check_health",
    "clear_health_cache",
    "filter_roster",
    "load_health_policies",
    "resolve_capacity_limits",
]
