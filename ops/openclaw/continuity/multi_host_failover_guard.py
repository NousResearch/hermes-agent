#!/usr/bin/env python3
"""A6 multi-host failover guard helpers.

Deterministically classify host health signals so transient network jitter does not
escalate to failover, while sustained multi-host failures still trigger reset paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GuardPolicy:
    jitter_grace_ms: int = 500
    hard_timeout_ms: int = 900
    consecutive_failures_for_host_down: int = 2
    failed_host_quorum_for_reset: int = 2


@dataclass(frozen=True)
class TickAssessment:
    trigger_set: list[str]
    host_status: dict[str, str]
    host_failure_streaks: dict[str, int]
    degraded_host_count: int
    failed_host_count: int


def _to_nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    if parsed < 0:
        return int(default)
    return parsed


def policy_from_env(env: Mapping[str, str] | None = None) -> GuardPolicy:
    env_map = dict(env or {})
    return GuardPolicy(
        jitter_grace_ms=_to_nonnegative_int(env_map.get("OPENCLAW_A6_MULTI_HOST_JITTER_GRACE_MS"), 500),
        hard_timeout_ms=_to_nonnegative_int(env_map.get("OPENCLAW_A6_MULTI_HOST_HARD_TIMEOUT_MS"), 900),
        consecutive_failures_for_host_down=max(
            1,
            _to_nonnegative_int(env_map.get("OPENCLAW_A6_MULTI_HOST_CONSECUTIVE_FAILURES_FOR_DOWN"), 2),
        ),
        failed_host_quorum_for_reset=max(
            1,
            _to_nonnegative_int(env_map.get("OPENCLAW_A6_MULTI_HOST_FAILED_HOST_QUORUM_FOR_RESET"), 2),
        ),
    )


def evaluate_tick(
    *,
    host_samples: list[Mapping[str, Any]],
    prior_failure_streaks: Mapping[str, int] | None,
    prior_warning_active: bool,
    policy: GuardPolicy,
) -> TickAssessment:
    streaks = {str(k): max(0, int(v)) for k, v in dict(prior_failure_streaks or {}).items()}
    host_status: dict[str, str] = {}

    degraded_hosts = 0
    failed_hosts = 0

    for sample in host_samples:
        host_id = str(sample.get("host_id") or "").strip()
        if not host_id:
            continue

        raw_latency = sample.get("latency_ms")
        latency_ms = _to_nonnegative_int(raw_latency, policy.hard_timeout_ms + 1)
        ok = bool(sample.get("ok", True))
        timed_out = bool(sample.get("timed_out", False))

        hard_failed = timed_out or (not ok) or latency_ms >= policy.hard_timeout_ms
        if hard_failed:
            next_streak = streaks.get(host_id, 0) + 1
            streaks[host_id] = next_streak
            if next_streak >= policy.consecutive_failures_for_host_down:
                host_status[host_id] = "failed"
                failed_hosts += 1
            else:
                host_status[host_id] = "transient_hard_fail"
            continue

        streaks[host_id] = 0
        if latency_ms >= policy.jitter_grace_ms:
            host_status[host_id] = "degraded_jitter"
            degraded_hosts += 1
        else:
            host_status[host_id] = "healthy"

    trigger_set: list[str] = []
    if failed_hosts >= policy.failed_host_quorum_for_reset:
        trigger_set.append("TR_RESET_REQUIRED")
    elif degraded_hosts > 0:
        trigger_set.append("TR_WARN_THRESHOLD_REACHED")
    elif prior_warning_active:
        trigger_set.append("TR_WARN_THRESHOLD_CLEARED_STABLE")

    return TickAssessment(
        trigger_set=trigger_set,
        host_status=host_status,
        host_failure_streaks=streaks,
        degraded_host_count=degraded_hosts,
        failed_host_count=failed_hosts,
    )
