"""Execution-time Kanban provider/model remediation backed by account usage meters.

The dispatcher calls this immediately before spawning a worker.  The policy is
fail-closed: an unavailable or malformed meter never becomes permission to
spend a provider quota.  Nous is intentionally excluded from execution
fallbacks for this policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agent.account_usage import AccountUsageSnapshot
from agent import account_usage


@dataclass(frozen=True)
class ProviderMeterCandidate:
    provider: str
    model: str


_PROVIDER_PRIORITY = {"anthropic": 0, "openai-codex": 1}


def _has_headroom(snapshot: AccountUsageSnapshot | None, minimum_remaining_percent: int) -> bool:
    if snapshot is None or not snapshot.available or not snapshot.windows:
        return False
    numeric = [window.used_percent for window in snapshot.windows]
    if any(value is None for value in numeric):
        return False
    # A provider is usable when at least one concrete usage window has
    # headroom.  Different providers expose different window sets; requiring
    # every window would incorrectly reject Claude when its short window is
    # healthy but a longer informational window is exhausted.
    return any(100.0 - float(value) >= minimum_remaining_percent for value in numeric)


def _has_headroom_all(snapshot: AccountUsageSnapshot | None, minimum_remaining_percent: int) -> bool:
    if snapshot is None or not snapshot.available or not snapshot.windows:
        return False
    numeric = [window.used_percent for window in snapshot.windows]
    if any(value is None for value in numeric):
        return False
    return all(100.0 - float(value) >= minimum_remaining_percent for value in numeric)


def choose_available_candidate(
    candidates: tuple[ProviderMeterCandidate, ...] | list[ProviderMeterCandidate],
    readings: Mapping[str, AccountUsageSnapshot | None],
    *,
    minimum_remaining_percent: int = 10,
) -> ProviderMeterCandidate | None:
    """Choose a metered non-Nous candidate, preferring Claude over Codex."""
    try:
        threshold = int(minimum_remaining_percent)
    except (TypeError, ValueError):
        threshold = 10
    threshold = max(0, min(100, threshold))
    eligible = [
        candidate for candidate in candidates
        if candidate.provider.strip().lower() != "nous"
        and candidate.model.strip()
        and _has_headroom(readings.get(candidate.provider.strip().lower()), threshold)
        and (
            candidate.provider.strip().lower() != "openai-codex"
            or _has_headroom_all(readings.get(candidate.provider.strip().lower()), threshold)
        )
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (_PROVIDER_PRIORITY.get(candidate.provider.strip().lower(), 99),),
    )


def remediate_candidate(
    candidates: tuple[ProviderMeterCandidate, ...] | list[ProviderMeterCandidate],
    *,
    minimum_remaining_percent: int = 10,
) -> ProviderMeterCandidate | None:
    """Fetch provider meters and return the execution override, if safe."""
    readings: dict[str, AccountUsageSnapshot | None] = {}
    for provider in {candidate.provider.strip().lower() for candidate in candidates}:
        if provider == "nous":
            continue
        # Resolve through the module at call time so the dispatcher observes
        # refreshed meters and tests/operators can replace the live lookup
        # without re-importing this policy module.
        readings[provider] = account_usage.fetch_account_usage(provider)
    return choose_available_candidate(
        candidates, readings, minimum_remaining_percent=minimum_remaining_percent
    )
