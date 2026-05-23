from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class QuotaWindowStatus:
    label: str
    remaining_percent: Optional[float] = None
    used_percent: Optional[float] = None
    reset_at: Optional[datetime] = None


@dataclass(frozen=True)
class QuotaResourceStatus:
    provider: str
    ok: bool
    status: str
    checked_at: datetime
    remaining_percent: Optional[float] = None
    windows: dict[str, QuotaWindowStatus] | None = None
    plan: Optional[str] = None
    stale: bool = False
    error: Optional[str] = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _window_key(label: str) -> str:
    normalized = str(label or "").strip().lower().replace(" ", "_")
    if normalized in {"current_session", "primary", "primary_window"}:
        return "session"
    if normalized in {"current_week", "secondary", "secondary_window"}:
        return "weekly"
    return normalized or "unknown"


def _remaining_from_used(used_percent: Optional[float]) -> Optional[float]:
    if used_percent is None:
        return None
    remaining = 100.0 - float(used_percent)
    if remaining < 0:
        return 0.0
    if remaining > 100:
        return 100.0
    return round(remaining, 2)


def quota_status_for_remaining(remaining_percent: Optional[float]) -> str:
    if remaining_percent is None:
        return "degraded"
    remaining = float(remaining_percent)
    if remaining <= 0:
        return "exhausted"
    if remaining < 5:
        return "critical"
    if remaining < 10:
        return "warning"
    if remaining < 20:
        return "economy"
    return "ok"


def build_quota_resource_status(
    snapshot: Optional[Any],
    *,
    checked_at: Optional[datetime] = None,
    stale: bool = False,
    error: Optional[str] = None,
) -> QuotaResourceStatus:
    now = checked_at or _utc_now()
    if snapshot is None:
        return QuotaResourceStatus(
            provider="openai-codex",
            ok=False,
            status="degraded",
            checked_at=now,
            remaining_percent=None,
            windows={},
            stale=stale,
            error=error or "quota unavailable",
        )

    windows: dict[str, QuotaWindowStatus] = {}
    remaining_values: list[float] = []
    for window in snapshot.windows:
        remaining = _remaining_from_used(window.used_percent)
        if remaining is not None:
            remaining_values.append(remaining)
        windows[_window_key(window.label)] = QuotaWindowStatus(
            label=window.label,
            remaining_percent=remaining,
            used_percent=None if window.used_percent is None else float(window.used_percent),
            reset_at=window.reset_at,
        )

    constrained_remaining = min(remaining_values) if remaining_values else None
    status = quota_status_for_remaining(constrained_remaining)
    return QuotaResourceStatus(
        provider=snapshot.provider,
        ok=status != "degraded" and not snapshot.unavailable_reason,
        status=status if not snapshot.unavailable_reason else "degraded",
        checked_at=now,
        remaining_percent=constrained_remaining,
        windows=windows,
        plan=snapshot.plan,
        stale=stale,
        error=snapshot.unavailable_reason or error,
    )


class CodexQuotaResourceCache:
    """Small TTL cache for Codex quota resource status.

    UI callers should use this instead of fetching account usage in render paths.
    Fetch failures return the previous value as stale when possible; otherwise they
    return a degraded status and never masquerade as exhausted quota.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 90,
        fetcher: Callable[[], Optional[Any]] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.ttl = timedelta(seconds=max(1, int(ttl_seconds)))
        if fetcher is None:
            def default_fetcher() -> Optional[Any]:
                from agent.account_usage import fetch_account_usage

                return fetch_account_usage("openai-codex")

            fetcher = default_fetcher

        self.fetcher = fetcher
        self.clock = clock
        self._cached: Optional[QuotaResourceStatus] = None
        self._cached_at: Optional[datetime] = None

    def get(self, *, force_refresh: bool = False) -> QuotaResourceStatus:
        now = self.clock()
        if (
            not force_refresh
            and self._cached is not None
            and self._cached_at is not None
            and now - self._cached_at < self.ttl
        ):
            return self._cached

        try:
            snapshot = self.fetcher()
            status = build_quota_resource_status(snapshot, checked_at=now)
        except Exception as exc:
            if self._cached is not None:
                status = QuotaResourceStatus(
                    provider=self._cached.provider,
                    ok=False,
                    status="degraded",
                    checked_at=now,
                    remaining_percent=self._cached.remaining_percent,
                    windows=self._cached.windows or {},
                    plan=self._cached.plan,
                    stale=True,
                    error=str(exc) or "quota unavailable",
                )
            else:
                status = build_quota_resource_status(None, checked_at=now, error=str(exc) or "quota unavailable")

        self._cached = status
        self._cached_at = now
        return status


_default_codex_quota_cache = CodexQuotaResourceCache()


def get_codex_quota_resource_status(*, force_refresh: bool = False) -> QuotaResourceStatus:
    return _default_codex_quota_cache.get(force_refresh=force_refresh)
