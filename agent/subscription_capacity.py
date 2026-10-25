"""Fail-closed admission and cross-process leases for subscription workers.

The router is provider-neutral.  The Kanban integration in
``agent.subscription_capacity_runtime`` applies it only to explicitly selected native subscription
providers; paid gateways and aliases remain separate operator-selected routes.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow, fetch_account_usage
from hermes_constants import hermes_home_key

ADMISSION_MAX_AGE = timedelta(seconds=60)
DISPLAY_MAX_AGE = timedelta(minutes=5)
DEFAULT_LEASE_TTL = timedelta(minutes=30)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class ProviderPolicy:
    """An explicitly admitted subscription path.

    ``allowed_sources`` and, when configured, ``expected_identity`` bind quota data to the
    intended account.  Model matching is prefix-based so dated model variants do not require a
    policy edit.  Gateway aliases are intentionally not representable here.
    """

    provider: str
    protected_floor_percent: float
    eligible_tasks: frozenset[str]
    model_prefixes: tuple[str, ...]
    allowed_sources: frozenset[str]
    expected_identity: Optional[str] = None
    max_bounded_leases: int = 1

    def __post_init__(self) -> None:
        if not self.provider or not self.eligible_tasks or not self.model_prefixes or not self.allowed_sources:
            raise ValueError("provider, eligible_tasks, model_prefixes and allowed_sources are required")
        if not 0 < self.protected_floor_percent < 100:
            raise ValueError("protected_floor_percent must be positive and below 100")
        if self.max_bounded_leases < 1:
            raise ValueError("max_bounded_leases must be positive")

    def eligible(self, task: str, model: str) -> bool:
        return task in self.eligible_tasks and any(model.startswith(prefix) for prefix in self.model_prefixes)


def default_subscription_policies(*, claude_floor_percent: float) -> tuple[ProviderPolicy, ...]:
    """Policies for native Codex and Claude subscriptions.

    Codex keeps the operator-mandated 10% voice allowance.  Claude has no inferred 10% rule:
    callers must provide the positive floor that their own operating policy configured.
    """

    tasks = frozenset({"builder", "reviewer", "repair", "analysis"})
    return (
        ProviderPolicy("openai-codex", 10.0, tasks, ("gpt-", "codex-"), frozenset({"usage_api"})),
        ProviderPolicy("anthropic", claude_floor_percent, tasks, ("claude-",), frozenset({"oauth_usage_api"})),
    )


@dataclass(frozen=True)
class CapacityRequest:
    request_id: str
    task: str
    owner: str
    # None is deliberately "unbounded": Hermes does not pretend to forecast a turn's usage.
    reserved_percent: Optional[float] = None

    def __post_init__(self) -> None:
        if not all((self.request_id, self.task, self.owner)):
            raise ValueError("request_id, task and owner are required")
        if self.reserved_percent is not None and not 0 < self.reserved_percent < 100:
            raise ValueError("reserved_percent must be positive and below 100")


@dataclass(frozen=True)
class CapacityAssessment:
    provider: str
    admitted: bool
    reason: str
    effective_remaining_percent: Optional[float] = None
    post_floor_slack_percent: Optional[float] = None
    next_reset_at: Optional[datetime] = None
    active_leases: int = 0


@dataclass(frozen=True)
class CapacityLease:
    lease_id: str
    request_id: str
    provider: str
    task: str
    model: str
    owner: str
    reserved_percent: Optional[float]
    expires_at: datetime


@dataclass(frozen=True)
class AdmissionResult:
    lease: Optional[CapacityLease]
    assessments: tuple[CapacityAssessment, ...]


class SubscriptionCapacityRouter:
    """Choose a subscription provider and atomically reserve its capacity."""

    def __init__(
        self,
        db_path: Path,
        policies: Iterable[ProviderPolicy],
        *,
        coordination_scope: str,
        fetcher: Callable[[str], Optional[AccountUsageSnapshot]] = fetch_account_usage,
        now: Callable[[], datetime] = _utc_now,
        lease_ttl: timedelta = DEFAULT_LEASE_TTL,
        near_tie_percent: float = 2.0,
    ) -> None:
        self.db_path = Path(db_path)
        policy_list = tuple(policies)
        self.policies = {policy.provider: policy for policy in policy_list}
        if len(self.policies) == 0:
            raise ValueError("at least one subscription policy is required")
        if len(self.policies) != len(policy_list):
            raise ValueError("subscription provider policies must be unique")
        if lease_ttl <= timedelta(0) or near_tie_percent < 0:
            raise ValueError("lease_ttl must be positive and near_tie_percent cannot be negative")
        self.coordination_scope = str(coordination_scope or "").strip()
        if not self.coordination_scope:
            raise ValueError("coordination_scope must name the shared subscription-account lease pool")
        self.fetcher = fetcher
        self.now = now
        self.lease_ttl = lease_ttl
        self.near_tie_percent = near_tie_percent
        self._cache: dict[tuple[str, str], AccountUsageSnapshot] = {}
        self._cache_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS subscription_capacity_leases (
                        lease_id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
                        provider TEXT NOT NULL, task TEXT NOT NULL, model TEXT NOT NULL,
                        owner TEXT NOT NULL, reserved_percent REAL, expires_at TEXT NOT NULL
                    )"""
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS subscription_capacity_meta "
                    "(singleton INTEGER PRIMARY KEY, scope TEXT NOT NULL)"
                )
                row = conn.execute("SELECT scope FROM subscription_capacity_meta WHERE singleton = 1").fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO subscription_capacity_meta(singleton, scope) VALUES (1, ?)",
                        (self.coordination_scope,),
                    )
                elif row["scope"] != self.coordination_scope:
                    raise ValueError(
                        f"capacity database belongs to coordination scope {row['scope']!r}, "
                        f"not {self.coordination_scope!r}"
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def snapshot(
        self, provider: str, *, purpose: str = "admission", force_refresh: bool = False,
    ) -> Optional[AccountUsageSnapshot]:
        if purpose not in {"admission", "display"}:
            raise ValueError("purpose must be admission or display")
        max_age = ADMISSION_MAX_AGE if purpose == "admission" else DISPLAY_MAX_AGE
        now = _aware(self.now())
        cache_key = (hermes_home_key(), provider)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if not force_refresh and cached is not None and timedelta(0) <= now - _aware(cached.fetched_at) <= max_age:
                return cached
        try:
            fresh = self.fetcher(provider)
        except Exception:
            fresh = None
        if fresh is not None and not isinstance(fresh, AccountUsageSnapshot):
            fresh = None
        if fresh is not None:
            with self._cache_lock:
                self._cache[cache_key] = fresh
        return fresh

    @staticmethod
    def _window_applies(window: AccountUsageWindow, model: str) -> bool:
        return not window.model_prefixes or any(model.startswith(prefix) for prefix in window.model_prefixes)

    def _assess_snapshot(
        self, policy: ProviderPolicy, request: CapacityRequest, model: str,
        snapshot: Optional[AccountUsageSnapshot],
        *, active_leases: int = 0, reserved_percent: float = 0.0, has_unbounded: bool = False,
    ) -> CapacityAssessment:
        now = _aware(self.now())
        if snapshot is None:
            return CapacityAssessment(policy.provider, False, "usage unavailable", active_leases=active_leases)
        if snapshot.provider != policy.provider:
            return CapacityAssessment(policy.provider, False, "provider identity mismatch", active_leases=active_leases)
        if snapshot.source not in policy.allowed_sources:
            return CapacityAssessment(policy.provider, False, "usage source not allowed", active_leases=active_leases)
        if policy.expected_identity is not None and snapshot.identity != policy.expected_identity:
            return CapacityAssessment(
                policy.provider, False, "account identity unavailable or mismatched", active_leases=active_leases,
            )
        age = now - _aware(snapshot.fetched_at)
        if age < timedelta(0) or age > ADMISSION_MAX_AGE:
            return CapacityAssessment(policy.provider, False, "usage snapshot stale", active_leases=active_leases)
        relevant = [window for window in snapshot.windows if self._window_applies(window, model)]
        if not relevant:
            return CapacityAssessment(policy.provider, False, "no relevant quota windows", active_leases=active_leases)
        if any(window.used_percent is None for window in relevant):
            return CapacityAssessment(policy.provider, False, "quota utilization unknown", active_leases=active_leases)
        utilizations = [float(window.used_percent) for window in relevant if window.used_percent is not None]
        if any(not math.isfinite(used) or used < 0 or used > 100 for used in utilizations):
            return CapacityAssessment(policy.provider, False, "quota utilization invalid", active_leases=active_leases)
        if any(window.reset_at is None or _aware(window.reset_at) <= now for window in relevant):
            return CapacityAssessment(
                policy.provider, False, "quota reset is unknown or expired", active_leases=active_leases,
            )
        if has_unbounded:
            return CapacityAssessment(
                policy.provider, False, "provider has an active unbounded lease", active_leases=active_leases,
            )
        if request.reserved_percent is None and active_leases:
            return CapacityAssessment(
                policy.provider, False, "unbounded lease requires an idle provider", active_leases=active_leases,
            )
        if request.reserved_percent is not None and active_leases >= policy.max_bounded_leases:
            return CapacityAssessment(
                policy.provider, False, "bounded lease limit reached", active_leases=active_leases,
            )
        remaining = min(100.0 - used for used in utilizations)
        requested = request.reserved_percent or 0.0
        slack = remaining - policy.protected_floor_percent - reserved_percent - requested
        reset = min(_aware(window.reset_at) for window in relevant if window.reset_at is not None)
        return CapacityAssessment(
            policy.provider, slack >= 0, "admitted" if slack >= 0 else "protected floor would be crossed",
            remaining, slack, reset, active_leases,
        )

    def _lease_state(self, conn: sqlite3.Connection, provider: str, now: datetime) -> tuple[int, float, bool]:
        conn.execute("DELETE FROM subscription_capacity_leases WHERE expires_at <= ?", (now.isoformat(),))
        rows = conn.execute(
            "SELECT reserved_percent FROM subscription_capacity_leases WHERE provider = ?", (provider,),
        ).fetchall()
        return len(rows), sum(float(row[0] or 0.0) for row in rows), any(row[0] is None for row in rows)

    def acquire(self, request: CapacityRequest, *, candidate_models: Mapping[str, str]) -> AdmissionResult:
        candidates = tuple(candidate_models)
        # A durable request-id hit is a re-admission, not a free idempotency return. Hint the
        # provider fetch to bypass even the <=60s admission cache; the row is re-read under the
        # write transaction below before any result is returned.
        with self._connect() as conn:
            replay_row = conn.execute(
                "SELECT provider FROM subscription_capacity_leases WHERE request_id = ?", (request.request_id,),
            ).fetchone()
        replay_provider = replay_row["provider"] if replay_row is not None else None
        snapshots: dict[str, Optional[AccountUsageSnapshot]] = {}
        ineligible: list[CapacityAssessment] = []
        for provider in candidates:
            policy = self.policies.get(provider)
            if policy is None:
                ineligible.append(CapacityAssessment(provider, False, "not an admitted subscription provider"))
            elif not policy.eligible(request.task, candidate_models[provider]):
                ineligible.append(CapacityAssessment(provider, False, "task or model is not eligible"))
            else:
                snapshots[provider] = self.snapshot(
                    provider, purpose="admission", force_refresh=provider == replay_provider,
                )

        now = _aware(self.now())
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM subscription_capacity_leases WHERE expires_at <= ?", (now.isoformat(),))
            existing = conn.execute(
                "SELECT * FROM subscription_capacity_leases WHERE request_id = ?", (request.request_id,),
            ).fetchone()
            if existing is not None:
                lease = self._row_to_lease(existing)
                expected_model = candidate_models.get(lease.provider)
                if (
                    lease.owner != request.owner
                    or lease.task != request.task
                    or lease.reserved_percent != request.reserved_percent
                    or expected_model != lease.model
                ):
                    conn.commit()
                    return AdmissionResult(None, (*ineligible, CapacityAssessment(
                        lease.provider, False, "request id belongs to a different lease",
                    )))
                policy = self.policies.get(lease.provider)
                snapshot = snapshots.get(lease.provider)
                if policy is None or not policy.eligible(request.task, lease.model):
                    conn.commit()
                    return AdmissionResult(None, (*ineligible, CapacityAssessment(
                        lease.provider, False, "persisted lease is not eligible under current policy",
                    )))
                active, reserved, unbounded = self._lease_state(conn, lease.provider, now)
                replay = self._assess_snapshot(
                    policy, request, lease.model, snapshot,
                    active_leases=max(0, active - 1),
                    reserved_percent=max(0.0, reserved - (lease.reserved_percent or 0.0)),
                    has_unbounded=unbounded and lease.reserved_percent is not None,
                )
                conn.commit()
                if not replay.admitted:
                    return AdmissionResult(None, (*ineligible, replay))
                return AdmissionResult(lease, (*ineligible, replay))
            assessments = list(ineligible)
            for provider, snapshot in snapshots.items():
                policy = self.policies[provider]
                active, reserved, unbounded = self._lease_state(conn, provider, now)
                assessments.append(self._assess_snapshot(
                    policy, request, candidate_models[provider], snapshot, active_leases=active,
                    reserved_percent=reserved, has_unbounded=unbounded,
                ))
            eligible = [assessment for assessment in assessments if assessment.admitted]
            if not eligible:
                conn.commit()
                return AdmissionResult(None, tuple(assessments))
            best_slack = max(float(item.post_floor_slack_percent or 0.0) for item in eligible)
            near = [
                item for item in eligible
                if best_slack - float(item.post_floor_slack_percent or 0.0) <= self.near_tie_percent
            ]
            chosen = min(near, key=lambda item: (
                item.active_leases, item.next_reset_at or datetime.max.replace(tzinfo=timezone.utc), item.provider,
            ))
            expires_at = now + self.lease_ttl
            lease = CapacityLease(
                str(uuid.uuid4()), request.request_id, chosen.provider, request.task, candidate_models[chosen.provider],
                request.owner, request.reserved_percent, expires_at,
            )
            conn.execute(
                "INSERT INTO subscription_capacity_leases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (lease.lease_id, lease.request_id, lease.provider, lease.task, lease.model, lease.owner,
                 lease.reserved_percent, lease.expires_at.isoformat()),
            )
            conn.commit()
        return AdmissionResult(lease, tuple(assessments))

    @staticmethod
    def _row_to_lease(row: sqlite3.Row) -> CapacityLease:
        return CapacityLease(
            row["lease_id"], row["request_id"], row["provider"], row["task"], row["model"], row["owner"],
            row["reserved_percent"], datetime.fromisoformat(row["expires_at"]),
        )

    def check_phase(self, lease: CapacityLease) -> CapacityAssessment:
        """Refresh quota at a phase boundary and prove the same lease remains valid."""
        now = _aware(self.now())
        # Authenticate the durable lease before letting caller-supplied fields select a provider fetch.
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM subscription_capacity_leases WHERE lease_id = ?", (lease.lease_id,),
            ).fetchone()
        if row is None or datetime.fromisoformat(row["expires_at"]) <= now:
            return CapacityAssessment(lease.provider, False, "lease missing or expired")
        if self._row_to_lease(row) != lease:
            return CapacityAssessment(lease.provider, False, "lease record does not match supplied lease")

        snapshot = self.snapshot(lease.provider, purpose="admission", force_refresh=True)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM subscription_capacity_leases WHERE lease_id = ?", (lease.lease_id,),
            ).fetchone()
            if row is None or datetime.fromisoformat(row["expires_at"]) <= now:
                conn.rollback()
                return CapacityAssessment(lease.provider, False, "lease missing or expired")
            if self._row_to_lease(row) != lease:
                conn.rollback()
                return CapacityAssessment(lease.provider, False, "lease record does not match supplied lease")
            active, reserved, unbounded = self._lease_state(conn, lease.provider, now)
            conn.commit()
        policy = self.policies.get(lease.provider)
        if policy is None or not policy.eligible(lease.task, lease.model):
            return CapacityAssessment(lease.provider, False, "lease policy no longer eligible")
        # Exclude this lease from the aggregate before reassessing it.
        own_reserved = lease.reserved_percent or 0.0
        return self._assess_snapshot(
            policy,
            CapacityRequest(lease.request_id, lease.task, lease.owner, lease.reserved_percent),
            lease.model, snapshot,
            active_leases=max(0, active - 1), reserved_percent=max(0.0, reserved - own_reserved),
            has_unbounded=unbounded and lease.reserved_percent is not None,
        )

    def release(self, lease_id: str, *, owner: str) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM subscription_capacity_leases WHERE lease_id = ? AND owner = ?", (lease_id, owner),
            )
            conn.commit()
            return cursor.rowcount == 1
