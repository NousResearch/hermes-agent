"""Profile-scoped durable provider/model availability circuit.

Every pre-inference surface uses this store so a provider-declared quota reset is
not rediscovered by starting another model session.  SQLite supplies atomic
merge-max cooldown writes and a single post-reset probe lease across processes.
"""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

_QUOTA_RE = re.compile(
    r"weekly(?:/monthly)? limit exhausted|monthly limit exhausted|quota (?:is )?exhausted|"
    r"usage limit(?: exhausted| reached)|billing (?:limit|quota|exhausted)|subscription (?:limit|exhausted)",
    re.IGNORECASE,
)
_RATE_LIMIT_RE = re.compile(r"\b429\b|rate[ -]?limit", re.IGNORECASE)
_UNKNOWN_BACKOFF = timedelta(minutes=15)
_PROBE_LEASE = timedelta(minutes=5)


@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    model: str
    credential_scope: str = "default"
    reasoning_effort: str | None = None

    def normalized(self) -> "ProviderRoute":
        return ProviderRoute(
            self.provider.strip().lower(),
            self.model.strip().lower(),
            self.credential_scope.strip() or "default",
            self.reasoning_effort,
        )


@dataclass(frozen=True)
class ProviderHealth:
    route: ProviderRoute
    until: datetime
    reset_at: datetime | None
    kind: str
    source: str
    reason: str
    last_failure: datetime
    probe_owner: str | None = None
    probe_until: datetime | None = None


@dataclass(frozen=True)
class RouteDecision:
    route: ProviderRoute | None
    skipped: tuple[ProviderRoute, ...] = ()
    deferred_until: datetime | None = None
    reason: str | None = None
    probe: bool = False


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> float:
    return _utc(value).timestamp()


def _datetime(value: float | None) -> datetime | None:
    return datetime.fromtimestamp(value, tz=timezone.utc) if value is not None else None


class ProviderHealthStore:
    """Durable circuit under one profile home; credentials never cross homes."""

    def __init__(self, hermes_home: str | os.PathLike[str]) -> None:
        home = Path(hermes_home).expanduser()
        home.mkdir(parents=True, exist_ok=True)
        self.path = home / "provider_health.db"
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS provider_health ("
                "provider TEXT NOT NULL, model TEXT NOT NULL, credential_scope TEXT NOT NULL, "
                "until REAL NOT NULL, reset_at REAL, kind TEXT NOT NULL, source TEXT NOT NULL, "
                "reason TEXT NOT NULL, last_failure REAL NOT NULL, probe_owner TEXT, probe_until REAL, "
                "PRIMARY KEY(provider, model, credential_scope))"
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _key(route: ProviderRoute) -> tuple[str, str, str]:
        normalized = route.normalized()
        return normalized.provider, normalized.model, normalized.credential_scope

    @staticmethod
    def _state(row: sqlite3.Row | None) -> ProviderHealth | None:
        if row is None:
            return None
        return ProviderHealth(
            ProviderRoute(row["provider"], row["model"], row["credential_scope"]),
            _datetime(row["until"]),
            _datetime(row["reset_at"]),
            row["kind"],
            row["source"],
            row["reason"],
            _datetime(row["last_failure"]),
            row["probe_owner"],
            _datetime(row["probe_until"]),
        )

    def get(self, route: ProviderRoute) -> ProviderHealth | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM provider_health WHERE provider=? AND model=? AND credential_scope=?",
                self._key(route),
            ).fetchone()
        return self._state(row)

    def record_failure(
        self,
        route: ProviderRoute,
        error: str,
        *,
        source: str,
        now: datetime | None = None,
        owner: str | None = None,
    ) -> ProviderHealth:
        now = _utc(now or datetime.now(timezone.utc))
        text = str(error or "")
        confirmed_quota = bool(_QUOTA_RE.search(text))
        kind = "quota" if confirmed_quota else "rate_limit" if _RATE_LIMIT_RE.search(text) else "transient"
        reset_at = None
        if confirmed_quota:
            from cron.rate_limit_backoff import _declared_reset

            reset_at = _declared_reset(text, now)
            reset_at = _utc(reset_at) if reset_at is not None and reset_at > now else None
        proposed_until = reset_at or (now + _UNKNOWN_BACKOFF)
        key = self._key(route)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT until, reset_at FROM provider_health WHERE provider=? AND model=? AND credential_scope=?",
                key,
            ).fetchone()
            until_ts = max(_timestamp(proposed_until), float(row["until"]) if row else 0.0)
            reset_ts = _timestamp(reset_at) if reset_at else None
            if row and row["reset_at"] is not None:
                reset_ts = max(float(row["reset_at"]), reset_ts or 0.0)
            conn.execute(
                "INSERT INTO provider_health(provider,model,credential_scope,until,reset_at,kind,source,reason,last_failure,probe_owner,probe_until) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(provider,model,credential_scope) DO UPDATE SET "
                "until=excluded.until, reset_at=excluded.reset_at, kind=excluded.kind, source=excluded.source, "
                "reason=excluded.reason, last_failure=excluded.last_failure, probe_owner=NULL, probe_until=NULL",
                (*key, until_ts, reset_ts, kind, source, text[:1000], _timestamp(now), None, None),
            )
            row = conn.execute(
                "SELECT * FROM provider_health WHERE provider=? AND model=? AND credential_scope=?",
                key,
            ).fetchone()
        return self._state(row)

    def record_success(self, route: ProviderRoute, *, owner: str | None = None) -> bool:
        key = self._key(route)
        with self._connect() as conn:
            if owner:
                cur = conn.execute(
                    "DELETE FROM provider_health WHERE provider=? AND model=? AND credential_scope=? "
                    "AND (probe_owner IS NULL OR probe_owner=?)",
                    (*key, owner),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM provider_health WHERE provider=? AND model=? AND credential_scope=?",
                    key,
                )
        return cur.rowcount > 0

    def release_probe(self, route: ProviderRoute, *, owner: str) -> bool:
        """Release an acquired probe without clearing the route's health history."""
        key = self._key(route)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE provider_health SET probe_owner=NULL, probe_until=NULL "
                "WHERE provider=? AND model=? AND credential_scope=? AND probe_owner=?",
                (*key, owner),
            )
        return cur.rowcount > 0

    def decide(
        self,
        routes: Iterable[ProviderRoute],
        *,
        owner: str,
        now: datetime | None = None,
    ) -> RouteDecision:
        now = _utc(now or datetime.now(timezone.utc))
        skipped: list[ProviderRoute] = []
        deadlines: list[datetime] = []
        reasons: list[str] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for route in routes:
                key = self._key(route)
                row = conn.execute(
                    "SELECT * FROM provider_health WHERE provider=? AND model=? AND credential_scope=?",
                    key,
                ).fetchone()
                if row is None:
                    return RouteDecision(route=route, skipped=tuple(skipped))
                state = self._state(row)
                if state.until > now:
                    skipped.append(route)
                    deadlines.append(state.until)
                    reasons.append(state.reason)
                    continue
                if state.probe_owner and state.probe_owner != owner and state.probe_until and state.probe_until > now:
                    skipped.append(route)
                    deadlines.append(state.probe_until)
                    reasons.append("post-reset probe already owned")
                    continue
                probe_until = now + _PROBE_LEASE
                conn.execute(
                    "UPDATE provider_health SET probe_owner=?, probe_until=? "
                    "WHERE provider=? AND model=? AND credential_scope=?",
                    (owner, _timestamp(probe_until), *key),
                )
                return RouteDecision(route=route, skipped=tuple(skipped), probe=True)
        return RouteDecision(
            route=None,
            skipped=tuple(skipped),
            deferred_until=min(deadlines) if deadlines else None,
            reason="; ".join(dict.fromkeys(reasons))[:1000] or "all allowed routes unavailable",
        )


def _agent_owner(agent) -> str:
    task_id = str(os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if task_id:
        return f"kanban:{task_id}"
    return f"agent:{str(getattr(agent, 'session_id', '') or 'unknown')}"


def agent_route_decision(
    agent,
    route: ProviderRoute,
    *,
    now: datetime | None = None,
) -> RouteDecision:
    """Apply the shared circuit to one in-process fallback candidate."""
    from hermes_constants import get_hermes_home

    return ProviderHealthStore(get_hermes_home()).decide(
        [route], owner=_agent_owner(agent), now=now
    )


def record_agent_success(agent) -> bool:
    """Close a route circuit after its owning model request succeeds."""
    provider = str(getattr(agent, "provider", "") or "").strip()
    model = str(getattr(agent, "model", "") or "").strip()
    if not provider or not model:
        return False
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    if not (home / "provider_health.db").exists():
        return False
    scope = str(getattr(agent, "_credential_pool_entry_id", "") or "default")
    store = ProviderHealthStore(home)
    cleared = store.record_success(
        ProviderRoute(provider, model, credential_scope=scope),
        owner=_agent_owner(agent),
    )
    if scope != "default":
        cleared = store.record_success(
            ProviderRoute(provider, model), owner=_agent_owner(agent)
        ) or cleared
    return cleared


def record_agent_failure(
    agent,
    error: str,
    *,
    reason: str,
    now: datetime | None = None,
) -> ProviderHealth | None:
    """Persist a classified availability failure for the exact active route."""
    if reason not in {"billing", "rate_limit", "upstream_rate_limit"}:
        return None
    provider = str(getattr(agent, "provider", "") or "").strip()
    model = str(getattr(agent, "model", "") or "").strip()
    if not provider or not model:
        return None
    from hermes_constants import get_hermes_home

    scope = str(getattr(agent, "_credential_pool_entry_id", "") or "default")
    route = ProviderRoute(provider, model, credential_scope=scope)
    owner = _agent_owner(agent)
    store = ProviderHealthStore(get_hermes_home())
    state = store.record_failure(
        route,
        str(error),
        source=owner,
        owner=owner,
        now=now,
    )
    # Confirmed subscription/account quota text is also indexed at the
    # provider/model scope consulted before a credential is selected. Generic
    # 429s remain isolated to the exact credential entry above.
    if scope != "default" and _QUOTA_RE.search(str(error or "")):
        state = store.record_failure(
            ProviderRoute(provider, model),
            str(error),
            source=owner,
            owner=owner,
            now=now,
        )
    return state
