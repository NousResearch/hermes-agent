"""Daily engagement rollup, per-conversation session tally and turns-before-switch.

Engagement is accumulated locally per UTC day in the profile's metrics store and emitted once the
day closes (the first interaction of a later day): one ``hermes.engagement.surface_day.count`` row
per surface used, and one ``hermes.engagement.day.count`` row with the day's total active time, how
many surfaces it touched and the model that served most attended turns. Active time is the sum of
the gaps between consecutive interactions (a turn starting or ending), each capped at
``IDLE_CAP_MS``. Days-active-per-week and next-day/next-week return by model are derived
server-side from these daily rows and the package ``install_id``: nothing here keeps a weekly
window or any identifier of its own.

The day row also carries ``active_profile_count_bucket``: how many distinct profiles of this host
had a user turn that UTC day. Every profile's interactions are folded into ONE host accumulator,
the root (default) profile's day state, as opaque local hashes of the profile home (a set, so a
profile is counted once whichever process or multiplexed runtime served it). Only the root
profile's day row reports the count; every other profile's row carries ``0`` so no row claims the
host total twice. When the root profile has collection off, nothing is written to its store and
the count is not reported.

Only closed enums, buckets, catalog-sanitized provider/model names and profile hashes are stored
in the state row; the hashes never leave it.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import shared_metrics_contract as contract

# A gap longer than this is idle time: it counts only up to the cap.
IDLE_CAP_MS = 5 * 60_000
STATE_KEY = "engagement_day_state"
_MINUTE_MS = 60_000
_ACTIVE_THRESHOLDS = (
    (5 * _MINUTE_MS, "lt_5m"), (30 * _MINUTE_MS, "5m_to_30m"), (120 * _MINUTE_MS, "30m_to_2h"),
    (360 * _MINUTE_MS, "2h_to_6h"),
)
_SWITCH_THRESHOLDS = ((2, "1"), (4, "2_to_3"), (11, "4_to_10"), (31, "11_to_30"))
# Task entrypoints that are a person using Hermes. Unattended cron runs (counted by hermes.cron.run),
# delegated children, background review forks, batch and API/python embedding are not engagement.
_ENGAGED_ENTRYPOINTS = frozenset({"gateway_message", "interactive"})
_INTERACTION_METRICS = frozenset({contract.TASK_STARTED_METRIC, contract.TASK_FINISHED_METRIC})
_ALL = "*"

# Test / live-probe seam: wall clock in epoch seconds.
_now: Callable[[], float] = time.time


def active_minutes_bucket(active_ms: int) -> str:
    return "0" if active_ms <= 0 else contract._bucket(active_ms, _ACTIVE_THRESHOLDS, "gte_6h")


def surfaces_used_bucket(count: int) -> str:
    return str(count) if count < 4 else "gte_4"


def turns_before_switch_bucket(turns: int) -> str:
    return contract._bucket(turns, _SWITCH_THRESHOLDS, "gte_31")


# ---- producer side (relay subscriber) ------------------------------------------------------------

def interaction_surface(classified: list[tuple[str, dict, int]]) -> str | None:
    """The engagement surface of a task start/end counter, else None."""
    for metric_name, dimensions, _ in classified:
        if metric_name in _INTERACTION_METRICS and dimensions.get("entrypoint") in _ENGAGED_ENTRYPOINTS:
            surface = dimensions.get("execution_surface", "")
            return surface if surface in contract.ENGAGEMENT_SURFACES else None
    return None


def engaged_turn(start_fields: dict[str, str] | None, user_turn: bool) -> bool:
    """A turn a person sent on an engagement surface (its model counts toward the day's primary model)."""
    return user_turn and (start_fields or {}).get("entrypoint") in _ENGAGED_ENTRYPOINTS


def turn_route(event: Any) -> tuple[str, str] | None:
    """(provider, model) of one validated engagement turn mark."""
    if not contract._valid_shape(event, **contract._MARK_SHAPE):
        return None
    if contract._event_text(event, "name") != contract.ENGAGEMENT_TURN_MARK:
        return None
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != {"model", "provider"}:
        return None
    limits = contract._MODEL_ROUTE_MAX_LENGTHS
    route = tuple(data[k] for k in ("provider", "model"))
    if not all(isinstance(v, str) and v == contract._metric_identifier(v, max_length=limits[k])
               for k, v in zip(("provider", "model"), route)):
        return None
    return route  # type: ignore[return-value]


def record(store: Any, resource: dict[str, str], *, surface: str | None, route: tuple[str, str] | None) -> None:
    """The clock is read inside each write transaction: writers then apply in clock order, so no
    process closes a day another already closed. A busy store defers the interaction, never drops it."""
    profile_home = Path(store.database_path).parent.parent.parent
    root_home = _root_home(profile_home)
    owner = root_home == profile_home
    profile = _profile_hash(profile_home) if surface is not None else None
    store.update_rollup_state(STATE_KEY, lambda state: apply(
        state, now_ms=_now_ms(), resource=resource, surface=surface, route=route,
        profile=profile if owner else None, owner=owner,
    ))
    if profile is not None and not owner and _collects(root_home):
        _root_store(root_home).update_rollup_state(STATE_KEY, lambda state: apply(
            state, now_ms=_now_ms(), resource=resource, profile=profile, owner=True,
        ))


def _now_ms() -> int:
    return int(_now() * 1000)


def _root_home(profile_home: Path) -> Path:
    from hermes_constants import get_default_hermes_root

    try:
        return get_default_hermes_root(home=profile_home).resolve()
    except Exception:
        return profile_home.resolve()


def _profile_hash(profile_home: Path) -> str:
    return hashlib.sha256(str(profile_home.resolve()).encode()).hexdigest()[:16]


def _collects(home: Path) -> bool:
    """The root profile's own collection consent (read-only; never touches its runtime)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        from hermes_cli.config import read_raw_config_readonly

        config: Any = read_raw_config_readonly() or {}
    except Exception:
        return False
    finally:
        reset_hermes_home_override(token)
    for key in ("telemetry", "shared_metrics"):
        config = config.get(key) if isinstance(config, dict) else None
    return isinstance(config, dict) and config.get("enabled") is True


_root_stores: dict[Path, Any] = {}
_root_stores_lock = threading.Lock()


def _root_store(home: Path) -> Any:
    from .shared_metrics import SharedMetricsStore

    with _root_stores_lock:
        if home not in _root_stores:
            root = home / "telemetry" / "shared_metrics"
            _root_stores[home] = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
        return _root_stores[home]


# ---- the rollup (pure) ---------------------------------------------------------------------------

def _utc_day(now_ms: int) -> str:
    return datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).date().isoformat()


def _fresh(day: str, resource: dict[str, str], owner: bool) -> dict[str, Any]:
    return {"day": day, "resource": resource, "surfaces": {}, "models": {}, "profiles": [], "owner": owner}


def _valid_state(state: Any) -> bool:
    return (
        isinstance(state, dict) and isinstance(state.get("day"), str)
        and isinstance(state.get("surfaces"), dict) and isinstance(state.get("models"), dict)
    )


def apply(
    state: dict[str, Any] | None, *, now_ms: int, resource: dict[str, str],
    surface: str | None = None, route: tuple[str, str] | None = None, profile: str | None = None,
    owner: bool = False,
) -> tuple[dict[str, Any], list[tuple[str, dict, dict, str]]]:
    """Fold one interaction (``surface``), attended turn (``route``) and/or a profile's user turn
    (``profile``, host accumulator only) into the day's state; close the stored day first when the
    clock has moved past it."""
    today, rows = _utc_day(now_ms), []
    if state is not None and not _valid_state(state):
        state = None
    if state is not None and state["day"] < today:
        # The only close path. A stored day ahead of the clock (a step back across midnight) keeps
        # the interaction: the stored day never moves backwards, so no day is re-opened or emitted twice.
        rows, state = close_day(state), None
    state = state or _fresh(today, resource, owner)
    state["owner"] = owner
    profiles = state.setdefault("profiles", [])
    if profile is not None and isinstance(profiles, list) and profile not in profiles:
        profiles.append(profile)
    if surface in contract.ENGAGEMENT_SURFACES:
        for stream in (surface, _ALL):
            active_ms, last_ms = state["surfaces"].get(stream) or (0, None)
            if isinstance(last_ms, int):
                active_ms += min(max(0, now_ms - last_ms), IDLE_CAP_MS)
            state["surfaces"][stream] = [active_ms, max(now_ms, last_ms or 0)]
    if route is not None:
        key = " ".join(route)
        state["models"][key] = int(state["models"].get(key, 0)) + 1
    return state, rows


def _primary(models: dict[str, Any]) -> tuple[str, str]:
    counted = [(int(n), key) for key, n in models.items() if isinstance(n, int) and key.count(" ") == 1]
    if not counted:
        return "none", "none"
    provider, model = max(counted)[1].split(" ")
    return provider, model


def close_day(state: dict[str, Any]) -> list[tuple[str, dict, dict, str]]:
    """The rows one closed day reports: none for a day with no engaged surface, except the root
    profile's host row (0 minutes, 0 surfaces) carrying the day's active-profile count."""
    day, resource = state["day"], state.get("resource") or {}
    streams = {k: v for k, v in state["surfaces"].items() if isinstance(v, list) and len(v) == 2}
    surfaces = {k: v for k, v in streams.items() if k in contract.ENGAGEMENT_SURFACES}
    profiles = state.get("profiles") if state.get("owner") is True else None
    active_profiles = len(profiles) if isinstance(profiles, list) else 0
    if not surfaces and not active_profiles:
        return []
    rows = [
        (contract.ENGAGEMENT_SURFACE_METRIC,
         {"active_minutes_bucket": active_minutes_bucket(int(v[0])), "surface": name}, resource, day)
        for name, v in sorted(surfaces.items())
    ]
    provider, model = _primary(state["models"])
    rows.append((contract.ENGAGEMENT_DAY_METRIC, {
        "active_minutes_bucket": active_minutes_bucket(int((streams.get(_ALL) or [0])[0])),
        "active_profile_count_bucket": contract.size_bucket(active_profiles),
        "primary_model": model, "primary_provider": provider,
        "surfaces_used_count": surfaces_used_bucket(len(surfaces)),
    }, resource, day))
    return rows


# ---- per-conversation state kept by the relay runtime -------------------------------------------

@dataclass
class RouteRun:
    """Consecutive turns the conversation's current model served since the last /model switch."""

    route: dict[str, str] | None = None
    turns: int = 0

    def observe(self, route: dict[str, str]) -> None:
        if route == self.route:
            self.turns += 1
        else:
            self.route, self.turns = route, 1

    def take(self) -> tuple[dict[str, str], int] | None:
        """The model being left and its turn count; resets (the next model starts from zero)."""
        taken = (self.route, self.turns) if self.route is not None and self.turns else None
        self.route, self.turns = None, 0
        return taken


def switch_after_fields(route: dict[str, str], turns: int) -> dict[str, str]:
    return {**route, "turns_before_switch_bucket": turns_before_switch_bucket(turns)}


@dataclass
class SessionTally:
    """A session summary that merges the segments compression rotation splits one conversation into."""

    start_fields: dict[str, str] | None = None
    turns: int = 0
    failed_turns: int = 0
    last_outcome: str = "unknown"
    first_turn_ns: int = 0
    last_turn_ns: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    replies: int = 0

    def absorb(self, other: SessionTally) -> None:
        if not other.turns:
            return
        if not self.turns or other.first_turn_ns < self.first_turn_ns:
            self.start_fields, self.first_turn_ns = other.start_fields, other.first_turn_ns
        if other.last_turn_ns >= self.last_turn_ns:
            self.last_outcome, self.last_turn_ns = other.last_outcome, other.last_turn_ns
        self.turns += other.turns
        self.failed_turns += other.failed_turns
        self.model_calls += other.model_calls
        self.tool_calls += other.tool_calls
        self.replies += other.replies

    def fields(self) -> dict[str, str] | None:
        if not self.turns or self.start_fields is None:
            return None
        from .shared_metrics_fields import session_fields

        return {
            **session_fields(
                self.start_fields, turns=self.turns, failed_turns=self.failed_turns, last_outcome=self.last_outcome,
                active_ms=max(0, self.last_turn_ns - self.first_turn_ns) // 1_000_000,
            ),
            # Messages the conversation added: each user turn, each primary-model reply, each tool result.
            "message_count_bucket": contract.long_size_bucket(self.turns + self.replies + self.tool_calls),
            "user_turn_count_bucket": contract.long_size_bucket(self.replies),
            "model_call_count_bucket": contract.long_size_bucket(self.model_calls),
            "tool_call_count_bucket": contract.long_size_bucket(self.tool_calls),
        }
