"""Body-state collection for the proprioception plugin.

Two external senses plus one internal:

* **Dashboard rollup** — ``GET /api/home`` on the Onyx Command Center
  (a local HttpListener that already aggregates ~19 systems into
  ``{verdict, needs[], systems[{id, state, label, detail, cat}]}``).
  We deliberately reuse it instead of probing the systems ourselves:
  one collector on the machine, not two drifting ones.
* **Gateway runtime status** — ``{HERMES_HOME}/gateway_state.json`` via
  :func:`gateway.status.read_runtime_status` (file read, no HTTP, works
  even when the API server is disabled).
* **Context estimate** — computed by the caller from the live message
  list and passed into the heartbeat, never fetched here.

A failing source is *data* ("sensor offline"), never an exception into
the agent loop.  The snapshot is cached process-wide with a short TTL so
concurrent sessions share one fetch.

Latency contract: the fetch happens OUTSIDE the cache lock. While one
thread refreshes, other threads are served the previous (stale) snapshot
immediately instead of queueing behind the HTTP call — a slow dashboard
must never serialize turn prologues across sessions.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# States (from the dashboard) that mean "something needs attention".
# Anything not in this set — "ok" and the informational "info" — is calm.
ATTENTION_STATES = frozenset({"warn", "warning", "down", "error", "crit", "critical"})


@dataclass
class Snapshot:
    """One reading of the body. Sensor failures are recorded, not raised."""

    fetched_at: float  # time.monotonic() — interval math only, never displayed
    dashboard: Optional[Dict[str, Any]] = None
    dashboard_error: str = ""
    gateway: Optional[Dict[str, Any]] = None
    gateway_error: str = ""
    # >0 when the dashboard payload was served from the last-known-good
    # grace window: seconds since that data was actually fetched. The
    # heartbeat stays silent during grace, but the body_state tool must
    # disclose the staleness — the deliberate look never lies about age.
    dashboard_stale_for: float = 0.0

    @property
    def sensors_down(self) -> Tuple[str, ...]:
        out = []
        if self.dashboard is None:
            out.append("dashboard")
        if self.gateway is None:
            out.append("gateway-status")
        return tuple(out)

    @property
    def gateway_state(self) -> str:
        if self.gateway is None:
            return "unknown"
        return str(self.gateway.get("state") or self.gateway.get("gateway_state") or "?")


_CACHE_LOCK = threading.Lock()
_CACHED: Optional[Snapshot] = None
_REFRESH_IN_FLIGHT = False

# Last successful dashboard payload, kept so a couple of missed polls (the
# dashboard is a single-threaded listener; an occasional timeout is normal)
# don't read as "sensor lost" and produce loss/recovery chatter. A fetch
# failure only becomes sensor-down once the last good reading is older than
# the configurable grace (settings key ``stale_grace_seconds``).
_LAST_GOOD_DASHBOARD: Optional[Dict[str, Any]] = None
_LAST_GOOD_AT: float = 0.0


def _fetch_dashboard(url: str, timeout: float) -> Dict[str, Any]:
    import requests

    # (connect, read) tuple: requests' scalar timeout is per-read-chunk,
    # not wall-clock; the tuple at least bounds each phase separately.
    resp = requests.get(url, timeout=(timeout, timeout))
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or "systems" not in payload:
        raise ValueError("unexpected /api/home shape")
    return payload


def _fetch_gateway_status() -> Optional[Dict[str, Any]]:
    from gateway.status import read_runtime_status

    return read_runtime_status()


def _do_refresh(settings: Dict[str, Any]) -> Snapshot:
    """Perform the actual (potentially slow) collection. No locks held."""
    global _LAST_GOOD_DASHBOARD, _LAST_GOOD_AT
    snap = Snapshot(fetched_at=time.monotonic())
    try:
        snap.dashboard = _fetch_dashboard(
            str(settings["dashboard_url"]), float(settings["timeout_seconds"])
        )
        _LAST_GOOD_DASHBOARD = snap.dashboard
        _LAST_GOOD_AT = snap.fetched_at
    except Exception as exc:  # sensor down is data, not an error
        snap.dashboard_error = f"{type(exc).__name__}: {exc}"[:200]
        logger.debug("proprioception: dashboard fetch failed: %s", snap.dashboard_error)
        # Grace window: reuse the last good reading rather than flapping
        # to sensor-down on a missed poll or two — but record the data's
        # true age so the body_state tool can disclose it.
        grace = float(settings.get("stale_grace_seconds", 90))
        age = snap.fetched_at - _LAST_GOOD_AT
        if _LAST_GOOD_DASHBOARD is not None and age < grace:
            snap.dashboard = _LAST_GOOD_DASHBOARD
            snap.dashboard_stale_for = age
    try:
        snap.gateway = _fetch_gateway_status()
        if snap.gateway is None:
            snap.gateway_error = "no gateway_state.json"
    except Exception as exc:
        snap.gateway_error = f"{type(exc).__name__}: {exc}"[:200]
        logger.debug("proprioception: gateway status read failed: %s", snap.gateway_error)
    return snap


def get_snapshot(settings: Dict[str, Any], *, force: bool = False) -> Snapshot:
    """Return the current body snapshot, cached for ``cache_ttl_seconds``.

    Never raises. Thread-safe. At most one thread refreshes at a time;
    concurrent callers get the previous snapshot (stale-while-revalidate)
    rather than blocking behind the HTTP fetch.
    """
    global _CACHED, _REFRESH_IN_FLIGHT
    ttl = float(settings["cache_ttl_seconds"])
    with _CACHE_LOCK:
        cached = _CACHED
        fresh = cached is not None and (time.monotonic() - cached.fetched_at) < ttl
        if not force and fresh:
            return cached
        if _REFRESH_IN_FLIGHT and cached is not None and not force:
            return cached  # serve stale; another thread is already refreshing
        _REFRESH_IN_FLIGHT = True

    try:
        snap = _do_refresh(settings)
    except Exception as exc:  # _do_refresh guards internally; this is belt+braces
        logger.debug("proprioception: refresh failed outright: %s", exc)
        snap = Snapshot(fetched_at=time.monotonic(), dashboard_error=str(exc)[:200])
    finally:
        with _CACHE_LOCK:
            _CACHED = snap
            _REFRESH_IN_FLIGHT = False
    return snap


def fingerprint(snap: Snapshot) -> Tuple:
    """Reduce a snapshot to the fields whose *change* is material.

    Deliberately excludes free-text ``detail`` strings AND ``needs[]``
    text (both embed live numbers — VRAM, hour counts, GB free — that
    wobble between readings; diffing them would make the heartbeat
    chatty with no state transition behind it). System ``state`` fields
    plus the overall verdict and gateway state carry the signal; every
    fingerprint field has a renderer in the heartbeat, so a fingerprint
    change can never produce an empty change message.
    """
    dash_part: Tuple = ("sensor-down",)
    if snap.dashboard is not None:
        systems = snap.dashboard.get("systems") or []
        dash_part = (
            str(snap.dashboard.get("verdict", "")),
            tuple(
                sorted(
                    (str(s.get("id", "?")), str(s.get("state", "?")))
                    for s in systems
                    if isinstance(s, dict)
                )
            ),
        )
    return (dash_part, (snap.gateway_state,))


def diff_systems(
    prev: Optional[Snapshot], cur: Snapshot
) -> Tuple[Tuple[str, str, str], ...]:
    """Per-system (label, old_state, new_state) transitions between snapshots.

    Systems present only on one side are reported against ``"absent"``.
    Empty when either side has no dashboard data (sensor transitions are
    handled separately by the caller).
    """
    if prev is None or prev.dashboard is None or cur.dashboard is None:
        return ()

    def _by_id(s: Snapshot) -> Dict[str, Dict[str, Any]]:
        return {
            str(sys_.get("id", "?")): sys_
            for sys_ in (s.dashboard.get("systems") or [])
            if isinstance(sys_, dict)
        }

    old, new = _by_id(prev), _by_id(cur)
    out = []
    for sid in sorted(set(old) | set(new)):
        old_state = str(old[sid].get("state", "?")) if sid in old else "absent"
        new_state = str(new[sid].get("state", "?")) if sid in new else "absent"
        if old_state != new_state:
            label = str((new.get(sid) or old.get(sid) or {}).get("label", sid))
            out.append((label, old_state, new_state))
    return tuple(out)


def reset_cache_for_tests() -> None:
    global _CACHED, _REFRESH_IN_FLIGHT, _LAST_GOOD_DASHBOARD, _LAST_GOOD_AT
    with _CACHE_LOCK:
        _CACHED = None
        _REFRESH_IN_FLIGHT = False
        _LAST_GOOD_DASHBOARD = None
        _LAST_GOOD_AT = 0.0
