"""``hermes peer deploy`` — remote health gate for peer deployments.

The deploy health check polls the remote host after a code swap / restart and
rolls back when the gateway does not become healthy in time. Tier 1 compares the
``gateway_state.json`` file mtime against the restart time (freshness); Tier 2
requires every *live* platform to report ``state == "connected"``.

Tier 2 must only consider platform entries written by the CURRENT gateway
process. Gateway startup preserves plain platform entries in
``gateway_state.json`` across restarts, so the raw ``platforms`` map can carry a
stale row for an adapter that is no longer configured, written days earlier by a
now-dead process (issue #130708: a 9-day-old ``feishu`` row in ``retrying`` from
a dead pid kept ``ALL`` permanently larger than ``CONNECTED``, failing every
deploy while ``/health`` returned 200). The status writer stamps
``writer_pid``/``writer_start_time`` on every entry (``gateway/status.py``), so
the probe filters by exact ``(pid, start_time)`` writer identity — the same
ownership rule as ``hermes_cli/web_server_gateway.py::_owned_profile_platforms``
used by the dashboard. Legacy entries without identity never match and are
excluded.

Fail-open note (explicit per #130708): when ``owned`` is empty — no live
identity in the state file, or a state file predating the stamping — the gate
reports healthy ("nothing owned to gate on") instead of failing forever. The
dashboard treats that case as not-degraded for the same reason; a false
"degraded forever" is the worse failure mode. Tier 1 freshness and ``/health``
still guard a genuinely down gateway.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

#: How long the gate waits for the remote gateway to become healthy.
HEALTH_TIMEOUT_S = 90

#: Platform state that counts as serving for the deploy gate. Intentionally
#: strict: a live owned platform that is still ``retrying`` mid-reconnect fails
#: the gate (the Tier-2 intent — a listed-but-disconnected platform must never
#: read as healthy). Only rows the current process never wrote are excluded.
_CONNECTED_STATE = "connected"


def platform_state(entry: Any) -> str:
    """Normalized platform state for one ``platforms`` map entry."""
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("state") or entry.get("status") or "").strip().lower()


def owned_platforms(
    platforms: Any, pid: Any, start_time: Any
) -> Dict[str, Dict[str, Any]]:
    """Keep only platform entries stamped by the current gateway process.

    Mirrors ``hermes_cli/web_server_gateway.py::_owned_profile_platforms``:
    exact ``(writer_pid, writer_start_time) == (pid, start_time)`` match.
    Non-dict values and legacy entries without writer identity never match.
    """
    if not isinstance(platforms, dict):
        return {}
    live = (pid, start_time)
    return {
        key: value
        for key, value in platforms.items()
        if isinstance(value, dict)
        and (value.get("writer_pid"), value.get("writer_start_time")) == live
    }


def connected_owned_platforms(owned: Dict[str, Dict[str, Any]]) -> List[str]:
    """Sorted names of owned platforms whose state is connected."""
    return sorted(
        name for name, entry in owned.items() if platform_state(entry) == _CONNECTED_STATE
    )


def evaluate_platform_gate(
    runtime_status: Any,
) -> Tuple[bool, List[str], List[str], List[str]]:
    """Tier-2 health predicate over one ``gateway_state.json`` payload.

    Returns ``(healthy, all_owned, connected_owned, missing)`` where ``all``
    and ``connected`` derive from writer-identity-filtered (owned) platforms,
    so stale rows from dead gateway processes are ignored. Empty ``owned``
    reports healthy (fail-open; see module docstring).
    """
    runtime = runtime_status if isinstance(runtime_status, dict) else {}
    plats = runtime.get("platforms", {})
    owned = owned_platforms(plats, runtime.get("pid"), runtime.get("start_time"))
    all_p = sorted(owned)
    conn_p = connected_owned_platforms(owned)
    if not all_p:
        return True, all_p, conn_p, []
    missing = sorted(set(all_p) - set(conn_p))
    return (len(missing) == 0, all_p, conn_p, missing)


def parse_probe_output(output: str) -> Tuple[List[str], List[str]]:
    """Parse the remote probe's ``ALL``/``CONNECTED`` lines into name lists."""
    all_p: List[str] = []
    conn_p: List[str] = []
    for line in (output or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("ALL"):
            rest = stripped[3:].strip()
            all_p = [] if rest in ("", "none") else sorted(p for p in rest.split(",") if p)
        elif stripped.startswith("CONNECTED"):
            rest = stripped[9:].strip()
            conn_p = [] if rest in ("", "none") else sorted(p for p in rest.split(",") if p)
    return all_p, conn_p


def platforms_healthy(all_p: List[str], conn_p: List[str]) -> Tuple[bool, List[str]]:
    """Tier-2 set-equality check over probe-derived populations.

    Empty ``all_p`` reports healthy (fail-open; see module docstring) — there
    is no owned platform to wait for.
    """
    all_sorted = sorted(all_p or [])
    conn_sorted = sorted(conn_p or [])
    if not all_sorted:
        return True, []
    missing = sorted(set(all_sorted) - set(conn_sorted))
    return (all_sorted == conn_sorted, missing)


# stdlib-only snippet sent to the remote host. It reads the state path given
# as ``sys.argv[1]`` and prints the ``ALL``/``CONNECTED`` lines the gate parses.
# Both populations derive from writer-identity-filtered (owned) platforms so a
# stale row from a dead gateway process can never fail the gate (#130708).
HEALTH_PROBE_SCRIPT = """\
import json
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    d = json.load(fh)
plats = d.get('platforms', {}) or {}
live = (d.get('pid'), d.get('start_time'))
owned = {k: v for k, v in plats.items()
         if isinstance(v, dict)
         and (v.get('writer_pid'), v.get('writer_start_time')) == live}
def state(v):
    return str((v.get('state') or v.get('status') or '')).strip().lower() if isinstance(v, dict) else ''
conn = [k for k, v in owned.items() if state(v) == 'connected']
print('ALL', ','.join(sorted(owned)) or 'none')
print('CONNECTED', ','.join(sorted(conn)) or 'none')
"""


def render_probe_command(state_path: str) -> List[str]:
    """Local runnable form of the remote probe (``python3 -c <script> <path>``)."""
    return [sys.executable, "-c", HEALTH_PROBE_SCRIPT, state_path]


def probe_runtime_status_file(path: str) -> Optional[Dict[str, Any]]:
    """Read and JSON-decode a ``gateway_state.json`` file; None when unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


#: gateway_state values that count as alive for Tier-1 freshness (mirrors
#: ``gateway/status.py::_DRAINABLE_GATEWAY_STATES``).
REMOTE_FRESH_STATES = frozenset({"running", "degraded"})

#: Max age of the peer's ``updated_at`` before its liveness claim is suspect
#: (mirrors ``gateway/status.py::_RUNTIME_STATUS_STALE_TTL_S``).
FRESHNESS_TTL_S = 120


def evaluate_freshness(
    gateway_state: Any, updated_at: Any, now: Optional[float] = None,
) -> Tuple[bool, str]:
    """Tier-1 freshness over one ``/health/detailed`` payload.

    Returns ``(fresh, reason)``: the peer must report a live ``gateway_state``
    AND a recent ``updated_at``. A missing ``updated_at`` fails closed here
    (unlike Tier 2's fail-open): without any recency signal there is no proof
    the process that wrote the state is still alive.
    """
    import datetime
    import time as _time

    state = str(gateway_state or "").strip().lower()
    if state not in REMOTE_FRESH_STATES:
        return False, f"gateway_state is {gateway_state!r}, not a live state"
    if not isinstance(updated_at, str) or not updated_at.strip():
        return False, "no updated_at in the health payload; freshness unproven"
    try:
        stamp = datetime.datetime.fromisoformat(updated_at.strip().replace("Z", "+00:00"))
        age_s = (_time.time() if now is None else now) - stamp.timestamp()
    except (ValueError, OverflowError):
        return False, f"unparseable updated_at {updated_at!r}"
    if age_s > FRESHNESS_TTL_S:
        return False, f"updated_at is {age_s:.0f}s old (stale writer?)"
    return True, "live state with a fresh heartbeat"


def evaluate_remote_gate(payload: Any) -> Dict[str, Any]:
    """Tier-1 + Tier-2 over one ``GET /health/detailed`` response dict.

    Returns a JSON-able result with ``healthy`` (both tiers), the Tier-1
    verdict and the Tier-2 ``(all, connected, missing)`` populations.
    """
    payload = payload if isinstance(payload, dict) else {}
    fresh, fresh_reason = evaluate_freshness(payload.get("gateway_state"), payload.get("updated_at"))
    runtime_shaped = {
        "platforms": payload.get("platforms", {}),
        "pid": payload.get("pid"),
        "start_time": payload.get("start_time"),
    }
    tier2_ok, all_p, conn_p, missing = evaluate_platform_gate(runtime_shaped)
    return {
        "healthy": bool(fresh and tier2_ok),
        "tier1_fresh": fresh,
        "tier1_reason": fresh_reason,
        "tier2_healthy": tier2_ok,
        "all_owned": all_p,
        "connected_owned": conn_p,
        "missing": missing,
    }
