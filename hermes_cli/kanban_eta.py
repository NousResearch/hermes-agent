"""kanban_eta.py — clock-time ETA engine for the kanban dock (t_cf6770dc owns).

Copied VERBATIM into hermes_cli/ by t_f4b3a89f per KANBAN-DOCK-CONTRACT-2026-09-21 §4.
Only t_cf6770dc may edit afterwards, and only inside estimate_task_hours()'s v2 block
and the tests — signatures are binding (the REST layer and pane call them blind).

v2 (this revision, t_cf6770dc): estimate_task_hours() now calls the auxiliary
kanban_estimator LLM door — the same door plugins/kanban/dashboard/plugin_api.py's
_run_estimate() uses — calibrated for GLM-5.3-flash-class cheap workers. Any failure
returns (1.0, 'stub-table'); the signature is unchanged. project_backlog() and the
stub table are untouched.

Deep spec (verbatim, 2026-09-21): clock time ETAs "computed based on ... initial
estimate of time cost then updated continuously presuming continuous execution
maximum blast using cheaper GLM 5.3 or similar workers until backlog cleared",
rendered in 12-hour clock time.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import time
from typing import Any, Iterable, Mapping, Optional

# Sane default table (Deep's directive: "put a sane default table in and mark
# estimates 'est.'"). Calibrated for GLM-5.3-flash-class cheap workers. Retained
# in v2 as the calibration anchor baked into the estimator prompt (and the
# fallback contract: failures return DEFAULT_HOURS, 'stub-table').
DEFAULT_HOURS = 1.0
_STUB_TABLE = [
    (("rebuild", "migration", "migrate", "design", "architecture", "survey", "rewrite"), 3.0),
    (("fix", "verify", "wire", "route", "config", "repoint", "rewire", "restart"), 1.0),
]
_S_HOURS = 0.25

VALID_BANDS = ("P0", "P1", "P2", "P3")
BAND_CENTERS = {"P0": 8, "P1": 6, "P2": 4, "P3": 2}  # -> tasks.priority on band change (contract D3)
P_MAX_DEFAULT = 8  # max-blast lane count; config kanban.eta_max_parallel overrides

# --- v2 estimator (GLM-5.3-flash calibration) --------------------------------

EST_HOURS_MIN, EST_HOURS_MAX = 0.1, 24.0
EST_SOURCE_LLM = "glm-5.3-flash"
EST_SOURCE_STUB = "stub-table"
_AFFINITY_KEY = "kanban:eta-engine"  # headless aux door needs a stable relay-affinity key

_ESTIMATOR_SYSTEM_PROMPT = (
    "You estimate the wall-clock working time one cheap autonomous worker (a GLM-5.3-flash "
    "class model in an agent harness, with file/shell tools on a Windows dev machine) needs "
    "to take a kanban task from open to verified done — a realistic multi-turn agent run "
    "(reading files, tool calls, edits, tests, retries), not a single chat reply.\n"
    "Anchor on this baseline table, then adjust for the scope the description signals:\n"
    "  L (rebuild/migration/design/architecture/survey/rewrite): 3.0 h\n"
    "  M (fix/verify/wire/route/config/repoint/restart): 1.0 h\n"
    "  S (anything localized): 0.25 h\n"
    "Move up for breadth (many files, new subsystems, verification loops, ambiguity), down "
    "for one-liners. Stay within 0.1-24 h.\n"
    "Answer with ONLY a JSON object: "
    '{"est_hours": <number of wall-clock hours>, "complexity": "S"|"M"|"L", '
    '"rationale": "<one short sentence>"} '
    "Be honest that this is a rough guess."
)


def _cap(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _llm_user_msg(title: str, body: Optional[str]) -> str:
    return f"Title: {_cap(title, 400)}\n\nDescription:\n{_cap(body, 4000) or '(none)'}"


def _parse_est_hours(raw: str) -> Optional[float]:
    """Same tolerant JSON-blob extraction _run_estimate uses; None on any mismatch."""
    try:
        m = None if raw.lstrip().startswith("{") else re.search(r"\{.*\}", raw, re.DOTALL)
        obj = json.loads(m.group(0) if m else raw)
        if not isinstance(obj, dict):
            return None
        hours = float(obj.get("est_hours"))
    except Exception:
        return None
    if not math.isfinite(hours):
        return None
    return round(min(max(hours, EST_HOURS_MIN), EST_HOURS_MAX), 2)


def _estimate_hours_via_llm(title: str, body: Optional[str]) -> Optional[float]:
    """One aux kanban_estimator call -> clamped wall-clock hours, or None (never raises).

    Mirrors _run_estimate()'s headless discipline: the OpenCode Go relay rejects
    affinity-less requests with 400 MissingSessionID (#112043), so a stable
    per-engine scope key is declared unless one is already bound.
    Tests/CI can force the stub path with HERMES_KANBAN_ETA_STUB=1 (no LLM,
    no network) instead of relying on the aux import failing.
    """
    import os
    if os.environ.get("HERMES_KANBAN_ETA_STUB"):
        return None
    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return None
    affinity_token = None
    try:
        try:
            from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope
            if not get_affinity_scope():
                affinity_token = set_affinity_scope(_AFFINITY_KEY)
        except Exception:
            affinity_token = None  # no portal_tags -> proceed without affinity
        resp = call_llm(
            task="kanban_estimator",
            messages=[{"role": "system", "content": _ESTIMATOR_SYSTEM_PROMPT},
                      {"role": "user", "content": _llm_user_msg(title, body)}],
            temperature=0.0, max_tokens=200, timeout=60)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
    finally:
        if affinity_token is not None:
            try:
                from agent.portal_tags import reset_affinity_scope
                reset_affinity_scope(affinity_token)
            except Exception:
                pass
    return _parse_est_hours(raw)


def estimate_task_hours(title: str, body: Optional[str] = None) -> tuple[float, str]:
    """Initial per-task time-cost estimate. v2: aux kanban_estimator LLM call
    calibrated for GLM-5.3-flash-class workers (source 'glm-5.3-flash'); any
    failure -> (1.0, 'stub-table'). Never raises."""
    text = f"{title or ''} {body or ''}".lower()
    if not text.strip():
        return DEFAULT_HOURS, EST_SOURCE_STUB
    hours = _estimate_hours_via_llm(title or "", body)
    if hours is not None:
        return hours, EST_SOURCE_LLM
    # LLM door failed -> the §4 stub table is still the sane default (Deep's
    # directive); only when even it has no signal do we take the 1.0h default.
    for needles, stub_hours in _STUB_TABLE:
        if any(n in text for n in needles):
            return stub_hours, EST_SOURCE_STUB
    return DEFAULT_HOURS, EST_SOURCE_STUB


def _band_of(t: Mapping[str, Any]) -> str:
    b = t.get("p_band") or "P2"
    return b if b in VALID_BANDS else "P2"


def _dock_order_of(t: Mapping[str, Any]) -> tuple[int, int]:
    do = t.get("dock_order")
    return (1, 0) if do is None else (0, int(do))


def execution_order(tasks: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """(p_band, dock_order NULLS-last, created_at, id) — the pane and the
    projection agree on this single order (contract D3/D4)."""
    def key(t: Mapping[str, Any]) -> tuple:
        created = t.get("created_at") or 0
        return (_band_of(t), *_dock_order_of(t), created, str(t.get("id")))

    return sorted(tasks, key=key)


def _remaining_hours(t: Mapping[str, Any], now: float) -> float:
    est = t.get("est_hours")
    est = DEFAULT_HOURS if est is None else float(est)
    started = t.get("started_at")
    if t.get("status") == "running" and started:
        est = max(0.05, est - (now - float(started)) / 3600.0)
    return est


def project_backlog(
    tasks: Iterable[Mapping[str, Any]],
    now: Optional[float] = None,
    p_max: int = P_MAX_DEFAULT,
) -> dict[str, Any]:
    """Continuous max-blast projection on cheap flash-class workers.

    Model: P = min(p_max, open_count) lanes; running tasks pre-occupy lanes at
    now+remaining; queued tasks, in execution_order, start on the
    earliest-freeing lane. Returns epochs (seconds); rendering (12-hour clock)
    is the frontend's job, backend `now` is authoritative.
    """
    now = time.time() if now is None else float(now)
    open_tasks = [t for t in tasks if t.get("status") not in ("done", "archived")]
    order = execution_order(open_tasks)
    parallelism = min(p_max, len(order)) if p_max and p_max > 0 else max(1, len(order))

    per_task: dict[str, float] = {}
    # Min-heap of lane-free epochs; running tasks seed their lanes.
    lanes: list[float] = []
    for t in order:
        if t.get("status") == "running":
            finish = now + _remaining_hours(t, now) * 3600.0
            heapq.heappush(lanes, finish)
            per_task[str(t["id"])] = finish
    while len(lanes) < parallelism:
        heapq.heappush(lanes, now)

    for t in order:
        tid = str(t["id"])
        if tid in per_task:  # already running — lane pre-occupied
            continue
        start = heapq.heappop(lanes)
        finish = start + _remaining_hours(t, now) * 3600.0
        heapq.heappush(lanes, finish)
        per_task[tid] = finish

    backlog_clear_at = max(per_task.values()) if per_task else None
    return {"parallelism": parallelism, "backlog_clear_at": backlog_clear_at, "per_task": per_task}
