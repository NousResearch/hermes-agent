"""Hermes Metrics dashboard plugin backend.

Mounted at /api/plugins/hermes-metrics/ by the Hermes dashboard.

Read-only aggregation over:
  - ~/.hermes/data/spend_ledger.json + spend_throttle.json (written by the
    spend-guard poller — this API never recomputes spend, only reads it)
  - the kanban board DB (tasks / task_runs / task_events)
  - root + per-profile state.db sessions/messages tables

All SQLite opens use mode=ro so the dashboard can never contend with the
gateway's writers. Results are snapshot-cached for a short TTL because the
frontend polls every 30s.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from hermes_constants import get_hermes_home
except ImportError:
    import os as _os

    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (_os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"


try:
    from fastapi import APIRouter
except Exception:  # Allows local unit tests without dashboard dependencies.

    class APIRouter:  # type: ignore
        def get(self, *_args, **_kwargs):
            return lambda fn: fn

        def post(self, *_args, **_kwargs):
            return lambda fn: fn


router = APIRouter()

SNAPSHOT_TTL_SECONDS = 20
_LOCK = threading.Lock()
_CACHE: Dict[str, tuple[float, Any]] = {}

DAY = 86400
WINDOW_DAYS = 14


def _cached(key: str, compute):
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < SNAPSHOT_TTL_SECONDS:
            return hit[1]
    value = compute()
    with _LOCK:
        _CACHE[key] = (time.monotonic(), value)
    return value


def _ro_connect(path: Path) -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _session_dbs() -> List[tuple[str, Path]]:
    home = get_hermes_home()
    out: List[tuple[str, Path]] = []
    if (home / "state.db").exists():
        out.append(("default", home / "state.db"))
    profiles = home / "profiles"
    if profiles.is_dir():
        for entry in sorted(profiles.iterdir()):
            if entry.is_dir() and (entry / "state.db").exists():
                out.append((entry.name, entry / "state.db"))
    return out


def _kanban_conn() -> Optional[sqlite3.Connection]:
    return _ro_connect(get_hermes_home() / "kanban.db")


# ─── Spend ───────────────────────────────────────────────────────────────────


def compute_spend() -> Dict[str, Any]:
    home = get_hermes_home()
    ledger = _read_json(home / "data" / "spend_ledger.json")
    throttle = _read_json(home / "data" / "spend_throttle.json")
    caps: Dict[str, Any] = {}
    labels: Dict[str, str] = {}
    throttle_enabled = False
    try:
        import sys

        repo = home / "hermes-agent"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from agent import spend_meter

        cfg = spend_meter.load_spend_config()
        throttle_enabled = cfg.enabled and cfg.throttle_enabled
        for lane, vals in cfg.lanes.items():
            caps[lane] = vals.get("daily_cap_usd")
            labels[lane] = vals.get("label") or lane
    except Exception:
        pass
    history = []
    for date_str, day in sorted((ledger.get("history") or {}).items())[-WINDOW_DAYS:]:
        history.append(
            {
                "date": date_str,
                "lanes": {
                    lane: round(float(info.get("usd", 0)), 4)
                    for lane, info in (day.get("lanes") or {}).items()
                },
            }
        )
    history.append(
        {
            "date": ledger.get("date"),
            "lanes": {
                lane: round(float(info.get("usd", 0)), 4)
                for lane, info in (ledger.get("lanes") or {}).items()
            },
        }
    )
    return {
        "date": ledger.get("date"),
        "last_poll": ledger.get("last_poll"),
        "lanes": ledger.get("lanes") or {},
        "profiles": ledger.get("profiles") or {},
        "caps": caps,
        "labels": labels,
        "history": history,
        "pricing_gaps": ledger.get("pricing_gaps") or [],
        "account_usage": ledger.get("account_usage") or {},
        "routing": throttle.get("routing") or {},
        "throttle": {
            "enabled": throttle_enabled,
            "paused_lanes": throttle.get("paused") or {},
            "paused_profiles": throttle.get("paused_profiles") or {},
            "swapped_lanes": throttle.get("swapped") or {},
            "overrides": throttle.get("overrides") or {},
        },
    }


# ─── Usage (tokens per profile) ──────────────────────────────────────────────


def compute_usage() -> Dict[str, Any]:
    floor = time.time() - 7 * DAY
    profiles: List[Dict[str, Any]] = []
    models: Dict[str, Dict[str, int]] = {}
    for profile, db in _session_dbs():
        conn = _ro_connect(db)
        if conn is None:
            continue
        try:
            row = conn.execute(
                "SELECT COUNT(*) sessions, COALESCE(SUM(input_tokens),0) inp,"
                " COALESCE(SUM(output_tokens),0) out, COALESCE(SUM(cache_read_tokens),0) cr,"
                " COALESCE(SUM(cache_write_tokens),0) cw, COALESCE(SUM(api_call_count),0) api,"
                " COALESCE(SUM(tool_call_count),0) tools"
                " FROM sessions WHERE started_at >= ?",
                (floor,),
            ).fetchone()
            if row and row["sessions"]:
                # Context volume per API call — the fleet's dominant cost
                # driver (cache writes+reads dwarf output ~7:1). This is
                # the KPI the token-economy protocol should push down.
                ctx = row["inp"] + row["cr"] + row["cw"]
                profiles.append(
                    {
                        "profile": profile,
                        "sessions": row["sessions"],
                        "input_tokens": row["inp"],
                        "output_tokens": row["out"],
                        "cache_read_tokens": row["cr"],
                        "cache_write_tokens": row["cw"],
                        "api_calls": row["api"],
                        "tool_calls": row["tools"],
                        "context_per_call": int(ctx / row["api"]) if row["api"] else None,
                    }
                )
            for mrow in conn.execute(
                "SELECT model, COALESCE(SUM(input_tokens+output_tokens+cache_read_tokens"
                "+cache_write_tokens),0) total FROM sessions"
                " WHERE started_at >= ? AND model IS NOT NULL GROUP BY model",
                (floor,),
            ):
                bucket = models.setdefault(mrow["model"], {"total_tokens": 0})
                bucket["total_tokens"] += mrow["total"]
        except sqlite3.Error:
            pass
        finally:
            conn.close()
    profiles.sort(key=lambda p: -(p["input_tokens"] + p["output_tokens"]))
    return {"window_days": 7, "profiles": profiles, "models": models}


# ─── Kanban fleet ────────────────────────────────────────────────────────────


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct * (len(ordered) - 1)))))
    return ordered[idx]


def compute_kanban() -> Dict[str, Any]:
    conn = _kanban_conn()
    if conn is None:
        return {"error": "kanban.db unavailable"}
    floor = time.time() - WINDOW_DAYS * DAY
    try:
        by_status = {
            row["status"]: row["n"]
            for row in conn.execute(
                "SELECT status, COUNT(*) n FROM tasks GROUP BY status"
            )
        }
        by_assignee = {
            row["assignee"]: row["n"]
            for row in conn.execute(
                "SELECT COALESCE(assignee,'(none)') assignee, COUNT(*) n FROM tasks"
                " WHERE status NOT IN ('done','archived') GROUP BY 1 ORDER BY n DESC"
            )
        }
        throughput = [
            {"date": row["d"], "done": row["n"]}
            for row in conn.execute(
                "SELECT date(completed_at,'unixepoch','localtime') d, COUNT(*) n"
                " FROM tasks WHERE completed_at >= ? GROUP BY d ORDER BY d",
                (floor,),
            )
        ]
        durations = [
            row["dur"]
            for row in conn.execute(
                "SELECT completed_at - started_at dur FROM tasks"
                " WHERE completed_at >= ? AND started_at IS NOT NULL"
                " AND completed_at > started_at",
                (floor,),
            )
        ]
        now = time.time()
        running = [
            {
                "id": row["id"],
                "title": (row["title"] or "")[:80],
                "assignee": row["assignee"],
                "running_seconds": int(now - row["started_at"]) if row["started_at"] else None,
                "heartbeat_age_seconds": (
                    int(now - row["last_heartbeat_at"]) if row["last_heartbeat_at"] else None
                ),
            }
            for row in conn.execute(
                "SELECT id, title, assignee, started_at, last_heartbeat_at"
                " FROM tasks WHERE status = 'running' ORDER BY started_at"
            )
        ]
        oldest_ready = conn.execute(
            "SELECT MIN(created_at) t FROM tasks WHERE status = 'ready'"
        ).fetchone()
        oldest_ready_age = (
            int(now - oldest_ready["t"]) if oldest_ready and oldest_ready["t"] else None
        )
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    finally:
        conn.close()
    return {
        "window_days": WINDOW_DAYS,
        "by_status": by_status,
        "by_assignee_open": by_assignee,
        "throughput": throughput,
        "cycle_time_seconds": {
            "p50": _percentile(durations, 0.5),
            "p90": _percentile(durations, 0.9),
            "n": len(durations),
        },
        "running": running,
        "oldest_ready_age_seconds": oldest_ready_age,
    }


# ─── Agents (load + quality) ─────────────────────────────────────────────────

QUALITY_EVENT_KINDS = (
    "protocol_violation",
    "completion_blocked_hallucination",
    "gave_up",
)


def compute_agents() -> Dict[str, Any]:
    conn = _kanban_conn()
    if conn is None:
        return {"error": "kanban.db unavailable"}
    floor = time.time() - WINDOW_DAYS * DAY
    agents: Dict[str, Dict[str, Any]] = {}

    def bucket(profile: str) -> Dict[str, Any]:
        return agents.setdefault(
            profile,
            {
                "runs": 0,
                "outcomes": {},
                "avg_run_seconds": None,
                "events": {},
                "reviews_approved": 0,
                "reviews_rejected": 0,
                "sent_back": 0,
            },
        )

    try:
        for row in conn.execute(
            "SELECT profile, COALESCE(outcome, status) outcome, COUNT(*) n,"
            " AVG(CASE WHEN ended_at > started_at THEN ended_at - started_at END) avg_dur"
            " FROM task_runs WHERE started_at >= ? AND profile IS NOT NULL"
            " GROUP BY profile, COALESCE(outcome, status)",
            (floor,),
        ):
            entry = bucket(row["profile"])
            entry["runs"] += row["n"]
            entry["outcomes"][row["outcome"]] = row["n"]
        for row in conn.execute(
            "SELECT profile, AVG(ended_at - started_at) avg_dur FROM task_runs"
            " WHERE started_at >= ? AND ended_at > started_at AND profile IS NOT NULL"
            " GROUP BY profile",
            (floor,),
        ):
            bucket(row["profile"])["avg_run_seconds"] = round(row["avg_dur"] or 0)
        placeholders = ",".join("?" for _ in QUALITY_EVENT_KINDS)
        for row in conn.execute(
            f"SELECT COALESCE(t.assignee,'(none)') assignee, e.kind, COUNT(*) n"
            f" FROM task_events e JOIN tasks t ON t.id = e.task_id"
            f" WHERE e.kind IN ({placeholders}) AND e.created_at >= ?"
            f" GROUP BY 1, 2",
            (*QUALITY_EVENT_KINDS, floor),
        ):
            bucket(row["assignee"])["events"][row["kind"]] = row["n"]
        # Review verdicts are prose in task_runs.summary. A loose LIKE match
        # is ~88% noise (pm board-sweep summaries mention APPROVE
        # incidentally), so classify in Python with a line-anchored regex;
        # when both tokens appear, the LAST occurrence is the final verdict.
        # Attributed to the REVIEWED task's assignee — the profile whose
        # work was corrected.
        verdict_re = re.compile(
            r"(?mi)^\s*(?:verdict\s*[:=]?\s*)?(APPROVE|REQUEST-CHANGES)\b"
        )
        for row in conn.execute(
            "SELECT COALESCE(t.assignee,'(none)') assignee, r.summary"
            " FROM task_runs r JOIN tasks t ON t.id = r.task_id"
            " WHERE r.started_at >= ? AND"
            " (r.summary LIKE '%APPROVE%' OR r.summary LIKE '%REQUEST-CHANGES%')",
            (floor,),
        ):
            matches = verdict_re.findall(row["summary"] or "")
            if not matches:
                continue
            entry = bucket(row["assignee"])
            if matches[-1].upper() == "APPROVE":
                entry["reviews_approved"] += 1
            else:
                entry["reviews_rejected"] += 1
        for row in conn.execute(
            "SELECT COALESCE(t.assignee,'(none)') assignee, COUNT(*) n"
            " FROM task_events e JOIN tasks t ON t.id = e.task_id"
            " WHERE e.kind = 'blocked' AND e.created_at >= ?"
            " AND e.payload LIKE '%review-required%' GROUP BY 1",
            (floor,),
        ):
            bucket(row["assignee"])["sent_back"] = row["n"]
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    finally:
        conn.close()
    return {"window_days": WINDOW_DAYS, "agents": agents}


# ─── Refinements (user follow-up pressure) ───────────────────────────────────


def compute_refinements() -> Dict[str, Any]:
    floor = time.time() - WINDOW_DAYS * DAY
    out: Dict[str, Dict[str, Any]] = {}
    for profile, db in _session_dbs():
        conn = _ro_connect(db)
        if conn is None:
            continue
        try:
            row = conn.execute(
                "WITH user_msgs AS ("
                "  SELECT m.session_id, COUNT(*) c FROM messages m"
                "  JOIN sessions s ON s.id = m.session_id"
                "  WHERE m.role = 'user' AND s.started_at >= ? GROUP BY m.session_id"
                ") SELECT COUNT(*) sessions,"
                " COALESCE(SUM(CASE WHEN c > 1 THEN c - 1 ELSE 0 END),0) followups"
                " FROM user_msgs",
                (floor,),
            ).fetchone()
            rewinds = conn.execute(
                "SELECT COALESCE(SUM(rewind_count),0) r FROM sessions WHERE started_at >= ?",
                (floor,),
            ).fetchone()
            if row and row["sessions"]:
                out[profile] = {
                    "sessions_with_user_msgs": row["sessions"],
                    "followups": row["followups"],
                    "followups_per_session": round(row["followups"] / row["sessions"], 2),
                    "rewinds": rewinds["r"] if rewinds else 0,
                }
        except sqlite3.Error:
            pass
        finally:
            conn.close()
    return {"window_days": WINDOW_DAYS, "profiles": out}


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.get("/spend")
def get_spend():
    return _cached("spend", compute_spend)


@router.get("/usage")
def get_usage():
    return _cached("usage", compute_usage)


@router.get("/kanban")
def get_kanban():
    return _cached("kanban", compute_kanban)


@router.get("/agents")
def get_agents():
    return _cached("agents", compute_agents)


@router.get("/refinements")
def get_refinements():
    return _cached("refinements", compute_refinements)


@router.get("/overview")
def get_overview():
    """Single-fetch aggregate for the dashboard page (30s polling)."""
    return {
        "generated_at": time.time(),
        "spend": _cached("spend", compute_spend),
        "usage": _cached("usage", compute_usage),
        "kanban": _cached("kanban", compute_kanban),
        "agents": _cached("agents", compute_agents),
        "refinements": _cached("refinements", compute_refinements),
    }
