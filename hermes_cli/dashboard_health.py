"""Dashboard Mission Control API — live data layer (Hive 2).

Mount point: add to hermes_cli/web_server.py:
    from hermes_cli.dashboard_health import router as mission_router
    app.include_router(mission_router)

All endpoints require X-Hermes-Session-Token validated by existing middleware.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/dashboard", tags=["dashboard-mission"])

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
CLAUDE_PROJECTS_DIR = HOME / ".claude" / "projects"

# ---------------------------------------------------------------------------
# Pricing per million tokens (estimated API-equivalent; user is on Max plan)
# ---------------------------------------------------------------------------
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":        {"in": 3.0,  "out": 15.0,  "cr": 0.30,  "cw": 3.75},
    "claude-opus-4-7":          {"in": 15.0, "out": 75.0,  "cr": 1.50,  "cw": 18.75},
    "claude-haiku-4-5-20251001":{"in": 0.80, "out": 4.0,   "cr": 0.08,  "cw": 1.0},
    "claude-haiku-4-5":         {"in": 0.80, "out": 4.0,   "cr": 0.08,  "cw": 1.0},
}
_DEFAULT_PRICING = {"in": 3.0, "out": 15.0, "cr": 0.30, "cw": 3.75}

# Server-side cache: (value, expires_at)
_SNAPSHOT_CACHE: tuple[dict, float] | None = None
_SNAPSHOT_TTL = 30.0
_SPEND_CACHE: dict[str, tuple[dict, float]] = {}
_SPEND_TTL = 60.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Runtime health probes
# ---------------------------------------------------------------------------

def _tcp_latency(host: str, port: int, timeout: float = 1.0) -> tuple[str, Optional[float]]:
    """Return (status, latency_ms). status is 'online'|'offline'."""
    try:
        t0 = time.monotonic()
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return "online", round((time.monotonic() - t0) * 1000, 1)
    except Exception:
        return "offline", None


def _process_alive(name: str) -> tuple[str, Optional[float]]:
    """Check if a process name is running. Returns (status, approx_latency_ms)."""
    try:
        t0 = time.monotonic()
        result = subprocess.run(
            ["pgrep", "-x", name],
            capture_output=True, timeout=2
        )
        latency = round((time.monotonic() - t0) * 1000, 1)
        return ("online" if result.returncode == 0 else "offline", latency)
    except Exception:
        return ("unknown", None)


def _probe_codex() -> dict:
    status, latency = _process_alive("codex")
    # Also accept 'node' processes that may host codex
    if status == "offline":
        try:
            result = subprocess.run(
                ["pgrep", "-f", "codex"],
                capture_output=True, timeout=2
            )
            if result.returncode == 0:
                status = "online"
        except Exception:
            pass
    return {"name": "codex", "label": "Codex", "status": status,
            "latencyMs": latency, "lastChecked": _now()}


def _probe_claude_code() -> dict:
    status, latency = _process_alive("claude")
    return {"name": "claude-code", "label": "Claude Code", "status": status,
            "latencyMs": latency, "lastChecked": _now()}


def _probe_ruflo() -> dict:
    try:
        t0 = time.monotonic()
        result = subprocess.run(
            ["ruflo", "status"],
            capture_output=True, timeout=3
        )
        latency = round((time.monotonic() - t0) * 1000, 1)
        status = "online" if result.returncode == 0 else "degraded"
    except FileNotFoundError:
        status, latency = "offline", None
    except Exception:
        status, latency = "unknown", None
    return {"name": "ruflo", "label": "Ruflo", "status": status,
            "latencyMs": latency, "lastChecked": _now()}


def _probe_hermes() -> dict:
    """Probe gateway via gateway_state.json (gateway runs as a subprocess, no TCP server)."""
    try:
        t0 = time.monotonic()
        state_path = HERMES_HOME / "gateway_state.json"
        data = json.loads(state_path.read_text())
        latency = round((time.monotonic() - t0) * 1000, 1)
        gw_state = data.get("gateway_state", "unknown")
        pid = data.get("pid")
        # Verify PID is actually alive
        if pid:
            try:
                os.kill(int(pid), 0)
                status = "online" if gw_state == "running" else "degraded"
            except (ProcessLookupError, PermissionError):
                status = "offline"
        else:
            status = "offline"
        platforms = data.get("platforms", {})
        detail = ", ".join(f"{p}:{v.get('state','?')}" for p, v in platforms.items())
    except Exception:
        status, latency, detail = "unknown", None, None
    return {"name": "hermes", "label": "Hermes Subagents", "status": status,
            "latencyMs": latency, "port": 8642, "detail": detail,
            "lastChecked": _now()}


def _probe_kanban() -> dict:
    """Check kanban plugin status via dashboard HTTP (same process)."""
    try:
        import urllib.request
        t0 = time.monotonic()
        # Read session token from the running app module if available
        try:
            from hermes_cli import web_server as _ws
            token = getattr(_ws, "_SESSION_TOKEN", None)
        except Exception:
            token = None

        req = urllib.request.Request(
            "http://127.0.0.1:9119/api/plugins/kanban/status",
            headers={"X-Hermes-Session-Token": token or ""} if token else {}
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        latency = round((time.monotonic() - t0) * 1000, 1)
        active = data.get("active_tasks", data.get("queue_depth", 0))
        status = "online"
        detail = f"{active} active task{'s' if active != 1 else ''}"
    except Exception:
        # Fall back to checking if kanban DB exists and is accessible
        kanban_db = HERMES_HOME / "kanban.db"
        if kanban_db.exists():
            status, latency, detail = "online", None, "DB reachable"
        else:
            status, latency, detail = "unknown", None, None
    return {"name": "kanban", "label": "Kanban Dispatcher", "status": status,
            "latencyMs": latency, "port": 9119, "detail": detail,
            "lastChecked": _now()}


def _probe_cron() -> dict:
    """Check if cron has ≥1 enabled job."""
    try:
        t0 = time.monotonic()
        jobs_path = HERMES_HOME / "cron" / "jobs.json"
        data = json.loads(jobs_path.read_text())
        latency = round((time.monotonic() - t0) * 1000, 1)
        jobs = data.get("jobs", [])
        enabled = [j for j in jobs if j.get("enabled")]
        total = len(jobs)
        status = "online" if enabled else "degraded"
        detail = f"{len(enabled)}/{total} jobs enabled"
    except Exception:
        status, latency, detail = "unknown", None, None
    return {"name": "cron", "label": "Cron", "status": status,
            "latencyMs": latency, "detail": detail, "lastChecked": _now()}


_PROBE_MAP = {
    "codex": _probe_codex,
    "claude-code": _probe_claude_code,
    "ruflo": _probe_ruflo,
    "hermes": _probe_hermes,
    "kanban": _probe_kanban,
    "cron": _probe_cron,
}

_RUNTIME_ORDER = ["codex", "claude-code", "ruflo", "hermes", "kanban", "cron"]


def _probe_all() -> list[dict]:
    return [_PROBE_MAP[name]() for name in _RUNTIME_ORDER]


# ---------------------------------------------------------------------------
# Spend data from claude JSONL session files
# ---------------------------------------------------------------------------

def _compute_spend_points(days: int) -> list[dict]:
    """Scan ~/.claude/projects/**/*.jsonl for usage data. Returns SpendPoint list."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    daily: dict[str, dict[str, Any]] = {}  # date -> {model: {cost, tokens}}

    if not CLAUDE_PROJECTS_DIR.exists():
        return []

    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, errors="replace") as fh:
                    for raw in fh:
                        try:
                            obj = json.loads(raw)
                            ts = obj.get("timestamp", "")
                            msg = obj.get("message", {})
                            usage = msg.get("usage", {})
                            if not usage or not ts:
                                continue
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if dt < cutoff:
                                continue
                            date_str = dt.strftime("%Y-%m-%d")
                            model = msg.get("model") or "claude-sonnet-4-6"
                            pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
                            cost = (
                                usage.get("input_tokens", 0) * pricing["in"] / 1e6 +
                                usage.get("output_tokens", 0) * pricing["out"] / 1e6 +
                                usage.get("cache_read_input_tokens", 0) * pricing["cr"] / 1e6 +
                                usage.get("cache_creation_input_tokens", 0) * pricing["cw"] / 1e6
                            )
                            tokens = (
                                usage.get("input_tokens", 0) +
                                usage.get("output_tokens", 0)
                            )
                            key = (date_str, model)
                            if key not in daily:
                                daily[key] = {"cost": 0.0, "tokens": 0}
                            daily[key]["cost"] += cost
                            daily[key]["tokens"] += tokens
                        except Exception:
                            continue
            except Exception:
                continue

    # Collapse by date, taking the dominant model per day
    by_date: dict[str, dict[str, Any]] = {}
    for (date_str, model), vals in daily.items():
        if date_str not in by_date:
            by_date[date_str] = {"model": model, "cost": 0.0, "tokens": 0}
        by_date[date_str]["cost"] += vals["cost"]
        if vals["cost"] > by_date[date_str]["cost"]:
            by_date[date_str]["model"] = model
        by_date[date_str]["tokens"] += vals["tokens"]

    points = []
    for date_str in sorted(by_date.keys()):
        d = by_date[date_str]
        points.append({
            "date": date_str,
            "model": d["model"],
            "amountUsd": round(d["cost"], 4),
            "tokenCount": d["tokens"],
        })
    return points


def _get_spend(range_str: str) -> dict:
    global _SPEND_CACHE
    now = time.monotonic()
    cached = _SPEND_CACHE.get(range_str)
    if cached and now < cached[1]:
        return cached[0]

    days = {"1d": 1, "7d": 7, "30d": 30}.get(range_str, 7)
    points = _compute_spend_points(days)
    result = {"range": range_str, "points": points}
    _SPEND_CACHE[range_str] = (result, now + _SPEND_TTL)
    return result


# ---------------------------------------------------------------------------
# Session data from ~/.hermes/sessions/sessions.json
# ---------------------------------------------------------------------------

def _get_recent_sessions(limit: int = 5) -> list[dict]:
    sessions_path = HERMES_HOME / "sessions" / "sessions.json"
    try:
        raw = json.loads(sessions_path.read_text())
        if isinstance(raw, dict):
            entries = list(raw.values())
        elif isinstance(raw, list):
            entries = raw
        else:
            return []

        # Sort by updated_at descending
        entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
        result = []
        for e in entries[:limit]:
            preview = e.get("display_name") or e.get("platform") or "Session"
            preview = str(preview)[:80]
            result.append({
                "id": e.get("session_id", "unknown"),
                "preview": preview,
                "createdAt": e.get("created_at", _now()),
                "modelUsed": e.get("model"),
            })
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Spend summary (today / week)
# ---------------------------------------------------------------------------

def _get_spend_summary() -> tuple[float, float]:
    spend_7d = _get_spend("7d")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    today_total = 0.0
    week_total = 0.0
    for p in spend_7d["points"]:
        week_total += p["amountUsd"]
        if p["date"] == today_str:
            today_total += p["amountUsd"]
    return round(today_total, 4), round(week_total, 4)


# ---------------------------------------------------------------------------
# Streak calculation
# ---------------------------------------------------------------------------

def _compute_streak() -> int:
    """Count consecutive days with claude session activity, ending today."""
    try:
        proj_dir = CLAUDE_PROJECTS_DIR
        if not proj_dir.exists():
            return 0
        active_dates: set[str] = set()
        for project_dir in proj_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                try:
                    mtime = datetime.fromtimestamp(
                        jsonl_file.stat().st_mtime, tz=timezone.utc
                    )
                    active_dates.add(mtime.strftime("%Y-%m-%d"))
                except Exception:
                    continue

        if not active_dates:
            return 0

        streak = 0
        check_date = datetime.now(timezone.utc).date()
        while check_date.isoformat() in active_dates:
            streak += 1
            check_date -= timedelta(days=1)
        return streak
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Active model
# ---------------------------------------------------------------------------

def _get_active_model() -> str:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        model_cfg = cfg.get("model", "")
        if isinstance(model_cfg, dict):
            return (model_cfg.get("default") or model_cfg.get("name") or "claude-sonnet-4-6")
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
    except Exception:
        pass
    return "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Cron data
# ---------------------------------------------------------------------------

def _get_next_cron() -> Optional[dict]:
    try:
        jobs_path = HERMES_HOME / "cron" / "jobs.json"
        data = json.loads(jobs_path.read_text())
        jobs = data.get("jobs", [])
        now_str = datetime.now(timezone.utc).isoformat()
        enabled_with_next = [
            j for j in jobs
            if j.get("enabled") and j.get("next_run_at")
        ]
        if not enabled_with_next:
            # Fall back to staged dream-reflect.cron
            staged_path = HERMES_HOME / "cron.d" / "dream-reflect.cron"
            if staged_path.exists():
                content = staged_path.read_text()
                for line in content.splitlines():
                    if line.strip() and not line.strip().startswith("#"):
                        parts = line.split()
                        if len(parts) >= 6:
                            expr = " ".join(parts[:5])
                            return {
                                "name": "dream-reflect (staged)",
                                "schedule": expr,
                                "nextRun": now_str,
                            }
            return None
        # Pick soonest
        soonest = min(enabled_with_next, key=lambda j: j["next_run_at"])
        return {
            "name": soonest.get("name", "unnamed"),
            "schedule": soonest.get("schedule_display", soonest.get("schedule", {}).get("expr", "")),
            "nextRun": soonest["next_run_at"],
        }
    except Exception:
        return None


def _get_all_cron_jobs() -> list[dict]:
    try:
        jobs_path = HERMES_HOME / "cron" / "jobs.json"
        data = json.loads(jobs_path.read_text())
        jobs = data.get("jobs", [])
        result = []
        for j in jobs:
            result.append({
                "id": j.get("id"),
                "name": j.get("name"),
                "schedule": j.get("schedule_display") or j.get("schedule", {}).get("expr", ""),
                "enabled": j.get("enabled", False),
                "state": j.get("state", "unknown"),
                "nextRun": j.get("next_run_at"),
                "lastRun": j.get("last_run_at"),
                "lastStatus": j.get("last_status"),
                "staged": False,
            })

        # Check for staged dream-reflect
        staged_path = HERMES_HOME / "cron.d" / "dream-reflect.cron"
        if staged_path.exists():
            content = staged_path.read_text()
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") and "dream-reflect" in stripped.lower():
                    # Extract the commented cron expression
                    inner = stripped.lstrip("#").strip()
                    parts = inner.split(None, 5)
                    if len(parts) >= 6:
                        result.append({
                            "id": "dream-reflect-staged",
                            "name": "dream-reflect (staged — click to activate)",
                            "schedule": " ".join(parts[:5]),
                            "enabled": False,
                            "state": "staged",
                            "nextRun": None,
                            "lastRun": None,
                            "lastStatus": None,
                            "staged": True,
                            "activateCmd": f"crontab -l | sed 's/^#\\s*{parts[:5][0]}//' | crontab -",
                        })
                    break
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Dream data
# ---------------------------------------------------------------------------

def _get_last_dream() -> Optional[str]:
    dreams_dir = HERMES_HOME / "dreams"
    if not dreams_dir.exists():
        return None
    try:
        dream_files = sorted(dreams_dir.glob("*.md"), reverse=True)
        if not dream_files:
            return None
        content = dream_files[0].read_text(errors="replace")
        # Return first ~300 chars as the brief
        lines = [ln for ln in content.splitlines() if ln.strip() and not ln.startswith("#")]
        brief = " ".join(lines)[:300].strip()
        return brief or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Swarm status
# ---------------------------------------------------------------------------

def _get_swarm_status() -> Optional[dict]:
    """Best-effort swarm status from ruflo."""
    try:
        result = subprocess.run(
            ["ruflo", "swarm", "status", "--json"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return data
    except Exception:
        pass

    # Fall back to hive process scan — look for hive-* working dirs
    try:
        hive_dirs = list((HERMES_HOME / "ruflo-work").glob("*hive*")) if (HERMES_HOME / "ruflo-work").exists() else []
        active = [d for d in hive_dirs if d.is_dir()]
        if active:
            return {
                "id": "hive-local",
                "name": "Hive Mind Swarm",
                "topology": "hierarchical-mesh",
                "workerCount": len(active),
                "activeWorkers": 0,
                "queueDepth": 0,
                "lastActivity": _now(),
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Snapshot cache
# ---------------------------------------------------------------------------

def _build_snapshot() -> dict:
    runtimes = _probe_all()
    today_spend, week_spend = _get_spend_summary()
    return {
        "model": _get_active_model(),
        "spendToday": today_spend,
        "spendWeek": week_spend,
        "streakDays": _compute_streak(),
        "runtimes": runtimes,
        "swarm": _get_swarm_status(),
        "recentSessions": _get_recent_sessions(),
        "lastDream": _get_last_dream(),
        "nextCron": _get_next_cron(),
    }


def _get_snapshot() -> dict:
    global _SNAPSHOT_CACHE
    now = time.monotonic()
    if _SNAPSHOT_CACHE and now < _SNAPSHOT_CACHE[1]:
        return _SNAPSHOT_CACHE[0]
    snapshot = _build_snapshot()
    _SNAPSHOT_CACHE = (snapshot, now + _SNAPSHOT_TTL)
    return snapshot


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/mission", summary="Mission Control snapshot")
async def get_mission_snapshot() -> dict:
    """Combined MissionSnapshot with real live data. 30 s server-side cache."""
    return await asyncio.get_event_loop().run_in_executor(None, _get_snapshot)


@router.get("/health/runtime/{name}", summary="Single runtime health probe")
async def get_runtime_health(name: str) -> dict:
    """Live probe for one named runtime."""
    probe = _PROBE_MAP.get(name)
    if probe is None:
        return {"name": name, "label": name, "status": "unknown", "lastChecked": _now()}
    return await asyncio.get_event_loop().run_in_executor(None, probe)


@router.get("/spend", summary="Spend history sparkline")
async def get_spend(range: str = "7d") -> dict:
    """Spend history from claude session JSONL files. range=1d|7d|30d."""
    if range not in ("1d", "7d", "30d"):
        range = "7d"
    return await asyncio.get_event_loop().run_in_executor(None, _get_spend, range)


def _get_queue_depth(range_str: str) -> dict:
    """Daily task creation counts across all hermes kanban boards.
    Proxy for 'workload'; each task represents a unit of dispatch.
    """
    import glob, sqlite3
    days = {"1d": 1, "7d": 7, "30d": 30}.get(range_str, 7)
    aggregated: dict[str, int] = {}
    open_total = 0
    for path in glob.glob(str(HERMES_HOME / "kanban/boards/*/kanban.db")):
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            c = conn.cursor()
            c.execute(
                "SELECT date(created_at,'unixepoch') d, count(*) "
                "FROM tasks "
                "WHERE created_at >= strftime('%s','now',?) "
                "GROUP BY d",
                (f"-{days} days",),
            )
            for d, n in c.fetchall():
                aggregated[d] = aggregated.get(d, 0) + n
            c.execute(
                "SELECT count(*) FROM tasks "
                "WHERE status NOT IN ('done','archived','complete')"
            )
            open_total += c.fetchone()[0]
            conn.close()
        except Exception:
            continue
    points = [{"date": d, "count": n} for d, n in sorted(aggregated.items())]
    return {"range": range_str, "points": points, "openNow": open_total}


@router.get("/queue", summary="Queue depth — kanban task creation per day")
async def get_queue(range: str = "7d") -> dict:
    """Daily task creation across all kanban boards + current open count."""
    if range not in ("1d", "7d", "30d"):
        range = "7d"
    return await asyncio.get_event_loop().run_in_executor(None, _get_queue_depth, range)


@router.get("/swarm", summary="Swarm status")
async def get_swarm() -> dict:
    """Ruflo + Hermes subagents + Kanban dispatcher swarm status."""
    swarm = await asyncio.get_event_loop().run_in_executor(None, _get_swarm_status)
    if swarm is None:
        return {"active": False, "message": "No active swarm detected"}
    return {"active": True, **swarm}


@router.get("/cron", summary="Cron jobs — next firings and last runs")
async def get_cron() -> dict:
    """All cron jobs with next firing times and last run status."""
    jobs = await asyncio.get_event_loop().run_in_executor(None, _get_all_cron_jobs)
    return {"jobs": jobs, "count": len(jobs)}


@router.get("/dreams/latest", summary="Last overnight reflection brief")
async def get_latest_dream() -> dict:
    """Return the most recent dream-reflect brief from ~/.hermes/dreams/."""
    dreams_dir = HERMES_HOME / "dreams"
    if not dreams_dir.exists():
        return {"dream": None, "date": None, "message": "No dreams directory yet"}

    dream_files = sorted(dreams_dir.glob("*.md"), reverse=True)
    if not dream_files:
        return {"dream": None, "date": None, "message": "No dream files yet"}

    try:
        latest = dream_files[0]
        content = latest.read_text(errors="replace")
        date_str = latest.stem  # filename is YYYY-MM-DD.md
        return {"dream": content, "date": date_str, "filename": latest.name}
    except Exception as e:
        return {"dream": None, "date": None, "error": str(e)}


@router.get("/stream", summary="SSE runtime health stream (1 Hz chip updates)")
async def stream_health() -> StreamingResponse:
    """Server-Sent Events stream. One SSEHealthEvent per runtime per 10 s cycle.
    Heartbeat every 15 s to keep proxies alive. id: field for Last-Event-ID resume.
    """
    async def _generate():
        event_id = 0
        loop = asyncio.get_event_loop()
        last_heartbeat = time.monotonic()

        while True:
            # Probe all runtimes in executor to avoid blocking the event loop
            chips = await loop.run_in_executor(None, _probe_all)
            for chip in chips:
                event = {**chip, "eventType": "health"}
                yield f"id: {event_id}\ndata: {json.dumps(event)}\n\n"
                event_id += 1
                await asyncio.sleep(0.1)  # small gap between chips

            # Heartbeat: emit a comment line every 15 s
            now = time.monotonic()
            if now - last_heartbeat >= 15:
                yield ": heartbeat\n\n"
                last_heartbeat = now

            # Wait for the next cycle (10 s total, accounting for probe time)
            await asyncio.sleep(10)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
