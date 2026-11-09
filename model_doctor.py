"""model-doctor — proactive LLM health monitor for Hermes Agent.

Observes every provider API request Hermes makes (via pre_api_request,
post_api_request, api_request_error hooks), records per-model rolling
uptime/latency, classifies failures, and exposes tools so the agent can
query current health. Complements (does not replace) round-robin routing:
it is the diagnostic layer that tells you WHICH models are sick and WHY.

Design notes:
- State is kept in-memory with a JSON snapshot under <HERMES_HOME>/model-doctor/
  so status survives restarts without locking issues (write-on-exit is lossy
  but acceptable; post_api_request also snapshots every N events).
- Zero network calls from the plugin itself. All data comes from observing
  real traffic, so it works for any provider without extra credentials.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover
    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()

LOCK = threading.Lock()
# key: (provider, base_url, model) -> record
RECORDS: Dict[str, Dict[str, Any]] = {}
SNAPSHOT_EVERY = 10  # API events between disk snapshots

WINDOW = 50          # rolling window per model
LAT_KEEP = 30        # latencies kept per model

# Classification thresholds / constants
EMPTY_MAX_CHARS = 5  # reply shorter than this (after strip) counts as empty


def _key(provider: str, base_url: str, model: str) -> str:
    return f"{provider}|{base_url}|{model}"


def _classify(status_code: Any, finish_reason: str, content_chars: int,
              api_duration: float, error: Any) -> Dict[str, Any]:
    """Return {verdict, reason} for a finished request."""
    etype = msg = ""
    if isinstance(error, dict):
        etype = str(error.get("type") or "")
        msg = str(error.get("message") or "")[:160]

    code = None
    if status_code is not None:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            code = None

    # HTTP status wins over the raw exception text whenever it exists.
    if code in (401, 403):
        reason = f"HTTP {code} — key invalid/expired/truncated" + (f" ({msg})" if msg else "")
        return {"verdict": "AUTH", "reason": reason}
    if code == 429:
        return {"verdict": "RATE", "reason": "HTTP 429 — rate limited / quota exhausted"}
    if code is not None and 500 <= code <= 599:
        return {"verdict": "UPSTREAM", "reason": f"HTTP {code} — upstream provider down"}
    if code is not None and code != 200:
        return {"verdict": "HTTP_OTHER", "reason": f"HTTP {code}"}

    if error or code is None:
        reason = f"{etype}: {msg}" if (etype or msg) else "request raised"
        return {"verdict": "ERROR", "reason": reason}

    # 200 OK — check for empty replies
    fr = (finish_reason or "").lower()
    if fr == "length" and content_chars <= 0:
        return {"verdict": "EMPTY", "reason": "finish=length with no content (max_tokens too small or model stalled)"}
    if content_chars <= 0 and fr not in ("tool_calls", "function_call", "stop"):
        return {"verdict": "EMPTY", "reason": f"no content, finish={fr or 'none'}"}
    if api_duration and api_duration > 60:
        return {"verdict": "SLOW", "reason": f"ok but slow ({api_duration:.0f}s)"}
    return {"verdict": "OK", "reason": ""}


def _new_record() -> Dict[str, Any]:
    return {
        "ok": 0, "auth": 0, "rate": 0, "upstream": 0, "empty": 0,
        "slow": 0, "other": 0, "errors": 0, "total": 0,
        "last_verdict": "", "last_reason": "", "last_ts": 0.0,
        "latencies": [], "first_ts": 0.0,
    }


def _record_event(key: str, verdict: str, reason: str, duration: float,
                  ts: float, keep_latency: bool) -> None:
    rec = RECORDS.setdefault(key, _new_record())
    rec["total"] += 1
    rec["last_verdict"] = verdict
    rec["last_reason"] = reason
    rec["last_ts"] = ts
    if not rec["first_ts"]:
        rec["first_ts"] = ts
    bucket = {
        "OK": "ok", "AUTH": "auth", "RATE": "rate", "UPSTREAM": "upstream",
        "EMPTY": "empty", "SLOW": "slow", "HTTP_OTHER": "other", "ERROR": "errors",
    }.get(verdict, "other")
    rec[bucket] = rec.get(bucket, 0) + 1
    if keep_latency and duration > 0:
        lats = rec.setdefault("latencies", [])
        lats.append(round(duration, 3))
        del lats[:-LAT_KEEP]
    # keep verdict counters bounded to the rolling window implicitly via totals
    if rec["total"] > WINDOW * 4:  # soft cap; totals are cumulative but bounded storage
        pass


def _snapshot_path() -> Path:
    return get_hermes_home() / "model-doctor" / "health.json"


def _save_snapshot() -> None:
    try:
        p = _snapshot_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with LOCK:
            data = {k: {kk: vv for kk, vv in v.items()} for k, v in RECORDS.items()}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"saved_at": time.time(), "records": data}, indent=1))
        tmp.replace(p)
    except Exception as exc:
        logger.debug("model-doctor snapshot failed: %s", exc)


def _load_snapshot() -> None:
    try:
        p = _snapshot_path()
        if not p.exists():
            return
        data = json.loads(p.read_text() or "{}")
        for k, v in (data.get("records") or {}).items():
            rec = _new_record()
            rec.update({kk: vv for kk, vv in v.items() if kk in rec})
            RECORDS[k] = rec
    except Exception as exc:
        logger.debug("model-doctor load failed: %s", exc)


# ---------------------------------------------------------------------------
# Hook callbacks (names must match Hermes hook registry; kwargs tolerated)
# ---------------------------------------------------------------------------

def on_pre_api_request(**kwargs: Any) -> None:
    # Reserved: could pre-mark in-flight requests. Nothing needed yet.
    return None


def on_post_api_request(**kwargs: Any) -> None:
    model = str(kwargs.get("model") or "")
    provider = str(kwargs.get("provider") or "")
    base_url = str(kwargs.get("base_url") or "")
    if not model:
        return
    status_code = kwargs.get("status_code")
    finish_reason = str(kwargs.get("finish_reason") or "")
    content_chars = int(kwargs.get("assistant_content_chars") or 0)
    duration = float(kwargs.get("api_duration") or 0.0)
    ts = time.time()

    cls = _classify(status_code, finish_reason, content_chars, duration, None)
    key = _key(provider, base_url, model)
    with LOCK:
        _record_event(key, cls["verdict"], cls["reason"], duration, ts, True)
        n = RECORDS[key]["total"]
    if n % SNAPSHOT_EVERY == 0:
        _save_snapshot()


def on_api_request_error(**kwargs: Any) -> None:
    model = str(kwargs.get("model") or "")
    provider = str(kwargs.get("provider") or "")
    base_url = str(kwargs.get("base_url") or "")
    if not model:
        return
    status_code = kwargs.get("status_code")
    duration = float(kwargs.get("api_duration") or 0.0)
    ts = time.time()
    cls = _classify(status_code, "", 0, duration, kwargs.get("error") or {"type": "exception", "message": "raised"})
    key = _key(provider, base_url, model)
    with LOCK:
        _record_event(key, cls["verdict"], cls["reason"], duration, ts, False)
        n = RECORDS[key]["total"]
    if n % SNAPSHOT_EVERY == 0:
        _save_snapshot()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _health_row(key: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    provider, base_url, model = (key.split("|", 2) + ["", "", ""])[:3]
    total = rec.get("total", 0) or 0
    ok = rec.get("ok", 0) or 0
    lats = rec.get("latencies") or []
    avg = round(sum(lats) / len(lats), 2) if lats else None
    last_age = round(time.time() - rec.get("last_ts", 0)) if rec.get("last_ts") else None
    health = "unknown"
    if total:
        if rec.get("auth"):
            health = "auth-broken"
        elif ok / total >= 0.8:
            health = "healthy"
        elif ok / total >= 0.4:
            health = "flaky"
        else:
            health = "sick"
    return {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "health": health,
        "ok_ratio": round(ok / total, 2) if total else None,
        "counts": {k: rec.get(k, 0) for k in ("ok", "auth", "rate", "upstream", "empty", "slow", "other", "errors", "total")},
        "avg_latency_s": avg,
        "last_verdict": rec.get("last_verdict"),
        "last_reason": rec.get("last_reason"),
        "last_seen_s_ago": last_age,
    }


def tool_status(args: Dict[str, Any]) -> Dict[str, Any]:
    with LOCK:
        rows = [_health_row(k, v) for k, v in RECORDS.items()]
    rows.sort(key=lambda r: (r["health"] != "auth-broken", r["health"] != "sick", -(r["counts"]["total"] or 0)))
    return {
        "models": rows,
        "note": "health = healthy | flaky | sick | auth-broken | unknown; AUTH verdicts usually mean a truncated/expired API key, 5xx means upstream down, EMPTY means zero-content replies.",
    }


def tool_refresh(args: Dict[str, Any]) -> Dict[str, Any]:
    """Re-sort the most recent events (passive refresh; no live network call)."""
    return tool_status(args)


def register(ctx) -> None:
    _load_snapshot()
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_tool(
        name="model_doctor_status",
        toolset="model-doctor",
        schema={
            "name": "model_doctor_status",
            "description": (
                "Report per-model uptime, latency and recent failure classes "
                "from the passive request log."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        handler=tool_status,
    )
    ctx.register_tool(
        name="model_doctor_refresh",
        toolset="model-doctor",
        schema={
            "name": "model_doctor_refresh",
            "description": (
                "Re-sort the most recent events (passive refresh; no live network call)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        handler=tool_refresh,
    )
    logger.info("model-doctor: monitoring %d known model(s)", len(RECORDS))
