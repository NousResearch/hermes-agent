"""Read-only metric collection for the Context Cockpit.

Never writes. Never calls an LLM. Never opens secrets/env dumps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .sqlite_ro import open_readonly
from .live_lcm import read_live_lcm_snapshot

# Window-fill zones (absolute % of model window).
QUALITY_WATCH_PCT = 33
HOST_COMPRESS_PCT = 50
HYGIENE_PCT = 80
LCM_SOON_RATIO = 0.70
LCM_ACT_RATIO = 0.90

# Freshness: Desktop personal-ops often has a stale gateway_state.json because
# the Telegram gateway lives on telegram-ops. Prefer live process + state.db age.
DEFAULT_STALE_SEC = 120.0  # session DB quiet for 2m → soft stale (no live process)
DEFAULT_IDLE_SEC = 600.0  # Desktop running but no chat activity for 10m → soft idle
DEFAULT_OFFLINE_SEC = 900.0  # 15m with no live Hermes process → offline

FALLBACK_WINDOWS = {
    "deepseek/deepseek-v4-flash": 1_000_000,
    "deepseek/deepseek-v4-pro": 1_000_000,
    "z-ai/glm-5.2": 1_000_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-mini": 400_000,
    "anthropic/claude-sonnet-4": 200_000,
}


def _chars_to_tokens(text: str) -> int:
    return max(0, len(text or "") // 4)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return data or {}
    except Exception:
        return {}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception:
        return None


def _safe_float(row: Any, key: str) -> Optional[float]:
    if row is None:
        return None
    try:
        val = row[key]
    except Exception:
        return None
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def _safe_str(row: Any, key: str) -> Optional[str]:
    if row is None:
        return None
    try:
        val = row[key]
    except Exception:
        return None
    if val is None:
        return None
    return str(val)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _file_age_sec(path: Path, now: Optional[float] = None) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return max(0.0, (now or time.time()) - path.stat().st_mtime)
    except Exception:
        return None


def resolve_window(model: str, profile_dir: Path) -> int:
    cache = _load_yaml(profile_dir / "context_length_cache.yaml")
    lengths = (cache or {}).get("context_lengths", {}) if isinstance(cache, dict) else {}
    if isinstance(lengths, dict):
        for key, val in lengths.items():
            if key.startswith(model + "@") or key == model:
                try:
                    return int(val)
                except Exception:
                    pass
    if model in FALLBACK_WINDOWS:
        return FALLBACK_WINDOWS[model]
    slug = model.split("/")[-1] if "/" in model else model
    for k, v in FALLBACK_WINDOWS.items():
        if k.split("/")[-1] == slug:
            return v
    return 1_000_000


def hermes_liveness(profile_dir: Path, now: Optional[float] = None) -> Dict[str, Any]:
    """Best-effort live detection for Desktop + gateway.

    Desktop sessions often leave gateway_state.json stale (gateway may live on
    another profile). Prefer:
      1. Live Hermes / tui_gateway / Desktop process for this profile
      2. Fresh state.db mtime as session activity heartbeat
      3. gateway_state.json only when its PID is alive
    """
    now = now or time.time()
    out: Dict[str, Any] = {
        "running": False,
        "pid": None,
        "source": "none",
        "command": "",
        "gateway_state": None,
        "gateway_age_sec": _file_age_sec(profile_dir / "gateway_state.json", now),
        "processes_age_sec": _file_age_sec(profile_dir / "processes.json", now),
        "state_db_age_sec": _file_age_sec(profile_dir / "state.db", now),
        "heartbeat_age_sec": None,
        "heartbeat_source": None,
    }

    # processes.json (list of background jobs) — rarely populated for Desktop.
    procs = _load_json(profile_dir / "processes.json")
    if isinstance(procs, list):
        for entry in procs:
            if not isinstance(entry, dict):
                continue
            cmd = str(entry.get("command") or "")
            pid = entry.get("pid") or 0
            if "hermes" in cmd.lower() and _pid_alive(pid):
                out.update(
                    {
                        "running": True,
                        "pid": pid,
                        "source": "processes.json",
                        "command": cmd,
                        "heartbeat_age_sec": out["processes_age_sec"],
                        "heartbeat_source": "processes.json",
                    }
                )
                return out

    # Live Desktop / tui_gateway process scan (read-only /proc).
    try:
        import subprocess

        ps = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        profile_token = str(profile_dir)
        for line in (ps.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid_s, args = parts[0], parts[1]
            low = args.lower()
            if "context_visor" in low or "context_cockpit" in low:
                continue
            is_desktop = "/hermes" in low and "linux-unpacked/hermes" in low
            is_tui = "tui_gateway" in low
            is_cli = "hermes-agent" in low and " hermes " in f" {low} "
            if not (is_desktop or is_tui or is_cli):
                continue
            # Prefer processes clearly tied to this profile; Desktop is global.
            if is_tui and profile_token not in args and f"--profile {profile_dir.name}" not in args:
                # slash_worker may not embed profile path; still count as live.
                pass
            try:
                pid = int(pid_s)
            except Exception:
                continue
            if _pid_alive(pid):
                out.update(
                    {
                        "running": True,
                        "pid": pid,
                        "source": "process_scan",
                        "command": args[:200],
                        "heartbeat_age_sec": out["state_db_age_sec"],
                        "heartbeat_source": "process_scan+state.db",
                    }
                )
                return out
    except Exception:
        pass

    # gateway_state.json — only if PID alive (personal-ops gateway often disabled).
    gs = _load_json(profile_dir / "gateway_state.json") or {}
    gpid = gs.get("pid")
    gstate = gs.get("gateway_state")
    out["gateway_state"] = gstate
    if gpid and _pid_alive(gpid):
        age = out["gateway_age_sec"]
        out.update(
            {
                "running": bool(gstate == "running"),
                "pid": gpid,
                "source": "gateway_state.json",
                "command": " ".join(str(x) for x in (gs.get("argv") or [])[:6]),
                "heartbeat_age_sec": age,
                "heartbeat_source": "gateway_state.json",
            }
        )
        return out

    # No live process — use state.db age as soft heartbeat for "recently used".
    out["heartbeat_age_sec"] = out["state_db_age_sec"]
    out["heartbeat_source"] = "state.db"
    return out


def current_session(state_db: Path) -> Optional[Dict[str, Any]]:
    if not state_db.exists():
        return None
    try:
        conn = open_readonly(state_db)
    except Exception:
        return None
    try:
        row = conn.execute(
            """SELECT id, model, source, started_at, ended_at, message_count,
                      input_tokens, output_tokens, cache_read_tokens,
                      cache_write_tokens, reasoning_tokens, api_call_count,
                      system_prompt,
                      billing_provider, billing_mode, estimated_cost_usd,
                      actual_cost_usd, cost_status, cost_source, pricing_version
               FROM sessions
               ORDER BY started_at DESC LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}
    except Exception:
        return None
    finally:
        conn.close()


def lcm_telemetry(profile_dir: Path, lcm_db: Path, conversation_id: Optional[str]) -> Dict[str, Any]:
    """Read LCM compaction telemetry from metadata + lifecycle debt."""
    out: Dict[str, Any] = {"loaded": False}
    if not lcm_db.exists():
        return out
    out["loaded"] = True
    if not conversation_id:
        return out
    try:
        conn = open_readonly(lcm_db)
    except Exception:
        return out
    try:
        key = f"compaction_telemetry:{conversation_id}"
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row:
            try:
                payload = json.loads(row["value"])
                if isinstance(payload, dict):
                    out.update(payload)
            except Exception:
                pass
        lc = conn.execute(
            "SELECT debt_kind, debt_size_estimate, updated_at "
            "FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if lc:
            out["debt_kind"] = lc["debt_kind"]
            out["debt_size_estimate"] = lc["debt_size_estimate"]
        from .live_lcm import snapshot_matches_conversation

        snapshot = read_live_lcm_snapshot(profile_dir)
        if snapshot and snapshot_matches_conversation(snapshot, conversation_id):
            now = time.time()
            out["live_snapshot_loaded"] = True
            out["live_snapshot_age_sec"] = max(
                0.0, now - float(snapshot.get("collected_at") or now)
            )
            out["last_compression_status"] = snapshot.get("last_compression_status")
            out["last_compression_noop_reason"] = snapshot.get(
                "last_compression_noop_reason"
            )
            # Prefer rotate-preview fresh-tail when present; else engine config.
            rotate_preview = snapshot.get("rotate_preview") or {}
            fresh_tail = rotate_preview.get("fresh_tail_count")
            if fresh_tail is None:
                fresh_tail = snapshot.get("fresh_tail_count")
            out["fresh_tail_count"] = fresh_tail
            out["leaf_chunk_tokens"] = snapshot.get("leaf_chunk_tokens")
            out["rotate_preview_reason"] = rotate_preview.get("reason")
            out["total_message_count"] = rotate_preview.get("total_message_count")
            out["pre_tail_message_count"] = rotate_preview.get(
                "pre_tail_message_count"
            )
        return out
    except Exception:
        return out
    finally:
        conn.close()


def conversation_mass(state_db: Path, session_id: str) -> Dict[str, int]:
    """Sum token_count for active, non-compacted messages."""
    out = {"tokens": 0, "messages": 0, "chars": 0}
    if not state_db.exists() or not session_id:
        return out
    try:
        conn = open_readonly(state_db)
    except Exception:
        return out
    try:
        row = conn.execute(
            """SELECT COALESCE(SUM(COALESCE(token_count,0)),0) AS t,
                      COUNT(*) AS n,
                      COALESCE(SUM(LENGTH(COALESCE(content,''))),0) AS c
               FROM messages
               WHERE session_id = ? AND active = 1 AND compacted = 0""",
            (session_id,),
        ).fetchone()
        if row:
            out["tokens"] = int(row["t"] or 0)
            out["messages"] = int(row["n"] or 0)
            out["chars"] = int(row["c"] or 0)
        return out
    except Exception:
        return out
    finally:
        conn.close()


def skills_block_tokens(profile_dir: Path) -> int:
    snap = _load_json(profile_dir / ".skills_prompt_snapshot.json")
    if isinstance(snap, dict):
        text = snap.get("prompt") or snap.get("skills_prompt") or ""
        if text:
            return _chars_to_tokens(str(text))
        # Some snapshots store size hints.
        for key in ("token_estimate", "tokens", "approx_tokens"):
            if key in snap:
                try:
                    return int(snap[key])
                except Exception:
                    pass
    return 0


def system_prompt_tokens(system_prompt: Optional[str]) -> int:
    return _chars_to_tokens(system_prompt or "")


def _read_profile_env_value(profile_dir: Path, key: str) -> Optional[str]:
    env_path = profile_dir / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def lcm_context_threshold_ratio(profile_dir: Path) -> tuple[float, bool]:
    """Return (ratio, explicit). Default 0.50 if unset."""
    env_v = _read_profile_env_value(profile_dir, "LCM_CONTEXT_THRESHOLD")
    if env_v:
        try:
            return float(env_v), True
        except Exception:
            pass
    cfg = _load_yaml(profile_dir / "config.yaml")
    lcm = cfg.get("lcm") if isinstance(cfg, dict) else None
    if isinstance(lcm, dict) and "context_threshold" in lcm:
        try:
            return float(lcm["context_threshold"]), True
        except Exception:
            pass
    return 0.50, False


def threshold_tokens(model: str, window: int, profile_dir: Path) -> int:
    ratio, _ = lcm_context_threshold_ratio(profile_dir)
    return int(window * ratio)


def track_burn_rate(
    state: Dict[str, Any],
    *,
    real_last: int,
    api_calls: int,
    est_usd: Optional[float],
) -> Dict[str, Any]:
    """Delta burn across visor polls (in-memory only)."""
    now = time.time()
    prev = state.get("burn_prev") or {}
    out: Dict[str, Any] = {
        "tok_per_call": None,
        "usd_per_call_recent": None,
        "tok_per_min": None,
    }
    if prev:
        dt = max(0.001, now - float(prev.get("t", now)))
        d_tok = real_last - int(prev.get("real_last", real_last))
        d_calls = api_calls - int(prev.get("api_calls", api_calls))
        d_usd = None
        if est_usd is not None and prev.get("est_usd") is not None:
            d_usd = est_usd - float(prev["est_usd"])
        if d_calls > 0 and d_tok > 0:
            out["tok_per_call"] = d_tok / d_calls
        if d_calls > 0 and d_usd is not None and d_usd > 0:
            out["usd_per_call_recent"] = d_usd / d_calls
        if d_tok > 0:
            out["tok_per_min"] = d_tok / (dt / 60.0)
    state["burn_prev"] = {
        "t": now,
        "real_last": real_last,
        "api_calls": api_calls,
        "est_usd": est_usd,
    }
    return out


def track_model_change(state: Dict[str, Any], model: str) -> Optional[str]:
    prev = state.get("last_model")
    state["last_model"] = model
    if prev and prev != model and prev != "unknown":
        msg = f"Model changed: {prev} → {model}"
        state["alert"] = msg
        state["alert_until"] = time.time() + 300
        return msg
    until = float(state.get("alert_until") or 0)
    if until > time.time():
        return state.get("alert")
    state.pop("alert", None)
    return None


def collect_metrics(
    profile: str,
    profile_dir: Path,
    state: Optional[Dict[str, Any]] = None,
    *,
    now: Optional[float] = None,
    stale_sec: float = DEFAULT_STALE_SEC,
    offline_sec: float = DEFAULT_OFFLINE_SEC,
) -> Dict[str, Any]:
    """Collect a JSON-serializable metrics snapshot."""
    state = state if state is not None else {}
    now = now or time.time()
    live = hermes_liveness(profile_dir, now=now)
    sess = current_session(profile_dir / "state.db")
    model = (sess or {}).get("model") or "unknown"
    alert_msg = track_model_change(state, model) if model != "unknown" else state.get("alert")
    window = resolve_window(model, profile_dir)
    conv_id = (sess or {}).get("id")
    tele = lcm_telemetry(profile_dir, profile_dir / "lcm.db", conv_id)
    conv = (
        conversation_mass(profile_dir / "state.db", conv_id)
        if conv_id
        else {"tokens": 0, "messages": 0, "chars": 0}
    )

    sp_tokens = system_prompt_tokens((sess or {}).get("system_prompt"))
    skills_tokens = skills_block_tokens(profile_dir)
    tools_est = 6_000
    scaffold_tokens = sp_tokens + skills_tokens + tools_est

    real_last = tele.get("last_observed_prompt_tokens")
    if real_last is None or int(real_last or 0) <= 0:
        real_last = scaffold_tokens + conv["tokens"]
    real_last = int(real_last or 0)

    pct = (real_last / window * 100.0) if window else 0.0
    thresh_tok = threshold_tokens(model, window, profile_dir)
    thresh_ratio, thresh_explicit = lcm_context_threshold_ratio(profile_dir)
    thresh_pct = (thresh_tok / window * 100.0) if window else 0.0
    compressions = int(tele.get("total_compactions", 0) or 0)

    conv_est = max(0, real_last - scaffold_tokens)
    conv_share = (conv_est / real_last * 100.0) if real_last else 0.0

    api_calls = int((sess or {}).get("api_call_count") or 0)
    est_usd = _safe_float(sess, "estimated_cost_usd") if sess else None
    act_usd = _safe_float(sess, "actual_cost_usd") if sess else None
    burn = track_burn_rate(
        state, real_last=real_last, api_calls=api_calls, est_usd=est_usd
    )

    # Activity age must include live LCM / last API signals. During a long Desktop
    # turn, state.db mtime can lag for minutes while the chat is clearly active.
    activity_candidates: list[tuple[float, str]] = []
    hb_age = live.get("heartbeat_age_sec")
    if hb_age is not None:
        activity_candidates.append((float(hb_age), str(live.get("heartbeat_source") or "heartbeat")))
    snap_age = tele.get("live_snapshot_age_sec")
    if snap_age is not None:
        activity_candidates.append((float(snap_age), "live-lcm-snapshot"))
    last_api = tele.get("last_api_call_at")
    if last_api is not None:
        try:
            activity_candidates.append((max(0.0, now - float(last_api)), "last-api-call"))
        except Exception:
            pass
    if activity_candidates:
        activity_age_sec, activity_source = min(activity_candidates, key=lambda item: item[0])
    else:
        activity_age_sec, activity_source = None, None
    if activity_age_sec is not None:
        live["activity_age_sec"] = activity_age_sec
        live["activity_source"] = activity_source
        # Prefer the freshest activity signal for operator heartbeat display.
        if hb_age is None or activity_age_sec < float(hb_age):
            live["heartbeat_age_sec"] = activity_age_sec
            live["heartbeat_source"] = activity_source

    freshness = "fresh"
    if not live["running"]:
        age = activity_age_sec if activity_age_sec is not None else hb_age
        if age is None or age > offline_sec:
            freshness = "offline"
        elif age > stale_sec:
            freshness = "stale"
        else:
            freshness = "quiet"  # recent activity, no live process seen
    else:
        # Live Hermes/Desktop process — never claim OFFLINE/STALE from DB age alone.
        # Soft idle only when *all* activity signals are old (not mid-turn lag).
        idle_sec = max(float(stale_sec), DEFAULT_IDLE_SEC)
        if activity_age_sec is not None and activity_age_sec > idle_sec:
            freshness = "idle"

    return {
        "profile": profile,
        "profile_dir": str(profile_dir),
        "collected_at": now,
        "freshness": freshness,
        "stale_sec": stale_sec,
        "idle_sec": max(float(stale_sec), DEFAULT_IDLE_SEC),
        "offline_sec": offline_sec,
        "liveness": live,
        "session_id": conv_id,
        "model": model,
        "model_alert": alert_msg,
        "previous_model": state.get("last_model") if alert_msg else None,
        "window": window,
        "window_source": "context_length_cache_or_fallback",
        "prompt_tokens": real_last,
        "prompt_pct": pct,
        "scaffold_tokens": scaffold_tokens,
        "scaffold_breakdown": {
            "system": sp_tokens,
            "skills": skills_tokens,
            "tools_est": tools_est,
        },
        "conversation_tokens_est": conv_est,
        "conversation_share_pct": conv_share,
        "message_count": int(conv["messages"] or 0),
        "lcm": {
            "loaded": bool(tele.get("loaded")),
            "threshold_ratio": thresh_ratio,
            "threshold_explicit": thresh_explicit,
            "threshold_tokens": thresh_tok,
            "threshold_pct": thresh_pct,
            "compressions": compressions,
            "cache_state": tele.get("cache_state"),
            "activity_band": tele.get("activity_band"),
            "live_snapshot_loaded": bool(tele.get("live_snapshot_loaded")),
            "live_snapshot_age_sec": tele.get("live_snapshot_age_sec"),
            "last_compression_status": tele.get("last_compression_status"),
            "last_compression_noop_reason": tele.get("last_compression_noop_reason"),
            "fresh_tail_count": tele.get("fresh_tail_count"),
            "pre_tail_message_count": tele.get("pre_tail_message_count"),
            "rotate_preview_reason": tele.get("rotate_preview_reason"),
            "total_message_count": tele.get("total_message_count"),
            "leaf_chunk_tokens": tele.get("leaf_chunk_tokens"),
            "turns_since_leaf": tele.get("turns_since_leaf_compaction"),
            "peak_since_leaf": tele.get("peak_prompt_tokens_since_leaf_compaction"),
            "last_leaf_compaction_at": tele.get("last_leaf_compaction_at"),
            "last_compaction_duration_ms": tele.get("last_compaction_duration_ms"),
            "last_api_call_at": tele.get("last_api_call_at"),
            "debt_kind": tele.get("debt_kind"),
            "debt_size_estimate": tele.get("debt_size_estimate"),
            "context_pressure": tele.get("context_pressure"),
            "fill_of_lcm": (real_last / thresh_tok) if thresh_tok > 0 else 0.0,
        },
        "cost": {
            "estimated_usd": est_usd,
            "actual_usd": act_usd,
            "billing_provider": _safe_str(sess, "billing_provider") if sess else None,
            "billing_mode": _safe_str(sess, "billing_mode") if sess else None,
            "cost_status": _safe_str(sess, "cost_status") if sess else None,
            "cost_source": _safe_str(sess, "cost_source") if sess else None,
            "api_calls": api_calls,
            "input_tokens": int((sess or {}).get("input_tokens") or 0),
            "output_tokens": int((sess or {}).get("output_tokens") or 0),
            "cache_read_tokens": int((sess or {}).get("cache_read_tokens") or 0),
            "burn": burn,
            "usage_note": (
                "Desktop /usage may show Nous credits only; trust Visor for "
                "session OpenRouter cost."
            ),
        },
        "constants": {
            "quality_watch_pct": QUALITY_WATCH_PCT,
            "host_compress_pct": HOST_COMPRESS_PCT,
            "hygiene_pct": HYGIENE_PCT,
            "lcm_soon_ratio": LCM_SOON_RATIO,
            "lcm_act_ratio": LCM_ACT_RATIO,
        },
    }
