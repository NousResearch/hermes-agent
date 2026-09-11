"""Models tab — backend. Mounted at /api/plugins/fleet-models/ behind the dashboard's own auth.

Reads: models.yaml (the source of truth), the nine live configs (what Hermes will actually do — drift is
shown, never hidden), every profile's state.db (real usage, billed dollars, which host served), and
OpenRouter's public endpoints API (prices, uptime, status per host; cached 10 min, falls back to the
nightly snapshot). Writes go through core.apply only: validate → pre-image → compile → verify → history.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
_HERE = Path(__file__).resolve().parent


def _core():
    name = "fleet_models_core"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, _HERE.parent / "core.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _root() -> Path:
    return _core().fleet_root()


# ── OpenRouter market data ───────────────────────────────────────────────────────────────────
_MARKET: Dict[str, tuple] = {}
_MARKET_TTL = 600
_UA = {"User-Agent": "hermes-fleet-models/1.0", "Accept": "application/json"}


def _pct(v):
    return round(float(v), 2) if isinstance(v, (int, float)) else None


def _per_m(v):
    try:
        return round(float(v) * 1_000_000, 4)
    except (TypeError, ValueError):
        return None


def _normalise_api(model_id: str, data: dict) -> dict:
    eps = []
    for e in (data.get("endpoints") or []):
        pr = e.get("pricing") or {}
        params = e.get("supported_parameters") or []
        eps.append({
            "tag": e.get("tag"), "provider": e.get("provider_name"), "quant": e.get("quantization"),
            "in": _per_m(pr.get("prompt")), "out": _per_m(pr.get("completion")),
            "cache_read": _per_m(pr.get("input_cache_read")), "cache_write": _per_m(pr.get("input_cache_write")),
            "discount": pr.get("discount") or 0, "ctx": e.get("context_length"),
            "uptime_1d": _pct(e.get("uptime_last_1d")), "uptime_30m": _pct(e.get("uptime_last_30m")),
            "status": e.get("status"), "tools": "tools" in params, "reasoning": "reasoning" in params,
            "latency_ms": (e.get("latency_last_30m") or {}).get("p50") if isinstance(e.get("latency_last_30m"), dict) else e.get("latency_last_30m"),
            "tps": (e.get("throughput_last_30m") or {}).get("p50") if isinstance(e.get("throughput_last_30m"), dict) else e.get("throughput_last_30m"),
        })
    return {"model": model_id, "name": data.get("name"), "modality": (data.get("architecture") or {}).get("modality"),
            "endpoints": eps, "source": "live", "fetched_at": time.time()}


def _snapshot_market(model_id: str) -> Optional[dict]:
    root = _root()
    try:
        snap = json.loads((root / "state" / "or-prices.json").read_text())
        m = (snap.get("models") or {}).get(model_id)
        if m:
            eps = [{"tag": t, "provider": e.get("provider_name"), "quant": e.get("quant"), "in": e.get("in"), "out": e.get("out"),
                    "cache_read": e.get("cache_read"), "cache_write": e.get("cache_write"), "discount": e.get("discount") or 0,
                    "ctx": e.get("ctx"), "uptime_1d": _pct(e.get("uptime")), "uptime_30m": None, "status": e.get("status"),
                    "tools": e.get("tools"), "reasoning": None, "latency_ms": None, "tps": None}
                   for t, e in (m.get("endpoints") or {}).items()]
            return {"model": model_id, "endpoints": eps, "source": "snapshot", "fetched_at": snap.get("generated_at")}
    except (OSError, ValueError):
        pass
    return None


def market(model_id: str, fresh: bool = False) -> dict:
    hit = _MARKET.get(model_id)
    if hit and not fresh and time.time() - hit[0] < _MARKET_TTL:
        return hit[1]
    try:
        url = "https://openrouter.ai/api/v1/models/" + model_id + "/endpoints"
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=8) as r:
            data = json.loads(r.read().decode()).get("data") or {}
        out = _normalise_api(model_id, data)
    except Exception as exc:  # noqa: BLE001
        out = _snapshot_market(model_id) or {"model": model_id, "endpoints": [], "source": "unavailable", "error": str(exc)[:200]}
    _MARKET[model_id] = (time.time(), out)
    return out


def _uptime_snapshot(doc: dict) -> dict:
    """{model id: {endpoints: {tag: {uptime}}}} for validation of host rules (cached market data only)."""
    out = {}
    for m in (doc.get("models") or {}).values():
        if m.get("provider") != "openrouter":
            continue
        mk = market(m["id"]) if (m.get("rules") or {}).get("min_uptime") else (_MARKET.get(m["id"], (0, None))[1] or _snapshot_market(m["id"]))
        if mk and mk.get("endpoints"):
            out[m["id"]] = {"endpoints": {e["tag"]: {"uptime": e.get("uptime_1d")} for e in mk["endpoints"] if e.get("tag")}}
    return out


# ── usage from every ledger ──────────────────────────────────────────────────────────────────
def _ledgers(root: Path):
    out = [("root", root / "state.db")]
    out += [(p.parent.name, p) for p in sorted((root / "profiles").glob("*/state.db"))]
    return [(n, p) for n, p in out if p.exists()]


def usage(days: int = 7) -> dict:
    core = _core(); root = _root()
    try:
        doc = core.load_doc(root)
    except Exception:  # noqa: BLE001
        doc = {"models": {}}
    rates = {}
    for m in (doc.get("models") or {}).values():
        if m.get("provider") == "modelark":
            ce = m.get("cap_equivalent") or {}
            for n in [m.get("id"), *(m.get("served_as") or [])]:
                rates[str(n).lower()] = (float(ce.get("input") or 0), float(ce.get("output") or 0), float(ce.get("cache_read") or 0))
    since = time.time() - days * 86400
    rows, daily = [], {}
    for prof, db in _ledgers(root):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            q = ("SELECT model, COALESCE(provider_name,''), COALESCE(billing_base_url,''), COALESCE(task,''), "
                 "SUM(COALESCE(api_call_count,0)), SUM(COALESCE(input_tokens,0)), SUM(COALESCE(output_tokens,0)), "
                 "SUM(COALESCE(cache_read_tokens,0)), SUM(COALESCE(cache_write_tokens,0)), "
                 "SUM(CASE WHEN COALESCE(actual_cost_usd,0) > 0 THEN actual_cost_usd WHEN COALESCE(total_cost,0) > 0 THEN total_cost "
                 "ELSE COALESCE(estimated_cost_usd,0) END), "
                 "COALESCE(cost_source,''), CAST(last_seen/86400 AS INT) "
                 "FROM session_model_usage WHERE last_seen >= ? GROUP BY 1,2,3,4,11,12")
            for (model, host, base, task, n, inp, out, cr, cw, cost, src, day) in c.execute(q, (since,)):
                ark = "bytepluses.com" in base or src in ("modelark subscription", "modelark-proxy")
                r = rates.get((model or "").lower()) or ((0.66, 1.98, 0.022) if "pro" in (model or "") else (0.15, 0.60, 0.003))
                capeq = ((inp + cw) * r[0] + out * r[1] + cr * r[2]) / 1e6 if ark else 0.0
                billed = 0.0 if ark and src != "modelark-proxy" else float(cost or 0)
                if src == "modelark-proxy":
                    billed = 0.0  # 1.x rows stored the cap-equivalent AS cost; it was never invoiced
                rows.append({"profile": prof, "model": model, "host": host or ("ModelArk" if ark else ""), "task": task or "main",
                             "calls": int(n or 0), "input": int(inp or 0), "output": int(out or 0), "cache_read": int(cr or 0),
                             "billed_usd": round(billed, 6), "modelark": ark, "cap_equivalent_usd": round(capeq, 6), "day": int(day)})
                d = daily.setdefault(int(day), {"billed_usd": 0.0, "modelark_calls": 0, "calls": 0})
                d["billed_usd"] += billed; d["calls"] += int(n or 0)
                if ark:
                    d["modelark_calls"] += int(n or 0)
            c.close()
        except sqlite3.Error:
            continue
    # merge rows that differ only by day
    agg: Dict[tuple, dict] = {}
    for r in rows:
        k = (r["profile"], r["model"], r["host"], r["task"], r["modelark"])
        a = agg.get(k)
        if a is None:
            agg[k] = {x: v for x, v in r.items() if x != "day"}
        else:
            for x in ("calls", "input", "output", "cache_read", "billed_usd", "cap_equivalent_usd"):
                a[x] += r[x]
    series = [{"day": d, **{k: round(v, 6) if isinstance(v, float) else v for k, v in daily[d].items()}} for d in sorted(daily)]
    return {"days": days, "rows": sorted(agg.values(), key=lambda r: (-r["billed_usd"], -r["calls"])), "daily": series,
            "generated_at": time.time()}


def _caps(root: Path) -> dict:
    """Cost caps are Richie's alone — shown read-only."""
    try:
        cfg = _core()._load_plain(root / "config.yaml")
        k = cfg.get("kanban") or {}
        return {x: k.get(x) for x in ("default_max_cost", "max_cost_ceiling", "max_cost_hard_ceiling") if x in k}
    except Exception:  # noqa: BLE001
        return {}


def _runtime_status() -> dict:
    try:
        from agent import auxiliary_client as ac
        aux = bool(getattr(ac._build_call_kwargs, "_fleet_models_wrapped", False))
    except Exception:  # noqa: BLE001
        aux = False
    return {"aux_routing_seam": aux}


# ── routes ───────────────────────────────────────────────────────────────────────────────────
@router.get("/state")
def get_state():
    core = _core(); root = _root()
    try:
        doc = core.load_doc(root, fresh=True)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    live = {}
    drift = {}
    for p in core.PROFILES:
        try:
            cfg = core._load_plain(core.cfg_path(p, root))
            eff = core.effective(cfg)
            live[p] = {k: eff[k] for k in ("main", "subagents", "cron", "aux", "routing", "reasoning", "reasoning_overrides")}
            probs = core.verify_profile(doc, p, cfg)
            if probs:
                drift[p] = probs
        except Exception as exc:  # noqa: BLE001
            drift[p] = [f"config unreadable: {exc}"]
    errors, warnings = core.validate(doc)
    return {"doc": doc, "revision": int(doc.get("revision") or 0), "live": live, "drift": drift,
            "validation": {"errors": errors, "warnings": warnings}, "history": core.history(root, 40),
            "caps": _caps(root), "runtime": _runtime_status(), "profiles": core.PROFILES,
            "generated_at": time.time()}


@router.get("/usage")
def get_usage(days: int = 7):
    return usage(max(1, min(int(days), 90)))


@router.get("/market")
def get_market(model: str, fresh: bool = False):
    if not model or len(model) > 120 or ".." in model:
        raise HTTPException(400, "bad model id")
    return market(model, fresh=fresh)


class Change(BaseModel):
    doc: Dict[str, Any]
    base_revision: Optional[int] = None
    unlock: List[str] = []
    summary: str = ""


@router.post("/plan")
def post_plan(ch: Change):
    core = _core()
    r = core.apply(ch.doc, by="dashboard", root=_root(), base_revision=ch.base_revision,
                   unlock=tuple(ch.unlock), dry_run=True, snapshot=_uptime_snapshot(ch.doc))
    return r


@router.post("/apply")
def post_apply(ch: Change):
    core = _core()
    return core.apply(ch.doc, by="dashboard", summary=ch.summary[:200], root=_root(),
                      base_revision=ch.base_revision, unlock=tuple(ch.unlock), snapshot=_uptime_snapshot(ch.doc))


class Revert(BaseModel):
    id: str


@router.post("/revert")
def post_revert(rv: Revert):
    return _core().revert(rv.id, by="dashboard", root=_root())


class Probe(BaseModel):
    model: str
    host: str


_PROBES: List[float] = []
_PROBE_LOCK = threading.Lock()


@router.post("/probe")
def post_probe(pb: Probe):
    """routable(): one tiny call pinned to ONE host with fallbacks off. An unmatched slug in `order` fails
    SILENTLY on OpenRouter, so a host pin is only trusted once a call has actually been served by it."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:
            from hermes_cli.env_loader import load_hermes_dotenv  # type: ignore
            load_hermes_dotenv(); key = os.environ.get("OPENROUTER_API_KEY")
        except Exception:  # noqa: BLE001
            key = None
    if not key:
        raise HTTPException(503, "OPENROUTER_API_KEY not available to the dashboard process")
    with _PROBE_LOCK:
        now = time.time()
        _PROBES[:] = [t for t in _PROBES if now - t < 3600]
        if len(_PROBES) >= 30:
            raise HTTPException(429, "probe budget: 30 per hour")
        _PROBES.append(now)
    body = {"model": pb.model, "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "max_tokens": 64, "provider": {"only": [pb.host], "allow_fallbacks": False, "data_collection": "deny"},
            "usage": {"include": True}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", **_UA})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        served = data.get("provider")
        u = data.get("usage") or {}
        return {"ok": True, "host": pb.host, "served_by": served, "latency_ms": int((time.time() - t0) * 1000),
                "cost_usd": u.get("cost"), "routable": bool(served)}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        # 429 = the pin MATCHED a host that is busy right now (OpenRouter moves on to the next pinned host);
        # 404 "no endpoints" = the pin matches nothing and would be dropped SILENTLY in a real request.
        busy = e.code == 429
        return {"ok": False, "host": pb.host, "routable": busy, "rate_limited": busy, "status": e.code,
                "error": "busy right now — the pin matches; requests move on to the next host" if busy else msg,
                "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "host": pb.host, "routable": False, "error": str(exc)[:300]}
