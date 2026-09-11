"""Fleet Models page — backend. Mounted at /api/plugins/fleet-models/ behind the dashboard's own auth.

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


_CORE_LOCK = threading.Lock()


def _core():
    """core.py, loaded once per process. The dashboard serves /state and /usage on parallel threads, and the
    page asks for both at once: without the lock, the second thread found the half-executed module in
    sys.modules and failed with "no attribute 'fleet_root'" (a 500 on the first load after every restart)."""
    name = "fleet_models_core"
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "_fleet_models_ready", False):
        return mod
    with _CORE_LOCK:
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "_fleet_models_ready", False):
            spec = importlib.util.spec_from_file_location(name, _HERE.parent / "core.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(name, None)
                raise
            mod._fleet_models_ready = True
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
# A ledger row (session_model_usage) is one session × model × host × task, with the time of its first and last
# call. Short windows need more than "when did it end": each row's usage is spread evenly across its active span
# (first_seen → last_seen) and pro-rated into the window and into each bucket. Measured 2026-09-12: ~70% of calls
# sit in rows spanning 1–15 min and ~12% in rows over an hour, so 15-minute views are close estimates, not guesses.
# Buckets align to the VIEWER's local clock (IANA tz from the browser) — hours on the hour, days at local midnight,
# DST-safe — so "24 h" reads 09:00, 10:00… in Auckland rather than on UTC boundaries.
WINDOW_MIN, WINDOW_MAX = 900, 90 * 86400
NICE_BUCKETS = (60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800)
MAX_BUCKETS = 400
_TZ_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_+-/")


def natural_bucket(window: int) -> int:
    """The bucket a window reads best in when there is room: 24 h by the hour, 7 days by 6 h, 30 days by day."""
    for w, b in ((900, 60), (1800, 60), (3600, 120), (7200, 300), (10800, 300), (21600, 900), (43200, 1800),
                 (86400, 3600), (7 * 86400, 21600)):
        if window <= w:
            return b
    return 86400


def _tzinfo(name: Optional[str]):
    from datetime import timezone
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # pragma: no cover
        ZoneInfo = None
    cands = [name] if name else []
    cands.append(os.environ.get("TZ"))
    try:
        link = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in link:
            cands.append(link.split("zoneinfo/", 1)[1])
    except OSError:
        pass
    for c in cands:
        if c and ZoneInfo and len(c) <= 64 and set(c) <= _TZ_OK:
            try:
                return ZoneInfo(c), c
            except Exception:  # noqa: BLE001
                continue
    loc = __import__("datetime").datetime.now().astimezone().tzinfo
    return loc or timezone.utc, str(loc or "UTC")


def _floor_local(ts: float, b: int, tz) -> float:
    from datetime import datetime, timedelta
    dt = datetime.fromtimestamp(ts, tz)
    mid = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if b >= 86400:
        return (mid - timedelta(days=dt.date().toordinal() % (b // 86400))).timestamp()
    secs = dt.hour * 3600 + dt.minute * 60 + dt.second
    return (mid + timedelta(seconds=secs - secs % b)).timestamp()


def bucket_edges(since: float, now: float, b: int, tz) -> List[float]:
    """Local-clock bucket boundaries covering [since, now]. Stepping 1.5 buckets and re-flooring keeps days at
    midnight across a 23- or 25-hour DST day, and hours on the hour after the clocks change."""
    edges = [_floor_local(since, b, tz)]
    while edges[-1] < now and len(edges) <= MAX_BUCKETS + 2:
        cur = edges[-1]
        nxt = _floor_local(cur + 1.5 * b, b, tz)
        edges.append(nxt if nxt > cur else cur + b)
    return edges


def _ledgers(root: Path):
    out = [("root", root / "state.db")]
    out += [(p.parent.name, p) for p in sorted((root / "profiles").glob("*/state.db"))]
    return [(n, p) for n, p in out if p.exists()]


def _usage_rows(root: Path, since: float, now: float):
    """(profile, model, host, base, task, calls, in, out, cache_read, cache_write, cost, source, first, last)."""
    for prof, db in _ledgers(root):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            q = ("SELECT model, COALESCE(provider_name,''), COALESCE(billing_base_url,''), COALESCE(task,''), "
                 "COALESCE(api_call_count,0), COALESCE(input_tokens,0), COALESCE(output_tokens,0), "
                 "COALESCE(cache_read_tokens,0), COALESCE(cache_write_tokens,0), "
                 "CASE WHEN COALESCE(actual_cost_usd,0) > 0 THEN actual_cost_usd WHEN COALESCE(total_cost,0) > 0 THEN total_cost "
                 "ELSE COALESCE(estimated_cost_usd,0) END, COALESCE(cost_source,''), "
                 "COALESCE(first_seen, last_seen), last_seen "
                 "FROM session_model_usage WHERE last_seen >= ? AND COALESCE(first_seen, last_seen) <= ?")
            for r in c.execute(q, (since, now)):
                yield (prof,) + tuple(r)
            c.close()
        except sqlite3.Error:
            continue


def usage(window: int = 7 * 86400, bucket: Optional[int] = None, tz: Optional[str] = None,
          now: Optional[float] = None, root: Optional[Path] = None) -> dict:
    import bisect
    core = _core(); root = root or _root()
    window = max(WINDOW_MIN, min(int(window), WINDOW_MAX))
    b = int(bucket) if bucket in NICE_BUCKETS else natural_bucket(window)
    while window / b > MAX_BUCKETS:
        b = next((x for x in NICE_BUCKETS if x > b), b * 2)
    tzi, tzname = _tzinfo(tz)
    now = float(now if now is not None else time.time())
    since = now - window
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
    edges = bucket_edges(since, now, b, tzi)
    nb = len(edges) - 1
    ser = [{"billed_usd": 0.0, "calls": 0.0, "modelark_calls": 0.0} for _ in range(nb)]
    by_p: Dict[str, dict] = {}
    by_m: Dict[str, dict] = {}
    agg: Dict[tuple, dict] = {}

    def idx(t):
        return min(nb - 1, max(0, bisect.bisect_right(edges, t) - 1))

    for (prof, model, host, base, task, n, inp, out, cr, cw, cost, src, fs, ls) in _usage_rows(root, since, now):
        fs = float(fs if fs is not None else ls); ls = float(ls)
        span = ls - fs
        lo, hi = max(fs, since), min(ls, now)
        if span <= 1.0:
            parts = [(idx(ls), 1.0)] if since <= ls <= now else []
        elif hi > lo:
            parts = []
            for i in range(idx(lo), idx(hi) + 1):
                ov = min(hi, edges[i + 1]) - max(lo, edges[i])
                if ov > 0:
                    parts.append((i, ov / span))
        else:
            parts = []
        frac = sum(f for _, f in parts)
        if frac <= 0:
            continue
        ark = "bytepluses.com" in base or src in ("modelark subscription", "modelark-proxy")
        r = rates.get((model or "").lower()) or ((0.66, 1.98, 0.022) if "pro" in (model or "") else (0.15, 0.60, 0.003))
        capeq = ((inp + cw) * r[0] + out * r[1] + cr * r[2]) / 1e6 if ark else 0.0
        billed = 0.0 if ark else float(cost or 0)  # 1.x "modelark-proxy" rows stored the cap-equivalent AS cost; never invoiced
        calls = float(n or 0)
        pp = by_p.setdefault(prof, {"calls": [0.0] * nb, "billed": [0.0] * nb})
        mm = by_m.setdefault(model or "?", {"calls": [0.0] * nb, "billed": [0.0] * nb})
        for i, f in parts:
            s_ = ser[i]
            s_["calls"] += calls * f; s_["billed_usd"] += billed * f
            if ark:
                s_["modelark_calls"] += calls * f
            pp["calls"][i] += calls * f; pp["billed"][i] += billed * f
            mm["calls"][i] += calls * f; mm["billed"][i] += billed * f
        k = (prof, model, host or ("ModelArk" if ark else ""), task or "main", ark)
        a = agg.setdefault(k, {"profile": prof, "model": model, "host": k[2], "task": k[3], "modelark": ark, "calls": 0.0,
                               "input": 0.0, "output": 0.0, "cache_read": 0.0, "billed_usd": 0.0, "cap_equivalent_usd": 0.0})
        a["calls"] += calls * frac; a["input"] += inp * frac; a["output"] += out * frac; a["cache_read"] += cr * frac
        a["billed_usd"] += billed * frac; a["cap_equivalent_usd"] += capeq * frac

    rows = []
    for a in agg.values():
        for x in ("calls", "input", "output", "cache_read"):
            a[x] = int(round(a[x]))
        a["billed_usd"] = round(a["billed_usd"], 6); a["cap_equivalent_usd"] = round(a["cap_equivalent_usd"], 6)
        if a["calls"] or a["billed_usd"]:
            rows.append(a)
    series = [{"t": edges[i], "end": edges[i + 1], "partial": edges[i] < since or edges[i + 1] > now,
               "calls": round(s_["calls"], 3), "modelark_calls": round(s_["modelark_calls"], 3),
               "billed_usd": round(s_["billed_usd"], 6)} for i, s_ in enumerate(ser)]
    rnd = lambda d: {k: {"calls": [round(v, 3) for v in x["calls"]], "billed": [round(v, 6) for v in x["billed"]]} for k, x in d.items()}
    return {"window": window, "bucket": b, "since": since, "now": now, "tz": tzname, "days": round(window / 86400, 4),
            "method": "spread", "rows": sorted(rows, key=lambda r: (-r["billed_usd"], -r["calls"])), "series": series,
            "by_profile": rnd(by_p), "by_model": rnd(by_m), "generated_at": time.time()}


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
def get_usage(window: Optional[int] = None, bucket: Optional[int] = None, tz: Optional[str] = None, days: Optional[int] = None):
    """window = seconds (15 min … 90 days); bucket = one of NICE_BUCKETS (else the window's natural bucket);
    tz = the viewer's IANA zone. `days` is the 1.0.x form, kept for old tabs still open."""
    if window:
        return usage(int(window), bucket=bucket, tz=tz)
    # a 1.0.4 page still open in a browser: daily UTC buckets under the old "daily" key
    u = usage(int(days or 7) * 86400, bucket=86400, tz="UTC")
    u["daily"] = [{"day": int(x["t"] // 86400), "billed_usd": x["billed_usd"], "calls": int(round(x["calls"])),
                   "modelark_calls": int(round(x["modelark_calls"]))} for x in u["series"]]
    return u


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
