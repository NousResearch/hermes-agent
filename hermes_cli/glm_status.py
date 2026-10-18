"""GLM Coding Plan quota segment for the classic CLI status bar.

A daemon poller hits the Z.ai monitor API every ``HERMES_GLM_POLL_SECONDS``
(default 300) and caches the parsed quota; the status bar renders from the
cache and the poller nudges a repaint once a minute so the reset countdown
stays fresh even while idle.

Auth quirk: the monitor endpoint expects the raw API key in the
``Authorization`` header WITHOUT a ``Bearer`` prefix.

Kill switch: ``HERMES_GLM_STATUS_BAR=0`` hides the segment. Without a
``GLM_API_KEY``/``ZAI_API_KEY`` (environment or ``$HERMES_HOME/.env``) the
poller never starts and nothing renders.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

_POLL_SECONDS = max(60.0, float(os.environ.get("HERMES_GLM_POLL_SECONDS", "300") or 300))
_TICK_SECONDS = 60.0
_PROXY_FALLBACK = os.environ.get("HERMES_GLM_PROXY", "http://127.0.0.1:7897")
_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"

_env_cache: Optional[dict] = None
_env_cache_at = 0.0


def _hermes_env_values() -> dict:
    """GLM keys from $HERMES_HOME/.env (cached ~10s)."""
    global _env_cache, _env_cache_at
    now = time.monotonic()
    if _env_cache is not None and now - _env_cache_at < 10:
        return _env_cache
    vals: dict = {}
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    try:
        with open(os.path.join(home, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    vals[k.strip()] = v
    except OSError:
        pass
    _env_cache, _env_cache_at = vals, now
    return vals


def _api_key() -> Optional[str]:
    for k in ("GLM_API_KEY", "ZAI_API_KEY"):
        v = os.environ.get(k) or _hermes_env_values().get(k)
        if v:
            return v
    return None


def _http_get_json(url: str, key: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": key,  # no Bearer prefix — monitor API contract
            "Accept-Language": "en-US,en",
            "Content-Type": "application/json",
        },
    )

    def _open(handler=None):
        opener = urllib.request.build_opener(handler) if handler else urllib.request.build_opener()
        with opener.open(req, timeout=20) as r:
            return json.loads(r.read().decode())

    try:
        return _open()
    except (urllib.error.URLError, OSError, TimeoutError):
        # Direct connection failed -> one retry through the local proxy.
        return _open(urllib.request.ProxyHandler({"http": _PROXY_FALLBACK, "https": _PROXY_FALLBACK}))


def parse_limits(payload: dict) -> dict:
    """Map the monitor API payload to a flat quota record."""
    data = (payload or {}).get("data") or {}
    out = {
        "level": data.get("level") or "",
        "five_pct": None,
        "five_reset_ms": None,
        "week_pct": None,
        "week_reset_ms": None,
        "mcp_used": None,
        "mcp_total": None,
    }
    for item in data.get("limits") or []:
        kind, unit = item.get("type"), item.get("unit")
        if kind == "TOKENS_LIMIT" and unit == 3:
            out["five_pct"] = item.get("percentage")
            out["five_reset_ms"] = item.get("nextResetTime")
        elif kind == "TOKENS_LIMIT" and unit == 6:
            out["week_pct"] = item.get("percentage")
            out["week_reset_ms"] = item.get("nextResetTime")
        elif kind == "TIME_LIMIT" and unit == 5:
            out["mcp_used"] = item.get("currentValue")
            out["mcp_total"] = item.get("usage")
    return out


def fetch_quota() -> Optional[dict]:
    key = _api_key()
    if not key:
        return None
    payload = _http_get_json(_QUOTA_URL, key)
    if not (payload.get("success") or payload.get("code") == 200):
        raise RuntimeError(str(payload.get("msg") or "monitor API error"))
    return parse_limits(payload)


def _countdown(reset_ms: Optional[int]) -> str:
    if not reset_ms:
        return ""
    remaining = reset_ms / 1000.0 - time.time()
    if remaining <= 0:
        return "reset"
    hours, mins = int(remaining // 3600), int(remaining % 3600 // 60)
    if hours >= 48:
        return f"{hours // 24}d{hours % 24:02d}h"
    return f"{hours}h{mins:02d}m" if hours else f"{mins}m"


def _pct_kind(pct: Optional[int]) -> str:
    if pct is None:
        return "dim"
    if pct >= 95:
        return "crit"
    if pct >= 80:
        return "warn"
    return "good"


_SG_TZ = timezone(timedelta(hours=8))  # Singapore = Beijing wall clock


def _peak_fragment(model: str = "", now: Optional[datetime] = None) -> tuple:
    """Legacy-plan peak pricing reminder. Peak = Mon-Fri 14:00-18:00 UTC+8;
    GLM-5.3 bills 3x (1x off), Flash 1.2x (0.4x off). Returns ("", "") when
    off-peak and more than 60 min away so the bar stays quiet."""
    mult = "1.2" if "flash" in str(model).lower() else "3"
    now = now or datetime.now(_SG_TZ)
    start = None
    for i in range(7):
        day = (now + timedelta(days=i)).date()
        if day.weekday() >= 5:
            continue
        cand = datetime(day.year, day.month, day.day, 14, 0, tzinfo=now.tzinfo)
        if now < cand + timedelta(hours=4):
            start = cand
            break
    if not start:
        return "", ""
    if now < start:
        mins = int((start - now).total_seconds() // 60)
        if mins > 60:
            return "", ""
        return f" ⚡{mins}m后入峰{mult}×", "warn"
    return f" ⚡峰{mult}×", "warn"


def build_segment(data: dict, width: int, model: str = "") -> list:
    """Style-agnostic fragments: (kind, text), kind in
    strong/dim/good/warn/crit. Empty list hides the segment (no key, no
    data yet, or terminal narrower than 52 columns)."""
    if not data or width < 52:
        return []
    sep = " │ " if width >= 76 else " · "
    seg = [("dim", sep), ("strong", "GLM")]
    if data.get("level") and width >= 76:
        seg.append(("dim", f" {data['level']}"))
    seg.append(("dim", " 5h "))
    seg.append((_pct_kind(data.get("five_pct")), "--" if data.get("five_pct") is None else f"{data['five_pct']}%"))
    cd = _countdown(data.get("five_reset_ms"))
    if cd and width >= 76:
        seg.append(("dim", f" {cd}"))
    seg.append(("dim", " · wk "))
    seg.append((_pct_kind(data.get("week_pct")), "--" if data.get("week_pct") is None else f"{data['week_pct']}%"))
    if width >= 76:
        cd = _countdown(data.get("week_reset_ms"))
        if cd:
            seg.append(("dim", f" {cd}"))
    peak_text, peak_kind = _peak_fragment(model)
    if peak_text:
        seg.append((peak_kind, peak_text))
    return seg


class QuotaPoller:
    """Single daemon thread; keeps the last good data across failures."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._data: Optional[dict] = None
        self._error = ""
        self._refresh_cb: Optional[Callable[[], None]] = None

    def start(self, refresh: Optional[Callable[[], None]] = None) -> None:
        with self._lock:
            if refresh:
                self._refresh_cb = refresh
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="glm-status-poller", daemon=True)
            self._thread.start()

    def snapshot(self) -> Optional[dict]:
        with self._lock:
            return dict(self._data) if self._data else None

    def error(self) -> str:
        with self._lock:
            return self._error

    def _run(self) -> None:
        while True:
            try:
                data = fetch_quota()
            except Exception as exc:  # keep last good; surface the reason
                with self._lock:
                    self._error = str(exc) or type(exc).__name__
            else:
                if data:
                    with self._lock:
                        self._data, self._error = data, ""
            # Fetch once per poll interval; repaint once per minute so the
            # reset countdowns tick even while the session is idle.
            for _ in range(int(_POLL_SECONDS // _TICK_SECONDS)):
                self._nudge()
                time.sleep(_TICK_SECONDS)
            self._nudge()

    def _nudge(self) -> None:
        cb = self._refresh_cb
        if cb:
            try:
                cb()
            except Exception:
                pass


_poller = QuotaPoller()


def get_poller() -> QuotaPoller:
    return _poller
