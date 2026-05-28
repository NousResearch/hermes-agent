"""Small persistent health cache for managed-agent model fallback.

This is intentionally narrow: it records short cooldowns for model_refs that
just failed and lets managed-agent routing skip them on later delegate runs.
It is not a quota authority and should never block all models permanently.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

_WRITE_LOCK = threading.Lock()


def _health_path(path: Optional[Path] = None) -> Path:
    return Path(path) if path is not None else get_hermes_home() / "config" / "model-health.json"


def _cooldown_seconds(reason: str) -> int:
    normalized = (reason or "").strip().lower()
    if normalized in {"rate_limit", "rate_limited", "quota_exceeded", "billing"}:
        return 3600
    if normalized in {"timeout", "server_error"}:
        return 600
    if normalized == "empty_final_content":
        return 900
    return 300


def load_model_health(path: Optional[Path] = None) -> Dict[str, Any]:
    p = _health_path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"models": {}}


def save_model_health(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = _health_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tid = threading.get_ident()
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{tid}.tmp")
    with _WRITE_LOCK:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)


def mark_model_unhealthy(
    model_ref: str,
    *,
    reason: str = "unknown",
    provider: str = "",
    model: str = "",
    base_url: str = "",
    path: Optional[Path] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    ref = (model_ref or "").strip()
    if not ref:
        return {}
    ts = float(now if now is not None else time.time())
    cooldown = _cooldown_seconds(reason)
    data = load_model_health(path)
    models = data.setdefault("models", {})
    entry = {
        "model_ref": ref,
        "status": "cooldown",
        "reason": reason or "unknown",
        "provider": provider or "",
        "model": model or "",
        "base_url": base_url or "",
        "last_failure_at": ts,
        "cooldown_until": ts + cooldown,
        "cooldown_seconds": cooldown,
    }
    models[ref] = entry
    save_model_health(data, path)
    return entry


def get_model_health(model_ref: str, *, path: Optional[Path] = None, now: Optional[float] = None) -> Dict[str, Any]:
    ref = (model_ref or "").strip()
    if not ref:
        return {"status": "unknown"}
    ts = float(now if now is not None else time.time())
    data = load_model_health(path)
    entry = (data.get("models") or {}).get(ref)
    if not isinstance(entry, dict):
        return {"model_ref": ref, "status": "healthy", "cooldown_remaining_seconds": 0}
    cooldown_until = float(entry.get("cooldown_until") or 0)
    remaining = max(0, int(cooldown_until - ts))
    if remaining <= 0:
        return {
            **entry,
            "model_ref": ref,
            "status": "healthy",
            "cooldown_remaining_seconds": 0,
        }
    return {
        **entry,
        "model_ref": ref,
        "status": "cooldown",
        "cooldown_remaining_seconds": remaining,
    }


def is_model_in_cooldown(model_ref: str, *, path: Optional[Path] = None, now: Optional[float] = None) -> bool:
    return get_model_health(model_ref, path=path, now=now).get("status") == "cooldown"


def model_health_snapshot(*, path: Optional[Path] = None, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    data = load_model_health(path)
    refs = (data.get("models") or {}).keys()
    return {ref: get_model_health(ref, path=path, now=now) for ref in refs}
