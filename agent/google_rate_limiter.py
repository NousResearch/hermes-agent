"""Local Google API rate-limit guard for Hermes.

Reads ``google_api_rate_limits`` from ~/.hermes/config.yaml and throttles
Google/Gemini requests before they are sent.  This is intentionally local and
best-effort: it prevents Hermes itself from exceeding the configured project
limits, but it cannot see traffic made outside this machine/process.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    block = cfg.get("google_api_rate_limits")
    return block if isinstance(block, dict) else {}


def _state_path() -> Path:
    return _hermes_home() / "cache" / "google_rate_usage.json"


def _read_state() -> dict[str, Any]:
    path = _state_path()
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception:
        return {"requests": []}
    if not isinstance(data, dict):
        return {"requests": []}
    reqs = data.get("requests")
    if not isinstance(reqs, list):
        data["requests"] = []
    return data


def _write_state(data: dict[str, Any]) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(data, fh, separators=(",", ":"))
        tmp.replace(path)
    except Exception as exc:
        logger.debug("google rate-limit state write failed: %s", exc)


def _slug(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("models/", "")
    value = re.sub(r"[^a-z0-9.]+", "-", value)
    return value.strip("-")


def _is_google_request(provider: str, base_url: str) -> bool:
    p = (provider or "").strip().lower()
    u = (base_url or "").strip().lower()
    return p in {"gemini", "google", "google-gemini", "google-gemini-cli"} or "generativelanguage.googleapis.com" in u


def _find_model_limit(cfg: dict[str, Any], model: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    models = cfg.get("models") if isinstance(cfg.get("models"), dict) else {}
    wanted = _slug(model)
    if wanted in models and isinstance(models[wanted], dict):
        return wanted, models[wanted]
    for key, val in models.items():
        if not isinstance(val, dict):
            continue
        aliases = val.get("aliases") or []
        if isinstance(aliases, list) and wanted in {_slug(str(a)) for a in aliases}:
            return str(key), val
        display = _slug(str(val.get("display_name") or ""))
        if wanted and wanted == display:
            return str(key), val
    return None, None


def _rough_tokens(obj: Any) -> int:
    """Cheap token estimate for quota reservation; intentionally conservative."""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    # Mostly English/code; 4 chars/token is common. Add 10% safety margin.
    return max(1, int((len(text) / 4.0) * 1.10))


def _estimate_request_tokens(api_kwargs: dict[str, Any]) -> int:
    prompt_tokens = _rough_tokens(api_kwargs.get("messages") or api_kwargs.get("input") or api_kwargs)
    max_out = api_kwargs.get("max_completion_tokens", api_kwargs.get("max_tokens", 0)) or 0
    try:
        max_out = int(max_out)
    except Exception:
        max_out = 0
    # Reserve prompt + requested output when explicit. If output cap is absent,
    # don't invent a huge value; providers/models have their own defaults.
    return prompt_tokens + max(0, max_out)


def throttle_google_api(provider: str, base_url: str, model: str, api_kwargs: dict[str, Any], status_callback=None) -> None:
    """Sleep/raise before Google API requests based on configured RPM/TPM/RPD.

    Raises RuntimeError for disabled models (0 limits), over-size single
    requests, or exhausted daily quota. Otherwise sleeps until the rolling
    minute window has capacity, then records a quota reservation.
    """
    if not _is_google_request(provider, base_url):
        return
    cfg = _load_config()
    if not cfg.get("enabled", False):
        return
    key, limits = _find_model_limit(cfg, model or str(api_kwargs.get("model") or ""))
    if not key or not limits:
        return

    rpm = limits.get("rpm")
    tpm = limits.get("tpm")
    rpd = limits.get("rpd")
    display = limits.get("display_name") or key
    est_tokens = _estimate_request_tokens(api_kwargs)

    # 0 means unavailable for this key/project.
    for name, value in (("RPM", rpm), ("TPM", tpm), ("RPD", rpd)):
        if value == 0:
            raise RuntimeError(f"Google API model unavailable by configured quota: {display} has {name}=0")
    if isinstance(tpm, int) and tpm > 0 and est_tokens > tpm:
        raise RuntimeError(
            f"Google API request too large for {display}: estimated {est_tokens:,} tokens exceeds TPM {tpm:,}"
        )

    # Wait for rolling minute capacity. RPD exhaustion cannot be safely waited
    # in an interactive call, so raise instead.
    max_sleep = 120.0
    slept = 0.0
    while True:
        now = time.time()
        day_ago = now - 86400
        minute_ago = now - 60
        with _LOCK:
            state = _read_state()
            reqs = [r for r in state.get("requests", []) if isinstance(r, dict) and float(r.get("ts", 0) or 0) >= day_ago]
            same_day = [r for r in reqs if r.get("model") == key]
            if isinstance(rpd, int) and rpd > 0 and len(same_day) >= rpd:
                raise RuntimeError(f"Google API daily quota exhausted for {display}: {len(same_day)}/{rpd} RPD")
            same_min = [r for r in same_day if float(r.get("ts", 0) or 0) >= minute_ago]
            used_rpm = len(same_min)
            used_tpm = sum(int(r.get("tokens", 0) or 0) for r in same_min)
            wait_until = 0.0
            if isinstance(rpm, int) and rpm > 0 and used_rpm >= rpm:
                wait_until = max(wait_until, min(float(r.get("ts", now) or now) for r in same_min) + 60)
            if isinstance(tpm, int) and tpm > 0 and used_tpm + est_tokens > tpm:
                # Wait until enough old token reservations leave the window.
                total = used_tpm
                for r in sorted(same_min, key=lambda x: float(x.get("ts", now) or now)):
                    total -= int(r.get("tokens", 0) or 0)
                    candidate = float(r.get("ts", now) or now) + 60
                    if total + est_tokens <= tpm:
                        wait_until = max(wait_until, candidate)
                        break
            if wait_until <= now:
                reqs.append({"ts": now, "model": key, "tokens": est_tokens})
                state["requests"] = reqs
                _write_state(state)
                return

        delay = min(max(0.5, wait_until - now), max_sleep - slept)
        if delay <= 0:
            raise RuntimeError(f"Google API rate-limit wait exceeded {int(max_sleep)}s for {display}")
        msg = f"Google API rate limit for {display}; waiting {int(delay)}s"
        logger.info(msg)
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass
        time.sleep(delay)
        slept += delay
