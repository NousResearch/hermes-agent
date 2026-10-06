"""
Self Evolution Plugin — Model Configuration & Failover
======================================================

Handles runtime model resolution (primary / fallback / multimodal)
and thread-safe failover state management.

Extracted from reflection_engine.py for single-responsibility.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Model Configuration Resolution ────────────────────────────────────────


def resolve_config() -> dict:
    """Resolve model config via hermes unified runtime provider.

    Returns dict with:
        base_url, api_key, model, provider — primary text model
        fallback: {base_url, api_key, model, provider} — fallback text model
        multimodal: {base_url, api_key, model, provider} — vision model
    Returns empty dict if no provider is available.
    """
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.config import load_config

        runtime = resolve_runtime_provider()
        config = load_config()
        model_name = config.get("model", {}).get("default", "")

        result = {
            "base_url": runtime.get("base_url", ""),
            "api_key": runtime.get("api_key", ""),
            "model": runtime.get("model", model_name),
            "provider": runtime.get("provider", ""),
        }

        result["fallback"] = _resolve_fallback_config(config)
        result["multimodal"] = _resolve_multimodal_config(config)

        return result
    except Exception:
        logger.warning("Failed to resolve runtime provider", exc_info=True)
        return {}


def _resolve_fallback_config(config: dict = None) -> dict:
    """Resolve fallback text model from config.yaml fallback_providers."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        if config is None:
            from hermes_cli.config import load_config
            config = load_config()

        for fb in config.get("fallback_providers", []):
            fb_provider = (fb.get("provider") or "").strip()
            fb_model = (fb.get("model") or "").strip()
            if not fb_provider:
                continue
            try:
                rt = resolve_runtime_provider(requested=fb_provider)
                base_url = rt.get("base_url", "")
                api_key = rt.get("api_key", "")
                if base_url and fb_model:
                    return {
                        "base_url": base_url,
                        "api_key": api_key,
                        "model": fb_model,
                        "provider": rt.get("provider", ""),
                    }
            except Exception:
                pass

        for cp in config.get("custom_providers", []):
            base_url = (cp.get("base_url") or cp.get("api", "")).strip()
            if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                model = (cp.get("model") or "").strip()
                if not model:
                    model = _detect_local_model(
                        base_url,
                        (cp.get("api_key") or "").strip(),
                    )
                if model and "gemma-4-26b" not in model.lower():
                    return {
                        "base_url": base_url.rstrip("/"),
                        "api_key": (cp.get("api_key") or "").strip(),
                        "model": model,
                        "provider": "custom",
                    }

        return {}
    except Exception:
        logger.warning("Failed to resolve fallback config", exc_info=True)
        return {}


def _resolve_multimodal_config(config: dict = None) -> dict:
    """Resolve multimodal (vision) model config."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        if config is None:
            from hermes_cli.config import load_config
            config = load_config()

        aux = config.get("auxiliary", {})
        vision_cfg = aux.get("vision", {})
        vision_provider = (vision_cfg.get("provider") or "").strip().lower()
        if vision_provider and vision_provider != "auto":
            try:
                rt = resolve_runtime_provider(requested=vision_provider)
                if rt.get("base_url"):
                    return {
                        "base_url": rt.get("base_url", ""),
                        "api_key": rt.get("api_key", ""),
                        "model": vision_cfg.get("model") or rt.get("model", ""),
                        "provider": rt.get("provider", ""),
                    }
            except Exception:
                pass

        for cp in config.get("custom_providers", []):
            base_url = (cp.get("base_url") or cp.get("api", "")).strip()
            if base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
                api_key = (cp.get("api_key") or "").strip()
                key_env = (cp.get("key_env") or "").strip()
                if not api_key and key_env:
                    import os
                    api_key = os.getenv(key_env, "")
                model = (cp.get("model") or "").strip()
                if not model:
                    model = _detect_local_model(base_url, api_key)
                if model:
                    return {
                        "base_url": base_url.rstrip("/"),
                        "api_key": api_key,
                        "model": model,
                        "provider": "custom",
                    }

        return {}
    except Exception:
        logger.warning("Failed to resolve multimodal config", exc_info=True)
        return {}


# ── Failover State (thread-safe) ──────────────────────────────────────────

_active_model: str = "primary"
_last_health_check: float = 0.0
_HEALTH_CHECK_INTERVAL: int = 1800  # 30 minutes
_failover_lock = threading.Lock()


def _check_primary_health(config: dict) -> bool:
    """Quick health check: send a minimal request to the primary model."""
    try:
        import requests
        base_url = config.get("base_url", "")
        api_key = config.get("api_key", "")
        model = config.get("model", "")
        if not base_url or not model:
            return False
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 2,
            },
            timeout=15,
        )
        return resp.status_code == 200
    except Exception:
        return False


def get_active_text_config(config: dict) -> tuple:
    """Return (active_config_dict, is_fallback) based on failover state."""
    global _active_model, _last_health_check

    with _failover_lock:
        now = time.time()

        if _active_model == "fallback":
            if now - _last_health_check >= _HEALTH_CHECK_INTERVAL:
                _last_health_check = now
                if _check_primary_health(config):
                    _active_model = "primary"
                    logger.info("Primary model recovered, switching back")
                else:
                    logger.info("Primary model still unavailable, staying on fallback")

        fallback = config.get("fallback", {})
        if _active_model == "primary":
            return config, False
        elif fallback:
            return fallback, True
        else:
            return config, False


def switch_to_fallback():
    """Mark primary as down and switch to fallback."""
    global _active_model, _last_health_check
    with _failover_lock:
        _active_model = "fallback"
        _last_health_check = time.time()
    logger.warning("Primary model failed, switched to fallback")


def _detect_local_model(base_url: str, api_key: str = "") -> str:
    """Auto-detect a multimodal model from a local server."""
    try:
        import requests
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers=headers, timeout=5,
        )
        if resp.ok:
            models = resp.json().get("data", [])
            multimodal_hints = ["gemma-4", "qwen2-vl", "qwen-vl", "llava", "pixtral", "vision"]
            for m in models:
                mid = m.get("id", "").lower()
                for hint in multimodal_hints:
                    if hint in mid:
                        return m["id"]
    except Exception:
        pass
    return ""
