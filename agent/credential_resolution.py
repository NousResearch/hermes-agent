"""Credential and runtime lookup helpers for auxiliary LLM clients."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable, cast

from agent.credential_pool import load_pool
from hermes_cli.config import get_hermes_home
from utils import base_url_host_matches

logger = logging.getLogger(__name__)

_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_AUTH_JSON_PATH = get_hermes_home() / "auth.json"


def _select_pool_entry(provider: str) -> tuple[bool, Any | None]:
    """Return (pool_exists_for_provider, selected_entry)."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s: %s", provider, exc)
        return False, None
    if not pool or not pool.has_credentials():
        return False, None
    try:
        return True, pool.select()
    except Exception as exc:
        logger.debug("Auxiliary client: could not select pool entry for %s: %s", provider, exc)
        return True, None


def _peek_pool_entry(provider: str) -> Any | None:
    """Best-effort current/next pool entry without mutating selection order."""
    try:
        pool = load_pool(provider)
    except Exception as exc:
        logger.debug("Auxiliary client: could not load pool for %s (peek): %s", provider, exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    try:
        current_fn = getattr(pool, "current", None)
        if callable(current_fn):
            current = current_fn()
            if current is not None:
                return current
        peek_fn = getattr(pool, "peek", None)
        if callable(peek_fn):
            return peek_fn()
    except Exception as exc:
        logger.debug("Auxiliary client: could not peek pool entry for %s: %s", provider, exc)
    return None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _pool_runtime_base_url(entry: Any, fallback: str = "") -> str:
    if entry is None:
        return str(fallback or "").strip().rstrip("/")
    url = (
        getattr(entry, "runtime_base_url", None)
        or getattr(entry, "inference_base_url", None)
        or getattr(entry, "base_url", None)
        or fallback
    )
    return str(url or "").strip().rstrip("/")


def _read_nous_auth(
    *,
    select_pool_entry: Callable[[str], tuple[bool, Any | None]] = _select_pool_entry,
    pool_runtime_base_url: Callable[[Any, str], str] = _pool_runtime_base_url,
) -> dict[str, Any] | None:
    """Read and validate ~/.hermes/auth.json for an active Nous provider."""
    pool_present, entry = select_pool_entry("nous")
    if pool_present:
        if entry is None:
            return None
        return {
            "access_token": getattr(entry, "access_token", ""),
            "refresh_token": getattr(entry, "refresh_token", None),
            "agent_key": getattr(entry, "agent_key", None),
            "inference_base_url": pool_runtime_base_url(entry, _NOUS_DEFAULT_BASE_URL),
            "portal_base_url": getattr(entry, "portal_base_url", None),
            "client_id": getattr(entry, "client_id", None),
            "scope": getattr(entry, "scope", None),
            "token_type": getattr(entry, "token_type", "Bearer"),
            "source": "pool",
        }

    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text())
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        if not isinstance(provider, dict):
            return None
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return cast(dict[str, Any], provider)
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict[str, Any]) -> str:
    """Extract the Nous runtime credential from the compatibility field."""
    return str(provider.get("agent_key") or provider.get("access_token", ""))


def _nous_base_url() -> str:
    """Resolve the Nous inference base URL from env or default."""
    return os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)


def _resolve_nous_runtime_api(*, force_refresh: bool = False) -> tuple[str, str] | None:
    """Return fresh Nous runtime credentials when available."""
    try:
        from hermes_cli.auth import (
            NOUS_INFERENCE_AUTH_MODE_AUTO,
            NOUS_INFERENCE_AUTH_MODE_LEGACY,
            resolve_nous_runtime_credentials,
        )

        creds = resolve_nous_runtime_credentials(
            min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
            timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
            inference_auth_mode=(
                NOUS_INFERENCE_AUTH_MODE_LEGACY
                if force_refresh
                else NOUS_INFERENCE_AUTH_MODE_AUTO
            ),
        )
    except Exception as exc:
        logger.debug("Auxiliary Nous runtime credential resolution failed: %s", exc)
        return None

    api_key = str(creds.get("api_key") or "").strip()
    base_url = str(creds.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _resolve_xai_oauth_for_aux() -> tuple[str, str] | None:
    """Resolve a fresh xAI OAuth (api_key, base_url) for auxiliary clients."""
    try:
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        pool = load_pool("xai-oauth")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                api_key = str(
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                    or ""
                ).strip()
                base_url = str(
                    os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
                    or os.getenv("XAI_BASE_URL", "").strip().rstrip("/")
                    or getattr(entry, "runtime_base_url", None)
                    or getattr(entry, "base_url", None)
                    or DEFAULT_XAI_OAUTH_BASE_URL
                ).strip().rstrip("/")
                if api_key and base_url:
                    return api_key, base_url
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth pool credential resolution failed: %s", exc)

    try:
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
    except Exception as exc:
        logger.debug("Auxiliary xAI OAuth runtime credential resolution failed: %s", exc)
        return None

    api_key = str(creds.get("api_key") or "").strip()
    base_url = str(creds.get("base_url") or "").strip().rstrip("/")
    if not api_key or not base_url:
        return None
    return api_key, base_url


def _read_codex_access_token(
    *,
    select_pool_entry: Callable[[str], tuple[bool, Any | None]] = _select_pool_entry,
    pool_runtime_api_key: Callable[[Any], str] = _pool_runtime_api_key,
) -> str | None:
    """Read a valid, non-expired Codex OAuth access token from Hermes auth store."""
    pool_present, entry = select_pool_entry("openai-codex")
    if pool_present:
        token = pool_runtime_api_key(entry)
        if token:
            return token

    try:
        from hermes_cli.auth import _read_codex_tokens

        data = _read_codex_tokens()
        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            return None

        try:
            payload = access_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            exp = claims.get("exp", 0)
            if exp and time.time() > exp:
                logger.debug("Codex access token expired (exp=%s), skipping", exp)
                return None
        except Exception:
            pass

        return access_token.strip()
    except Exception as exc:
        logger.debug("Could not read Codex auth for auxiliary client: %s", exc)
        return None


_RUNTIME_MAIN_PROVIDER: str = ""
_RUNTIME_MAIN_MODEL: str = ""


def _read_main_model() -> str:
    """Read the user's configured main model from config.yaml."""
    override = _RUNTIME_MAIN_MODEL
    if isinstance(override, str) and override.strip():
        return override.strip()
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str) and model_cfg.strip():
            return model_cfg.strip()
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default", "")
            if isinstance(default, str) and default.strip():
                return default.strip()
    except Exception:
        pass
    return ""


def _read_main_provider() -> str:
    """Read the user's configured main provider from config.yaml."""
    override = _RUNTIME_MAIN_PROVIDER
    if isinstance(override, str) and override.strip():
        return override.strip().lower()
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            provider = model_cfg.get("provider", "")
            if isinstance(provider, str) and provider.strip():
                return provider.strip().lower()
    except Exception:
        pass
    return ""


def set_runtime_main(provider: str, model: str) -> None:
    """Record the live runtime provider/model for the current AIAgent."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    _RUNTIME_MAIN_PROVIDER = (provider or "").strip().lower()
    _RUNTIME_MAIN_MODEL = (model or "").strip()


def clear_runtime_main() -> None:
    """Clear the runtime override (e.g. on session end)."""
    global _RUNTIME_MAIN_PROVIDER, _RUNTIME_MAIN_MODEL
    _RUNTIME_MAIN_PROVIDER = ""
    _RUNTIME_MAIN_MODEL = ""


def _resolve_custom_runtime() -> tuple[str | None, str | None, str | None]:
    """Resolve the active custom/main endpoint the same way the main CLI does."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="custom")
    except Exception as exc:
        logger.debug("Auxiliary client: custom runtime resolution failed: %s", exc)
        runtime = None

    if not isinstance(runtime, dict):
        openai_base = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_base:
            return None, None, None
        runtime = {
            "base_url": openai_base,
            "api_key": openai_key,
        }

    custom_base = runtime.get("base_url")
    custom_key = runtime.get("api_key")
    custom_mode = runtime.get("api_mode")
    if not isinstance(custom_base, str) or not custom_base.strip():
        return None, None, None

    custom_base = custom_base.strip().rstrip("/")
    if base_url_host_matches(custom_base, "openrouter.ai"):
        return None, None, None

    if not isinstance(custom_key, str) or not custom_key.strip():
        custom_key = "no-key-required"

    if not isinstance(custom_mode, str) or not custom_mode.strip():
        custom_mode = None

    return custom_base, custom_key.strip(), custom_mode


def _current_custom_base_url() -> str:
    custom_base, _, _ = _resolve_custom_runtime()
    return custom_base or ""
