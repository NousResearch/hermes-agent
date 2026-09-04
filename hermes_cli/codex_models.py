"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Curated offline fallback (first-run, transient API failure). Only slugs the ChatGPT Codex
# OAuth backend actually accepts: the public API's "-pro" variants and the retired
# gpt-5.2-codex / gpt-5.1-codex-max / gpt-5.1-codex-mini return HTTP 400 there ("not supported
# when using Codex with a ChatGPT account"), so listing them leaked dead picker choices. If
# OpenAI re-enables any, live discovery (_fetch_models_from_api) picks them up automatically.
DEFAULT_CODEX_MODELS: List[str] = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    # Research preview exposed ONLY via the Codex OAuth backend for ChatGPT Pro subscribers —
    # not in the public API, so it stays out of the "openai" catalog in hermes_cli/models.py.
    # The backend reports ``supported_in_api: false`` for it; that flag describes API
    # availability, not Codex availability, so fetch/cache paths must not filter on it.
    "gpt-5.3-codex-spark"]

# gpt-5.3-codex-spark is in research preview and is exposed *only* via the Codex CLI / OAuth backend
# (chatgpt.com/backend-api/codex/models) for ChatGPT Pro subscribers. It is NOT available in the public
# OpenAI API, so it intentionally stays out of the "openai" provider catalog in hermes_cli/models.py — only
# the openai-codex (OAuth) provider surfaces it. The Codex backend reports ``supported_in_api: false`` for
# this slug; that flag describes API availability, not Codex backend availability, so the fetch/cache code
# paths below intentionally do not filter on it. PR #12994 removed this entry on the assumption it was
# unsupported — that was wrong; restored here. Keep it in the curated fallback so Pro users still see Spark
# in `/model` when live discovery is unavailable (offline first run, transient API failure).
_FORWARD_COMPAT_TEMPLATE_MODELS: List[tuple[str, tuple[str, ...]]] = [
    ("gpt-5.6-sol", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-terra", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.6-luna", ("gpt-5.5", "gpt-5.4")),
    ("gpt-5.5", ("gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex")),
    ("gpt-5.4-mini", ("gpt-5.3-codex",)),
    ("gpt-5.4", ("gpt-5.3-codex",)),
    # Spark surfaces whenever a compatible template is present; the backend (not Hermes)
    # gates real availability by ChatGPT Pro entitlement.
    ("gpt-5.3-codex-spark", ("gpt-5.3-codex",))]


def _dedupe(model_ids) -> List[str]:
    """Order-preserving dedupe."""
    return list(dict.fromkeys(model_ids))


def _add_forward_compat_models(model_ids: List[str]) -> List[str]:
    """Surface newer Codex slugs missing from live discovery when an older compatible template is
    present (Clawdbot-style synthetic forward-compat catalog)."""
    ordered = _dedupe(model_ids)
    seen = set(ordered)
    for synthetic_model, template_models in _FORWARD_COMPAT_TEMPLATE_MODELS:
        if synthetic_model not in seen and any(template in seen for template in template_models):
            ordered.append(synthetic_model)
            seen.add(synthetic_model)
    return ordered


def _add_context_variants(model_ids: List[str]) -> List[str]:
    """Insert ``<slug>-900k`` large-context picker variants after eligible base slugs.

    Base slugs keep the cheaper advertised 272K limit; the variant opts into the large window.
    The suffix is Hermes-side only — stripped before the id hits the wire (agent/transports/codex.py,
    agent/auxiliary_client.py).
    """
    from agent.model_metadata import CODEX_CONTEXT_VARIANT_SUFFIX, has_codex_context_variant

    out: List[str] = []
    present = set(model_ids)
    for model_id in model_ids:
        out.append(model_id)
        variant = model_id + CODEX_CONTEXT_VARIANT_SUFFIX
        if variant in present or variant in out:
            continue
        if has_codex_context_variant(model_id):
            out.append(variant)
    return out


def _finalize_codex_models(model_ids: List[str]) -> List[str]:
    """Forward-compat synthesis + large-context variant synthesis."""
    return _add_context_variants(_add_forward_compat_models(model_ids))


def _extract_chatgpt_account_id(access_token: str) -> Optional[str]:
    """Best-effort ``chatgpt_account_id`` from the OAuth JWT; None on any parse error.

    The Codex backend requires the ``ChatGPT-Account-Id`` header for the per-account catalog;
    without it ``GET /backend-api/codex/models`` returns ``{"models":[]}`` with HTTP 200, which
    masquerades as "no models" and silently degrades the picker to the curated fallback.
    """
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        acct_id = (
            claims.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
            if isinstance(claims, dict)
            else None)
        return acct_id if isinstance(acct_id, str) and acct_id else None
    except Exception:
        return None


def _ranked_slugs(entries: object) -> List[str]:
    """Visible slugs from a Codex catalog ``models`` list, sorted by (priority, slug), deduped.

    Does not filter on ``supported_in_api``: that flag describes the public OpenAI API, while the
    OAuth-backed Codex backend still accepts slugs marked false there (gpt-5.3-codex-spark).
    """
    sortable = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        visibility = item.get("visibility")
        if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        sortable.append((rank, slug.strip()))

    sortable.sort()
    return _dedupe(slug for _, slug in sortable)


def _fetch_models_from_api(access_token: str) -> List[str]:
    """Fetch available models from the Codex API. Returns visible models sorted by priority."""
    try:
        import httpx
        headers = {"Authorization": f"Bearer {access_token}"}
        acct_id = _extract_chatgpt_account_id(access_token)
        if acct_id:
            headers["ChatGPT-Account-Id"] = acct_id
        resp = httpx.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers=headers,
            timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        entries = data.get("models", []) if isinstance(data, dict) else []
    except Exception as exc:
        logger.debug("Failed to fetch Codex models from API: %s", exc)
        return []

    return _finalize_codex_models(_ranked_slugs(entries))


def _read_default_model(codex_home: Path) -> Optional[str]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _read_cache_models(codex_home: Path) -> List[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    return _ranked_slugs(entries if isinstance(entries, list) else [])


def get_codex_model_ids(access_token: Optional[str] = None) -> List[str]:
    """Available Codex model IDs: live API (if token) > config.toml default > local cache > defaults."""
    codex_home = Path(os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")).expanduser()
    if access_token:
        api_models = _fetch_models_from_api(access_token)
        if api_models:
            return _finalize_codex_models(api_models)
    default_model = _read_default_model(codex_home)
    return _finalize_codex_models(_dedupe([
        *([default_model] if default_model else []), *_read_cache_models(codex_home),
        *DEFAULT_CODEX_MODELS]))
