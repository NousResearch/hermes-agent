"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import os

logger = logging.getLogger(__name__)

DEFAULT_CODEX_MODELS: List[str] = [
    # GPT-5.6 series exposed by the ChatGPT Codex OAuth backend.
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.3-codex",
    # gpt-5.3-codex-spark is in research preview and is exposed *only* via
    # the Codex CLI / OAuth backend (chatgpt.com/backend-api/codex/models)
    # for ChatGPT Pro subscribers. It is NOT available in the public OpenAI
    # API, so it intentionally stays out of the "openai" provider catalog
    # in hermes_cli/models.py — only the openai-codex (OAuth) provider
    # surfaces it. The Codex backend reports ``supported_in_api: false`` for
    # this slug; that flag describes API availability, not Codex backend
    # availability, so the fetch/cache code paths below intentionally do
    # not filter on it. PR #12994 removed this entry on the assumption it
    # was unsupported — that was wrong; restored here. Keep it in the
    # curated fallback so Pro users still see Spark in `/model` when live
    # discovery is unavailable (offline first run, transient API failure).
    "gpt-5.3-codex-spark",
    # NOTE: gpt-5.2-codex / gpt-5.1-codex-max / gpt-5.1-codex-mini were
    # previously listed here but the chatgpt.com Codex backend returns
    # HTTP 400 "The '<model>' model is not supported when using Codex with
    # a ChatGPT account." for all three on every ChatGPT Pro account we've
    # tested (verified live 2026-05-27). Keeping them in the fallback list
    # leaked dead slugs into /model when live discovery was unavailable
    # (transient API failure, first-run before refresh) and surfaced HTTP 400
    # crashes on selection. The Codex CLI public catalog still references
    # these slugs, which is why they survived previously — but those entries
    # describe the public OpenAI API, not the OAuth-backed Codex backend
    # Hermes uses. Removed here. If OpenAI re-enables them on Codex backend,
    # live discovery will pick them up automatically via _fetch_models_from_api.
]

# These slugs are valid on other OpenAI product/API surfaces but the ChatGPT
# Codex OAuth endpoint has returned HTTP 400 for them.  Keep them out of
# offline config/cache fallbacks.  This is deliberately *not* applied to a
# successful live response: if the Codex endpoint starts advertising one of
# them for an account, live provider authority immediately re-enables it.
_LIVE_ONLY_CODEX_MODELS = frozenset(
    {
        "gpt-5.6-sol-pro",
        "gpt-5.6-terra-pro",
        "gpt-5.6-luna-pro",
    }
)


def _dedupe_model_ids(
    model_ids: List[str], *, exclude_live_only: bool = False
) -> List[str]:
    """Return model IDs once, preserving provider order."""
    ordered: List[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if exclude_live_only and model_id in _LIVE_ONLY_CODEX_MODELS:
            continue
        if model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)
    return ordered


def _fetch_models_from_api(access_token: str) -> Optional[List[str]]:
    """Fetch the live catalog; ``None`` means the fetch was not authoritative."""
    try:
        import httpx
        resp = httpx.get(
            "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            return None
        entries = data["models"]
    except Exception as exc:
        logger.debug("Failed to fetch Codex models from API: %s", exc)
        return None

    sortable = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        slug = slug.strip()
        # Codex CLI's catalog uses ``supported_in_api`` for the public OpenAI
        # API, not for the OAuth-backed Codex backend that this provider uses.
        # Some valid Codex CLI models (for example gpt-5.3-codex-spark) are
        # marked false here but are still accepted by the Codex route.
        visibility = item.get("visibility", "")
        if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        sortable.append((rank, slug))

    sortable.sort(key=lambda x: (x[0], x[1]))
    # A successful live response is authoritative.  Do not synthesize models
    # that the account-specific endpoint did not advertise.
    return _dedupe_model_ids([slug for _, slug in sortable])


def _read_default_model(codex_home: Path) -> Optional[str]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib
    except Exception:
        return None
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _read_cache_models(codex_home: Path) -> List[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    sortable = []
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug.strip():
                continue
            slug = slug.strip()
            # Do not filter on ``supported_in_api`` here.  It describes the
            # public OpenAI API, while Hermes openai-codex talks to the same
            # OAuth-backed Codex backend as Codex CLI.
            visibility = item.get("visibility")
            if isinstance(visibility, str) and visibility.strip().lower() in {"hide", "hidden"}:
                continue
            priority = item.get("priority")
            rank = int(priority) if isinstance(priority, (int, float)) else 10_000
            sortable.append((rank, slug))

    sortable.sort(key=lambda item: (item[0], item[1]))
    deduped: List[str] = []
    for _, slug in sortable:
        if slug not in deduped:
            deduped.append(slug)
    return deduped


def get_codex_model_ids(access_token: Optional[str] = None) -> List[str]:
    """Return available Codex model IDs, trying API first, then local sources.
    
    Resolution order: API (live, if token provided) > config.toml default >
    local cache > hardcoded defaults.
    """
    codex_home_str = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    codex_home = Path(codex_home_str).expanduser()
    ordered: List[str] = []

    # Try live API if we have a token
    if access_token:
        api_models = _fetch_models_from_api(access_token)
        if api_models is not None:
            return _dedupe_model_ids(api_models)

    # Fall back to local sources
    default_model = _read_default_model(codex_home)
    if default_model and default_model not in _LIVE_ONLY_CODEX_MODELS:
        ordered.append(default_model)

    for model_id in _read_cache_models(codex_home):
        if model_id in _LIVE_ONLY_CODEX_MODELS:
            continue
        if model_id not in ordered:
            ordered.append(model_id)

    for model_id in DEFAULT_CODEX_MODELS:
        if model_id not in ordered:
            ordered.append(model_id)

    return _dedupe_model_ids(ordered, exclude_live_only=True)
