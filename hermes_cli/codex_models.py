"""Codex model discovery from API, local cache, and config."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from models.codex_catalog import (
    DEFAULT_CODEX_MODELS,
    dedupe_model_ids,
    drop_undiscovered_astra,
    finalize_codex_models,
)

def codex_catalog_credential_identity() -> str:
    """Identity of the credential live discovery would use right now, for the catalog cache key.

    Access/refresh tokens rotate in place while the account-scoped catalog stays authoritative for
    the same ChatGPT principal, so the key is ``(chatgpt_account_id, sub)``, not the token. An
    expired token is its own state: ``_codex_catalog`` serves the static fallback for it, and that
    fallback must not outlive the refresh under the healthy principal's key. Opaque non-JWT tokens
    fall back to the token itself (the caller hashes every part before anything is persisted).
    """
    from hermes_cli.auth import _codex_access_token_is_expiring, resolve_codex_runtime_credentials

    try:
        token = str(resolve_codex_runtime_credentials(read_only=True).get("api_key") or "")
    except Exception:  # AuthError (no/exhausted creds) or the pytest seat belt: no live catalog either way
        token = ""
    if not token:
        return "missing"
    if _codex_access_token_is_expiring(token, 0):
        return "expired"
    from agent.credential_pool import _codex_principal_identity

    principal = _codex_principal_identity(token)
    return "/".join(principal) if principal else token


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
    return dedupe_model_ids(slug for _, slug in sortable)


def _fetch_models_from_api(access_token: str, base_url: Optional[str] = None) -> List[str]:
    """Fetch available models from the Codex API. Returns visible models sorted by priority.

    ``base_url`` is the host the credential is routed to (resolved together with it); the
    catalog is fetched there, never from a host the credential does not belong to (#121486).
    """
    try:
        from agent.model_metadata import install_context_metadata_hooks
        from models.metadata.context import _codex_catalog_probe_allowed
        from hermes_cli.auth_codex import _codex_base_url

        install_context_metadata_hooks()
        catalog_base = (base_url or "").strip().rstrip("/") or _codex_base_url()
        if not _codex_catalog_probe_allowed(access_token, catalog_base):
            return []
        import httpx
        # The per-account catalog needs ChatGPT-Account-ID (else ``{"models":[]}`` with HTTP 200
        # masquerades as "no models") and, for residency-enforced workspaces, the residency header.
        from agent.codex_headers import codex_account_headers
        headers = {"Authorization": f"Bearer {access_token}", **codex_account_headers(access_token)}
        from models.metadata.context import fetch_codex_catalog_entries
        entries, _status = fetch_codex_catalog_entries(
            lambda url: httpx.get(url, headers=headers, timeout=10), base_url=catalog_base)
    except Exception as exc:
        logger.debug("Failed to fetch Codex models from API: %s", exc)
        return []

    return finalize_codex_models(_ranked_slugs(entries))


def _read_default_model(codex_home: Path) -> Optional[str]:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib
        payload = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _read_cache_models(codex_home: Path) -> List[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []

    entries = raw.get("models") if isinstance(raw, dict) else None
    return _ranked_slugs(entries if isinstance(entries, list) else [])


def get_codex_model_ids(access_token: Optional[str] = None, base_url: Optional[str] = None) -> List[str]:
    """Available Codex model IDs: live API (if token) > config.toml default > local cache > defaults.

    Pass the ``base_url`` resolved together with ``access_token`` (runtime/pool route) so live
    discovery asks the credential's own host."""
    codex_home = Path(os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")).expanduser()
    if access_token:
        api_models = _fetch_models_from_api(access_token, base_url=base_url)
        if api_models:
            return finalize_codex_models(api_models)
    default_model = _read_default_model(codex_home)
    return finalize_codex_models(drop_undiscovered_astra(dedupe_model_ids([
        *([default_model] if default_model else []), *_read_cache_models(codex_home),
        *DEFAULT_CODEX_MODELS])))
