"""Explicit custom catalogue save. Never invoked by read-only inventory."""
from __future__ import annotations
import logging
from typing import Optional
from application_provider_discovery import _entry_api_mode, _entry_credentials
logger=logging.getLogger(__name__)

def _save_discovered_models_to_config(
    api_url: str, model_ids: list[str], *, api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None, credential_identity: str | None = None) -> None:
    """Persist a successful ``/v1/models`` probe into the matching ``custom_providers`` entry.

    Matches by base_url (slash-normalised), api_mode and headers. A failed config write is
    swallowed — the picker still shows the live models for this session."""
    from application_provider_secret_inputs import extra_headers_from_config as _extra_headers_from_config
    if not api_url or not model_ids:
        return
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        providers = cfg.get("custom_providers") or []
        if not isinstance(providers, list):
            return

        norm_url = api_url.strip().rstrip("/").lower()
        changed = False
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            entry_url = (entry.get("base_url", "") or entry.get("url", "")).strip()
            if entry_url.rstrip("/").lower() != norm_url or _entry_api_mode(entry) != api_mode:
                continue
            if headers is not None and _extra_headers_from_config(entry) != headers:
                continue
            if credential_identity is not None and _entry_credentials(entry, "key_env", "api_key_env")[2] != credential_identity:
                continue
            if not _discovered_catalog_stale(entry, model_ids):
                continue
            entry["models"] = {model_id: {} for model_id in model_ids}
            entry["models_discovered"] = True
            changed = True

        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
    except Exception:
        pass


def _discovered_catalog_stale(entry: dict, model_ids: list[str]) -> bool:
    """Whether a live probe may overwrite ``entry["models"]``.

    A ``models`` mapping or list of dicts is user-curated per-model metadata — never replaced.
    A mapping Hermes itself discovered (entry flag or legacy in-mapping sentinel) is ours to
    refresh, but only when stale; a legacy-shape entry is always rewritten so the save migrates
    it to the clean entry-level flag."""
    existing = entry.get("models")
    legacy_discovered = isinstance(existing, dict) and existing.get("__discovered_model_catalog__") is True
    entry_discovered = entry.get("models_discovered") is True or legacy_discovered
    if isinstance(existing, dict):
        return entry_discovered and (legacy_discovered or list(existing) != model_ids)
    if isinstance(existing, list):
        return not any(isinstance(m, dict) for m in existing) and existing != model_ids
    return True
