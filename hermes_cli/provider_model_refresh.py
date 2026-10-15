"""Opt-in refresh of the same provider catalogs read by /model."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


def background_refresh_interval(config: dict) -> int:
    """Zero disables polling; an enabled interval requires an explicit allowlist."""
    section = config.get("provider_models")
    settings = section.get("background_refresh") if isinstance(section, dict) else None
    if not isinstance(settings, dict):
        return 0
    seconds = settings.get("interval_seconds", 0)
    selected = settings.get("providers", [])
    if (isinstance(seconds, bool) or not isinstance(seconds, int)
            or not 60 <= seconds <= 86400 or not isinstance(selected, list)
            or not selected or any(not isinstance(p, str) or not p.strip() for p in selected)):
        return 0
    return seconds


def refresh_configured_provider_models(config: dict) -> tuple[int, int]:
    """Probe selected endpoints, returning (attempted, failed).

    The caller binds one profile's config and secret scope. Native cache writes
    retain the last valid entry when a probe is unavailable.
    """
    if not background_refresh_interval(config):
        return 0, 0

    from agent.command_token_source import materialize_probe_api_key
    from hermes_cli.config_providers import is_provider_enabled
    from hermes_cli.model_switch import (
        _entry_models_discovered, _extra_headers_from_config, _models_config_is_allowlist,
        _scoped_key_env,
    )
    from hermes_cli.model_switch_providers import (
        _discover_flag, _entry_api_mode, _entry_base_url, _entry_credentials,
        _display_prefix, _fetch_picker_live_models,
    )
    from hermes_cli.models import (
        _custom_endpoint_fingerprint, _load_provider_models_cache, cached_fetch_api_models,
    )
    from hermes_cli.providers import custom_provider_slug

    selected = {
        name.strip().lower()
        for name in config["provider_models"]["background_refresh"]["providers"]
    }
    entries = []
    legacy = config.get("custom_providers")
    for entry in legacy if isinstance(legacy, list) else []:
        if isinstance(entry, dict) and entry.get("name"):
            entries.append((custom_provider_slug(_display_prefix(str(entry["name"]))), entry))
    providers = config.get("providers")
    for key, entry in providers.items() if isinstance(providers, dict) else []:
        if isinstance(entry, dict) and is_provider_enabled(entry):
            entries.append((custom_provider_slug("", str(key)), entry))

    seen: set[tuple[str, str]] = set()
    attempted = failed = 0
    for slug, entry in entries:
        if slug not in selected or not _discover_flag(entry):
            continue
        url = str(_entry_base_url(entry) or "").strip().rstrip("/")
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                continue
        except ValueError:
            continue
        api_key, key_env, _ = _entry_credentials(entry, "key_env", "api_key_env")
        api_key = api_key or _scoped_key_env(key_env)
        explicit = _models_config_is_allowlist(
            entry.get("models"), _entry_models_discovered(entry))
        if not api_key and explicit:
            continue  # an explicitly pinned keyless catalog is not discoverable
        mode = _entry_api_mode(entry)
        headers = _extra_headers_from_config(entry) or None
        fingerprint = _custom_endpoint_fingerprint(api_key, mode, headers)
        normalized = url.lower()
        identity = (normalized, fingerprint)
        if identity in seen:
            continue
        seen.add(identity)
        attempted += 1
        if key_env and not api_key:
            failed += 1
            continue

        cache_key = f"custom:{normalized}#{fingerprint}"
        previous = _load_provider_models_cache().get(cache_key)
        previous_at = previous.get("at", 0) if isinstance(previous, dict) else 0
        if not isinstance(previous_at, (int, float)):
            previous_at = 0
        try:
            # Preserve native Ollama discovery and command-token materialization.
            cached_fetch_api_models(
                api_key, url, headers=headers, api_mode=mode, timeout=5.0,
                force_refresh=True,
                fetch_models=lambda key=api_key, endpoint=url, wire=mode, extra=headers, pin=explicit:
                    _fetch_picker_live_models(
                        materialize_probe_api_key(key), endpoint, "custom", pin,
                        headers=extra, timeout=5.0, api_mode=wire, cache=False,
                        force_native_refresh=True),
            )
            current = _load_provider_models_cache().get(cache_key)
            if not isinstance(current, dict) or (
                not current.get("models") and not current.get("native_catalog")
            ) or not isinstance(current.get("at"), (int, float)) or current["at"] <= previous_at:
                failed += 1
        except Exception:
            failed += 1
            logger.debug("Provider model catalog refresh failed for %s", slug, exc_info=True)
    return attempted, failed
