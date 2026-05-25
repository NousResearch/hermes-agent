"""User-configurable LLM request header helpers.

Hermes has several client construction paths (primary OpenAI SDK clients,
Anthropic SDK clients, fallback rebuilds, and auxiliary clients). Keep config
header parsing here so those paths share the same validation and merge rules.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

# These are owned by SDK auth/protocol behavior or by provider-specific Hermes
# adapters. Allowing config to override them makes auth leaks and broken
# streaming much easier than legitimate customization.
_RESERVED_CONFIG_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "anthropic-beta",
        "anthropic-version",
        "content-length",
        "connection",
        "transfer-encoding",
        "host",
    }
)


def normalize_config_headers(raw: Any, *, source: str = "headers") -> Dict[str, str]:
    """Return a sanitized header dict from config.

    Invalid entries are skipped with a warning. Values are stringified because
    the OpenAI and Anthropic SDKs expect HTTP header values, not structured
    objects.
    """
    if raw in (None, ""):
        return {}
    if not isinstance(raw, Mapping):
        logger.warning("%s must be a mapping of header name to value; ignored", source)
        return {}

    normalized: Dict[str, str] = {}
    key_by_lower: Dict[str, str] = {}
    for raw_name, raw_value in raw.items():
        name = str(raw_name or "").strip()
        if not name:
            logger.warning("%s contains an empty header name; skipped", source)
            continue
        if "\r" in name or "\n" in name or not _HEADER_NAME_RE.match(name):
            logger.warning("%s contains invalid header name %r; skipped", source, name)
            continue
        lower_name = name.lower()
        if lower_name in _RESERVED_CONFIG_HEADERS:
            logger.warning("%s attempts to override reserved header %r; skipped", source, name)
            continue
        if raw_value is None:
            continue
        value = raw_value if isinstance(raw_value, str) else str(raw_value)
        if "\r" in value or "\n" in value:
            logger.warning("%s contains invalid value for header %r; skipped", source, name)
            continue

        previous_key = key_by_lower.get(lower_name)
        if previous_key:
            normalized.pop(previous_key, None)
        normalized[name] = value
        key_by_lower[lower_name] = name

    return normalized


def merge_default_headers(*header_maps: Any) -> Dict[str, str]:
    """Merge SDK default_headers case-insensitively while preserving casing."""
    merged: Dict[str, str] = {}
    key_by_lower: Dict[str, str] = {}
    for header_map in header_maps:
        if not isinstance(header_map, Mapping):
            continue
        for raw_name, raw_value in header_map.items():
            if raw_name is None or raw_value is None:
                continue
            name = str(raw_name)
            if not name:
                continue
            lower_name = name.lower()
            previous_key = key_by_lower.get(lower_name)
            if previous_key and previous_key != name:
                merged.pop(previous_key, None)
            merged[name] = raw_value if isinstance(raw_value, str) else str(raw_value)
            key_by_lower[lower_name] = name
    return merged


def configured_default_headers(
    *,
    provider: str = "",
    base_url: str = "",
    model: str = "",
    config: Optional[Dict[str, Any]] = None,
    include_model_headers: bool = True,
) -> Dict[str, str]:
    """Resolve user-configured headers for the current provider/base/model."""
    cfg = _load_config(config)
    if not isinstance(cfg, dict):
        return {}

    merged: Dict[str, str] = {}
    for entry in _iter_custom_provider_entries(cfg):
        if not _custom_provider_entry_matches(
            entry,
            provider=provider,
            base_url=base_url,
            model=model,
        ):
            continue
        headers = normalize_config_headers(
            entry.get("headers"),
            source=_custom_provider_header_source(entry),
        )
        if headers:
            merged = merge_default_headers(merged, headers)

    if include_model_headers:
        model_headers = _model_config_headers(
            cfg,
            provider=provider,
            base_url=base_url,
            model=model,
        )
        if model_headers:
            merged = merge_default_headers(merged, model_headers)

    return merged


def apply_configured_default_headers(
    existing_headers: Any = None,
    *,
    provider: str = "",
    base_url: str = "",
    model: str = "",
    config: Optional[Dict[str, Any]] = None,
    include_model_headers: bool = True,
) -> Dict[str, str]:
    """Merge configured headers after existing SDK/provider defaults."""
    configured = configured_default_headers(
        provider=provider,
        base_url=base_url,
        model=model,
        config=config,
        include_model_headers=include_model_headers,
    )
    return merge_default_headers(existing_headers, configured)


def apply_configured_default_headers_to_kwargs(
    client_kwargs: Dict[str, Any],
    *,
    provider: str = "",
    base_url: str = "",
    model: str = "",
    config: Optional[Dict[str, Any]] = None,
    include_model_headers: bool = True,
) -> None:
    """In-place helper for OpenAI/Anthropic SDK kwargs dicts."""
    headers = apply_configured_default_headers(
        client_kwargs.get("default_headers"),
        provider=provider,
        base_url=base_url or str(client_kwargs.get("base_url") or ""),
        model=model,
        config=config,
        include_model_headers=include_model_headers,
    )
    if headers:
        client_kwargs["default_headers"] = headers
    else:
        client_kwargs.pop("default_headers", None)


def _load_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.debug("Could not load config for request headers: %s", exc)
        return {}


def _iter_custom_provider_entries(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    try:
        from hermes_cli.config import get_compatible_custom_providers

        entries = get_compatible_custom_providers(config)
    except Exception as exc:
        logger.debug("Could not normalize custom providers for request headers: %s", exc)
        entries = []
    for entry in entries or []:
        if isinstance(entry, dict):
            yield entry


def _normalize_provider_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace(" ", "-")


def _provider_names_for_entry(entry: Dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("provider_key", "name"):
        normalized = _normalize_provider_name(entry.get(key))
        if normalized:
            names.add(normalized)
            names.add(f"custom:{normalized}")
    return names


def _normalize_model_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate:
        return ""
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme and parsed.netloc:
            candidate = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except Exception:
        pass
    return candidate.rstrip("/")


def _base_url_variants(value: Any) -> set[str]:
    normalized = _normalize_base_url(value)
    if not normalized:
        return set()
    variants = {normalized}
    if normalized.lower().endswith("/v1"):
        variants.add(normalized[:-3].rstrip("/"))
    else:
        variants.add(f"{normalized}/v1")
    return variants


def _base_urls_match(a: Any, b: Any) -> bool:
    variants_a = _base_url_variants(a)
    variants_b = _base_url_variants(b)
    return bool(variants_a and variants_b and variants_a.intersection(variants_b))


def _entry_model_matches(entry: Dict[str, Any], model: str) -> bool:
    entry_model = _normalize_model_name(entry.get("model"))
    if not entry_model:
        return True
    current_model = _normalize_model_name(model)
    return bool(current_model and entry_model == current_model)


def _custom_provider_entry_matches(
    entry: Dict[str, Any],
    *,
    provider: str,
    base_url: str,
    model: str,
) -> bool:
    if not isinstance(entry.get("headers"), Mapping):
        return False
    if not _entry_model_matches(entry, model):
        return False

    provider_norm = _normalize_provider_name(provider)
    provider_match = bool(provider_norm and provider_norm in _provider_names_for_entry(entry))
    base_match = _base_urls_match(entry.get("base_url"), base_url)

    # Prefer the concrete endpoint when available. This prevents a custom
    # provider entry that happens to share a canonical provider name from
    # leaking headers into the built-in provider's unrelated base URL.
    return base_match or (provider_match and not _normalize_base_url(base_url))


def _custom_provider_header_source(entry: Dict[str, Any]) -> str:
    name = str(entry.get("provider_key") or entry.get("name") or "?").strip() or "?"
    return f"custom_providers[{name}].headers"


def _model_config_headers(
    config: Dict[str, Any],
    *,
    provider: str,
    base_url: str,
    model: str,
) -> Dict[str, str]:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return {}
    raw_headers = model_cfg.get("headers")
    if not isinstance(raw_headers, Mapping):
        return {}

    cfg_provider = _normalize_provider_name(model_cfg.get("provider"))
    current_provider = _normalize_provider_name(provider)
    cfg_base_url = _normalize_base_url(model_cfg.get("base_url"))
    cfg_model = _normalize_model_name(model_cfg.get("default") or model_cfg.get("model"))
    current_model = _normalize_model_name(model)

    base_matches = bool(cfg_base_url and _base_urls_match(cfg_base_url, base_url))
    provider_matches = bool(
        cfg_provider
        and current_provider
        and (
            cfg_provider == current_provider
            or (cfg_provider.startswith("custom:") and current_provider == "custom")
        )
    )
    model_matches = bool(cfg_model and current_model and cfg_model == current_model)

    if cfg_provider and current_provider and not provider_matches and not base_matches:
        return {}
    if cfg_base_url and base_url and not base_matches:
        return {}
    if cfg_model and current_model and not model_matches and not (provider_matches or base_matches):
        return {}

    return normalize_config_headers(raw_headers, source="model.headers")
