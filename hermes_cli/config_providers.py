"""Custom-provider entry normalization and per-route lookups (TLS, headers, context length, capabilities).

Split out of ``hermes_cli/config.py``; every name is re-imported there, so
``hermes_cli.config.<name>`` keeps resolving (and monkeypatching) as before. Functions that call
``get_compatible_custom_providers`` / ``load_config*`` import them lazily from ``hermes_cli.config``
so tests patching that module still intercept the call.
"""

import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from hermes_cli.route_identity import normalize_route_base_url

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.config")


# ``_normalize_custom_provider_entry`` runs on every ``load_picker_context()`` call, so a warning
# it emits would fire repeatedly for the same static config; on Windows that storm contends on
# ``concurrent-log-handler``'s cross-process rotation lock and can stall the gateway/serve event
# loop. Deduplicate per (provider, signature) for the process lifetime.
_PROVIDER_NORMALIZE_WARNED: set = set()


def _warn_once_per_provider(provider_key: str, signature: str, msg: str, *args: Any) -> None:
    """Emit ``logger.warning(msg, *args)`` at most once per (provider, signature)."""
    dedup_key = (provider_key or "?", signature)
    if dedup_key in _PROVIDER_NORMALIZE_WARNED:
        return
    _PROVIDER_NORMALIZE_WARNED.add(dedup_key)
    logger.warning(msg, *args)


# Values accepted by earlier releases (and natural spellings) → canonical transport names consumed
# by agent_init. Without this map an unrecognized api_mode was silently ignored and the transport
# fell through to hostname-based guessing, so ``api_mode: openai`` could flip to
# ``codex_responses`` after an update and break the provider.
_API_MODE_ALIASES = {
    # See #66543.
    "openai": "chat_completions",
    "openai_chat": "chat_completions",
    "openai-chat": "chat_completions",
    "chat-completions": "chat_completions",
    "chatcompletions": "chat_completions",
    "responses": "codex_responses",
    "openai_responses": "codex_responses",
    "openai-responses": "codex_responses",
    "anthropic": "anthropic_messages",
    "anthropic-messages": "anthropic_messages",
    "messages": "anthropic_messages",
    "bedrock": "bedrock_converse",
    "bedrock-converse": "bedrock_converse"}

_FALSE_WORDS = frozenset({"false", "0", "no", "off"})
_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})


def _canonical_api_mode(api_mode: str) -> str:
    """Map alias ``api_mode`` spellings to canonical transport names (unknown pass through)."""
    cleaned = api_mode.strip()
    return _API_MODE_ALIASES.get(cleaned.lower(), cleaned)


def coerce_provider_id(value: Any) -> str:
    """Provider identity fields are strings."""
    if value is None:
        return ""
    return str(value).strip()


def stringify_provider_map(providers: Any) -> dict:
    """Copy a ``providers:`` mapping so keys are strings (unquoted YAML ``2070:`` loads as int)."""
    if not isinstance(providers, dict):
        return {}
    out: Dict[str, Any] = {}
    for stored, value in providers.items():
        key = coerce_provider_id(stored)
        if key:
            out[key] = value
    return out


def find_provider_entry(providers: Any, key: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Return ``(stored_key, entry)`` matching *key* by string identity (exact hit, then scan)."""
    if not isinstance(providers, dict):
        return None, None
    want = coerce_provider_id(key)
    if not want:
        return None, None
    exact = providers.get(want)
    if isinstance(exact, dict):
        return want, exact
    for stored, entry in providers.items():
        if coerce_provider_id(stored) == want and isinstance(entry, dict):
            return stored, entry
    return None, None


# camelCase aliases commonly used in hand-written provider configs.
_CAMEL_ALIASES: Dict[str, str] = {
    "apiKey": "api_key",
    "baseUrl": "base_url",
    "apiMode": "api_mode",
    "keyEnv": "key_env",
    "apiKeyEnv": "key_env",  # OpenClaw-compatible + docs variant
    "defaultModel": "default_model",
    "contextLength": "context_length",
    "rateLimitDelay": "rate_limit_delay"}


_KNOWN_PROVIDER_KEYS = {
    # ``provider`` duplicates the ``providers.<name>`` mapping key and is unused here, but Hermes'
    # own config writer has historically emitted it. Accept it so self-written configs don't warn.
    "provider",
    "name", "api", "url", "base_url", "api_key", "key_env", "api_key_env", "key_cmd",
    "api_mode", "transport", "model", "default_model", "models", "models_discovered",
    "context_length", "rate_limit_delay", "request_timeout_seconds", "stale_timeout_seconds",
    "discover_models", "extra_body", "extra_headers", "capabilities", "ssl_ca_cert", "ssl_verify"}


def _pick_provider_base_url(entry: Dict[str, Any], provider_key: str) -> str:
    """First usable URL among ``base_url``/``url``/``api``, or "".

    URLs with unresolved ``${ENV_VAR}`` / ``{region}`` placeholders are accepted unvalidated: they
    expand at runtime, and rejecting them here would silently drop the provider.
    """
    for url_key in ("base_url", "url", "api"):
        raw_url = entry.get(url_key)
        if not (isinstance(raw_url, str) and raw_url.strip()):
            continue
        candidate = raw_url.strip()
        # Accept URLs containing unresolved placeholder tokens — both ``${ENV_VAR}`` env-refs and bare
        # ``{region}``-style templates — without URL validation. They are expanded at runtime, so a caller
        # reaching this normalizer with raw (un-expanded) config would otherwise see the provider silently
        # dropped (#14457).
        if re.search(r"\{[^}]+\}", candidate):
            return candidate
        parsed = urlparse(candidate)
        if parsed.scheme and parsed.netloc:
            return candidate
        logger.warning(
            "providers.%s: '%s' value '%s' is not a valid URL "
            "(no scheme or host) — skipped",
            provider_key or "?", url_key, candidate)
    return ""


def _normalize_provider_models(models: Any) -> Tuple[Dict[str, Any], bool]:
    """Normalize an entry's ``models`` to ``(models_dict, discovered_flag)``.

    The legacy in-mapping ``__discovered_model_catalog__`` sentinel is accepted and stripped; a
    plain list of ids or ``[{id: ...}]`` rows is converted so /model doesn't show (0) models.
    """
    discovered = False
    if isinstance(models, dict) and models:
        # Shallow-copy: `models` may alias a cached config sub-dict, and the normalized entry
        # escapes into long-lived runtime state.
        models_copy = dict(models)
        if models_copy.pop("__discovered_model_catalog__", None) is True:
            discovered = True
        models_copy.pop("__explicit_model_allowlist__", None)
        return models_copy, discovered
    if isinstance(models, list) and models:
        normalized_models: Dict[str, Any] = {}
        for item in models:
            if isinstance(item, str) and item.strip():
                normalized_models[item.strip()] = {}
                continue
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                model_id = item.get("name")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            normalized_models[model_id.strip()] = {
                k: v for k, v in item.items() if k not in {"id", "name"}}
        return normalized_models, discovered
    return {}, discovered


def _normalize_custom_provider_entry(
    entry: Any, *, provider_key: str = "") -> Optional[Dict[str, Any]]:
    """Return a runtime-compatible custom provider entry or ``None``."""
    if not isinstance(entry, dict):
        return None
    # Shallow-copy before alias normalization writes into the entry: callers pass live sub-dicts
    # from load_config_readonly()'s shared cache; mutating those violates its no-mutation contract
    # and leaks alias keys back into config.yaml on a later save_config(load_config()).
    entry = dict(entry)
    provider_key = coerce_provider_id(provider_key)
    # api_key_env is a documented snake_case alias for key_env (azure-foundry guide).
    if "api_key_env" in entry and "key_env" not in entry:
        entry["key_env"] = entry["api_key_env"]
    for camel, snake in _CAMEL_ALIASES.items():
        if camel in entry and snake not in entry:
            _warn_once_per_provider(
                provider_key, f"camel:{camel}",
                "providers.%s: camelCase key '%s' auto-mapped to '%s' "
                "(use snake_case to avoid this warning)",
                provider_key or "?", camel, snake)
            entry[snake] = entry[camel]
    unknown = set(entry.keys()) - _KNOWN_PROVIDER_KEYS - set(_CAMEL_ALIASES.keys())
    if unknown:
        _warn_once_per_provider(
            provider_key, "unknown:" + ",".join(sorted(unknown)),
            "providers.%s: unknown config keys ignored: %s",
            provider_key or "?", ", ".join(sorted(unknown)))

    base_url = _pick_provider_base_url(entry, provider_key)
    name = coerce_provider_id(entry.get("name")) or provider_key
    if not base_url or not name:
        return None
    normalized: Dict[str, Any] = {"name": name, "base_url": base_url}
    if provider_key:
        normalized["provider_key"] = provider_key

    def _stripped(*keys: str) -> str:
        val = None
        for k in keys:
            val = entry.get(k)
            if val:
                break
        return val.strip() if isinstance(val, str) else ""

    def _put(field: str, value: Any) -> None:
        if value:
            normalized[field] = value

    _put("api_key", _stripped("api_key"))
    key_env = _stripped("key_env", "api_key_env")
    _put("key_env", key_env)
    if key_env and entry.get("api_key_env") and not entry.get("key_env"):
        normalized["api_key_env"] = key_env
    api_mode = _stripped("api_mode", "transport")
    _put("api_mode", _canonical_api_mode(api_mode) if api_mode else "")
    _put("model", _stripped("model", "default_model"))

    # ``models_discovered`` marks a mapping auto-discovered by Hermes, not hand-curated.
    models_dict, discovered = _normalize_provider_models(entry.get("models"))
    _put("models", models_dict)
    if entry.get("models_discovered") is True or discovered:
        normalized["models_discovered"] = True

    capabilities = entry.get("capabilities")
    if isinstance(capabilities, dict):
        _put("capabilities", {
            key: value for key, value in capabilities.items()
            if isinstance(key, str) and isinstance(value, bool)})

    for field, ok in (
        ("context_length", lambda v: isinstance(v, int) and v > 0),
        ("rate_limit_delay", lambda v: isinstance(v, (int, float)) and v >= 0),
        ("discover_models", lambda v: isinstance(v, bool)),
    ):
        if ok(entry.get(field)):
            normalized[field] = entry[field]
    if isinstance(entry.get("extra_body"), dict):
        normalized["extra_body"] = dict(entry["extra_body"])

    # Per-provider extra HTTP headers may carry credentials — never log them downstream.
    _put("extra_headers", normalize_extra_headers(entry.get("extra_headers")))
    _put("ssl_ca_cert", _stripped("ssl_ca_cert"))

    ssl_verify = entry.get("ssl_verify")
    if isinstance(ssl_verify, bool):
        normalized["ssl_verify"] = ssl_verify
    elif isinstance(ssl_verify, str) and ssl_verify.strip():
        normalized["ssl_verify"] = ssl_verify.strip()

    return normalized


def _custom_provider_entry_to_provider_config(
    entry: Any, *, provider_key: str = "") -> Optional[Dict[str, Any]]:
    """Translate a legacy custom provider entry to the v12 providers shape."""
    normalized = _normalize_custom_provider_entry(entry, provider_key=provider_key)
    if normalized is None:
        return None

    provider_entry: Dict[str, Any] = {"api": normalized["base_url"]}
    for field in (
        "name", "api_key", "key_env", "models", "models_discovered", "context_length",
        "rate_limit_delay", "discover_models", "extra_body", "extra_headers",
        "ssl_ca_cert", "ssl_verify"):
        if field in normalized:
            provider_entry[field] = normalized[field]
    if "model" in normalized:
        provider_entry["default_model"] = normalized["model"]
    if "api_mode" in normalized:
        provider_entry["transport"] = normalized["api_mode"]
    return provider_entry


def providers_dict_to_custom_providers(providers_dict: Any) -> List[Dict[str, Any]]:
    """Normalize enabled ``providers`` config entries into the legacy custom-provider shape."""
    if not isinstance(providers_dict, dict):
        return []
    custom_providers: List[Dict[str, Any]] = []
    for key, entry in providers_dict.items():
        if isinstance(entry, dict) and not is_provider_enabled(entry):
            continue
        normalized = _normalize_custom_provider_entry(entry, provider_key=coerce_provider_id(key))
        if normalized is not None:
            custom_providers.append(normalized)
    return custom_providers


def get_compatible_custom_providers(
    config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Deduplicated list view over legacy ``custom_providers`` and v12+ ``providers``.

    Never materialised back into config.yaml (it would duplicate entries in UIs).
    """
    from hermes_cli.config import load_config
    if config is None:
        config = load_config()

    custom_providers = config.get("custom_providers")
    if custom_providers is not None and not isinstance(custom_providers, list):
        return []
    candidates = [_normalize_custom_provider_entry(e) for e in (custom_providers or [])]
    candidates += providers_dict_to_custom_providers(config.get("providers"))

    def _norm(entry: Dict[str, Any], field: str) -> str:
        return str(entry.get(field, "") or "").strip().lower()

    compatible: List[Dict[str, Any]] = []
    seen_provider_keys: set = set()
    seen_name_url_pairs: set = set()
    for entry in candidates:
        if entry is None:
            continue
        provider_key = _norm(entry, "provider_key")
        name = _norm(entry, "name")
        base_url = str(entry.get("base_url", "") or "").strip().rstrip("/").lower()
        pair = (name, base_url, _norm(entry, "model"))
        if provider_key and provider_key in seen_provider_keys:
            continue
        if name and base_url and pair in seen_name_url_pairs:
            continue
        compatible.append(entry)
        if provider_key:
            seen_provider_keys.add(provider_key)
        if name and base_url:
            seen_name_url_pairs.add(pair)
    return compatible


def _entries_for_route(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]]):
    """Yield entries whose normalized route identity equals *base_url*.

    None *custom_providers* → ``get_compatible_custom_providers(config)`` (failure → none).
    """
    from hermes_cli.config import get_compatible_custom_providers
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            custom_providers = []
    if not base_url or not isinstance(custom_providers, list):
        return
    target_url = normalize_route_base_url(base_url)
    if not target_url:
        return
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        entry_url = normalize_route_base_url(entry.get("base_url"))
        if entry_url and entry_url == target_url:
            yield entry


def _route_model_cfg(entry: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """Return ``entry.models[model]`` when both are mappings, else None."""
    models = entry.get("models")
    if not isinstance(models, dict):
        return None
    model_cfg = models.get(model)
    return model_cfg if isinstance(model_cfg, dict) else None


def _route_model_cfgs(
    model: str,
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield the ``models.<model>`` mapping of every entry matching *base_url*."""
    for entry in _entries_for_route(base_url, custom_providers, config):
        model_cfg = _route_model_cfg(entry, model)
        if model_cfg is not None:
            yield model_cfg


def _coerce_ssl_verify(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        return False if lowered in _FALSE_WORDS else True if lowered in _TRUE_WORDS else None
    return None


def get_custom_provider_tls_settings(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return TLS settings from a matching ``custom_providers`` / ``providers`` entry."""
    for entry in _entries_for_route(base_url, custom_providers, config):
        out: Dict[str, Any] = {}
        ca = entry.get("ssl_ca_cert")
        if isinstance(ca, str) and ca.strip():
            out["ssl_ca_cert"] = ca.strip()
        verify = _coerce_ssl_verify(entry.get("ssl_verify"))
        if verify is not None:
            out["ssl_verify"] = verify
        return out
    return {}


def apply_custom_provider_tls_to_client_kwargs(
    client_kwargs: Dict[str, Any],
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None) -> None:
    """Attach per-provider TLS knobs to OpenAI client kwargs when matched."""
    tls = get_custom_provider_tls_settings(base_url, custom_providers, config)
    if tls.get("ssl_ca_cert"):
        client_kwargs["ssl_ca_cert"] = tls["ssl_ca_cert"]
    if "ssl_verify" in tls:
        client_kwargs["ssl_verify"] = tls["ssl_verify"]


def normalize_extra_headers(extra_headers: Any) -> Dict[str, str]:
    """Normalize a raw ``extra_headers`` value into a ``dict[str, str]``.

    SECURITY: header values routinely carry credentials (Cloudflare Access service tokens, proxy
    auth, custom bearer schemes). Callers must never log the returned values.
    """
    if not isinstance(extra_headers, dict) or not extra_headers:
        return {}
    return {str(k): str(v) for k, v in extra_headers.items() if v is not None}


def get_custom_provider_extra_headers(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """``extra_headers`` of the first route-matching entry declaring any, else ``{}``.
    SECURITY: values may carry credentials — callers must never log them."""
    for entry in _entries_for_route(base_url, custom_providers, config):
        headers = normalize_extra_headers(entry.get("extra_headers"))
        if headers:
            return headers
    return {}


def apply_custom_provider_extra_headers_to_client_kwargs(
    client_kwargs: Dict[str, Any],
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None) -> None:
    """Merge per-provider ``extra_headers`` onto OpenAI client ``default_headers`` (provider wins
    over SDK defaults, most specific level). SECURITY: values may carry credentials; never log."""
    extra_headers = get_custom_provider_extra_headers(base_url, custom_providers, config)
    if not extra_headers:
        return
    merged = dict(client_kwargs.get("default_headers") or {})
    merged.update(extra_headers)
    client_kwargs["default_headers"] = merged


def get_custom_provider_context_length(
    model: str,
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Per-model ``context_length`` override from a route-matching entry, or ``None``.

    Before this helper existed, the lookup was duplicated in ``run_agent.py``'s startup path only; every
    other path (notably ``/model`` switch) fell back to the 128K default. See #15779.
    """
    from hermes_cli.config import get_compatible_custom_providers
    if not model or not base_url:
        return None
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            if config is None:
                return None
            raw = config.get("custom_providers")
            custom_providers = raw if isinstance(raw, list) else []

    for model_cfg in _route_model_cfgs(model, base_url, custom_providers, config):
        raw_ctx = model_cfg.get("context_length")
        if raw_ctx is None:
            continue
        try:
            ctx = int(raw_ctx)
        except (TypeError, ValueError):
            continue
        if ctx > 0:
            return ctx
    return None


def get_custom_provider_model_capability(
    model: str,
    base_url: str,
    capability: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None) -> Optional[bool]:
    """Explicit boolean capability for one custom-provider model, or ``None``. Scoped to the
    normalized route + exact runtime model id so aliases can declare capabilities."""
    from hermes_cli.config import get_compatible_custom_providers, load_config_readonly
    if not model or not base_url or not capability:
        return None
    if custom_providers is None:
        try:
            if config is None:
                # Read-only path: entries are never mutated and get_compatible_custom_providers
                # shallow-copies each one, so the no-deepcopy cache is safe (~135us saved per call).
                config = load_config_readonly()
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            return None

    for model_cfg in _route_model_cfgs(model, base_url, custom_providers, config):
        value = model_cfg.get(capability)
        if isinstance(value, bool):
            return value
    return None


def is_provider_enabled(provider_cfg: Optional[Dict[str, Any]]) -> bool:
    """Whether a ``providers.<name>`` block is enabled: default True; only an explicit
    ``enabled: false`` hides it from the picker, ``/models``, runtime resolver and doctor."""
    if not isinstance(provider_cfg, dict):
        return True
    flag = provider_cfg.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    # YAML can produce strings for "true"/"false" depending on quoting.
    if isinstance(flag, str):
        return flag.strip().lower() not in _FALSE_WORDS
    return bool(flag)
