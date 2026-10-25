from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _configured_timeout(provider_id: str, model: str | None, model_key: str, provider_key: str) -> float | None:
    """Per-model ``providers.<id>.models.<model>.<model_key>`` wins over ``providers.<id>.<provider_key>``."""
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    if not isinstance(provider_config, dict):
        return None
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get(model_key))
        if timeout is not None:
            return timeout
    return _coerce_timeout(provider_config.get(provider_key))


def get_provider_request_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "timeout_seconds", "request_timeout_seconds")


def get_provider_stale_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds")


def get_provider_follow_redirects(
    provider_id: str,
    model: str | None = None,
    *,
    base_url: str = "",
    requested_provider: str = "",
) -> bool | None:
    """Explicit redirect policy for the active route; unset/invalid leaves SDK defaults alone.

    The current URL wins over a stale requested custom identity after rotation. An
    explicit model override, including an invalid value, never inherits a provider opt-in.
    """
    from hermes_cli.config import get_compatible_custom_providers, load_config_readonly
    from hermes_cli.config_read_errors import FailedConfigRead
    from hermes_cli.config_providers import (
        _coerce_ssl_verify, _normalize_custom_provider_entry, _normalize_provider_models, find_provider_entry,
    )
    from hermes_cli.providers import custom_provider_aliases
    from hermes_cli.route_identity import normalize_route_base_url
    from agent.auxiliary_client import _to_openai_base_url
    from urllib.parse import urlsplit, urlunsplit

    def route_normalize(url: str) -> str:
        # SDK clients move query parameters into default_query. Normalize both sides
        # before the shared SDK path rewrite (host casing/default ports included).
        try:
            parsed = urlsplit(url)
        except ValueError:
            return ""
        clean = urlunsplit(parsed._replace(query="", fragment=""))
        return normalize_route_base_url(_to_openai_base_url(normalize_route_base_url(clean)))

    if not provider_id:
        return None
    config = load_config_readonly()
    if not isinstance(config, dict) or isinstance(config, FailedConfigRead):
        return None
    identity = (requested_provider if provider_id == "custom" else provider_id).strip().lower()
    # The picker view deduplicates display-name/URL pairs, not durable identities.
    # Read the selected key from the original mapping before consulting that view.
    providers = config.get("providers")
    _, provider_config = find_provider_entry(providers, identity)
    if provider_config is None and isinstance(providers, dict):
        provider_config = next((entry for key, entry in providers.items()
                                if isinstance(entry, dict) and identity in custom_provider_aliases("", key)), None)
    active_route = route_normalize(base_url) if base_url else ""
    if provider_config is not None and base_url:
        normalized = _normalize_custom_provider_entry(provider_config, provider_key=identity)
        if normalized is None or not active_route or route_normalize(normalized["base_url"]) != active_route:
            provider_config = None
    entries = get_compatible_custom_providers(config)
    route_entries = [entry for entry in entries if active_route and
                     route_normalize(entry.get("base_url", "")) == active_route]
    if provider_config is None:
        matches = [entry for entry in route_entries if identity in custom_provider_aliases(
            entry.get("name", ""), entry.get("provider_key", ""))]
        provider_config = matches[0] if matches else route_entries[0] if len(route_entries) == 1 else None
    if provider_config is None:
        # Ambiguous or unknown rotated custom routes must not inherit another route's opt-in.
        if base_url and (route_entries or provider_id == "custom" or provider_id.startswith("custom:")):
            return None
        _, provider_config = find_provider_entry(config.get("providers"), identity)
        if provider_config is None:
            identity = identity.strip().lower()
            provider_config = next((entry for entry in entries if identity in custom_provider_aliases(
                entry.get("name", ""), entry.get("provider_key", ""))), None)
    if provider_config is None:
        return None
    models, _ = _normalize_provider_models(provider_config.get("models"))
    model_config = models.get(model) if model else None
    if isinstance(model_config, dict) and "follow_redirects" in model_config:
        return _coerce_ssl_verify(model_config["follow_redirects"])
    return _coerce_ssl_verify(provider_config.get("follow_redirects"))


def _get_model_config(provider_config: dict[str, object], model: str | None) -> dict[str, object] | None:
    if not model:
        return None
    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    return model_config if isinstance(model_config, dict) else None
