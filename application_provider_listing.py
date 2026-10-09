"""Profile-scoped provider list for application presentation, from canonical registry."""
from __future__ import annotations

from typing import Any, Mapping


def _scoped_env_key(name: str) -> str:
    from agent.secret_scope import current_secret_scope, get_secret, is_multiplex_active
    if current_secret_scope() is not None or is_multiplex_active():
        return str(get_secret(name, "") or "").strip()
    from agent.credential_pool import get_env_prefer_dotenv
    return str(get_env_prefer_dotenv(name) or "").strip()


def _has_credentials(provider: str, config: Mapping[str, Any]) -> bool:
    try:
        from hermes_cli.auth import get_auth_status, has_usable_secret
        if provider == "custom":
            model = config.get("model")
            return bool(str(model.get("base_url") or "").strip()) if isinstance(model, Mapping) else False
        if provider == "openrouter":
            return has_usable_secret(_scoped_env_key("OPENROUTER_API_KEY"))
        status = get_auth_status(provider)
        return bool(status.get("logged_in") or status.get("configured"))
    except Exception:
        return False


def list_available_providers(config: Mapping[str, Any] | None = None) -> list[dict]:
    """Preserve the existing config.get provider payload without a shadow registry."""
    if config is None:
        from hermes_cli.config import load_config
        config = load_config()
    from hermes_cli.provider_catalog import provider_catalog
    from providers import get_provider_profile
    return [
        {
            "id": descriptor.slug,
            "label": descriptor.label,
            "aliases": list(getattr(get_provider_profile(descriptor.slug), "aliases", ()) or ()),
            "authenticated": _has_credentials(descriptor.slug, config or {}),
        }
        for descriptor in provider_catalog()
    ]
