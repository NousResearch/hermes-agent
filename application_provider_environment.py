"""Profile-scoped acquisition of non-secret, explicitly declared endpoint facts."""
from __future__ import annotations

import os

from providers.environment import declared_endpoint_override
from providers.identity import normalize_provider
from providers.registry import get_provider_profile


def scoped_endpoint_override(
    provider: str, *, base_url_env_var: str = "",
    explicit: str = "", config: dict | None = None,
) -> str:
    """Configured endpoint > declared profile environment > no override.

    Only the declaration (base_url_env_var) marks an environment variable
    as an endpoint; credential env_vars are never inspected here.
    """
    from hermes_cli.config import get_env_value_prefer_dotenv, load_config

    profile = get_provider_profile(provider)
    var = str(getattr(profile, "base_url_env_var", "") or base_url_env_var)
    if config is None:
        config = load_config()
    model = config.get("model") if isinstance(config, dict) else None
    configured = ""
    if isinstance(model, dict) and normalize_provider(model.get("provider") or "") == normalize_provider(provider):
        configured = str(model.get("base_url") or "")
    scoped = ""
    if var:
        from agent.secret_scope import current_secret_scope, get_secret, is_multiplex_active
        if current_secret_scope() is not None or is_multiplex_active():
            scoped = str(get_secret(var, "") or "")
        else:
            scoped = str(get_env_value_prefer_dotenv(var) or os.environ.get(var) or "")
    return declared_endpoint_override(
        base_url_env_var=var, explicit=explicit, configured=configured,
        environment={var: scoped} if var else {},
    )
