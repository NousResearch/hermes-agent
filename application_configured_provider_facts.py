"""Application acquisition of scoped facts for canonical provider decisions."""
from __future__ import annotations

from collections.abc import Mapping


def requested_provider(config: Mapping | None = None, *, environment: str | None = None) -> str:
    """Explicit config wins over the scoped inference-provider environment."""
    if config is None:
        from hermes_cli.config import load_config
        config = load_config()
    model = config.get("model") if isinstance(config, Mapping) else None
    chosen = str(model.get("provider") or "").strip().lower() if isinstance(model, Mapping) else ""
    if chosen:
        return chosen
    if environment is None:
        from agent.secret_scope import get_secret
        environment = get_secret("HERMES_INFERENCE_PROVIDER", "")
    return str(environment or "").strip().lower() or "auto"


def custom_identity(
    *, base_url: str = "", model: str = "", config_provider: str = "",
    config: Mapping | None = None,
) -> str:
    """Resolve durable custom identity from this profile's configured route facts."""
    from hermes_cli.config import (
        get_compatible_custom_providers, load_config, stringify_provider_map,
    )
    from providers.configured import configured_custom_identity
    from providers.route_identity import normalize_route_base_url

    cfg = config if config is not None else load_config()
    model_cfg = cfg.get("model") if isinstance(cfg, Mapping) else None
    active = (config_provider or (
        str(model_cfg.get("provider") or "") if isinstance(model_cfg, Mapping) else ""
    ))
    if not active:
        from agent.secret_scope import get_secret
        active = str(get_secret("HERMES_INFERENCE_PROVIDER", "") or "")
    identity = configured_custom_identity(
        base_url=base_url, model=model, config_provider=active,
        providers=stringify_provider_map(cfg.get("providers")),
        custom_providers=get_compatible_custom_providers(cfg),
    )
    if identity:
        return identity
    if base_url:
        # The managed server has no user-defined provider entry; prove endpoint
        # ownership instead of inferring it from localhost/port/model.
        from hermes_cli.local_runtime.endpoint import _state_endpoint

        managed = _state_endpoint()
        if managed and (
            normalize_route_base_url(base_url)
            == normalize_route_base_url(managed.get("base_url"))
        ):
            return "llamacpp"
    return ""
