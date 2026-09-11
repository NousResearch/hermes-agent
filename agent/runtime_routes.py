"""Preserve resolved route data across client and agent ownership boundaries."""

from collections.abc import Mapping

from agent.runtime_bundle import ResolvedRuntime, mutable_config_copy


def record_client_runtime(client, **fields):
    """Attach construction data for later fallback adoption, never an owned HTTP client."""
    fields.pop("http_client", None)
    client._hermes_resolved_runtime = ResolvedRuntime.from_mapping(fields)
    return client


def runtime_for_endpoint(runtime, base_url):
    """Keep same-route settings; resolve only destination settings after an endpoint change."""
    from hermes_cli.route_identity import normalize_route_base_url

    raw = mutable_config_copy(runtime) if isinstance(runtime, Mapping) else {}
    if raw and normalize_route_base_url(raw.get("base_url")) == normalize_route_base_url(base_url):
        return ResolvedRuntime.from_mapping(raw).with_updates(base_url=base_url or "")
    for key in ("extra_headers", "default_headers", "default_query", "ssl_ca_cert", "ssl_verify", "client_kwargs"):
        raw.pop(key, None)
    from agent.agent_init import _host_default_headers_factory
    from providers import get_provider_profile
    build_headers = _host_default_headers_factory(base_url or "")
    profile = get_provider_profile(raw.get("provider", ""))
    if build_headers is not None:
        raw["default_headers"] = build_headers(raw.get("api_key", ""), base_url or "")
    elif profile and profile.default_headers:
        raw["default_headers"] = dict(profile.default_headers)
    from hermes_cli.config import (
        apply_custom_provider_tls_to_client_kwargs, get_compatible_custom_providers,
        get_custom_provider_extra_headers, load_config_readonly,
    )
    cfg = load_config_readonly()
    entries = get_compatible_custom_providers(cfg)
    apply_custom_provider_tls_to_client_kwargs(raw, base_url or "", entries, config=cfg)
    raw["extra_headers"] = get_custom_provider_extra_headers(base_url or "", entries, config=cfg)
    raw["base_url"] = base_url or ""
    return ResolvedRuntime.from_mapping(raw)


def runtime_from_client(client, *, provider, model, api_mode, base_url):
    """Recover the exact builder snapshot; legacy SDK clients expose a smaller transport contract."""
    wire = getattr(client, "_real_client", None)
    resolved = getattr(client, "_hermes_resolved_runtime", None)
    if not isinstance(resolved, ResolvedRuntime):
        resolved = getattr(wire, "_hermes_resolved_runtime", None)
    raw = resolved.as_dict() if isinstance(resolved, ResolvedRuntime) else {}
    for key in ("default_query", "timeout", "ssl_ca_cert", "ssl_verify"):
        value = getattr(client, key, None)
        if key not in raw and isinstance(value, (Mapping, str, int, float, bool)):
            raw[key] = mutable_config_copy(value)
    if "default_headers" not in raw:
        headers = getattr(client, "_custom_headers", None)
        if isinstance(headers, Mapping):
            raw["default_headers"] = mutable_config_copy(headers)
    headers = getattr(client, "_hermes_runtime_extra_headers", None)
    if isinstance(headers, Mapping) and headers:
        raw["extra_headers"] = mutable_config_copy(headers)
    raw.update(provider=provider, model=model, api_mode=api_mode, base_url=base_url,
               api_key=getattr(client, "api_key", raw.get("api_key", "")))
    raw.setdefault("requested_provider", provider)
    return ResolvedRuntime.from_mapping(raw)


def build_resolved_agent_client(agent, runtime):
    """Build a fresh owned client from a resolved input, without resolving ambient credentials."""
    from agent.runtime_bundle import build_client_bundle
    runtime = ResolvedRuntime.from_mapping(runtime).with_updates(
        provider=agent.provider, model=agent.model, api_mode=agent.api_mode,
        requested_provider=agent.requested_provider,
    )
    bundle = build_client_bundle(
        runtime,
        openai_builder=lambda kwargs: agent._create_openai_client(
            kwargs, reason="agent_init", shared=True, runtime=runtime,
        ),
    )
    agent.install_runtime(bundle, reason="agent_init")
