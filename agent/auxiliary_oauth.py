"""Keep a main session's OAuth-proxy capability scoped to its auxiliary route."""

from hermes_cli.route_identity import normalize_route_base_url


def runtime_oauth_proxy(main_runtime, provider: str, base_url: str) -> bool | None:
    """Inherit OAuth semantics only for the main provider at the same endpoint."""
    if not isinstance(main_runtime, dict):
        return None
    capabilities = main_runtime.get("capabilities")
    if not isinstance(capabilities, dict) or not isinstance(
        capabilities.get("anthropic_oauth_proxy"), bool
    ):
        return None
    runtime_base = main_runtime.get("base_url")
    if not base_url or not runtime_base:
        return None
    if normalize_route_base_url(base_url) != normalize_route_base_url(runtime_base):
        return None
    target = str(provider or "").lower().removeprefix("custom:")
    source = (
        str(
            main_runtime.get("requested_provider") or main_runtime.get("provider") or ""
        )
        .lower()
        .removeprefix("custom:")
    )
    if target == source or target in {"auto", "main", "custom"}:
        return capabilities["anthropic_oauth_proxy"]
    return None
