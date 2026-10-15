"""Manual fallback runtime projection for the gateway; no credentials persist here."""
import re


def _runtime_kwargs_for_agent_constructor(runtime: dict) -> dict:
    """Remove gateway-only metadata; credentials stay in this ephemeral projection."""
    return {k: v for k, v in runtime.items() if k not in {"manual_fallback_index", "has_header_auth"}}


def _resolve_fallback_entry_agent_kwargs(entry: dict, *, config: dict) -> tuple[str, dict]:
    """Validate an explicit selection using native config/auth/pool precedence.

    This is a runtime projection, not a credential snapshot or a billing boundary.
    """
    from urllib.parse import urlparse
    import httpx
    from agent.azure_identity_adapter import is_token_provider
    from hermes_cli.auth import has_usable_secret
    from hermes_cli.fallback_config import resolve_entry_api_key
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider, _parse_api_mode, _loopback_hostname

    from gateway.run import _runtime_agent_kwargs, _deep_merge_request_overrides

    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    if not provider or not model or re.search(r"\$\{[^}]*\}", provider + model):
        raise ValueError("Fallback entry needs resolved provider and model")
    key = resolve_entry_api_key(entry, strict=True)
    resolved = resolve_runtime_provider(
        requested=provider, target_model=model, explicit_base_url=entry.get("base_url"),
        explicit_api_key=key, config=config,
    )
    runtime = _runtime_agent_kwargs(resolved)
    runtime["capabilities"] = {**(resolved.get("capabilities") or {}), **(entry.get("capabilities") or {})}
    runtime["max_tokens"] = entry.get("max_output_tokens", resolved.get("max_output_tokens"))
    runtime["request_overrides"] = _deep_merge_request_overrides(
        resolved.get("request_overrides"), entry.get("request_overrides"),
    )
    headers = resolved.get("extra_headers") or {}
    if any(re.search(r"\$\{[^}]*\}", str(value)) for value in headers.values()):
        raise ValueError("Unresolved fallback header authentication")
    runtime["has_header_auth"] = bool(headers)
    # Preserve the named entry, including an empty header block: URL-only lookup
    # can select a different account when two providers share one endpoint.
    if provider.startswith("custom:"):
        runtime["extra_headers"] = dict(headers)
    if "api_mode" in entry:
        mode = _parse_api_mode(entry["api_mode"])
        if mode is None:
            raise ValueError("Invalid fallback api_mode")
        runtime["api_mode"] = mode
    actual = str(runtime.get("provider") or "").strip()
    mode = runtime.get("api_mode")
    if not actual:
        raise ValueError("Fallback provider is unavailable")
    native = {"bedrock_converse", "codex_app_server"}
    if mode in native and mode != resolved.get("api_mode"):
        raise ValueError("Incompatible fallback transport")
    if not runtime.get("command") and actual != "moa" and mode not in native:
        url = str(runtime.get("base_url") or "")
        parsed = urlparse(url)
        _ = parsed.port
        # urlparse accepts spaces in host/path; the HTTP client does not route
        # those endpoints as configured (and may silently percent-encode them).
        if (not url or any(char.isspace() for char in url)
                or re.search(r"\$\{[^}]*\}", url) or parsed.scheme not in {"http", "https"}
                or not parsed.hostname or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Invalid fallback endpoint")
        httpx.URL(url)
        key = runtime.get("api_key")
        usable = has_usable_secret(key) and not re.search(r"\$\{[^}]*\}", str(key))
        if key == "no-key-required":
            # Preserve native local-noauth eligibility; never accept that sentinel remotely.
            usable = _loopback_hostname(parsed.hostname)
        if not (usable or is_token_provider(key) or runtime["has_header_auth"]):
            raise ValueError("Fallback authentication is unavailable")
    runtime["fallback_service_tier_override"] = entry.get("service_tier_override")
    return normalize_model_for_provider(model, provider), runtime


def _manual_fallback_runtime(index) -> tuple[str, dict]:
    """Resolve the current durable index from fresh effective user config, never last-good cache."""
    from gateway.run import _gateway_config_home
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.fallback_config import get_fallback_chain
    try:
        cfg = load_user_config_effective(_gateway_config_home() / "config.yaml", fail_closed=True)
        chain = get_fallback_chain(cfg)
        if type(index) is not int or not 0 <= index < len(chain):
            raise ValueError("Invalid manual selection")
        model, runtime = _resolve_fallback_entry_agent_kwargs(chain[index], config=cfg)
        runtime.update(manual_fallback_index=index, fallback_model=chain[index + 1:])
        return model, runtime
    # health: allow BLE001 -- provider errors may contain credentials; preserve the redacted failure boundary
    except Exception:
        raise RuntimeError(
            "Manual fallback is unavailable. Fix its configuration or use /fallback off or /new."
        ) from None
