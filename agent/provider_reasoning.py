"""Bind catalog capabilities to the actual client route, outside request construction."""
from urllib.parse import urlparse


def prepare_client_reasoning(client, *, provider=None):
    """Called on client initialization and before inference (also handles credential rotation)."""
    if client is None:
        return None
    from providers import get_provider_profile
    base_url = str(getattr(client, "base_url", "") or "")
    # The route, not a Grok model name, determines which account's catalog is authoritative.
    name = "xai" if urlparse(base_url).hostname == "api.x.ai" else provider
    profile = get_provider_profile(name) if name else None
    if profile is None:
        return None
    credential = getattr(client, "api_key", None)
    # The SDK stores callable credentials separately and leaves api_key empty/stale.
    source = vars(client).get("_api_key_provider")
    if name == "xai" and callable(source):
        credential = source()
    bound = profile.prepare_reasoning_catalog(api_key=credential, base_url=base_url)
    client._hermes_reasoning_profile = bound
    return bound


def xai_auxiliary_reasoning_fields(client, model, extra_body, warning_state):
    """Cache-only: both auxiliary modes use the same policy as primary Responses requests."""
    from agent.transports.codex import _resolve_reasoning, _reasoning_fields
    config = extra_body.get("reasoning") if isinstance(extra_body, dict) else None
    if not isinstance(config, dict):
        return {}
    params = {
        "is_xai_responses": True, "reasoning_config": config,
        "reasoning_profile": getattr(client, "_hermes_reasoning_profile", None),
        "reasoning_warning_state": warning_state,
    }
    effort, enabled = _resolve_reasoning(model, params)
    return _reasoning_fields(model, params, effort=effort, enabled=enabled,
        replay_encrypted_reasoning=True, is_xai_responses=True, is_github_responses=False)
