"""Refresh transport leases for conversations attached to the managed local router."""

from typing import Any

from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES, _state_endpoint


def _identity(agent: Any) -> tuple[str, str, str]:
    return (
        str(getattr(agent, "requested_provider", "") or "").lower(),
        str(getattr(agent, "base_url", "") or "").rstrip("/"),
        str(getattr(agent, "api_key", "") or ""),
    )


def remember_managed_endpoint(agent: Any) -> None:
    """Remember ownership at attachment, before the endpoint can move.

    A custom URL, a named provider, or an externally detected server is not a
    managed lease. Matching the initial endpoint and key keeps those routes pinned.
    """
    agent._managed_local_endpoint = None
    alias, base, key = _identity(agent)
    if alias not in LLAMACPP_ALIASES or agent.api_mode != "chat_completions":
        return
    from hermes_cli.runtime_provider import _get_named_custom_provider

    if _get_named_custom_provider(alias):
        return
    endpoint = _state_endpoint()
    if endpoint and (base, key) == (
        endpoint["base_url"].rstrip("/"),
        endpoint["api_key"],
    ):
        agent._managed_local_endpoint = (alias, base, key)


def refresh_managed_endpoint(agent: Any, api_error: Exception | None = None) -> bool:
    """Rebind a moved managed lease at turn start or before connection-error backoff."""
    lease = getattr(agent, "_managed_local_endpoint", None)
    if lease is None or lease != _identity(agent):
        return False
    if api_error is not None:
        from openai import APIConnectionError

        if not isinstance(api_error, APIConnectionError):
            return False
    endpoint = _state_endpoint()
    if not endpoint:
        return False
    base, key = endpoint["base_url"].rstrip("/"), endpoint["api_key"]
    if (base, key) == lease[1:]:
        return False
    kwargs = {**agent._client_kwargs, "base_url": base, "api_key": key}
    client = agent._create_openai_client(
        kwargs, reason="managed_endpoint_refresh", shared=True
    )
    if agent.client is not None:
        agent._retire_shared_openai_client(
            agent.client, reason="managed_endpoint_refresh"
        )
    agent.client, agent.base_url, agent.api_key = client, base, key
    agent._client_kwargs = kwargs
    if hasattr(agent, "_transport_cache"):
        agent._transport_cache.clear()
    # Only transport changed: keep the prompt, token accounting and compaction policy.
    compressor = agent.context_compressor
    if str(getattr(compressor, "base_url", "") or "").rstrip("/") == lease[1]:
        compressor.base_url, compressor.api_key = base, key
    primary = agent._primary_runtime
    if str(primary.get("base_url") or "").rstrip("/") == lease[1]:
        primary.update(base_url=base, api_key=key, client_kwargs=dict(kwargs))
        if str(primary.get("compressor_base_url") or "").rstrip("/") == lease[1]:
            primary.update(compressor_base_url=base, compressor_api_key=key)
    agent._managed_local_endpoint = (lease[0], base, key)
    return True
