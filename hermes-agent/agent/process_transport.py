"""External-process providers (DirectSDK, ACP agents) versus local inference servers.

An external-process provider's ``process://<name>`` base URL has no dots, so ``is_local_endpoint``
reads it as an unqualified LAN host. It is a hosted model behind a child process: there is no shared
server slot to wait out and no local server to probe (#129153). It does keep a local server's
patience, because the child is silent until the model's first token. Every rule here is keyed on the
provider's ``auth_type``, never on the URL.
"""

from __future__ import annotations

from typing import Any


def is_local_server(base_url: Any, provider: Any) -> bool:
    """A local or LAN HTTP server: one request slot, Ollama/llama.cpp probes, local-server copy."""
    from agent.model_metadata import is_local_endpoint
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    return bool(base_url) and is_local_endpoint(str(base_url)) and not _is_external_process_provider(str(provider or ""))


def waits_like_local(base_url: Any, provider: Any) -> bool:
    """Silent for minutes before the first token, so the stream detectors use the local ceiling."""
    from agent.model_metadata import is_local_endpoint
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    return _is_external_process_provider(str(provider or "")) or (bool(base_url) and is_local_endpoint(str(base_url)))


def implicit_non_stream_stale_timeout(agent: Any, uses_implicit_default: bool) -> float | None:
    """The buffered-call stall budget when the cloud formula does not apply, else None.

    A local server with nothing configured gets no kill. An external process with nothing configured
    gets the request timeout its client is already given, for every model: reasoning floors (240s for
    opus) are for HTTP gateways that idle-kill (#149555). The bound stays finite because a client
    blocked writing a prompt to a child that stopped reading never reaches its own deadline (ACP).
    """
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    if _is_external_process_provider(str(agent.provider or "")):
        if agent._stale_timeout_is_explicit():
            return None
        from agent.chat_completion_helpers import cap_to_run_budget

        return cap_to_run_budget(agent, agent._resolved_api_call_timeout())
    if uses_implicit_default and is_local_server(getattr(agent, "_base_url", None) or agent.base_url, agent.provider):
        return float("inf")
    return None
