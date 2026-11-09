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

