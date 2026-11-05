"""Constructor kwargs for rebuilding a router-resolved client."""

from typing import Any

from agent.acp_client import ACPClient


def _client_kwargs_from_routed(client, timeout) -> dict[str, Any]:
    """Preserve the routed client's headers and ACP transport when rebuilding it."""
    kwargs = {"api_key": client.api_key, "base_url": str(client.base_url)}
    if timeout is not None:
        kwargs["timeout"] = timeout
    headers = (
        getattr(client, "_custom_headers", None)
        or getattr(client, "default_headers", None)
        or getattr(client, "_default_headers", None)
    )
    if headers:
        kwargs["default_headers"] = dict(headers)
    if isinstance(client, ACPClient):
        kwargs.update(command=client._acp_command, args=list(client._acp_args), acp_cwd=client._acp_cwd)
    return kwargs
