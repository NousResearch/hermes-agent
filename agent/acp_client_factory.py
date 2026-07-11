"""Factory helpers for ACP subprocess-backed chat clients."""

from __future__ import annotations

from typing import Any

ACP_PROVIDERS = frozenset({"copilot-acp", "devin-acp"})


def is_acp_provider(provider: str | None = None, base_url: str | None = None) -> bool:
    """Return True when *provider* or *base_url* points at an ACP subprocess backend."""
    p = (provider or "").strip().lower()
    if p in ACP_PROVIDERS:
        return True
    url = (base_url or "").strip().lower()
    return url.startswith("acp://") or url.startswith("acp+tcp://")


def create_acp_client(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate the correct ACP client for *provider* / *base_url*."""
    p = (provider or "").strip().lower()
    url = (base_url or kwargs.get("base_url") or "").strip().lower()
    if p == "devin-acp" or url.startswith("acp://devin"):
        from agent.devin_acp_client import DevinACPClient

        return DevinACPClient(base_url=base_url, **kwargs)

    from agent.copilot_acp_client import CopilotACPClient

    return CopilotACPClient(base_url=base_url, **kwargs)
