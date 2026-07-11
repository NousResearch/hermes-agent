"""Factory helpers for ACP subprocess-backed chat clients."""

from __future__ import annotations

from typing import Any

ACP_PROVIDERS = frozenset({"copilot-acp", "devin-acp", "grok-acp"})


def is_acp_provider(provider: str | None = None, base_url: str | None = None) -> bool:
    """Return True when *provider* or *base_url* points at an ACP subprocess backend."""
    p = (provider or "").strip().lower()
    if p in ACP_PROVIDERS:
        return True
    url = (base_url or "").strip().lower()
    return url.startswith("acp://") or url.startswith("acp+tcp://")


def _fill_missing_acp_invocation(provider: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fill empty command/args from the external-process credential resolver.

    Call sites sometimes construct an ACP client with only provider/base_url.
    When that happens, prefer the shared auth resolver (env + PATH) over each
    client's private defaults — keeps command/args consistent with
    ``resolve_runtime_provider`` / ``hermes status``.
    """
    command = kwargs.get("command") or kwargs.get("acp_command")
    raw_args = kwargs.get("args")
    if raw_args is None:
        raw_args = kwargs.get("acp_args")
    has_args = isinstance(raw_args, (list, tuple)) and len(raw_args) > 0
    if command and has_args:
        return kwargs

    p = (provider or "").strip().lower()
    if p not in ACP_PROVIDERS:
        return kwargs

    try:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        creds = resolve_external_process_provider_credentials(p)
    except Exception:
        return kwargs

    filled = dict(kwargs)
    if not command:
        resolved = str(creds.get("command") or "").strip()
        if resolved:
            filled["command"] = resolved
    if not has_args:
        cred_args = list(creds.get("args") or [])
        if cred_args:
            filled["args"] = cred_args
    return filled


def create_acp_client(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate the correct ACP client for *provider* / *base_url*."""
    p = (provider or "").strip().lower()
    url = (base_url or kwargs.get("base_url") or "").strip().lower()
    client_kwargs = _fill_missing_acp_invocation(p, kwargs)

    if p == "devin-acp" or url.startswith("acp://devin"):
        from agent.devin_acp_client import DevinACPClient

        return DevinACPClient(base_url=base_url, **client_kwargs)

    if p == "grok-acp" or url.startswith("acp://grok"):
        from agent.grok_acp_client import GrokACPClient

        return GrokACPClient(base_url=base_url, **client_kwargs)

    from agent.copilot_acp_client import CopilotACPClient

    return CopilotACPClient(base_url=base_url, **client_kwargs)
