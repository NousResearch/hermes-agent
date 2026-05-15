"""ACP auth helpers — detect and advertise Hermes authentication methods."""

from __future__ import annotations

from typing import Any, Optional


TERMINAL_SETUP_AUTH_METHOD_ID = "hermes-setup"


def detect_provider() -> Optional[str]:
    """Resolve the active Hermes runtime provider, or None if unavailable."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider()
        api_key = runtime.get("api_key")
        provider = runtime.get("provider")
        if isinstance(api_key, str) and api_key.strip() and isinstance(provider, str) and provider.strip():
            return provider.strip().lower()
    except Exception:
        return None
    return None


def has_provider() -> bool:
    """Return True if Hermes can resolve any runtime provider credentials."""
    return detect_provider() is not None


def build_auth_methods() -> list[Any]:
    """Return registry-compatible ACP auth methods for Hermes.

    The official ACP registry validates that agents advertise at least one
    usable auth method during the initial handshake. A fresh Zed install may
    not have Hermes provider credentials configured yet, so Hermes always
    advertises a terminal setup method. When credentials are already present,
    it also advertises the resolved provider as the default agent-managed
    runtime credential method.
    """
    from acp.schema import AuthMethodAgent, TerminalAuthMethod

    methods: list[Any] = []
    provider = detect_provider()
    if provider:
        methods.append(
            AuthMethodAgent(
                id=provider,
                name=f"{provider} runtime credentials",
                description=(
                    "Authenticate Hermes using the currently configured "
                    f"{provider} runtime credentials."
                ),
            )
        )

    methods.append(
        TerminalAuthMethod(
            id=TERMINAL_SETUP_AUTH_METHOD_ID,
            name="Configure Hermes provider",
            description=(
                "Open Hermes' interactive model/provider setup in a terminal. "
                "Use this when Hermes has not been configured on this machine yet."
            ),
            type="terminal",
            args=["--setup"],
        )
    )
    return methods
