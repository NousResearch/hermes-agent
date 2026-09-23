"""ACP auth helpers — detect and advertise Hermes authentication methods."""

from __future__ import annotations

from typing import Any, Optional


TERMINAL_SETUP_AUTH_METHOD_ID = "hermes-setup"


def detect_provider() -> Optional[str]:
    """Resolve the active Hermes runtime provider, or None if unavailable.

    A callable ``api_key`` (Azure Foundry Entra ID bearer-token provider, see
    :mod:`agent.azure_identity_adapter`) counts as a valid credential; otherwise
    Entra-configured Foundry deployments would default to ``"openrouter"`` and
    the ACP auth handshake would reject the legitimate provider."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider()
        api_key, provider = runtime.get("api_key"), runtime.get("provider")
        if isinstance(provider, str) and provider.strip() and (
                (isinstance(api_key, str) and api_key.strip()) or callable(api_key)):
            return provider.strip().lower()
    except Exception:
        pass
    return None


def build_auth_methods() -> list[Any]:
    """Return registry-compatible ACP auth methods for Hermes.

    The ACP registry requires at least one usable auth method in the initial
    handshake. A fresh Zed install may have no Hermes credentials yet, so the
    terminal setup method is always advertised; when credentials resolve, the
    provider is also advertised as the default agent-managed runtime method."""
    from acp.schema import AuthMethodAgent, TerminalAuthMethod

    methods: list[Any] = []
    provider = detect_provider()
    if provider:
        methods.append(AuthMethodAgent(
            id=provider, name=f"{provider} runtime credentials",
            description=f"Authenticate Hermes using the currently configured {provider} runtime credentials.",
        ))
    methods.append(TerminalAuthMethod(
        id=TERMINAL_SETUP_AUTH_METHOD_ID, name="Configure Hermes provider", type="terminal", args=["--setup"],
        description=("Open Hermes' interactive model/provider setup in a terminal. "
                     "Use this when Hermes has not been configured on this machine yet."),
    ))
    return methods


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def has_provider() -> bool:
    """Return True if Hermes can resolve any runtime provider credentials."""
    return detect_provider() is not None
# ---- END PLUGIN-COMPAT ----
