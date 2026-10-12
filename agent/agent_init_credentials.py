"""Credential display for ``agent_init`` client construction."""


def _print_key_banner(key, label: str, warn_missing: bool = False) -> None:
    """Masked credential line. ``key`` may be a callable Entra ID bearer provider (Azure
    Foundry) — never invoke or inspect it. Keys ≤ 12 chars (incl. "dummy-key") are not shown."""
    from agent.azure_identity_adapter import is_token_provider
    if is_token_provider(key):
        print("🔑 Using credentials: Microsoft Entra ID")
    elif isinstance(key, str) and len(key) > 12:
        print(f"🔑 Using {label}: {key[:8]}...{key[-4:]}")
    elif warn_missing:
        print("⚠️  Warning: API key appears invalid or missing")
