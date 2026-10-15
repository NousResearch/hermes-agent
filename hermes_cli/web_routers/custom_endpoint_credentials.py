"""Credential selection for Desktop custom endpoints, using the runtime's key_cmd contract."""

from typing import Any, Literal

from fastapi import HTTPException

from hermes_cli.web_models import CustomEndpointUpdate

CredentialSource = Literal["api_key", "command", "azure_cli"]
# Project only UTC epoch expiry. Azure's older expiresOn is in the host's local timezone.
AZURE_CLI_KEY_CMD = (
    'az account get-access-token --scope https://ai.azure.com/.default '
    '--query "{access_token:accessToken,expires_on:expires_on}" --output json --only-show-errors'
)


def credential_source(entry: dict[str, Any]) -> CredentialSource:
    command = str(entry.get("key_cmd") or "").strip()
    if command == AZURE_CLI_KEY_CMD:
        return "azure_cli"
    return "command" if command else "api_key"


def select_credentials(entry: dict[str, Any], body: CustomEndpointUpdate) -> dict[str, Any]:
    """Apply an explicit auth choice; old clients leave unrepresented fields untouched."""
    selected = dict(entry)
    source = body.credential_source
    command = body.key_cmd.strip() if body.key_cmd is not None else None
    if source is None and command is None:
        return selected
    if source == "api_key":
        if command:
            raise HTTPException(400, "Choose either an API key or a credential command.")
        selected.pop("key_cmd", None)
        return selected
    if body.api_key and body.api_key.strip():
        raise HTTPException(400, "Choose either an API key or a credential command.")
    if source == "azure_cli":
        command = AZURE_CLI_KEY_CMD
    else:
        command = command or str(entry.get("key_cmd") or "").strip()
    if not command:
        raise HTTPException(400, "Enter a credential command first.")
    selected["key_cmd"] = command
    for key in ("api_key", "key_env", "api_key_env"):
        selected.pop(key, None)
    return selected


def probe_headers(body: CustomEndpointUpdate) -> dict[str, str]:
    """Resolve and mint inside the caller's profile scope, off the event loop.

    Test uses the same named-provider resolver as chat. Never swallow a helper failure and
    retry anonymously; helper stdout, stderr and command text may all contain credentials.
    """
    from agent.command_token_source import CommandTokenError
    from hermes_cli.config import load_config
    from hermes_cli.web_routers.config_env import _pop_legacy_custom_provider, _resolve_custom_endpoint_entry
    from hermes_cli.runtime_provider_custom import _resolve_named_custom_runtime

    cfg = load_config()
    providers = cfg.get("providers")
    providers = dict(providers) if isinstance(providers, dict) else {}
    stored_key, existing = _resolve_custom_endpoint_entry(providers, body.id or body.name)
    if existing is None:
        existing = _pop_legacy_custom_provider(cfg, body.id or body.name)
    endpoint_id = stored_key or body.id or body.name or "desktop-endpoint-probe"
    entry = select_credentials(existing or {}, body)
    entry.update(name=body.name, base_url=body.base_url, model=body.model)
    if body.api_key is not None:
        entry["api_key"] = body.api_key.strip()
        entry.pop("key_env", None)
        entry.pop("api_key_env", None)
    providers[endpoint_id] = entry
    cfg["providers"] = providers
    runtime = _resolve_named_custom_runtime(
        requested_provider=f"custom:{endpoint_id}", explicit_api_key=body.api_key,
        target_model=body.model, config=cfg,
    )
    if credential_source(entry) == "azure_cli":
        from hermes_platform.resolver import LookupContext, locate_command
        from tools.environments.local import served_profile_child_env

        child_env = served_profile_child_env(inherit_credentials=True)
        context = LookupContext(path=child_env.get("PATH"), pathext=child_env.get("PATHEXT"))
        if not locate_command("az", context).command:
            raise CommandTokenError(
                "Azure CLI (az) was not found on the machine running this profile. "
                "Install it there and restart Hermes so it can find az on PATH."
            )
    key = runtime["api_key"] if runtime else ""
    try:
        token = key() if callable(key) else key
    except CommandTokenError as exc:
        if credential_source(entry) == "azure_cli":
            raise CommandTokenError(
                "Azure CLI could not provide a token. Install Azure CLI and run az login "
                "on the machine running this profile, then test again."
            ) from exc
        raise
    import httpx

    headers = httpx.Headers({"Accept": "application/json"})
    if isinstance(token, str) and token and token != "no-key-required":
        headers["Authorization"] = f"Bearer {token}"
    if runtime:
        # Match the SDK: provider headers win, including differently cased Authorization.
        headers.update(runtime.get("extra_headers") or {})
    return {key.title(): value for key, value in headers.items()}
