"""Profile-scoped setup for the built-in Microsoft Foundry provider."""

from typing import Annotated, Literal, Optional, Self
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.azure_identity_adapter import EntraIdentityConfig, materialize_bearer_for_http
from hermes_cli.config import (
    clear_model_endpoint_credentials, get_config_path, get_env_value,
    load_config, require_readable_config_before_write, save_config, save_env_value,
)
from hermes_cli.provider_connection_probe import _endpoint_probe_client, _probe_transport_route
from hermes_cli.web_routers._common import (
    _CONFIG_MUTATION_LOCK, config_scoped_to_thread, destructive_profile,
    http_failure, is_redacted_credential_preview,
)

router = APIRouter(prefix="/api/providers/azure-foundry")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DefaultCredential(_StrictModel):
    credential_source: Literal["default"] = "default"
    scope: str = ""


class CliCredential(_StrictModel):
    credential_source: Literal["azure_cli"] = "azure_cli"
    scope: str = ""
    tenant_id: str = ""
    subscription: str = ""

    @model_validator(mode="after")
    def validate_account(self) -> Self:
        EntraIdentityConfig.from_dict(self.model_dump())
        return self


class ApiKeyAuth(_StrictModel):
    kind: Literal["api_key"] = "api_key"
    api_key: Optional[str] = None


class EntraAuth(_StrictModel):
    kind: Literal["entra_id"] = "entra_id"
    entra: Annotated[DefaultCredential | CliCredential, Field(discriminator="credential_source")]


class FoundryUpdate(_StrictModel):
    base_url: str
    model: str
    api_mode: Literal["chat_completions", "codex_responses", "anthropic_messages"]
    auth: Annotated[ApiKeyAuth | EntraAuth, Field(discriminator="kind")]


def _settings(cfg: dict) -> dict:
    model = cfg.get("model")
    current = isinstance(model, dict) and model.get("provider") == "azure-foundry"
    model = model if current else {}
    auth: dict = {"kind": "api_key"}
    if model.get("auth_mode") == "entra_id":
        entra = EntraIdentityConfig.from_dict(model.get("entra"))
        source = {"credential_source": entra.credential_source, "scope": entra.scope}
        if entra.credential_source == "azure_cli":
            source.update({key: value for key, value in (("tenant_id", entra.tenant_id),
                                                          ("subscription", entra.subscription)) if value})
        auth = {"kind": "entra_id", "entra": source}
    return {
        "base_url": model.get("base_url") or get_env_value("AZURE_FOUNDRY_BASE_URL") or "",
        "model": model.get("default") or "",
        "api_mode": model.get("api_mode") or "codex_responses",
        "auth": auth,
        "api_key_set": bool(get_env_value("AZURE_FOUNDRY_API_KEY")),
        "is_current": current,
    }


def _candidate_model(cfg: dict, body: FoundryUpdate) -> dict:
    from hermes_cli.azure_detect import normalize_foundry_base_url

    try:
        base = normalize_foundry_base_url(body.base_url, body.api_mode)
        parsed = urlsplit(base)
        valid = parsed.scheme in ("https", "http") and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        valid = False
    if not valid or parsed.query or parsed.fragment:
        raise HTTPException(400, "Enter an endpoint URL with an http or https scheme and host, without credentials or query parameters.")
    if not body.model:
        raise HTTPException(400, "Enter your Azure deployment name.")
    old_model = cfg.get("model")
    model = dict(old_model) if isinstance(old_model, dict) else {}
    clear_model_endpoint_credentials(model, clear_api_mode=False)
    model.pop("key_cmd", None)
    model.pop("transport", None)
    model.update(provider="azure-foundry", default=body.model, base_url=base,
                 api_mode=body.api_mode, auth_mode=body.auth.kind)
    if isinstance(body.auth, EntraAuth):
        previous = model.get("entra") if isinstance(model.get("entra"), dict) else {}
        entra = {key: value for key, value in body.auth.entra.model_dump().items() if value}
        # Retain the existing non-UI knob when editing a Foundry configuration.
        if isinstance(old_model, dict) and old_model.get("provider") == "azure-foundry" and "exclude_interactive_browser" in previous:
            entra["exclude_interactive_browser"] = previous["exclude_interactive_browser"]
        EntraIdentityConfig.from_dict(entra)
        model["entra"] = entra
    else:
        model.pop("entra", None)
    return model


def _api_key(auth: ApiKeyAuth) -> str:
    value = auth.api_key or get_env_value("AZURE_FOUNDRY_API_KEY") or ""
    if not value or is_redacted_credential_preview(value) or value.startswith("${"):
        raise HTTPException(400, "Enter a Foundry API key, or choose Microsoft Entra ID.")
    return value


@router.get("")
async def get_foundry(profile: Optional[str] = None):
    with http_failure("GET Foundry settings failed", 500, detail="Could not load Foundry settings."):
        return await config_scoped_to_thread(profile, lambda: _settings(load_config()))


def _save(body: FoundryUpdate) -> dict:
    with _CONFIG_MUTATION_LOCK:
        cfg = require_readable_config_before_write(get_config_path())
        model = _candidate_model(cfg, body)
        if isinstance(body.auth, ApiKeyAuth):
            key = _api_key(body.auth)
            if body.auth.api_key:
                save_env_value("AZURE_FOUNDRY_API_KEY", key)
        cfg["model"] = model
        save_config(cfg)
        return {"ok": True, **_settings(cfg)}


@router.put("")
async def save_foundry(body: FoundryUpdate, profile: Optional[str] = None):
    profile = destructive_profile(profile, "Save Foundry settings")
    with http_failure("Save Foundry settings failed", 500, detail="Could not save Foundry settings."):
        return await config_scoped_to_thread(profile, lambda: _save(body))


def _probe_runtime(body: FoundryUpdate) -> dict:
    from hermes_cli.runtime_provider_backends import _resolve_azure_foundry_runtime

    model = _candidate_model(load_config(), body)
    key = _api_key(body.auth) if isinstance(body.auth, ApiKeyAuth) else None
    runtime = _resolve_azure_foundry_runtime(requested_provider="azure-foundry", model_cfg=model,
                                            explicit_api_key=key, target_model=body.model)
    token = materialize_bearer_for_http(runtime["api_key"])
    headers = {"Authorization": f"Bearer {token}"}
    base = runtime["base_url"]
    if runtime["api_mode"] == "anthropic_messages":
        from hermes_cli.azure_detect import _AZURE_ANTHROPIC_API_VERSION

        base += "/v1"
        headers["anthropic-version"] = _AZURE_ANTHROPIC_API_VERSION
    return {"base_url": runtime["base_url"], "inference_base_url": base,
            "api_mode": runtime["api_mode"], "headers": headers}


@router.post("/validate")
async def validate_foundry(body: FoundryUpdate, profile: Optional[str] = None):
    profile = destructive_profile(profile, "Test Foundry credentials")
    try:
        runtime = await config_scoped_to_thread(profile, lambda: _probe_runtime(body))
    except HTTPException:
        raise
    except Exception:  # health: allow BLE001 -- SDK errors can embed CLI output; return a fixed hint without logging credentials
        # SDK errors may embed command output. Keep credential failures actionable without echoing it.
        cli = isinstance(body.auth, EntraAuth) and isinstance(body.auth.entra, CliCredential)
        message = ("Azure CLI could not provide a token. Install Azure CLI and run az login on the machine "
                   "running this profile; check the selected tenant and subscription." if cli else
                   "Could not obtain Foundry credentials. Check this profile's API key or Azure identity settings.")
        return {"ok": False, "reachable": True, "message": message}
    async with _endpoint_probe_client(runtime["base_url"], 10.0) as client:
        failure = await _probe_transport_route(client, runtime["inference_base_url"], runtime["api_mode"], body.model,
                                               runtime["headers"], require_success=True, bearer_only=isinstance(body.auth, EntraAuth))
    return {"ok": not failure, "reachable": True, "message": failure,
            "resolved_base_url": runtime["base_url"], "transport_checked": runtime["api_mode"]}
