"""Azure Foundry wizard (OpenAI-style or Anthropic-style transport, API-key or Entra ID auth).

Imports of hermes_cli.config / azure_detect stay lazy (tests patch them at call time).
Prompt strings and config write order are behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hermes_cli.config import clear_model_endpoint_credentials
from hermes_cli.model_setup_flows_common import _HTTP, _ask, _commit_model_config, _load_config_model_section, _say


def _azure_mode_label(mode: str) -> str:
    return "OpenAI-style" if mode == "chat_completions" else "Anthropic-style"


@dataclass
class _AzureCurrent:
    """The active Azure Foundry settings shown as prompt defaults."""
    base_url: str = ""
    api_mode: str = ""
    auth_mode: str = "api_key"
    entra: dict = field(default_factory=dict)
    api_key: str = ""


def _azure_current(config) -> _AzureCurrent:
    from hermes_cli.config import get_env_value

    cur = _AzureCurrent(api_key=get_env_value("AZURE_FOUNDRY_API_KEY") or "")
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("provider") == "azure-foundry":
        cur.base_url = str(model_cfg.get("base_url", "") or "")
        cur.api_mode = str(model_cfg.get("api_mode", "") or "")
        cur.auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        _cur_entra = model_cfg.get("entra") or {}
        cur.entra = _cur_entra if isinstance(_cur_entra, dict) else {}
    return cur


def _azure_entra_preflight(current_entra: dict):
    """Entra ID credential preflight. Returns ``(token_provider, entra_overrides)``; ``None``
    when the user cancelled; ``False`` when the adapter is missing (fall back to API key)."""
    try:
        from agent.azure_identity_adapter import (
            EntraIdentityConfig, SCOPE_AI_AZURE_DEFAULT, build_token_provider, describe_active_credential,
            has_azure_identity_installed)
    except ImportError as exc:
        _say("", f"⚠ Could not import azure-identity adapter: {exc}", "  Falling back to API key auth.")
        return False

    print()
    if not has_azure_identity_installed():
        _say("◐ The 'azure-identity' package is not installed yet.",
             "  Hermes will install it now (the preflight below triggers the lazy-install). "
             "To skip lazy installs, run:  pip install azure-identity")

    # Only the optional scope override is persisted; identity selection (tenant,
    # user-assigned MI, workload identity, SP) stays in AZURE_* SDK env vars.
    entra_overrides: dict = {}
    _persisted_scope_override = str(current_entra.get("scope") or "").strip()
    entra_scope = _persisted_scope_override or SCOPE_AI_AZURE_DEFAULT
    if _persisted_scope_override:
        entra_overrides["scope"] = _persisted_scope_override

    _say("", "◐ Probing Microsoft Entra ID credential chain (up to 10s)...")
    _config = EntraIdentityConfig(scope=entra_scope)
    info = describe_active_credential(config=_config, timeout_seconds=10.0)
    if info.get("ok"):
        env_sources = info.get("env_sources") or []
        tag = ", ".join(env_sources) if env_sources else "default chain"
        print(f"✓ Entra ID token acquired ({tag}, scope={entra_scope})")
    else:
        err = info.get("error") or "credential chain exhausted"
        hint = info.get("hint") or (
            "Run `az login`, attach a managed identity to this VM, or set AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET."
        )
        _say(f"⚠ {err}", f"  Hint: {hint}")
        ans = _ask("Save Entra config anyway and validate later? [Y/n]: ", raw=True)
        if ans is None:
            return None
        if ans.lower() not in ("", "y", "yes"):
            print("Cancelled.")
            return None

    # Best-effort token provider for the detection probe; on failure the probe falls back
    # to manual entry.
    try:
        token_provider = build_token_provider(config=_config)
    except Exception as exc:
        print(f"⚠ Could not build token provider for probing: {exc}")
        token_provider = None
    return token_provider, entra_overrides


def _azure_pick_model(discovered_models: list, current_model: str):
    """Model/deployment step of the Azure flow; None when cancelled."""
    if not discovered_models:
        model_name = _ask(f"Model / deployment name [{current_model or 'e.g. gpt-5.4, claude-sonnet-4-6'}]: ")
        return None if model_name is None else (model_name or current_model)
    print("Available models on this endpoint:")
    for i, mid in enumerate(discovered_models[:30], start=1):
        print(f"  {i:>2}. {mid}")
    if len(discovered_models) > 30:
        print(f"  ... and {len(discovered_models) - 30} more (type name manually if not shown)")
    print()
    pick = _ask(f"Pick by number, or type a deployment name [{current_model or discovered_models[0]}]: ", raw=True)
    if pick is None:
        return None
    if not pick:
        return current_model or discovered_models[0]
    if pick.isdigit() and 1 <= int(pick) <= min(len(discovered_models), 30):
        return discovered_models[int(pick) - 1]
    return pick


def _azure_detect_transport(effective_url: str, effective_key: str, token_provider, current_api_mode: str):
    """Probe the endpoint; returns ``(api_mode, discovered_models)`` (manual pick of the API
    format when detection is incomplete) or None when cancelled."""
    from hermes_cli import azure_detect

    _say("", "◐ Probing endpoint to auto-detect transport and models...")
    detection = azure_detect.detect(effective_url, api_key=effective_key, token_provider=token_provider)
    discovered_models: list[str] = list(detection.models)
    api_mode: str = detection.api_mode or ""
    if api_mode:
        print(f"✓ Detected API transport: {_azure_mode_label(api_mode)}")
        if detection.reason:
            print(f"    ({detection.reason})")
        if discovered_models:
            print(f"✓ Found {len(discovered_models)} deployed model(s) on this endpoint")
        return api_mode, discovered_models
    _say(f"⚠ Auto-detection incomplete: {detection.reason}", "",
         "Select the API format your Azure Foundry endpoint uses:",
         "  1. OpenAI-style  (POST /v1/chat/completions)",
         "     For: GPT models, Llama, Mistral, and most open models",
         "  2. Anthropic-style  (POST /v1/messages)",
         "     For: Claude models deployed via Anthropic API format")
    default_choice = "2" if current_api_mode == "anthropic_messages" else "1"
    mode_choice = _ask(f"API format [1/2] ({default_choice}): ", raw=True)
    if mode_choice is None:
        return None
    return ("anthropic_messages" if (mode_choice or default_choice) == "2" else "chat_completions"), discovered_models


def _model_flow_azure_foundry(config, current_model=""):
    """Azure Foundry: endpoint, auth mode, API mode, model. Two transports (OpenAI-style
    ``/v1/chat/completions``, Anthropic-style ``/v1/messages``) and two auth modes: API key
    (``AZURE_FOUNDRY_API_KEY``) or Microsoft Entra ID (keyless RBAC via ``azure-identity``; the
    ``Azure AI User`` role covers both transports). Detection: ``/anthropic`` URL suffix → Anthropic;
    ``GET <base>/models`` → OpenAI-style + picker; Anthropic Messages probe; manual entry."""
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli import azure_detect

    cur = _azure_current(config)
    _say("", "Azure Foundry Configuration", "=" * 50, "",
         "Azure Foundry can host models with either OpenAI-style or",
         "Anthropic-style API endpoints.  Hermes will probe your",
         "endpoint to auto-detect the transport and the deployed",
         "models when possible.", "")
    if cur.base_url:
        print(f"  Current endpoint:  {cur.base_url}")
    if cur.api_mode:
        print(f"  Current API mode:  {_azure_mode_label(cur.api_mode)}")
    if cur.auth_mode == "entra_id":
        print("  Current auth mode: Microsoft Entra ID (keyless)")
    elif cur.api_key:
        print(f"  Current auth mode: API key ({cur.api_key[:8]}...)")
    print()

    # Step 1: endpoint URL
    _placeholder = cur.base_url or (
        "e.g. https://<resource>.openai.azure.com/openai/v1 or https://<resource>.services.ai.azure.com/anthropic")
    base_url = _ask(f"API endpoint URL [{_placeholder}]: ")
    if base_url is None:
        return
    effective_url = (base_url or cur.base_url).rstrip("/")
    if not effective_url:
        print("No endpoint URL provided. Cancelled.")
        return
    if not effective_url.startswith(_HTTP):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return

    # Step 2: authentication mode
    _say("", "Authentication:", "  1. API key                  (AZURE_FOUNDRY_API_KEY in .env)",
         "  2. Microsoft Entra ID       (managed identity / workload identity / az login)",
         "     Recommended by Microsoft. Works for both OpenAI-style and Anthropic-style endpoints.",
         "     Requires the 'Azure AI User' role on the Foundry resource.")
    _auth_default = "2" if cur.auth_mode == "entra_id" else "1"
    auth_choice = _ask(f"Authentication mode [1/2] ({_auth_default}): ", raw=True)
    if auth_choice is None:
        return
    use_entra = (auth_choice or _auth_default) == "2"

    # Step 3: credentials (key OR Entra preflight)
    effective_key: str = ""
    entra_overrides: dict = {}
    token_provider = None  # callable when entra
    if use_entra:
        preflight = _azure_entra_preflight(cur.entra)
        if preflight is None:
            return
        if preflight is False:
            use_entra = False
        else:
            token_provider, entra_overrides = preflight
    if not use_entra:
        print()
        api_key = _ask(f"API key [{cur.api_key[:8] + '...' if cur.api_key else 'required'}]: ", secret=True)
        if api_key is None:
            return
        effective_key = api_key or cur.api_key
        if not effective_key:
            print("No API key provided. Cancelled.")
            return

    # Step 4: auto-detect transport + models
    detected = _azure_detect_transport(effective_url, effective_key, token_provider, cur.api_mode)
    if detected is None:
        return
    api_mode, discovered_models = detected

    # Step 5: model name
    print()
    effective_model = _azure_pick_model(discovered_models, current_model)
    if effective_model is None:
        return
    if not effective_model:
        print("No model name provided. Cancelled.")
        return

    # Step 6: context-length lookup
    ctx_len = azure_detect.lookup_context_length(effective_model, effective_url, api_key=effective_key, token_provider=token_provider)

    # Step 7: persist
    if not use_entra:
        save_env_value("AZURE_FOUNDRY_API_KEY", effective_key)
    cfg, model = _load_config_model_section()
    model["provider"] = "azure-foundry"
    model["base_url"] = effective_url
    model["api_mode"] = api_mode
    model["default"] = effective_model
    model["auth_mode"] = "entra_id" if use_entra else "api_key"
    clear_model_endpoint_credentials(model, clear_api_mode=False)
    # Persist only a non-default Entra scope so config.yaml stays tidy.
    clean_entra = {k: v for k in ("scope",) if (v := entra_overrides.get(k))}
    if use_entra and clean_entra:
        model["entra"] = clean_entra
    else:
        model.pop("entra", None)
    if ctx_len:
        model["context_length"] = ctx_len
    _commit_model_config(cfg)
    config["model"] = dict(model)

    # Clear conflicting env vars so auxiliary clients don't pick up a stale OpenAI base URL / key.
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if get_env_value(var):
            save_env_value(var, "")

    _say("", "✓ Azure Foundry configured:", f"    Endpoint:       {effective_url}",
         f"    API mode:       {_azure_mode_label(api_mode)}",
         f"    Auth:           {'Microsoft Entra ID (keyless)' if use_entra else 'API key'}",
         f"    Model:          {effective_model}",
         f"    Context length: {ctx_len:,} tokens" if ctx_len else "    Context length: not auto-detected (will fall back at runtime)",
         "")
