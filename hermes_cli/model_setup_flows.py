"""Per-provider model-selection wizard flows for ``hermes setup`` / ``hermes model``.

main / config / auth / models helpers are imported lazily inside bodies: avoids the main.py import
cycle and lets tests patch ``hermes_cli.config.load_config`` etc. at call time. The shared skeleton
lives in :mod:`hermes_cli.model_setup_flows_common`; the custom / Azure / Bedrock flows live in their
own ``model_setup_flows_*`` modules.
"""

from __future__ import annotations

import contextlib
import argparse
import os

from hermes_cli.config import clear_model_endpoint_credentials
from hermes_cli.model_setup_flows_common import (
    _HTTP, _activate_provider_model, _ask, _commit_model_config, _curses_choice,
    _ensure_dict_section, _ensure_flow_api_key, _finish_model,
    _load_config_model_section, _models_dev_merged, _oauth_gate, _persist_model, _pick_model_or_prompt,
    _print_numbered, _prompt_auth_credentials_choice,
    _run_login, _say, _show_curated)
from hermes_cli.model_setup_flows_custom import _model_flow_custom, _model_flow_named_custom
from hermes_cli.model_setup_flows_azure import _model_flow_azure_foundry
from hermes_cli.model_setup_flows_bedrock import _model_flow_bedrock


def _env_base_url(base_url_env: str) -> str:
    """Base-URL override from ``.env`` then the process environment ('' when unset)."""
    from hermes_cli.config import get_env_value
    if not base_url_env:
        return ""
    return get_env_value(base_url_env) or os.getenv(base_url_env, "")


def _prompt_base_url_override(effective_base: str, base_url_env: str) -> str:
    """Optional ``Base URL [...]`` prompt; a valid override is saved to *base_url_env*."""
    from hermes_cli.config import save_env_value
    override = _ask(f"Base URL [{effective_base}]: ", cancel_msg="", on_cancel="")
    if override and base_url_env:
        if not override.startswith(_HTTP):
            print("  Invalid URL — must start with http:// or https://. Keeping current value.")
        else:
            save_env_value(base_url_env, override)
            return override
    return effective_base


def _report_live_models(model_list, source: str) -> None:
    if model_list:
        print(f"  Found {len(model_list)} model(s) from {source}")


def _model_flow_openrouter(config, current_model=""):
    """OpenRouter provider: ensure API key, then pick model."""
    from hermes_constants import OPENROUTER_BASE_URL
    from hermes_cli.auth import ProviderConfig, _prompt_model_selection

    # OpenRouter isn't in PROVIDER_REGISTRY so we synthesize a minimal pconfig.
    pconfig = ProviderConfig(id="openrouter", name="OpenRouter", auth_type="api_key", api_key_env_vars=("OPENROUTER_API_KEY",))
    existing_key, _resolved, abort = _ensure_flow_api_key(
        "openrouter", pconfig, missing_hint=("Get one at: https://openrouter.ai/keys", ""))
    if abort:
        return

    from hermes_cli.models import model_ids
    from hermes_cli.models_pricing import get_pricing_for_provider
    openrouter_models = model_ids(force_refresh=True)
    # Live pricing is non-blocking — empty dict on failure.
    pricing = get_pricing_for_provider("openrouter", force_refresh=True)
    selected = _prompt_model_selection(
        openrouter_models, current_model=current_model, pricing=pricing, confirm_provider="openrouter",
        confirm_base_url=OPENROUTER_BASE_URL, confirm_api_key=_resolved or existing_key)
    _finish_model(selected, "openrouter", f"Default model set to: {selected} (via OpenRouter)",
                  base_url=OPENROUTER_BASE_URL, api_mode="chat_completions")


def _model_flow_ai_gateway(config, current_model=""):
    """Vercel AI Gateway provider: ensure API key, then pick model with pricing."""
    from hermes_constants import AI_GATEWAY_BASE_URL
    from hermes_cli.main_provider_setup import _prompt_api_key
    from hermes_cli.auth import PROVIDER_REGISTRY, _prompt_model_selection
    from hermes_cli.config import get_env_value
    pconfig = PROVIDER_REGISTRY["ai-gateway"]
    existing_key = get_env_value("AI_GATEWAY_API_KEY") or ""
    if not existing_key:
        _say("Create API key here: https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai-gateway&title=AI+Gateway",
             "Add a payment method to get $5 in free credits.", "")
    _resolved, abort = _prompt_api_key(pconfig, existing_key, provider_id="ai-gateway")
    if abort:
        return

    from hermes_cli.models import ai_gateway_model_ids
    from hermes_cli.models_pricing import get_pricing_for_provider
    models_list = ai_gateway_model_ids(force_refresh=True)
    pricing = get_pricing_for_provider("ai-gateway", force_refresh=True)
    selected = _prompt_model_selection(models_list, current_model=current_model, pricing=pricing)
    # Inline credentials are deliberately left untouched here (historical behavior).
    _finish_model(selected, "ai-gateway", f"Default model set to: {selected} (via Vercel AI Gateway)",
                  base_url=AI_GATEWAY_BASE_URL, api_mode="chat_completions", clear_creds=False)


def _model_flow_moa(config, current_model=""):
    """Mixture of Agents virtual provider: pick a preset (list always shown, even with one entry),
    persist it, print the breakdown. No credential step — presets reference configured providers."""
    from hermes_cli.auth import _save_model_choice
    from hermes_cli.moa_config import normalize_moa_config
    moa = normalize_moa_config(config.get("moa") if isinstance(config, dict) else {})
    presets = moa.get("presets") or {}
    if not presets:
        print("No MoA presets configured. Run `hermes moa configure <name>` first.")
        return

    names = list(presets.keys())
    default_name = moa.get("default_preset") or names[0]
    # Rows show the aggregator so the picker is informative before drilling in.
    rows = []
    for n in names:
        agg = presets[n].get("aggregator") or {}
        agg_label = f"{agg.get('provider')}:{agg.get('model')}" if agg else ""
        ref_count = len(presets[n].get("reference_models") or [])
        suffix = "  ← default" if n == default_name else ""
        rows.append(f"{n}  (agg {agg_label}, {ref_count} refs){suffix}")
    default_idx = names.index(default_name) if default_name in names else 0

    title = "Select a Mixture of Agents preset:"
    idx = _curses_choice(title, rows, default_idx)
    if idx is None:
        _print_numbered(title, rows, default_idx)
        raw = _ask(f"  Choice [1-{len(rows)}]: ", raw=True, cancel_msg="No change.")
        if raw is None:
            return
        try:
            idx = default_idx if not raw else max(0, min(len(rows) - 1, int(raw) - 1))
        except ValueError:
            print("No change.")
            return
    if idx < 0:
        print("No change.")
        return

    selected_name = names[idx]
    cfg, model = _load_config_model_section()
    model["default"] = selected_name
    model["provider"] = "moa"
    # Virtual local provider: drop stale endpoint credentials AND base_url (which
    # clear_model_endpoint_credentials intentionally leaves alone).
    clear_model_endpoint_credentials(model, clear_api_mode=True)
    model.pop("base_url", None)
    _commit_model_config(cfg)
    _save_model_choice(selected_name)

    preset = presets[selected_name]
    _say("", f"Default model set to: {selected_name} (via Mixture of Agents)", f"  Preset: {selected_name}", "  Reference models:")
    for i, slot in enumerate(preset.get("reference_models") or [], start=1):
        print(f"    {i}. {slot.get('provider')}:{slot.get('model')}")
    agg = preset.get("aggregator") or {}
    print(f"  Aggregator:  {agg.get('provider')}:{agg.get('model')}")


def _nous_login_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        portal_url=getattr(args, "portal_url", None), inference_url=getattr(args, "inference_url", None),
        client_id=getattr(args, "client_id", None), scope=getattr(args, "scope", None),
        no_browser=bool(getattr(args, "no_browser", False)), timeout=getattr(args, "timeout", None) or 15.0,
        ca_bundle=getattr(args, "ca_bundle", None), insecure=bool(getattr(args, "insecure", False)))


def _nous_model_catalog(free_tier: bool, portal_url: str, model_ids: list, pricing: dict):
    """Free/paid-tier catalog for the Nous picker: ``(model_ids, pricing, unavailable_models,
    unavailable_message, policy_narrowed)`` or None (message already printed) when nothing is selectable."""
    from hermes_cli.models_pricing import nous_policy_allowed_ids, restrict_to_nous_policy
    from hermes_cli.models import (
        partition_nous_models_by_tier,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )

    # Free users: union with the Portal's freeRecommendedModels (newly launched free models appear
    # before the curated list catches up), then partition selectable/unavailable by Portal pricing.
    # Paid users: paidRecommendedModels, no partition. Org policy narrows BEFORE the tier split so a
    # rescued id still has to pass the free/paid predicate.
    unavailable_models: list[str] = []
    unavailable_message = ""
    _policy_allowed = nous_policy_allowed_ids()
    if free_tier:
        try:
            from hermes_cli.nous_account import format_nous_portal_entitlement_message, get_nous_portal_account_info
            _account_info = get_nous_portal_account_info(force_fresh=True)
            unavailable_message = format_nous_portal_entitlement_message(_account_info, capability="paid Nous models") or ""
        except Exception:
            unavailable_message = ""
        model_ids, pricing = union_with_portal_free_recommendations(model_ids, pricing, portal_url)
    else:
        model_ids, pricing = union_with_portal_paid_recommendations(model_ids, pricing, portal_url)
    _before_policy = model_ids
    model_ids = restrict_to_nous_policy(model_ids, _policy_allowed, rescue_empty=True)
    _policy_narrowed = model_ids != _before_policy
    if free_tier:
        model_ids, unavailable_models = partition_nous_models_by_tier(model_ids, pricing, free_tier=True)

    if not model_ids and not unavailable_models:
        print("No models available for Nous Portal after filtering.")
        return None
    if free_tier and not model_ids:
        print("No free models currently available.")
        if unavailable_models:
            from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL
            _url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print(unavailable_message or f"Upgrade at {_url} to access paid models.")
        return None
    return model_ids, pricing, unavailable_models, unavailable_message, _policy_narrowed


def _nous_verified_credentials(creds_or_none=None):
    """Resolve Nous runtime credentials; on failure print the diagnosis (re-login when the
    session expired) and return None."""
    from hermes_cli.auth import (
        AuthError, PROVIDER_REGISTRY, _login_nous, format_auth_error, resolve_nous_runtime_credentials)

    try:
        return resolve_nous_runtime_credentials()
    except Exception as exc:
        relogin = isinstance(exc, AuthError) and exc.relogin_required
        msg = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
        if relogin:
            _say(f"Session expired: {msg}", "Re-authenticating with Nous Portal...\n")
            try:
                _login_nous(_nous_login_args(None), PROVIDER_REGISTRY["nous"])
            except Exception as login_exc:
                print(f"Re-login failed: {login_exc}")
            return None
        print(f"Could not verify credentials: {msg}")
        return None


def _nous_persist_selection(selected: str, creds: dict) -> dict:
    """Nous persist step: model choice + provider state, then rewrite ``model`` on a fresh
    config (the caller's may carry stale custom-provider fields) and clear a conflicting
    OPENAI_BASE_URL / OPENAI_API_KEY. Returns the saved config."""
    from hermes_cli.auth import _save_model_choice, _update_config_for_provider
    from hermes_cli.config import get_env_value, load_config, save_config, save_env_value
    _save_model_choice(selected)
    inference_url = creds.get("base_url", "")
    _update_config_for_provider("nous", inference_url)
    config = load_config()
    current_model_cfg = config.get("model")
    if isinstance(current_model_cfg, dict):
        model_cfg = dict(current_model_cfg)
    elif isinstance(current_model_cfg, str) and current_model_cfg.strip():
        model_cfg = {"default": current_model_cfg.strip()}
    else:
        model_cfg = {}
    model_cfg["provider"] = "nous"
    model_cfg["default"] = selected
    if inference_url and inference_url.strip():
        model_cfg["base_url"] = inference_url.rstrip("/")
    else:
        model_cfg.pop("base_url", None)
    clear_model_endpoint_credentials(model_cfg)
    config["model"] = model_cfg
    if get_env_value("OPENAI_BASE_URL"):
        save_env_value("OPENAI_BASE_URL", "")
        save_env_value("OPENAI_API_KEY", "")
    save_config(config)
    return config


def _model_flow_nous(config, current_model="", args=None):
    """Nous Portal provider: ensure logged in, then pick model."""
    from hermes_cli.auth import get_provider_auth_state, _prompt_model_selection, _login_nous, PROVIDER_REGISTRY
    from hermes_cli.config import load_config
    from hermes_cli.nous_subscription import prompt_enable_tool_gateway
    state = get_provider_auth_state("nous")
    if not state or not state.get("access_token"):
        _say("Not logged into Nous Portal. Starting login...", "")

        def _login_then_offer_gateway(login_args, pconfig):
            _login_nous(login_args, pconfig)
            # Offer Tool Gateway enablement for paid subscribers
            with contextlib.suppress(Exception):
                prompt_enable_tool_gateway(load_config() or {})

        # login_nous already handles model selection + config update
        _run_login(_login_then_offer_gateway, _nous_login_args(args), PROVIDER_REGISTRY["nous"])
        return

    # Already logged in — the curated list (agentic models users know from OpenRouter)
    # instead of the hundreds returned by the live /models endpoint.
    from hermes_cli.models import check_nous_free_tier, get_curated_nous_model_ids
    from hermes_cli.models_pricing import get_pricing_for_provider
    model_ids = get_curated_nous_model_ids()
    if not model_ids:
        print("No curated models available for Nous Portal.")
        return

    # Verify credentials are still valid (catches expired sessions early)
    creds = _nous_verified_credentials()
    if creds is None:
        return

    pricing = get_pricing_for_provider("nous")
    # Force fresh account data so recent credit purchases are reflected immediately.
    free_tier = check_nous_free_tier(force_fresh=True)
    if not free_tier:
        from hermes_cli.auth import resolve_nous_runtime_credentials
        try:
            creds = resolve_nous_runtime_credentials(force_refresh=True) or creds
        except Exception:
            # Runtime inference has its own paid-entitlement recovery; don't block.
            pass

    # Portal URL is needed for upgrade links and the recommendations endpoints.
    _nous_portal_url = ""
    with contextlib.suppress(Exception):
        _nous_portal_url = (get_provider_auth_state("nous") or {}).get("portal_base_url", "")

    catalog = _nous_model_catalog(free_tier, _nous_portal_url, model_ids, pricing)
    if catalog is None:
        return
    model_ids, pricing, unavailable_models, unavailable_message, _policy_narrowed = catalog

    from hermes_cli.nous_account import nous_policy_notice
    _policy_notice = nous_policy_notice(removed=_policy_narrowed)
    if _policy_notice:
        print(_policy_notice)
    print(f'Showing {len(model_ids)} curated models — use "Enter custom model name" for others.')

    selected = _prompt_model_selection(
        model_ids, current_model=current_model, pricing=pricing, unavailable_models=unavailable_models,
        portal_url=_nous_portal_url, unavailable_message=unavailable_message, confirm_provider="nous",
        confirm_base_url=creds.get("base_url", ""), confirm_api_key=creds.get("api_key", ""))
    if not selected:
        print("No change.")
        return
    config = _nous_persist_selection(selected, creds)
    print(f"Default model set to: {selected} (via Nous Portal)")
    # Offer Tool Gateway enablement for paid subscribers
    prompt_enable_tool_gateway(config)


def _model_flow_openai_codex(config, current_model=""):
    """OpenAI Codex provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_codex_auth_status, _prompt_model_selection, _login_openai_codex, PROVIDER_REGISTRY, DEFAULT_CODEX_BASE_URL,
    )
    from hermes_cli.codex_models import get_codex_model_ids
    if not _oauth_gate(
        bool(get_codex_auth_status().get("logged_in")), "OpenAI Codex", _login_openai_codex, argparse.Namespace(),
        PROVIDER_REGISTRY["openai-codex"], recheck=lambda: get_codex_auth_status().get("logged_in")):
        return

    # Prefer the credential pool (where `hermes auth` stores device_code tokens),
    # fall back to legacy provider state.
    _codex_token = None
    with contextlib.suppress(Exception):
        _codex_status = get_codex_auth_status()
        _codex_token = _codex_status.get("api_key") if _codex_status.get("logged_in") else None
    if not _codex_token:
        with contextlib.suppress(Exception):
            from hermes_cli.auth import resolve_codex_runtime_credentials
            _codex_token = resolve_codex_runtime_credentials().get("api_key")

    codex_models = get_codex_model_ids(access_token=_codex_token)
    selected = _prompt_model_selection(
        codex_models, current_model=current_model, confirm_provider="openai-codex",
        confirm_base_url=DEFAULT_CODEX_BASE_URL, confirm_api_key=_codex_token or "")
    _activate_provider_model(selected, "openai-codex", DEFAULT_CODEX_BASE_URL,
                             f"Default model set to: {selected} (via OpenAI Codex)")


def _model_flow_claude_agent_sdk(_config, current_model=""):
    """Claude Agent SDK provider: subscription OAuth is owned by the SDK's
    Claude Code CLI subprocess — there is no Hermes login flow and no API
    key to collect. Report credential status structurally, then pick a
    model (the CLI re-checks auth authoritatively at session start)."""
    from hermes_cli.auth import (
        get_claude_agent_sdk_auth_status,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    status = get_claude_agent_sdk_auth_status()
    if status.get("logged_in"):
        print(f"  Claude subscription credentials: ✓ ({status.get('source', '')})")
    else:
        print("No Claude subscription credential detected.")
        print("The SDK authenticates itself: run `claude setup-token` (or `claude")
        print("login`) on this machine, or set CLAUDE_CODE_OAUTH_TOKEN. Continuing —")
        print("the SDK re-checks at session start (macOS Keychain logins are not")
        print("probed here).")
    print()

    # Same Claude model ids the anthropic picker offers; the SDK also accepts
    # an unset model (CLI default), but the picker keeps an explicit choice.
    models = list(
        _PROVIDER_MODELS.get("claude-agent-sdk")
        or _PROVIDER_MODELS.get("anthropic")
        or []
    )
    selected = _prompt_model_selection(models, current_model=current_model)
    if selected:
        _save_model_choice(selected)
        # Empty base URL on purpose: the runtime resolver returns base_url ""
        # for this provider (the SDK owns the endpoint), and passing "" also
        # clears any stale base_url from a previous provider.
        _update_config_for_provider("claude-agent-sdk", "")
        print(f"Default model set to: {selected} (via Claude Agent SDK — Claude subscription)")
    else:
        print("No change.")


def _model_flow_xai_oauth(_config, current_model="", *, args=None):
    """xAI Grok OAuth (SuperGrok / Premium+) provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_xai_oauth_auth_status, _prompt_model_selection, resolve_xai_oauth_runtime_credentials, _login_xai_oauth,
        DEFAULT_XAI_OAUTH_BASE_URL, PROVIDER_REGISTRY)
    from hermes_cli.models import provider_model_ids
    login_args = argparse.Namespace(no_browser=bool(getattr(args, "no_browser", False)), timeout=getattr(args, "timeout", None))
    if not _oauth_gate(
        bool(get_xai_oauth_auth_status().get("logged_in")), "xAI Grok OAuth (SuperGrok / Premium+)", _login_xai_oauth,
        login_args, PROVIDER_REGISTRY["xai-oauth"], fresh_name="xAI OAuth"):
        return

    # ``resolve_xai_oauth_runtime_credentials`` only reads the auth.json singleton, but
    # credentials may live only in the pool (``hermes auth add xai-oauth``) — fall back to
    # the default base URL so the picker still completes.
    base_url = DEFAULT_XAI_OAUTH_BASE_URL
    with contextlib.suppress(Exception):
        creds = resolve_xai_oauth_runtime_credentials()
        base_url = (creds.get("base_url") or "").strip().rstrip("/") or base_url

    models = provider_model_ids("xai-oauth")
    selected = _prompt_model_selection(models, current_model=current_model or (models[0] if models else "grok-4.6"))
    _activate_provider_model(selected, "xai-oauth", base_url,
                             f"Default model set to: {selected} (via xAI Grok OAuth — SuperGrok / Premium+)")


def _model_flow_qwen_oauth(_config, current_model=""):
    """Qwen OAuth provider: reuse local Qwen CLI login, then pick model."""
    from hermes_cli.main_provider_setup import _DEFAULT_QWEN_PORTAL_MODELS
    from hermes_cli.auth import (
        get_qwen_auth_status, resolve_qwen_runtime_credentials, _prompt_model_selection, DEFAULT_QWEN_BASE_URL)
    from hermes_cli.models import fetch_api_models
    status = get_qwen_auth_status()
    if not status.get("logged_in"):
        _say("Not logged into Qwen CLI OAuth.", "Run: qwen auth qwen-oauth",
             *([f"Expected credentials file: {status.get('auth_file')}"] if status.get("auth_file") else []),
             *([f"Error: {status.get('error')}"] if status.get("error") else []))
        return

    # Try live model discovery, fall back to curated list.
    models = None
    with contextlib.suppress(Exception):
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=True)
        models = fetch_api_models(creds["api_key"], creds["base_url"])
    if not models:
        models = list(_DEFAULT_QWEN_PORTAL_MODELS)

    default = current_model or (models[0] if models else "qwen3-coder-plus")
    selected = _prompt_model_selection(models, current_model=default, confirm_provider="qwen-oauth", confirm_base_url=DEFAULT_QWEN_BASE_URL)
    _activate_provider_model(selected, "qwen-oauth", DEFAULT_QWEN_BASE_URL, f"Default model set to: {selected} (via Qwen OAuth)")


def _model_flow_minimax_oauth(config, current_model="", args=None):
    """MiniMax OAuth provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_provider_auth_state, _prompt_model_selection, resolve_minimax_oauth_runtime_credentials, AuthError,
        format_auth_error, _login_minimax_oauth, PROVIDER_REGISTRY)

    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        _say("Not logged into MiniMax. Starting OAuth login...", "")
        mock_args = argparse.Namespace(
            region=getattr(args, "region", None) or "global", no_browser=bool(getattr(args, "no_browser", False)),
            timeout=getattr(args, "timeout", None) or 15.0)
        if not _run_login(_login_minimax_oauth, mock_args, PROVIDER_REGISTRY["minimax-oauth"]):
            return

    try:
        creds = resolve_minimax_oauth_runtime_credentials()
    except AuthError as exc:
        print(format_auth_error(exc))
        return

    from hermes_cli.models import _PROVIDER_MODELS
    model_ids = _PROVIDER_MODELS.get("minimax-oauth", [])
    selected = _prompt_model_selection(model_ids, current_model, confirm_provider="minimax-oauth", confirm_base_url=creds["base_url"])
    _activate_provider_model(selected, "minimax-oauth", creds["base_url"], f"\u2713 Using MiniMax model: {selected}", no_change=None)


def _copilot_model_list(live_ids) -> list:
    """Live GitHub Copilot ids, or the curated fallback with a warning."""
    from hermes_cli.models import _PROVIDER_MODELS
    if live_ids:
        model_list = [model_id for model_id in live_ids if model_id]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
        return model_list
    model_list = _PROVIDER_MODELS.get("copilot", [])
    if model_list:
        _say("  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults.",
             '    Use "Enter custom model name" if you do not see your model.')
    return model_list


def _copilot_catalog(api_key: str):
    """``(catalog, catalog_ids, normalize)`` for a GitHub token; *normalize* canonicalizes a
    model id against the catalog (identity when unknown)."""
    from hermes_cli.models import fetch_github_model_catalog, normalize_copilot_model_id
    catalog = fetch_github_model_catalog(api_key)
    ids = [item.get("id", "") for item in catalog if item.get("id")] if catalog else []

    def _normalize(mid):
        return normalize_copilot_model_id(mid, catalog=catalog, api_key=api_key) or mid

    return catalog, ids, _normalize


def _copilot_obtain_token() -> bool:
    """No Copilot token yet: offer device-code login or manual entry. False = stop."""
    from hermes_cli.config import save_env_value
    _say("No GitHub token configured for GitHub Copilot.", "", "  Supported token types:",
         "    → OAuth token (gho_*)          via `copilot login` or device code flow",
         "    → Fine-grained PAT (github_pat_*)  with Copilot Requests permission",
         "    → GitHub App token (ghu_*)     via environment variable",
         "    ✗ Classic PAT (ghp_*)          NOT supported by Copilot API", "", "  Options:",
         "    1. Login with GitHub (OAuth device code flow)", "    2. Enter a token manually", "    3. Cancel", "")
    choice = _ask("  Choice [1-3]: ", raw=True, cancel_msg="")
    if choice is None:
        return False
    if choice == "1":
        try:
            from hermes_cli.copilot_auth import copilot_device_code_login
            token = copilot_device_code_login()
            if not token:
                print("  Login cancelled or failed.")
                return False
            save_env_value("COPILOT_GITHUB_TOKEN", token)
            _say("  Copilot token saved.", "")
        except Exception as exc:
            print(f"  Login failed: {exc}")
            return False
        return True
    if choice == "2":
        new_key = _ask("  Token (COPILOT_GITHUB_TOKEN): ", secret=True, cancel_msg="")
        if new_key is None:
            return False
        if not new_key:
            print("  Cancelled.")
            return False
        # Validate token type
        with contextlib.suppress(ImportError):
            from hermes_cli.copilot_auth import validate_copilot_token
            valid, msg = validate_copilot_token(new_key)
            if not valid:
                print(f"  ✗ {msg}")
                return False
        save_env_value("COPILOT_GITHUB_TOKEN", new_key)
        _say("  Token saved.", "")
        return True
    print("  Cancelled.")
    return False


def _model_flow_copilot(config, current_model=""):
    """GitHub Copilot flow using env vars, gh CLI, or OAuth device code."""
    from hermes_cli.main_provider_setup import _prompt_reasoning_effort_selection
    from hermes_cli.setup import _current_reasoning_effort, _set_reasoning_effort
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    from hermes_cli.config import load_config
    from hermes_cli.models import fetch_api_models, github_model_reasoning_efforts, copilot_model_api_mode
    provider_id = "copilot"
    pconfig = PROVIDER_REGISTRY[provider_id]
    creds = resolve_api_key_provider_credentials(provider_id)
    api_key = creds.get("api_key", "")
    source = creds.get("source", "")
    if not api_key:
        if not _copilot_obtain_token():
            return
        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = creds.get("api_key", "")
    else:
        if source in {"GITHUB_TOKEN", "GH_TOKEN"}:
            from hermes_cli.env_loader import format_secret_source_suffix
            _say(f"  GitHub token: {api_key[:8]}... ✓ ({source}{format_secret_source_suffix(source)})", "")
        else:
            _say("  GitHub token: ✓ (from `gh auth token`)" if source == "gh auth token" else "  GitHub token: ✓", "")

    effective_base = pconfig.inference_base_url
    catalog, live_models, _normalize = _copilot_catalog(api_key)
    if not catalog:
        live_models = fetch_api_models(api_key, effective_base)

    selected = _pick_model_or_prompt(
        _copilot_model_list(live_models), "Model name: ", current_model=_normalize(current_model),
        confirm_provider=provider_id, confirm_base_url=effective_base, confirm_api_key=api_key)
    if not selected:
        print("No change.")
        return
    selected = _normalize(selected)
    current_effort = _current_reasoning_effort(load_config())
    reasoning_efforts = github_model_reasoning_efforts(selected, catalog=catalog, api_key=api_key)
    selected_effort = None
    if reasoning_efforts:
        print(f"  {selected} supports reasoning controls.")
        selected_effort = _prompt_reasoning_effort_selection(reasoning_efforts, current_effort=current_effort)

    def _finish(cfg, _model):
        if selected_effort is not None:
            _set_reasoning_effort(cfg, selected_effort)

    _persist_model(selected, provider_id, base_url=effective_base,
                   api_mode=copilot_model_api_mode(selected, catalog=catalog, api_key=api_key), finish=_finish)
    print(f"Default model set to: {selected} (via {pconfig.name})")
    if reasoning_efforts:
        if selected_effort == "none":
            print("Reasoning disabled for this model.")
        elif selected_effort:
            print(f"Reasoning effort set to: {selected_effort}")


def _model_flow_copilot_acp(config, current_model=""):
    """GitHub Copilot ACP flow using the local Copilot CLI."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY, get_external_process_provider_status, resolve_api_key_provider_credentials,
        resolve_external_process_provider_credentials)

    del config
    provider_id = "copilot-acp"
    pconfig = PROVIDER_REGISTRY[provider_id]
    status = get_external_process_provider_status(provider_id)
    resolved_command = status.get("resolved_command") or status.get("command") or "copilot"
    effective_base = status.get("base_url") or pconfig.inference_base_url

    _say("  GitHub Copilot ACP delegates Hermes turns to `copilot --acp`.",
         "  Hermes currently starts its own ACP subprocess for each request.",
         "  Hermes uses your selected model as a hint for the Copilot ACP session.",
         f"  Command: {resolved_command}", f"  Backend marker: {effective_base}", "")
    try:
        creds = resolve_external_process_provider_credentials(provider_id)
    except Exception as exc:
        _say(f"  ⚠ {exc}", "  Set HERMES_COPILOT_ACP_COMMAND or COPILOT_CLI_PATH if Copilot CLI is installed elsewhere.")
        return
    effective_base = creds.get("base_url") or effective_base

    catalog_api_key = ""
    with contextlib.suppress(Exception):
        catalog_api_key = resolve_api_key_provider_credentials("copilot").get("api_key", "")
    _catalog, catalog_ids, _normalize = _copilot_catalog(catalog_api_key)
    selected = _pick_model_or_prompt(
        _copilot_model_list(catalog_ids), "Model name: ", current_model=_normalize(current_model),
        confirm_provider=provider_id, confirm_base_url=effective_base, confirm_api_key=catalog_api_key)
    if selected:
        selected = _normalize(selected)
    _finish_model(selected, provider_id, f"Default model set to: {selected} (via {pconfig.name})",
                  base_url=effective_base, api_mode="chat_completions")


def _model_flow_kimi(config, current_model=""):
    """Kimi / Moonshot model selection; the endpoint is chosen by key prefix (no URL prompt):
    ``sk-kimi-*`` → api.kimi.com/coding/v1 (Kimi Coding Plan), other keys → Moonshot."""
    from hermes_cli.auth import PROVIDER_REGISTRY, KIMI_CODE_BASE_URL
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.models import _PROVIDER_MODELS
    provider_id = "kimi-coding"
    pconfig = PROVIDER_REGISTRY[provider_id]
    base_url_env = pconfig.base_url_env_var or ""

    _, existing_key, abort = _ensure_flow_api_key(provider_id, pconfig)
    if abort:
        return

    is_coding_plan = existing_key.startswith("sk-kimi-")
    if is_coding_plan:
        effective_base = KIMI_CODE_BASE_URL
        print(f"  Detected Kimi Coding Plan key → {effective_base}")
    else:
        effective_base = pconfig.inference_base_url
        print(f"  Using Moonshot endpoint → {effective_base}")
    # Clear any manual base URL override so auto-detection works at runtime
    if base_url_env and get_env_value(base_url_env):
        save_env_value(base_url_env, "")
    print()

    model_list = _PROVIDER_MODELS.get("kimi-coding" if is_coding_plan else "moonshot", [])
    selected = _pick_model_or_prompt(
        model_list, "Enter model name: ", current_model=current_model, confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=existing_key)
    # api_mode is dropped so the runtime auto-detects it from the URL.
    _finish_model(selected, provider_id, f"Default model set to: {selected} (via {'Kimi Coding' if is_coding_plan else 'Moonshot'})",
                  base_url=effective_base, drop_api_mode=True)


def _model_flow_stepfun(config, current_model=""):
    """StepFun Step Plan flow with region-specific endpoints."""
    from hermes_cli.main_provider_setup import _infer_stepfun_region, _prompt_provider_choice, _stepfun_base_url_for_region
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import save_env_value
    from hermes_cli.models import _PROVIDER_MODELS, fetch_api_models
    provider_id = "stepfun"
    pconfig = PROVIDER_REGISTRY[provider_id]
    base_url_env = pconfig.base_url_env_var or ""

    _, existing_key, abort = _ensure_flow_api_key(provider_id, pconfig)
    if abort:
        return

    current_base = _env_base_url(base_url_env)
    if not current_base:
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            current_base = str(model_cfg.get("base_url") or "").strip()
    current_region = _infer_stepfun_region(current_base or pconfig.inference_base_url)

    regions = [(key, f"{name} ({_stepfun_base_url_for_region(key)})") for key, name in
               (("international", "International"), ("china", "China"))]
    # Active region first, marked; then the other; then Cancel.
    ordered_regions = ([(k, f"{label}  ← currently active") for k, label in regions if k == current_region]
                       + [(k, label) for k, label in regions if k != current_region] + [("cancel", "Cancel")])

    region_idx = _prompt_provider_choice([label for _, label in ordered_regions])
    if region_idx is None or ordered_regions[region_idx][0] == "cancel":
        print("No change.")
        return
    effective_base = _stepfun_base_url_for_region(ordered_regions[region_idx][0])
    if base_url_env:
        save_env_value(base_url_env, effective_base)

    model_list = fetch_api_models(existing_key, effective_base)
    if model_list:
        print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print(f"  Could not auto-detect models from {pconfig.name} API — showing Step Plan fallback catalog.")

    selected = _pick_model_or_prompt(
        model_list, "Model name: ", current_model=current_model, confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=existing_key)
    model = _finish_model(selected, provider_id, f"Default model set to: {selected} (via {pconfig.name})",
                          base_url=effective_base, drop_api_mode=True)
    if model is not None:
        # Sync the caller's config dict so the setup wizard's final save_config(config) preserves our model
        # settings. Without this, the wizard overwrites model.provider/base_url with the stale values from
        # its own config dict (#4172).
        config["model"] = dict(model)


def _model_flow_vertex(config, current_model=""):
    """Google Vertex AI (Gemini via the OpenAI-compatible endpoint). Auth is OAuth2 (service-account
    JSON or ADC): the credential *path* lives in .env (VERTEX_CREDENTIALS_PATH /
    GOOGLE_APPLICATION_CREDENTIALS); project ID and region are non-secret, saved under ``vertex:``."""
    from hermes_cli.auth import _prompt_model_selection
    from hermes_cli.config import load_config, get_env_value
    from hermes_cli.models import _PROVIDER_MODELS

    # 1. Credential source detection (fast, no network / no google-auth import).
    sa_path = (get_env_value("VERTEX_CREDENTIALS_PATH") or get_env_value("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if sa_path:
        print(f"  Vertex credentials: service account JSON ({sa_path}) ✓")
    else:
        _say("  Vertex credentials: Application Default Credentials (ADC)",
             "    Vertex uses OAuth2, not a static API key. Either:",
             "      • run 'gcloud auth application-default login', or",
             "      • set VERTEX_CREDENTIALS_PATH in ~/.hermes/.env to a service account JSON")
    print()

    vertex_cfg = load_config().get("vertex")
    if not isinstance(vertex_cfg, dict):
        vertex_cfg = {}

    # 2. Project ID (optional — falls back to the project embedded in creds).
    current_project = str(vertex_cfg.get("project_id") or "").strip()
    project_input = _ask(f"  GCP project ID [{current_project or 'from credentials'}]: ", cancel_msg="")
    if project_input is None:
        return
    project_id = project_input or current_project

    # 3. Region (default global — required for the Gemini 3.x previews).
    current_region = str(vertex_cfg.get("region") or "global").strip() or "global"
    region_input = _ask(f"  Vertex region [{current_region}]: ", cancel_msg="")
    if region_input is None:
        return
    region = region_input or current_region

    # 4. Model selection (curated list — Vertex has no /models listing route).
    model_list = _PROVIDER_MODELS.get("vertex", []) or ["google/gemini-3-pro-preview", "google/gemini-3-flash-preview"]
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    base_url_preview = f"https://{host}/v1beta1/projects/<project>/locations/{region}/endpoints/openapi"
    selected = _prompt_model_selection(model_list, current_model=current_model, confirm_provider="vertex", confirm_base_url=base_url_preview)

    def _finish(cfg, _model):
        vcfg = _ensure_dict_section(cfg, "vertex")
        vcfg["project_id"] = project_id
        vcfg["region"] = region

    # base_url is computed at runtime from project+region; do not pin it.
    # api_mode is dropped: chat_completions is the profile default.
    _finish_model(selected, "vertex", f"  Default model set to: {selected} (via Google Vertex AI, {region})", no_change="  No change.",
                  drop_base_url=True, drop_api_mode=True, finish=_finish)


def _select_zai_endpoint(current_base: str) -> str:
    """Picker for the official Z.AI endpoints (``ZAI_ENDPOINTS`` in ``hermes_cli.auth``, kept in
    sync with the probe list) plus a custom-proxy option. Returns the selected base URL;
    *current_base* on cancel/error."""
    from hermes_cli.main_provider_setup import _prompt_provider_choice
    from hermes_cli.auth import ZAI_ENDPOINTS
    options = [(label, url) for _, url, _, label in ZAI_ENDPOINTS]
    normalized_current = (current_base or "").strip().rstrip("/")

    # Default to the active endpoint when known; a custom URL defaults to "Custom proxy".
    default_idx = next((idx for idx, (_, url) in enumerate(options) if normalized_current == url.rstrip("/")),
                       len(options) if normalized_current else 0)
    choices = [f"{label} ({url})" for label, url in options] + ["Custom proxy URL"]
    selected = _prompt_provider_choice(choices, default=default_idx, title="Select Z.AI / GLM endpoint:")
    if selected is None:
        return current_base
    if selected != len(options):
        return options[selected][1].rstrip("/")
    override = _ask(f"Custom base URL [{current_base}]: ", cancel_msg="")
    if not override:
        return current_base
    if not override.startswith(_HTTP):
        print("  Invalid URL — must start with http:// or https://. Keeping current value.")
        return current_base
    return override.rstrip("/")


_GEMINI_FREE_TIER_NOTICE = (
    "", "❌ This Google API key is on the free tier (<= 250 requests/day for gemini-2.5-flash).",
    "   Hermes typically makes 3-10 API calls per user turn (tool iterations + auxiliary tasks),",
    "   so the free tier is exhausted after a handful of messages and cannot sustain",
    "   an agent session.", "",
    "   To use Gemini with Hermes, enable billing on your Google Cloud project and regenerate",
    "   the key in a billing-enabled project: https://aistudio.google.com/apikey", "",
    "   Alternatives with workable free usage: DeepSeek, OpenRouter (free models), Groq, Nous.", "",
    "Not saving Gemini as the default provider.")


def _gemini_tier_ok(existing_key: str, pconfig, base_url_env: str) -> bool:
    """Gemini free-tier gate: free-tier daily quotas (<= 250 RPD for Flash) are exhausted in a
    handful of agent turns, so refuse a free-tier key. The probe is best-effort; network or
    auth errors fall through without blocking."""
    try:
        from agent.gemini_native_adapter import probe_gemini_tier
    except Exception:
        return True
    print("  Checking Gemini API tier...")
    tier = probe_gemini_tier(existing_key, _env_base_url(base_url_env) or pconfig.inference_base_url)
    if tier == "free":
        _say(*_GEMINI_FREE_TIER_NOTICE)
        return False
    # "unknown" (network/auth/unexpected response): don't block; the runtime 429 handler
    # surfaces free-tier guidance if needed.
    _say("  Tier check: paid ✓" if tier == "paid" else "  Tier check: could not verify (proceeding anyway).", "")
    return True


def _lmstudio_models(pconfig, curated, api_key, base_url):
    """LM Studio: live /api/v1/models probe only."""
    from hermes_cli.auth import AuthError
    from hermes_cli.models_local import fetch_lmstudio_models
    try:
        model_list = fetch_lmstudio_models(api_key=api_key, base_url=base_url)
    except AuthError as exc:
        _say(f"  LM Studio rejected the request: {exc}", "  Set LM_API_KEY (or update it) to match the server's bearer token.")
        model_list = []
    _report_live_models(model_list, "LM Studio")
    return model_list


def _ollama_cloud_models(pconfig, curated, api_key, base_url):
    """Ollama Cloud: forced live refresh so newly released models appear the moment the user
    enters their key, not when the disk cache TTL expires."""
    from hermes_cli.models import fetch_ollama_cloud_models
    model_list = fetch_ollama_cloud_models(api_key=api_key, base_url=base_url, force_refresh=True)
    _report_live_models(model_list, "Ollama Cloud")
    return model_list


def _opencode_free_models(pconfig, curated, api_key, base_url):
    """Keyless tier: the curated list is synced against anonymous live probes (models.dev's
    cost.input==0 filter lags reality)."""
    if curated:
        print(f'  Showing {len(curated)} keyless free models — use "Enter custom model name" for others.')
    return curated


def _novita_models(pconfig, curated, api_key, base_url):
    """Novita: live first, then models.dev, then curated."""
    from hermes_cli.models import fetch_api_models
    live_models = fetch_api_models(api_key, base_url)
    if live_models:
        _report_live_models(live_models, f"{pconfig.name} API")
        return live_models
    model_list = _models_dev_merged("novita", curated)
    if model_list:
        _report_live_models(model_list, "models.dev registry")
        return model_list
    _show_curated(curated)
    return curated


# provider id -> (pconfig, curated, api_key_for_probe, effective_base) -> model list
_SPECIAL_MODEL_LISTS = {
    "lmstudio": _lmstudio_models,
    "ollama-cloud": _ollama_cloud_models,
    "opencode-free": _opencode_free_models,
    "novita": _novita_models}


def _api_key_provider_model_list(provider_id: str, pconfig, existing_key: str, key_env: str, effective_base: str) -> list:
    """Model list for an API-key provider: models.dev registry (cached, agentic/tool-capable filter)
    → curated static list (offline insurance) → live /models probe (small providers without
    models.dev data). Providers in ``_SPECIAL_MODEL_LISTS`` have their own resolution."""
    from hermes_cli.config import get_env_value
    from hermes_cli.models import _PROVIDER_MODELS, fetch_api_models
    curated = _PROVIDER_MODELS.get(provider_id, [])
    api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")

    special = _SPECIAL_MODEL_LISTS.get(provider_id)
    if special is not None:
        return special(pconfig, curated, api_key_for_probe, effective_base)
    # models.dev first (tool-capable, noise-filtered), merged with curated so newly added
    # models still appear.
    model_list = _models_dev_merged(provider_id, curated)
    if model_list:
        _report_live_models(model_list, "models.dev registry")
        return model_list
    if curated and len(curated) >= 8:
        # Substantial curated list — use it directly, skip live probe
        _show_curated(curated)
        return curated
    live_models = fetch_api_models(api_key_for_probe, effective_base)
    if live_models and len(live_models) >= len(curated):
        _report_live_models(live_models, f"{pconfig.name} API")
        return live_models
    _show_curated(curated)  # may be empty: falls through to raw input
    return curated


def _model_flow_api_key_provider(config, provider_id, current_model=""):
    """Generic flow for API-key providers (z.ai, MiniMax, OpenCode, etc.)."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import save_env_value, load_config
    from hermes_cli.models import opencode_model_api_mode, normalize_opencode_model_id
    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""
    is_opencode = provider_id in {"opencode-zen", "opencode-go", "opencode-free"}

    # OpenCode Free is keyless — the tier is served anonymously and any unrecognized
    # bearer 401s, so there is no key to prompt for.
    if provider_id == "opencode-free":
        print("  OpenCode Free is keyless — no API key or account needed.")
        existing_key = ""
    else:
        _, existing_key, abort = _ensure_flow_api_key(provider_id, pconfig)
        if abort:
            return
    if provider_id == "gemini" and existing_key and not _gemini_tier_ok(existing_key, pconfig, base_url_env):
        return

    # Optional base URL override. Precedence: env var → config.yaml model.base_url → registry
    # default; reading config.yaml keeps a saved remote URL from being overwritten with
    # localhost when the user just presses Enter.
    current_base = _env_base_url(base_url_env)
    if not current_base:
        with contextlib.suppress(Exception):
            _m = load_config().get("model") or {}
            if str(_m.get("provider") or "").strip().lower() == provider_id:
                current_base = str(_m.get("base_url") or "").strip()
    effective_base = current_base or pconfig.inference_base_url

    if provider_id == "zai":
        # Four official endpoints with separate billing paths — a picker lets users match
        # the endpoint to their key type.
        chosen_base = _select_zai_endpoint(effective_base)
        if chosen_base and chosen_base != effective_base and base_url_env:
            save_env_value(base_url_env, chosen_base)
        effective_base = chosen_base
    else:
        effective_base = _prompt_base_url_override(effective_base, base_url_env)

    model_list = _api_key_provider_model_list(provider_id, pconfig, existing_key, key_env, effective_base)
    if is_opencode:
        model_list = [normalize_opencode_model_id(provider_id, mid) for mid in model_list]
        current_model = normalize_opencode_model_id(provider_id, current_model)
        model_list = list(dict.fromkeys(mid for mid in model_list if mid))

    # Per-model pricing when the provider supports it; get_pricing_for_provider() is memoized
    # and returns {} otherwise — never a blocking fetch beyond the catalog lookup above.
    pricing: dict = {}
    if model_list:
        try:
            from hermes_cli.models_pricing import get_pricing_for_provider
            pricing = get_pricing_for_provider(provider_id) or {}
        except Exception:
            pricing = {}
    selected = _pick_model_or_prompt(
        model_list, "Model name: ", current_model=current_model, pricing=pricing, confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=existing_key)
    if selected and is_opencode:
        selected = normalize_opencode_model_id(provider_id, selected)
    # OpenCode pins its api_mode; everyone else drops it so the runtime auto-detects.
    _finish_model(
        selected, provider_id, f"Default model set to: {selected} (via {pconfig.name})", base_url=effective_base,
        api_mode=opencode_model_api_mode(provider_id, selected) if selected and is_opencode else None,
        drop_api_mode=not is_opencode)


def _anthropic_authenticate() -> bool:
    """Interactive Anthropic auth (OAuth subscription or API key). False = flow must stop."""
    from hermes_cli.main_provider_setup import _run_anthropic_oauth_flow
    from hermes_cli.config import save_env_value, save_anthropic_api_key
    _say("", "  Choose authentication method:", "", "    1. Claude Pro/Max subscription (OAuth login)",
         "    2. Anthropic API key (pay-per-token)", "    3. Cancel", "")
    choice = _ask("  Choice [1/2/3]: ", raw=True, cancel_msg="")
    if choice is None:
        return False
    if choice == "1":
        return _run_anthropic_oauth_flow(save_env_value)
    if choice == "2":
        _say("", "  Get an API key at: https://platform.claude.com/settings/keys", "")
        api_key = _ask("  API key (sk-ant-...): ", secret=True, cancel_msg="")
        if api_key is None:
            return False
        if not api_key:
            print("  Cancelled.")
            return False
        save_anthropic_api_key(api_key, save_fn=save_env_value)
        print("  ✓ API key saved.")
        return True
    print("  No change.")
    return False


def _model_flow_anthropic(config, current_model=""):
    """Flow for Anthropic provider — OAuth subscription, API key, or Claude Code creds."""
    from hermes_cli.auth import get_anthropic_key
    from hermes_cli.models import _PROVIDER_MODELS

    # Check ALL credential sources
    existing_key = get_anthropic_key()
    cc_available = False
    with contextlib.suppress(Exception):
        from agent.anthropic_credentials import read_claude_code_credentials, is_claude_code_token_valid, _is_oauth_token
        cc_creds = read_claude_code_credentials()
        if cc_creds and is_claude_code_token_valid(cc_creds):
            cc_available = True

    # Stale-OAuth guard: an expired OAuth token with no valid cc_creds fallback is treated
    # as missing so the re-auth path is offered.
    existing_is_stale_oauth = bool(existing_key and _is_oauth_token(existing_key) and not cc_available)
    has_creds = (bool(existing_key) and not existing_is_stale_oauth) or cc_available
    needs_auth = not has_creds

    if has_creds:
        if existing_key:
            from hermes_cli.env_loader import format_secret_source_suffix
            from hermes_cli.auth import PROVIDER_REGISTRY

            # Surface which env var supplied the key so Bitwarden users see "(from Bitwarden)".
            source_suffix = ""
            for var in PROVIDER_REGISTRY["anthropic"].api_key_env_vars:
                if os.getenv(var, "").strip() == existing_key:
                    source_suffix = format_secret_source_suffix(var)
                    if source_suffix:
                        break
            print(f"  Anthropic credentials: {existing_key[:12]}... ✓{source_suffix}")
        elif cc_available:
            print("  Claude Code credentials: ✓ (auto-detected)")
        print()
        choice = _prompt_auth_credentials_choice("Anthropic credentials:")
        if choice == "reauth":
            needs_auth = True
        elif choice == "cancel":
            return
        # "use" (default): proceed to model selection with existing creds

    if needs_auth and not _anthropic_authenticate():
        return
    print()

    selected = _pick_model_or_prompt(
        _PROVIDER_MODELS.get("anthropic", []), "Model name (e.g., claude-sonnet-4-20250514): ",
        current_model=current_model, confirm_provider="anthropic")
    # Clear base_url: resolve_runtime_provider() always hardcodes Anthropic's URL, and a
    # stale value can contaminate other providers on a later switch.
    _finish_model(selected, "anthropic", f"Default model set to: {selected} (via Anthropic)", drop_base_url=True, drop_api_mode=True)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import subprocess  # noqa: F401,E402
import urllib.parse  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'BEDROCK_GEO_PREFIXES': ('hermes_cli.model_setup_flows_bedrock', 'BEDROCK_GEO_PREFIXES'),
    'bedrock_model_routable_from_region': ('hermes_cli.model_setup_flows_bedrock', 'bedrock_model_routable_from_region'),
    'bedrock_region_geo_prefix': ('hermes_cli.model_setup_flows_bedrock', 'bedrock_region_geo_prefix'),
    'custom_provider_slug': ('hermes_cli.providers', 'custom_provider_slug'),
    'line_input': ('hermes_cli.cli_output', 'line_input'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
