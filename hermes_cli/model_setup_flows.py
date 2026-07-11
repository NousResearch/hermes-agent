"""Per-provider model-selection wizard flows for ``hermes setup`` / ``hermes model``.

Extracted from ``hermes_cli/main.py`` as part of the god-file decomposition
campaign (``~/.hermes/plans/god-file-decomposition.md``, Phase 2 — splitting
main.py handler/flow bodies out of the module). These 18 ``_model_flow_*``
functions are the interactive provider-setup branches dispatched by
``select_provider_and_model`` (which stays in main.py).

Behavior-neutral: each function is lifted verbatim. ``select_provider_and_model``
in main.py re-imports them (``from hermes_cli.model_setup_flows import *``-style
explicit import) so existing call sites — and test monkeypatches that target
``hermes_cli.main._model_flow_*`` — keep resolving against main.py's namespace.

main.py-internal helpers the flows call (``_prompt_api_key``, ``_save_custom_provider``,
the reasoning-effort/stepfun/qwen helpers, ``_run_anthropic_oauth_flow``, …) are
imported lazily inside the flows (``from hermes_cli.main import ...`` resolves at
call time, when main.py is fully loaded) so this module never imports
``hermes_cli.main`` at import time -> no import cycle.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from hermes_cli.config import clear_model_endpoint_credentials


def _prune_replaced_custom_model_config_credentials(
    base_url: str,
    *,
    provider_name: str = "",
) -> None:
    """Drop stale ``model_config`` credentials from inactive custom pools.

    ``model_config`` means "the credential currently stored under
    ``model.api_key``". After an explicit custom-endpoint switch, any old
    custom pool still carrying that source points at the previous endpoint and
    can be selected before the freshly saved config is tried.
    """
    try:
        from agent.credential_pool import (
            CUSTOM_POOL_PREFIX,
            get_custom_provider_pool_key,
        )
        from hermes_cli.auth import read_credential_pool, write_credential_pool

        active_pool_key = get_custom_provider_pool_key(
            base_url,
            provider_name=provider_name or None,
        )
        if not active_pool_key:
            return
        pools = read_credential_pool(None)
        if not isinstance(pools, dict):
            return
        for pool_key, entries in pools.items():
            if (
                not isinstance(pool_key, str)
                or not pool_key.startswith(CUSTOM_POOL_PREFIX)
                or pool_key == active_pool_key
                or not isinstance(entries, list)
            ):
                continue
            retained = []
            removed_ids = []
            changed = False
            for entry in entries:
                if isinstance(entry, dict) and entry.get("source") == "model_config":
                    changed = True
                    entry_id = entry.get("id")
                    if entry_id:
                        removed_ids.append(str(entry_id))
                    continue
                retained.append(entry)
            if changed:
                write_credential_pool(pool_key, retained, removed_ids=removed_ids)
    except Exception:
        return


def _prompt_auth_credentials_choice(title: str) -> str:
    """Prompt for reuse / reauthenticate / cancel with the standard radio UI.

    Returns one of ``"use"``, ``"reauth"``, ``"cancel"``. Falls back to a
    numbered prompt when curses is unavailable (piped stdin, non-TTY).
    """
    choices = [
        "Use existing credentials",
        "Reauthenticate (new OAuth login)",
        "Cancel",
    ]
    try:
        from hermes_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice(title, choices, 0)
        if idx >= 0:
            print()
            return ("use", "reauth", "cancel")[idx]
    except Exception:
        pass

    print(title)
    for i, label in enumerate(choices, 1):
        marker = "→" if i == 1 else " "
        print(f"  {marker} {i}. {label}")
    print()
    try:
        choice = input("  Choice [1/2/3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

    if choice == "2":
        return "reauth"
    if choice == "3":
        return "cancel"
    return "use"


def _model_flow_openrouter(config, current_model=""):
    """OpenRouter provider: ensure API key, then pick model."""
    from hermes_cli.main import _prompt_api_key
    from hermes_constants import OPENROUTER_BASE_URL
    from hermes_cli.auth import (
        ProviderConfig,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import get_env_value

    # Route through _prompt_api_key so users can replace a stale/broken key
    # in-flow (K/R/C) instead of having to edit ~/.hermes/.env by hand. The
    # previous bypass-when-key-exists branch left no way to recover from a
    # bad paste short of re-running `hermes setup` from scratch. OpenRouter
    # isn't in PROVIDER_REGISTRY so we synthesize a minimal pconfig.
    pconfig = ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        auth_type="api_key",
        api_key_env_vars=("OPENROUTER_API_KEY",),
    )
    existing_key = get_env_value("OPENROUTER_API_KEY") or ""
    if not existing_key:
        print("Get one at: https://openrouter.ai/keys")
        print()
    _resolved, abort = _prompt_api_key(pconfig, existing_key, provider_id="openrouter")
    if abort:
        return

    from hermes_cli.models import model_ids, get_pricing_for_provider

    openrouter_models = model_ids(force_refresh=True)

    # Fetch live pricing (non-blocking — returns empty dict on failure)
    pricing = get_pricing_for_provider("openrouter", force_refresh=True)

    selected = _prompt_model_selection(
        openrouter_models,
        current_model=current_model,
        pricing=pricing,
        confirm_provider="openrouter",
        confirm_base_url=OPENROUTER_BASE_URL,
        confirm_api_key=_resolved or existing_key,
    )
    if selected:
        _save_model_choice(selected)

        # Update config provider and deactivate any OAuth provider
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "openrouter"
        model["base_url"] = OPENROUTER_BASE_URL
        model["api_mode"] = "chat_completions"
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        save_config(cfg)
        deactivate_provider()
        print(f"Default model set to: {selected} (via OpenRouter)")
    else:
        print("No change.")


def _print_moa_preset(name: str, preset: dict) -> None:
    """Print the full reference-models + aggregator breakdown for a preset."""
    print(f"  Preset: {name}")
    print("  Reference models:")
    for idx, slot in enumerate(preset.get("reference_models") or [], start=1):
        print(f"    {idx}. {slot.get('provider')}:{slot.get('model')}")
    agg = preset.get("aggregator") or {}
    print(f"  Aggregator:  {agg.get('provider')}:{agg.get('model')}")


def _model_flow_moa(config, current_model=""):
    """Mixture of Agents virtual provider: pick a preset, then persist it.

    Unlike the other provider flows there is no credential step — MoA is a
    virtual provider whose presets reference already-configured providers. We
    always show the preset list (even when there is only one) so the user sees
    what they are selecting, then print the full preset breakdown on selection.
    """
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import load_config, save_config
    from hermes_cli.moa_config import normalize_moa_config

    moa = normalize_moa_config(config.get("moa") if isinstance(config, dict) else {})
    presets = moa.get("presets") or {}
    if not presets:
        print("No MoA presets configured. Run `hermes moa configure <name>` first.")
        return

    names = list(presets.keys())
    default_name = moa.get("default_preset") or names[0]

    # Build labelled rows showing the aggregator so the picker is informative
    # even before drilling into the full breakdown.
    rows = []
    for n in names:
        agg = (presets[n].get("aggregator") or {})
        agg_label = f"{agg.get('provider')}:{agg.get('model')}" if agg else ""
        ref_count = len(presets[n].get("reference_models") or [])
        suffix = "  ← default" if n == default_name else ""
        rows.append(f"{n}  (agg {agg_label}, {ref_count} refs){suffix}")

    default_idx = names.index(default_name) if default_name in names else 0

    try:
        from hermes_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice("Select a Mixture of Agents preset:", rows, default_idx)
    except Exception:
        print("Select a Mixture of Agents preset:")
        for i, row in enumerate(rows, 1):
            marker = "→" if (i - 1) == default_idx else " "
            print(f"  {marker} {i}. {row}")
        try:
            raw = input(f"  Choice [1-{len(rows)}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("No change.")
            return
        if not raw:
            idx = default_idx
        else:
            try:
                idx = max(0, min(len(rows) - 1, int(raw) - 1))
            except ValueError:
                print("No change.")
                return

    if idx is None or idx < 0:
        print("No change.")
        return

    selected_name = names[idx]
    preset = presets[selected_name]

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    model["default"] = selected_name
    model["provider"] = "moa"
    # MoA is a virtual local provider — drop any stale endpoint credentials and
    # base_url so auto-resolution doesn't keep pointing at the previous real
    # provider. (clear_model_endpoint_credentials handles api_key/api_mode but
    # intentionally leaves base_url, so pop it here.)
    clear_model_endpoint_credentials(model, clear_api_mode=True)
    model.pop("base_url", None)
    save_config(cfg)
    _save_model_choice(selected_name)
    deactivate_provider()

    print()
    print(f"Default model set to: {selected_name} (via Mixture of Agents)")
    _print_moa_preset(selected_name, preset)


def _model_flow_nous(config, current_model="", args=None):
    """Nous Portal provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_provider_auth_state,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        resolve_nous_runtime_credentials,
        AuthError,
        format_auth_error,
        _login_nous,
        PROVIDER_REGISTRY,
    )
    from hermes_cli.config import (
        get_env_value,
        load_config,
        save_config,
        save_env_value,
    )
    from hermes_cli.nous_subscription import prompt_enable_tool_gateway

    state = get_provider_auth_state("nous")
    if not state or not state.get("access_token"):
        print("Not logged into Nous Portal. Starting login...")
        print()
        try:
            mock_args = argparse.Namespace(
                portal_url=getattr(args, "portal_url", None),
                inference_url=getattr(args, "inference_url", None),
                client_id=getattr(args, "client_id", None),
                scope=getattr(args, "scope", None),
                no_browser=bool(getattr(args, "no_browser", False)),
                timeout=getattr(args, "timeout", None) or 15.0,
                ca_bundle=getattr(args, "ca_bundle", None),
                insecure=bool(getattr(args, "insecure", False)),
            )
            _login_nous(mock_args, PROVIDER_REGISTRY["nous"])
            # Offer Tool Gateway enablement for paid subscribers
            try:
                _refreshed = load_config() or {}
                prompt_enable_tool_gateway(_refreshed)
            except Exception:
                pass
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return
        # login_nous already handles model selection + config update
        return

    # Already logged in — use curated model list (same as OpenRouter defaults).
    # The live /models endpoint returns hundreds of models; the curated list
    # shows only agentic models users recognize from OpenRouter.
    from hermes_cli.models import (
        get_curated_nous_model_ids,
        get_pricing_for_provider,
        check_nous_free_tier,
        partition_nous_models_by_tier,
        union_with_portal_free_recommendations,
        union_with_portal_paid_recommendations,
    )

    model_ids = get_curated_nous_model_ids()
    if not model_ids:
        print("No curated models available for Nous Portal.")
        return

    # Verify credentials are still valid (catches expired sessions early)
    try:
        creds = resolve_nous_runtime_credentials()
    except Exception as exc:
        relogin = isinstance(exc, AuthError) and exc.relogin_required
        msg = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
        if relogin:
            print(f"Session expired: {msg}")
            print("Re-authenticating with Nous Portal...\n")
            try:
                mock_args = argparse.Namespace(
                    portal_url=None,
                    inference_url=None,
                    client_id=None,
                    scope=None,
                    no_browser=False,
                    timeout=15.0,
                    ca_bundle=None,
                    insecure=False,
                )
                _login_nous(mock_args, PROVIDER_REGISTRY["nous"])
            except Exception as login_exc:
                print(f"Re-login failed: {login_exc}")
            return
        print(f"Could not verify credentials: {msg}")
        return

    # Fetch live pricing (non-blocking — returns empty dict on failure)
    pricing = get_pricing_for_provider("nous")

    # Force fresh account data for model selection so recent credit purchases
    # are reflected immediately.
    free_tier = check_nous_free_tier(force_fresh=True)
    if not free_tier:
        try:
            refreshed_creds = resolve_nous_runtime_credentials(
                force_refresh=True,
            )
            if refreshed_creds:
                creds = refreshed_creds
        except Exception:
            # Runtime inference has its own paid-entitlement recovery path; do
            # not block model selection if this opportunistic refresh fails.
            pass

    # Resolve portal URL early — needed both for upgrade links and for the
    # freeRecommendedModels endpoint below.
    _nous_portal_url = ""
    try:
        _nous_state = get_provider_auth_state("nous")
        if _nous_state:
            _nous_portal_url = _nous_state.get("portal_base_url", "")
    except Exception:
        pass

    # For free users: partition models into selectable/unavailable based on
    # whether they are free per the Portal-reported pricing.  First augment
    # with the Portal's freeRecommendedModels list so newly-launched free
    # models show up even if this CLI build's hardcoded curated list and
    # docs-hosted manifest haven't caught up yet.
    #
    # For paid users: mirror the same idea with paidRecommendedModels so
    # newly-launched paid models surface in the picker too — independent
    # of CLI release cadence.
    unavailable_models: list[str] = []
    unavailable_message = ""
    if free_tier:
        try:
            from hermes_cli.nous_account import (
                format_nous_portal_entitlement_message,
                get_nous_portal_account_info,
            )

            _account_info = get_nous_portal_account_info(force_fresh=True)
            unavailable_message = (
                format_nous_portal_entitlement_message(
                    _account_info,
                    capability="paid Nous models",
                )
                or ""
            )
        except Exception:
            unavailable_message = ""
        model_ids, pricing = union_with_portal_free_recommendations(
            model_ids, pricing, _nous_portal_url,
        )
        model_ids, unavailable_models = partition_nous_models_by_tier(
            model_ids, pricing, free_tier=True
        )
    else:
        model_ids, pricing = union_with_portal_paid_recommendations(
            model_ids, pricing, _nous_portal_url,
        )

    if not model_ids and not unavailable_models:
        print("No models available for Nous Portal after filtering.")
        return

    if free_tier and not model_ids:
        print("No free models currently available.")
        if unavailable_models:
            from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL

            _url = (_nous_portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print(unavailable_message or f"Upgrade at {_url} to access paid models.")
        return

    print(
        f'Showing {len(model_ids)} curated models — use "Enter custom model name" for others.'
    )

    selected = _prompt_model_selection(
        model_ids,
        current_model=current_model,
        pricing=pricing,
        unavailable_models=unavailable_models,
        portal_url=_nous_portal_url,
        unavailable_message=unavailable_message,
        confirm_provider="nous",
        confirm_base_url=creds.get("base_url", ""),
        confirm_api_key=creds.get("api_key", ""),
    )
    if selected:
        _save_model_choice(selected)
        # Reactivate Nous as the provider and update config
        inference_url = creds.get("base_url", "")
        _update_config_for_provider("nous", inference_url)
        # Reload after the auth helper writes provider state. The incoming
        # config object may still contain stale custom-provider fields.
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
        # Clear any custom endpoint that might conflict
        if get_env_value("OPENAI_BASE_URL"):
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
        save_config(config)
        print(f"Default model set to: {selected} (via Nous Portal)")
        # Offer Tool Gateway enablement for paid subscribers
        prompt_enable_tool_gateway(config)
    else:
        print("No change.")

def _model_flow_openai_codex(config, current_model=""):
    """OpenAI Codex provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_codex_auth_status,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        _login_openai_codex,
        PROVIDER_REGISTRY,
        DEFAULT_CODEX_BASE_URL,
    )
    from hermes_cli.codex_models import get_codex_model_ids

    status = get_codex_auth_status()
    if status.get("logged_in"):
        print("  OpenAI Codex credentials: ✓")
        print()
        choice = _prompt_auth_credentials_choice("OpenAI Codex credentials:")

        if choice == "reauth":
            print("Starting a fresh OpenAI Codex login...")
            print()
            try:
                mock_args = argparse.Namespace()
                _login_openai_codex(
                    mock_args,
                    PROVIDER_REGISTRY["openai-codex"],
                    force_new_login=True,
                )
            except SystemExit:
                print("Login cancelled or failed.")
                return
            except Exception as exc:
                print(f"Login failed: {exc}")
                return
            status = get_codex_auth_status()
            if not status.get("logged_in"):
                print("Login failed.")
                return
        elif choice == "cancel":
            return
    else:
        print("Not logged into OpenAI Codex. Starting login...")
        print()
        try:
            mock_args = argparse.Namespace()
            _login_openai_codex(mock_args, PROVIDER_REGISTRY["openai-codex"])
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return

    _codex_token = None
    # Prefer credential pool (where `hermes auth` stores device_code tokens),
    # fall back to legacy provider state.
    try:
        _codex_status = get_codex_auth_status()
        if _codex_status.get("logged_in"):
            _codex_token = _codex_status.get("api_key")
    except Exception:
        pass
    if not _codex_token:
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            _codex_creds = resolve_codex_runtime_credentials()
            _codex_token = _codex_creds.get("api_key")
        except Exception:
            pass

    codex_models = get_codex_model_ids(access_token=_codex_token)

    selected = _prompt_model_selection(
        codex_models,
        current_model=current_model,
        confirm_provider="openai-codex",
        confirm_base_url=DEFAULT_CODEX_BASE_URL,
        confirm_api_key=_codex_token or "",
    )
    if selected:
        _save_model_choice(selected)
        _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)
        print(f"Default model set to: {selected} (via OpenAI Codex)")
    else:
        print("No change.")

def _model_flow_xai_oauth(_config, current_model="", *, args=None):
    """xAI Grok OAuth (SuperGrok / Premium+) provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_xai_oauth_auth_status,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        resolve_xai_oauth_runtime_credentials,
        _login_xai_oauth,
        DEFAULT_XAI_OAUTH_BASE_URL,
        PROVIDER_REGISTRY,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    status = get_xai_oauth_auth_status()
    if status.get("logged_in"):
        print("  xAI Grok OAuth (SuperGrok / Premium+) credentials: ✓")
        print()
        choice = _prompt_auth_credentials_choice(
            "xAI Grok OAuth (SuperGrok / Premium+) credentials:"
        )

        if choice == "reauth":
            print("Starting a fresh xAI OAuth login...")
            print()
            try:
                mock_args = argparse.Namespace(
                    no_browser=bool(getattr(args, "no_browser", False)),
                    timeout=getattr(args, "timeout", None),
                )
                _login_xai_oauth(
                    mock_args,
                    PROVIDER_REGISTRY["xai-oauth"],
                    force_new_login=True,
                )
            except SystemExit:
                print("Login cancelled or failed.")
                return
            except Exception as exc:
                print(f"Login failed: {exc}")
                return
        elif choice == "cancel":
            return
    else:
        print("Not logged into xAI Grok OAuth (SuperGrok / Premium+). Starting login...")
        print()
        try:
            mock_args = argparse.Namespace(
                no_browser=bool(getattr(args, "no_browser", False)),
                timeout=getattr(args, "timeout", None),
            )
            _login_xai_oauth(mock_args, PROVIDER_REGISTRY["xai-oauth"])
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return

    # Resolve a usable base URL.  ``resolve_xai_oauth_runtime_credentials``
    # only reads from the auth.json singleton — but credentials may legitimately
    # live only in the pool (e.g. after ``hermes auth add xai-oauth``).  Fall
    # back to the default base URL in that case so the model picker still
    # completes successfully instead of bailing out with
    # ``Could not resolve xAI OAuth credentials``.
    base_url = DEFAULT_XAI_OAUTH_BASE_URL
    try:
        creds = resolve_xai_oauth_runtime_credentials()
        base_url = (creds.get("base_url") or "").strip().rstrip("/") or base_url
    except Exception:
        pass

    models = list(_PROVIDER_MODELS.get("xai-oauth") or _PROVIDER_MODELS.get("xai") or [])
    selected = _prompt_model_selection(models, current_model=current_model or (models[0] if models else "grok-build-0.1"))
    if selected:
        _save_model_choice(selected)
        _update_config_for_provider("xai-oauth", base_url)
        print(f"Default model set to: {selected} (via xAI Grok OAuth — SuperGrok / Premium+)")
    else:
        print("No change.")

def _model_flow_qwen_oauth(_config, current_model=""):
    """Qwen OAuth provider: reuse local Qwen CLI login, then pick model."""
    from hermes_cli.main import _DEFAULT_QWEN_PORTAL_MODELS
    from hermes_cli.auth import (
        get_qwen_auth_status,
        resolve_qwen_runtime_credentials,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        DEFAULT_QWEN_BASE_URL,
    )
    from hermes_cli.models import fetch_api_models

    status = get_qwen_auth_status()
    if not status.get("logged_in"):
        print("Not logged into Qwen CLI OAuth.")
        print("Run: qwen auth qwen-oauth")
        auth_file = status.get("auth_file")
        if auth_file:
            print(f"Expected credentials file: {auth_file}")
        if status.get("error"):
            print(f"Error: {status.get('error')}")
        return

    # Try live model discovery, fall back to curated list.
    models = None
    try:
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=True)
        models = fetch_api_models(creds["api_key"], creds["base_url"])
    except Exception:
        pass
    if not models:
        models = list(_DEFAULT_QWEN_PORTAL_MODELS)

    default = current_model or (models[0] if models else "qwen3-coder-plus")
    selected = _prompt_model_selection(
        models,
        current_model=default,
        confirm_provider="qwen-oauth",
        confirm_base_url=DEFAULT_QWEN_BASE_URL,
    )
    if selected:
        _save_model_choice(selected)
        _update_config_for_provider("qwen-oauth", DEFAULT_QWEN_BASE_URL)
        print(f"Default model set to: {selected} (via Qwen OAuth)")
    else:
        print("No change.")

def _model_flow_minimax_oauth(config, current_model="", args=None):
    """MiniMax OAuth provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_provider_auth_state,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        resolve_minimax_oauth_runtime_credentials,
        AuthError,
        format_auth_error,
        _login_minimax_oauth,
        PROVIDER_REGISTRY,
    )

    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        print("Not logged into MiniMax. Starting OAuth login...")
        print()
        try:
            mock_args = argparse.Namespace(
                region=getattr(args, "region", None) or "global",
                no_browser=bool(getattr(args, "no_browser", False)),
                timeout=getattr(args, "timeout", None) or 15.0,
            )
            _login_minimax_oauth(mock_args, PROVIDER_REGISTRY["minimax-oauth"])
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return

    try:
        creds = resolve_minimax_oauth_runtime_credentials()
    except AuthError as exc:
        print(format_auth_error(exc))
        return

    from hermes_cli.models import _PROVIDER_MODELS

    model_ids = _PROVIDER_MODELS.get("minimax-oauth", [])
    selected = _prompt_model_selection(
        model_ids,
        current_model,
        confirm_provider="minimax-oauth",
        confirm_base_url=creds["base_url"],
    )
    if not selected:
        return
    _save_model_choice(selected)
    _update_config_for_provider("minimax-oauth", creds["base_url"])
    print(f"\u2713 Using MiniMax model: {selected}")


def _model_flow_custom(config):
    """Custom endpoint: collect URL, API key, and model name.

    Automatically saves the endpoint to ``custom_providers`` in config.yaml
    so it appears in the provider menu on subsequent runs.
    """
    from hermes_cli.main import _auto_provider_name, _prompt_custom_api_mode_selection, _save_custom_provider
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import get_env_value, load_config, save_config
    from hermes_cli.secret_prompt import masked_secret_prompt

    current_url = get_env_value("OPENAI_BASE_URL") or ""
    current_key = get_env_value("OPENAI_API_KEY") or ""

    print("Custom OpenAI-compatible endpoint configuration:")
    if current_url:
        print(f"  Current URL: {current_url}")
    if current_key:
        print(f"  Current key: {current_key[:8]}...")
    print()

    try:
        base_url = input(
            f"API base URL [{current_url or 'e.g. https://api.example.com/v1'}]: "
        ).strip()
        api_key = masked_secret_prompt(
            f"API key [{current_key[:8] + '...' if current_key else 'optional'}]: "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not base_url and not current_url:
        print("No URL provided. Cancelled.")
        return

    # Validate URL format
    effective_url = base_url or current_url
    if not effective_url.startswith(("http://", "https://")):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return

    effective_key = api_key or current_key

    # Hint: most local model servers (Ollama, vLLM, llama.cpp) require /v1
    # in the base URL for OpenAI-compatible chat completions.  Prompt the
    # user if the URL looks like a local server without /v1.
    _url_lower = effective_url.rstrip("/").lower()
    _looks_local = any(
        h in _url_lower
        for h in ("localhost", "127.0.0.1", "0.0.0.0", ":11434", ":8080", ":5000")
    )
    if _looks_local and not _url_lower.endswith("/v1"):
        print()
        print("  Hint: Did you mean to add /v1 at the end?")
        print("  Most local model servers (Ollama, vLLM, llama.cpp) require it.")
        print(f"  e.g. {effective_url.rstrip('/')}/v1")
        try:
            _add_v1 = input("  Add /v1? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            _add_v1 = "n"
        if _add_v1 in {"", "y", "yes"}:
            effective_url = effective_url.rstrip("/") + "/v1"
            if base_url:
                base_url = effective_url
            print(f"  Updated URL: {effective_url}")
        print()

    from hermes_cli.models import probe_api_models

    probe = probe_api_models(effective_key, effective_url)
    if probe.get("used_fallback") and probe.get("resolved_base_url"):
        print(
            f"Warning: endpoint verification worked at {probe['resolved_base_url']}/models, "
            f"not the exact URL you entered. Saving the working base URL instead."
        )
        effective_url = probe["resolved_base_url"]
        if base_url:
            base_url = effective_url
    elif probe.get("models") is not None:
        print(
            f"Verified endpoint via {probe.get('probed_url')} "
            f"({len(probe.get('models') or [])} model(s) visible)"
        )
    else:
        print(
            f"Warning: could not verify this endpoint via {probe.get('probed_url')}. "
            f"Hermes will still save it."
        )
        if probe.get("suggested_base_url"):
            suggested = probe["suggested_base_url"]
            if suggested.endswith("/v1"):
                print(
                    f"  If this server expects /v1 in the path, try base URL: {suggested}"
                )
            else:
                print(f"  If /v1 should not be in the base URL, try: {suggested}")

    # Prompt for API compatibility mode explicitly so codex-compatible custom
    # providers don't silently fall back to chat_completions.
    current_model_cfg = config.get("model")
    current_api_mode = ""
    if isinstance(current_model_cfg, dict):
        current_api_mode = str(current_model_cfg.get("api_mode") or "").strip()
    api_mode = _prompt_custom_api_mode_selection(
        effective_url,
        current_api_mode=current_api_mode,
    )
    if api_mode:
        print(f"  API mode: {api_mode}")
    else:
        print("  API mode: auto-detect")

    # Select model — use probe results when available, fall back to manual input
    model_name = ""
    detected_models = probe.get("models") or []
    try:
        if len(detected_models) == 1:
            print(f"  Detected model: {detected_models[0]}")
            confirm = input("  Use this model? [Y/n]: ").strip().lower()
            if confirm in {"", "y", "yes"}:
                model_name = detected_models[0]
            else:
                model_name = input("Model name (e.g. gpt-4, llama-3-70b): ").strip()
        elif len(detected_models) > 1:
            print("  Available models:")
            for i, m in enumerate(detected_models, 1):
                print(f"    {i}. {m}")
            pick = input(
                f"  Select model [1-{len(detected_models)}] or type name: "
            ).strip()
            if pick.isdigit() and 1 <= int(pick) <= len(detected_models):
                model_name = detected_models[int(pick) - 1]
            elif pick:
                model_name = pick
        else:
            model_name = input("Model name (e.g. gpt-4, llama-3-70b): ").strip()

        context_length_str = input(
            "Context length in tokens [leave blank for auto-detect]: "
        ).strip()

        # Prompt for a display name — shown in the provider menu on future runs
        default_name = _auto_provider_name(effective_url)
        display_name = input(f"Display name [{default_name}]: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    context_length = None
    if context_length_str:
        try:
            context_length = int(
                context_length_str.replace(",", "")
                .replace("k", "000")
                .replace("K", "000")
            )
            if context_length <= 0:
                context_length = None
        except ValueError:
            print(f"Invalid context length: {context_length_str} — will auto-detect.")
            context_length = None

    if model_name:
        _save_model_choice(model_name)

        # Update config and deactivate any OAuth provider
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "custom"
        model["base_url"] = effective_url
        if effective_key:
            model["api_key"] = effective_key
        if api_mode:
            model["api_mode"] = api_mode
        else:
            model.pop("api_mode", None)
        save_config(cfg)
        deactivate_provider()

        # Sync the caller's config dict so the setup wizard's final
        # save_config(config) preserves our model settings.  Without
        # this, the wizard overwrites model.provider/base_url with
        # the stale values from its own config dict (#4172).
        config["model"] = dict(model)

        print(f"Default model set to: {model_name} (via {effective_url})")
    else:
        if base_url or api_key:
            deactivate_provider()
        # Even without a model name, persist the custom endpoint on the
        # caller's config dict so the setup wizard doesn't lose it.
        _caller_model = config.get("model")
        if not isinstance(_caller_model, dict):
            _caller_model = {"default": _caller_model} if _caller_model else {}
        _caller_model["provider"] = "custom"
        _caller_model["base_url"] = effective_url
        if effective_key:
            _caller_model["api_key"] = effective_key
        if api_mode:
            _caller_model["api_mode"] = api_mode
        else:
            _caller_model.pop("api_mode", None)
        config["model"] = _caller_model
        print("Endpoint saved. Use `/model` in chat or `hermes model` to set a model.")

    # Auto-save to custom_providers so it appears in the menu next time
    _save_custom_provider(
        effective_url,
        effective_key,
        model_name or "",
        context_length=context_length,
        name=display_name,
        api_mode=api_mode,
    )
    _prune_replaced_custom_model_config_credentials(
        effective_url,
        provider_name=display_name,
    )


def _model_flow_azure_foundry(config, current_model=""):
    """Azure Foundry provider: configure endpoint, auth mode, API mode, and model.

    Azure Foundry supports both OpenAI-style (``/v1/chat/completions``) and
    Anthropic-style (``/v1/messages``) endpoints, and two authentication
    modes:

    * **API key** (default) — uses ``AZURE_FOUNDRY_API_KEY`` from .env.
    * **Microsoft Entra ID** — keyless, RBAC-based auth via the
      ``azure-identity`` SDK (Managed Identity / Workload Identity / az
      login / VS Code / azd / service principal env vars). Works on both
      OpenAI-style and Anthropic-style endpoints — Microsoft RBAC is
      per-resource and the same ``Azure AI User`` role grants
      both. For OpenAI-style the OpenAI SDK's native callable
      ``api_key=`` contract is used; for Anthropic-style an
      ``httpx.Client`` with a request event hook (built by
      :func:`agent.azure_identity_adapter.build_bearer_http_client`)
      mints a fresh JWT per request because the Anthropic SDK does not
      accept a callable ``auth_token`` natively.

    The wizard auto-detects the transport and available models when
    possible:

    * URLs ending in ``/anthropic`` → Anthropic Messages API.
    * Successful ``GET <base>/models`` probe → OpenAI-style + populates
      a picker with the returned deployment / model IDs.
    * Anthropic Messages probe fallback when ``/models`` fails.
    * Manual entry when every probe fails (private endpoints, etc.).

    Context lengths for the chosen model are resolved via the standard
    :func:`agent.model_metadata.get_model_context_length` chain
    (models.dev, provider metadata, hardcoded family fallbacks).
    """
    from hermes_cli.auth import _save_model_choice, deactivate_provider  # noqa: F401
    from hermes_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from hermes_cli import azure_detect

    # ── Load current Azure Foundry configuration ─────────────────────
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict) and model_cfg.get("provider") == "azure-foundry":
        current_base_url = str(model_cfg.get("base_url", "") or "")
        current_api_mode = str(model_cfg.get("api_mode", "") or "")
        current_auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        _cur_entra = model_cfg.get("entra") or {}
        current_entra = _cur_entra if isinstance(_cur_entra, dict) else {}
    else:
        current_base_url = ""
        current_api_mode = ""
        current_auth_mode = "api_key"
        current_entra = {}

    current_api_key = get_env_value("AZURE_FOUNDRY_API_KEY") or ""

    print()
    print("Azure Foundry Configuration")
    print("=" * 50)
    print()
    print("Azure Foundry can host models with either OpenAI-style or")
    print("Anthropic-style API endpoints.  Hermes will probe your")
    print("endpoint to auto-detect the transport and the deployed")
    print("models when possible.")
    print()

    if current_base_url:
        print(f"  Current endpoint:  {current_base_url}")
    if current_api_mode:
        _lbl = (
            "OpenAI-style"
            if current_api_mode == "chat_completions"
            else "Anthropic-style"
        )
        print(f"  Current API mode:  {_lbl}")
    if current_auth_mode == "entra_id":
        print("  Current auth mode: Microsoft Entra ID (keyless)")
    elif current_api_key:
        print(f"  Current auth mode: API key ({current_api_key[:8]}...)")
    print()

    # ── Step 1: endpoint URL ─────────────────────────────────────────
    try:
        _placeholder = (
            current_base_url
            or "e.g. https://<resource>.openai.azure.com/openai/v1 "
              "or https://<resource>.services.ai.azure.com/anthropic"
        )
        base_url = input(
            f"API endpoint URL [{_placeholder}]: "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    effective_url = (base_url or current_base_url).rstrip("/")
    if not effective_url:
        print("No endpoint URL provided. Cancelled.")
        return
    if not effective_url.startswith(("http://", "https://")):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return

    # ── Step 2: authentication mode ──────────────────────────────────
    print()
    print("Authentication:")
    print("  1. API key                  (AZURE_FOUNDRY_API_KEY in .env)")
    print("  2. Microsoft Entra ID       (managed identity / workload identity / az login)")
    print("     Recommended by Microsoft. Works for both OpenAI-style and Anthropic-style endpoints.")
    print("     Requires the 'Azure AI User' role on the Foundry resource.")
    try:
        _auth_default = "2" if current_auth_mode == "entra_id" else "1"
        auth_choice = (
            input(f"Authentication mode [1/2] ({_auth_default}): ").strip()
            or _auth_default
        )
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return
    use_entra = auth_choice == "2"
    auth_mode_label = "entra_id" if use_entra else "api_key"

    # ── Step 3: credentials (key OR Entra preflight) ─────────────────
    effective_key: str = ""
    entra_overrides: dict = {}
    token_provider = None  # callable when entra
    entra_scope = ""

    if use_entra:
        try:
            from agent.azure_identity_adapter import (
                EntraIdentityConfig,
                SCOPE_AI_AZURE_DEFAULT,
                build_token_provider,
                describe_active_credential,
                has_azure_identity_installed,
            )
        except ImportError as exc:
            print()
            print(f"⚠ Could not import azure-identity adapter: {exc}")
            print("  Falling back to API key auth.")
            use_entra = False
            auth_mode_label = "api_key"

    if use_entra:
        print()
        if not has_azure_identity_installed():
            print("◐ The 'azure-identity' package is not installed yet.")
            print(
                "  Hermes will install it now (the preflight below "
                "triggers the lazy-install). To skip lazy installs, "
                "run:  pip install azure-identity"
            )

        # Preserve only the optional scope override. Identity selection
        # (tenant, user-assigned MI, workload identity, service principal)
        # stays in Azure SDK env vars such as AZURE_CLIENT_ID.
        _persisted_scope_override = str(current_entra.get("scope") or "").strip()
        entra_scope = _persisted_scope_override or SCOPE_AI_AZURE_DEFAULT

        entra_overrides = {}
        if _persisted_scope_override:
            entra_overrides["scope"] = _persisted_scope_override

        print()
        print("◐ Probing Microsoft Entra ID credential chain (up to 10s)...")
        _config = EntraIdentityConfig(
            scope=entra_scope,
        )
        info = describe_active_credential(config=_config, timeout_seconds=10.0)
        if info.get("ok"):
            env_sources = info.get("env_sources") or []
            tag = ", ".join(env_sources) if env_sources else "default chain"
            print(f"✓ Entra ID token acquired ({tag}, scope={entra_scope})")
        else:
            err = info.get("error") or "credential chain exhausted"
            hint = info.get("hint") or (
                "Run `az login`, attach a managed identity to this VM, or "
                "set AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET."
            )
            print(f"⚠ {err}")
            print(f"  Hint: {hint}")
            try:
                ans = input("Save Entra config anyway and validate later? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return
            if ans and ans not in ("y", "yes"):
                print("Cancelled.")
                return

        # Build the token provider for the detection probe (best-effort —
        # if the credential chain failed above, this will silently return
        # None inside azure_detect and the probe falls back to manual).
        try:
            token_provider = build_token_provider(config=_config)
        except Exception as exc:
            print(f"⚠ Could not build token provider for probing: {exc}")
            token_provider = None
    else:
        print()
        from hermes_cli.secret_prompt import masked_secret_prompt

        try:
            api_key = masked_secret_prompt(
                f"API key [{current_api_key[:8] + '...' if current_api_key else 'required'}]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return

        effective_key = api_key or current_api_key
        if not effective_key:
            print("No API key provided. Cancelled.")
            return

    # ── Step 4: auto-detect transport + models ───────────────────────
    print()
    print("◐ Probing endpoint to auto-detect transport and models...")
    detection = azure_detect.detect(
        effective_url,
        api_key=effective_key,
        token_provider=token_provider,
    )

    discovered_models: list[str] = list(detection.models)
    api_mode: str = detection.api_mode or ""

    if api_mode:
        mode_label = (
            "OpenAI-style" if api_mode == "chat_completions" else "Anthropic-style"
        )
        print(f"✓ Detected API transport: {mode_label}")
        if detection.reason:
            print(f"    ({detection.reason})")
        if discovered_models:
            print(
                f"✓ Found {len(discovered_models)} deployed model(s) on this endpoint"
            )
    else:
        print(f"⚠ Auto-detection incomplete: {detection.reason}")
        print()
        print("Select the API format your Azure Foundry endpoint uses:")
        print("  1. OpenAI-style  (POST /v1/chat/completions)")
        print("     For: GPT models, Llama, Mistral, and most open models")
        print("  2. Anthropic-style  (POST /v1/messages)")
        print("     For: Claude models deployed via Anthropic API format")
        try:
            default_choice = "2" if current_api_mode == "anthropic_messages" else "1"
            mode_choice = (
                input(f"API format [1/2] ({default_choice}): ").strip()
                or default_choice
            )
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
        api_mode = "anthropic_messages" if mode_choice == "2" else "chat_completions"

    # ── Step 5: model name ───────────────────────────────────────────
    print()
    effective_model = ""
    if discovered_models:
        print("Available models on this endpoint:")
        for i, mid in enumerate(discovered_models[:30], start=1):
            print(f"  {i:>2}. {mid}")
        if len(discovered_models) > 30:
            print(
                f"  ... and {len(discovered_models) - 30} more (type name manually if not shown)"
            )
        print()
        try:
            pick = input(
                f"Pick by number, or type a deployment name [{current_model or discovered_models[0]}]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
        if not pick:
            effective_model = current_model or discovered_models[0]
        elif pick.isdigit() and 1 <= int(pick) <= min(len(discovered_models), 30):
            effective_model = discovered_models[int(pick) - 1]
        else:
            effective_model = pick
    else:
        try:
            model_name = input(
                f"Model / deployment name [{current_model or 'e.g. gpt-5.4, claude-sonnet-4-6'}]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
        effective_model = model_name or current_model

    if not effective_model:
        print("No model name provided. Cancelled.")
        return

    # ── Step 6: context-length lookup ────────────────────────────────
    ctx_len = azure_detect.lookup_context_length(
        effective_model,
        effective_url,
        api_key=effective_key,
        token_provider=token_provider,
    )

    # ── Step 7: persist ──────────────────────────────────────────────
    if not use_entra:
        save_env_value("AZURE_FOUNDRY_API_KEY", effective_key)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model

    model["provider"] = "azure-foundry"
    model["base_url"] = effective_url
    model["api_mode"] = api_mode
    model["default"] = effective_model
    model["auth_mode"] = auth_mode_label
    clear_model_endpoint_credentials(model, clear_api_mode=False)
    if use_entra:
        # Persist only the non-default Entra scope so config.yaml stays tidy.
        # Azure identity selection stays in standard AZURE_* env vars.
        clean_entra: dict = {}
        for key in ("scope",):
            val = entra_overrides.get(key)
            if val:
                clean_entra[key] = val
        if clean_entra:
            model["entra"] = clean_entra
        elif "entra" in model:
            del model["entra"]
    else:
        if "entra" in model:
            del model["entra"]
    if ctx_len:
        model["context_length"] = ctx_len

    save_config(cfg)
    deactivate_provider()
    config["model"] = dict(model)

    # Clear any conflicting env vars so auxiliary clients don't poison
    # themselves with a stale OpenAI base URL / key.
    if get_env_value("OPENAI_BASE_URL"):
        save_env_value("OPENAI_BASE_URL", "")
    if get_env_value("OPENAI_API_KEY"):
        save_env_value("OPENAI_API_KEY", "")

    mode_label = "OpenAI-style" if api_mode == "chat_completions" else "Anthropic-style"
    auth_label = (
        "Microsoft Entra ID (keyless)" if use_entra else "API key"
    )
    print()
    print("✓ Azure Foundry configured:")
    print(f"    Endpoint:       {effective_url}")
    print(f"    API mode:       {mode_label}")
    print(f"    Auth:           {auth_label}")
    print(f"    Model:          {effective_model}")
    if ctx_len:
        print(f"    Context length: {ctx_len:,} tokens")
    else:
        print("    Context length: not auto-detected (will fall back at runtime)")
    print()

def _model_flow_named_custom(config, provider_info):
    """Handle a named custom provider from config.yaml custom_providers list.

    Always probes the endpoint's /models API to let the user pick a model.
    If a model was previously saved, it is pre-selected in the menu.
    Falls back to the saved model if probing fails.
    """
    from hermes_cli.main import _custom_provider_api_key_config_value, _custom_provider_base_url_config_value, _save_custom_provider
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import load_config, save_config
    from hermes_cli.models import fetch_api_models

    name = provider_info["name"]
    base_url = provider_info["base_url"]
    api_mode = provider_info.get("api_mode", "")
    api_key = provider_info.get("api_key", "")
    key_env = provider_info.get("key_env", "")
    saved_model = provider_info.get("model", "")
    provider_key = (provider_info.get("provider_key") or "").strip()

    # Resolve key from env var if api_key not set directly
    if not api_key and key_env:
        api_key = os.environ.get(key_env, "")
    config_api_key = _custom_provider_api_key_config_value(provider_info, api_key)

    # Honor ``discover_models: false`` (default True) — when discovery is
    # disabled, use the configured ``models:`` list verbatim and skip the
    # live /models probe. This lets operators restrict the picker to the
    # subset their plan actually serves instead of the endpoint's full
    # catalog (#18726: Baidu Qianfan returns 100+ models for a 2-3 model
    # plan). Same semantics as the slash-command picker (model_switch.py
    # sections 3 & 4): default discovers, false keeps the explicit list.
    discover = provider_info.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    configured_models: list[str] = []
    cfg_models = provider_info.get("models", {})
    if isinstance(cfg_models, dict):
        configured_models = [str(m) for m in cfg_models if str(m).strip()]
    elif isinstance(cfg_models, list):
        configured_models = [
            str(m) for m in cfg_models if isinstance(m, str) and m.strip()
        ]

    print(f"  Provider: {name}")
    print(f"  URL:      {base_url}")
    if saved_model:
        print(f"  Current:  {saved_model}")
    print()

    if not discover and configured_models:
        # Discovery disabled with an explicit list — use it verbatim, no probe.
        print(f"Using configured models (discover_models: false): {len(configured_models)}")
        models = configured_models
    else:
        print("Fetching available models...")
        fetch_kwargs = {"timeout": 8.0}
        if api_mode:
            fetch_kwargs["api_mode"] = api_mode
        models = fetch_api_models(api_key, base_url, **fetch_kwargs)
        # If the probe came back empty but the operator configured an explicit
        # list, fall back to it rather than forcing manual entry.
        if not models and configured_models:
            models = configured_models

    if models:
        default_idx = 0
        if saved_model and saved_model in models:
            default_idx = models.index(saved_model)

        print(f"Found {len(models)} model(s):\n")
        try:
            from hermes_cli.curses_ui import curses_radiolist

            menu_items = [
                f"{m} (current)" if m == saved_model else m for m in models
            ] + ["Cancel"]
            idx = curses_radiolist(
                f"Select model from {name}:",
                menu_items,
                selected=default_idx,
                cancel_returns=-1,
                searchable=True,
            )
            print()
            if idx < 0 or idx >= len(models):
                print("Cancelled.")
                return
            model_name = models[idx]
        except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
            for i, m in enumerate(models, 1):
                suffix = " (current)" if m == saved_model else ""
                print(f"  {i}. {m}{suffix}")
            print(f"  {len(models) + 1}. Cancel")
            print()
            try:
                val = input(f"Choice [1-{len(models) + 1}]: ").strip()
                if not val:
                    print("Cancelled.")
                    return
                idx = int(val) - 1
                if idx < 0 or idx >= len(models):
                    print("Cancelled.")
                    return
                model_name = models[idx]
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return
    elif saved_model:
        print("Could not fetch models from endpoint.")
        try:
            model_name = input(f"Model name [{saved_model}]: ").strip() or saved_model
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
    else:
        print("Could not fetch models from endpoint. Enter model name manually.")
        try:
            model_name = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
        if not model_name:
            print("No model specified. Cancelled.")
            return

    # Activate and save the model to the custom_providers entry
    _save_model_choice(model_name)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    if provider_key:
        model["provider"] = "custom:" + provider_key.strip().lower().replace(" ", "-")
        model.pop("base_url", None)
        model.pop("api_key", None)
    else:
        model["provider"] = "custom"
        model["base_url"] = _custom_provider_base_url_config_value(
            provider_info, base_url
        )
        if config_api_key:
            model["api_key"] = config_api_key
    # Apply api_mode from custom_providers entry, or clear stale value
    custom_api_mode = provider_info.get("api_mode", "")
    if custom_api_mode:
        model["api_mode"] = custom_api_mode
    else:
        model.pop("api_mode", None)  # let runtime auto-detect from URL
    save_config(cfg)
    deactivate_provider()

    # Persist the selected model back to whichever schema owns this endpoint.
    if provider_key:
        cfg = load_config()
        providers_cfg = cfg.get("providers")
        if isinstance(providers_cfg, dict):
            provider_entry = providers_cfg.get(provider_key)
            if isinstance(provider_entry, dict):
                provider_entry["default_model"] = model_name
                # Only persist an inline api_key when the user originally had
                # one (either a literal secret or a ``${VAR}`` template). When
                # the entry relies on ``key_env``, do not synthesize a
                # ``${key_env}`` api_key — the runtime already resolves the
                # key from ``key_env`` directly, and writing the resolved
                # secret (or even a synthesized template) would silently
                # downgrade credential hygiene on entries that intentionally
                # keep plaintext out of ``config.yaml``. See issue #15803.
                original_api_key_ref = str(
                    provider_info.get("api_key_ref", "") or ""
                ).strip()
                original_api_key = str(provider_info.get("api_key", "") or "").strip()
                had_inline_api_key = bool(original_api_key_ref or original_api_key)
                if (
                    had_inline_api_key
                    and config_api_key
                    and not str(provider_entry.get("api_key", "") or "").strip()
                ):
                    provider_entry["api_key"] = config_api_key
                if key_env and not str(provider_entry.get("key_env", "") or "").strip():
                    provider_entry["key_env"] = key_env
                cfg["providers"] = providers_cfg
                save_config(cfg)
    else:
        # Save model name to the custom_providers entry for next time
        _save_custom_provider(base_url, config_api_key, model_name, api_mode=api_mode)

    print(f"\n✅ Model set to: {model_name}")
    print(f"   Provider: {name} ({base_url})")

def _model_flow_copilot(config, current_model=""):
    """GitHub Copilot flow using env vars, gh CLI, or OAuth device code."""
    from hermes_cli.main import _current_reasoning_effort, _prompt_reasoning_effort_selection, _set_reasoning_effort
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
        resolve_api_key_provider_credentials,
    )
    from hermes_cli.config import save_env_value, load_config, save_config
    from hermes_cli.models import (
        _PROVIDER_MODELS,
        fetch_api_models,
        fetch_github_model_catalog,
        github_model_reasoning_efforts,
        copilot_model_api_mode,
        normalize_copilot_model_id,
    )

    provider_id = "copilot"
    pconfig = PROVIDER_REGISTRY[provider_id]

    creds = resolve_api_key_provider_credentials(provider_id)
    api_key = creds.get("api_key", "")
    source = creds.get("source", "")

    if not api_key:
        print("No GitHub token configured for GitHub Copilot.")
        print()
        print("  Supported token types:")
        print(
            "    → OAuth token (gho_*)          via `copilot login` or device code flow"
        )
        print("    → Fine-grained PAT (github_pat_*)  with Copilot Requests permission")
        print("    → GitHub App token (ghu_*)     via environment variable")
        print("    ✗ Classic PAT (ghp_*)          NOT supported by Copilot API")
        print()
        print("  Options:")
        print("    1. Login with GitHub (OAuth device code flow)")
        print("    2. Enter a token manually")
        print("    3. Cancel")
        print()
        try:
            choice = input("  Choice [1-3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == "1":
            try:
                from hermes_cli.copilot_auth import copilot_device_code_login

                token = copilot_device_code_login()
                if token:
                    save_env_value("COPILOT_GITHUB_TOKEN", token)
                    print("  Copilot token saved.")
                    print()
                else:
                    print("  Login cancelled or failed.")
                    return
            except Exception as exc:
                print(f"  Login failed: {exc}")
                return
        elif choice == "2":
            from hermes_cli.secret_prompt import masked_secret_prompt

            try:
                new_key = masked_secret_prompt("  Token (COPILOT_GITHUB_TOKEN): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not new_key:
                print("  Cancelled.")
                return
            # Validate token type
            try:
                from hermes_cli.copilot_auth import validate_copilot_token

                valid, msg = validate_copilot_token(new_key)
                if not valid:
                    print(f"  ✗ {msg}")
                    return
            except ImportError:
                pass
            save_env_value("COPILOT_GITHUB_TOKEN", new_key)
            print("  Token saved.")
            print()
        else:
            print("  Cancelled.")
            return

        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = creds.get("api_key", "")
        source = creds.get("source", "")
    else:
        if source in {"GITHUB_TOKEN", "GH_TOKEN"}:
            from hermes_cli.env_loader import format_secret_source_suffix
            bw_suffix = format_secret_source_suffix(source)
            print(f"  GitHub token: {api_key[:8]}... ✓ ({source}{bw_suffix})")
        elif source == "gh auth token":
            print("  GitHub token: ✓ (from `gh auth token`)")
        else:
            print("  GitHub token: ✓")
        print()

    effective_base = pconfig.inference_base_url

    catalog = fetch_github_model_catalog(api_key)
    live_models = (
        [item.get("id", "") for item in catalog if item.get("id")]
        if catalog
        else fetch_api_models(api_key, effective_base)
    )
    normalized_current_model = (
        normalize_copilot_model_id(
            current_model,
            catalog=catalog,
            api_key=api_key,
        )
        or current_model
    )
    if live_models:
        model_list = [model_id for model_id in live_models if model_id]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print(
                "  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults."
            )
            print('    Use "Enter custom model name" if you do not see your model.')

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=normalized_current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=api_key,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        selected = (
            normalize_copilot_model_id(
                selected,
                catalog=catalog,
                api_key=api_key,
            )
            or selected
        )
        initial_cfg = load_config()
        current_effort = _current_reasoning_effort(initial_cfg)
        reasoning_efforts = github_model_reasoning_efforts(
            selected,
            catalog=catalog,
            api_key=api_key,
        )
        selected_effort = None
        if reasoning_efforts:
            print(f"  {selected} supports reasoning controls.")
            selected_effort = _prompt_reasoning_effort_selection(
                reasoning_efforts, current_effort=current_effort
            )

        _save_model_choice(selected)

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model["api_mode"] = copilot_model_api_mode(
            selected,
            catalog=catalog,
            api_key=api_key,
        )
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        if selected_effort is not None:
            _set_reasoning_effort(cfg, selected_effort)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via {pconfig.name})")
        if reasoning_efforts:
            if selected_effort == "none":
                print("Reasoning disabled for this model.")
            elif selected_effort:
                print(f"Reasoning effort set to: {selected_effort}")
    else:
        print("No change.")

def _model_flow_copilot_acp(config, current_model=""):
    """GitHub Copilot ACP flow using the local Copilot CLI."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
        get_external_process_provider_status,
        resolve_api_key_provider_credentials,
        resolve_external_process_provider_credentials,
    )
    from hermes_cli.models import (
        _PROVIDER_MODELS,
        fetch_github_model_catalog,
        normalize_copilot_model_id,
    )
    from hermes_cli.config import load_config, save_config

    del config

    provider_id = "copilot-acp"
    pconfig = PROVIDER_REGISTRY[provider_id]

    status = get_external_process_provider_status(provider_id)
    resolved_command = (
        status.get("resolved_command") or status.get("command") or "copilot"
    )
    effective_base = status.get("base_url") or pconfig.inference_base_url

    print("  GitHub Copilot ACP delegates Hermes turns to `copilot --acp`.")
    print("  Hermes currently starts its own ACP subprocess for each request.")
    print("  Hermes uses your selected model as a hint for the Copilot ACP session.")
    print(f"  Command: {resolved_command}")
    print(f"  Backend marker: {effective_base}")
    print()

    try:
        creds = resolve_external_process_provider_credentials(provider_id)
    except Exception as exc:
        print(f"  ⚠ {exc}")
        print(
            "  Set HERMES_COPILOT_ACP_COMMAND or COPILOT_CLI_PATH if Copilot CLI is installed elsewhere."
        )
        return

    effective_base = creds.get("base_url") or effective_base

    catalog_api_key = ""
    try:
        catalog_creds = resolve_api_key_provider_credentials("copilot")
        catalog_api_key = catalog_creds.get("api_key", "")
    except Exception:
        pass

    catalog = fetch_github_model_catalog(catalog_api_key)
    normalized_current_model = (
        normalize_copilot_model_id(
            current_model,
            catalog=catalog,
            api_key=catalog_api_key,
        )
        or current_model
    )

    if catalog:
        model_list = [item.get("id", "") for item in catalog if item.get("id")]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
    else:
        model_list = _PROVIDER_MODELS.get("copilot", [])
        if model_list:
            print(
                "  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults."
            )
            print('    Use "Enter custom model name" if you do not see your model.')

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=normalized_current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=catalog_api_key,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if not selected:
        print("No change.")
        return

    selected = (
        normalize_copilot_model_id(
            selected,
            catalog=catalog,
            api_key=catalog_api_key,
        )
        or selected
    )
    _save_model_choice(selected)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    model["provider"] = provider_id
    model["base_url"] = effective_base
    model["api_mode"] = "chat_completions"
    clear_model_endpoint_credentials(model, clear_api_mode=False)
    save_config(cfg)
    deactivate_provider()

    print(f"Default model set to: {selected} (via {pconfig.name})")

def _model_flow_devin_acp(config, current_model=""):
    """Devin CLI ACP flow using the local Devin CLI (`devin acp`)."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
        get_external_process_provider_status,
        resolve_external_process_provider_credentials,
    )
    from hermes_cli.models import _PROVIDER_MODELS
    from hermes_cli.config import load_config, save_config

    del config

    provider_id = "devin-acp"
    pconfig = PROVIDER_REGISTRY[provider_id]

    status = get_external_process_provider_status(provider_id)
    resolved_command = (
        status.get("resolved_command") or status.get("command") or "devin"
    )
    effective_base = status.get("base_url") or pconfig.inference_base_url

    print("  Devin CLI ACP delegates Hermes turns to `devin acp`.")
    print("  Hermes keeps the ACP process warm and continues sessions when history grows.")
    print("  Authenticate with: devin auth login")
    print(f"  Command: {resolved_command}")
    print(f"  Backend marker: {effective_base}")
    if status.get("auth_present") is True:
        print("  Local credentials: detected")
    elif status.get("auth_present") is False:
        print("  Local credentials: missing")
    if status.get("hint"):
        print(f"  ⚠ {status['hint']}")
    print()

    try:
        creds = resolve_external_process_provider_credentials(provider_id)
    except Exception as exc:
        print(f"  ⚠ {exc}")
        print(
            "  Install Devin CLI (https://docs.devin.ai/cli) or set "
            "HERMES_DEVIN_ACP_COMMAND / DEVIN_CLI_PATH."
        )
        return

    effective_base = creds.get("base_url") or effective_base
    # Prefer live Devin CLI discovery (invalid --model → Available: …), then
    # curated snapshot so the CLI `hermes model` flow matches Desktop picker.
    try:
        from hermes_cli.models import cached_provider_model_ids

        model_list = cached_provider_model_ids(provider_id) or _PROVIDER_MODELS.get(
            "devin-acp", ["devin-acp"]
        )
    except Exception:
        model_list = _PROVIDER_MODELS.get("devin-acp", ["devin-acp"])
    default_pick = current_model or (model_list[0] if model_list else "devin-acp")
    selected = _prompt_model_selection(
        model_list,
        current_model=default_pick,
        confirm_provider=provider_id,
        confirm_base_url=effective_base,
        confirm_api_key="",
    )
    if not selected:
        try:
            selected = input("Model name [devin-acp]: ").strip() or "devin-acp"
        except (KeyboardInterrupt, EOFError):
            selected = None

    if not selected:
        print("No change.")
        return

    _save_model_choice(selected)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    model["provider"] = provider_id
    model["base_url"] = effective_base
    model["api_mode"] = "chat_completions"
    clear_model_endpoint_credentials(model, clear_api_mode=False)
    save_config(cfg)
    deactivate_provider()

    print(f"Default model set to: {selected} (via {pconfig.name})")

def _model_flow_kimi(config, current_model=""):
    """Kimi / Moonshot model selection with automatic endpoint routing.

    - sk-kimi-* keys   → api.kimi.com/coding/v1  (Kimi Coding Plan)
    - Other keys        → api.moonshot.ai/v1      (legacy Moonshot)

    No manual base URL prompt — endpoint is determined by key prefix.
    """
    from hermes_cli.main import _prompt_api_key
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        KIMI_CODE_BASE_URL,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    provider_id = "kimi-coding"
    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""

    # Step 1: Check / prompt for API key
    existing_key = ""
    for ev in pconfig.api_key_env_vars:
        existing_key = get_env_value(ev) or os.getenv(ev, "")
        if existing_key:
            break

    existing_key, abort = _prompt_api_key(
        pconfig, existing_key, provider_id=provider_id
    )
    if abort:
        return

    # Step 2: Auto-detect endpoint from key prefix
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

    # Step 3: Model selection — show appropriate models for the endpoint
    model_list = _PROVIDER_MODELS.get("kimi-coding" if is_coding_plan else "moonshot", [])

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("Enter model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Update config with provider and base URL
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model.pop("api_mode", None)  # let runtime auto-detect from URL
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        save_config(cfg)
        deactivate_provider()

        endpoint_label = "Kimi Coding" if is_coding_plan else "Moonshot"
        print(f"Default model set to: {selected} (via {endpoint_label})")
    else:
        print("No change.")

def _model_flow_stepfun(config, current_model=""):
    """StepFun Step Plan flow with region-specific endpoints."""
    from hermes_cli.main import _infer_stepfun_region, _prompt_api_key, _prompt_provider_choice, _stepfun_base_url_for_region
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from hermes_cli.models import _PROVIDER_MODELS, fetch_api_models

    provider_id = "stepfun"
    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""

    existing_key = ""
    for ev in pconfig.api_key_env_vars:
        existing_key = get_env_value(ev) or os.getenv(ev, "")
        if existing_key:
            break

    existing_key, abort = _prompt_api_key(
        pconfig, existing_key, provider_id=provider_id
    )
    if abort:
        return

    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    if not current_base:
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            current_base = str(model_cfg.get("base_url") or "").strip()
    current_region = _infer_stepfun_region(current_base or pconfig.inference_base_url)

    region_choices = [
        (
            "international",
            f"International ({_stepfun_base_url_for_region('international')})",
        ),
        ("china", f"China ({_stepfun_base_url_for_region('china')})"),
    ]
    ordered_regions = []
    for region_key, label in region_choices:
        if region_key == current_region:
            ordered_regions.insert(0, (region_key, f"{label}  ← currently active"))
        else:
            ordered_regions.append((region_key, label))
    ordered_regions.append(("cancel", "Cancel"))

    region_idx = _prompt_provider_choice([label for _, label in ordered_regions])
    if region_idx is None or ordered_regions[region_idx][0] == "cancel":
        print("No change.")
        return

    selected_region = ordered_regions[region_idx][0]
    effective_base = _stepfun_base_url_for_region(selected_region)
    if base_url_env:
        save_env_value(base_url_env, effective_base)

    live_models = fetch_api_models(existing_key, effective_base)
    if live_models:
        model_list = live_models
        print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print(
                f"  Could not auto-detect models from {pconfig.name} API — "
                "showing Step Plan fallback catalog."
            )

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model.pop("api_mode", None)
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        save_config(cfg)
        deactivate_provider()

        config["model"] = dict(model)
        print(f"Default model set to: {selected} (via {pconfig.name})")
    else:
        print("No change.")

def _model_flow_bedrock_api_key(config, region, current_model=""):
    """Bedrock API Key mode — uses the OpenAI-compatible bedrock-mantle endpoint.

    For developers who don't have an AWS account but received a Bedrock API Key
    from their AWS admin. Works like any OpenAI-compatible endpoint.
    """
    from hermes_cli.auth import (
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import (
        load_config,
        save_config,
        get_env_value,
        save_env_value,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    mantle_base_url = f"https://bedrock-mantle.{region}.api.aws/v1"

    # Prompt for API key
    existing_key = get_env_value("AWS_BEARER_TOKEN_BEDROCK") or ""
    if existing_key:
        from hermes_cli.env_loader import format_secret_source_suffix
        source_suffix = format_secret_source_suffix("AWS_BEARER_TOKEN_BEDROCK")
        print(f"  Bedrock API Key: {existing_key[:12]}... ✓{source_suffix}")
    else:
        print(f"  Endpoint: {mantle_base_url}")
        print()
        from hermes_cli.secret_prompt import masked_secret_prompt

        try:
            api_key = masked_secret_prompt("  Bedrock API Key: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not api_key:
            print("  Cancelled.")
            return
        save_env_value("AWS_BEARER_TOKEN_BEDROCK", api_key)
        existing_key = api_key
        print("  ✓ API key saved.")
    print()

    # Model selection — use static list (mantle doesn't need boto3 for discovery)
    model_list = _PROVIDER_MODELS.get("bedrock", [])
    print(f"  Showing {len(model_list)} curated models")

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider="custom",
            confirm_base_url=mantle_base_url,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("  Model ID: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Save as custom provider pointing to bedrock-mantle
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "custom"
        model["base_url"] = mantle_base_url
        model.pop("api_mode", None)  # chat_completions is the default
        clear_model_endpoint_credentials(model, clear_api_mode=False)

        # Also save region in bedrock config for reference
        bedrock_cfg = cfg.get("bedrock", {})
        if not isinstance(bedrock_cfg, dict):
            bedrock_cfg = {}
        bedrock_cfg["region"] = region
        cfg["bedrock"] = bedrock_cfg

        # Save the API key env var name so hermes knows where to find it
        save_env_value("OPENAI_API_KEY", existing_key)
        save_env_value("OPENAI_BASE_URL", mantle_base_url)

        save_config(cfg)
        deactivate_provider()

        print(f"  Default model set to: {selected} (via Bedrock API Key, {region})")
        print(f"  Endpoint: {mantle_base_url}")
    else:
        print("  No change.")

def _model_flow_bedrock(config, current_model=""):
    """AWS Bedrock provider: verify credentials, pick region, discover models.

    Uses the native Converse API via boto3 — not the OpenAI-compatible endpoint.
    Auth is handled by the AWS SDK default credential chain (env vars, profile,
    instance role), so no API key prompt is needed.
    """
    from hermes_cli.auth import (
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import load_config, save_config
    from hermes_cli.models import _PROVIDER_MODELS

    # 1. Check for AWS credentials
    try:
        from agent.bedrock_adapter import (
            has_aws_credentials,
            resolve_aws_auth_env_var,
            resolve_bedrock_region,
            discover_bedrock_models,
        )
    except ImportError:
        print("  ✗ boto3 is not installed. Install it with:")
        print("    pip install boto3")
        print()
        return

    if not has_aws_credentials():
        print("  ⚠ No AWS credentials detected via environment variables.")
        print("  Bedrock will use boto3's default credential chain (IMDS, SSO, etc.)")
        print()

    auth_var = resolve_aws_auth_env_var()
    if auth_var:
        print(f"  AWS credentials: {auth_var} ✓")
    else:
        print("  AWS credentials: boto3 default chain (instance role / SSO)")
    print()

    # 2. Region selection
    current_region = resolve_bedrock_region()
    try:
        region_input = input(f"  AWS Region [{current_region}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    region = region_input or current_region

    # 2b. Authentication mode
    print("  Choose authentication method:")
    print()
    print("    1. IAM credential chain (recommended)")
    print("       Works with EC2 instance roles, SSO, env vars, aws configure")
    print("    2. Bedrock API Key")
    print("       Enter your Bedrock API Key directly — also supports")
    print("       team scenarios where an admin distributes keys")
    print()
    try:
        auth_choice = input("  Choice [1]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return

    if auth_choice == "2":
        _model_flow_bedrock_api_key(config, region, current_model)
        return

    # 3. Model discovery — try live API first, fall back to static list
    print(f"  Discovering models in {region}...")
    live_models = discover_bedrock_models(region)

    if live_models:
        _EXCLUDE_PREFIXES = (
            "stability.",
            "cohere.embed",
            "twelvelabs.",
            "us.stability.",
            "us.cohere.embed",
            "us.twelvelabs.",
            "global.cohere.embed",
            "global.twelvelabs.",
        )
        _EXCLUDE_SUBSTRINGS = ("safeguard", "voxtral", "palmyra-vision")
        filtered = []
        for m in live_models:
            mid = m["id"]
            if any(mid.startswith(p) for p in _EXCLUDE_PREFIXES):
                continue
            if any(s in mid.lower() for s in _EXCLUDE_SUBSTRINGS):
                continue
            filtered.append(m)

        # Deduplicate: prefer inference profiles (us.*, global.*) over bare
        # foundation model IDs.
        profile_base_ids = set()
        for m in filtered:
            mid = m["id"]
            if mid.startswith(("us.", "global.")):
                base = mid.split(".", 1)[1] if "." in mid[3:] else mid
                profile_base_ids.add(base)

        deduped = []
        for m in filtered:
            mid = m["id"]
            if not mid.startswith(("us.", "global.")) and mid in profile_base_ids:
                continue
            deduped.append(m)

        _RECOMMENDED = [
            "us.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-opus-4-6",
            "us.anthropic.claude-haiku-4-5",
            "us.amazon.nova-pro",
            "us.amazon.nova-lite",
            "us.amazon.nova-micro",
            "deepseek.v3",
            "us.meta.llama4-maverick",
            "us.meta.llama4-scout",
        ]

        def _sort_key(m):
            mid = m["id"]
            for i, rec in enumerate(_RECOMMENDED):
                if mid.startswith(rec):
                    return (0, i, mid)
            if mid.startswith("global."):
                return (1, 0, mid)
            return (2, 0, mid)

        deduped.sort(key=_sort_key)
        model_list = [m["id"] for m in deduped]
        print(
            f"  Found {len(model_list)} text model(s) (filtered from {len(live_models)} total)"
        )
    else:
        model_list = _PROVIDER_MODELS.get("bedrock", [])
        if model_list:
            print(
                f"  Using {len(model_list)} curated models (live discovery unavailable)"
            )
        else:
            print(
                "  No models found. Check IAM permissions for bedrock:ListFoundationModels."
            )
            return

    # 4. Model selection
    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider="bedrock",
            confirm_base_url=f"https://bedrock-runtime.{region}.amazonaws.com",
        )
    else:
        try:
            selected = input("  Model ID: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "bedrock"
        model["base_url"] = f"https://bedrock-runtime.{region}.amazonaws.com"
        model.pop("api_mode", None)  # bedrock_converse is auto-detected
        clear_model_endpoint_credentials(model, clear_api_mode=False)

        bedrock_cfg = cfg.get("bedrock", {})
        if not isinstance(bedrock_cfg, dict):
            bedrock_cfg = {}
        bedrock_cfg["region"] = region
        cfg["bedrock"] = bedrock_cfg

        save_config(cfg)
        deactivate_provider()

        print(f"  Default model set to: {selected} (via AWS Bedrock, {region})")
    else:
        print("  No change.")


def _model_flow_vertex(config, current_model=""):
    """Google Vertex AI provider: Gemini via the OpenAI-compatible endpoint.

    Auth is OAuth2 — short-lived tokens minted from a service-account JSON or
    Application Default Credentials (ADC). No static API key. The credential
    *path* lives in .env (VERTEX_CREDENTIALS_PATH / GOOGLE_APPLICATION_CREDENTIALS);
    project ID and region are non-secret and saved to config.yaml under vertex:.
    """
    from hermes_cli.auth import (
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import load_config, save_config, get_env_value
    from hermes_cli.models import _PROVIDER_MODELS

    # 1. Credential source detection (fast, no network / no google-auth import).
    sa_path = (
        get_env_value("VERTEX_CREDENTIALS_PATH")
        or get_env_value("GOOGLE_APPLICATION_CREDENTIALS")
        or ""
    ).strip()
    if sa_path:
        print(f"  Vertex credentials: service account JSON ({sa_path}) ✓")
    else:
        print("  Vertex credentials: Application Default Credentials (ADC)")
        print("    Vertex uses OAuth2, not a static API key. Either:")
        print("      • run 'gcloud auth application-default login', or")
        print("      • set VERTEX_CREDENTIALS_PATH in ~/.hermes/.env to a service account JSON")
    print()

    cfg = load_config()
    vertex_cfg = cfg.get("vertex")
    if not isinstance(vertex_cfg, dict):
        vertex_cfg = {}

    # 2. Project ID (optional — falls back to the project embedded in creds).
    current_project = str(vertex_cfg.get("project_id") or "").strip()
    try:
        project_input = input(
            f"  GCP project ID [{current_project or 'from credentials'}]: "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    project_id = project_input or current_project

    # 3. Region (default global — required for the Gemini 3.x previews).
    current_region = str(vertex_cfg.get("region") or "global").strip() or "global"
    try:
        region_input = input(f"  Vertex region [{current_region}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    region = region_input or current_region

    # 4. Model selection (curated list — Vertex has no /models listing route).
    model_list = _PROVIDER_MODELS.get("vertex", []) or [
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    ]
    base_url_preview = (
        "https://aiplatform.googleapis.com/v1beta1/projects/<project>/"
        f"locations/{region}/endpoints/openapi"
        if region == "global"
        else f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/<project>/"
        f"locations/{region}/endpoints/openapi"
    )
    selected = _prompt_model_selection(
        model_list,
        current_model=current_model,
        confirm_provider="vertex",
        confirm_base_url=base_url_preview,
    )

    if selected:
        _save_model_choice(selected)

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "vertex"
        # base_url is computed at runtime from project+region; do not pin it.
        model.pop("base_url", None)
        model.pop("api_mode", None)  # chat_completions is the profile default
        clear_model_endpoint_credentials(model, clear_api_mode=False)

        vcfg = cfg.get("vertex")
        if not isinstance(vcfg, dict):
            vcfg = {}
        vcfg["project_id"] = project_id
        vcfg["region"] = region
        cfg["vertex"] = vcfg

        save_config(cfg)
        deactivate_provider()

        print(f"  Default model set to: {selected} (via Google Vertex AI, {region})")
    else:
        print("  No change.")

def _select_zai_endpoint(current_base: str) -> str:
    """Present a picker for Z.AI endpoint selection during setup.

    Offers the four official Z.AI endpoints (Global, China, Coding Plan
    Global, Coding Plan China) plus a custom-proxy option.  The list is
    sourced from ``ZAI_ENDPOINTS`` in ``hermes_cli.auth`` so it stays in
    sync with the probe list.

    Returns the selected base URL.  Falls back to *current_base* on cancel
    or error.
    """
    from hermes_cli.main import _prompt_provider_choice
    from hermes_cli.auth import ZAI_ENDPOINTS

    # Build label + URL pairs from the shared endpoint list.
    options = [(label, url) for _, url, _, label in ZAI_ENDPOINTS]
    normalized_current = (current_base or "").strip().rstrip("/")

    # Default to the currently-active option if it matches one of the
    # known endpoints; otherwise default to the first (Global).
    default_idx = 0
    for idx, (_, url) in enumerate(options):
        if normalized_current == url.rstrip("/"):
            default_idx = idx
            break
    else:
        if normalized_current:
            # A custom URL is active — offer "Custom proxy" as the default.
            default_idx = len(options)

    choices = [f"{label} ({url})" for label, url in options]
    choices.append("Custom proxy URL")

    selected = _prompt_provider_choice(
        choices,
        default=default_idx,
        title="Select Z.AI / GLM endpoint:",
    )
    if selected is None:
        return current_base

    if selected == len(options):
        # Custom proxy URL
        try:
            override = input(f"Custom base URL [{current_base}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return current_base
        if not override:
            return current_base
        if not override.startswith(("http://", "https://")):
            print("  Invalid URL — must start with http:// or https://. Keeping current value.")
            return current_base
        return override.rstrip("/")

    return options[selected][1].rstrip("/")


def _model_flow_api_key_provider(config, provider_id, current_model=""):
    """Generic flow for API-key providers (z.ai, MiniMax, OpenCode, etc.)."""
    from hermes_cli.main import _prompt_api_key
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from hermes_cli.models import (
        _PROVIDER_MODELS,
        fetch_api_models,
        opencode_model_api_mode,
        normalize_opencode_model_id,
    )

    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""

    # Check / prompt for API key
    existing_key = ""
    for ev in pconfig.api_key_env_vars:
        existing_key = get_env_value(ev) or os.getenv(ev, "")
        if existing_key:
            break

    existing_key, abort = _prompt_api_key(
        pconfig, existing_key, provider_id=provider_id
    )
    if abort:
        return

    # Gemini free-tier gate: free-tier daily quotas (<= 250 RPD for Flash)
    # are exhausted in a handful of agent turns, so refuse to wire up the
    # provider with a free-tier key. Probe is best-effort; network or auth
    # errors fall through without blocking.
    if provider_id == "gemini" and existing_key:
        try:
            from agent.gemini_native_adapter import probe_gemini_tier
        except Exception:
            probe_gemini_tier = None
        if probe_gemini_tier is not None:
            print("  Checking Gemini API tier...")
            probe_base = (
                (get_env_value(base_url_env) if base_url_env else "")
                or os.getenv(base_url_env or "", "")
                or pconfig.inference_base_url
            )
            tier = probe_gemini_tier(existing_key, probe_base)
            if tier == "free":
                print()
                print(
                    "❌ This Google API key is on the free tier "
                    "(<= 250 requests/day for gemini-2.5-flash)."
                )
                print(
                    "   Hermes typically makes 3-10 API calls per user turn "
                    "(tool iterations + auxiliary tasks),"
                )
                print(
                    "   so the free tier is exhausted after a handful of "
                    "messages and cannot sustain"
                )
                print("   an agent session.")
                print()
                print(
                    "   To use Gemini with Hermes, enable billing on your "
                    "Google Cloud project and regenerate"
                )
                print(
                    "   the key in a billing-enabled project: "
                    "https://aistudio.google.com/apikey"
                )
                print()
                print(
                    "   Alternatives with workable free usage: DeepSeek, "
                    "OpenRouter (free models), Groq, Nous."
                )
                print()
                print("Not saving Gemini as the default provider.")
                return
            if tier == "paid":
                print("  Tier check: paid ✓")
            else:
                # "unknown" -- network issue, auth problem, unexpected response.
                # Don't block; the runtime 429 handler will surface free-tier
                # guidance if the key turns out to be free tier.
                print("  Tier check: could not verify (proceeding anyway).")
            print()

    # Optional base URL override.
    # Precedence: env var → config.yaml model.base_url → registry default.
    # Reading config.yaml prevents silently overwriting a saved remote URL
    # (e.g. a remote LM Studio endpoint) with localhost when the user just
    # presses Enter at the prompt below.
    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    if not current_base:
        try:
            _m = load_config().get("model") or {}
            if str(_m.get("provider") or "").strip().lower() == provider_id:
                current_base = str(_m.get("base_url") or "").strip()
        except Exception:
            pass
    effective_base = current_base or pconfig.inference_base_url

    if provider_id == "zai":
        # Z.AI has four official endpoints (Global, China, Coding Plan
        # Global, Coding Plan China) with separate billing paths.  Present
        # a picker instead of a plain text input so users can explicitly
        # choose the endpoint that matches their key type.
        chosen_base = _select_zai_endpoint(effective_base)
        if chosen_base and chosen_base != effective_base and base_url_env:
            save_env_value(base_url_env, chosen_base)
        effective_base = chosen_base
    else:
        try:
            override = input(f"Base URL [{effective_base}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            override = ""
        if override and base_url_env:
            if not override.startswith(("http://", "https://")):
                print(
                    "  Invalid URL — must start with http:// or https://. Keeping current value."
                )
            else:
                save_env_value(base_url_env, override)
                effective_base = override

    # Model selection — resolution order:
    #   1. models.dev registry (cached, filtered for agentic/tool-capable models)
    #   2. Curated static fallback list (offline insurance)
    #   3. Live /models endpoint probe (small providers without models.dev data)
    #
    # LM Studio: live /api/v1/models probe (no models.dev catalog).
    # Ollama Cloud: merged discovery (live API + models.dev + disk cache).
    if provider_id == "lmstudio":
        from hermes_cli.auth import AuthError
        from hermes_cli.models import fetch_lmstudio_models

        api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")
        try:
            model_list = fetch_lmstudio_models(
                api_key=api_key_for_probe, base_url=effective_base
            )
        except AuthError as exc:
            print(f"  LM Studio rejected the request: {exc}")
            print("  Set LM_API_KEY (or update it) to match the server's bearer token.")
            model_list = []
        if model_list:
            print(f"  Found {len(model_list)} model(s) from LM Studio")
    elif provider_id == "ollama-cloud":
        from hermes_cli.models import fetch_ollama_cloud_models

        api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")
        # During setup, force a live refresh so the picker reflects newly
        # released models (e.g. deepseek v4 flash, kimi k2.6) the moment
        # the user enters their key — not an hour later when the disk
        # cache TTL expires.
        model_list = fetch_ollama_cloud_models(
            api_key=api_key_for_probe,
            base_url=effective_base,
            force_refresh=True,
        )
        if model_list:
            print(f"  Found {len(model_list)} model(s) from Ollama Cloud")
    elif provider_id == "novita":
        from hermes_cli.models import fetch_api_models

        api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")
        curated = _PROVIDER_MODELS.get(provider_id, [])
        live_models = fetch_api_models(api_key_for_probe, effective_base)
        if live_models:
            model_list = live_models
            print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
        else:
            mdev_models: list = []
            try:
                from agent.models_dev import list_agentic_models

                mdev_models = list_agentic_models(provider_id)
            except Exception:
                pass
            if mdev_models:
                seen = {m.lower() for m in mdev_models}
                model_list = list(mdev_models)
                for m in curated:
                    if m.lower() not in seen:
                        model_list.append(m)
                        seen.add(m.lower())
                print(f"  Found {len(model_list)} model(s) from models.dev registry")
            else:
                model_list = curated
                if model_list:
                    print(
                        f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.'
                    )
    else:
        curated = _PROVIDER_MODELS.get(provider_id, [])

        # Try models.dev first — returns tool-capable models, filtered for noise
        mdev_models: list = []
        try:
            from agent.models_dev import list_agentic_models

            mdev_models = list_agentic_models(provider_id)
        except Exception:
            pass

        if mdev_models:
            # Merge models.dev with curated list so newly added models
            # (not yet in models.dev) still appear in the picker.
            if curated:
                seen = {m.lower() for m in mdev_models}
                merged = list(mdev_models)
                for m in curated:
                    if m.lower() not in seen:
                        merged.append(m)
                        seen.add(m.lower())
                model_list = merged
            else:
                model_list = mdev_models
            print(f"  Found {len(model_list)} model(s) from models.dev registry")
        elif curated and len(curated) >= 8:
            # Curated list is substantial — use it directly, skip live probe
            model_list = curated
            print(
                f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.'
            )
        else:
            api_key_for_probe = existing_key or (
                get_env_value(key_env) if key_env else ""
            )
            live_models = fetch_api_models(api_key_for_probe, effective_base)
            if live_models and len(live_models) >= len(curated):
                model_list = live_models
                print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
            else:
                model_list = curated
                if model_list:
                    print(
                        f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.'
                    )
            # else: no defaults either, will fall through to raw input

    if provider_id in {"opencode-zen", "opencode-go"}:
        model_list = [
            normalize_opencode_model_id(provider_id, mid) for mid in model_list
        ]
        current_model = normalize_opencode_model_id(provider_id, current_model)
        model_list = list(dict.fromkeys(mid for mid in model_list if mid))

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        if provider_id in {"opencode-zen", "opencode-go"}:
            selected = normalize_opencode_model_id(provider_id, selected)

        _save_model_choice(selected)

        # Update config with provider, base URL, and provider-specific API mode
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        if provider_id in {"opencode-zen", "opencode-go"}:
            model["api_mode"] = opencode_model_api_mode(provider_id, selected)
        else:
            model.pop("api_mode", None)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via {pconfig.name})")
    else:
        print("No change.")

def _model_flow_anthropic(config, current_model=""):
    """Flow for Anthropic provider — OAuth subscription, API key, or Claude Code creds."""
    from hermes_cli.main import _run_anthropic_oauth_flow
    from hermes_cli.auth import (
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from hermes_cli.config import (
        save_env_value,
        load_config,
        save_config,
        save_anthropic_api_key,
    )
    from hermes_cli.models import _PROVIDER_MODELS

    # Check ALL credential sources
    from hermes_cli.auth import get_anthropic_key

    existing_key = get_anthropic_key()
    cc_available = False
    try:
        from agent.anthropic_adapter import (
            read_claude_code_credentials,
            is_claude_code_token_valid,
            _is_oauth_token,
        )

        cc_creds = read_claude_code_credentials()
        if cc_creds and is_claude_code_token_valid(cc_creds):
            cc_available = True
    except Exception:
        pass

    # Stale-OAuth guard: if the only existing cred is an expired OAuth token
    # (no valid cc_creds to fall back on), treat it as missing so the re-auth
    # path is offered instead of silently accepting a broken token.
    existing_is_stale_oauth = False
    if existing_key and _is_oauth_token(existing_key) and not cc_available:
        existing_is_stale_oauth = True

    has_creds = (bool(existing_key) and not existing_is_stale_oauth) or cc_available
    needs_auth = not has_creds

    if has_creds:
        # Show what we found
        if existing_key:
            from hermes_cli.env_loader import format_secret_source_suffix
            from hermes_cli.auth import PROVIDER_REGISTRY

            # Surface which env var supplied the key so users with
            # Bitwarden see "(from Bitwarden)" — without this, a detected
            # BSM key looks identical to a key in .env and users assume
            # nothing is wired up.
            source_suffix = ""
            for var in PROVIDER_REGISTRY["anthropic"].api_key_env_vars:
                if os.getenv(var, "").strip() == existing_key:
                    source_suffix = format_secret_source_suffix(var)
                    if source_suffix:
                        break
            print(
                f"  Anthropic credentials: {existing_key[:12]}... ✓{source_suffix}"
            )
        elif cc_available:
            print("  Claude Code credentials: ✓ (auto-detected)")
        print()
        choice = _prompt_auth_credentials_choice("Anthropic credentials:")

        if choice == "reauth":
            needs_auth = True
        elif choice == "cancel":
            return
        # choice == "use" or default: use existing, proceed to model selection

    if needs_auth:
        # Show auth method choice
        print()
        print("  Choose authentication method:")
        print()
        print("    1. Claude Pro/Max subscription (OAuth login)")
        print("    2. Anthropic API key (pay-per-token)")
        print("    3. Cancel")
        print()
        try:
            choice = input("  Choice [1/2/3]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == "1":
            if not _run_anthropic_oauth_flow(save_env_value):
                return

        elif choice == "2":
            print()
            print("  Get an API key at: https://platform.claude.com/settings/keys")
            print()
            from hermes_cli.secret_prompt import masked_secret_prompt

            try:
                api_key = masked_secret_prompt("  API key (sk-ant-...): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return
            if not api_key:
                print("  Cancelled.")
                return
            save_anthropic_api_key(api_key, save_fn=save_env_value)
            print("  ✓ API key saved.")

        else:
            print("  No change.")
            return
    print()

    # Model selection
    model_list = _PROVIDER_MODELS.get("anthropic", [])
    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider="anthropic",
        )
    else:
        try:
            selected = input("Model name (e.g., claude-sonnet-4-20250514): ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Update config with provider — clear base_url since
        # resolve_runtime_provider() always hardcodes Anthropic's URL.
        # Leaving a stale base_url in config can contaminate other
        # providers if the user switches without running 'hermes model'.
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "anthropic"
        model.pop("base_url", None)
        clear_model_endpoint_credentials(model)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via Anthropic)")
    else:
        print("No change.")
