"""Custom OpenAI-compatible endpoint wizards: the ad-hoc ``custom`` flow and the
``custom_providers`` / ``providers.<key>`` named-endpoint flow.

Imports of hermes_cli.main / auth / config / models stay lazy (main.py import cycle;
tests patch them at call time). Prompt strings and config write order are behavior.
"""

from __future__ import annotations

import contextlib
import os
import urllib.parse

from hermes_cli.cli_output import line_input
from hermes_cli.providers import custom_provider_slug
from hermes_cli.model_setup_flows_common import (
    _HTTP, _ask, _commit_model_config, _load_config_model_section,
    _prune_replaced_custom_model_config_credentials, _radiolist, _say)


def _parse_context_length(text: str):
    """``128k`` / ``128,000`` -> int; None when blank, non-positive, or unparsable (warns)."""
    if not text:
        return None
    try:
        value = int(text.replace(",", "").replace("k", "000").replace("K", "000"))
    except ValueError:
        print(f"Invalid context length: {text} — will auto-detect.")
        return None
    return value if value > 0 else None


def _probe_custom_endpoint(effective_key: str, effective_url: str) -> tuple[dict, str]:
    """Verify a custom endpoint via ``probe_api_models`` and report; returns
    ``(probe, effective_url)`` where the URL may be the working fallback base."""
    from hermes_cli.models import probe_api_models
    probe = probe_api_models(effective_key, effective_url)
    if probe.get("used_fallback") and probe.get("resolved_base_url"):
        print(f"Warning: endpoint verification worked at {probe['resolved_base_url']}/models, "
              f"not the exact URL you entered. Saving the working base URL instead.")
        effective_url = probe["resolved_base_url"]
    elif probe.get("models") is not None:
        print(f"Verified endpoint via {probe.get('probed_url')} ({len(probe.get('models') or [])} model(s) visible)")
    else:
        print(f"Warning: could not verify this endpoint via {probe.get('probed_url')}. Hermes will still save it.")
        suggested = probe.get("suggested_base_url")
        if suggested and suggested.endswith("/v1"):
            print(f"  If this server expects /v1 in the path, try base URL: {suggested}")
        elif suggested:
            print(f"  If /v1 should not be in the base URL, try: {suggested}")
    return probe, effective_url


def _pick_detected_model(detected_models: list) -> str:
    """Model-name step of the custom flow: confirm a single detection, number-pick from
    several, or type one. Raises KeyboardInterrupt/EOFError like the prompts it wraps."""
    manual = "Model name (e.g. gpt-4, llama-3-70b): "
    if len(detected_models) == 1:
        print(f"  Detected model: {detected_models[0]}")
        if input("  Use this model? [Y/n]: ").strip().lower() in {"", "y", "yes"}:
            return detected_models[0]
        return line_input(manual).strip()
    if len(detected_models) > 1:
        print("  Available models:")
        for i, m in enumerate(detected_models, 1):
            print(f"    {i}. {m}")
        pick = input(f"  Select model [1-{len(detected_models)}] or type name: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(detected_models):
            return detected_models[int(pick) - 1]
        return pick
    return line_input(manual).strip()


def _model_flow_custom(config):
    """Custom endpoint: collect URL, API key, and model name; also saved to ``custom_providers`` so
    it appears in the provider menu on subsequent runs."""
    from hermes_cli.main_provider_setup import _auto_provider_name, _prompt_custom_api_mode_selection, _save_custom_provider
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import custom_endpoint_key_env, get_env_value, save_env_value
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
        base_url = line_input(f"API base URL [{current_url or 'e.g. https://api.example.com/v1'}]: ").strip()
        api_key = masked_secret_prompt(f"API key [{current_key[:8] + '...' if current_key else 'optional'}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not base_url and not current_url:
        print("No URL provided. Cancelled.")
        return
    effective_url = base_url or current_url
    if not effective_url.startswith(_HTTP):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return
    effective_key = api_key or current_key

    # Most local servers (Ollama, vLLM, llama.cpp) need /v1 for OpenAI-compatible
    # chat completions — offer to append it when the URL looks local without it.
    _url_lower = effective_url.rstrip("/").lower()
    _looks_local = any(h in _url_lower for h in ("localhost", "127.0.0.1", "0.0.0.0", ":11434", ":8080", ":5000"))
    if _looks_local and not _url_lower.endswith("/v1"):
        _say("", "  Hint: Did you mean to add /v1 at the end?",
             "  Most local model servers (Ollama, vLLM, llama.cpp) require it.", f"  e.g. {effective_url.rstrip('/')}/v1")
        if _ask("  Add /v1? [Y/n]: ", raw=True, cancel_msg=None, on_cancel="n").lower() in {"", "y", "yes"}:
            effective_url = effective_url.rstrip("/") + "/v1"
            print(f"  Updated URL: {effective_url}")
        print()

    probe, effective_url = _probe_custom_endpoint(effective_key, effective_url)

    # Ask for the API mode explicitly so codex-compatible custom providers don't
    # silently fall back to chat_completions.
    current_model_cfg = config.get("model")
    current_api_mode = str(current_model_cfg.get("api_mode") or "").strip() if isinstance(current_model_cfg, dict) else ""
    api_mode = _prompt_custom_api_mode_selection(effective_url, current_api_mode=current_api_mode)
    print(f"  API mode: {api_mode}" if api_mode else "  API mode: auto-detect")

    # Select model — use probe results when available, fall back to manual input
    try:
        model_name = _pick_detected_model(probe.get("models") or [])
        context_length_str = line_input("Context length in tokens [leave blank for auto-detect]: ").strip()
        # Display name — shown in the provider menu on future runs
        default_name = _auto_provider_name(effective_url)
        display_name = line_input(f"Display name [{default_name}]: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return
    context_length = _parse_context_length(context_length_str)

    # The key goes to .env and config.yaml only references it. Keyed on host:port
    # so two servers on one machine keep separate credentials.
    # See #69449.
    custom_key_env = ""
    if effective_key:
        _parsed = urllib.parse.urlparse(effective_url)
        _identity = _parsed.hostname or ""
        if _parsed.port:
            _identity = f"{_identity}_{_parsed.port}"
        custom_key_env = custom_endpoint_key_env(_identity)
        save_env_value(custom_key_env, effective_key)
        print(f"  API key saved to .env as {custom_key_env}")

    def _apply_endpoint(model: dict) -> None:
        model["provider"] = "custom"
        model["base_url"] = effective_url
        if custom_key_env:
            model["api_key"] = f"${{{custom_key_env}}}"
        if api_mode:
            model["api_mode"] = api_mode
        else:
            model.pop("api_mode", None)

    if model_name:
        _save_model_choice(model_name)
        cfg, model = _load_config_model_section()
        _apply_endpoint(model)
        _commit_model_config(cfg)
        # Sync the caller's config dict so the setup wizard's final save_config(config)
        # doesn't overwrite model.provider/base_url with its stale values.
        config["model"] = dict(model)
        print(f"Default model set to: {model_name} (via {effective_url})")
    else:
        if base_url or api_key:
            deactivate_provider()
        # Even without a model name, persist the endpoint on the caller's config dict.
        _caller_model = config.get("model")
        if not isinstance(_caller_model, dict):
            _caller_model = {"default": _caller_model} if _caller_model else {}
        _apply_endpoint(_caller_model)
        config["model"] = _caller_model
        print("Endpoint saved. Use `/model` in chat or `hermes model` to set a model.")

    # Auto-save to custom_providers so it appears in the menu next time
    _save_custom_provider(effective_url, effective_key, model_name or "", context_length=context_length,
                          name=display_name, api_mode=api_mode, key_env=custom_key_env)
    _prune_replaced_custom_model_config_credentials(effective_url, provider_name=display_name)


def _configured_model_ids(cfg_models) -> list[str]:
    """Model ids from a ``custom_providers[].models`` mapping or list (marker keys skipped)."""
    if isinstance(cfg_models, dict):
        markers = {"__explicit_model_allowlist__", "__discovered_model_catalog__"}
        return [str(m) for m in cfg_models if m not in markers and str(m).strip()]
    out: list[str] = []
    if isinstance(cfg_models, list):
        for entry in cfg_models:
            if isinstance(entry, dict):
                model_id = str(entry.get("id") or entry.get("model") or "").strip()
            else:
                model_id = str(entry).strip() if isinstance(entry, str) else ""
            if model_id:
                out.append(model_id)
    return out


def _discover_named_custom_models(provider_info: dict, api_key: str, configured_models: list, explicit_catalog: bool):
    """Live catalog probe for a named custom endpoint (native ``/api/tags`` for Ollama).
    Returns ``(models, native_catalog_empty)``; persists the live catalog as a side effect."""
    from hermes_cli.config import normalize_extra_headers
    from hermes_cli.models import fetch_api_models, _get_ollama_native_headers
    from hermes_cli.models_local import (
        fetch_ollama_local_models,
        _normalize_openai_base_url,
        should_use_ollama_native_catalog,
    )

    name, base_url = provider_info["name"], provider_info["base_url"]
    api_mode = provider_info.get("api_mode", "")
    provider_key = (provider_info.get("provider_key") or "").strip()
    print("Fetching available models...")
    fetch_kwargs = {"timeout": 8.0}
    if api_mode:
        fetch_kwargs["api_mode"] = api_mode
    native_catalog_provider = "ollama" if provider_key.lower() == "ollama" or name.strip().lower() == "ollama" else "custom"
    extra_headers = normalize_extra_headers(provider_info.get("extra_headers")) or {}
    candidate_headers = _get_ollama_native_headers(base_url, api_key=api_key)
    for key in tuple(candidate_headers):
        if any(key.lower() == existing.lower() for existing in extra_headers):
            del candidate_headers[key]
    candidate_headers.update(extra_headers)
    caller_has_authorization = any(key.lower() == "authorization" for key in extra_headers)
    if api_key and not caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
        candidate_headers["Authorization"] = f"Bearer {api_key}"
    use_native = should_use_ollama_native_catalog(native_catalog_provider, base_url, headers=candidate_headers or None)
    native_headers_arg = candidate_headers or None if use_native else (extra_headers or None)
    native_catalog_empty = False
    if use_native:
        if explicit_catalog and configured_models:
            live_models = configured_models
        else:
            live_models = fetch_ollama_local_models(base_url, timeout=8.0, headers=native_headers_arg)
            native_catalog_empty = live_models == []
            if live_models is None:
                live_models = fetch_api_models(api_key, _normalize_openai_base_url(base_url), headers=native_headers_arg, **fetch_kwargs)
                native_catalog_empty = False
    else:
        live_models = fetch_api_models(api_key, base_url, headers=native_headers_arg, **fetch_kwargs)
    models = configured_models if explicit_catalog else [] if native_catalog_empty else (live_models or configured_models)
    # Persist the live catalog to the custom_providers entry so no-probe surfaces
    # (dashboard, desktop, ACP) show the full list; mirrors model_switch.py's
    # _save_discovered_models_to_config. A failed save is non-fatal.
    if live_models:
        with contextlib.suppress(Exception):
            from hermes_cli.model_switch_providers import _save_discovered_models_to_config
            _save_discovered_models_to_config(base_url, live_models, api_mode=api_mode, headers=extra_headers or None)
    return models, native_catalog_empty


def _pick_named_custom_model(name: str, models: list, saved_model: str):
    """Searchable radiolist over *models* (numbered prompt without curses); None = cancelled."""
    default_idx = models.index(saved_model) if saved_model and saved_model in models else 0
    print(f"Found {len(models)} model(s):\n")
    menu_items = [f"{m} (current)" if m == saved_model else m for m in models] + ["Cancel"]
    idx = _radiolist(f"Select model from {name}:", menu_items, default_idx, searchable=True)
    if idx is not None:
        print()
    else:
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}{' (current)' if m == saved_model else ''}")
        _say(f"  {len(models) + 1}. Cancel", "")
        try:
            val = input(f"Choice [1-{len(models) + 1}]: ").strip()
            if not val:
                print("Cancelled.")
                return None
            idx = int(val) - 1
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return None
    if idx < 0 or idx >= len(models):
        print("Cancelled.")
        return None
    return models[idx]


def _model_flow_named_custom(config, provider_info):
    """Named custom provider from ``custom_providers`` / ``providers.<key>``: probes the model
    catalog (native ``/api/tags`` for endpoints conservatively identified as Ollama); a previously
    saved model is pre-selected and is the fallback when probing fails."""
    from hermes_cli.main_provider_setup import _custom_provider_api_key_config_value, _custom_provider_base_url_config_value, _save_custom_provider
    from hermes_cli.auth import _save_model_choice
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_switch import _entry_models_discovered, _models_config_is_allowlist
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

    # ``discover_models: false`` (default True) uses the configured ``models:`` list
    # verbatim and skips the live probe, so operators can restrict the picker to the
    # subset their plan serves. Same semantics as the slash-command picker.
    # This lets operators restrict the picker to the subset their plan actually serves instead of the
    # endpoint's full catalog (#18726: Baidu Qianfan returns 100+ models for a 2-3 model plan).
    discover = provider_info.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    cfg_models = provider_info.get("models", {})
    explicit_catalog = _models_config_is_allowlist(cfg_models, _entry_models_discovered(provider_info))
    configured_models = _configured_model_ids(cfg_models)

    _say(f"  Provider: {name}", f"  URL:      {base_url}")
    if saved_model:
        print(f"  Current:  {saved_model}")
    print()

    native_catalog_empty = False
    if not discover:
        # Never probe. The active model is a usable sole choice, not a catalog.
        models = configured_models or ([saved_model] if saved_model else [])
        print(f"Using configured models (discover_models: false): {len(models)}")
    else:
        models, native_catalog_empty = _discover_named_custom_models(provider_info, api_key, configured_models, explicit_catalog)

    if models:
        model_name = _pick_named_custom_model(name, models, saved_model)
        if model_name is None:
            return
    elif saved_model and not native_catalog_empty:
        print("Could not fetch models from endpoint.")
        model_name = _ask(f"Model name [{saved_model}]: ")
        if model_name is None:
            return
        model_name = model_name or saved_model
    else:
        print("Could not fetch models from endpoint. Enter model name manually.")
        model_name = _ask("Model name: ")
        if model_name is None:
            return
        if not model_name:
            print("No model specified. Cancelled.")
            return

    # Activate and save the model to the custom_providers entry
    _save_model_choice(model_name)
    cfg, model = _load_config_model_section()
    if provider_key:
        model["provider"] = custom_provider_slug(name, provider_key)
        model.pop("base_url", None)
        model.pop("api_key", None)
    else:
        model["provider"] = "custom"
        model["base_url"] = _custom_provider_base_url_config_value(provider_info, base_url)
        if config_api_key:
            model["api_key"] = config_api_key
    # Apply api_mode from custom_providers entry, or clear stale value
    if api_mode:
        model["api_mode"] = api_mode
    else:
        model.pop("api_mode", None)  # let runtime auto-detect from URL
    _commit_model_config(cfg)

    # Persist the selected model back to whichever schema owns this endpoint.
    if provider_key:
        cfg = load_config()
        providers_cfg = cfg.get("providers")
        provider_entry = providers_cfg.get(provider_key) if isinstance(providers_cfg, dict) else None
        if isinstance(provider_entry, dict):
            provider_entry["default_model"] = model_name
            # Only persist an inline api_key when the user originally had one
            # (literal or ``${VAR}``). Entries relying on ``key_env`` must not get
            # a synthesized api_key — the runtime resolves key_env directly and
            # writing it would downgrade credential hygiene.
            had_inline_api_key = bool(
                str(provider_info.get("api_key_ref", "") or "").strip() or str(provider_info.get("api_key", "") or "").strip()
            )
            if had_inline_api_key and config_api_key and not str(provider_entry.get("api_key", "") or "").strip():
                provider_entry["api_key"] = config_api_key
            if key_env and not str(provider_entry.get("key_env", "") or "").strip():
                provider_entry["key_env"] = key_env
            cfg["providers"] = providers_cfg
            save_config(cfg)
    else:
        # Save model name to the custom_providers entry for next time
        _save_custom_provider(base_url, config_api_key, model_name, api_mode=api_mode)

    _say(f"\n✅ Model set to: {model_name}", f"   Provider: {name} ({base_url})")
