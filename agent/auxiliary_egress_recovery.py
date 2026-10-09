"""Recover a denied auxiliary request only through an explicitly local route."""
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from hermes_constants import get_hermes_home

from agent.llm_egress_firewall import DestinationClass, classify_destination



def local_fallback_entry(entry, *, main_runtime=None):
    """Screen the router's concrete endpoint before credentials or catalogs resolve.

    Built-in OAuth/discovery arms do not honor entry URLs. Named custom arms
    use their provider configuration instead; ordinary API-key arms honor an
    explicit URL. Never infer locality from a provider name or transport mode.
    """
    from agent import auxiliary_client as auxiliary

    raw_provider = str(entry.get("provider") or "").strip()
    if not raw_provider or not str(entry.get("model") or "").strip():
        return None
    provider = auxiliary._normalize_aux_provider(raw_provider)
    if provider in auxiliary._EXPLICIT_PROVIDER_BRANCHES and provider != "custom":
        return None
    base_url = str(entry.get("base_url") or "").strip()
    from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES, resolve_llamacpp_endpoint
    from hermes_cli.runtime_provider import _get_named_custom_provider
    named = _get_named_custom_provider(raw_provider.lower(), metadata_only=True)
    if (named is None and raw_provider.lower() != provider
            and raw_provider.lower() not in LLAMACPP_ALIASES):
        named = _get_named_custom_provider(provider, metadata_only=True)
    if raw_provider.lower() in LLAMACPP_ALIASES and not base_url and named is None:
        from hermes_cli.config import load_config_readonly
        endpoint = resolve_llamacpp_endpoint(config=load_config_readonly(), wait_for_boot_s=0) or {}
        base_url = str(endpoint.get("base_url") or "").strip()
        entry = {**entry, "api_key": entry.get("api_key") or endpoint.get("api_key") or "no-key-required"}
    elif named:
        # Current named routing composes caller endpoint overrides over saved defaults.
        base_url = base_url or str(named.get("base_url") or "").strip()
    elif provider == "custom":
        # Pin a concrete endpoint so the custom resolver cannot discover a
        # remote provider when the main/task configuration is incomplete.
        try:
            from agent.secret_scope import get_secret
            from hermes_cli.config import load_config_readonly
            from hermes_cli.runtime_provider import _config_base_url_trustworthy_for_bare_custom

            model_cfg = load_config_readonly().get("model") or {}
            configured_base = str(model_cfg.get("base_url") or "").strip() if isinstance(model_cfg, dict) else ""
            configured_provider = str(model_cfg.get("provider") or "").strip().lower() if isinstance(model_cfg, dict) else ""
            if not _config_base_url_trustworthy_for_bare_custom(configured_base, configured_provider):
                configured_base = ""
            # Match the terminal resolver; OPENAI_BASE_URL does not select its endpoint.
            # Only the explicit main candidate may reuse the session's runtime URL.
            base_url = (
                base_url
                or str((main_runtime or {}).get("base_url") or "").strip()
                or get_secret("CUSTOM_BASE_URL", "").strip()
                or configured_base
                or get_secret("OPENROUTER_BASE_URL", "").strip()
            )
        except Exception:
            logger.debug("Optional metadata or provenance operation failed", exc_info=True)
            return None
    else:
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY

            registered = PROVIDER_REGISTRY.get(provider)
            if (
                registered is None or registered.auth_type != "api_key"
                or provider in {"anthropic", "copilot", "azure-foundry"}
            ):
                return None
            env_url = (
                auxiliary._scoped_key_env(registered.base_url_env_var).strip()
                if registered.base_url_env_var else ""
            )
            # Z.AI probes remote endpoints while resolving credentials,
            # before the router applies an entry's explicit URL override.
            if provider == "zai" and classify_destination(
                provider, env_url, "chat_completions"
            ) is not DestinationClass.LOOPBACK:
                return None
            base_url = base_url or env_url or registered.inference_base_url
        except Exception:
            logger.debug("Optional metadata or provenance operation failed", exc_info=True)
            return None
    if classify_destination(provider, base_url, "chat_completions") is not DestinationClass.LOOPBACK:
        return None
    return {**entry, "base_url": base_url}



def is_local_fallback_client(client, provider):
    """Verify the resolved physical endpoint before any metadata probe."""
    base_url = getattr(client, "base_url", None)
    api_mode = getattr(client, "api_mode", None)
    return classify_destination(
        provider, str(base_url) if base_url is not None else None,
        api_mode if isinstance(api_mode, str) else None,
    ) in {DestinationClass.LOCAL_PROCESS, DestinationClass.LOOPBACK}


def local_fallback_steps(route, step_factory):
    # Resolve through the caller module so its routing/cache seams remain authoritative.
    from agent import auxiliary_client as auxiliary

    candidates = (
        lambda: try_configured_fallback_chain(
            route.task, route.resolved_provider or "auto", reason="egress blocked",
            failed_model=route.final_model, local_only=True,
        ),
        lambda: try_main_agent_model_fallback(
            route.resolved_provider, route.task, reason="egress blocked",
            failed_model=route.final_model, local_only=True,
        ),
    )
    for candidate in candidates:
        client, model, label = candidate()
        if client is None:
            continue
        provider = auxiliary._fallback_provider_from_label(label)
        destination = classify_destination(
            provider, str(getattr(client, "base_url", "") or ""), "chat_completions",
        )
        if destination not in {DestinationClass.LOCAL_PROCESS, DestinationClass.LOOPBACK}:
            continue
        auxiliary._record_route_info(route.route_info, provider, model)
        response = yield step_factory("fallback", (client, model, label))
        if response is not None:
            return response
    return None


def auxiliary_egress_binding(
    client: Any,
    *,
    provider: str | None,
    model: str | None,
    api_mode: str | None,
) -> tuple[Any, Any] | None:
    """Build the complete identity and route for protected auxiliary calls."""
    from agent import auxiliary_client as auxiliary

    normalized_provider = auxiliary._normalize_aux_provider(provider)
    from agent.llm_egress_runtime import provider_uses_egress_firewall

    if not provider_uses_egress_firewall(normalized_provider):
        return None
    from agent.source_provenance import DEFAULT_POLICY_DIGEST

    runtime = auxiliary._normalize_main_runtime(None)
    raw_runtime = auxiliary._RUNTIME_MAIN_CONTEXT.get() or {}
    relay = auxiliary._RELAY_AUX_CALL_CONTEXT.get() or {}
    request_id = str(relay.get("request_id") or f"aux-{uuid.uuid4().hex}")
    session_id = str(
        runtime.get("session_id")
        or raw_runtime.get("session_id")
        or f"aux-session:{request_id}"
    )
    turn_id = str(
        raw_runtime.get("turn_id")
        or f"{session_id}:aux:{str(relay.get('task') or 'call')}"
    )
    policy_digest = str(
        raw_runtime.get("policy_digest")
        or raw_runtime.get("llm_egress_policy_digest")
        or DEFAULT_POLICY_DIGEST
    )
    # SDK clients expose httpx.URL; authorize the auxiliary endpoint before
    # considering the main runtime's fallback endpoint.
    candidate_base_url = str(getattr(client, "base_url", "") or "")
    if not isinstance(candidate_base_url, str) or not candidate_base_url.startswith(
        ("http://", "https://")
    ):
        candidate_base_url = raw_runtime.get("base_url")
    if not isinstance(candidate_base_url, str) or not candidate_base_url.startswith(
        ("http://", "https://")
    ):
        if normalized_provider == "openai-codex":
            candidate_base_url = "https://chatgpt.com/backend-api/codex"
        elif normalized_provider == "anthropic":
            candidate_base_url = "https://api.anthropic.com/v1"
        else:
            candidate_base_url = auxiliary._NOUS_DEFAULT_BASE_URL
    base_url = candidate_base_url
    resolved_api_mode = str(
        api_mode
        or (
            "codex_responses"
            if normalized_provider == "openai-codex"
            else "chat_completions"
        )
    )
    agent_attrs = {
        "provider": normalized_provider,
        "model": str(model or ""),
        "base_url": base_url,
        "api_mode": resolved_api_mode,
        "session_id": session_id,
        "_current_turn_id": turn_id,
        "_current_api_request_id": request_id,
        "_llm_egress_policy_digest": policy_digest,
        "_llm_egress_state_dir": Path(get_hermes_home()) / "egress",
    }
    if str(relay.get("task") or "") == "compression":
        agent_attrs.update(
            _llm_egress_max_serialized_bytes=2_000_000,
            _llm_egress_max_conservative_tokens=666_667,
            _llm_egress_max_sanitized_bytes=2_000_000,
            _llm_egress_max_sanitized_segment_bytes=32_768,
            _llm_egress_preserve_segment_boundaries=True,
            _llm_egress_max_granted_serialized_bytes=2_000_000,
            _llm_egress_max_granted_conservative_tokens=666_667,
        )
    agent = SimpleNamespace(**agent_attrs)
    route = SimpleNamespace(
        provider=normalized_provider,
        model=str(model or ""),
        base_url=base_url,
        api_mode=resolved_api_mode,
    )
    return agent, route


def authorize_auxiliary_request(client: Any, kwargs: dict[str, Any], callback, *, provider: str | None, api_mode: str | None, metadata: dict[str, Any] | None):
    binding = auxiliary_egress_binding(
        client, provider=provider, model=kwargs.get("model"), api_mode=api_mode,
    )
    if binding is None:
        return callback(kwargs)
    from agent.llm_egress_runtime import dispatch_authorized_agent_request
    agent, route = binding
    return dispatch_authorized_agent_request(agent, kwargs, callback, route=route)



def local_main_supports_vision(provider, model, entry):
    """Use live-route metadata without remote catalog or credential discovery."""
    from agent import auxiliary_client as auxiliary
    from agent.image_routing import _supports_vision_override
    from agent.models_dev import get_model_capabilities
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    runtime = auxiliary._normalize_main_runtime(None)
    requested = (runtime.get("requested_provider") or "") if (
        runtime.get("provider") == provider and runtime.get("model") == model
    ) else ""
    model_cfg = cfg.get("model") or {}
    configured_model = (model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name") or "") if isinstance(model_cfg, dict) else model_cfg
    configured_provider = model_cfg.get("provider") or "auto" if isinstance(model_cfg, dict) else "auto"
    configured_route = configured_model == model and auxiliary._normalize_aux_provider(
        configured_provider) == auxiliary._normalize_aux_provider(requested or provider)
    if not configured_route:
        cfg = {**cfg, "model": {}}
    override = _supports_vision_override(cfg, provider, model, requested_provider=requested)
    if override is not None:
        return override
    from hermes_cli.local_runtime.capabilities import is_managed_provider
    from hermes_cli.local_runtime.endpoint import managed_get_json, managed_root

    endpoint = managed_root()
    if (endpoint and is_managed_provider(provider, entry["base_url"])
            and endpoint[0].rstrip("/") == entry["base_url"].removesuffix("/v1").rstrip("/")):
        try:
            modalities = managed_get_json(*endpoint, f"/props?model={model}", timeout_s=3).get("modalities")
            if isinstance(modalities, dict) and isinstance(modalities.get("vision"), bool):
                return modalities["vision"]
        except Exception:
            logger.debug("Optional metadata or provenance operation failed", exc_info=True)
            pass
    caps = get_model_capabilities(provider, model, allow_network=False)
    return True if caps is None or caps.supports_vision is None else caps.supports_vision


def try_main_agent_model_fallback(
    failed_provider: str, task: str | None = None, reason: str = "error",
    failed_model: str | None = None, failed_base_url: str = "", failure_scope: Any = None,
    local_only: bool = False,
) -> tuple[Any | None, str | None, str]:
    """Last-resort fallback to the main agent provider + model after the configured chain is exhausted.
    ``failed_model`` scoping per ``auxiliary._failed_backend_skip``; same-URL custom endpoints serve many models,
    so a hung aux model says nothing about the main model's health. Returns (client, model, label) or (None, None, "")."""
    from agent import auxiliary_client as auxiliary
    main_provider = (auxiliary._read_main_provider() or "").strip()
    main_model = (auxiliary._read_main_model() or "").strip()
    if main_provider.lower() == "moa":
        # MoA virtual provider: fall back to the preset's aggregator (the acting model).
        _agg_provider, _agg_model = auxiliary._resolve_moa_aggregator(main_model)
        if not _agg_provider or not _agg_model:
            return None, None, ""
        main_provider, main_model = _agg_provider, _agg_model
    if not main_provider or not main_model or main_provider.lower() in {"auto", ""}:
        return None, None, ""
    local_entry = None
    if local_only:
        from agent.auxiliary_egress_recovery import local_fallback_entry, local_main_supports_vision
        runtime = auxiliary._normalize_main_runtime(None)
        if (main_provider != "custom" or runtime.get("provider") != main_provider
                or runtime.get("model") != main_model or not runtime.get("base_url")):
            runtime = {}
        local_entry = local_fallback_entry(
            {"provider": main_provider, "model": main_model, "api_key": runtime.get("api_key"),
             "base_url": runtime.get("base_url")},
            main_runtime=runtime)
        if local_entry is None:
            return None, None, ""
    if task == "vision" and (
            main_provider in auxiliary._PROVIDERS_WITHOUT_VISION or not (
                local_main_supports_vision(main_provider, main_model, local_entry) if local_only
                else auxiliary._main_model_supports_vision(main_provider, main_model))):
        # Same capability gate as the auto-route (_vision_main_provider_client): handing an image to a
        # text-only main model turns a transient 429 into a guaranteed 400 (#108349).
        auxiliary.logger.info("Auxiliary vision: %s on %s — main agent provider %s accepts no image input, not falling back",
                    reason, failed_provider, main_provider)
        return None, None, ""
    main_base_url = (local_entry["base_url"] if local_only
                     else auxiliary._custom_health_base_url(main_provider))
    if auxiliary._failed_backend_skip(
            failed_provider, failed_model, failed_base_url=failed_base_url,
            failure_scope=failure_scope)(main_provider, main_model, main_base_url):
        return None, None, ""
    if auxiliary._is_provider_unhealthy(main_provider, main_base_url):
        auxiliary._log_skip_unhealthy(main_provider, task, base_url=main_base_url)
        return None, None, ""
    try:
        client, resolved_model = (auxiliary._resolve_fallback_entry(local_entry) if local_entry is not None
                                  else auxiliary.resolve_provider_client(provider=main_provider, model=main_model))
    except Exception:
        logger.debug("Optional metadata or provenance operation failed", exc_info=True)
        client, resolved_model = None, None
    if client is None:
        return None, None, ""
    if local_only:
        from agent.auxiliary_egress_recovery import is_local_fallback_client
        if not is_local_fallback_client(client, main_provider):
            return None, None, ""
    label = f"main-agent({main_provider})"
    auxiliary.logger.info("Auxiliary %s: %s on %s — falling back to main agent model %s (%s)",
                task or "call", reason, failed_provider, label, resolved_model or main_model)
    return client, resolved_model or main_model, label



def try_configured_fallback_chain(
    task: str, failed_provider: str, reason: str = "error", failed_model: str | None = None, *,
    failed_base_url: str = "", failure_scope: Any = None, local_only: bool = False,
) -> tuple[Any | None, str | None, str]:
    """Try auxiliary.<task>.fallback_chain entries in order (each needs ``provider``; model/base_url/api_key optional).
    ``failed_model`` scoping per ``auxiliary._failed_backend_skip`` (sibling models on the same provider still
    run after a model-scoped failure). Returns (client, model, provider_label) or (None, None, "")."""
    from agent import auxiliary_client as auxiliary
    if not task:
        return None, None, ""
    chain = auxiliary._get_auxiliary_task_config(task).get("fallback_chain")
    if not chain or not isinstance(chain, list):
        return None, None, ""
    skip = auxiliary._failed_backend_skip(
        failed_provider, failed_model, failed_base_url=failed_base_url, failure_scope=failure_scope)
    tried = []
    min_ctx = auxiliary._task_minimum_context_length(task)
    for i, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        if local_only:
            from agent.auxiliary_egress_recovery import local_fallback_entry
            entry = local_fallback_entry(entry)
            if entry is None:
                continue
        fb_provider = str(entry.get("provider", "")).strip()
        if not fb_provider:
            continue
        fb_model_raw = str(entry.get("model", "")).strip()
        fb_base_url = auxiliary._custom_health_base_url(fb_provider, entry.get("base_url"))
        if skip(fb_provider, fb_model_raw, fb_base_url):
            continue
        if auxiliary._is_provider_unhealthy(fb_provider, fb_base_url):
            auxiliary._log_skip_unhealthy(fb_provider, task, base_url=fb_base_url)
            tried.append(f"fallback_chain[{i}]({fb_provider}) (unhealthy)")
            continue
        fb_model = fb_model_raw or None
        label = f"fallback_chain[{i}]({fb_provider})"
        try:
            fb_client, resolved_model = auxiliary._resolve_fallback_entry(entry)
        except Exception:
            logger.debug("Optional metadata or provenance operation failed", exc_info=True)
            fb_client, resolved_model = None, None
        if fb_client is not None:
            if local_only:
                from agent.auxiliary_egress_recovery import is_local_fallback_client
                if not is_local_fallback_client(fb_client, fb_provider):
                    continue
            context_entry = {**entry, "base_url": str(fb_client.base_url)} if local_only else entry
            too_small = auxiliary._context_too_small(
                context_entry, fb_provider, resolved_model, min_ctx, task=task, label=label, name_model=True,
            ) if resolved_model else None
            if too_small:
                tried.append(too_small)
                continue
            auxiliary.logger.info("Auxiliary %s: %s on %s — configured fallback to %s (%s)",
                        task, reason, failed_provider, label, resolved_model or fb_model or "default")
            return fb_client, resolved_model or fb_model, label
        tried.append(label)
    if tried:
        auxiliary.logger.debug("Auxiliary %s: configured fallback_chain exhausted (tried: %s)", task, ", ".join(tried))
    return None, None, ""
