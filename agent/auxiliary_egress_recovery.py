"""Recover a denied auxiliary request only through an explicitly local route."""
import os

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
    if provider == "custom":
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
            return None
    else:
        try:
            from hermes_cli.runtime_provider import _get_named_custom_provider

            original_provider = raw_provider.lower()
            named = (
                _get_named_custom_provider(original_provider, metadata_only=True)
                if original_provider != provider else None
            )
            if named is None:
                named = _get_named_custom_provider(provider, metadata_only=True)
            if named:
                # The named-provider router ignores an entry's URL override.
                base_url = str(named.get("base_url") or "").strip()
            else:
                from hermes_cli.auth import PROVIDER_REGISTRY

                registered = PROVIDER_REGISTRY.get(provider)
                if (
                    registered is None or registered.auth_type != "api_key"
                    or provider in {"anthropic", "copilot", "azure-foundry"}
                ):
                    return None
                env_url = (
                    os.getenv(registered.base_url_env_var, "").strip()
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
        lambda: auxiliary._try_configured_fallback_chain(
            route.task, route.resolved_provider or "auto", reason="egress blocked",
            failed_model=route.final_model, local_only=True,
        ),
        lambda: auxiliary._try_main_agent_model_fallback(
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
