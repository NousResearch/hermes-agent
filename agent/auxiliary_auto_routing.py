"""Live runtime identity and model selection for automatic auxiliary routes."""

from urllib.parse import urlsplit


def compression_main_runtime(compressor):
    """Keep a matching turn's named identity and live credential/endpoint, not a config snapshot."""
    from agent import auxiliary_client as aux

    own = {field: getattr(compressor, field) for field in (
        "model", "provider", "base_url", "api_key", "api_mode"
    )}
    live = aux._RUNTIME_MAIN_CONTEXT.get() or {}
    own_url = urlsplit(str(own["base_url"] or ""))._replace(query="", fragment="")
    live_url = urlsplit(str(live.get("base_url") or ""))._replace(query="", fragment="")
    if live.get("provider") == own["provider"] and live.get("model") == own["model"] and live_url == own_url:
        return {**own, **live}
    return own


def compatible_auto_model(override, default, base_url=""):
    """Reject aggregator-shaped ids on native routes, as the historical auto-model contract does."""
    from utils import base_url_host_matches

    if override and "/" in override and not (
        default and "/" in default or base_url_host_matches(base_url, "openrouter.ai")
    ):
        return default
    return override or default


def main_route_target(runtime, task):
    """Main provider/model after named identity, optional fast tier and MoA substitution."""
    from types import SimpleNamespace
    from agent import auxiliary_client as aux
    from agent.gemini_routing import effective_gemini_provider

    main_provider = effective_gemini_provider(SimpleNamespace(**runtime)) or aux._read_main_provider() or ""
    main_model = str(runtime.get("model") or aux._read_main_model() or "")
    base_url, api_key, api_mode = runtime.get("base_url", ""), runtime.get("api_key", ""), runtime.get("api_mode", "")
    if aux._task_prefers_fast_model(task) and main_provider not in {"auto", ""}:
        fast_model = aux._get_aux_model_for_provider(main_provider, prefer_fast=True)
        if fast_model and fast_model != main_model:
            aux.logger.debug("Auxiliary task %s: preferring fast model %s over main model %s", task, fast_model, main_model)
            main_model = fast_model
    if main_provider == "moa":
        agg_provider, agg_model = aux._resolve_moa_aggregator(main_model)
        if agg_provider and agg_model:
            main_provider, main_model = agg_provider, agg_model
            base_url = api_key = api_mode = ""
    return main_provider, main_model, base_url, api_key, api_mode


def discovery_chain(model_override):
    """Pass the model to discovery constructors before they select a mixed endpoint's wire."""
    from agent import auxiliary_client as aux

    return [
        ("openrouter", lambda: aux._try_openrouter(model=model_override)),
        ("nous", lambda: aux.resolve_provider_client("nous", model_override)),
        ("local/custom", lambda: aux._try_custom_endpoint(model_override=model_override)),
        ("api-key", lambda: aux._resolve_api_key_provider(model_override=model_override)),
    ]
