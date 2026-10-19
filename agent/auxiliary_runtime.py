"""Session-owned runtime snapshots for auxiliary work."""
import copy
from typing import Any


def current_main_runtime(agent) -> dict[str, Any]:
    """Freeze the live session identity and effort before starting auxiliary work."""
    runtime = {
        key: getattr(agent, key, '') or ''
        for key in ('model', 'provider', 'base_url', 'api_key', 'api_mode', 'auth_mode', 'session_id')
    }
    runtime['reasoning_config'] = copy.deepcopy(getattr(agent, 'reasoning_config', None))
    return runtime


def _aux_route_pins_target(resolved_provider: str | None, resolved_model: str | None) -> bool:
    """Whether task config pinned an explicit provider/model for this call.

    A pinned target (``auxiliary.<task>.provider/model``, a summary_model override, or a
    per-call ``model=``) owns its own reasoning policy; only a route left to ``auto``
    inherits the session's.
    """
    return (
        str(resolved_provider or "").strip().lower() not in {"", "auto", "main"}
        or bool(str(resolved_model or "").strip())
    )


def _aux_route_matches_main_runtime(
    request_provider: str | None, final_model: str | None, base_url: str,
    main_runtime: dict[str, Any],
) -> bool:
    """Whether the concrete auxiliary route IS the session's main deployment.

    Provider and model must match; base_url only disambiguates when both sides carry one
    (default-OpenAI sessions have no explicit URL on either side). Deployment identity is
    compared from normalized route fields — anything stricter would silently disable
    inheritance for default routes.
    """
    def _norm(value: Any) -> str:
        return str(value or "").strip().rstrip("/").lower()

    from agent.auxiliary_client import _normalize_aux_provider, _fallback_provider_from_label

    route_provider = _normalize_aux_provider(
        _fallback_provider_from_label(str(request_provider or "")))
    if route_provider != _normalize_aux_provider(str(main_runtime.get("provider") or "")):
        return False
    if _norm(final_model) != _norm(main_runtime.get("model")):
        return False
    main_base, route_base = _norm(main_runtime.get("base_url")), _norm(base_url)
    return not (main_base and route_base and main_base != route_base)


def inherited_compression_reasoning(
    task, reasoning_config, effective_extra_body, resolved_provider, resolved_model,
    request_provider, final_model, base_url, main_runtime,
):
    """Inherit effort only for an unpinned compression call on its owning route."""
    if (
        task == "compression"
        and reasoning_config is None
        and "reasoning" not in effective_extra_body
        and not _aux_route_pins_target(resolved_provider, resolved_model)
        and _aux_route_matches_main_runtime(
            request_provider, final_model, base_url, main_runtime)
    ):
        inherited = main_runtime.get("reasoning_config")
        if isinstance(inherited, dict):
            reasoning_config = copy.deepcopy(inherited)
    return reasoning_config


def normalize_runtime_fields(main_runtime, fields) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in fields:
        value = main_runtime.get(field)
        if field == "reasoning_config" and isinstance(value, dict):
            # Deep copy: the snapshot must not mutate under a concurrent in-session effort change.
            normalized[field] = copy.deepcopy(value)
        elif field == "api_key" and callable(value) and not isinstance(value, str):
            normalized[field] = value
        elif isinstance(value, str) and value.strip():
            normalized[field] = value.strip()
    for identity_field in ("provider", "requested_provider"):
        identity = normalized.get(identity_field)
        if isinstance(identity, str):
            normalized[identity_field] = identity.lower()
    return normalized


def responses_reasoning_intent(kwargs, extra_body):
    """Preserve an explicit Responses policy before the chat effort spelling."""
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, dict):
        return reasoning
    effort = kwargs.get("reasoning_effort")
    return {"effort": effort} if effort is not None else None


def apply_responses_options(resp_kwargs, kwargs, model, host, *, is_codex_backend, is_xai):
    """Apply route-supported Responses tier and explicit reasoning options."""
    extra_body = kwargs.get("extra_body") or {}
    if isinstance(extra_body, dict):
        # service_tier (fast mode) is a top-level Responses field; xAI's endpoint rejects it.
        service_tier = extra_body.get("service_tier")
        if isinstance(service_tier, str) and service_tier.strip() and not is_xai:
            resp_kwargs["service_tier"] = service_tier.strip()
        reasoning_cfg = responses_reasoning_intent(kwargs, extra_body)
        if isinstance(reasoning_cfg, dict):
            # Shared per-model vocabulary with the main transport ("max" only where the model publishes it; "minimal"/"ultra"
            # clamp to a listed level; ``()`` = the model takes no ``reasoning`` field at all — gpt-4o/4.1 on api.openai.com,
            # #76255). ``enabled: False`` goes on the wire as ``effort: none`` where the vocabulary has it,
            # since an omitted field leaves the model's default effort on (#75227).
            from agent.reasoning_effort import clamp_effort
            from agent.transports.codex import _codex_efforts_for_route
            supported = _codex_efforts_for_route(model, host, is_codex_backend=is_codex_backend)
            if supported and reasoning_cfg.get("enabled") is not False:
                # Truthy-only: Codex 400s on e.g. {"effort": null}, so falsy → default.
                effort = clamp_effort(reasoning_cfg.get("effort") or "medium", supported)
                resp_kwargs["reasoning"] = {"effort": effort, "summary": "auto"}
                resp_kwargs["include"] = ["reasoning.encrypted_content"]
            elif "none" in supported and not is_xai:
                resp_kwargs["reasoning"] = {"effort": "none"}
