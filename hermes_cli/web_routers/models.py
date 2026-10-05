"""Model assignment dashboard routes: model info/options/recommended default, auxiliary + MoA slots, /api/model/set.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are resolved late at call time (cycle-safe).
"""

import asyncio
import concurrent.futures
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_config import (
    _AUX_TASK_SLOTS, _UNSET, _apply_model_assignment_sync, _dashboard_code_skew_guard,
    _prepare_main_assignment,
)
from agent.model_metadata import is_local_endpoint
from starlette.concurrency import run_in_threadpool
from hermes_cli.web_models import ModelAssignment, MoaConfigPayload, MoaModelSlot, ReasoningEffortUpdate
from hermes_cli.web_routers._common import _CONFIG_MUTATION_LOCK, config_write_scope, http_failure

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
load_config = late("load_config", "hermes_cli.config")
read_user_config_raw = late("read_user_config_raw", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")


_EMPTY_MODEL_INFO: dict = {
    "model": "", "provider": "", "auto_context_length": 0, "config_context_length": 0,
    "effective_context_length": 0, "capabilities": {},
}
_CAPABILITY_FIELDS = ("supports_tools", "supports_vision", "supports_reasoning", "context_window",
                      "max_output_tokens", "model_family")


def _main_model_fields(model_cfg) -> tuple[str, str]:
    """(model, provider) from config's ``model`` section, which may be a plain string."""
    if isinstance(model_cfg, dict):
        return model_cfg.get("default", model_cfg.get("name", "")), model_cfg.get("provider", "")
    return (str(model_cfg) if model_cfg else ""), ""


def _load_config_scoped(profile: Optional[str]) -> dict:
    with _profile_scope(profile):
        return load_config()


# Blocking budget for /api/model/info's context-length resolution. The resolver
# chain (agent.model_metadata.get_model_context_length) runs several sequential
# provider probes, each with its own multi-second timeout, so an unreachable or
# blackholed model.base_url can hold this response for tens of seconds — and the
# Desktop Model Settings page waits on it (#63214).
_MODEL_INFO_PROBE_BUDGET_S = 5.0


def _bounded_context_length_probe(model: str, base_url: str, provider: str) -> int:
    """``get_model_context_length`` with the route's blocking budget.

    On timeout the abandoned probe keeps running in its worker thread (bounded
    by its own per-request timeouts) while the response degrades to
    ``auto_context_length = 0`` ("auto-detected: unknown").
    """
    from agent.model_metadata import get_model_context_length

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-info-probe")
    try:
        return pool.submit(
            get_model_context_length, model=model, base_url=base_url, provider=provider,
            config_context_length=None
        ).result(timeout=_MODEL_INFO_PROBE_BUDGET_S)
    except concurrent.futures.TimeoutError:
        _log.warning(
            "GET /api/model/info: context-length probe for %r at %s exceeded %.1fs — returning unknown",
            model, base_url or "<default>", _MODEL_INFO_PROBE_BUDGET_S,
        )
        return 0
    finally:
        # wait=False: never block the response (or interpreter exit) on the
        # abandoned probe.
        pool.shutdown(wait=False)


@router.get("/api/model/info")
def get_model_info(profile: Optional[str] = None):
    """Resolved metadata for the configured model: auto-detected vs configured
    context length (so the UI can show "Auto-detected: 200K" beside the
    override) plus models.dev capabilities when available."""
    try:
        model_cfg = _load_config_scoped(profile).get("model", "")
        model_name, provider = _main_model_fields(model_cfg)
        base_url = model_cfg.get("base_url", "") if isinstance(model_cfg, dict) else ""
        config_ctx = model_cfg.get("context_length") if isinstance(model_cfg, dict) else None

        if not model_name:
            return dict(_EMPTY_MODEL_INFO, provider=provider)

        try:
            # config_context_length=None: ignore the override — we want the auto value.
            # Bounded: the resolver's provider probes can hang for tens of seconds
            # when model.base_url is unreachable (#63214).
            auto_ctx = _bounded_context_length_probe(model_name, base_url, provider)
        except Exception:
            auto_ctx = 0

        config_ctx_int = config_ctx if isinstance(config_ctx, int) and config_ctx > 0 else 0

        caps = {}
        try:
            from agent.models_dev import get_model_capabilities
            mc = get_model_capabilities(provider=provider, model=model_name)
            if mc is not None:
                caps = {name: getattr(mc, name) for name in _CAPABILITY_FIELDS}
        except Exception:
            pass

        return {
            "model": model_name, "provider": provider, "auto_context_length": auto_ctx,
            "config_context_length": config_ctx_int,
            "effective_context_length": config_ctx_int or auto_ctx,  # what the agent actually uses
            "capabilities": caps,
        }
    except HTTPException:
        # Unknown/invalid profile must surface as 404, not degrade into a
        # 200 with empty model info (which would render as "no model set").
        raise
    except Exception:
        _log.exception("GET /api/model/info failed")
        return dict(_EMPTY_MODEL_INFO)


@router.get("/api/model/options")
async def get_model_options(
    profile: Optional[str] = None,
    refresh: bool = False,
    include_unconfigured: bool = False,
    explicit_only: bool = False,
):
    """Authenticated providers + curated model lists — REST twin of the ``model.options``
    JSON-RPC on tui_gateway, same response shape so ``ModelPickerDialog`` shares the types.
    ``profile`` scopes the picker context so the Models page reads the SAME profile
    /api/model/set writes. ``refresh`` busts the per-provider model-id disk cache
    (picker's explicit "Refresh Models"); normal opens stay on the 1h cache."""
    with http_failure("GET /api/model/options failed", 500, detail="Failed to list model options"):
        skew_msg = _dashboard_code_skew_guard()
        if skew_msg:
            _log.warning("GET /api/model/options refused: %s", skew_msg)
            raise HTTPException(status_code=503, detail=f"Restart required: {skew_msg}")

        from hermes_cli.inventory import build_model_options_payload, load_picker_context

        def _build_payload_scoped() -> dict:
            # Full sync picker build off the event loop under the requested profile.
            # _config_profile_scope (contextvar only, no skill-module lock): the build can
            # block 15s on a models.dev cache miss, and _profile_scope's RLock held across
            # that starves concurrent /api/config and freezes the server.
            with _config_profile_scope(profile):
                return build_model_options_payload(
                    load_picker_context(), explicit_only=bool(explicit_only),
                    include_unconfigured=bool(include_unconfigured), refresh=bool(refresh))

        return await run_in_threadpool(_build_payload_scoped)


def _nous_recommended_default() -> dict:
    from hermes_cli.models import recommended_nous_default_model
    return recommended_nous_default_model()


@router.get("/api/model/recommended-default")
def get_recommended_default_model(provider: str = "", profile: Optional[str] = None):
    """Recommended default model for a freshly-authenticated provider, mirroring
    ``hermes model``'s curation so GUI onboarding lands on a sensible default.
    Nous honors the user's free/paid tier. Any other provider gets the preferred
    silent default when its curated list carries it, else the first curated model —
    aggregator lists lead with the priciest Anthropic flagship, which must never be
    the model a user lands on without explicitly picking it.
    Response: {"provider", "model", "free_tier": bool | None} — free_tier only for
    Nous; ``model`` may be empty (caller degrades gracefully)."""
    slug = (provider or "").strip().lower()

    if slug == "nous":
        try:
            # The tier, Portal URL and recommendation caches are all per profile home.
            with _config_profile_scope(profile):
                return _nous_recommended_default()
        except HTTPException:
            raise  # an unknown ?profile= is the scope's 404, not an empty recommendation
        except Exception:
            _log.exception("GET /api/model/recommended-default (nous) failed")
            return {"provider": "nous", "model": "", "free_tier": None}

    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context
        from hermes_cli.models import pick_silent_default_model

        # build_models_payload -> list_authenticated_providers -> _save_discovered_models_to_config:
        # this GET lazily PERSISTS discovered custom-provider models, so it needs the scope too.
        with _config_profile_scope(profile):
            payload = build_models_payload(load_picker_context())
        for row in payload.get("providers", []):
            if str(row.get("slug", "")).lower() == slug:
                models = [str(m) for m in (row.get("models") or [])]
                return {"provider": slug, "model": pick_silent_default_model(models, provider=slug), "free_tier": None}
        return {"provider": slug, "model": "", "free_tier": None}
    except HTTPException:
        raise  # an unknown ?profile= is the scope's 404, not an empty recommendation
    except Exception:
        _log.exception("GET /api/model/recommended-default failed")
        return {"provider": slug, "model": "", "free_tier": None}


@router.get("/api/model/auxiliary")
def get_auxiliary_models(profile: Optional[str] = None):
    """Current auxiliary task assignments: ``{"tasks": [{task, provider, model,
    base_url}, ...], "main": {provider, model}}``. ``profile`` scopes the read —
    without it the Models page would show the dashboard profile's pins while
    /api/model/set wrote the selected profile's."""
    with http_failure("GET /api/model/auxiliary failed", 500, detail="Failed to read auxiliary config"):
        cfg = _load_config_scoped(profile)
        aux_cfg = cfg.get("auxiliary", {})
        if not isinstance(aux_cfg, dict):
            aux_cfg = {}

        tasks = []
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux_cfg.get(slot, {}) if isinstance(aux_cfg.get(slot), dict) else {}
            base_url = str(slot_cfg.get("base_url", "") or "")
            tasks.append({
                "task": slot, "provider": str(slot_cfg.get("provider", "auto") or "auto"),
                "model": str(slot_cfg.get("model", "") or ""), "base_url": base_url,
                "reasoning_effort": str(slot_cfg.get("reasoning_effort") or "") or None,
                # Lets the UI tell a free local/LAN pin from a forgotten paid-provider pin.
                "local_endpoint": is_local_endpoint(base_url),
            })

        model, provider = _main_model_fields(cfg.get("model", {}))
        main = {"provider": str(provider or ""), "model": str(model or "")}
        dcfg = cfg.get("delegation", {}) if isinstance(cfg.get("delegation"), dict) else {}
        delegation = {
            "provider": str(dcfg.get("provider", "") or provider),
            "model": str(dcfg.get("model", "") or model),
            # Empty is an explicit inheritance state, not the parent's rendered value.
            "reasoning_effort": _reasoning_effort_display(dcfg.get("reasoning_effort", "")),
            "has_base_url": bool(dcfg.get("base_url")),
            "has_api_key": bool(dcfg.get("api_key")),
            "max_iterations": int(dcfg.get("max_iterations", 250) or 250),
            "max_concurrent_children": int(dcfg.get("max_concurrent_children", 10) or 10),
            "max_spawn_depth": int(dcfg.get("max_spawn_depth", 1) or 1),
        }
        return {"tasks": tasks, "main": main, "delegation": delegation}


def _reasoning_effort_display(value) -> str:
    """Normalize raw values into the select contract without mutating config."""
    if value is False or (isinstance(value, str) and value.strip().lower() in {"false", "disabled", "none"}):
        return "none"
    if value is None or value == "":
        return ""
    known = {"minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    text = str(value)
    return text if text in known else "__custom__"


def _reasoning_effort_custom(value) -> str:
    if value is None or value is False or value == "":
        return ""
    text = str(value)
    return text if _reasoning_effort_display(value) == "__custom__" else ""


def _main_model(cfg: dict) -> str:
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        value = model_cfg.get("default") or model_cfg.get("model") or model_cfg.get("name", "")
    else:
        value = model_cfg
    return value.strip() if isinstance(value, str) else ""


def _reasoning_payload(cfg: dict) -> dict:
    from hermes_constants import resolve_per_model_reasoning_effort, resolve_reasoning_config
    agent = cfg.get("agent", {}) if isinstance(cfg.get("agent"), dict) else {}
    delegation = cfg.get("delegation", {}) if isinstance(cfg.get("delegation"), dict) else {}
    model = _main_model(cfg)
    override = resolve_per_model_reasoning_effort(model, agent.get("reasoning_overrides")) if model else None
    effective = resolve_reasoning_config(cfg, model)
    raw_value = agent.get("reasoning_effort", "")
    return {
        "main_raw": _reasoning_effort_display(raw_value),
        "delegation_raw": _reasoning_effort_display(delegation.get("reasoning_effort", "")),
        "main_custom": _reasoning_effort_custom(raw_value),
        "delegation_custom": _reasoning_effort_custom(delegation.get("reasoning_effort", "")),
        "main_effective": "none" if effective == {"enabled": False} else (str(effective.get("effort", "")) if isinstance(effective, dict) else ""),
        "main_source": "model_override" if override is not None else ("global" if raw_value not in (None, "") else "provider_default"),
        "main_model": model,
    }


@router.get("/api/model/reasoning-effort")
def get_reasoning_effort(profile: Optional[str] = None):
    with _profile_scope(profile):
        return _reasoning_payload(read_user_config_raw())


@router.put("/api/model/reasoning-effort")
def set_reasoning_effort(body: ReasoningEffortUpdate, profile: Optional[str] = None):
    profile_scope = _profile_scope(body.profile or profile)
    with profile_scope, _CONFIG_MUTATION_LOCK:
        cfg = read_user_config_raw()
        if body.scope == "main" and body.target == "model":
            current_model = _main_model(cfg)
            if not body.model or body.model != current_model:
                raise HTTPException(status_code=409, detail="main model changed; refresh before saving model reasoning")
            agent = cfg.setdefault("agent", {})
            if not isinstance(agent, dict):
                agent = cfg["agent"] = {}
            overrides = agent.setdefault("reasoning_overrides", {})
            if not isinstance(overrides, dict):
                overrides = agent["reasoning_overrides"] = {}
            from hermes_constants import resolve_per_model_reasoning_effort
            matching_keys = {
                key for key, value in overrides.items()
                if isinstance(key, str) and resolve_per_model_reasoning_effort(current_model, {key: value}) is not None
            }
            if body.effort == "__custom__":
                raise HTTPException(status_code=400, detail="custom effort must be supplied as a concrete value")
            if body.effort:
                # Normalize aliases already present, then write the canonical current model key.
                for key in matching_keys:
                    if key != current_model:
                        overrides.pop(key, None)
                overrides[current_model] = body.effort
            else:
                for key in matching_keys:
                    overrides.pop(key, None)
        else:
            section_name = "agent" if body.scope == "main" else "delegation"
            section = cfg.setdefault(section_name, {})
            if not isinstance(section, dict):
                section = cfg[section_name] = {}
            if body.scope == "main" and body.effort == "__custom__":
                raise HTTPException(status_code=400, detail="custom effort must be supplied as a concrete value")
            if body.effort:
                section["reasoning_effort"] = body.effort
            else:
                section.pop("reasoning_effort", None)
        save_config(cfg)
        result = _reasoning_payload(read_user_config_raw())
        raw = result["main_raw"] if body.scope == "main" else result["delegation_raw"]
        if body.scope == "main" and body.target == "model":
            verified_cfg = read_user_config_raw()
            verified_agent = verified_cfg.get("agent", {}) if isinstance(verified_cfg.get("agent"), dict) else {}
            from hermes_constants import resolve_per_model_reasoning_effort as resolve_model_override
            verified = resolve_model_override(body.model or "", verified_agent.get("reasoning_overrides"))
            verified_raw = "none" if verified == {"enabled": False} else (verified.get("effort", "") if isinstance(verified, dict) else "")
            ok = verified_raw == body.effort if body.effort else verified is None
            result.update({"ok": ok, "scope": body.scope, "raw": verified_raw})
            return result
        if body.scope == "main" and body.target == "global":
            verified_cfg = read_user_config_raw()
            verified_agent = verified_cfg.get("agent", {}) if isinstance(verified_cfg.get("agent"), dict) else {}
            verified_raw = verified_agent.get("reasoning_effort", "")
            ok = verified_raw == body.effort if body.effort else "reasoning_effort" not in verified_agent
            result.update({"ok": ok, "scope": body.scope, "raw": _reasoning_effort_display(verified_raw)})
            return result
        if body.scope == "delegation":
            verified_cfg = read_user_config_raw()
            verified_delegation = verified_cfg.get("delegation", {}) if isinstance(verified_cfg.get("delegation"), dict) else {}
            verified_raw = verified_delegation.get("reasoning_effort", "")
            ok = verified_raw == body.effort if body.effort else "reasoning_effort" not in verified_delegation
            result.update({"ok": ok, "scope": body.scope, "raw": _reasoning_effort_display(verified_raw)})
            return result
        result.update({"ok": True, "scope": body.scope, "raw": raw})
        return result


@router.get("/api/model/moa")
def get_moa_models(profile: Optional[str] = None):
    """Return the configured Mixture-of-Agents provider/model slots."""
    with http_failure("GET /api/model/moa failed", 500, detail="Failed to read MoA config"):
        from hermes_cli.moa_config import normalize_moa_config

        with _profile_scope(profile):
            cfg = load_config()
            return normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})


_MOA_PRESET_FIELDS = (
    "reference_temperature", "aggregator_temperature", "reference_timeout",
    "degraded_reference_policy", "fanout", "enabled",
)


def _slot_dict(slot: MoaModelSlot) -> dict:
    # Drop unset optionals so saved slots stay minimal ({provider, model}).
    return {k: v for k, v in slot.dict().items() if v is not None}


def _preset_dict(preset) -> dict:
    """Raw preset dict from a MoaPresetPayload or the flat MoaConfigPayload fields."""
    return {
        "reference_models": [_slot_dict(slot) for slot in preset.reference_models],
        "aggregator": _slot_dict(preset.aggregator),
        **{name: getattr(preset, name) for name in _MOA_PRESET_FIELDS},
    }


@router.put("/api/model/moa")
def set_moa_models(body: MoaConfigPayload, profile: Optional[str] = None):
    """Persist the Mixture-of-Agents provider/model slots."""
    with http_failure("PUT /api/model/moa failed", 500, detail="Failed to save MoA config"):
        from hermes_cli.moa_config import normalize_moa_config, validate_moa_payload

        # load→mutate→save runs on a worker thread (sync-def endpoint); the
        # desktop's debounced PUT /api/config autosave races it, so the whole
        # span holds _CONFIG_MUTATION_LOCK or one of the two saves is dropped.
        with config_write_scope(body.profile or profile):
            cfg = load_config()
            if body.presets:
                raw = {
                    "default_preset": body.default_preset,
                    "active_preset": body.active_preset,
                    "presets": {name: _preset_dict(preset) for name, preset in body.presets.items()},
                }
            else:
                raw = _preset_dict(body)  # legacy flat payload from older clients

            # Reject-don't-repair: normalize_moa_config() silently swaps any preset with
            # incomplete slots for the hardcoded defaults — correct tolerance at READ time,
            # silent data loss at WRITE time (desktop autosave of a half-filled slot replaced
            # the user's whole preset). Refuse loudly so no client can corrupt config here.
            # See #64156.
            problems = validate_moa_payload(raw)
            if problems:
                raise HTTPException(status_code=422, detail="Invalid MoA config: " + "; ".join(problems))
            normalized = normalize_moa_config(raw)
            # Merge, don't overwrite: hand-edited keys not in MoaConfigPayload (save_traces, trace_dir) survive.
            # See issue #58819. Write ONLY the moa section (merge_existing deep-merges it over the
            # on-disk raw file): saving the whole default-expanded ``cfg`` snapshot re-persisted
            # every other section too, so a Desktop MoA autosave could wipe a chain another
            # surface wrote meanwhile (#89184, ``fallback_providers: []``).
            moa_section = dict(cfg.get("moa") or {})
            moa_section.update(normalized)
            save_config({"moa": moa_section}, merge_existing=True)
            return {"ok": True, **normalized}


@router.post("/api/model/set")
async def set_model_assignment(body: ModelAssignment, profile: Optional[str] = None):
    """Assign a model to the main slot or an auxiliary task slot. Writes
    ``~/.hermes/config.yaml`` — applies to **new** sessions only; a running chat
    PTY hot-swaps via the ``/model`` slash command instead."""
    scope, task = (body.scope or "").strip().lower(), (body.task or "").strip().lower()
    provider, model = (body.provider or "").strip(), (body.model or "").strip()
    base_url, api_key = (body.base_url or "").strip(), (body.api_key or "").strip()

    if scope not in {"main", "auxiliary", "delegation"}:
        raise HTTPException(status_code=400, detail="scope must be 'main', 'auxiliary', or 'delegation'")
    if scope == "delegation" and task:
        raise HTTPException(status_code=400, detail="task is not supported for delegation assignments")
    if scope == "delegation" and api_key:
        raise HTTPException(status_code=400, detail="API keys are not accepted for delegation; use the provider's configured credentials")
    if scope == "delegation" and "api_key" in body.model_fields_set:
        raise HTTPException(status_code=400, detail="API keys are not accepted for delegation; use the provider's configured credentials")

    if scope == "delegation" and body.confirm_clear_routing and not body.reset_routing:
        # Confirmation also authorizes clearing custom routing during a provider switch.
        pass

    with http_failure("POST /api/model/set failed", 500, detail="Failed to save model assignment"):
        # #99859 (R2): the options picker already refuses on code skew; the WRITE path
        # must too — a stale process persisting a post-update model string is the
        # invalid-model-serving failure the reporter hit.
        skew_msg = _dashboard_code_skew_guard()
        if skew_msg:
            _log.warning("POST /api/model/set refused: %s", skew_msg)
            raise HTTPException(status_code=503, detail=f"Restart required: {skew_msg}")

        # Expensive-model warning runs BEFORE the profile scope is entered: _profile_scope
        # must never be held across an await (the RLock is reentrant per-thread, so a second
        # coroutine interleaving on the event-loop thread could cross-restore module globals).
        if model and not body.confirm_expensive_model:
            try:
                from hermes_cli.model_selection_guards import combined_selection_warning

                # Pricing lookup can hit models.dev / a /models endpoint on a cache miss — off the loop.
                warning = await asyncio.to_thread(combined_selection_warning, model, provider=provider, base_url=base_url)
            except Exception:
                warning = None
            if warning is not None:
                return {"ok": False, "scope": scope, "provider": provider, "model": model,
                        "confirm_required": True, "confirm_message": warning.message}

        reasoning_effort = body.reasoning_effort if "reasoning_effort" in body.model_fields_set else _UNSET

        if scope == "delegation" and body.reset_routing:
            with _CONFIG_MUTATION_LOCK, _profile_scope(body.profile or profile):
                cfg = read_user_config_raw()
                delegation = cfg.get("delegation") if isinstance(cfg.get("delegation"), dict) else {}
                has_routing = any(delegation.get(key) for key in ("base_url", "api_key", "api_mode"))
                if has_routing and not body.confirm_clear_routing:
                    return {"ok": False, "scope": scope, "routing_confirmation_required": True,
                            "confirm_message": "Resetting delegation routing removes the custom endpoint and credentials. Confirm to continue."}
                delegation = dict(delegation)
                for key in ("provider", "model", "base_url", "api_key", "api_mode"):
                    delegation.pop(key, None)
                cfg["delegation"] = delegation
                save_config(cfg)
                verified = read_user_config_raw().get("delegation", {})
                return {"ok": not any(verified.get(key) for key in ("provider", "model", "base_url", "api_key", "api_mode")),
                        "scope": scope, "routing_reset": True, "has_base_url": False, "has_api_key": False}

        def _apply_assignment():
            # Network/catalog work must finish before entering the config mutation lock;
            # only load→apply→save is serialized with dashboard config autosave.
            if scope == "main":
                with _profile_scope(body.profile or profile):
                    prepared = _prepare_main_assignment(load_config(), provider, model, base_url, api_key)
            else:
                prepared = None

            with _CONFIG_MUTATION_LOCK, _profile_scope(body.profile or profile):
                if scope == "delegation":
                    cfg = read_user_config_raw()
                    delegation = cfg.setdefault("delegation", {})
                    if not isinstance(delegation, dict):
                        delegation = cfg["delegation"] = {}
                    configured_provider = str(delegation.get("provider", "") or "").strip()
                    if not configured_provider:
                        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
                        configured_provider = str(model_cfg.get("provider", "") or "").strip()
                    old_provider = configured_provider.strip().lower()
                    new_provider = provider.strip().lower() or old_provider
                    provider_changed = bool(new_provider and old_provider != new_provider)
                    custom_routing = any(delegation.get(key) for key in ("base_url", "api_key", "api_mode"))
                    if provider_changed and custom_routing and not body.confirm_clear_routing:
                        return {"ok": False, "scope": scope, "provider": provider, "model": model,
                                "routing_confirmation_required": True,
                                "confirm_message": "Changing provider will remove custom routing. Confirm to continue."}
                    if provider_changed and custom_routing and body.confirm_clear_routing:
                        for key in ("base_url", "api_key", "api_mode"):
                            delegation.pop(key, None)
                    if provider: delegation["provider"] = provider
                    else: delegation.pop("provider", None)
                    if model: delegation["model"] = model
                    else: delegation.pop("model", None)
                    # Model-only and unchanged-provider assignment retain all configured routing.
                    save_config(cfg)
                    return {"ok": True, "scope": scope, "provider": provider, "model": model,
                            "has_base_url": bool(delegation.get("base_url")),
                            "has_api_key": bool(delegation.get("api_key"))}
                return _apply_model_assignment_sync(
                    scope, provider, model, task, base_url, api_key,
                    reasoning_effort=reasoning_effort, prepared=prepared)

        return await asyncio.to_thread(_apply_assignment)
