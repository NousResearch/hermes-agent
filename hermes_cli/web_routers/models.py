"""Model assignment dashboard routes: model info/options/recommended default, auxiliary + MoA slots, /api/model/set.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are resolved late at call time (cycle-safe).
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_config import (
    _AUX_TASK_SLOTS, _apply_model_assignment_sync, _dashboard_code_skew_guard,
)
from starlette.concurrency import run_in_threadpool
from hermes_cli.web_models import ModelAssignment, MoaConfigPayload, MoaModelSlot
from hermes_cli.web_routers._common import http_failure

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

# Late-bound so a test's monkeypatch on the owning module wins at call time.
_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_profile_scope = late("_profile_scope", "hermes_cli.web_server_profiles")
load_config = late("load_config", "hermes_cli.config")
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
            from agent.model_metadata import get_model_context_length
            # config_context_length=None: ignore the override — we want the auto value
            auto_ctx = get_model_context_length(model=model_name, base_url=base_url, provider=provider,
                                                config_context_length=None)
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
    from hermes_cli import models as m
    from hermes_cli import models_pricing as mp
    from hermes_cli.auth import get_provider_auth_state

    model_ids = m.get_curated_nous_model_ids()
    pricing = mp.get_pricing_for_provider("nous") or {}
    free_tier = m.check_nous_free_tier(force_fresh=True)

    try:
        portal_url = (get_provider_auth_state("nous") or {}).get("portal_base_url", "") or ""
    except Exception:
        portal_url = ""

    # This endpoint picks the model a user lands on without choosing it, so an unreachable
    # one is worse than in a picker. Narrow to policy BEFORE the tier split, so a rescued
    # id still has to pass the free/paid predicate.
    policy_allowed = mp.nous_policy_allowed_ids()
    union = m.union_with_portal_free_recommendations if free_tier else m.union_with_portal_paid_recommendations
    model_ids, pricing = union(model_ids, pricing, portal_url)
    model_ids = mp.restrict_to_nous_policy(model_ids, policy_allowed, rescue_empty=True)
    if free_tier:
        model_ids, _unavailable = m.partition_nous_models_by_tier(model_ids, pricing, free_tier=True)

    model = m.pick_silent_default_model(model_ids, provider="nous")
    return {"provider": "nous", "model": model, "free_tier": bool(free_tier)}


@router.get("/api/model/recommended-default")
def get_recommended_default_model(provider: str = ""):
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
            return _nous_recommended_default()
        except Exception:
            _log.exception("GET /api/model/recommended-default (nous) failed")
            return {"provider": "nous", "model": "", "free_tier": None}

    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context
        from hermes_cli.models import pick_silent_default_model

        payload = build_models_payload(load_picker_context())
        for row in payload.get("providers", []):
            if str(row.get("slug", "")).lower() == slug:
                models = [str(m) for m in (row.get("models") or [])]
                return {"provider": slug, "model": pick_silent_default_model(models, provider=slug), "free_tier": None}
        return {"provider": slug, "model": "", "free_tier": None}
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
            tasks.append({
                "task": slot, "provider": str(slot_cfg.get("provider", "auto") or "auto"),
                "model": str(slot_cfg.get("model", "") or ""), "base_url": str(slot_cfg.get("base_url", "") or ""),
            })

        model, provider = _main_model_fields(cfg.get("model", {}))
        return {"tasks": tasks, "main": {"provider": str(provider or ""), "model": str(model or "")}}


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
    "degraded_reference_policy", "max_tokens", "reference_max_tokens", "fanout", "enabled",
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

        with _profile_scope(body.profile or profile):
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
            # See issue #58819.
            cfg.setdefault("moa", {}).update(normalized)
            save_config(cfg)
            return {"ok": True, **normalized}


@router.post("/api/model/set")
async def set_model_assignment(body: ModelAssignment, profile: Optional[str] = None):
    """Assign a model to the main slot or an auxiliary task slot. Writes
    ``~/.hermes/config.yaml`` — applies to **new** sessions only; a running chat
    PTY hot-swaps via the ``/model`` slash command instead."""
    scope, task = (body.scope or "").strip().lower(), (body.task or "").strip().lower()
    provider, model = (body.provider or "").strip(), (body.model or "").strip()
    base_url, api_key = (body.base_url or "").strip(), (body.api_key or "").strip()

    if scope not in {"main", "auxiliary"}:
        raise HTTPException(status_code=400, detail="scope must be 'main' or 'auxiliary'")

    with http_failure("POST /api/model/set failed", 500, detail="Failed to save model assignment"):
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

        def _apply_assignment():
            with _profile_scope(body.profile or profile):
                return _apply_model_assignment_sync(scope, provider, model, task, base_url, api_key)

        return await asyncio.to_thread(_apply_assignment)
