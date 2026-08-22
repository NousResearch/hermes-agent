"""Structured, silent runtime options for messaging-gateway sessions.

Host UIs use this instead of injecting human-facing slash commands. The
gateway remains the single owner of provider resolution, capability checks,
session state, and restart persistence.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Mapping

from gateway.session_state import SERVICE_TIER_UNSET

logger = logging.getLogger(__name__)


def _reasoning_label(value: dict | None) -> str | None:
    if value is None:
        return None
    if value.get("enabled") is False:
        return "none"
    return str(value.get("effort") or "medium")


async def apply_gateway_session_options(
    runner: Any,
    source: Any,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and atomically apply per-session model/reasoning/fast state."""
    allowed = {
        "model",
        "provider",
        "reasoning_effort",
        "fast",
        "confirm_model_selection",
        "initial",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        return {
            "status": "rejected",
            "code": "invalid_options",
            "error": f"unknown session option(s): {', '.join(unknown)}",
        }

    normalized_source = await asyncio.to_thread(
        runner._normalize_source_for_session_key, source
    )
    session_key = runner._session_key_for_source(normalized_source)
    if not session_key:
        return {
            "status": "rejected",
            "code": "invalid_session",
            "error": "session source did not resolve to a session key",
        }
    if runner._is_session_running(session_key):
        return {
            "status": "rejected",
            "code": "session_busy",
            "error": "session options can only change while the session is idle",
        }

    locks = getattr(runner, "_session_options_locks", None)
    if locks is None:
        locks = {}
        runner._session_options_locks = locks
    lock = locks.setdefault(session_key, asyncio.Lock())
    async with lock:
        profile_scope = contextlib.nullcontext()
        if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
            from gateway.run import _profile_runtime_scope

            profile_scope = _profile_runtime_scope(
                runner._resolve_profile_home_for_source(normalized_source)
            )
        with profile_scope:
            return await _apply_scoped(
                runner, normalized_source, session_key, options
            )


async def _apply_scoped(
    runner: Any,
    source: Any,
    session_key: str,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from hermes_cli.model_selection_guards import combined_selection_warning
    from hermes_cli.model_switch import switch_model
    from hermes_cli.models import resolve_fast_mode_overrides
    from hermes_constants import parse_reasoning_effort

    runner._rehydrate_session_runtime_options(session_key)
    state = runner._session_state(session_key)
    current_model, current_runtime = runner._resolve_session_agent_runtime(
        source=source,
        session_key=session_key,
    )
    current_provider = str(current_runtime.get("provider") or "openrouter")
    current_base_url = str(current_runtime.get("base_url") or "")
    current_api_key = str(current_runtime.get("api_key") or "")

    model_override = state.conversation.model_override
    reasoning_override = state.conversation.reasoning_override
    current_tier = state.conversation.service_tier_override
    service_tier_override = (
        None
        if current_tier is SERVICE_TIER_UNSET
        else ("priority" if current_tier == "priority" else "normal")
    )
    effective_model = current_model
    effective_provider = current_provider
    model_warning = ""
    changed_fields: list[str] = []

    if "provider" in options and "model" not in options:
        return {
            "status": "rejected",
            "code": "invalid_options",
            "error": "provider requires model",
        }

    if "model" in options:
        requested_model = str(options.get("model") or "").strip()
        requested_provider = str(options.get("provider") or "").strip()
        if not requested_model:
            model_override = None
            user_config = _load_gateway_config()
            effective_model = _resolve_gateway_model(user_config)
            effective_provider = str(
                ((user_config.get("model") or {}).get("provider")
                 if isinstance(user_config.get("model"), dict) else "")
                or current_provider
            )
        else:
            user_config = _load_gateway_config()
            user_providers = user_config.get("providers")
            try:
                from hermes_cli.config import get_compatible_custom_providers

                custom_providers = get_compatible_custom_providers(user_config)
            except Exception:
                custom_providers = user_config.get("custom_providers")
            result = await asyncio.to_thread(
                switch_model,
                raw_input=requested_model,
                current_provider=current_provider,
                current_model=current_model,
                current_base_url=current_base_url,
                current_api_key=current_api_key,
                is_global=False,
                explicit_provider=requested_provider or None,
                user_providers=user_providers,
                custom_providers=custom_providers,
            )
            if not result.success:
                return {
                    "status": "rejected",
                    "code": "model_rejected",
                    "error": result.error_message or "model switch failed",
                }
            selection_warning = await asyncio.to_thread(
                combined_selection_warning,
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or current_base_url,
                api_key=result.api_key or current_api_key,
                model_info=result.model_info,
            )
            if selection_warning is not None and not bool(
                options.get("confirm_model_selection")
            ):
                return {
                    "status": "confirmation_required",
                    "code": "model_confirmation_required",
                    "title": selection_warning.title,
                    "message": selection_warning.message,
                }
            model_override = {
                "model": result.new_model,
                "provider": result.target_provider,
                "api_key": result.api_key,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
            }
            effective_model = result.new_model
            effective_provider = result.target_provider
            model_warning = str(result.warning_message or "")
        changed_fields.append("model")

    if "reasoning_effort" in options:
        effort = str(options.get("reasoning_effort") or "").strip().lower()
        if not effort:
            reasoning_override = None
        else:
            try:
                reasoning_override = parse_reasoning_effort(effort)
            except Exception as exc:
                return {
                    "status": "rejected",
                    "code": "reasoning_rejected",
                    "error": str(exc),
                }
            if reasoning_override is None:
                return {
                    "status": "rejected",
                    "code": "reasoning_rejected",
                    "error": f"unsupported reasoning effort: {effort}",
                }
        changed_fields.append("reasoning_effort")

    if "fast" in options:
        requested_fast = options.get("fast")
        if not isinstance(requested_fast, bool):
            return {
                "status": "rejected",
                "code": "fast_rejected",
                "error": "fast must be a boolean",
            }
        if requested_fast:
            overrides = resolve_fast_mode_overrides(effective_model)
            if overrides is None:
                return {
                    "status": "rejected",
                    "code": "fast_unsupported",
                    "error": "fast mode is not available for this model",
                }
            service_tier_override = "priority"
        else:
            service_tier_override = "normal"
        changed_fields.append("fast")

    if not changed_fields:
        return {
            "status": "accepted",
            "session_key": session_key,
            "applied": [],
            "effective": {
                "model": effective_model,
                "provider": effective_provider,
                "reasoning_effort": _reasoning_label(reasoning_override),
                "fast": service_tier_override == "priority",
            },
        }

    # Create/resolve the routing entry only after every requested value has
    # passed validation. Persistence happens before process-local mutation so
    # a disk failure cannot leave a half-applied runtime.
    await runner.async_session_store.get_or_create_session(source)
    persisted = await runner.async_session_store.set_runtime_options(
        session_key,
        model_override=model_override,
        reasoning_override=reasoning_override,
        service_tier_override=service_tier_override,
    )
    if not persisted:
        return {
            "status": "rejected",
            "code": "session_missing",
            "error": "session disappeared while applying runtime options",
        }

    state.conversation.model_override = (
        dict(model_override) if model_override is not None else None
    )
    state.conversation.reasoning_override = (
        dict(reasoning_override) if reasoning_override is not None else None
    )
    state.conversation.service_tier_override = (
        "priority" if service_tier_override == "priority" else None
    ) if service_tier_override is not None else SERVICE_TIER_UNSET
    state.persistent.runtime_options_rehydrated = True
    runner._evict_cached_agent(session_key)

    if "model" in changed_fields and not bool(options.get("initial")):
        from hermes_cli.model_switch import format_model_for_display

        if not hasattr(runner, "_pending_model_notes"):
            runner._pending_model_notes = {}
        runner._pending_model_notes[session_key] = (
            f"[Note: model was just switched from "
            f"{format_model_for_display(current_model)} to "
            f"{format_model_for_display(effective_model)} via "
            f"{effective_provider}. Adjust your self-identification accordingly.]"
        )

    session_db = getattr(runner, "_session_db", None)
    if session_db is not None and model_override is not None:
        try:
            entry = await runner.async_session_store.lookup_by_session_key(session_key)
            if entry is not None:
                await session_db.update_session_model(
                    entry.session_id,
                    effective_model,
                    provider=effective_provider,
                )
        except Exception:
            logger.debug("Failed to mirror structured model option", exc_info=True)

    return {
        "status": "accepted",
        "session_key": session_key,
        "applied": changed_fields,
        "effective": {
            "model": effective_model,
            "provider": effective_provider,
            "reasoning_effort": _reasoning_label(reasoning_override),
            "fast": service_tier_override == "priority",
        },
        **({"warning": model_warning} if model_warning else {}),
    }
