"""New-session hooks run once, before the first turn of a session takes its lease.

* Topic/channel-bound skill auto-load (``event.auto_skill``).
* Optional session-start model picker (``gateway.session_start_model_picker``): sends the
  ``/model`` picker when a NEW session opens on a listed platform and holds the first turn until
  the user picks, so the chosen model answers the very first message. The model/runtime resolve
  at agent-run time (``_resolve_session_agent_runtime`` -> ``_apply_session_model_override``),
  after this hook, so a session-scoped override committed during the wait applies to this turn.
  ``gateway.session_start_model_picker_timeout`` bounds the wait (default 60s; 0 = no wait).
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("gateway.run")

_PICKER_KEY = "session_start_model_picker"
_PICKER_TIMEOUT_KEY = "session_start_model_picker_timeout"
_PICKER_TIMEOUT_DEFAULT = 60.0


async def run_session_start_hooks(runner, event, source, session_key, quick_key) -> None:
    """Run the new-session hooks for the turn about to start."""
    auto = getattr(event, "auto_skill", None)
    if auto:
        _hmwa_auto_load_skills(event, auto, quick_key, session_key)
    try:
        await _maybe_send_session_start_model_picker(runner, event, source, session_key)
    except Exception:  # health: allow BLE001 -- an optional picker must never block the turn it decorates
        logger.warning("[Gateway] Session-start model picker failed (session %s)", session_key, exc_info=True)


def _hmwa_auto_load_skills(event, _auto, _quick_key, session_key):
    """Prepend topic/channel-bound skill payload(s) to ``event.text`` on a new session."""
    _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
    try:
        from agent.skill_commands import _load_skill_payload, _build_skill_message
        _combined_parts: list[str] = []
        _loaded_names: list[str] = []
        for _sname in _skill_names:
            _loaded = _load_skill_payload(_sname, task_id=_quick_key)
            if not _loaded:
                logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
                continue
            _loaded_skill, _skill_dir, _display_name = _loaded
            _part = _build_skill_message(
                _loaded_skill, _skill_dir,
                f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                f"Follow its instructions for this session.]",
            )
            if _part:
                _combined_parts.append(_part)
                _loaded_names.append(_sname)
        if _combined_parts:
            _combined_parts.append(event.text)  # user's original text after the payloads
            event.text = "\n\n".join(_combined_parts)
            logger.info("[Gateway] Auto-loaded skill(s) %s for session %s", _loaded_names, session_key)
    except Exception:  # health: allow BLE001 -- moved as-is; a bad skill payload must not block the turn
        logger.warning("[Gateway] Failed to auto-load skill(s) %s", _skill_names, exc_info=True)


def _picker_setting(cfg: dict, key: str, default=None):
    """Top-level key wins by presence, else nested ``gateway.<key>`` (canonical gateway config rule)."""
    if key in cfg:
        return cfg[key]
    section = cfg.get("gateway")
    return section.get(key, default) if isinstance(section, dict) else default


def _picker_timeout(cfg: dict) -> float:
    raw = _picker_setting(cfg, _PICKER_TIMEOUT_KEY, _PICKER_TIMEOUT_DEFAULT)
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return _PICKER_TIMEOUT_DEFAULT


async def _maybe_send_session_start_model_picker(runner, event, source, session_key) -> None:
    """Send the ``/model`` picker on a NEW session on a listed platform and wait for the pick.

    Reuses the ``/model`` picker path verbatim (``_send_model_picker`` + the same
    ``_ModelSwitchContext`` the slash command builds), so the selection lands as a normal
    session-scoped override, like a bare ``/model``. Absent key or unlisted platform = no-op.
    """
    from gateway.run import _hermes_home, _load_gateway_config

    cfg = _load_gateway_config() or {}
    platforms = _picker_setting(cfg, _PICKER_KEY)
    if not isinstance(platforms, list) or not platforms:
        return
    platform = getattr(source, "platform", None)
    platform_name = getattr(platform, "value", str(platform or "")).lower()
    if platform_name not in {str(p).strip().lower() for p in platforms}:
        return
    adapter = runner._delivery_adapter_for(source)
    if adapter is None or getattr(type(adapter), "send_model_picker", None) is None:
        return  # platform has no interactive picker

    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_cli.model_switch import resolve_persist_behavior

    # Same profile resolution as /model (_handle_model_command_locked): on a multiplexed gateway the
    # routed profile's config feeds the picker, and the click (which arrives from the adapter outside
    # the turn's profile scope) commits the switch inside that profile's runtime scope.
    profile_home = None
    if getattr(getattr(runner, "config", None), "multiplex_profiles", False):
        profile_home = runner._resolve_profile_home_for_source(source)
    ctx = _ModelSwitchContext(
        session_key=session_key, source=source, config_path=(profile_home or _hermes_home) / "config.yaml",
        persist_global=resolve_persist_behavior(False, True, is_once=False, explicit_provider=""),
    )
    ctx.read_config()
    ctx.apply_override(runner._session_model_overrides.get(session_key, {}))
    listing_kwargs = dict(
        current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
        current_model=ctx.current_model, user_providers=ctx.user_provs,
        custom_providers=ctx.custom_provs, excluded_providers=ctx.excluded_provs,
        non_blocking_catalogs=True, probe_custom_providers=False, probe_current_custom_provider=True,
    )
    selected = asyncio.Event()

    async def _switch(model_id: str, provider_slug: str) -> str:
        result, error = await runner._perform_model_switch(ctx, model_id, provider_slug, source)
        if error is not None:
            return error
        return await runner._commit_model_switch(result, ctx, source=source, picker=True)

    async def _on_model_selected(_chat_id: str, model_id: str, provider_slug: str) -> str:
        try:
            if profile_home is None:
                return await _switch(model_id, provider_slug)
            from gateway.run import _profile_runtime_scope
            with _profile_runtime_scope(profile_home):
                return await _switch(model_id, provider_slug)
        finally:
            selected.set()  # a failed switch releases the waiting turn too

    if not await runner._send_model_picker(event, source, adapter, session_key, listing_kwargs, _on_model_selected):
        return
    logger.info("[Gateway] Session-start model picker sent for %s (session %s)", platform_name, session_key)
    timeout = _picker_timeout(cfg)
    if timeout <= 0:
        return
    try:
        await asyncio.wait_for(selected.wait(), timeout=timeout)
        logger.info("[Gateway] Session-start model picker resolved (session %s)", session_key)
    except TimeoutError:
        logger.info("[Gateway] Session-start model picker timed out after %.0fs (session %s); using %s",
                    timeout, session_key, ctx.current_model)
        await adapter.send(
            source.chat_id,
            f"No model selected within {timeout:.0f}s; continuing with "
            f"`{ctx.current_model or 'the default model'}`.",
            metadata=runner._thread_metadata_for_source(source),
        )
