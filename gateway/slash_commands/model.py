"""/model, /codex-runtime slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from typing import Optional
import asyncio

from agent.i18n import t
from gateway.platforms.base import MessageEvent
from hermes_cli.config import clear_model_endpoint_credentials
from utils import base_url_host_matches

from gateway.slash_commands._shared import _model_switch_skew_guard, logger

class ModelCommandsMixin:
    """/model, /codex-runtime handlers."""

    def _typed_command_prefix_for(self, platform) -> str:
        """Return the prefix users can always type to reach Hermes commands.

        Reads the adapter's ``typed_command_prefix`` capability flag
        (default "/"). Slack and Matrix return "!" because typed "/"
        commands are blocked in Slack threads / reserved by Matrix clients;
        their adapters rewrite "!command" to "/command" on receive.
        Instruction text built for those platforms must show the prefix
        that actually works when typed.
        """
        adapter = self.adapters.get(platform) if getattr(self, "adapters", None) else None
        return getattr(adapter, "typed_command_prefix", "/") if adapter is not None else "/"

    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /model command — switch model.

        Supports:
          /model                              — interactive picker (Telegram/Discord) or text list
          /model <name>                       — switch model (this session only)
          /model <name> --once                — switch for the next turn only
          /model <name> --session             — switch for this session only (explicit)
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model
        """
        from gateway.run import _hermes_home, _load_gateway_config
        from hermes_cli.model_switch import (
            switch_model as _switch_model, parse_model_switch_args,
            resolve_persist_behavior,
            list_authenticated_providers,
            list_picker_providers,
        )
        from hermes_cli.providers import get_label

        raw_args = event.get_command_args().strip()
        source = event.source
        _command_profile_home = None
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            _command_profile_home = getattr(
                self, "_resolve_profile_home_for_source"
            )(source)

        # Parse --provider, --global, --session, --once, and --refresh flags
        # via the shared single-owner parser (hermes_cli.model_switch).
        request = parse_model_switch_args(raw_args)
        model_input = request.target
        explicit_provider = request.explicit_provider
        is_global_flag = request.is_global
        force_refresh = request.force_refresh
        is_session = request.is_session
        one_turn = request.is_once
        if request.errors:
            # Gateway decoration: "❌ " prefix over the canonical error copy.
            return f"❌ {request.error_messages()[0]}"
        persist_global = resolve_persist_behavior(
            is_global_flag,
            is_session,
            is_once=one_turn,
            explicit_provider=explicit_provider,
        )

        # --refresh: bust the disk cache so the picker shows live data.
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
            except Exception:
                pass

        # Read current model/provider from config
        current_model = ""
        current_provider = "openrouter"
        current_base_url = ""
        current_api_key = ""
        user_provs = None
        custom_provs = None
        excluded_provs = []
        config_path = (_command_profile_home or _hermes_home) / "config.yaml"
        try:
            cfg = _load_gateway_config()
            if cfg:
                model_cfg = cfg.get("model", {})
                if isinstance(model_cfg, dict):
                    current_model = model_cfg.get("default", "")
                    current_provider = model_cfg.get("provider", current_provider)
                    current_base_url = model_cfg.get("base_url", "")
                user_provs = cfg.get("providers")
                try:
                    from hermes_cli.config import get_compatible_custom_providers
                    custom_provs = get_compatible_custom_providers(cfg)
                except Exception:
                    custom_provs = cfg.get("custom_providers")
                _excl = cfg.get("model_catalog", {}).get("excluded_providers")
                if isinstance(_excl, list):
                    excluded_provs = _excl
        except Exception:
            pass

        # Check for session override. Normalize the source the same way a normal
        # message turn does
        # (Telegram DM topic recovery) before deriving the override key, so
        # the override is stored under the key the next message turn reads
        # (#30479).
        source = await asyncio.to_thread(self._normalize_source_for_session_key, source)
        session_key = self._session_key_for_source(source)
        override = self._session_model_overrides.get(session_key, {})
        restore_snapshot = (
            self._snapshot_session_model_override(session_key) if one_turn else None
        )
        if override:
            current_model = override.get("model", current_model)
            current_provider = override.get("provider", current_provider)
            current_base_url = override.get("base_url", current_base_url)
            current_api_key = override.get("api_key", current_api_key)

        # No args: show interactive picker (Telegram/Discord) or text list
        if not model_input and not explicit_provider:
            # Try interactive picker if the platform supports it
            adapter = getattr(self, "_adapter_for_source")(source)
            has_picker = (
                adapter is not None
                and getattr(type(adapter), "send_model_picker", None) is not None
            )

            if has_picker:
                try:
                    # Offload blocking provider-listing (can fall through to a
                    # synchronous urllib HTTP fetch on a stale cache) off the
                    # event loop so the gateway doesn't freeze. See #41289.
                    providers = await asyncio.to_thread(
                        list_picker_providers,
                        current_provider=current_provider,
                        current_base_url=current_base_url,
                        current_model=current_model,
                        user_providers=user_provs,
                        custom_providers=custom_provs,
                        max_models=50,
                        include_moa=True,
                        excluded_providers=excluded_provs,
                    )
                except Exception:
                    providers = []

                if providers:
                    # Build a callback closure for when the user picks a model.
                    # Captures self + locals needed for the switch logic.
                    _self = self
                    _session_key = session_key
                    _cur_model = current_model
                    _cur_provider = current_provider
                    _cur_base_url = current_base_url
                    _cur_api_key = current_api_key
                    _picker_profile_home = _command_profile_home

                    async def _on_model_selected_scoped(
                        _chat_id: str, model_id: str, provider_slug: str
                    ) -> str:
                        """Perform the model switch and return confirmation text."""
                        skew_error = _model_switch_skew_guard()
                        if skew_error:
                            return skew_error
                        # Offload the switch off the event loop — switch_model()
                        # can fall through to a synchronous models.dev HTTP fetch
                        # (requests.get, 15s timeout) on a cold/expired cache,
                        # which freezes the gateway otherwise. See #20525, #41289.
                        result = await asyncio.to_thread(
                            _switch_model,
                            raw_input=model_id,
                            current_provider=_cur_provider,
                            current_model=_cur_model,
                            current_base_url=_cur_base_url,
                            current_api_key=_cur_api_key,
                            is_global=persist_global,
                            explicit_provider=provider_slug,
                            user_providers=user_provs,
                            custom_providers=custom_provs,
                        )
                        if not result.success:
                            return t("gateway.model.error_prefix", error=result.error_message)

                        try:
                            from hermes_cli.context_switch_guard import (
                                enrich_model_switch_warnings_for_gateway,
                            )

                            # Offload: merge_preflight_compression_warning()
                            # calls the sync resolve_display_context_length()
                            # provider probe ladder — must not run on the loop.
                            await asyncio.to_thread(
                                enrich_model_switch_warnings_for_gateway,
                                result,
                                _self,
                                session_key=_session_key,
                                source=event.source,
                                custom_providers=custom_provs,
                                load_gateway_config=_load_gateway_config,
                            )
                        except Exception as exc:
                            logger.debug("preflight-compression switch warning failed: %s", exc)

                        # Update cached agent in-place
                        cached_entry = None
                        _cache_lock = getattr(_self, "_agent_cache_lock", None)
                        _cache = getattr(_self, "_agent_cache", None)
                        if _cache_lock and _cache is not None:
                            with _cache_lock:
                                cached_entry = _cache.get(_session_key)
                        if cached_entry and cached_entry[0] is not None:
                            try:
                                cached_entry[0].switch_model(
                                    new_model=result.new_model,
                                    new_provider=result.target_provider,
                                    api_key=result.api_key,
                                    base_url=result.base_url,
                                    api_mode=result.api_mode,
                                )
                            except Exception as exc:
                                # The in-place swap rolled the agent back to the
                                # OLD working model/client and re-raised.  Abort
                                # the rest of the commit: do NOT persist the
                                # failed model to the DB, do NOT set a session
                                # override pointing at the broken model, and do
                                # NOT evict the working cached agent.  Otherwise
                                # the next message rebuilds a dead agent from the
                                # broken override and the conversation is lost
                                # (#50163).  A failed switch must be a no-op.
                                logger.warning(
                                    "Picker model switch failed for cached agent: %s", exc
                                )
                                return t(
                                    "gateway.model.error_prefix",
                                    error=(
                                        f"Model switch to {result.new_model} failed ({exc}); "
                                        f"staying on {_cur_model}."
                                    ),
                                )

                        # Persist the new model to the session DB so the
                        # dashboard shows the updated model (#34850).
                        _sess_db = getattr(_self, "_session_db", None)
                        if _sess_db is not None:
                            try:
                                _sess_entry = await _self.async_session_store.get_or_create_session(
                                    event.source
                                )
                                await _sess_db.update_session_model(
                                    _sess_entry.session_id, result.new_model
                                )
                            except Exception as exc:
                                logger.debug(
                                    "Failed to persist model switch to DB: %s", exc
                                )

                        # Store model note + session override.  Use display
                        # form (strips opaque Palantir prefix) for the user-
                        # visible note; session-override map still gets the
                        # full opaque ID, which is what the wire needs.
                        from hermes_cli.model_switch import format_model_for_display
                        _display_cur = format_model_for_display(_cur_model)
                        _display_new = format_model_for_display(result.new_model)
                        if not hasattr(_self, "_pending_model_notes"):
                            _self._pending_model_notes = {}
                        _self._pending_model_notes[_session_key] = (
                            f"[Note: model was just switched from {_display_cur} to {_display_new} "
                            f"via {result.provider_label or result.target_provider}. "
                            f"Adjust your self-identification accordingly.]"
                        )
                        _self._session_model_overrides[_session_key] = {
                            "model": result.new_model,
                            "provider": result.target_provider,
                            "api_key": result.api_key,
                            "base_url": result.base_url,
                            "api_mode": result.api_mode,
                        }

                        # Write-through the non-secret parts to the session
                        # store so the picked model survives a gateway restart
                        # (api_key is never persisted).
                        try:
                            await _self.async_session_store.set_model_override(
                                _session_key,
                                _self._session_model_overrides[_session_key],
                            )
                        except Exception:
                            logger.debug(
                                "Failed to persist session model override",
                                exc_info=True,
                            )

                        # Evict cached agent so the next turn creates a fresh
                        # agent from the override rather than relying on the
                        # stale cache signature to trigger a rebuild.
                        _self._evict_cached_agent(_session_key)

                        # Persist to config (default) unless --session opted out,
                        # mirroring the text /model command path above so a picked
                        # model survives across sessions like a typed one (#49066).
                        if persist_global:
                            try:
                                # Write-back round-trip: raw read is correct
                                # (merged defaults must not be persisted).
                                from hermes_cli.config import read_user_config_raw
                                _persist_cfg = read_user_config_raw(config_path)
                                _raw_model = _persist_cfg.get("model")
                                if isinstance(_raw_model, dict):
                                    _persist_model_cfg = _raw_model
                                elif isinstance(_raw_model, str) and _raw_model.strip():
                                    _persist_model_cfg = {"default": _raw_model.strip()}
                                    _persist_cfg["model"] = _persist_model_cfg
                                else:
                                    _persist_model_cfg = {}
                                    _persist_cfg["model"] = _persist_model_cfg
                                try:
                                    from hermes_cli.route_identity import should_clear_context_pin_async

                                    if await should_clear_context_pin_async(
                                        _persist_model_cfg.get("default")
                                        or _persist_model_cfg.get("model"),
                                        result.new_model,
                                        _persist_model_cfg.get("base_url"),
                                        result.base_url,
                                        _persist_model_cfg.get("provider"),
                                        result.target_provider,
                                    ):
                                        _persist_model_cfg.pop("context_length", None)
                                except Exception:
                                    _persist_model_cfg.pop("context_length", None)
                                _persist_model_cfg["default"] = result.new_model
                                _persist_model_cfg["provider"] = result.target_provider
                                # Named providers always resolve base_url/api_mode fresh,
                                # so any leftover is cleared unconditionally below. Custom
                                # providers have no registry entry to re-derive from, so
                                # they need an explicit set-or-clear here — the previous
                                # lone `if result.base_url:` left a stale base_url behind
                                # when switching to a custom provider whose resolver
                                # returned an empty base_url (#25107).
                                _is_custom_target = str(result.target_provider or "").strip().lower() == "custom"
                                if result.base_url:
                                    _persist_model_cfg["base_url"] = result.base_url
                                elif _is_custom_target:
                                    _persist_model_cfg.pop("base_url", None)
                                if _is_custom_target:
                                    if result.api_mode:
                                        _persist_model_cfg["api_mode"] = result.api_mode
                                    else:
                                        _persist_model_cfg.pop("api_mode", None)
                                else:
                                    clear_model_endpoint_credentials(_persist_model_cfg, clear_base_url=True)
                                from hermes_cli.config import save_config
                                save_config(_persist_cfg)
                            except Exception as e:
                                logger.warning("Failed to persist model switch: %s", e)

                        # Build confirmation text.  Use display form so opaque
                        # Palantir IDs (ri.language-model-service..*) get
                        # shortened to their trailing slug for the UI.
                        plabel = result.provider_label or result.target_provider
                        lines = [t("gateway.model.switched", model=format_model_for_display(result.new_model))]
                        lines.append(t("gateway.model.provider_label", provider=plabel))
                        mi = result.model_info
                        from hermes_cli.model_switch import resolve_display_context_length_async
                        _sw_config_ctx = None
                        _sw_model_cfg = {}
                        try:
                            _sw_cfg = _load_gateway_config()
                            _sw_model_cfg = _sw_cfg.get("model", {})
                            if isinstance(_sw_model_cfg, dict):
                                _sw_raw = _sw_model_cfg.get("context_length")
                                if _sw_raw is not None:
                                    _sw_config_ctx = int(_sw_raw)
                        except Exception:
                            pass
                        if not isinstance(_sw_model_cfg, dict):
                            _sw_model_cfg = {}
                        ctx = await resolve_display_context_length_async(
                            result.new_model,
                            result.target_provider,
                            base_url=result.base_url or current_base_url or "",
                            api_key=result.api_key or current_api_key or "",
                            model_info=mi,
                            custom_providers=custom_provs,
                            config_context_length=_sw_config_ctx,
                            configured_model=(
                                _sw_model_cfg.get("default")
                                or _sw_model_cfg.get("model")
                            ),
                            configured_provider=_sw_model_cfg.get("provider"),
                            configured_base_url=_sw_model_cfg.get("base_url"),
                        )
                        if ctx:
                            lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
                        if mi:
                            if mi.max_output:
                                lines.append(t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}"))
                            lines.append(t("gateway.model.capabilities_label", capabilities=mi.format_capabilities()))
                        if result.warning_message:
                            lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))
                        if persist_global:
                            lines.append(t("gateway.model.saved_global"))
                        else:
                            lines.append(t("gateway.model.session_only_hint"))
                        return "\n".join(lines)

                    async def _on_model_selected(
                        _chat_id: str, model_id: str, provider_slug: str
                    ) -> str:
                        if _picker_profile_home is None:
                            return await _on_model_selected_scoped(
                                _chat_id, model_id, provider_slug
                            )
                        from gateway.run import _profile_runtime_scope

                        with _profile_runtime_scope(_picker_profile_home):
                            return await _on_model_selected_scoped(
                                _chat_id, model_id, provider_slug
                            )

                    metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                    result = await adapter.send_model_picker(
                        chat_id=source.chat_id,
                        providers=providers,
                        current_model=current_model,
                        current_provider=current_provider,
                        session_key=session_key,
                        on_model_selected=_on_model_selected,
                        metadata=metadata,
                    )
                    if result.success:
                        return None  # Picker sent — adapter handles the response

            # Fallback: text list (for platforms without picker or if picker failed)
            provider_label = get_label(current_provider)
            lines = [t("gateway.model.current_label", model=current_model or "unknown", provider=provider_label), ""]

            try:
                # Offload blocking provider-listing off the event loop so the
                # gateway doesn't freeze on a stale-cache HTTP fetch. See #41289.
                providers = await asyncio.to_thread(
                    list_authenticated_providers,
                    current_provider=current_provider,
                    current_base_url=current_base_url,
                    current_model=current_model,
                    user_providers=user_provs,
                    custom_providers=custom_provs,
                    max_models=5,
                    excluded_providers=excluded_provs,
                )
                for p in providers:
                    tag = t("gateway.model.current_tag") if p["is_current"] else ""
                    lines.append(f"**{p['name']}** `--provider {p['slug']}`{tag}:")
                    if p["models"]:
                        model_strs = ", ".join(f"`{m}`" for m in p["models"])
                        extra = t("gateway.model.more_models_suffix", count=p["total_models"] - len(p["models"])) if p["total_models"] > len(p["models"]) else ""
                        lines.append(f"  {model_strs}{extra}")
                    elif p.get("api_url"):
                        lines.append(f"  `{p['api_url']}`")
                    lines.append("")
            except Exception:
                pass

            lines.append(t("gateway.model.usage_switch_model"))
            lines.append(t("gateway.model.usage_switch_provider"))
            lines.append(t("gateway.model.usage_persist"))
            return "\n".join(lines)

        # Perform the switch
        skew_error = _model_switch_skew_guard()
        if skew_error:
            return skew_error
        # Offload the switch off the event loop — switch_model() can fall
        # through to a synchronous models.dev HTTP fetch (requests.get, 15s
        # timeout) on a cold/expired cache, which freezes the gateway
        # otherwise. See #20525, #41289.
        result = await asyncio.to_thread(
            _switch_model,
            raw_input=model_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key=current_api_key,
            is_global=persist_global,
            explicit_provider=explicit_provider,
            user_providers=user_provs,
            custom_providers=custom_provs,
        )

        if not result.success:
            return t("gateway.model.error_prefix", error=result.error_message)

        try:
            from hermes_cli.context_switch_guard import (
                enrich_model_switch_warnings_for_gateway,
            )

            # Offload: merge_preflight_compression_warning() calls the sync
            # resolve_display_context_length() provider probe ladder — must
            # not run on the loop.
            await asyncio.to_thread(
                enrich_model_switch_warnings_for_gateway,
                result,
                self,
                session_key=session_key,
                source=source,
                custom_providers=custom_provs,
                load_gateway_config=_load_gateway_config,
            )
        except Exception as exc:
            logger.debug("preflight-compression switch warning failed: %s", exc)

        async def _finish_switch() -> str:
            """Apply the resolved switch (agent, session, config) and build the reply."""
            # If there's a cached agent, update it in-place
            cached_entry = None
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached_entry = _cache.get(session_key)

            if cached_entry and cached_entry[0] is not None:
                try:
                    cached_entry[0].switch_model(
                        new_model=result.new_model,
                        new_provider=result.target_provider,
                        api_key=result.api_key,
                        base_url=result.base_url,
                        api_mode=result.api_mode,
                    )
                except Exception as exc:
                    # In-place swap rolled the agent back to the OLD working
                    # model/client and re-raised.  Abort the commit: skip DB
                    # persist, session override, cache eviction, and config
                    # write so a failed switch is a no-op rather than a dead
                    # conversation (#50163).  Without this early return the
                    # next message rebuilds a broken agent from the override.
                    logger.warning("In-place model switch failed for cached agent: %s", exc)
                    return t(
                        "gateway.model.error_prefix",
                        error=(
                            f"Model switch to {result.new_model} failed ({exc}); "
                            f"staying on {current_model}."
                        ),
                    )

            # Persist the new model to the session DB so the dashboard
            # shows the updated model (#34850).
            _sess_db = getattr(self, "_session_db", None)
            if _sess_db is not None:
                try:
                    _sess_entry = await self.async_session_store.get_or_create_session(source)
                    # If this session was auto-reset, consume the flag so the
                    # next regular message's cleanup does not wipe the model
                    # override just stored below (Closes #48031).
                    if getattr(_sess_entry, "was_auto_reset", False):
                        _sess_entry.was_auto_reset = False
                    await _sess_db.update_session_model(
                        _sess_entry.session_id, result.new_model
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to persist model switch to DB: %s", exc
                    )

            # Store a note to prepend to the next user message so the model
            # knows about the switch (avoids system messages mid-history).
            # Display form strips opaque Palantir RID prefixes; the override
            # map below keeps the full ID for the wire.
            from hermes_cli.model_switch import format_model_for_display
            if not hasattr(self, "_pending_model_notes"):
                self._pending_model_notes = {}
            self._pending_model_notes[session_key] = (
                f"[Note: model was just switched from {format_model_for_display(current_model)} to {format_model_for_display(result.new_model)} "
                f"via {result.provider_label or result.target_provider}. "
                f"{'This override applies to the next turn only. ' if one_turn else ''}"
                f"Adjust your self-identification accordingly.]"
            )

            # Store session override so next agent creation uses the new model
            self._session_model_overrides[session_key] = {
                "model": result.new_model,
                "provider": result.target_provider,
                "api_key": result.api_key,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
            }
            if one_turn:
                if not hasattr(self, "_pending_one_turn_model_restores"):
                    self._pending_one_turn_model_restores = {}
                self._pending_one_turn_model_restores[session_key] = (
                    restore_snapshot or {"had_override": False, "override": None}
                )
            elif hasattr(self, "_pending_one_turn_model_restores"):
                self._pending_one_turn_model_restores.pop(session_key, None)

            # Write-through the non-secret parts (model/provider/base_url) to
            # the session store so the override survives a gateway restart.
            # api_key/api_mode are never persisted — they are re-resolved via
            # runtime provider resolution on rehydration.
            #
            # /model --once is intentionally EXCLUDED from the write-through:
            # a one-turn override must never survive a restart. The persisted
            # value stays at the pre-once state (the prior session override,
            # or nothing), which is exactly what the finally-restore reverts
            # the in-memory dict to. (#29923 review defect: the original
            # implementation wrote through, so a crash before the restore
            # rehydrated the once-model permanently.)
            if not one_turn:
                try:
                    await self.async_session_store.set_model_override(
                        session_key,
                        self._session_model_overrides[session_key],
                    )
                except Exception:
                    logger.debug(
                        "Failed to persist session model override", exc_info=True
                    )

            # Evict cached agent so the next turn creates a fresh agent from the
            # override rather than relying on cache signature mismatch detection.
            self._evict_cached_agent(session_key)

            # Persist to config (default) unless --session opted out
            if persist_global:
                try:
                    # Write-back round-trip: raw read is correct (merged
                    # defaults must not be persisted back to the user's file).
                    from hermes_cli.config import read_user_config_raw
                    cfg = read_user_config_raw(config_path)
                    # Coerce scalar/None ``model:`` into a dict before mutation —
                    # otherwise ``cfg.setdefault("model", {})`` returns the existing
                    # scalar and the next assignment raises
                    # ``TypeError: 'str' object does not support item assignment``.
                    # Reproduces when ``config.yaml`` has ``model: <name>`` (flat
                    # string) instead of the proper nested ``model: {default: ...}``.
                    raw_model = cfg.get("model")
                    if isinstance(raw_model, dict):
                        model_cfg = raw_model
                    elif isinstance(raw_model, str) and raw_model.strip():
                        model_cfg = {"default": raw_model.strip()}
                        cfg["model"] = model_cfg
                    else:
                        model_cfg = {}
                        cfg["model"] = model_cfg
                    try:
                        from hermes_cli.route_identity import should_clear_context_pin_async

                        if await should_clear_context_pin_async(
                            model_cfg.get("default") or model_cfg.get("model"),
                            result.new_model,
                            model_cfg.get("base_url"),
                            result.base_url,
                            model_cfg.get("provider"),
                            result.target_provider,
                        ):
                            model_cfg.pop("context_length", None)
                    except Exception:
                        model_cfg.pop("context_length", None)
                    model_cfg["default"] = result.new_model
                    model_cfg["provider"] = result.target_provider
                    # See the picker handler above for why custom providers need an
                    # explicit set-or-clear instead of the old lone truthy check (#25107).
                    _is_custom_target = str(result.target_provider or "").strip().lower() == "custom"
                    if result.base_url:
                        model_cfg["base_url"] = result.base_url
                    elif _is_custom_target:
                        model_cfg.pop("base_url", None)
                    if _is_custom_target:
                        if result.api_mode:
                            model_cfg["api_mode"] = result.api_mode
                        else:
                            model_cfg.pop("api_mode", None)
                    else:
                        clear_model_endpoint_credentials(model_cfg, clear_base_url=True)
                    from hermes_cli.config import save_config
                    save_config(cfg)
                except Exception as e:
                    logger.warning("Failed to persist model switch: %s", e)

            # Build confirmation message with full metadata
            provider_label = result.provider_label or result.target_provider
            lines = [t("gateway.model.switched", model=format_model_for_display(result.new_model))]
            lines.append(t("gateway.model.provider_label", provider=provider_label))

            # Context: always resolve via the provider-aware chain so Codex OAuth,
            # Copilot, and Nous-enforced caps win over the raw models.dev entry.
            mi = result.model_info
            from hermes_cli.model_switch import resolve_display_context_length_async
            _sw2_config_ctx = None
            _sw2_model_cfg = {}
            try:
                _sw2_cfg = _load_gateway_config()
                _sw2_model_cfg = _sw2_cfg.get("model", {})
                if isinstance(_sw2_model_cfg, dict):
                    _sw2_raw = _sw2_model_cfg.get("context_length")
                    if _sw2_raw is not None:
                        _sw2_config_ctx = int(_sw2_raw)
            except Exception:
                pass
            if not isinstance(_sw2_model_cfg, dict):
                _sw2_model_cfg = {}
            ctx = await resolve_display_context_length_async(
                result.new_model,
                result.target_provider,
                base_url=result.base_url or current_base_url or "",
                api_key=result.api_key or current_api_key or "",
                model_info=mi,
                custom_providers=custom_provs,
                config_context_length=_sw2_config_ctx,
                configured_model=(
                    _sw2_model_cfg.get("default")
                    or _sw2_model_cfg.get("model")
                ),
                configured_provider=_sw2_model_cfg.get("provider"),
                configured_base_url=_sw2_model_cfg.get("base_url"),
            )
            if ctx:
                lines.append(t("gateway.model.context_label", tokens=f"{ctx:,}"))
            if mi:
                if mi.max_output:
                    lines.append(t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}"))
                lines.append(t("gateway.model.capabilities_label", capabilities=mi.format_capabilities()))

            # Cache notice
            cache_enabled = (
                (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
                or result.api_mode == "anthropic_messages"
            )
            if cache_enabled:
                lines.append(t("gateway.model.prompt_caching_enabled"))

            if result.warning_message:
                lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))

            if persist_global:
                lines.append(t("gateway.model.saved_global"))
            elif one_turn:
                lines.append("    (next turn only — restores after one response)")
            else:
                lines.append(t("gateway.model.session_only_hint"))

            return "\n".join(lines)

        # Expensive-model confirmation gate (typed /model <name> path).
        # The pickers (Telegram/Discord inline keyboards, TUI, dashboard)
        # already confirm via their own UI affordances; this covers the
        # direct text command, which previously bypassed the guard.
        # expensive_model_warning() may hit models.dev or a /models endpoint
        # on a cache miss, so run it off the event loop.
        _cost_warning = None
        try:
            from hermes_cli.model_cost_guard import expensive_model_warning

            _cost_warning = await asyncio.to_thread(
                expensive_model_warning,
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or current_base_url or "",
                api_key=result.api_key or current_api_key or "",
                model_info=result.model_info,
            )
        except Exception:
            _cost_warning = None
        if _cost_warning is not None:
            async def _on_cost_confirm(choice: str) -> str:
                if choice == "cancel":
                    return (
                        f"🟡 Model switch cancelled. Current model unchanged "
                        f"({current_model or 'unknown'})."
                    )
                # "once" and "always" both proceed — there is no persistent
                # opt-out for the cost guard (each expensive switch should be
                # an explicit decision).
                return await _finish_switch()

            _p = self._typed_command_prefix_for(event.source.platform)
            return await self._request_slash_confirm(
                event=event,
                command="model",
                title="Expensive Model Warning",
                message=(
                    f"⚠️ **Expensive Model Warning**\n\n{_cost_warning.message}\n\n"
                    f"_Text fallback: reply `{_p}approve` to switch or `{_p}cancel` to keep "
                    "the current model._"
                ),
                handler=_on_cost_confirm,
            )

        return await _finish_switch()

    async def _handle_codex_runtime_command(self, event: MessageEvent) -> str:
        """Handle /codex-runtime command in the gateway.

        Same surface as the CLI handler in cli.py:
            /codex-runtime                  — show current state
            /codex-runtime auto             — Hermes default runtime
            /codex-runtime codex_app_server — codex subprocess runtime
            /codex-runtime on / off         — synonyms

        On change, the cached agent for this session is evicted so the next
        message creates a fresh AIAgent with the new api_mode wired in
        (avoids prompt-cache invalidation mid-session)."""
        from hermes_cli import codex_runtime_switch as crs

        raw_args = event.get_command_args().strip() if event else ""
        new_value, errors = crs.parse_args(raw_args)
        if errors:
            return "❌ " + "\n❌ ".join(errors)

        # Load + persist via the same helpers used for /model and /yolo
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            return f"❌ Could not load config: {exc}"
        cfg = load_config()

        result = crs.apply(
            cfg,
            new_value,
            persist_callback=(save_config if new_value is not None else None),
        )

        # On a real change, evict the cached agent so the new runtime takes
        # effect on the next message rather than waiting for cache TTL.
        if result.success and new_value is not None and result.requires_new_session:
            try:
                session_key = self._session_key_for_source(event.source)
                self._evict_cached_agent(session_key)
            except Exception:
                logger.debug("could not evict cached agent after codex-runtime change",
                             exc_info=True)

        prefix = "✓" if result.success else "✗"
        return f"{prefix} {result.message}"
