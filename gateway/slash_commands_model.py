"""Gateway slash commands that switch or tune the model route:
/model, /codex-runtime, /reasoning, /fast, /personality.

Split out of ``gateway/slash_commands.py``; bound onto ``GatewayRunner`` through
``GatewaySlashCommandsMixin``. Origin internals are imported lazily inside the bodies to avoid
the import cycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from typing import Any, Optional

from agent.i18n import t
from gateway.platforms.base import MessageEvent
from hermes_cli.config import atomic_config_write, clear_model_endpoint_credentials
from utils import base_url_host_matches

logger = logging.getLogger("gateway.run")  # log-record parity with gateway/run.py

# /fast argument -> (service tier, persisted value, i18n label key; None = value.upper()).
_FAST_SELECTIONS = {
    "fast": ("priority", "fast", "gateway.fast.label_fast"),
    "on": ("priority", "fast", "gateway.fast.label_fast"),
    "normal": (None, "normal", "gateway.fast.label_normal"),
    "off": (None, "normal", "gateway.fast.label_normal"),
    "auto": ("auto", "auto", None),
    "cold": ("cold", "cold", None),
}

# /reasoning display-toggle arguments -> show_reasoning value.
_REASONING_DISPLAY_TOGGLES = {"show": True, "on": True, "hide": False, "off": False}


def _model_switch_skew_guard() -> Optional[str]:
    """Refuse a model switch when the gateway is running stale code: a first-time lazy import on
    a new code path can crash on a stale cached dependency. Scoped to the highest-risk trigger."""
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return t(
        "gateway.model.error_prefix",
        error=(
            f"This gateway is running code from {boot_rev} but the checkout on "
            f"disk is now {disk_rev}. Switching models would risk a stale-module "
            f"crash — restart the gateway to load the new code: hermes gateway restart"
        ),
    )


async def _persist_model_switch_to_config(result, config_path) -> None:
    """Write-through a resolved /model switch to ``config_path`` (model.default/provider/base_url).

    Raw read: merged defaults must not be persisted back. A scalar/None ``model:`` is coerced to a
    dict first. Named providers re-resolve base_url/api_mode, so leftovers are cleared; custom
    providers have no registry entry to re-derive from and need an explicit set-or-clear.
    """
    from hermes_cli.config import read_user_config_raw, save_config

    cfg = read_user_config_raw(config_path)
    raw_model = cfg.get("model")
    if isinstance(raw_model, dict):
        model_cfg = raw_model
    elif isinstance(raw_model, str) and raw_model.strip():
        model_cfg = cfg["model"] = {"default": raw_model.strip()}
    else:
        model_cfg = cfg["model"] = {}
    try:
        from hermes_cli.route_identity import should_clear_context_pin_async
        clear_pin = await should_clear_context_pin_async(
            model_cfg.get("default") or model_cfg.get("model"), result.new_model,
            model_cfg.get("base_url"), result.base_url, model_cfg.get("provider"), result.target_provider,
        )
    except Exception:
        clear_pin = True
    if clear_pin:
        model_cfg.pop("context_length", None)
    model_cfg["default"] = result.new_model
    model_cfg["provider"] = result.target_provider
    is_custom_target = str(result.target_provider or "").strip().lower() == "custom"
    if result.base_url:
        model_cfg["base_url"] = result.base_url
    elif is_custom_target:
        model_cfg.pop("base_url", None)
    if not is_custom_target:
        clear_model_endpoint_credentials(model_cfg, clear_base_url=True)
    elif result.api_mode:
        model_cfg["api_mode"] = result.api_mode
    else:
        model_cfg.pop("api_mode", None)
    save_config(cfg)


@dataclasses.dataclass
class _ModelSwitchContext:
    """Everything a /model switch needs beyond the target: current route + persistence policy."""

    session_key: str
    source: Any
    config_path: Any
    persist_global: bool
    one_turn: bool = False
    restore_snapshot: Optional[dict] = None
    current_model: str = ""
    current_provider: str = "openrouter"
    current_base_url: str = ""
    current_api_key: str = ""
    user_provs: Any = None
    custom_provs: Any = None
    excluded_provs: list = dataclasses.field(default_factory=list)

    def read_config(self) -> None:
        """Fill the current route from ``config_path``; fail-open to the defaults."""
        from gateway.run import _load_gateway_config
        try:
            cfg = _load_gateway_config(config_path=self.config_path)
            if not cfg:
                return
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                self.current_model = model_cfg.get("default", "")
                self.current_provider = model_cfg.get("provider", self.current_provider)
                self.current_base_url = model_cfg.get("base_url", "")
            self.user_provs = cfg.get("providers")
            try:
                from hermes_cli.config import get_compatible_custom_providers
                self.custom_provs = get_compatible_custom_providers(cfg)
            except Exception:
                self.custom_provs = cfg.get("custom_providers")
            excl = cfg.get("model_catalog", {}).get("excluded_providers")
            if isinstance(excl, list):
                self.excluded_provs = excl
        except Exception:
            pass

    def apply_override(self, override: dict) -> None:
        """A session /model override supersedes the configured route."""
        if override:
            self.current_model = override.get("model", self.current_model)
            self.current_provider = override.get("provider", self.current_provider)
            self.current_base_url = override.get("base_url", self.current_base_url)
            self.current_api_key = override.get("api_key", self.current_api_key)



def _model_provider_listing_lines(providers) -> list[str]:
    """Text-list body for ``/model`` with no args on platforms without a picker."""
    lines: list[str] = []
    for p in providers:
        tag = t("gateway.model.current_tag") if p["is_current"] else ""
        lines.append(f"**{p['name']}** `--provider {p['slug']}`{tag}:")
        if p["models"]:
            model_strs = ", ".join(f"`{m}`" for m in p["models"])
            hidden = p["total_models"] - len(p["models"])
            extra = t("gateway.model.more_models_suffix", count=hidden) if hidden > 0 else ""
            lines.append(f"  {model_strs}{extra}")
        elif p.get("api_url"):
            lines.append(f"  `{p['api_url']}`")
        lines.append("")
    return lines


class GatewayModelCommandsMixin:
    """Model-route slash commands (/model, /codex-runtime, /reasoning, /fast, /personality)."""

    # ----------------------------------------------------------------- /model

    async def _perform_model_switch(
        self, ctx: _ModelSwitchContext, raw_input: str, explicit_provider, source
    ):
        """Resolve a /model switch off-loop. Returns ``(result, None)`` or ``(None, error_text)``."""
        from gateway.run import _load_gateway_config
        from hermes_cli.model_switch import switch_model

        skew_error = _model_switch_skew_guard()
        if skew_error:
            return None, skew_error
        # Off-loop: switch_model() can hit a synchronous models.dev fetch (15s) on a cold cache.
        result = await asyncio.to_thread(
            switch_model, raw_input=raw_input, current_provider=ctx.current_provider,
            current_model=ctx.current_model, current_base_url=ctx.current_base_url,
            current_api_key=ctx.current_api_key, is_global=ctx.persist_global,
            explicit_provider=explicit_provider, user_providers=ctx.user_provs,
            custom_providers=ctx.custom_provs,
        )
        if not result.success:
            return None, t("gateway.model.error_prefix", error=result.error_message)
        try:
            from hermes_cli.context_switch_guard import enrich_model_switch_warnings_for_gateway
            # Off-loop: merge_preflight_compression_warning() runs the sync provider probe ladder.
            await asyncio.to_thread(
                enrich_model_switch_warnings_for_gateway, result, self, session_key=ctx.session_key,
                source=source, custom_providers=ctx.custom_provs,
                load_gateway_config=_load_gateway_config,
            )
        except Exception as exc:
            logger.debug("preflight-compression switch warning failed: %s", exc)
        return result, None

    def _switch_cached_agent_model(self, result, ctx: _ModelSwitchContext, picker: bool) -> Optional[str]:
        """In-place swap on the cached agent; returns the error reply when it failed.

        The agent rolls back to the OLD model/client and re-raises; the commit (DB, override,
        eviction, config) is aborted so the next message doesn't rebuild a broken agent.
        """
        cached_agent = self._cached_agent_for(ctx.session_key)
        if cached_agent is None:
            return None
        try:
            cached_agent.switch_model(
                new_model=result.new_model, new_provider=result.target_provider,
                api_key=result.api_key, base_url=result.base_url, api_mode=result.api_mode,
                capabilities=getattr(result, "runtime_capabilities", None),
            )
        except Exception as exc:
            logger.warning(
                "%s model switch failed for cached agent: %s", "Picker" if picker else "In-place", exc
            )
            return t(
                "gateway.model.error_prefix",
                error=f"Model switch to {result.new_model} failed ({exc}); staying on {ctx.current_model}.",
            )
        return None

    async def _record_model_switch(
        self, result, ctx: _ModelSwitchContext, *, source, one_turn: bool, picker: bool
    ) -> None:
        """Persist a committed switch: session DB, next-turn note, override map, config write-through."""
        from hermes_cli.model_switch import format_model_for_display

        # Persist the new model to the session DB so the dashboard shows the updated model (#34850).
        _sess_db = getattr(self, "_session_db", None)
        if _sess_db is not None:  # so the dashboard shows the updated model
            try:
                _sess_entry = await self.async_session_store.get_or_create_session(source)
                # Typed path: consume the auto-reset flag so the next message's cleanup does not
                # wipe the override stored below.
                if not picker and getattr(_sess_entry, "was_auto_reset", False):
                    # See #48031.
                    _sess_entry.was_auto_reset = False
                await _sess_db.update_session_model(
                    _sess_entry.session_id, result.new_model, provider=result.target_provider,
                )
            except Exception as exc:
                logger.debug("Failed to persist model switch to DB: %s", exc)
        # Prepended to the next user message (no system messages mid-history). Display form strips
        # opaque Palantir RID prefixes; the override map keeps the full ID for the wire.
        if not hasattr(self, "_pending_model_notes"):
            self._pending_model_notes = {}
        self._pending_model_notes[ctx.session_key] = (
            f"[Note: model was just switched from {format_model_for_display(ctx.current_model)} to "
            f"{format_model_for_display(result.new_model)} "
            f"via {result.provider_label or result.target_provider}. "
            f"{'This override applies to the next turn only. ' if one_turn else ''}"
            f"Adjust your self-identification accordingly.]"
        )
        self._session_model_overrides[ctx.session_key] = {
            "model": result.new_model, "provider": result.target_provider, "api_key": result.api_key,
            "base_url": result.base_url, "api_mode": result.api_mode,
            "request_overrides": dict(result.request_overrides or {}),
            "capabilities": dict(result.runtime_capabilities or {}),
        }
        if one_turn:
            if not hasattr(self, "_pending_one_turn_model_restores"):
                self._pending_one_turn_model_restores = {}
            snapshot = ctx.restore_snapshot or {"had_override": False, "override": None}
            self._pending_one_turn_model_restores[ctx.session_key] = snapshot
        elif not picker and hasattr(self, "_pending_one_turn_model_restores"):
            self._pending_one_turn_model_restores.pop(ctx.session_key, None)
        # Non-secret write-through so the override survives a restart (api_key/api_mode are
        # re-resolved on rehydration); a --once override must NOT outlive a restart.
        # Write-through the non-secret parts (model/provider/base_url) to the session store so the override
        # survives a gateway restart. api_key/api_mode are never persisted — they are re-resolved via
        # runtime provider resolution on rehydration. /model --once is intentionally EXCLUDED from the
        # write-through: a one-turn override must never survive a restart. The persisted value stays at the
        # pre-once state (the prior session override, or nothing), which is exactly what the finally-restore
        # reverts the in-memory dict to. (#29923 review defect: the original implementation wrote through,
        # so a crash before the restore rehydrated the once-model permanently.)
        if not one_turn:
            try:
                await self.async_session_store.set_model_override(
                    ctx.session_key, self._session_model_overrides[ctx.session_key]
                )
            except Exception:
                logger.debug("Failed to persist session model override", exc_info=True)
        self._evict_cached_agent(ctx.session_key)  # next turn builds fresh from the override
        if ctx.persist_global:
            try:
                await _persist_model_switch_to_config(result, ctx.config_path)
            except Exception as e:
                logger.warning("Failed to persist model switch: %s", e)

    async def _model_switch_confirmation(
        self, result, ctx: _ModelSwitchContext, *, one_turn: bool, picker: bool
    ) -> str:
        """Confirmation text with full metadata (display form shortens opaque Palantir IDs)."""
        from gateway.run import _load_gateway_config
        from hermes_cli.model_switch import format_model_for_display, resolve_display_context_length_async

        lines = [
            t("gateway.model.switched", model=format_model_for_display(result.new_model)),
            t("gateway.model.provider_label", provider=result.provider_label or result.target_provider),
        ]
        # Provider-aware chain: Codex OAuth, Copilot and Nous caps win over the raw models.dev entry.
        mi = result.model_info
        model_cfg: dict = {}
        config_ctx = None
        with contextlib.suppress(Exception):  # fail-open on config read errors
            model_cfg = _load_gateway_config().get("model", {})
            if isinstance(model_cfg, dict) and model_cfg.get("context_length") is not None:
                config_ctx = int(model_cfg["context_length"])
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        ctx_len = await resolve_display_context_length_async(
            result.new_model, result.target_provider,
            base_url=result.base_url or ctx.current_base_url or "",
            api_key=result.api_key or ctx.current_api_key or "", model_info=mi,
            custom_providers=ctx.custom_provs, config_context_length=config_ctx,
            configured_model=model_cfg.get("default") or model_cfg.get("model"),
            configured_provider=model_cfg.get("provider"),
            configured_base_url=model_cfg.get("base_url"),
        )
        if ctx_len:
            lines.append(t("gateway.model.context_label", tokens=f"{ctx_len:,}"))
        if mi and mi.max_output:
            lines.append(t("gateway.model.max_output_label", tokens=f"{mi.max_output:,}"))
        if mi:
            lines.append(t("gateway.model.capabilities_label", capabilities=mi.format_capabilities()))
        openrouter_claude = (
            base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower()
        )
        if not picker and (openrouter_claude or result.api_mode == "anthropic_messages"):
            lines.append(t("gateway.model.prompt_caching_enabled"))
        if result.warning_message:
            lines.append(t("gateway.model.warning_prefix", warning=result.warning_message))
        if ctx.persist_global:
            lines.append(t("gateway.model.saved_global"))
        elif one_turn:
            lines.append("    (next turn only — restores after one response)")
        else:
            lines.append(t("gateway.model.session_only_hint"))
        return "\n".join(lines)

    async def _commit_model_switch(
        self, result, ctx: _ModelSwitchContext, *, source, picker: bool = False
    ) -> str:
        """Apply a resolved switch (cached agent, session, config) and build the confirmation; shared
        by the typed path and the picker callback (``picker=True`` never carries --once)."""
        one_turn = False if picker else ctx.one_turn
        error = self._switch_cached_agent_model(result, ctx, picker)
        if error is not None:
            return error
        await self._record_model_switch(result, ctx, source=source, one_turn=one_turn, picker=picker)
        return await self._model_switch_confirmation(result, ctx, one_turn=one_turn, picker=picker)

    async def _send_model_picker(self, event: MessageEvent, source, adapter, session_key: str, listing_kwargs: dict, on_model_selected) -> bool:
        """Send the interactive /model picker; False when nothing was sent (text fallback). *source*
        is session-key-normalized so the picker's thread metadata lands where the next turn reads."""
        from hermes_cli.model_switch_providers import list_picker_providers
        try:  # off-loop: listing can hit a synchronous HTTP fetch on a stale cache
            # Offload blocking provider-listing (can fall through to a synchronous urllib HTTP fetch on a
            # stale cache) off the event loop so the gateway doesn't freeze. See #41289.
            # Offload blocking provider-listing off the event loop so the gateway doesn't freeze on a
            # stale-cache HTTP fetch. See #41289.
            providers = await asyncio.to_thread(
                list_picker_providers, max_models=50, include_moa=True, **listing_kwargs
            )
        except Exception:
            providers = []
        if not providers:
            return False
        result = await adapter.send_model_picker(
            chat_id=source.chat_id, providers=providers,
            current_model=listing_kwargs["current_model"],
            current_provider=listing_kwargs["current_provider"], session_key=session_key,
            on_model_selected=on_model_selected,
            metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
        )
        return bool(result.success)

    async def _model_listing_reply(
        self, event: MessageEvent, ctx: _ModelSwitchContext, profile_home
    ) -> Optional[str]:
        """``/model`` with no args: interactive picker where supported, else the text list."""
        from hermes_cli.model_switch import list_authenticated_providers
        from hermes_cli.providers import get_label

        listing_kwargs = dict(
            current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
            current_model=ctx.current_model, user_providers=ctx.user_provs,
            custom_providers=ctx.custom_provs, excluded_providers=ctx.excluded_provs,
        )
        adapter = self._adapter_for_source(ctx.source)
        if adapter is not None and getattr(type(adapter), "send_model_picker", None) is not None:
            async def _picker_switch(model_id: str, provider_slug: str) -> str:
                # The picker callback binds the raw event source (pre-normalization).
                result, error = await self._perform_model_switch(ctx, model_id, provider_slug, event.source)
                if error is not None:
                    return error
                return await self._commit_model_switch(result, ctx, source=event.source, picker=True)

            async def _on_model_selected(_chat_id: str, model_id: str, provider_slug: str) -> str:
                if profile_home is None:
                    return await _picker_switch(model_id, provider_slug)
                from gateway.run import _profile_runtime_scope
                with _profile_runtime_scope(profile_home):
                    return await _picker_switch(model_id, provider_slug)

            if await self._send_model_picker(event, ctx.source, adapter, ctx.session_key, listing_kwargs, _on_model_selected):
                return None  # Picker sent — adapter handles the response

        lines = [t("gateway.model.current_label", model=ctx.current_model or "unknown", provider=get_label(ctx.current_provider)), ""]
        try:  # off-loop: listing can hit a stale-cache HTTP fetch
            providers = await asyncio.to_thread(list_authenticated_providers, max_models=5, **listing_kwargs)
            lines.extend(_model_provider_listing_lines(providers))
        except Exception:
            pass
        lines.append(t("gateway.model.usage_switch_model"))
        lines.append(t("gateway.model.usage_switch_provider"))
        lines.append(t("gateway.model.usage_persist"))
        return "\n".join(lines)

    async def _model_selection_guard_reply(
        self, event: MessageEvent, ctx: _ModelSwitchContext, result
    ) -> tuple[bool, Optional[str]]:
        """Selection-guard confirmation for the typed path (pickers confirm via their own UI).

        The unified registry (cost + data-policy guards) runs off-loop — pricing lookups may hit
        models.dev on a cache miss. Returns ``(fired, reply)``; the reply is None when the platform
        rendered confirm buttons itself.
        """
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning
            warning = await asyncio.to_thread(
                combined_selection_warning, result.new_model, provider=result.target_provider,
                base_url=result.base_url or ctx.current_base_url or "",
                api_key=result.api_key or ctx.current_api_key or "", model_info=result.model_info,
            )
        except Exception:
            warning = None
        if warning is None:
            return False, None

        async def _on_cost_confirm(choice: str) -> str:
            if choice == "cancel":
                return f"🟡 Model switch cancelled. Current model unchanged ({ctx.current_model or 'unknown'})."
            # "once" and "always" both proceed — selection guards have no persistent opt-out.
            return await self._commit_model_switch(result, ctx, source=ctx.source)

        _p = self._typed_command_prefix_for(event.source)
        message = (
            f"⚠️ **{warning.title}**\n\n{warning.message}\n\n"
            f"_Text fallback: reply `{_p}approve` to switch or `{_p}cancel` to keep the current model._"
        )
        return True, await self._request_slash_confirm(
            event=event, command="model", title=warning.title, message=message, handler=_on_cost_confirm,
        )

    async def _handle_model_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /model command — switch model."""
        from gateway.run import _hermes_home
        from hermes_cli.model_switch import parse_model_switch_args, resolve_persist_behavior

        profile_home = None
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            profile_home = self._resolve_profile_home_for_source(event.source)
        request = parse_model_switch_args(event.get_command_args().strip())  # single-owner parser
        if request.errors:
            return f"❌ {request.error_messages()[0]}"  # gateway decoration over canonical copy
        if request.force_refresh:  # bust the disk cache so the picker shows live data
            with contextlib.suppress(Exception):
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
        # Normalize like a message turn (Telegram DM topic recovery) before deriving the override
        # key, so the override lands under the key the next turn reads.
        # Check for session override. See #30479.
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        ctx = _ModelSwitchContext(
            # Gateway routing columns — forward ALL of them at CREATE time, same fix as the
            # compression-rotation bug in agent/conversation_compression.py. Without these, the branched
            # child row has NULL routing columns until switch_session() below calls
            # _record_gateway_session_peer() — a crash/kill anywhere between here and there (most plausibly
            # mid-history-copy, since each append_message call a few lines down is independently
            # best-effort) leaves the branch permanently unroutable: unreachable by chat/thread lookup, and
            # unreachable via /resume's IDOR guard too (which requires the row's chat_id/thread_id to match
            # the caller's). user_id is critical for the fallback lookup path (hermes_state.py:1994-2009)
            # that searches by the complete peer tuple when session_key doesn't match. origin_json and
            # display_name complete the identity (same shape as the reset path's db_create_kwargs in
            # gateway/session.py, #82633) so consumers that read routing/presentation data from state.db
            # (mcp_serve, mirror, channel directory) see the branch row fully formed with zero backfill gap.
            session_key=session_key,
            source=source,
            config_path=(profile_home or _hermes_home) / "config.yaml",
            persist_global=resolve_persist_behavior(
                request.is_global, request.is_session, is_once=request.is_once,
                explicit_provider=request.explicit_provider,
            ),
            one_turn=request.is_once,
            restore_snapshot=self._snapshot_session_model_override(session_key) if request.is_once else None,
        )
        ctx.read_config()
        ctx.apply_override(self._session_model_overrides.get(session_key, {}))
        if not request.target and not request.explicit_provider:
            return await self._model_listing_reply(event, ctx, profile_home)
        result, error = await self._perform_model_switch(ctx, request.target, request.explicit_provider, source)
        if error is not None:
            return error
        guard_fired, guard_reply = await self._model_selection_guard_reply(event, ctx, result)
        if guard_fired:
            return guard_reply
        return await self._commit_model_switch(result, ctx, source=source)

    # -------------------------------------------------- /codex-runtime, /personality

    async def _handle_codex_runtime_command(self, event: MessageEvent) -> str:
        """Handle /codex-runtime; a real change evicts the cached agent so the new api_mode applies
        on the next message (avoids prompt-cache invalidation mid-session)."""
        from hermes_cli import codex_runtime_switch as crs

        new_value, errors = crs.parse_args(event.get_command_args().strip() if event else "")
        if errors:
            return "❌ " + "\n❌ ".join(errors)
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            return f"❌ Could not load config: {exc}"
        result = crs.apply(
            load_config(), new_value, persist_callback=(save_config if new_value is not None else None),
        )
        if result.success and new_value is not None and result.requires_new_session:
            try:
                self._evict_cached_agent(self._session_key_for_source(event.source))
            except Exception:
                logger.debug("could not evict cached agent after codex-runtime change", exc_info=True)
        return f"{'✓' if result.success else '✗'} {result.message}"

    async def _handle_personality_command(self, event: MessageEvent) -> str:
        """Handle /personality — list or set a personality (hermes_cli.personality owns the state)."""
        from gateway.run import _load_gateway_config
        from hermes_cli.personality import (
            active_personality_name,
            available_personalities,
            describe_personality,
            persist_personality,
            resolve_personality,
        )

        args = event.get_command_args().strip()
        try:
            config = _load_gateway_config()
        except Exception:
            config = {}
        personalities = available_personalities(config)
        if not args:
            current = active_personality_name(config)
            lines = [t("gateway.personality.header"), t("gateway.personality.none_option")]
            for name, prompt in personalities.items():
                marker = " ✓" if name == current else ""
                lines.append(
                    t("gateway.personality.item", name=f"{name}{marker}", preview=describe_personality(prompt))
                )
            lines.append(t("gateway.personality.usage"))
            return "\n".join(lines)
        try:
            name, _new_prompt = resolve_personality(args, config)
        except ValueError:
            available = "`none`, " + ", ".join(f"`{n}`" for n in personalities)
            return t("gateway.personality.unknown", name=args.lower(), available=available)
        # Persists the selection only (never agent.system_prompt, a user-owned overlay) into the
        # routed profile's config.yaml; the next turn re-resolves the prompt — no process-global state.
        if not persist_personality(name):
            return t("gateway.personality.save_failed", error="config write failed")
        if not name:
            return t("gateway.personality.cleared")
        return t("gateway.personality.set_to", name=name)

    # ----------------------------------------------------------- /reasoning, /fast

    def _save_gateway_config_key(self, key_path: str, value) -> bool:
        """Save a dot-separated key to config.yaml (shared by /reasoning, /fast and their pickers)."""
        from gateway.slash_commands import _nested_dict
        from gateway.run import _gateway_config_home
        from hermes_cli.config import read_user_config_raw
        config_path = _gateway_config_home() / "config.yaml"
        try:
            user_config = read_user_config_raw(config_path)  # raw: never persist merged defaults
            *parents, leaf = key_path.split(".")
            _nested_dict(user_config, *parents)[leaf] = value
            atomic_config_write(config_path, user_config)
            return True
        except Exception as e:
            logger.error("Failed to save config key %s: %s", key_path, e)
            return False

    def _set_reasoning_override(self, session_key: str, value) -> None:
        """Store (or clear with None) the session reasoning override and drop the cached agent."""
        self._set_session_reasoning_override(session_key, value)
        self._evict_cached_agent(session_key)

    def _apply_reasoning_selection(
        self, session_key: str, platform_key: str, value: str, persist_global: bool = False,
    ) -> str:
        """Apply a /reasoning argument (typed or picked) and return the reply."""
        from hermes_constants import parse_reasoning_effort

        value = (value or "").strip().lower()
        show = _REASONING_DISPLAY_TOGGLES.get(value)
        if show is not None:  # per-platform display toggle
            self._show_reasoning = show
            self._save_gateway_config_key(f"display.platforms.{platform_key}.show_reasoning", show)
            key = "gateway.reasoning.display_set_on" if show else "gateway.reasoning.display_set_off"
            return t(key, platform=platform_key)
        if value == "reset":
            if persist_global:
                return t("gateway.reasoning.reset_global_unsupported")
            self._set_session_reasoning_override(session_key, None)
            self._reasoning_config = self._load_reasoning_config()
            self._evict_cached_agent(session_key)
            return t("gateway.reasoning.reset_done")

        parsed = parse_reasoning_effort(value)
        if parsed is None:
            return t("gateway.reasoning.unknown_arg", arg=value)
        self._reasoning_config = parsed
        if persist_global:
            if self._save_gateway_config_key("agent.reasoning_effort", value):
                self._set_reasoning_override(session_key, None)
                return t("gateway.reasoning.set_global", effort=value)
            self._set_reasoning_override(session_key, parsed)
            return t("gateway.reasoning.set_global_save_failed", effort=value)
        self._set_reasoning_override(session_key, parsed)
        return t("gateway.reasoning.set_session", effort=value)

    async def _try_send_choice_picker(
        self,
        event: MessageEvent,
        session_key: str,
        title: str,
        choices: list,
        on_choice_selected,
        *,
        reusable: bool = False,
    ) -> bool:
        """Send a native picker, or return False for the caller's text fallback.

        reusable opts into ChoicePage/ChoiceProgress callback results. A str
        still finishes the menu. No callback arity or one-shot call changes.
        """
        from gateway.choice_picker import ChoicePage

        adapter = self._adapter_for_source(event.source)
        has_picker = (
            adapter is not None
            and getattr(type(adapter), "send_choice_picker", None) is not None
        )
        if not has_picker:
            return False
        if (
            reusable
            and getattr(type(adapter), "supports_choice_pages", False) is not True
        ):
            return False
        try:
            if reusable:
                page = ChoicePage(title, choices)
                choices = page.choices
            metadata = dict(
                self._thread_metadata_for_source(
                    event.source, self._reply_anchor_for_event(event)
                )
                or {}
            )
            requester_user_id = getattr(event.source, "user_id", None)
            if requester_user_id is not None:
                metadata["requester_user_id"] = str(requester_user_id)
            if reusable:
                if not metadata.get("requester_user_id"):
                    return False
                metadata["choice_pages"] = True
            result = await adapter.send_choice_picker(
                chat_id=event.source.chat_id,
                title=title,
                choices=choices,
                session_key=session_key,
                on_choice_selected=on_choice_selected,
                metadata=metadata,
            )
            return bool(getattr(result, "success", False))
        except Exception as exc:
            logger.warning("send_choice_picker failed, falling back to text: %s", exc)
            return False

    async def _handle_reasoning_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reasoning command — manage reasoning effort and display toggle."""
        from gateway.run import _platform_config_key
        from hermes_constants import VALID_REASONING_EFFORTS

        raw_args = event.get_command_args().strip()
        args, persist_global = self._parse_reasoning_command_args(raw_args)
        # Normalize (Telegram DM topic recovery) so the override key matches the next turn's.
        # See #30479.
        _reasoning_source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(_reasoning_source)
        self._show_reasoning = self._load_show_reasoning()
        # Effective model (session /model override wins) so per-model reasoning_overrides display.
        _session_model = str(
            ((getattr(self, "_session_model_overrides", {}) or {}).get(session_key) or {}).get("model") or ""
        )
        self._reasoning_config = self._resolve_session_reasoning_config(
            source=event.source, session_key=session_key, model=_session_model,
        )
        platform_key = _platform_config_key(event.source.platform)
        if raw_args:  # typed path — same applier the picker uses
            return self._apply_reasoning_selection(session_key, platform_key, args, persist_global=persist_global)
        rc = self._reasoning_config
        if rc is None:
            level, current_effort = t("gateway.reasoning.level_default"), "medium"
        elif rc.get("enabled") is False:
            level, current_effort = t("gateway.reasoning.level_disabled"), "none"
        else:
            level = current_effort = rc.get("effort", "medium")
        display_state = t("gateway.reasoning.display_on") if self._show_reasoning else t("gateway.reasoning.display_off")
        has_session_override = session_key in (getattr(self, "_session_reasoning_overrides", {}) or {})
        scope = t("gateway.reasoning.scope_session") if has_session_override else t("gateway.reasoning.scope_global")

        async def _on_reasoning_choice(_chat_id: str, value: str) -> str:
            return self._apply_reasoning_selection(session_key, platform_key, value)

        picker_sent = await self._try_send_choice_picker(
            event,
            session_key,
            title=t("gateway.reasoning.picker_title", level=level, scope=scope, display=display_state),
            choices=[
                {"value": "none", "label": t("gateway.reasoning.choice_none"), "is_current": current_effort == "none"},
                *({"value": lv, "label": lv, "is_current": lv == current_effort} for lv in VALID_REASONING_EFFORTS),
                *({"value": v, "label": t(f"gateway.reasoning.choice_{v}"), "is_current": False}
                  for v in ("reset", "show", "hide")),
            ],
            on_choice_selected=_on_reasoning_choice,
        )
        if picker_sent:
            return None  # Picker sent — adapter handles the response
        return t("gateway.reasoning.status", level=level, scope=scope, display=display_state)

    def _apply_fast_selection(self, session_key: str, value: str, persist: bool = False) -> str:
        """Apply a /fast argument (typed or picked) and return the reply."""
        selection = _FAST_SELECTIONS.get(value)
        if selection is None:
            return t("gateway.fast.unknown_arg", arg=value)
        tier, saved_value, label_key = selection
        label = t(label_key) if label_key else value.upper()
        self._service_tier = tier
        if persist and self._save_gateway_config_key("agent.service_tier", saved_value):
            self._set_session_service_tier_override(session_key, None, clear=True)  # global wins
            self._evict_cached_agent(session_key)
            return t("gateway.fast.saved", label=label)
        # Session override — also the fallback after a failed config write (as /reasoning --global).
        self._set_session_service_tier_override(session_key, tier)
        self._evict_cached_agent(session_key)
        return t("gateway.fast.session_only", label=label)

    async def _handle_fast_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /fast — the CLI Priority Processing toggle; session-scoped unless ``--global``
        (persists agent.service_tier, parity with /model)."""
        from gateway.run import _load_gateway_config, _resolve_gateway_model
        from hermes_cli.models import model_supports_fast_mode

        # The /reasoning parser strips --global (any position) and normalizes unicode dashes.
        args, persist_global = self._parse_reasoning_command_args(event.get_command_args().strip().lower())
        session_key = self._session_key_for_source(event.source)
        self._service_tier = self._resolve_session_service_tier(session_key=session_key)
        if not model_supports_fast_mode(_resolve_gateway_model(_load_gateway_config())):
            return t("gateway.fast.not_supported")
        if args and args != "status":
            return self._apply_fast_selection(session_key, args, persist=persist_global)
        mode = "fast" if self._service_tier == "priority" else (self._service_tier or "normal")
        status = {"fast": t("gateway.fast.status_fast"), "normal": t("gateway.fast.status_normal")}.get(mode, mode)

        async def _on_fast_choice(_chat_id: str, value: str) -> str:
            return self._apply_fast_selection(session_key, value, persist=persist_global)

        picker_sent = await self._try_send_choice_picker(
            event,
            session_key,
            title=t("gateway.fast.picker_title", mode=status),
            choices=[
                {"value": v, "label": t(f"gateway.fast.choice_{v}"), "is_current": mode == v}
                for v in ("fast", "normal", "auto", "cold")
            ],
            on_choice_selected=_on_fast_choice,
        )
        if picker_sent:
            return None  # Picker sent — adapter handles the response
        return t("gateway.fast.status", mode=status)
