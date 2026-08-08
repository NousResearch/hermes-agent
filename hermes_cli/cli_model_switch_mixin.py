"""Model-switch subsystem for the interactive CLI (god-file decomposition).

This module hosts the /model picker, expensive-switch guard, one-turn runtime
snapshot/restore, and switch-application methods lifted out of ``cli.py``'s
``HermesCLI`` class. ``HermesCLI`` inherits ``CLIModelSwitchMixin`` so every
``self.<handler>`` call resolves unchanged via the MRO — behavior-neutral.

Import discipline mirrors the other ``hermes_cli.cli_*_mixin`` modules:
  * Only stdlib, cycle-free dependencies are imported at module top level below.
  * cli.py-internal symbols (``_cprint``, ``save_config_value``,
    ``base_url_host_matches``) and third-party/agent imports are resolved
    LAZILY inside each method via ``from cli import ...`` / ``from ... import``
    — resolved at call time when ``cli`` is fully loaded, so this module never
    imports ``cli`` at top level (no cycle).
"""

from __future__ import annotations

import copy
import threading


class CLIModelSwitchMixin:
    """Mixin holding interactive-CLI model-switch handlers."""

    def _open_model_picker(self, providers: list, current_model: str, current_provider: str, user_provs=None, custom_provs=None) -> None:
        """Open prompt_toolkit-native /model picker modal."""
        self._capture_modal_input_snapshot()
        default_idx = next((i for i, p in enumerate(providers) if p.get("is_current")), 0)
        self._model_picker_state = {
            "stage": "provider",
            "providers": providers,
            "selected": default_idx,
            "current_model": current_model,
            "current_provider": current_provider,
            "user_provs": user_provs,
            "custom_provs": custom_provs,
        }
        self._invalidate(min_interval=0.0)

    def _confirm_expensive_model_switch(self, result) -> bool:
        """Ask for explicit confirmation before applying costly model switches."""
        if not getattr(result, "success", False):
            return True
        try:
            from hermes_cli.model_cost_guard import expensive_model_warning

            warning = expensive_model_warning(
                result.new_model,
                provider=result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "",
                model_info=result.model_info,
            )
        except Exception:
            warning = None
        if warning is None:
            return True

        choices = [
            ("once", "Switch anyway", "Use this model for the current Hermes session."),
            ("cancel", "Cancel", "Keep the current model."),
        ]
        raw = self._prompt_text_input_modal(
            title="!!! Expensive Model Warning !!!",
            detail=warning.message,
            choices=choices,
            timeout=120,
        )
        choice = self._normalize_slash_confirm_choice(raw, choices)
        return choice == "once"

    def _confirm_and_apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None
    ) -> None:
        try:
            if result.success and not self._confirm_expensive_model_switch(result):
                _cprint("  Model switch cancelled.")
                return
            self._apply_model_switch_result(
                result, persist_global, custom_providers=custom_providers
            )
        except Exception as exc:
            _cprint(f"  ✗ Model selection failed: {exc}")

    def _close_model_picker(self) -> None:
        self._model_picker_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _snapshot_model_runtime(self) -> dict:
        """Capture current CLI and agent model runtime for one-turn restore."""
        agent = getattr(self, "agent", None)
        return {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "agent_primary_runtime": copy.deepcopy(
                getattr(agent, "_primary_runtime", None)
            ) if agent is not None else None,
        }

    def _restore_model_runtime_snapshot(self, snapshot: dict | None) -> None:
        """Restore a model runtime captured before a one-turn override."""
        if not snapshot:
            return
        for key in (
            "model",
            "provider",
            "requested_provider",
            "_explicit_api_key",
            "_explicit_base_url",
            "api_key",
            "base_url",
            "api_mode",
        ):
            if key in snapshot:
                setattr(self, key, snapshot.get(key))

        agent = getattr(self, "agent", None)
        if agent is None:
            return

        primary = snapshot.get("agent_primary_runtime")
        if primary and hasattr(agent, "_restore_primary_runtime"):
            try:
                agent._primary_runtime = copy.deepcopy(primary)
                agent._fallback_activated = True
                agent._rate_limited_until = 0
                if agent._restore_primary_runtime():
                    return
            except Exception:
                logger.debug("CLI one-turn model restore via primary runtime failed", exc_info=True)

        if hasattr(agent, "switch_model"):
            try:
                agent.switch_model(
                    new_model=snapshot.get("model", ""),
                    new_provider=snapshot.get("provider", ""),
                    api_key=snapshot.get("api_key", ""),
                    base_url=snapshot.get("base_url", ""),
                    api_mode=snapshot.get("api_mode", ""),
                )
            except Exception as exc:
                logger.warning("CLI one-turn model restore failed: %s", exc)

    @staticmethod
    def _compute_model_picker_viewport(
        selected: int,
        scroll_offset: int,
        n: int,
        term_rows: int,
        reserved_below: int = 6,
        panel_chrome: int = 6,
        min_visible: int = 3,
    ) -> tuple[int, int]:
        """Resolve (scroll_offset, visible) for the /model picker viewport.

        ``reserved_below`` matches the approval / clarify panels — input area,
        status bar, and separators below the panel. ``panel_chrome`` covers
        this panel's own borders + blanks + hint row. The remaining rows hold
        the scrollable list, with the offset slid to keep ``selected`` on screen.
        """
        max_visible = max(min_visible, term_rows - reserved_below - panel_chrome)
        if n <= max_visible:
            return 0, n
        visible = max_visible
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + visible:
            scroll_offset = selected - visible + 1
        scroll_offset = max(0, min(scroll_offset, n - visible))
        return scroll_offset, visible

    def _clear_persisted_context_for_model_switch(self, result) -> None:
        """Drop a global context pin when its configured owner changes."""
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.route_identity import should_clear_context_pin

            config = load_config_readonly()
            model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
            if not isinstance(model_cfg, dict) or "context_length" not in model_cfg:
                return
            if should_clear_context_pin(
                model_cfg.get("default") or model_cfg.get("model"),
                result.new_model,
                model_cfg.get("base_url"),
                result.base_url,
                model_cfg.get("provider"),
                result.target_provider,
            ):
                save_config_value("model.context_length", None)
        except Exception:
            save_config_value("model.context_length", None)

    def _apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None
    ) -> None:
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        if self.agent is not None:
            try:
                from hermes_cli.context_switch_guard import merge_preflight_compression_warning

                # Prefer the fresh inventory list (same source as switch_model /
                # TUI); fall back to the agent-init snapshot.
                _cp = (
                    custom_providers
                    if custom_providers is not None
                    else getattr(self.agent, "_custom_providers", None)
                )
                merge_preflight_compression_warning(
                    result,
                    agent=self.agent,
                    messages=list(self.conversation_history or []),
                    custom_providers=_cp,
                    config_context_length=getattr(self.agent, "_config_context_length", None),
                )
            except Exception as exc:
                logger.debug("preflight-compression switch warning failed: %s", exc)

        old_model = self.model
        # Snapshot the CLI-level credential/runtime fields BEFORE mutating them
        # so a failed in-place agent swap can roll the whole CLI back to the old
        # working model.  Otherwise the broken credentials staged below leak into
        # the next turn's resolution even though the agent itself rolled back
        # (#50163).
        _cli_snapshot = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
        }
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the
        # previous provider (e.g. Ollama api_key/base_url) don't leak into
        # the new provider's credential resolution on the next turn.
        self._explicit_api_key = result.api_key
        self._explicit_base_url = result.base_url
        if result.api_key:
            self.api_key = result.api_key
        if result.base_url:
            self.base_url = result.base_url
        if result.api_mode:
            self.api_mode = result.api_mode

        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
                )
            except Exception as exc:
                # The agent rolled itself back to the old working model/client.
                # Roll the CLI's own staged fields back too and abort the rest
                # of the commit (note + success print) so a failed switch is a
                # no-op rather than a dead session (#50163).
                for _k, _v in _cli_snapshot.items():
                    setattr(self, _k, _v)
                _cprint(
                    f"  ⚠ Model switch to {result.new_model} failed ({exc}); "
                    f"staying on {old_model}."
                )
                return

        from hermes_cli.model_switch import format_model_for_display
        _display_old = format_model_for_display(old_model)
        _display_new = format_model_for_display(result.new_model)

        self._pending_model_switch_note = (
            f"[Note: model was just switched from {_display_old} to {_display_new} "
            f"via {result.provider_label or result.target_provider}. "
            f"Adjust your self-identification accordingly.]"
        )

        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {_display_new}")
        _cprint(f"    Provider: {provider_label}")

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry
        # (e.g. gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
        mi = result.model_info
        try:
            from hermes_cli.model_switch import resolve_display_context_length
            ctx = resolve_display_context_length(
                result.new_model,
                result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "",
                model_info=mi,
                config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None,
                custom_providers=getattr(self.agent, "_custom_providers", None) if self.agent else None,
            )
            if ctx:
                _cprint(f"    Context: {ctx:,} tokens")
        except Exception:
            pass
        if mi:
            if mi.max_output:
                _cprint(f"    Max output: {mi.max_output:,} tokens")
            _cprint(f"    Capabilities: {mi.format_capabilities()}")

        cache_enabled = (
            (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            _cprint("    Prompt caching: enabled")
        if result.warning_message:
            _cprint(f"    ⚠ {result.warning_message}")
        if persist_global:
            HermesCLI._clear_persisted_context_for_model_switch(self, result)
            save_config_value("model.default", result.new_model)
            save_config_value("model.provider", result.target_provider)
            # base_url/api_mode were previously never persisted here, so a
            # global switch left the OLD provider's endpoint/wire-protocol in
            # config.yaml. result.base_url/api_mode are always freshly
            # resolved for the target provider (see model_switch.py), so sync
            # them every time; None clears a value the new provider doesn't
            # need (#25106).
            save_config_value("model.base_url", result.base_url or None)
            save_config_value("model.api_mode", result.api_mode or None)
            _cprint("    Saved to config.yaml (--global)")
        else:
            _cprint("    (session only — add --global to persist)")

    def _handle_model_picker_selection(self, persist_global: bool = False) -> None:
        state = self._model_picker_state
        if not state:
            return
        selected = state.get("selected", 0)
        stage = state.get("stage")
        if stage == "provider":
            providers = state.get("providers") or []
            if selected >= len(providers):
                self._close_model_picker()
                return
            provider_data = providers[selected]
            # Use the curated model list from list_authenticated_providers()
            # (same lists as `hermes model` and gateway pickers).
            # Only fall back to the live provider catalog when the curated
            # list is empty (e.g. user-defined endpoints with no curated list).
            model_list = provider_data.get("models", [])
            if not model_list:
                try:
                    from hermes_cli.models import provider_model_ids
                    live = provider_model_ids(provider_data["slug"])
                    if live:
                        model_list = live
                except Exception:
                    pass
            state["stage"] = "model"
            state["provider_data"] = provider_data
            state["model_list"] = model_list
            state["selected"] = 0
            self._invalidate(min_interval=0.0)
            return
        if stage == "model":
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            back_idx = len(model_list)
            cancel_idx = len(model_list) + 1
            if selected == back_idx:
                state["stage"] = "provider"
                state["selected"] = next((i for i, p in enumerate(state.get("providers") or []) if p.get("slug") == provider_data.get("slug")), 0)
                self._invalidate(min_interval=0.0)
                return
            if selected >= cancel_idx:
                self._close_model_picker()
                return
            if selected < len(model_list):
                from hermes_cli.model_switch import switch_model
                chosen_model = model_list[selected]
                result = switch_model(
                    raw_input=chosen_model,
                    current_provider=self.provider or "",
                    current_model=self.model or "",
                    current_base_url=self.base_url or "",
                    current_api_key=self.api_key or "",
                    is_global=persist_global,
                    explicit_provider=provider_data.get("slug"),
                    user_providers=state.get("user_provs"),
                    custom_providers=state.get("custom_provs"),
                )
                # Capture before close — picker state is cleared on close.
                _picker_custom_provs = state.get("custom_provs")
                self._close_model_picker()
                if getattr(self, "_app", None):
                    threading.Thread(
                        target=self._confirm_and_apply_model_switch_result,
                        args=(result, persist_global, _picker_custom_provs),
                        daemon=True,
                    ).start()
                else:
                    self._confirm_and_apply_model_switch_result(
                        result, persist_global, custom_providers=_picker_custom_provs
                    )
                return
            self._close_model_picker()

    def _handle_model_switch(self, cmd_original: str):
        """Handle /model command — switch model.

        Supports:
          /model                              — show current model + usage hints
          /model <name>                       — switch model (this session only)
          /model <name> --once                — switch for the next turn only
          /model <name> --session             — switch for this session only (explicit)
          /model <name> --global              — switch and persist to config.yaml
          /model <name> --provider <provider> — switch provider + model
          /model --provider <provider>        — switch to provider, auto-detect model

        Persistence defaults to off (``model.persist_switch_by_default`` in
        config.yaml, default False — switches are session-scoped). Use
        ``--global`` to persist, or ``--once`` for the next turn only.
        """
        from hermes_cli.model_switch import (
            switch_model,
            parse_model_switch_args,
            resolve_persist_behavior,
        )
        from hermes_cli.providers import get_label

        # Parse args from the original command
        parts = cmd_original.split(None, 1)  # split off '/model'
        raw_args = parts[1].strip() if len(parts) > 1 else ""

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
            # CLI decoration: "  ✗ " prefix over the canonical error copy.
            _cprint(f"  ✗ {request.error_messages()[0]}")
            return
        # Resolve the effective persistence once: --global forces persist,
        # --session/--once force session-scope, otherwise defer to
        # model.persist_switch_by_default (defaults to False so /model is
        # session-scoped unless the user opts in).
        persist_global = resolve_persist_behavior(
            is_global_flag, is_session, is_once=one_turn,
            explicit_provider=explicit_provider,
        )

        # --refresh: wipe the on-disk picker cache before building the
        # provider list. Forces a live re-fetch of every authed provider's
        # /v1/models endpoint on this open.
        if force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
                _cprint("  Cleared model picker cache. Refreshing...")
            except Exception:
                pass

        # Single inventory context — replaces the inline config-slice the
        # dashboard / TUI used to duplicate. Overlay live session state
        # via with_overrides (truthy-only) so empty self.* attrs don't
        # clobber disk config.
        from hermes_cli.inventory import build_models_payload, load_picker_context

        try:
            ctx = load_picker_context().with_overrides(
                current_provider=self.provider or "",
                current_model=self.model or "",
                current_base_url=self.base_url or "",
            )
        except Exception:
            ctx = None

        # switch_model() + _open_model_picker still need the raw provider
        # dicts; ConfigContext is the canonical source for both.
        user_provs = ctx.user_providers if ctx is not None else None
        custom_provs = ctx.custom_providers if ctx is not None else None

        # No args at all: open prompt_toolkit-native picker modal
        if not model_input and not explicit_provider:
            model_display = self.model or "unknown"
            provider_display = get_label(self.provider) if self.provider else "unknown"

            try:
                if ctx is None:
                    raise RuntimeError("inventory context unavailable")
                providers = build_models_payload(
                    ctx,
                    probe_custom_providers=force_refresh,
                    probe_current_custom_provider=not force_refresh,
                )["providers"]
            except Exception:
                providers = []

            if not providers:
                _cprint("  No authenticated providers found.")
                _cprint("")
                _cprint("  /model <name>                        switch model (persists)")
                _cprint("  /model <name> --once                 switch for the next turn only")
                _cprint("  /model <name> --session              switch for this session only")
                _cprint("  /model --provider <slug>             switch provider")
                _cprint("  /model --refresh                     re-fetch live model lists")
                return

            self._open_model_picker(
                providers,
                model_display,
                provider_display,
                user_provs=user_provs,
                custom_provs=custom_provs,
            )
            return

        # Perform the switch
        result = switch_model(
            raw_input=model_input,
            current_provider=self.provider or "",
            current_model=self.model or "",
            current_base_url=self.base_url or "",
            current_api_key=self.api_key or "",
            is_global=persist_global,
            explicit_provider=explicit_provider,
            user_providers=user_provs,
            custom_providers=custom_provs,
        )

        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return

        if self.agent is not None:
            try:
                from hermes_cli.context_switch_guard import merge_preflight_compression_warning

                merge_preflight_compression_warning(
                    result,
                    agent=self.agent,
                    messages=list(self.conversation_history or []),
                    # Same fresh inventory list passed to switch_model above.
                    custom_providers=custom_provs
                    if custom_provs is not None
                    else getattr(self.agent, "_custom_providers", None),
                    config_context_length=getattr(self.agent, "_config_context_length", None),
                )
            except Exception as exc:
                logger.debug("preflight-compression switch warning failed: %s", exc)

        if not self._confirm_expensive_model_switch(result):
            _cprint("  Model switch cancelled.")
            return

        # Apply to CLI state.
        # Update requested_provider so _ensure_runtime_credentials() doesn't
        # overwrite the switch on the next turn (it re-resolves from this).
        old_model = self.model
        _one_turn_restore_snapshot = self._snapshot_model_runtime() if one_turn else None
        # Snapshot CLI-level fields before mutation so a failed in-place swap
        # rolls the whole CLI back to the old working model (#50163).
        _cli_snapshot = {
            "model": self.model,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "_explicit_api_key": getattr(self, "_explicit_api_key", None),
            "_explicit_base_url": getattr(self, "_explicit_base_url", None),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
        }
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the
        # previous provider (e.g. Ollama api_key/base_url) don't leak into
        # the new provider's credential resolution on the next turn.
        self._explicit_api_key = result.api_key
        self._explicit_base_url = result.base_url
        if result.api_key:
            self.api_key = result.api_key
        if result.base_url:
            self.base_url = result.base_url
        if result.api_mode:
            self.api_mode = result.api_mode

        # Apply to running agent (in-place swap)
        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=result.new_model,
                    new_provider=result.target_provider,
                    api_key=result.api_key,
                    base_url=result.base_url,
                    api_mode=result.api_mode,
                )
            except Exception as exc:
                # Agent rolled itself back; roll the CLI back too and abort so a
                # failed switch is a no-op rather than a dead session (#50163).
                for _k, _v in _cli_snapshot.items():
                    setattr(self, _k, _v)
                _cprint(
                    f"  ⚠ Model switch to {result.new_model} failed ({exc}); "
                    f"staying on {old_model}."
                )
                return

        # Store a note to prepend to the next user message so the model
        # knows a switch occurred (avoids injecting system messages mid-history
        # which breaks providers and prompt caching).
        from hermes_cli.model_switch import format_model_for_display
        _display_old = format_model_for_display(old_model)
        _display_new = format_model_for_display(result.new_model)

        self._pending_model_switch_note = (
            f"[Note: model was just switched from {_display_old} to {_display_new} "
            f"via {result.provider_label or result.target_provider}. "
            f"{'This override applies to the next turn only. ' if one_turn else ''}"
            f"Adjust your self-identification accordingly.]"
        )
        if one_turn:
            self._pending_one_turn_model_restore = _one_turn_restore_snapshot
        else:
            self._pending_one_turn_model_restore = None

        # Display confirmation with full metadata
        provider_label = result.provider_label or result.target_provider
        _cprint(f"  ✓ Model switched: {_display_new}")
        _cprint(f"    Provider: {provider_label}")

        # Context: always resolve via the provider-aware chain so Codex OAuth,
        # Copilot, and Nous-enforced caps win over the raw models.dev entry
        # (e.g. gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
        mi = result.model_info
        from hermes_cli.model_switch import resolve_display_context_length
        ctx = resolve_display_context_length(
            result.new_model,
            result.target_provider,
            base_url=result.base_url or self.base_url or "",
            api_key=result.api_key or self.api_key or "",
            model_info=mi,
            config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None,
            custom_providers=getattr(self.agent, "_custom_providers", None) if self.agent else None,
        )
        if ctx:
            _cprint(f"    Context: {ctx:,} tokens")
        if mi:
            if mi.max_output:
                _cprint(f"    Max output: {mi.max_output:,} tokens")
            _cprint(f"    Capabilities: {mi.format_capabilities()}")

        # Cache notice
        cache_enabled = (
            (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
            or result.api_mode == "anthropic_messages"
        )
        if cache_enabled:
            _cprint("    Prompt caching: enabled")

        # Warning from validation
        if result.warning_message:
            _cprint(f"    ⚠ {result.warning_message}")

        # Persistence
        if persist_global:
            HermesCLI._clear_persisted_context_for_model_switch(self, result)
            save_config_value("model.default", result.new_model)
            save_config_value("model.provider", result.target_provider)
            # See _apply_model_switch_result above for why base_url/api_mode
            # must be synced on every global switch (#25106).
            save_config_value("model.base_url", result.base_url or None)
            save_config_value("model.api_mode", result.api_mode or None)
            _cprint("    Saved to config.yaml")
        elif one_turn:
            _cprint("    (next turn only — restores after one response)")
        else:
            _cprint("    (session only — add --global to persist)")



def __getattr__(name):
    """PEP 562 lazy bridge for cli.py-internal symbols.

    The moved methods reference ``_cprint``, ``save_config_value``,
    ``base_url_host_matches``, ``logger``, and the ``HermesCLI`` class by
    bare name (verbatim bodies).  Resolving them lazily at first attribute
    access keeps this module free of a top-level ``import cli`` (which would
    cycle) while preserving byte-identical method bodies.
    """
    if name in ("_cprint", "save_config_value", "base_url_host_matches", "logger", "HermesCLI"):
        from cli import (  # noqa: F401
            HermesCLI,
            _cprint,
            base_url_host_matches,
            logger,
            save_config_value,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
