"""Model picker, /model switch application, runtime snapshot/restore, and codex runtime handling for the interactive CLI

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal
symbols are imported LAZILY inside each method (``from cli import ...``) — the mixin
never imports ``cli`` at module load time (import cycle).

Tests drive the switch paths with bare stubs lacking most mixin methods, so shared steps are
module-level functions taking ``cli`` and siblings are called as ``HermesCLI.<name>(self, ...)``.
"""

from __future__ import annotations

import copy
import sys
import threading

from rich.markup import escape as _escape
from utils import base_url_host_matches

# CLI-level fields describing the active model route; snapshotted before a switch / one-turn
# override and restored wholesale on rollback.
_RUNTIME_FIELDS = (
    "model", "provider", "requested_provider", "_explicit_api_key", "_explicit_base_url",
    "api_key", "base_url", "api_mode")


def _runtime_fields(cli) -> dict:
    return {key: getattr(cli, key, None) for key in _RUNTIME_FIELDS}


def _heal_bare_custom_provider(provider, *, base_url, model):
    """Bare ``custom`` is a billing class, not a routable identity: persisting/restoring it makes a
    later resume hard-fail once the config default leaves the custom endpoint. Recover the durable
    ``custom:<name>`` menu key from the endpoint, else drop the provider (None)."""
    if str(provider or "").strip().lower() != "custom":
        return provider
    try:
        # Heal bare "custom" persisted by older builds / gateway turns: it's the resolved billing class, not
        # a routable identity. (Stricter than the TUI gateway's recovery, which keeps bare "custom" when a
        # base_url exists — the CLI's resolve path would hard-fail on it, #14676.)
        from hermes_cli.runtime_provider import canonical_custom_identity
        return canonical_custom_identity(base_url=base_url or None, model=model or None) or None
    except Exception:
        return None


def _merge_preflight_warning(cli, result, custom_providers) -> None:
    """Fold the context-compression preflight warning into ``result`` (fail-soft)."""
    from cli import logger
    if cli.agent is None:
        return
    try:
        from hermes_cli.context_switch_guard import merge_preflight_compression_warning
        # Prefer the fresh inventory list (same source as switch_model / TUI); fall back
        # to the agent-init snapshot.
        merge_preflight_compression_warning(
            result,
            agent=cli.agent,
            messages=list(cli.conversation_history or []),
            custom_providers=custom_providers if custom_providers is not None
            else getattr(cli.agent, "_custom_providers", None),
            config_context_length=getattr(cli.agent, "_config_context_length", None))
    except Exception as exc:
        logger.debug("preflight-compression switch warning failed: %s", exc)


def _print_switch_summary(cli, result, old_model, *, one_turn: bool, strict_context: bool) -> None:
    """Record the next-turn switch note and print the "Model switched" block.

    The note is prepended to the next user message (a mid-history system message would break
    providers and prompt caching). ``strict_context``: the typed /model path lets
    context-resolution errors propagate; the picker path swallows them.
    """
    from cli import _cprint
    from hermes_cli.model_switch import format_model_for_display, resolve_display_context_length
    _display_old = format_model_for_display(old_model)
    _display_new = format_model_for_display(result.new_model)
    cli._pending_model_switch_note = (
        f"[Note: model was just switched from {_display_old} to {_display_new} "
        f"via {result.provider_label or result.target_provider}. "
        f"{'This override applies to the next turn only. ' if one_turn else ''}"
        f"Adjust your self-identification accordingly.]")
    _cprint(f"  ✓ Model switched: {_display_new}")
    _cprint(f"    Provider: {result.provider_label or result.target_provider}")

    # Provider-aware context chain: Codex OAuth / Copilot / Nous caps win over the raw
    # models.dev entry (gpt-5.5 is 1.05M on openai but 272K on Codex OAuth).
    mi = result.model_info
    agent = cli.agent
    try:
        ctx = resolve_display_context_length(
            result.new_model, result.target_provider,
            base_url=result.base_url or cli.base_url or "",
            api_key=result.api_key or cli.api_key or "", model_info=mi,
            config_context_length=getattr(agent, "_config_context_length", None) if agent else None,
            custom_providers=getattr(agent, "_custom_providers", None) if agent else None)
    except Exception:
        if strict_context:
            raise
        ctx = None
    if ctx:
        _cprint(f"    Context: {ctx:,} tokens")
    if mi:
        if mi.max_output:
            _cprint(f"    Max output: {mi.max_output:,} tokens")
        _cprint(f"    Capabilities: {mi.format_capabilities()}")
    cache_enabled = (
        (base_url_host_matches(result.base_url or "", "openrouter.ai") and "claude" in result.new_model.lower())
        or result.api_mode == "anthropic_messages")
    if cache_enabled:
        _cprint("    Prompt caching: enabled")
    if result.warning_message:
        _cprint(f"    ⚠ {result.warning_message}")


def _switch_model_from(
    cli, raw_input, *, is_global, explicit_provider, user_providers, custom_providers):
    """``switch_model`` seeded with this CLI's live route."""
    from hermes_cli.model_switch import switch_model
    return switch_model(
        raw_input=raw_input, current_provider=cli.provider or "", current_model=cli.model or "",
        current_base_url=cli.base_url or "", current_api_key=cli.api_key or "", is_global=is_global,
        explicit_provider=explicit_provider, user_providers=user_providers,
        custom_providers=custom_providers)


def _run_confirm_and_apply(cli, target, *args) -> None:
    """Run a confirm+apply sequence off the UI thread when the TUI is live.

    The expensive-model modal blocks its thread on a response queue (_prompt_text_input_modal);
    on the prompt_toolkit main thread that freezes rendering, so the modal never appears and the
    switch silently cancels after the 120s timeout.
    """
    if getattr(cli, "_app", None):
        threading.Thread(target=target, args=args, daemon=True).start()
    else:
        target(*args)


def _commit_model_switch(
    cli, result, *, persist_global: bool, one_turn: bool = False, picker: bool = False) -> None:
    """Stage + swap, print the summary, persist (session row unless --once; config on --global).
    ``picker``: tolerate context-resolution errors and label the config write "(--global)"; the
    typed path additionally records the one-turn restore snapshot."""
    from cli import HermesCLI, _cprint
    old_model = cli.model
    snapshot = cli._snapshot_model_runtime() if one_turn else None
    if not cli._stage_and_swap_model(result, old_model):
        return
    if not picker:
        cli._pending_one_turn_model_restore = snapshot
    _print_switch_summary(cli, result, old_model, one_turn=one_turn, strict_context=not picker)
    if persist_global:
        _persist_global_switch(cli, result)
        _cprint("    Saved to config.yaml (--global)" if picker else "    Saved to config.yaml")
    elif one_turn:
        _cprint("    (next turn only — restores after one response)")
    else:
        _cprint("    (session only — add --global to persist)")
    # The row records what THIS session runs even on --global (else a later resume restores the
    # stale creation-time model); --once is restored after one turn and never touches the row.
    if not one_turn:
        HermesCLI._persist_model_switch_to_session(cli, result)


def _persist_global_switch(cli, result) -> None:
    """Write the switched route to config.yaml (--global). base_url/api_mode are freshly resolved
    for the target provider, so sync them every time (None clears a value the new provider doesn't
    need) — otherwise the OLD provider's endpoint/wire-protocol lingers in config.yaml."""
    from cli import HermesCLI, save_config_value
    HermesCLI._clear_persisted_context_for_model_switch(cli, result)
    save_config_value("model.default", result.new_model)
    save_config_value("model.provider", result.target_provider)
    # base_url/api_mode were previously never persisted here, so a global switch left the OLD provider's
    # endpoint/wire-protocol in config.yaml. result.base_url/api_mode are always freshly resolved for the
    # target provider (see model_switch.py), so sync them every time; None clears a value the new provider
    # doesn't need (#25106).
    # See _apply_model_switch_result above for why base_url/api_mode must be synced on every global switch
    # (#25106).
    save_config_value("model.base_url", result.base_url or None)
    save_config_value("model.api_mode", result.api_mode or None)


def _show_model_picker(cli, ctx, force_refresh: bool) -> None:
    """``/model`` with no args: open the picker, or print usage when nothing is authed."""
    from cli import _cprint
    from hermes_cli.inventory import build_models_payload
    from hermes_cli.providers import get_label
    try:
        if ctx is None:
            raise RuntimeError("inventory context unavailable")
        providers = build_models_payload(
            ctx, probe_custom_providers=force_refresh,
            probe_current_custom_provider=not force_refresh,
        )["providers"]
    except Exception:
        providers = []
    if not providers:
        _cprint("  No authenticated providers found.")
        _cprint("")
        _cprint("  /model <name>                        switch model (this session)")
        _cprint("  /model <name> --global               switch model and persist as default")
        _cprint("  /model <name> --once                 switch for the next turn only")
        _cprint("  /model <name> --session              switch for this session only")
        _cprint("  /model --provider <slug>             switch provider")
        _cprint("  /model --refresh                     re-fetch live model lists")
        return
    cli._open_model_picker(
        providers, cli.model or "unknown", get_label(cli.provider) if cli.provider else "unknown",
        user_provs=ctx.user_providers if ctx is not None else None,
        custom_provs=ctx.custom_providers if ctx is not None else None)


class CLIModelSwitchMixin:
    """Model picker, /model switch application, runtime snapshot/restore, and codex runtime handling for the interactive CLI"""

    def _normalize_model_for_provider(self, resolved_provider: str) -> bool:
        """Normalize provider-specific model IDs and routing."""
        from cli import _split_model_config_default
        current_model = str(self.model or "").strip()
        if isinstance(self.model, dict):
            current_model, _ = _split_model_config_default(self.model)
        changed = False

        def _adopt(canonical, notice) -> None:
            """Adopt ``canonical`` when it differs; ``notice(new)`` builds the warning text."""
            nonlocal current_model, changed
            if canonical and canonical != current_model:
                if not self._model_is_default:
                    self._console_print(f"[yellow]⚠️  {notice(canonical)}[/]")
                self.model = canonical
                current_model = canonical
                changed = True

        def _adopt_with_mode(normalize, api_mode_of, notice) -> bool:
            """Provider families that also own the wire protocol: adopt id, then sync api_mode."""
            nonlocal changed
            try:
                _adopt(normalize(current_model), notice)
                resolved_mode = api_mode_of(current_model)
                if resolved_mode != self.api_mode:
                    self.api_mode = resolved_mode
                    changed = True
            except Exception:
                pass
            return changed

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS, normalize_model_for_provider)
            if resolved_provider not in _AGGREGATOR_PROVIDERS:
                _adopt(
                    normalize_model_for_provider(current_model, resolved_provider),
                    lambda new: (
                        f"Normalized model '{current_model}' to '{new}' for {resolved_provider}."))
        except Exception:
            pass

        if resolved_provider == "copilot":
            from hermes_cli.models import copilot_model_api_mode, normalize_copilot_model_id
            return _adopt_with_mode(
                lambda m: normalize_copilot_model_id(m, api_key=self.api_key),
                lambda m: copilot_model_api_mode(m, api_key=self.api_key),
                lambda new: f"Normalized Copilot model '{current_model}' to '{new}'.")

        from hermes_cli.models import opencode_provider_family
        if opencode_provider_family(resolved_provider) is not None:
            from hermes_cli.models import normalize_opencode_model_id, opencode_model_api_mode
            return _adopt_with_mode(
                lambda m: normalize_opencode_model_id(resolved_provider, m),
                lambda m: opencode_model_api_mode(resolved_provider, m),
                lambda new: (
                    f"Stripped provider prefix from '{current_model}'; "
                    f"using '{new}' for {resolved_provider}."))

        if resolved_provider != "openai-codex":
            return changed

        # 1. Strip provider prefix ("openai/gpt-5.4" → "gpt-5.4")
        if "/" in current_model:
            slug = current_model.split("/", 1)[1]
            if not self._model_is_default:
                self._console_print(
                    f"[yellow]⚠️  Stripped provider prefix from '{current_model}'; "
                    f"using '{slug}' for OpenAI Codex.[/]")
            self.model = slug
            current_model = slug
            changed = True

        # 2. Replace untouched default with a Codex model
        if self._model_is_default:
            fallback_model = "gpt-5.3-codex"
            try:
                from hermes_cli.codex_models import get_codex_model_ids
                available = get_codex_model_ids(access_token=self.api_key if self.api_key else None)
                if available:
                    fallback_model = available[0]
            except Exception:
                pass
            if current_model != fallback_model:
                self.model = fallback_model
                changed = True
        return changed

    def _persist_model_switch_to_session(self, result) -> None:
        """Persist a session-scoped /model switch to the session DB row.

        Writes the model column plus the route in both shapes readers use — nested
        ``gateway_runtime`` (CLI --resume) and top-level keys (TUI session.resume) — from one
        or-None dict so stale keys are DELETED (``_merge_model_config_json`` only deletes on
        explicit None) and the shapes never diverge.

        Writes the model column plus the runtime route so ``--resume`` (CLI, reads ``gateway_runtime``) and
        ``session.resume`` (TUI/desktop, reads top-level ``model_config`` keys via
        ``_stored_session_runtime_overrides``) both restore the switched provider instead of recombining the
        model with the ambient default (#79536). Mirrors the gateway's ``update_session_model()`` call.
        getattr: tests drive the switch paths with ``object.__new__`` stubs.
        """
        from cli import logger
        db = getattr(self, "_session_db", None)
        sid = getattr(self, "session_id", None)
        if not db or not sid:
            return
        route = {
            "provider": _heal_bare_custom_provider(
                result.target_provider, base_url=result.base_url, model=result.new_model,
            ) or None,
            # Both shapes use the same or-None discipline so stale keys from a previous switch are deleted
            # (not merely omitted) in BOTH the nested gateway_runtime dict (CLI reader) and the top-level
            # keys (TUI gateway reader). _merge_model_config_json only deletes on explicit None, so falsy
            # values must be converted, not filtered. Deriving the top-level from **route guarantees the two
            # shapes can never diverge — the asymmetry that caused the original stale-key bug (#85261
            # simplify-code review).
            "base_url": result.base_url or None,
            "api_mode": result.api_mode or None}
        try:
            db.update_session_model(sid, result.new_model)
            db.patch_session_model_config(sid, {"gateway_runtime": route, **route})
        except Exception:
            logger.debug("Failed to persist model switch to session DB", exc_info=True)

    def _restore_session_model(self, session_meta: dict, *, quiet: bool = False) -> None:
        """Restore model/provider from the session DB row on every resume path.

        Skips when no model is recorded or the CLI got an explicit ``-m`` (user intent wins).
        A different stored provider gets its credentials re-resolved — the ambient ``api_key``
        must not be sent to the session's endpoint; on failure the ambient credentials are kept
        so the session still opens (the first turn surfaces the auth error).
        """
        from cli import logger
        stored_model = (session_meta or {}).get("model")
        if not stored_model or getattr(self, "_explicit_model_override", False):
            return
        # Canonical row reader: model_config.gateway_runtime, else the TUI's top-level keys.
        from hermes_state import SessionDB as _SessionDB
        _stored_runtime = _SessionDB.session_gateway_runtime(session_meta)
        stored_base_url = _stored_runtime.get("base_url") or None
        stored_api_mode = _stored_runtime.get("api_mode") or None
        # Stricter than the TUI gateway's recovery (which keeps bare "custom" when a
        # base_url exists) — the CLI's resolve path would hard-fail on it.
        stored_provider = _heal_bare_custom_provider(
            _stored_runtime.get("provider") or None, base_url=stored_base_url, model=stored_model)
        model_changed = stored_model != self.model
        provider_changed = bool(stored_provider) and stored_provider != self.provider
        if not model_changed and not provider_changed:
            return
        self.model = stored_model
        if stored_provider:
            self.provider = stored_provider
            self.requested_provider = stored_provider
            if stored_base_url:
                self.base_url = stored_base_url
            if stored_api_mode:
                self.api_mode = stored_api_mode
        if provider_changed:
            # Launch-time explicit overrides belong to the AMBIENT provider and would poison
            # _ensure_runtime_credentials for the restored one. api_key is never persisted to
            # the session DB — runtime provider resolution owns credentials.
            self._explicit_api_key = None
            self._explicit_base_url = stored_base_url
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                resolved = resolve_runtime_provider(requested=stored_provider)
                if resolved.get("api_key"):
                    self.api_key = resolved["api_key"]
                    self._credential_pool = resolved.get("credential_pool")
                if not stored_base_url and resolved.get("base_url"):
                    self.base_url = resolved["base_url"]
                if not stored_api_mode and resolved.get("api_mode"):
                    self.api_mode = resolved["api_mode"]
            except Exception:
                logger.debug(
                    "Credential re-resolution for resumed session provider "
                    "%s failed; keeping ambient credentials",
                    stored_provider, exc_info=True)
        # Mid-chat /resume swaps the live agent; on startup --resume _init_agent picks up
        # self.model / self.provider.
        if self.agent is not None:
            try:
                self.agent.switch_model(
                    new_model=self.model, new_provider=self.provider, api_key=self.api_key or "",
                    base_url=self.base_url or "", api_mode=self.api_mode or "")
            except Exception:
                logger.debug("In-place agent model swap on resume failed", exc_info=True)
        msg = f"Model restored from session: {stored_model}"
        if stored_provider:
            msg += f" ({stored_provider})"
        if quiet:
            print(msg, file=sys.stderr)
        else:
            self._console_print(f"[dim]{_escape(msg)}[/dim]")

    def _open_model_picker(self, providers: list, current_model: str, current_provider: str, user_provs=None, custom_provs=None) -> None:
        """Open prompt_toolkit-native /model picker modal."""
        self._capture_modal_input_snapshot()
        self._model_picker_state = {
            "stage": "provider",
            "providers": providers,
            "selected": next((i for i, p in enumerate(providers) if p.get("is_current")), 0),
            "current_model": current_model,
            "current_provider": current_provider,
            "user_provs": user_provs,
            "custom_provs": custom_provs,
            "filter": ""}
        self._invalidate(min_interval=0.0)

    def _confirm_expensive_model_switch(self, result) -> bool:
        """Ask for explicit confirmation before applying costly model switches."""
        if not getattr(result, "success", False):
            return True
        try:
            from hermes_cli.model_selection_guards import combined_selection_warning
            warning = combined_selection_warning(
                result.new_model, provider=result.target_provider,
                base_url=result.base_url or self.base_url or "",
                api_key=result.api_key or self.api_key or "", model_info=result.model_info)
        except Exception:
            warning = None
        if warning is None:
            return True
        choices = [
            ("once", "Switch anyway", "Use this model for the current Hermes session."),
            ("cancel", "Cancel", "Keep the current model.")]
        raw = self._prompt_text_input_modal(
            title=f"!!! {warning.title} !!!", detail=warning.message, choices=choices, timeout=120)
        return self._normalize_slash_confirm_choice(raw, choices) == "once"

    def _confirm_and_apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None) -> None:
        from cli import _cprint
        try:
            if result.success and not self._confirm_expensive_model_switch(result):
                _cprint("  Model switch cancelled.")
                return
            self._apply_model_switch_result(
                result, persist_global, custom_providers=custom_providers)
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
            **_runtime_fields(self),
            "agent_primary_runtime": copy.deepcopy(
                getattr(agent, "_primary_runtime", None)
            ) if agent is not None else None}

    def _restore_model_runtime_snapshot(self, snapshot: dict | None) -> None:
        """Restore a model runtime captured before a one-turn override."""
        from cli import logger
        if not snapshot:
            return
        for key in _RUNTIME_FIELDS:
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
                    new_model=snapshot.get("model", ""), new_provider=snapshot.get("provider", ""),
                    api_key=snapshot.get("api_key", ""), base_url=snapshot.get("base_url", ""),
                    api_mode=snapshot.get("api_mode", ""),
                    capabilities=snapshot.get("capabilities"))
            except Exception as exc:
                logger.warning("CLI one-turn model restore failed: %s", exc)

    @staticmethod
    def _filter_model_picker_entries(entries: list, query: str) -> list:
        """Return (original_index, label) pairs matching ``query`` (case-insensitive subsequence;
        empty matches all). The ORIGINAL index keeps a filtered selection resolving to exactly one
        concrete model — filtering never introduces fuzzy *resolution*."""
        pairs = list(enumerate(entries))
        q = (query or "").strip().lower()
        if not q:
            return pairs

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        return [(i, e) for (i, e) in pairs if _subseq(q, str(e).lower())]

    @staticmethod
    def _compute_model_picker_viewport(
        selected: int, scroll_offset: int, n: int, term_rows: int, reserved_below: int = 6,
        panel_chrome: int = 6, min_visible: int = 3) -> tuple[int, int]:
        """Resolve (scroll_offset, visible) for the /model picker viewport. ``reserved_below``
        matches the approval/clarify panels (input, status bar, separators); ``panel_chrome`` is
        borders + blanks + hint row. The offset slides to keep ``selected`` on screen."""
        max_visible = max(min_visible, term_rows - reserved_below - panel_chrome)
        if n <= max_visible:
            return 0, n
        visible = max_visible
        if selected < scroll_offset:
            scroll_offset = selected
        elif selected >= scroll_offset + visible:
            scroll_offset = selected - visible + 1
        return max(0, min(scroll_offset, n - visible)), visible

    def _clear_persisted_context_for_model_switch(self, result) -> None:
        """Drop a global context pin when its configured owner changes."""
        from cli import save_config_value
        try:
            from hermes_cli.config import load_config_readonly
            from hermes_cli.route_identity import should_clear_context_pin
            config = load_config_readonly()
            model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
            if not isinstance(model_cfg, dict) or "context_length" not in model_cfg:
                return
            if should_clear_context_pin(
                model_cfg.get("default") or model_cfg.get("model"), result.new_model,
                model_cfg.get("base_url"), result.base_url,
                model_cfg.get("provider"), result.target_provider):
                save_config_value("model.context_length", None)
        except Exception:
            save_config_value("model.context_length", None)

    def _stage_and_swap_model(self, result, old_model) -> bool:
        """Stage ``result`` onto the CLI fields, then swap the live agent in place.

        CLI fields are snapshotted first so a failed agent swap rolls the whole CLI back —
        otherwise the staged broken credentials leak into the next turn even though the agent
        rolled back. Returns False after printing the failure (a failed switch is a no-op).
        """
        from cli import _cprint
        _cli_snapshot = _runtime_fields(self)
        self.model = result.new_model
        self.provider = result.target_provider
        self.requested_provider = result.target_provider
        # Always overwrite explicit overrides so stale credentials from the previous provider
        # (e.g. Ollama api_key/base_url) don't leak into the next resolution.
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
                    new_model=result.new_model, new_provider=result.target_provider,
                    api_key=result.api_key, base_url=result.base_url, api_mode=result.api_mode,
                    capabilities=getattr(result, "runtime_capabilities", None))
            except Exception as exc:
                # The agent rolled itself back to the old working model/client. Roll the CLI's own staged
                # fields back too and abort the rest of the commit (note + success print) so a failed switch
                # is a no-op rather than a dead session (#50163).
                # Agent rolled itself back; roll the CLI back too and abort so a failed switch is a no-op
                # rather than a dead session (#50163).
                for _k, _v in _cli_snapshot.items():
                    setattr(self, _k, _v)
                _cprint(
                    f"  ⚠ Model switch to {result.new_model} failed ({exc}); "
                    f"staying on {old_model}.")
                return False
        return True

    def _apply_model_switch_result(
        self, result, persist_global: bool, custom_providers=None) -> None:
        """Picker-path commit (see _commit_model_switch)."""
        from cli import _cprint
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return
        _merge_preflight_warning(self, result, custom_providers)
        _commit_model_switch(self, result, persist_global=persist_global, picker=True)

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
            # Curated list (same as `hermes model` / gateway pickers); live catalog only when
            # it is empty (user-defined endpoints).
            model_list = provider_data.get("models", [])
            if not model_list:
                try:
                    from hermes_cli.models import provider_model_ids
                    model_list = provider_model_ids(provider_data["slug"]) or model_list
                except Exception:
                    pass
            state.update(
                stage="model", provider_data=provider_data, model_list=model_list,
                selected=0, filter="", _filtered_pairs=None)
            self._invalidate(min_interval=0.0)
            return
        if stage == "model":
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            # Map the row through the active fuzzy filter; pairs carry the ORIGINAL index.
            filtered_pairs = state.get("_filtered_pairs")
            if filtered_pairs is None:
                filtered_pairs = list(enumerate(model_list))
            visible_labels = [e for (_i, e) in filtered_pairs]
            back_idx = len(visible_labels)
            if selected == back_idx:
                state.update(
                    stage="provider", filter="", _filtered_pairs=None,
                    selected=next((i for i, p in enumerate(state.get("providers") or [])
                                   if p.get("slug") == provider_data.get("slug")), 0))
                self._invalidate(min_interval=0.0)
                return
            if selected > back_idx:  # cancel row (and anything past it)
                self._close_model_picker()
                return
            if 0 <= selected < back_idx:
                result = _switch_model_from(
                    self, visible_labels[selected], is_global=persist_global,
                    explicit_provider=provider_data.get("slug"),
                    user_providers=state.get("user_provs"),
                    custom_providers=state.get("custom_provs"))
                # Capture before close — picker state is cleared on close.
                _picker_custom_provs = state.get("custom_provs")
                self._close_model_picker()
                _run_confirm_and_apply(
                    self, self._confirm_and_apply_model_switch_result,
                    result, persist_global, _picker_custom_provs)
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

        Switches are session-scoped unless ``model.persist_switch_by_default`` or ``--global``.
        """
        from cli import _cprint
        from hermes_cli.model_switch import parse_model_switch_args, resolve_persist_behavior

        parts = cmd_original.split(None, 1)  # split off '/model'
        request = parse_model_switch_args(parts[1].strip() if len(parts) > 1 else "")
        if request.errors:
            # CLI decoration: "  ✗ " prefix over the canonical error copy.
            _cprint(f"  ✗ {request.error_messages()[0]}")
            return
        one_turn = request.is_once
        persist_global = resolve_persist_behavior(
            request.is_global, request.is_session, is_once=one_turn,
            explicit_provider=request.explicit_provider)

        # --refresh: wipe the picker cache so every authed provider's /v1/models is re-fetched.
        if request.force_refresh:
            try:
                from hermes_cli.models import clear_provider_models_cache
                clear_provider_models_cache()
                _cprint("  Cleared model picker cache. Refreshing...")
            except Exception:
                pass

        # Live session state is overlaid truthy-only so empty self.* attrs don't clobber config.
        from hermes_cli.inventory import load_picker_context
        try:
            ctx = load_picker_context().with_overrides(
                current_provider=self.provider or "", current_model=self.model or "",
                current_base_url=self.base_url or "")
        except Exception:
            ctx = None
        # switch_model() + _open_model_picker still need the raw provider dicts.
        user_provs = ctx.user_providers if ctx is not None else None
        custom_provs = ctx.custom_providers if ctx is not None else None

        if not request.target and not request.explicit_provider:
            return _show_model_picker(self, ctx, request.force_refresh)

        result = _switch_model_from(
            self, request.target, is_global=persist_global,
            explicit_provider=request.explicit_provider,
            user_providers=user_provs, custom_providers=custom_provs)
        if not result.success:
            _cprint(f"  ✗ {result.error_message}")
            return
        _merge_preflight_warning(self, result, custom_provs)
        _run_confirm_and_apply(
            self, self._confirm_and_apply_cli_model_switch,
            result, persist_global, one_turn, custom_provs)

    def _confirm_and_apply_cli_model_switch(
        self, result, persist_global: bool, one_turn: bool, custom_provs=None) -> None:
        """Confirm an expensive model switch and apply it (typed /model path). Runs on a worker
        thread when the TUI is active (see _run_confirm_and_apply) so the modal can render."""
        from cli import _cprint
        if not self._confirm_expensive_model_switch(result):
            _cprint("  Model switch cancelled.")
            return
        _commit_model_switch(self, result, persist_global=persist_global, one_turn=one_turn)

    def _handle_codex_runtime(self, cmd_original: str) -> None:
        """Handle /codex-runtime — toggle the codex app-server runtime opt-in.

        Usage:
            /codex-runtime                       — show current state
            /codex-runtime auto                  — Hermes default (chat_completions)
            /codex-runtime codex_app_server      — hand turns to codex subprocess
            /codex-runtime on / off              — synonyms for the above
        """
        from cli import _cprint
        from hermes_cli import codex_runtime_switch as crs

        parts = cmd_original.split(None, 1)
        new_value, errors = crs.parse_args(parts[1].strip() if len(parts) > 1 else "")
        if errors:
            for err in errors:
                _cprint(f"❌ {err}")
            return
        try:
            from hermes_cli.config import load_config, save_config
        except Exception as exc:
            _cprint(f"❌ could not load config: {exc}")
            return
        result = crs.apply(
            load_config(), new_value,
            persist_callback=(save_config if new_value is not None else None))
        prefix = "✓" if result.success else "✗"
        for line in result.message.splitlines():
            _cprint(f"  {prefix} {line}" if line.startswith("openai_runtime") else f"    {line}")
        if result.success and result.requires_new_session:
            _cprint("    Tip: `/reset` starts a new session immediately.")

    def _should_handle_model_command_inline(self, text: str, has_images: bool = False) -> bool:
        """Return True when /model should be handled immediately on the UI thread."""
        from cli import _looks_like_slash_command
        if not text or has_images or not _looks_like_slash_command(text):
            return False
        try:
            from hermes_cli.commands import resolve_command
            cmd = resolve_command(text.split(None, 1)[0].lower().lstrip('/'))
            return bool(cmd and cmd.name == "model")
        except Exception:
            return False

    def _cmd_moa(self, cmd_original: str):
        """/moa one-shot: run one prompt through the default MoA preset, then restore the prior
        model (a session-long MoA switch goes through the picker's virtual MoA provider)."""
        from cli import _cprint, _slash_args
        from hermes_cli.moa_config import moa_usage, normalize_moa_config

        payload = _slash_args(cmd_original)
        if not payload:
            _cprint(f"  {moa_usage()}")
            return True
        moa_cfg = self.config.get("moa") if isinstance(self.config, dict) else {}
        preset = normalize_moa_config(moa_cfg)["default_preset"]
        self._pending_moa_restore_model = {
            key: getattr(self, key, None)
            for key in (
                "requested_provider", "provider", "model", "api_key", "base_url", "api_mode")}
        self.requested_provider = "moa"
        self.provider = "moa"
        self.model = preset
        self.api_key = "moa-virtual-provider"
        self.base_url = "moa://local"
        self.api_mode = "chat_completions"
        self.agent = None
        self._pending_moa_disable_after_turn = True
        self._pending_agent_seed = payload
        _cprint(f"  MoA one-shot queued with preset {preset}; previous model will be restored after this turn.")
