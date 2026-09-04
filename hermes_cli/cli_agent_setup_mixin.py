"""Agent construction + session-resume display for ``HermesCLI``: credential resolution,
per-turn agent config, first-use build, resume preload + recap. ``cli.py`` helpers are
imported lazily inside each method (import cycle)."""

from __future__ import annotations

import sys

from rich.markup import escape as _escape

from utils import base_url_host_matches


def _single_query_clarify_callback(question: str, choices=None, multi_select=False) -> str:
    """Headless clarify answer for ``hermes chat -q``.

    A -q turn never builds the prompt_toolkit app, so the interactive clarify modal
    can never be painted or answered — the CLI callback would poll until
    ``agent.clarify_timeout`` while the caller sees a silent hang. Mirror the oneshot
    path and answer immediately instead.

    The oneshot path answers immediately via ``_oneshot_clarify_callback``; single-query turns need the same
    headless behavior (#94943).
    """
    prefix = f"[single-query mode: no user available to answer {question!r}. "
    if choices:
        what = "subset" if multi_select else "option"
        return f"{prefix}Pick the best {what} from {choices} using your own judgment and continue.]"
    return f"{prefix}Make the most reasonable assumption you can and continue.]"


def _current_runtime(cli) -> dict:
    """Snapshot the CLI's resolved provider routing as an AIAgent runtime dict.

    getattr guards stay: tests build minimal shells lacking these attributes."""
    return {
        "api_key": cli.api_key,
        "base_url": cli.base_url,
        "provider": cli.provider,
        "requested_provider": getattr(cli, "requested_provider", cli.provider),
        "api_mode": cli.api_mode,
        "command": cli.acp_command,
        "args": list(cli.acp_args or []),
        "credential_pool": getattr(cli, "_credential_pool", None)}


def _route_signature(model, runtime: dict) -> tuple:
    """Hashable identity of (model, routing) used to detect when the agent must be rebuilt."""
    return (
        model, runtime.get("provider"), runtime.get("requested_provider"), runtime.get("base_url"),
        runtime.get("api_mode"), runtime.get("command"), tuple(runtime.get("args") or ()))


def _keyless_custom_base(base_url) -> bool:
    """Custom/local endpoints (llama.cpp, ollama, vLLM) often need no auth; only a
    non-OpenRouter base_url qualifies."""
    return bool(
        isinstance(base_url, str)
        and base_url
        and not base_url_host_matches(base_url, "openrouter.ai"))


def _compression_descendant(session_db, session_id):
    """If ``session_id`` is the (empty) head of a compression chain, return the
    descendant that actually holds the messages; else None. Fails open on DB errors."""
    try:
        resolved_id = session_db.resolve_resume_session_id(session_id)
    except Exception:
        return None
    return resolved_id if resolved_id and resolved_id != session_id else None


def _user_display_text(content) -> str:
    """Recap text for a user row; multimodal lists become text parts + ``[image]`` markers."""
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if part.get("type") == "text" else "[image]"
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "image_url"))
    return "" if content is None else str(content)


def _tool_calls_summary(tool_calls) -> str:
    """``[N tool call(s): name, ...]`` with up to 4 distinct names."""
    names = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
        if name not in names:
            names.append(name)
    names_str = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
    noun = "call" if len(tool_calls) == 1 else "calls"
    return f"[{len(tool_calls)} tool {noun}: {names_str}]"


# display_kind -> recap event line; ``hidden`` rows are skipped before this lookup.
_RESUME_EVENT_TEXT = {
    "model_switch": "model changed",
    "async_delegation_complete": "background delegation completed",
    "auto_continue": "resumed interrupted turn"}

def _collect_resume_entries(display_history, disp: dict, clean_assistant):
    """Displayable ``(role, text)`` recap entries from stored history, truncated per the
    ``display.resume_*`` config; system and tool-result rows are skipped. Returns
    ``(entries, index of last assistant entry, its un-truncated text)``.

    Stored history is untrusted for display: text is sanitized so replay can't clear the
    screen, retitle the window or restyle the panel. Pure-reasoning assistant rows with no
    visible output are skipped, as are tool-call-only rows when ``resume_skip_tool_only``.
    """
    from tools.ansi_strip import sanitize_display_text as _sanitize_display_text
    max_user_len = int(disp.get("resume_max_user_chars", 300))
    max_asst_len = int(disp.get("resume_max_assistant_chars", 200))
    max_asst_lines = int(disp.get("resume_max_assistant_lines", 3))
    skip_tool_only = disp.get("resume_skip_tool_only", True)
    entries: list = []
    last_asst_idx = None
    last_asst_full = None
    for msg in display_history:
        role = msg.get("role", "")
        display_kind = msg.get("display_kind")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []
        if display_kind == "hidden":
            continue
        if display_kind in _RESUME_EVENT_TEXT:
            entries.append(("event", _RESUME_EVENT_TEXT[display_kind]))
            continue
        if role == "user":
            text = _sanitize_display_text(_user_display_text(content))
            if len(text) > max_user_len:
                text = text[:max_user_len] + "..."
            entries.append(("user", text))
        elif role == "assistant":
            text = clean_assistant("" if content is None else str(content))
            parts, full_parts = [], []
            if text:
                full_parts.append(text)
                lines = text.splitlines()
                if len(lines) > max_asst_lines:
                    text = "\n".join(lines[:max_asst_lines]) + " ..."
                if len(text) > max_asst_len:
                    text = text[:max_asst_len] + "..."
                parts.append(text)
            if tool_calls:
                parts.append(_tool_calls_summary(tool_calls))
                full_parts.append(parts[-1])
            if not text and (skip_tool_only or not tool_calls):
                continue
            entries.append(("assistant", " ".join(parts)))
            last_asst_idx = len(entries) - 1
            last_asst_full = " ".join(full_parts)
    return entries, last_asst_idx, last_asst_full


# (skin key, fallback) for recap panel colors: body text, session label, border, assistant label.
_RESUME_SKIN_COLORS = (
    ("banner_text", "#FFF8DC"), ("session_label", "#DAA520"), ("session_border", "#8B8682"),
    ("ui_ok", "#8FBC8F"))


def _resume_panel_colors() -> tuple:
    """Active-skin colors for ``_RESUME_SKIN_COLORS`` (fallbacks when no skin loads)."""
    try:
        from hermes_cli.skin_engine import get_active_skin
        _skin = get_active_skin()
        return tuple(_skin.get_color(key, default) for key, default in _RESUME_SKIN_COLORS)
    except Exception:
        return tuple(default for _, default in _RESUME_SKIN_COLORS)


class CLIAgentSetupMixin:
    """Agent construction + session-resume display methods for ``HermesCLI``."""

    def _ensure_runtime_credentials(self) -> bool:
        """Re-resolve provider credentials before agent use so key rotation / token
        refresh are picked up without restarting the CLI. False on auth failure."""
        from cli import ChatConsole, logger
        from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
        _primary_exc = None
        runtime = None
        try:
            runtime = resolve_runtime_provider(
                requested=self.requested_provider, explicit_api_key=self._explicit_api_key,
                explicit_base_url=self._explicit_base_url)
        except Exception as exc:
            _primary_exc = exc
        if _primary_exc is not None:
            runtime = self._resolve_fallback_runtime(_primary_exc)
            if runtime is not None:
                _primary_exc = None
        if runtime is None:
            message = format_runtime_provider_error(_primary_exc) if _primary_exc else "Provider resolution failed."
            ChatConsole().print(f"[bold red]{message}[/]")
            return False
        api_key = runtime.get("api_key")
        base_url = runtime.get("base_url")
        resolved_provider = runtime.get("provider", "openrouter")
        resolved_routing = (
            resolved_provider, runtime.get("api_mode", self.api_mode), runtime.get("command"),
            list(runtime.get("args") or []))
        # A callable api_key is a bearer-token provider (Azure Entra ID): the OpenAI SDK
        # invokes it per request, so skip string validation / placeholder substitution.
        if not callable(api_key) and not (isinstance(api_key, str) and api_key):
            if _keyless_custom_base(base_url):
                # Placeholder key so the SDK doesn't reject the keyless local endpoint.
                api_key = "no-key-required"
                logger.debug(
                    "No API key for custom endpoint %s (source=%s), "
                    "using placeholder — local servers typically ignore auth",
                    base_url, runtime.get("source", ""))
            else:
                _prov = (resolved_provider or self.requested_provider or "").strip()
                if _prov and _prov != "auto":
                    print(f"\n⚠️  No API key found for provider '{_prov}'.")
                else:
                    print("\n⚠️  No inference provider is configured.")
                print("   Run 'hermes model' to choose a provider, or "
                      "'hermes setup' for first-time setup.")
                return False
        if not isinstance(base_url, str) or not base_url:
            print("\n⚠️  Provider resolver returned an empty base URL. "
                  "Check your provider config or run: hermes setup")
            return False
        credentials_changed = api_key != self.api_key or base_url != self.base_url
        routing_changed = resolved_routing != (self.provider, self.api_mode, self.acp_command, self.acp_args)
        self.provider, self.api_mode, self.acp_command, self.acp_args = resolved_routing
        self._credential_pool = runtime.get("credential_pool")
        self._provider_source = runtime.get("source")
        self.api_key = api_key
        self.base_url = base_url

        # A custom_provider entry's explicit `model` wins when the CLI model is unset or
        # is just the provider slug/display name (`hermes chat --model <provider-name>`
        # would otherwise send the provider name as the model string -> 400).
        runtime_model = runtime.get("model")
        if runtime_model and isinstance(runtime_model, str) and (
            not self.model or self.model == self.provider or self.model == runtime.get("name")):
            self.model = runtime_model

        # Still empty (e.g. `hermes auth add` without `hermes model`): fall back to the
        # provider's first catalog model so the API doesn't reject an empty model.
        if not self.model and resolved_provider:
            try:
                from hermes_cli.models import get_default_model_for_provider
                _default = get_default_model_for_provider(resolved_provider)
                if _default:
                    self.model = _default
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        _default, resolved_provider)
            except Exception:
                pass

        # Normalize model for the resolved provider (e.g. swap non-Codex models on openai-codex).
        # Fixes #651.
        model_changed = self._normalize_model_for_provider(resolved_provider)

        # AIAgent/OpenAI client holds auth at init, so rebuild on key/routing/model change.
        if (credentials_changed or routing_changed or model_changed) and self.agent is not None:
            self.agent = None
            self._active_agent_route_signature = None
        return True

    def _resolve_fallback_runtime(self, primary_exc):
        """Primary provider resolution failed: on an AuthError try each fallback entry in
        order and switch the CLI's requested_provider/model to the first that resolves.
        None when the error is not auth-related or no fallback resolves."""
        from cli import _cprint, logger
        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider
        if not isinstance(primary_exc, AuthError):
            return None
        _fb_chain = self._fallback_model if isinstance(self._fallback_model, list) else []
        for _fb in _fb_chain:
            _fb_provider = (_fb.get("provider") or "").strip().lower()
            _fb_model = (_fb.get("model") or "").strip()
            if not _fb_provider or not _fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key
                _fb_kwargs = {"requested": _fb_provider}
                if _fb.get("base_url"):
                    _fb_kwargs["explicit_base_url"] = _fb["base_url"]
                _fb_api_key = resolve_entry_api_key(_fb)
                if _fb_api_key:
                    _fb_kwargs["explicit_api_key"] = _fb_api_key
                runtime = resolve_runtime_provider(**_fb_kwargs)
                logger.warning(
                    "Primary provider auth failed (%s). Falling through to fallback: %s/%s",
                    primary_exc, _fb_provider, _fb_model)
                _cprint(f"⚠️  Primary auth failed — switching to fallback: {_fb_provider} / {_fb_model}")
                self.requested_provider = _fb_provider
                self.model = _fb_model
                return runtime
            except Exception:
                continue
        return None

    def _runtime_credentials_ready(self) -> bool:
        """Silently probe whether any inference provider can be resolved.

        Never prints or mutates CLI state, so the interactive first-run path can route a
        keyless install into onboarding before the user types into a chat that can't work.

        See #62935.
        """
        from hermes_cli.runtime_provider import resolve_runtime_provider
        try:
            runtime = resolve_runtime_provider(
                requested=self.requested_provider, explicit_api_key=self._explicit_api_key,
                explicit_base_url=self._explicit_base_url)
        except Exception:
            return False
        if not isinstance(runtime, dict):
            return False
        api_key = runtime.get("api_key")
        base_url = runtime.get("base_url")
        if callable(api_key) or (isinstance(api_key, str) and api_key):
            return bool(base_url)
        return _keyless_custom_base(base_url)

    def _offer_first_run_setup(self) -> bool:
        """Offer the provider picker when no provider is configured at all (interactive
        startup, TTY). Runs the same flow as ``hermes model`` so onboarding has a single
        source of truth. True when a provider was configured."""
        from cli import _cprint, logger
        _cprint("")
        _cprint("⚕ No inference provider is configured yet — let's fix that.")
        _cprint("  You'll pick a provider (Nous Portal OAuth is the fastest; "
                "no API key needed) and a model.")
        try:
            answer = input("  Set up a provider now? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            answer = "n"
        if answer in {"n", "no"}:
            _cprint("  Skipped. Run 'hermes model' or 'hermes setup' any time.")
            return False
        try:
            from hermes_cli.main import select_provider_and_model
            select_provider_and_model()
        except (KeyboardInterrupt, EOFError, SystemExit):
            print()
            _cprint("  Setup cancelled. Run 'hermes model' any time.")
            return False
        except Exception as exc:
            logger.debug("first-run provider setup failed: %s", exc)
            _cprint(f"  ⚠️  Provider setup failed: {exc}")
            _cprint("  Run 'hermes model' to try again.")
            return False

        # Re-sync CLI state from what the picker persisted so the next turn uses it without a restart.
        try:
            from hermes_cli.config import load_config
            _model_cfg = (load_config().get("model") or {})
            if isinstance(_model_cfg, dict):
                self.requested_provider = (_model_cfg.get("provider") or "").strip() or self.requested_provider
                _new_model = (_model_cfg.get("default") or _model_cfg.get("model") or "").strip()
                self.model = _new_model or self.model
        except Exception as exc:
            logger.debug("first-run config re-sync failed: %s", exc)
        # Force credential re-resolution + agent rebuild on next use.
        self.agent = None
        self._active_agent_route_signature = None
        if self._runtime_credentials_ready():
            _cprint("  ✓ Provider configured — you're ready to chat.")
            return True
        _cprint("  Provider setup didn't complete. Run 'hermes model' to retry.")
        return False

    def _resolve_turn_agent_config(self, user_message: str) -> dict:
        """Effective model/runtime config for one turn — always the session's primary
        provider. With `/fast` on (service_tier == "priority") attach request_overrides;
        auto/cold tiers are applied per request by agent.fast_mode instead."""
        from hermes_cli.models import resolve_fast_mode_overrides
        runtime = _current_runtime(self)
        route = {"model": self.model, "runtime": runtime, "signature": _route_signature(self.model, runtime)}
        overrides = None
        if getattr(self, "service_tier", None) == "priority":
            try:
                overrides = resolve_fast_mode_overrides(
                    route["model"], provider=runtime["provider"], base_url=runtime["base_url"])
            except Exception:
                pass
        route["request_overrides"] = overrides
        return route

    def _follow_compression_chain(self, session_meta, announce):
        """If the resumed id is an empty compression-chain head, announce and switch to
        the descendant holding the messages; returns the (possibly refreshed) meta."""
        resolved_id = _compression_descendant(self._session_db, self.session_id)
        if resolved_id:
            announce(resolved_id)
            self.session_id = resolved_id
            session_meta = self._session_db.get_session(self.session_id) or session_meta
        return session_meta

    def _restore_session_state(self, session_meta, *, quiet: bool = False) -> None:
        """Restore cwd / yolo / model from the resumed session's metadata."""
        self._restore_session_cwd(session_meta, quiet=quiet)
        self._restore_session_yolo(session_meta, quiet=quiet)
        self._restore_session_model(session_meta, quiet=quiet)

    def _reopen_session(self) -> None:
        """Clear ended_at so the resumed session is active again (best effort)."""
        try:
            self._session_db.reopen_session(self.session_id)
        except Exception:
            pass

    def _load_resumed_history_late(self) -> bool:
        """Late resume path: validate the session and load its history from the DB when
        _preload_resumed_session() (called from run()) did not already populate it.
        False when the resume must abort (missing session / over the safe-resume limit)."""
        from cli import ChatConsole, _DIM, _RST, _accent_hex, _cprint
        session_meta = self._session_db.get_session(self.session_id)
        # Quiet mode (tool_progress_mode == "off") routes resume status lines to
        # stderr so stdout stays machine-readable for `$(hermes chat -Q --resume ...)`.
        # Without this, the resume banner pollutes captured stdout. See #11793.
        _quiet_mode = getattr(self, "tool_progress_mode", "full") == "off"

        def _say(plain: str, rich: str) -> None:
            if _quiet_mode:
                print(plain, file=sys.stderr)
            else:
                ChatConsole().print(rich)
        if not session_meta:
            hint = "Use a session ID from a previous CLI run (hermes sessions list)."
            if _quiet_mode:
                print(f"Session not found: {self.session_id}", file=sys.stderr)
                print(hint, file=sys.stderr)
            else:
                _cprint(f"\033[1;31mSession not found: {self.session_id}{_RST}")
                _cprint(f"{_DIM}{hint}{_RST}")
            return False
        session_meta = self._follow_compression_chain(
            session_meta,
            lambda rid: ChatConsole().print(
                f"[dim]Session {_escape(self.session_id)} was compressed into "
                f"{_escape(rid)}; resuming the descendant with your "
                f"transcript.[/dim]"))
        if getattr(self, "_resume_history_error", None):
            return False
        # Only the TIP session's rows are loaded here (no ancestors), so use the
        # tip-only count — the full-lineage count would over-reject compressed sessions.
        resume_limit_error = self._resume_history_limit_error(tip_only=True)
        if resume_limit_error:
            self._resume_history_error = resume_limit_error
            _say(
                f"Cannot resume session: {resume_limit_error}",
                f"[bold red]Cannot resume session:[/] {_escape(resume_limit_error)}")
            return False
        restored = self._session_db.get_messages_as_conversation(self.session_id, repair_alternation=True)
        if restored:
            restored = [m for m in restored if m.get("role") != "session_meta"]
            self.conversation_history = restored
            msg_count = len([m for m in restored if m.get("role") == "user"])
            title_part = f" \"{session_meta['title']}\"" if session_meta.get("title") else ""
            counts = f"({msg_count} user message{'s' if msg_count != 1 else ''}, {len(restored)} total messages)"
            _say(
                f"↻ Resumed session {self.session_id}{title_part} {counts}",
                f"[bold {_accent_hex()}]↻ Resumed session[/] [bold]{_escape(self.session_id)}[/]"
                f"[bold {_accent_hex()}]{_escape(title_part)}[/] {counts}")
            self._restore_session_state(session_meta, quiet=_quiet_mode)
        else:
            _say(
                f"Session {self.session_id} found but has no messages. Starting fresh.",
                f"[bold {_accent_hex()}]Session {_escape(self.session_id)} found but has no messages. Starting fresh.[/]",
            )
        self._reopen_session()
        return True

    def _init_agent(self, *, model_override: str = None, runtime_override: dict = None, request_overrides: dict | None = None) -> bool:
        """Build the agent on first use; when resuming, restore history from SQLite.
        Returns True on success."""
        from cli import ChatConsole, _cprint, _prepare_deferred_agent_startup, logger
        from run_agent import AIAgent
        if self.agent is not None:
            return True

        # Join the background preloaded-skills load (--skills/-s) BEFORE the agent
        # snapshots self.system_prompt below. No-op when nothing was requested.
        self.finalize_preloaded_skills()
        _prepare_deferred_agent_startup()
        self._install_tool_callbacks()
        self._ensure_tirith_security()
        if not self._ensure_runtime_credentials():
            return False
        from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build
        ensure_mcp_discovery_before_agent_build(
            logger=logger, single_query=getattr(self, "_single_query_mode", False))
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.warning("SQLite session store not available — session will NOT be indexed: %s", e)
        if (
            self._resumed and self._session_db and not self.conversation_history
            and not self._load_resumed_history_late()):
            return False
        try:
            runtime = runtime_override or _current_runtime(self)
            effective_model = model_override or self.model
            # -q never builds the prompt_toolkit app, so the clarify modal can't be
            # answered — answer headless instead of polling until clarify_timeout.
            clarify_callback = (
                # See #94943.
                _single_query_clarify_callback
                if getattr(self, "_single_query_mode", False)
                else self._clarify_callback)
            self.agent = AIAgent(
                model=effective_model, api_key=runtime.get("api_key"),
                base_url=runtime.get("base_url"), provider=runtime.get("provider"),
                requested_provider=runtime.get("requested_provider"),
                api_mode=runtime.get("api_mode"), acp_command=runtime.get("command"),
                acp_args=runtime.get("args"), credential_pool=runtime.get("credential_pool"),
                max_tokens=self.max_tokens, max_iterations=self.max_turns,
                run_budget_seconds=getattr(self, "run_budget_seconds", None),
                enabled_toolsets=self.enabled_toolsets, disabled_toolsets=self.disabled_toolsets,
                verbose_logging=self.verbose, quiet_mode=not self.verbose,
                tool_progress_mode=getattr(self, "tool_progress_mode", "all"),
                ephemeral_system_prompt=self.system_prompt if self.system_prompt else None,
                prefill_messages=self.prefill_messages or None,
                reasoning_config=self.reasoning_config, service_tier=self.service_tier,
                request_overrides=request_overrides, providers_allowed=self._providers_only,
                providers_ignored=self._providers_ignore, providers_order=self._providers_order,
                provider_sort=self._provider_sort,
                provider_require_parameters=self._provider_require_params,
                provider_data_collection=self._provider_data_collection,
                openrouter_min_coding_score=self._openrouter_min_coding_score,
                session_id=self.session_id, platform="cli", session_db=self._session_db,
                clarify_callback=clarify_callback,
                reasoning_callback=self._current_reasoning_callback(),
                fallback_model=self._fallback_model, thinking_callback=self._on_thinking,
                checkpoints_enabled=self.checkpoints_enabled,
                checkpoint_max_snapshots=self.checkpoint_max_snapshots,
                checkpoint_max_total_size_mb=self.checkpoint_max_total_size_mb,
                checkpoint_max_file_size_mb=self.checkpoint_max_file_size_mb,
                pass_session_id=self.pass_session_id, skip_context_files=self.ignore_rules,
                skip_memory=self.ignore_rules, tool_progress_callback=self._on_tool_progress,
                tool_start_callback=self._on_tool_start if self._inline_diffs_enabled else None,
                tool_complete_callback=self._on_tool_complete if self._inline_diffs_enabled else None,
                stream_delta_callback=self._stream_delta if self.streaming_enabled else None,
                tool_gen_callback=self._on_tool_gen_start if self.streaming_enabled else None,
                notice_callback=self._on_notice, notice_clear_callback=self._on_notice_clear,
                reaction_callback=self._on_reaction)
            # Reference for atexit memory-provider shutdown: ``_run_cleanup`` in cli.py
            # reads ``cli._active_agent_ref``, so this MUST write the ``cli`` module's
            # global — a ``global`` statement here would bind this module's namespace.
            # When this code lived in cli.py a bare ``global _active_agent_ref`` worked; after the god-file
            # extraction into this mixin a ``global`` here would bind *this module's* namespace, leaving
            # ``cli._active_agent_ref`` None forever — so memory shutdown never ran on /exit (#49287).
            import cli as _cli
            _cli._active_agent_ref = self.agent
            # Route agent status output through prompt_toolkit so ANSI escapes
            # aren't garbled by patch_stdout's StdoutProxy.
            # See #2262.
            self.agent._print_fn = _cprint
            # Hydrate credits notices at session OPEN (parity with the TUI) so a depletion
            # warning shows before the first message. Idempotent + fail-open in the helper.
            try:
                from agent.credits_tracker import seed_credits_at_session_start
                seed_credits_at_session_start(self.agent)
            except Exception:
                pass
            self._active_agent_route_signature = _route_signature(effective_model, runtime)

            # Force-create DB row on /title intent, then apply title.
            if self._pending_title and self._session_db:
                try:
                    self.agent._ensure_db_session()
                    if self.agent._session_db_created:
                        self._session_db.set_session_title(self.session_id, self._pending_title)
                        _cprint(f"  Session title applied: {self._pending_title}")
                        self._pending_title = None
                    # else: row creation failed transiently — keep _pending_title for retry
                except Exception as e:
                    _cprint(f"  Could not apply pending title: {e}")
                    # Keep _pending_title so it can be retried after row creation succeeds
            return True
        except Exception as e:
            console = ChatConsole()
            console.print(f"[bold red]Failed to initialize agent: {e}[/]")
            from hermes_constants import partial_update_hint
            for line in partial_update_hint(e):
                console.print(line)
            return False

    def _resume_history_limit_error(self, tip_only: bool = False):
        """Return a safe-resume error without materializing transcript rows.

        ``tip_only`` matches call sites that load only the tip session's rows — counting
        the full lineage there would over-reject heavily-compressed sessions with a small
        tip. Generic guard failures fail OPEN; only a genuine over-limit result blocks."""
        if not self._session_db:
            return None
        from cli import logger
        from hermes_state import SessionResumeTooLargeError
        try:
            safety_check = getattr(self._session_db, "assert_resume_safe", None)
            if not callable(safety_check):
                return None
            safety_check(self.session_id, **({"tip_only": True} if tip_only else {}))
        except SessionResumeTooLargeError as exc:
            return str(exc)
        except Exception as exc:
            logger.warning(
                "Resume safety check failed for %s (proceeding without guard): %s",
                self.session_id, exc)
        return None

    def _preload_resumed_session(self) -> bool:
        """Load a resumed session's history early (from run(), before the first chat) so
        it can be displayed; ``_init_agent()`` then skips its own DB round-trip. Sets
        ``self.conversation_history`` and prints the status line. True if history loaded."""
        from cli import _accent_hex
        if not self._resumed or not self._session_db:
            return False
        session_meta = self._session_db.get_session(self.session_id)
        if not session_meta:
            self._console_print(f"[bold red]Session not found: {self.session_id}[/]")
            self._console_print("[dim]Use a session ID from a previous CLI run (hermes sessions list).[/]")
            return False
        session_meta = self._follow_compression_chain(
            session_meta,
            lambda rid: self._console_print(
                f"[dim]Session {self.session_id} was compressed into "
                f"{rid}; resuming the descendant with your transcript.[/]"))
        resume_limit_error = self._resume_history_limit_error()
        if resume_limit_error:
            self._resume_history_error = resume_limit_error
            self._console_print(f"[bold red]Cannot resume session:[/] {resume_limit_error}")
            return False
        restored, display_history = self._session_db.get_resume_conversations(self.session_id)
        accent_color = _accent_hex()
        if not restored:
            self._console_print(
                f"[{accent_color}]Session {self.session_id} found but has no "
                f"messages. Starting fresh.[/]")
            return False
        restored = [m for m in restored if m.get("role") != "session_meta"]
        self.conversation_history = restored
        self._resume_display_history = [m for m in display_history if m.get("role") != "session_meta"]
        from agent.context_compressor import is_user_originated_turn
        # Count only user-originated turns: legacy compaction handoffs are durable
        # role=user rows without display_kind.
        msg_count = len([m for m in self._resume_display_history if is_user_originated_turn(m)])
        title_part = f' "{session_meta["title"]}"' if session_meta.get("title") else ""
        self._console_print(
            f"[{accent_color}]↻ Resumed session [bold]{self.session_id}[/bold]"
            f"{title_part} "
            f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
            f"{len(restored)} total messages)[/]")
        self._restore_session_state(session_meta)
        self._reopen_session()
        return True

    def _display_resumed_history(self):
        """Render a dim Rich-panel recap of the previous conversation, capped at the last
        ``resume_exchanges`` user/assistant exchanges with a hidden-count indicator."""
        from cli import CLI_CONFIG, _record_output_history_entry, _strip_reasoning_tags, _suspend_output_history
        from tools.ansi_strip import sanitize_display_text as _sanitize_display_text
        display_history = getattr(self, "_resume_display_history", self.conversation_history)
        if not display_history or self.resume_display == "minimal":
            return
        _disp = CLI_CONFIG.get("display", {})
        entries, _last_asst_idx, _last_asst_full = _collect_resume_entries(
            display_history, _disp, lambda t: _sanitize_display_text(_strip_reasoning_tags(t)))
        if not entries:
            return
        skipped = max(0, len(entries) - int(_disp.get("resume_exchanges", 10)) * 2)
        entries = entries[skipped:]
        # Show the last assistant entry in full so the user sees where they left off.
        if _last_asst_idx is not None and _last_asst_full:
            adj_idx = _last_asst_idx - skipped
            if 0 <= adj_idx < len(entries):
                entries[adj_idx] = ("assistant_last", _last_asst_full)
        from rich.panel import Panel
        from rich.text import Text
        _history_text_c, _session_label_c, _session_border_c, _assistant_label_c = (
            _resume_panel_colors())

        # role -> (label, label style, body style, continuation indent)
        role_styles = {
            "user": ("  ● You: ", f"dim bold {_session_label_c}", "dim", " " * 9),
            "assistant": ("  ◆ Hermes: ", f"dim bold {_assistant_label_c}", "dim", " " * 12),
            "assistant_last": ("  ◆ Hermes: ", f"bold {_assistant_label_c}", "", " " * 12),  # full, non-dim
        }
        lines = Text()
        if skipped:
            lines.append(f"  ... {skipped} earlier messages ...\n\n", style="dim italic")
        for i, (role, text) in enumerate(entries):
            if role == "event":
                lines.append(f"  ◈ {text}\n", style="dim italic")
            else:
                label, label_style, body_style, indent = role_styles[role]
                lines.append(label, style=label_style)
                first, *rest = text.splitlines() or [""]  # first line inline, rest indented
                lines.append(first + "\n", style=body_style)
                for ml in rest:
                    lines.append(f"{indent}{ml}\n", style=body_style)
            if i < len(entries) - 1:
                lines.append("")  # small gap
        panel = Panel(
            lines, title=f"[dim {_session_label_c}]Previous Conversation[/]",
            border_style=f"dim {_session_border_c}", padding=(0, 1), style=_history_text_c)
        _record_output_history_entry(lambda: self._render_resume_history_panel_lines(panel))
        with _suspend_output_history():
            self._console_print(panel)
