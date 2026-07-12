"""Agent-construction and session-resume display methods for ``HermesCLI``.

Extracted from ``cli.py`` as part of the god-file decomposition campaign
(``~/.hermes/plans/god-file-decomposition.md``, Phase 4 step 2). This mixin holds
the agent lifecycle/setup cluster: runtime-credential resolution, per-turn agent
config, first-use agent construction, and resumed-session preload + history recap.

Behavior-neutral: every method is lifted verbatim from ``HermesCLI``. ``self.*``
calls resolve unchanged via the MRO. Neutral dependencies are imported at module
top level; ``cli.py``-internal helpers/constants are imported lazily inside each
method (``from cli import ...`` resolves at call time, when ``cli`` is fully
loaded) so this module never imports ``cli`` at import time -> no import cycle.
"""

from __future__ import annotations

import sys

from rich.markup import escape as _escape


class CLIAgentSetupMixin:
    """Agent construction + session-resume display methods for ``HermesCLI``."""

    def _ensure_runtime_credentials(self) -> bool:
        """
        Ensure runtime credentials are resolved before agent use.
        Re-resolves provider credentials so key rotation and token refresh
        are picked up without restarting the CLI.
        Returns True if credentials are ready, False on auth failure.
        """
        from cli import ChatConsole, _cprint, logger
        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )

        # MoA virtual provider (#56828): recognize `-m moa:<preset>` /
        # `--provider moa` here so classic non-interactive chat resolves the
        # virtual provider instead of sending the literal model string to a
        # real API. Idempotent: once normalized, re-resolution is a no-op.
        from hermes_cli.moa_config import resolve_moa_request

        _cli_config = getattr(self, "config", None)
        try:
            _moa = resolve_moa_request(
                self.requested_provider,
                self.model,
                _cli_config if isinstance(_cli_config, dict) else {},
                model_explicit=not getattr(self, "_model_is_default", True),
            )
        except ValueError as exc:
            ChatConsole().print(f"[bold red]{exc}[/]")
            return False
        if _moa is not None:
            self.requested_provider, self.model = _moa

        _primary_exc = None
        runtime = None
        try:
            runtime = resolve_runtime_provider(
                requested=self.requested_provider,
                explicit_api_key=self._explicit_api_key,
                explicit_base_url=self._explicit_base_url,
            )
        except Exception as exc:
            _primary_exc = exc

        # Primary provider auth failed — try fallback providers before giving up.
        if runtime is None and _primary_exc is not None:
            from hermes_cli.auth import AuthError
            if isinstance(_primary_exc, AuthError):
                _fb_chain = self._fallback_model if isinstance(self._fallback_model, list) else []
                for _fb in _fb_chain:
                    _fb_provider = (_fb.get("provider") or "").strip().lower()
                    _fb_model = (_fb.get("model") or "").strip()
                    if not _fb_provider or not _fb_model:
                        continue
                    try:
                        runtime = resolve_runtime_provider(requested=_fb_provider)
                        logger.warning(
                            "Primary provider auth failed (%s). Falling through to fallback: %s/%s",
                            _primary_exc, _fb_provider, _fb_model,
                        )
                        _cprint(f"⚠️  Primary auth failed — switching to fallback: {_fb_provider} / {_fb_model}")
                        self.requested_provider = _fb_provider
                        self.model = _fb_model
                        _primary_exc = None
                        break
                    except Exception:
                        continue

        if runtime is None:
            message = format_runtime_provider_error(_primary_exc) if _primary_exc else "Provider resolution failed."
            ChatConsole().print(f"[bold red]{message}[/]")
            return False

        api_key = runtime.get("api_key")
        base_url = runtime.get("base_url")
        resolved_provider = runtime.get("provider", "openrouter")
        resolved_api_mode = runtime.get("api_mode", self.api_mode)
        resolved_acp_command = runtime.get("command")
        resolved_acp_args = list(runtime.get("args") or [])
        resolved_credential_pool = runtime.get("credential_pool")
        # A callable api_key is a bearer-token provider (Azure Foundry
        # Entra ID — ``azure_identity_adapter.build_token_provider``).
        # The OpenAI SDK accepts ``Callable[[], str]`` for ``api_key`` and
        # invokes it before every request. Skip the string-only validation
        # and placeholder substitution for callables.
        _is_callable_provider = callable(api_key) and not isinstance(api_key, str)
        if not _is_callable_provider and (not isinstance(api_key, str) or not api_key):
            # Custom / local endpoints (llama.cpp, ollama, vLLM, etc.) often
            # don't require authentication.  When a base_url IS configured but
            # no API key was found, use a placeholder so the OpenAI SDK
            # doesn't reject the request and local servers just ignore it.
            _source = runtime.get("source", "")
            _has_custom_base = isinstance(base_url, str) and base_url and "openrouter.ai" not in base_url
            if _has_custom_base:
                api_key = "no-key-required"
                logger.debug(
                    "No API key for custom endpoint %s (source=%s), "
                    "using placeholder — local servers typically ignore auth",
                    base_url, _source,
                )
            else:
                print("\n⚠️  Provider resolver returned an empty API key. "
                      "Set OPENROUTER_API_KEY or run: hermes setup")
                return False
        if not isinstance(base_url, str) or not base_url:
            print("\n⚠️  Provider resolver returned an empty base URL. "
                  "Check your provider config or run: hermes setup")
            return False

        credentials_changed = api_key != self.api_key or base_url != self.base_url
        routing_changed = (
            resolved_provider != self.provider
            or resolved_api_mode != self.api_mode
            or resolved_acp_command != self.acp_command
            or resolved_acp_args != self.acp_args
        )
        self.provider = resolved_provider
        self.api_mode = resolved_api_mode
        self.acp_command = resolved_acp_command
        self.acp_args = resolved_acp_args
        self._credential_pool = resolved_credential_pool
        self._provider_source = runtime.get("source")
        self.api_key = api_key
        self.base_url = base_url

        # When a custom_provider entry carries an explicit `model` field,
        # use it as the effective model name.  Without this, running
        # `hermes chat --model <provider-name>` sends the provider name
        # (e.g. "my-provider") as the model string to the API instead of
        # the configured model (e.g. "qwen3.6-plus"), causing 400 errors.
        runtime_model = runtime.get("model")
        if runtime_model and isinstance(runtime_model, str):
            # Only use runtime model if: model is unset, or model equals provider name
            should_use_runtime_model = (
                not self.model or  # No model configured yet
                self.model == self.provider or  # Model is the provider slug
                self.model == runtime.get("name")  # Model matches provider display name
            )
            if should_use_runtime_model:
                self.model = runtime_model

        # If model is still empty (e.g. user ran `hermes auth add openai-codex`
        # without `hermes model`), fall back to the provider's first catalog
        # model so the API call doesn't fail with "model must be non-empty".
        if not self.model and resolved_provider:
            try:
                from hermes_cli.models import get_default_model_for_provider
                _default = get_default_model_for_provider(resolved_provider)
                if _default:
                    self.model = _default
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        _default, resolved_provider,
                    )
            except Exception:
                pass

        # Normalize model for the resolved provider (e.g. swap non-Codex
        # models when provider is openai-codex).  Fixes #651.
        model_changed = self._normalize_model_for_provider(resolved_provider)

        # AIAgent/OpenAI client holds auth at init time, so rebuild if key,
        # routing, or the effective model changed.
        if (credentials_changed or routing_changed or model_changed) and self.agent is not None:
            self.agent = None
            self._active_agent_route_signature = None

        return True

    def _resolve_turn_agent_config(self, user_message: str) -> dict:
        """Build the effective model/runtime config for a single user turn.

        Always uses the session's primary model/provider.  If the user has
        toggled `/fast` on and the current model supports Priority
        Processing / Anthropic fast mode, attach `request_overrides` so the
        API call is marked accordingly.
        """
        from hermes_cli.models import resolve_fast_mode_overrides

        runtime = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "provider": self.provider,
            "api_mode": self.api_mode,
            "command": self.acp_command,
            "args": list(self.acp_args or []),
            "credential_pool": getattr(self, "_credential_pool", None),
        }
        route = {
            "model": self.model,
            "runtime": runtime,
            "signature": (
                self.model,
                runtime["provider"],
                runtime["base_url"],
                runtime["api_mode"],
                runtime["command"],
                tuple(runtime["args"]),
            ),
        }

        service_tier = getattr(self, "service_tier", None)
        if not service_tier:
            route["request_overrides"] = None
            return route

        try:
            overrides = resolve_fast_mode_overrides(route["model"])
        except Exception:
            overrides = None
        route["request_overrides"] = overrides
        return route

    def _init_agent(self, *, model_override: str = None, runtime_override: dict = None, request_overrides: dict | None = None) -> bool:
        """
        Initialize the agent on first use.
        When resuming a session, restores conversation history from SQLite.
        
        Returns:
            bool: True if successful, False otherwise
        """
        from cli import AIAgent, ChatConsole, _DIM, _RST, _accent_hex, _cprint, _prepare_deferred_agent_startup, logger
        if self.agent is not None:
            return True

        _prepare_deferred_agent_startup()
        self._install_tool_callbacks()
        self._ensure_tirith_security()

        if not self._ensure_runtime_credentials():
            return False

        from hermes_cli.mcp_startup import wait_for_mcp_discovery

        wait_for_mcp_discovery()

        # Initialize SQLite session store for CLI sessions (if not already done in __init__)
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.warning("SQLite session store not available — session will NOT be indexed: %s", e)
        
        # If resuming, validate the session exists and load its history.
        # _preload_resumed_session() may have already loaded it (called from
        # run() for immediate display).  In that case, conversation_history
        # is non-empty and we skip the DB round-trip.
        if self._resumed and self._session_db and not self.conversation_history:
            session_meta = self._session_db.get_session(self.session_id)
            # In quiet mode (`hermes chat -Q` / --quiet, surfaced via
            # tool_progress_mode == "off"), resume status lines go to stderr
            # so stdout stays machine-readable for automation wrappers that
            # do `$(hermes chat -Q --resume <id> -q "...")`. Without this,
            # the resume banner pollutes captured stdout. See #11793.
            _quiet_mode = getattr(self, "tool_progress_mode", "full") == "off"
            if not session_meta:
                if _quiet_mode:
                    print(f"Session not found: {self.session_id}", file=sys.stderr)
                    print(
                        "Use a session ID from a previous CLI run (hermes sessions list).",
                        file=sys.stderr,
                    )
                else:
                    _cprint(f"\033[1;31mSession not found: {self.session_id}{_RST}")
                    _cprint(f"{_DIM}Use a session ID from a previous CLI run (hermes sessions list).{_RST}")
                return False
            # If the requested session is the (empty) head of a compression
            # chain, walk to the descendant that actually holds the messages.
            # See #15000 and SessionDB.resolve_resume_session_id.
            try:
                resolved_id = self._session_db.resolve_resume_session_id(self.session_id)
            except Exception:
                resolved_id = self.session_id
            if resolved_id and resolved_id != self.session_id:
                ChatConsole().print(
                    f"[dim]Session {_escape(self.session_id)} was compressed into "
                    f"{_escape(resolved_id)}; resuming the descendant with your "
                    f"transcript.[/dim]"
                )
                self.session_id = resolved_id
                resolved_meta = self._session_db.get_session(self.session_id)
                if resolved_meta:
                    session_meta = resolved_meta
            restored = self._session_db.get_messages_as_conversation(self.session_id)
            if restored:
                restored = [m for m in restored if m.get("role") != "session_meta"]
                self.conversation_history = restored
                msg_count = len([m for m in restored if m.get("role") == "user"])
                title_part = ""
                if session_meta.get("title"):
                    title_part = f" \"{session_meta['title']}\""
                if _quiet_mode:
                    print(
                        f"↻ Resumed session {self.session_id}{title_part} "
                        f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                        f"{len(restored)} total messages)",
                        file=sys.stderr,
                    )
                else:
                    ChatConsole().print(
                        f"[bold {_accent_hex()}]↻ Resumed session[/] "
                        f"[bold]{_escape(self.session_id)}[/]"
                        f"[bold {_accent_hex()}]{_escape(title_part)}[/] "
                        f"({msg_count} user message{'s' if msg_count != 1 else ''}, {len(restored)} total messages)"
                    )
                self._restore_session_cwd(session_meta, quiet=_quiet_mode)
            else:
                if _quiet_mode:
                    print(
                        f"Session {self.session_id} found but has no messages. Starting fresh.",
                        file=sys.stderr,
                    )
                else:
                    ChatConsole().print(
                        f"[bold {_accent_hex()}]Session {_escape(self.session_id)} found but has no messages. Starting fresh.[/]"
                    )
            # Re-open the session (clear ended_at so it's active again)
            try:
                self._session_db._conn.execute(
                    "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                    (self.session_id,),
                )
                self._session_db._conn.commit()
            except Exception:
                pass
        
        try:
            runtime = runtime_override or {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "provider": self.provider,
                "api_mode": self.api_mode,
                "command": self.acp_command,
                "args": list(self.acp_args or []),
                "credential_pool": getattr(self, "_credential_pool", None),
            }
            effective_model = model_override or self.model
            self.agent = AIAgent(
                model=effective_model,
                api_key=runtime.get("api_key"),
                base_url=runtime.get("base_url"),
                provider=runtime.get("provider"),
                api_mode=runtime.get("api_mode"),
                acp_command=runtime.get("command"),
                acp_args=runtime.get("args"),
                credential_pool=runtime.get("credential_pool"),
                max_tokens=self.max_tokens,
                max_iterations=self.max_turns,
                enabled_toolsets=self.enabled_toolsets,
                disabled_toolsets=self.disabled_toolsets,
                verbose_logging=self.verbose,
                quiet_mode=not self.verbose,
                tool_progress_mode=getattr(self, "tool_progress_mode", "all"),
                ephemeral_system_prompt=self.system_prompt if self.system_prompt else None,
                prefill_messages=self.prefill_messages or None,
                reasoning_config=self.reasoning_config,
                service_tier=self.service_tier,
                request_overrides=request_overrides,
                providers_allowed=self._providers_only,
                providers_ignored=self._providers_ignore,
                providers_order=self._providers_order,
                provider_sort=self._provider_sort,
                provider_require_parameters=self._provider_require_params,
                provider_data_collection=self._provider_data_collection,
                openrouter_min_coding_score=self._openrouter_min_coding_score,
                session_id=self.session_id,
                platform="cli",
                session_db=self._session_db,
                clarify_callback=self._clarify_callback,
                reasoning_callback=self._current_reasoning_callback(),

                fallback_model=self._fallback_model,
                thinking_callback=self._on_thinking,
                checkpoints_enabled=self.checkpoints_enabled,
                checkpoint_max_snapshots=self.checkpoint_max_snapshots,
                checkpoint_max_total_size_mb=self.checkpoint_max_total_size_mb,
                checkpoint_max_file_size_mb=self.checkpoint_max_file_size_mb,
                pass_session_id=self.pass_session_id,
                skip_context_files=self.ignore_rules,
                skip_memory=self.ignore_rules,
                tool_progress_callback=self._on_tool_progress,
                tool_start_callback=self._on_tool_start if self._inline_diffs_enabled else None,
                tool_complete_callback=self._on_tool_complete if self._inline_diffs_enabled else None,
                stream_delta_callback=self._stream_delta if self.streaming_enabled else None,
                tool_gen_callback=self._on_tool_gen_start if self.streaming_enabled else None,
                notice_callback=self._on_notice,
                notice_clear_callback=self._on_notice_clear,
                reaction_callback=self._on_reaction,
            )
            # Store reference for atexit memory provider shutdown.
            # NOTE: this MUST write to the ``cli`` module's global, not a
            # local module global. ``_run_cleanup`` (in cli.py) reads
            # ``cli._active_agent_ref`` to decide whether to fire the memory
            # provider's ``on_session_end`` hook. When this code lived in
            # cli.py a bare ``global _active_agent_ref`` worked; after the
            # god-file extraction into this mixin a ``global`` here would bind
            # *this module's* namespace, leaving ``cli._active_agent_ref`` None
            # forever — so memory shutdown never ran on /exit (#49287).
            import cli as _cli
            _cli._active_agent_ref = self.agent
            # Route agent status output through prompt_toolkit so ANSI escape
            # sequences aren't garbled by patch_stdout's StdoutProxy (#2262).
            self.agent._print_fn = _cprint
            # Hydrate credits notices at session OPEN (parity with the TUI), so a
            # depletion / usage-band warning shows before the first message. The
            # notice_callback is bound above → _on_notice renders the line. Idempotent
            # + fail-open inside the helper; harmless for non-Nous providers.
            try:
                from agent.credits_tracker import seed_credits_at_session_start

                seed_credits_at_session_start(self.agent)
            except Exception:
                pass
            self._active_agent_route_signature = (
                effective_model,
                runtime.get("provider"),
                runtime.get("base_url"),
                runtime.get("api_mode"),
                runtime.get("command"),
                tuple(runtime.get("args") or ()),
            )

            # Force-create DB row on /title intent, then apply title.
            if self._pending_title and self._session_db and self.agent:
                try:
                    self.agent._ensure_db_session()
                    if self.agent._session_db_created:
                        self._session_db.set_session_title(self.session_id, self._pending_title)
                        _cprint(f"  Session title applied: {self._pending_title}")
                        self._pending_title = None
                    # else: row creation failed transiently — keep _pending_title for retry
                except (ValueError, Exception) as e:
                    _cprint(f"  Could not apply pending title: {e}")
                    # Keep _pending_title so it can be retried after row creation succeeds
            return True
        except Exception as e:
            ChatConsole().print(f"[bold red]Failed to initialize agent: {e}[/]")
            return False

    def _preload_resumed_session(self) -> bool:
        """Load a resumed session's history from the DB early (before first chat).

        Called from run() so the conversation history is available for display
        before the user sends their first message.  Sets
        ``self.conversation_history`` and prints the one-liner status.  Returns
        True if history was loaded, False otherwise.

        The corresponding block in ``_init_agent()`` checks whether history is
        already populated and skips the DB round-trip.
        """
        from cli import _accent_hex
        if not self._resumed or not self._session_db:
            return False

        session_meta = self._session_db.get_session(self.session_id)
        if not session_meta:
            self._console_print(
                f"[bold red]Session not found: {self.session_id}[/]"
            )
            self._console_print(
                "[dim]Use a session ID from a previous CLI run "
                "(hermes sessions list).[/]"
            )
            return False

        # If the requested session is the (empty) head of a compression chain,
        # walk to the descendant that actually holds the messages. See #15000.
        try:
            resolved_id = self._session_db.resolve_resume_session_id(self.session_id)
        except Exception:
            resolved_id = self.session_id
        if resolved_id and resolved_id != self.session_id:
            self._console_print(
                f"[dim]Session {self.session_id} was compressed into "
                f"{resolved_id}; resuming the descendant with your transcript.[/]"
            )
            self.session_id = resolved_id
            resolved_meta = self._session_db.get_session(self.session_id)
            if resolved_meta:
                session_meta = resolved_meta

        restored = self._session_db.get_messages_as_conversation(self.session_id)
        if restored:
            restored = [m for m in restored if m.get("role") != "session_meta"]
            self.conversation_history = restored
            msg_count = len([m for m in restored if m.get("role") == "user"])
            title_part = ""
            if session_meta.get("title"):
                title_part = f' "{session_meta["title"]}"'
            accent_color = _accent_hex()
            self._console_print(
                f"[{accent_color}]↻ Resumed session [bold]{self.session_id}[/bold]"
                f"{title_part} "
                f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                f"{len(restored)} total messages)[/]"
            )
            self._restore_session_cwd(session_meta)
        else:
            accent_color = _accent_hex()
            self._console_print(
                f"[{accent_color}]Session {self.session_id} found but has no "
                f"messages. Starting fresh.[/]"
            )
            return False

        # Re-open the session (clear ended_at so it's active again)
        try:
            self._session_db._conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ?",
                (self.session_id,),
            )
            self._session_db._conn.commit()
        except Exception:
            pass

        return True

    def _display_resumed_history(self):
        """Render a compact recap of previous conversation messages.

        Uses Rich markup with dim/muted styling so the recap is visually
        distinct from the active conversation.  Caps the display at the
        last ``MAX_DISPLAY_EXCHANGES`` user/assistant exchanges and shows
        an indicator for earlier hidden messages.
        """
        from cli import CLI_CONFIG, _record_output_history_entry, _strip_reasoning_tags, _suspend_output_history
        if not self.conversation_history:
            return

        # Check config: resume_display setting
        if self.resume_display == "minimal":
            return

        # Read limits from config (with hardcoded defaults)
        _disp = CLI_CONFIG.get("display", {})
        MAX_DISPLAY_EXCHANGES = int(_disp.get("resume_exchanges", 10))
        MAX_USER_LEN = int(_disp.get("resume_max_user_chars", 300))
        MAX_ASST_LEN = int(_disp.get("resume_max_assistant_chars", 200))
        MAX_ASST_LINES = int(_disp.get("resume_max_assistant_lines", 3))
        SKIP_TOOL_ONLY = _disp.get("resume_skip_tool_only", True)

        # Collect displayable entries (skip system, tool-result messages)
        entries = []  # list of (role, display_text)
        _last_asst_idx = None       # index of last assistant entry
        _last_asst_full = None      # un-truncated display text for last assistant
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            if role == "system":
                continue
            if role == "tool":
                continue

            if role == "user":
                text = "" if content is None else str(content)
                # Handle multimodal content (list of dicts)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            parts.append("[image]")
                    text = " ".join(parts)
                if len(text) > MAX_USER_LEN:
                    text = text[:MAX_USER_LEN] + "..."
                entries.append(("user", text))

            elif role == "assistant":
                text = "" if content is None else str(content)
                text = _strip_reasoning_tags(text)
                parts = []
                full_parts = []  # un-truncated version
                if text:
                    full_parts.append(text)
                    lines = text.splitlines()
                    if len(lines) > MAX_ASST_LINES:
                        text = "\n".join(lines[:MAX_ASST_LINES]) + " ..."
                    if len(text) > MAX_ASST_LEN:
                        text = text[:MAX_ASST_LEN] + "..."
                    parts.append(text)
                if tool_calls:
                    tc_count = len(tool_calls)
                    # Extract tool names
                    names = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
                        if name not in names:
                            names.append(name)
                    names_str = ", ".join(names[:4])
                    if len(names) > 4:
                        names_str += ", ..."
                    noun = "call" if tc_count == 1 else "calls"
                    tc_summary = f"[{tc_count} tool {noun}: {names_str}]"
                    parts.append(tc_summary)
                    full_parts.append(tc_summary)
                if not parts:
                    # Skip pure-reasoning messages that have no visible output
                    continue
                # Skip tool-call-only entries when SKIP_TOOL_ONLY is enabled
                has_text = bool(text)
                if SKIP_TOOL_ONLY and not has_text and tool_calls:
                    continue
                entries.append(("assistant", " ".join(parts)))
                _last_asst_idx = len(entries) - 1
                _last_asst_full = " ".join(full_parts)

        if not entries:
            return

        # Determine if we need to truncate
        skipped = 0
        if len(entries) > MAX_DISPLAY_EXCHANGES * 2:
            skipped = len(entries) - MAX_DISPLAY_EXCHANGES * 2
            entries = entries[skipped:]

        # Replace last assistant entry with full (un-truncated) text
        # so the user can see where they left off without wasting tokens.
        if _last_asst_idx is not None and _last_asst_full:
            adj_idx = _last_asst_idx - skipped
            if 0 <= adj_idx < len(entries):
                entries[adj_idx] = ("assistant_last", _last_asst_full)

        # Build the display using Rich
        from rich.panel import Panel
        from rich.text import Text

        try:
            from hermes_cli.skin_engine import get_active_skin
            _skin = get_active_skin()
            _history_text_c = _skin.get_color("banner_text", "#FFF8DC")
            _session_label_c = _skin.get_color("session_label", "#DAA520")
            _session_border_c = _skin.get_color("session_border", "#8B8682")
            _assistant_label_c = _skin.get_color("ui_ok", "#8FBC8F")
        except Exception:
            _history_text_c = "#FFF8DC"
            _session_label_c = "#DAA520"
            _session_border_c = "#8B8682"
            _assistant_label_c = "#8FBC8F"

        lines = Text()
        if skipped:
            lines.append(
                f"  ... {skipped} earlier messages ...\n\n",
                style="dim italic",
            )

        for i, (role, text) in enumerate(entries):
            if role == "user":
                lines.append("  ● You: ", style=f"dim bold {_session_label_c}")
                # Show first line inline, indent rest
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="dim")
                for ml in msg_lines[1:]:
                    lines.append(f"         {ml}\n", style="dim")
            elif role == "assistant_last":
                # Last assistant response shown in full, non-dim
                lines.append("  ◆ Hermes: ", style=f"bold {_assistant_label_c}")
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="")
                for ml in msg_lines[1:]:
                    lines.append(f"            {ml}\n", style="")
            else:
                lines.append("  ◆ Hermes: ", style=f"dim bold {_assistant_label_c}")
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="dim")
                for ml in msg_lines[1:]:
                    lines.append(f"            {ml}\n", style="dim")
            if i < len(entries) - 1:
                lines.append("")  # small gap

        panel = Panel(
            lines,
            title=f"[dim {_session_label_c}]Previous Conversation[/]",
            border_style=f"dim {_session_border_c}",
            padding=(0, 1),
            style=_history_text_c,
        )
        _record_output_history_entry(lambda: self._render_resume_history_panel_lines(panel))
        with _suspend_output_history():
            self._console_print(panel)
