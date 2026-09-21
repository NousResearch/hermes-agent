"""Classic-TUI run loop phases: input processing, after-turn bookkeeping, startup banner/prewarm/maintenance, prompt_toolkit application build, signal handlers and shutdown.

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO.
"""

from __future__ import annotations

import errno
import logging
import os
import queue
import shutil
import sys
import threading
import time
from agent.interrupt_compat import request_hard_interrupt
from agent.pet import render as pet_render
from contextlib import suppress
from hermes_constants import get_hermes_home
from prompt_toolkit.application import Application
from rich.markup import escape as _escape

# Log-record parity with the origin module.
logger = logging.getLogger("cli")


class CLITuiRuntimeMixin:
    """Classic-TUI run loop phases: input processing, after-turn bookkeeping, startup banner/prewarm/maintenance, prompt_toolkit application build, signal handlers and shutdown."""

    def _tui_process_loop(self):
        """REPL worker thread: drain ``_pending_input``, run idle housekeeping, dispatch each input."""
        while not self._should_exit:
            try:
                try:
                    user_input = self._pending_input.get(timeout=0.1)
                except queue.Empty:
                    if not self._agent_running:
                        self._tui_idle_tick()
                    continue
                self._tui_process_one_input(user_input)
            except Exception as e:
                if isinstance(e, OSError) and e.errno == errno.EIO:
                    self._mark_terminal_io_broken("process_loop")
                    logger.warning("process_loop EIO — freezing UI paints (#81521): %s", e)
                    continue
                logger.warning("process_loop unhandled error (msg may be lost): %s", e)

    def _tui_idle_tick(self):
        """Idle housekeeping between inputs (agent not running)."""
        self._check_config_mcp_changes()  # auto-reload MCP on mcp_servers change
        # Termios drift heal first: a drifted tty makes the CLI look dead while the loop is healthy.
        for step in (
            self._check_termios_drift,
            lambda: self._drain_process_notifications("cli-idle"),
            self._maybe_fire_loop_tick,
            self._maybe_resume_parked_goal,
        ):
            with suppress(Exception):
                step()

    def _tui_process_one_input(self, user_input):
        """Route one submitted input: file drop, /resume pick, ! shell, slash command, or a chat turn."""
        from cli import _DIM, _PASTE_REF_RE, _RST, _cprint, _detect_file_drop, _looks_like_slash_command, _strip_leaked_bracketed_paste_wrappers, _strip_leaked_terminal_responses_with_meta
        from tools.process_registry_notifications import TimelineNotification
        user_input, is_voice_input, is_seeded_query = self._tui_unwrap_input(user_input)
        if not user_input:
            return
        notification_preview = user_input if isinstance(user_input, TimelineNotification) else None
        self._status_bar_suppressed_after_resize = False  # input ends post-resize suppression

        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input

        if isinstance(user_input, str):
            user_input = _strip_leaked_bracketed_paste_wrappers(user_input)
            user_input, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(user_input)
            if _had_mouse_reports:
                self._recover_terminal_input_modes(reason="mouse reports leaked into submitted input")

        # A typed bare stop phrase ends an active voice chat (transcripts are checked earlier).
        if not is_voice_input and self._typed_voice_stop(user_input):
            return

        # File drops are detected before any dispatch; seeded -q prompts are literal text.
        _file_drop = _detect_file_drop(user_input) if isinstance(user_input, str) and not is_seeded_query else None
        if _file_drop:
            _drop_path = _file_drop["path"]
            _remainder = _file_drop["remainder"]
            if _file_drop["is_image"]:
                submit_images.append(_drop_path)
                user_input = _remainder or f"[User attached image: {_drop_path.name}]"
                _cprint(f"  📎 Auto-attached image: {_drop_path.name}")
            else:
                _cprint(f"  📄 Detected file: {_drop_path.name}")
                user_input = f"[User attached file: {_drop_path}]" + (f"\n{_remainder}" if _remainder else "")
        elif isinstance(user_input, str):
            # A bare number right after a bare `/resume` selects that session (never sent to the agent).
            if self._pending_resume_sessions and self._consume_pending_resume_selection(user_input):
                return
            if not is_seeded_query:
                if self.handle_bang_shell(user_input):
                    return
                if _looks_like_slash_command(user_input):
                    user_input = self._tui_run_slash_input(user_input)
                    if user_input is None:
                        return

        if isinstance(user_input, str) and _PASTE_REF_RE.search(user_input):
            user_input = self._expand_paste_references(user_input)
        print()
        self._print_user_message_preview(notification_preview or user_input)

        if submit_images:
            n = len(submit_images)
            _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

        self._agent_running = self._interactive_turn = True
        self._pet_turn_error = self._pet_reasoning = False
        self._turn_summary_begin()
        self._app.invalidate()
        try:
            self.chat(notification_preview or user_input, images=submit_images or None, voice_input=is_voice_input)
        finally:
            self._tui_after_turn()

    def _tui_run_slash_input(self, user_input: str):
        """Dispatch a slash command. Returns the pending agent seed to run as a chat turn, else None."""
        from cli import _cprint
        _cprint(f"\n⚙️  {user_input}")
        try:
            if not self.process_command(user_input):
                self._should_exit = True
                if self._app.is_running:
                    self._app.exit()
        except KeyboardInterrupt:
            # Ctrl+C during a slow slash command returns to the prompt instead of exiting.
            _cprint("\n[dim]Command interrupted.[/dim]")
            return None
        _seed, self._pending_agent_seed = self._pending_agent_seed, None
        return _seed or None

    def _tui_after_turn(self):
        """Post-turn bookkeeping after chat() returns (normal, error, or interrupt)."""
        from cli import _DIM, _RST, _cprint
        self._agent_running = self._pet_reasoning = False
        self._spinner_text = self._last_scrollback_tool = ""
        self._tool_start_time = 0.0
        self._pending_tool_info.clear()
        self._pet_react_turn_end()
        self._turn_summary_emit()
        self._interactive_turn = False
        self._app.invalidate()

        # After an interrupt the renderer may have drifted (leaked CPR text, VT100 parser
        # stalled mid-escape): drain stray bytes and force a clean redraw.
        if self._last_turn_interrupted:
            self._recover_terminal_after_interrupt()

        # Re-queue any messages that arrived in _interrupt_queue while the agent was running and were never
        # claimed by the explicit interrupt path. See _drain_interrupt_queue_to_pending_input for the full
        # rationale. Regression of #17666 / #18760 — the drain block from the original PR #17939 was
        # deferred as "worth its own review" and never re-landed (#20271).
        self._drain_interrupt_queue_to_pending_input()

        # /goal continuation (queued user input still preempts), then /loop tick completion.
        for hook, what in (
            (self._maybe_continue_goal_after_turn, "goal continuation"),
            (self._maybe_complete_loop_tick_after_turn, "loop completion"),
        ):
            try:
                hook()
            except Exception as _exc:
                logging.debug("%s hook failed: %s", what, _exc)

        # Continuous voice: restart recording off-thread (beep + recorder start would block process_loop).
        if self._voice_mode and self._voice_continuous and not self._voice_recording:
            def _restart_recording():
                try:
                    if self._voice_tts:
                        self._voice_tts_done.wait(timeout=60)
                        time.sleep(0.3)
                    # A barge-in capture already owns the mic and submits the interruption itself.
                    if self._voice_barge_capture.is_set():
                        return
                    self._voice_start_recording()
                    self._app.invalidate()
                except Exception as e:
                    _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
            threading.Thread(target=_restart_recording, daemon=True).start()

        with suppress(Exception):
            self._drain_process_notifications("cli-post-turn")

    def _tui_signal_handler(self, signum, frame):
        """SIGHUP/SIGTERM -> graceful shutdown.

        The agent is hard-interrupted first so its daemon thread can kill the tool's
        setsid subprocess group before the main thread unwinds (else an orphan child).
        ``logger.debug`` is guarded: logging is not reentrant-safe and a shutdown race
        can raise ``KeyError`` inside the handler, bypassing prompt_toolkit's unwind.
        """
        from cli import _arm_exit_watchdog_on_shutdown_signal, _interrupt_agent_for_signal
        with suppress(Exception):
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
        # Arm the backstop IMMEDIATELY: if the unwind wedges, _run_cleanup never arms its own.
        # Shutdown intent is now unambiguous — arm the exit backstop IMMEDIATELY, before the graceful unwind
        # below. If any step of that unwind wedges (main thread parked in a syscall, prompt_toolkit teardown
        # never returning), _run_cleanup never runs and would never arm its own watchdog — leaving a "dead"
        # CLI alive for minutes (#65998 class).
        # Arm the exit backstop now that shutdown intent is unambiguous — covers wedges in the unwind below
        # that would otherwise leave the process alive with no watchdog (#65998 class).
        _arm_exit_watchdog_on_shutdown_signal()
        if self._agent_running:
            _interrupt_agent_for_signal(self.agent, signum)
        # Prefer app.exit() over raising KeyboardInterrupt: a KBI from a signal handler
        # lands in a pt Task ("Unhandled exception in event loop" + "Press ENTER to
        # continue..."); call_soon_threadsafe lets the loop unwind normally.
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            _loop = getattr(_app, "loop", None)
            if _loop is not None:
                _loop.call_soon_threadsafe(_app.exit)
                return  # clean unwind — no traceback, no ENTER pause
        except Exception:
            pass
        raise KeyboardInterrupt()  # fallback for non-prompt_toolkit contexts

    def _tui_print_startup(self):
        """Startup output: light-mode probe, banner, advisories, resume/welcome lines, tips."""
        from cli import _accent_hex, _detect_light_mode
        with suppress(Exception):  # light-mode probe before pt grabs the tty (cached)
            _detect_light_mode()
        # Scroll the cursor to the last row so banner, responses and prompt pin to the bottom.
        with suppress(Exception):
            _term_lines = shutil.get_terminal_size().lines
            if _term_lines > 2:
                print("\n" * (_term_lines - 1), end="", flush=True)

        self.show_banner()
        self._show_security_advisories()
        self._show_browser_backend_notice()
        self._show_pending_writes_notice()

        # First-run: an unconfigured install routes into provider onboarding instead of
        # a chat that spins ~30s and fails with a provider-specific error. TTY only. A
        # configured profile whose credential is benched or signed out gets the reason instead.
        try:
            self._maybe_offer_first_run_setup()
        except Exception:
            logger.debug("first-run setup offer failed", exc_info=True)

        if self._resumed and self._preload_resumed_session():
            self._display_resumed_history()

        _welcome_skin = None  # stays None when the skin engine failed
        _welcome_text = "Welcome to Hermes Agent! Type your message or /help for commands."
        _welcome_color = "#FFF8DC"
        try:
            from hermes_cli.skin_engine import get_active_skin
            _welcome_skin = get_active_skin()
            _welcome_text = _welcome_skin.get_branding("welcome", _welcome_text)
            _welcome_color = _welcome_skin.get_color("banner_text", _welcome_color)
        except Exception:
            pass
        self._console_print(f"[{_welcome_color}]{_welcome_text}[/]")

        self._tui_startup_prewarm_and_warnings(_welcome_skin)
        self._print_random_tip()

        self._tui_startup_background_maintenance()
        # Before the background preload is folded in (at agent init), show the REQUESTED names.
        _skills_for_line = self.preloaded_skills or list(self._preload_skills_requested or [])
        if _skills_for_line and not self._startup_skills_line_shown:
            self._console_print(f"[bold {_accent_hex()}]Activated skills:[/] {', '.join(_skills_for_line)}")
            self._startup_skills_line_shown = True
        self._console_print()

    def _tui_startup_prewarm_and_warnings(self, _welcome_skin):
        """Idle-window prewarms (picker cache, agent runtime imports) plus the redaction-off and OpenClaw-residue banners."""
        # Warm the /model picker cache off-thread (else its first open blocks ~1-2s).
        with suppress(Exception):
            from hermes_cli.model_switch_providers import prewarm_picker_cache_async
            prewarm_picker_cache_async()

        # Pre-import the agent runtime (~1.5s: run_agent + OpenAI SDK) off-thread; the import
        # lock makes an early submit block on the remaining work rather than redo it.
        # Skipped when Termux defers agent startup on purpose.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            def _prewarm_agent_runtime() -> None:
                try:
                    import run_agent  # noqa: F401  (imports model_tools + tool registry)
                    import openai  # noqa: F401
                except Exception:
                    logger.debug("agent runtime pre-import failed", exc_info=True)

            threading.Thread(target=_prewarm_agent_runtime, name="agent-runtime-prewarm", daemon=True).start()

        # Redaction is ON by default; be loud when the operator turned it off.
        with suppress(Exception):
            # The redactor snapshots its state at import time so any toggle now won't affect the running
            # process — we just want the operator to see that they're running without the safety net. See
            # #17691.
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            if _redact_raw.lower() not in {"1", "true", "yes", "on"}:
                self._console_print(
                    "[bold red]⚠  Secret redaction is DISABLED[/] "
                    f"(HERMES_REDACT_SECRETS={_redact_raw}). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set "
                    "[cyan]security.redact_secrets: true[/] in config.yaml "
                    "to re-enable."
                )
        # One-time banner when ~/.openclaw/ is left over from a migration.
        try:
            from agent.onboarding import (
                OPENCLAW_RESIDUE_FLAG, detect_openclaw_residue, is_seen, mark_seen, openclaw_residue_hint_cli,
            )
            if not is_seen(self.config, OPENCLAW_RESIDUE_FLAG) and detect_openclaw_residue():
                try:
                    _resid_color = _welcome_skin.get_color("banner_dim", "#B8860B")
                except Exception:
                    _resid_color = "#B8860B"
                self._console_print(f"[{_resid_color}]{openclaw_residue_hint_cli()}[/]")
                try:
                    from hermes_cli.config import get_config_path as _get_cfg_path_resid
                    mark_seen(_get_cfg_path_resid(), OPENCLAW_RESIDUE_FLAG)
                except Exception:
                    pass  # banner fires again next session
        except Exception:
            pass

    def _tui_startup_background_maintenance(self):
        """Best-effort startup passes: curator skill maintenance, personal + org skill sync.

        Off the main thread: the curator's deterministic pass snapshots and prunes the whole
        skills tree (a due weekly pass held the prompt for 6 minutes on a large library), and
        the sync pulls can hit the network. The REPL must never wait on housekeeping."""
        threading.Thread(target=self._run_startup_maintenance, name="startup-maintenance", daemon=True).start()

    def _run_startup_maintenance(self):
        with suppress(Exception):
            from agent.curator import maybe_run_curator
            maybe_run_curator(
                idle_for_seconds=float("inf"),  # CLI startup = fully idle
                on_summary=lambda msg: self._console_print(f"[dim #6b7684]💾 {msg}[/]"),
            )

        # Skill sync (personal, then org-shared): inert unless the access gate is open
        # and a sync base URL is configured. The org pull is gated on a real org role on
        # the token (only issued for multi-member orgs), so a solo account never hits
        # the network here. Both fail-quiet.
        try:
            from tools.skills_sync_client import maybe_pull_skills
            from tools.skills_sync_client_org import maybe_pull_org_skills
        except Exception:
            return
        for pull in (maybe_pull_skills, maybe_pull_org_skills):
            with suppress(Exception):
                pull()

    def _tui_build_application(self, layout, kb, style):
        """Construct the prompt_toolkit Application for the REPL."""
        from cli import CLI_CONFIG, EditingMode, _STEADY_CURSOR, _select_classic_cli_pt_output
        _cpr_disabled_output = _select_classic_cli_pt_output(sys.stdout)

        # Kitty placeholders encode the image id in exact foreground RGB, so the whole app
        # runs 24-bit there. ColorDepth is imported lazily for tests that stub prompt_toolkit.
        extra_kw = {}
        if pet_render.supports_kitty_placeholders():
            from prompt_toolkit.output import ColorDepth

            extra_kw["color_depth"] = ColorDepth.DEPTH_24_BIT
        if _cpr_disabled_output is not None:
            extra_kw["output"] = _cpr_disabled_output
        if _STEADY_CURSOR is not None:
            extra_kw["cursor"] = _STEADY_CURSOR
        if EditingMode is not None:
            # Vi editing mode when display.vim_mode is on.
            # EMACS is prompt_toolkit's own default, so non-opted-in behaviour is unchanged.
            extra_kw["editing_mode"] = EditingMode.VI if self._vim_mode else EditingMode.EMACS
        return Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            # 0 (default) avoids fighting terminal auto-scroll in non-fullscreen mode.
            refresh_interval=float(CLI_CONFIG.get("display", {}).get("cli_refresh_interval", 0)),
            # Erase the bottom chrome on exit instead of freezing a copy into scrollback.
            # Without this, prompt_toolkit's render_as_done teardown repaints the chrome one last time and
            # leaves it stranded above the exit summary — so a dead status bar + empty prompt sit between
            # the conversation transcript and the "Resume this session" block, and stack with the next
            # session's UI on resume (#38252). The actual conversation transcript is printed through
            # patch_stdout into normal scrollback and is unaffected; only the managed chrome is erased.
            # Applies to every exit path (/exit, /quit, EOF, Ctrl+C).
            erase_when_done=True,
            **extra_kw,
        )

    def _tui_install_signal_handlers(self):
        """SIGTERM/SIGHUP -> graceful shutdown; Windows absorbs SIGINT (see body)."""
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, self._tui_signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, self._tui_signal_handler)

            # Windows: absorb SIGINT. Win32 delivers spurious CTRL_C_EVENT when children spawn
            # from background threads, which would unwind app.run() mid-turn. Real Ctrl+C is
            # bound by prompt_toolkit. Never call agent.interrupt() here (fake user message).
            if sys.platform == "win32":
                _signal.signal(_signal.SIGINT, lambda signum, frame: None)
        except Exception:
            pass  # restricted environments

    def _tui_stdin_usable(self) -> bool:
        """Validate fd 0 before prompt_toolkit starts; on macOS fall back to a select() loop when kqueue can't watch it (uv-managed Python)."""
        try:
            os.fstat(0)
        except OSError:
            print(
                "Error: stdin (fd 0) is not available.\n"
                "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
            )
            return False
        if sys.platform == "darwin":
            import selectors as _selectors
            try:
                if hasattr(_selectors, "KqueueSelector"):
                    _kq = _selectors.KqueueSelector()
                    try:
                        _kq.register(0, _selectors.EVENT_READ)
                        _kq.unregister(0)
                    finally:
                        _kq.close()
            except (OSError, ValueError, KeyError):
                import asyncio as _aio_probe

                class _SelectEventLoopPolicy(_aio_probe.DefaultEventLoopPolicy):
                    def new_event_loop(self):
                        return _aio_probe.SelectorEventLoop(_selectors.SelectSelector())

                _aio_probe.set_event_loop_policy(_SelectEventLoopPolicy())
        return True

    def _tui_shutdown(self):
        """Teardown after the app exits: interrupt agent, stop voice/pet, persist + close session, cleanup, exit summary."""
        from cli import _DIM, _RST, _cprint, _invoke_interrupted_session_end, _run_cleanup, set_approval_callback, set_secret_capture_callback, set_sudo_password_callback
        self._should_exit = True
        self._pet_stop_anim()
        # Without this line the terminal sits silent through the whole cleanup window.
        with suppress(Exception):
            print(f"{_DIM}Shutting down… (finalizing session){_RST}", flush=True)
        if self.agent and self._agent_running:
            with suppress(Exception):
                request_hard_interrupt(self.agent)
        if self._voice_recorder:
            with suppress(Exception):
                self._voice_recorder.shutdown()
            self._voice_recorder = None
        with suppress(Exception):
            from tools.voice_mode import cleanup_temp_recordings
            cleanup_temp_recordings()
        from agent.vault_backends.unlock import (lock as _vault_lock, set_code_prompt_callback,
                                                 set_save_login_prompt_callback, set_unlock_prompt_callback)
        for _unset in (set_sudo_password_callback, set_approval_callback, set_secret_capture_callback,
                       set_unlock_prompt_callback, set_save_login_prompt_callback, set_code_prompt_callback):
            _unset(None)
        _vault_lock()  # session tokens for external password managers die with the session
        # On SIGHUP/SIGTERM the agent thread may be reaped before its own persistence runs.
        self._persist_active_session_before_close()

        if self._session_db and self.agent:
            try:
                self._session_db.end_session(self.agent.session_id, "cli_close")
            except (Exception, KeyboardInterrupt) as e:
                logger.debug("Could not close session in DB: %s", e)
            if not self._delete_session_on_exit:
                # Drop the empty row of a start-and-quit session so /resume stays clean.
                try:
                    self._discard_session_if_empty(self.agent.session_id)
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not prune empty session: %s", e)
            else:
                # /exit --delete: remove transcripts + SQLite history.
                try:
                    _sid = self.agent.session_id
                    if self._session_db.delete_session(_sid, sessions_dir=get_hermes_home() / "sessions"):
                        _cprint(f"  {_DIM}✓ Session {_escape(_sid)} deleted{_RST}")
                    else:
                        _cprint(f"  {_DIM}✗ Session {_escape(_sid)} not found for deletion{_RST}")
                except (Exception, KeyboardInterrupt) as e:
                    logger.debug("Could not delete session on exit: %s", e)
        # run_conversation() fires on_session_end on normal completion; only fire here mid-turn.
        if self.agent and self._agent_running:
            _invoke_interrupted_session_end(self.agent, self.agent.session_id, "shutdown")
        _run_cleanup()
        self._print_exit_summary()
        self._release_active_session()
