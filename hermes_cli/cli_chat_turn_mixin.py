"""chat() and its per-turn phases: image routing, staging, agent thread, interrupt monitor, rendering.

Mixin bound onto ``HermesCLI`` via the MRO. cli.py-internal symbols are imported LAZILY
inside each method — importing ``cli`` at module load time would be a cycle.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time

from pathlib import Path
from rich import box as rich_box
from rich.panel import Panel
from typing import Optional


class CLIChatTurnMixin:
    """chat() and its per-turn phase helpers."""

    def chat(
        self,
        message,
        images: list = None,
        voice_input: bool = False,
        *,
        origin_work_id: str = "",
        work_generation: int = 0,
        work_delivery_id: str = "",
        work_claim_id: str = "",
    ) -> Optional[str]:
        """Run one user turn; returns the agent's response, or None on error.

        Input typed while the agent runs goes to ``_interrupt_queue`` (separate from
        ``_pending_input`` so process_loop and the interrupt monitor never compete); an
        interrupting message is re-queued as the next turn. ``voice_input`` gates the
        concise voice-response prefix.

        Args: message: The user's message (str or multimodal content list) images: Optional list of Path
        objects for attached images voice_input: True when the message came from voice transcription (gates
        the concise voice-response prefix, #65827)
        """
        from cli import ChatConsole, _ChatTurn, _DIM, _RST, _accent_hex, _cprint, set_secret_capture_callback
        # Single-query and direct chat callers do not go through run().
        set_secret_capture_callback(self._secret_capture_callback)
        # Reset per turn; only a real interrupt flips it, so early returns leave it False.
        self._last_turn_interrupted = False

        if not self._ensure_runtime_credentials():
            return None

        turn_route = self._resolve_turn_agent_config(message)
        if turn_route["signature"] != self._active_agent_route_signature:
            self.agent = None
        if self.agent is None:
            _cprint(f"{_DIM}Initializing agent...{_RST}")
        if not self._init_agent(model_override=turn_route["model"], runtime_override=turn_route["runtime"],
                                request_overrides=turn_route.get("request_overrides")):
            return None
        agent = self.agent
        if agent is None:
            return None
        message = self._chat_route_images(message, images)

        if isinstance(message, str):
            message, blocked = self._chat_expand_context_references(message)
            if blocked is not None:
                return blocked
            # Lone surrogates (rich-text clipboard paste) crash the OpenAI SDK's JSON serialization.
            from agent.message_sanitization import _sanitize_surrogates
            message = _sanitize_surrogates(message)

        self._chat_stage_user_message(agent, message)

        ChatConsole().print(f"[{_accent_hex()}]{'─' * 40}[/]")
        print(flush=True)

        turn = _ChatTurn(
            origin_work_id=origin_work_id,
            work_generation=work_generation,
            work_delivery_id=work_delivery_id,
            work_claim_id=work_claim_id,
        )
        try:
            self._reset_stream_state()
            # Not part of _reset_stream_state: must persist across intermediate turn
            # boundaries (tool-calling loops), reset once per user turn.
            self._reasoning_shown_this_turn = False
            self._chat_setup_turn_audio(turn, message, voice_input)
            # Per-prompt elapsed timer — frozen when the agent thread finishes.
            self._prompt_start_time = time.time()
            self._prompt_duration = 0.0
            # Daemon: closing the terminal tab (SIGHUP) must not be kept alive by it.
            agent_thread = threading.Thread(target=self._chat_run_agent, args=(turn, message), daemon=True)
            agent_thread.start()
            interrupt_msg = self._chat_monitor_agent_thread(turn, agent_thread)
            self._chat_settle_turn(turn)
            if isinstance(turn.result, dict) and turn.result.get("waiting_on_delegates"):
                return None
            return self._chat_render_turn(turn, agent_thread, interrupt_msg)
        except Exception as e:
            if origin_work_id:
                from cli import _InternalContinuation, _retry_failed_closeout_continuation

                _retry_failed_closeout_continuation(
                    _InternalContinuation(
                        message,
                        origin_work_id,
                        work_generation,
                        work_delivery_id,
                        work_claim_id,
                    )
                )
            print(f"Error: {e}")
            return None
        finally:
            self._chat_release_turn_audio(turn)

    def _chat_release_turn_audio(self, turn):
        """Every exit path: stop the thinking sound, send the TTS sentinel, cut TTS only if abnormal."""
        from cli import logger
        if turn.thinking_started:
            try:
                from tools.voice_mode import stop_thinking_sound
                stop_thinking_sound()
            except Exception:
                pass
        # Safety-net sentinel for exception paths that skipped _chat_settle_turn's; a
        # duplicate is harmless (stream_tts_to_speaker exits on the first None).
        # stop_event only on abnormal exit: after a normal drain it would race the
        # playback worker and cut the final sentence mid-audio.
        if turn.text_queue is not None:
            try:
                turn.text_queue.put_nowait(None)
            except Exception:
                pass
        if turn.stop_event is not None and not turn.tts_normal_exit:
            logger.info("TTS CUT: exception finally block setting stop_event")
            turn.stop_event.set()
        if turn.tts_thread is not None and turn.tts_thread.is_alive():
            turn.tts_thread.join(timeout=5)

    def _chat_expand_context_references(self, message: str):
        """Expand ``@file:``/``@diff``/``@folder:`` references.

        Returns ``(message, blocked)``; ``blocked`` is the refusal text to return instead
        of running the turn when injection was refused, else None.
        """
        from cli import _DIM, _RST, _cprint
        if "@" not in message:
            return message, None
        try:
            from agent.context_references import preprocess_context_references
            from agent.model_metadata import get_model_context_length
            _ctx_len = get_model_context_length(
                self.model, base_url=self.base_url or "", api_key=self.api_key or "",
                provider=self.provider or "",
                config_context_length=getattr(self.agent, "_config_context_length", None) if self.agent else None)
            _ctx_result = preprocess_context_references(message, cwd=os.getcwd(), context_length=_ctx_len)
            if _ctx_result.expanded or _ctx_result.blocked:
                if _ctx_result.references:
                    _cprint(f"  {_DIM}[@ context: {len(_ctx_result.references)} ref(s), "
                            f"{_ctx_result.injected_tokens} tokens]{_RST}")
                for w in _ctx_result.warnings:
                    _cprint(f"  {_DIM}⚠ {w}{_RST}")
                if _ctx_result.blocked:
                    return message, ("\n".join(_ctx_result.warnings) or "Context injection refused.")
                message = _ctx_result.message
        except Exception as e:
            logging.debug("@ context reference expansion failed: %s", e)
        return message, None

    def _chat_route_images(self, message, images):
        """Attach images natively (vision model) or pre-describe them as text; returns the message to send.

        "native" → OpenAI-style content parts (adapters translate per provider); "text" →
        vision_analyze each image and prepend the description. Decision: agent/image_routing.py.
        """
        from cli import _DIM, _RST, _cprint, _split_model_config_default
        if not images:
            return message
        text = message if isinstance(message, str) else ""
        try:
            from agent.image_routing import build_native_content_parts, decide_image_input_mode
            from hermes_cli.config import load_config

            _img_model = (_split_model_config_default(self.model)[0]
                          if isinstance(self.model, dict) else str(self.model or ""))
            _img_provider = (_split_model_config_default(self.provider)[1]
                             if isinstance(self.provider, dict) else str(self.provider or ""))
            _img_mode = decide_image_input_mode(
                _img_provider.strip(), _img_model.strip(), load_config(),
                requested_provider=(self.requested_provider or "").strip(),
            )
        except Exception as _img_exc:
            logging.debug("image_routing decision failed, defaulting to text: %s", _img_exc)
            _img_mode = "text"

        if _img_mode == "native":
            try:
                _img_str_paths = [str(p) for p in images]
                _parts, _skipped = build_native_content_parts(text, _img_str_paths)
                if _skipped:
                    _cprint(f"  {_DIM}⚠ skipped {len(_skipped)} unreadable image path(s){_RST}")
                if any(p.get("type") == "image_url" for p in _parts):
                    _img_names = ", ".join(Path(p).name for p in _img_str_paths)
                    _cprint(f"  {_DIM}📎 attaching {len(images)} image(s) natively "
                            f"(model supports vision): {_img_names}{_RST}")
                    return _parts
                # All images unreadable — fall back to text enrichment.
            except Exception as _img_exc:
                logging.warning("native image attach failed, falling back to text: %s", _img_exc)
        return self._preprocess_images_with_vision(text, images)

    def _chat_stage_user_message(self, agent, message):
        """Append the staged user dict to the transcript under the agent's persist lock."""
        # Copy before appending: mutating ``agent._session_messages`` in this UI-only step
        # would expose a duplicate-prone snapshot to terminal-close persistence.
        if self.conversation_history is getattr(agent, "_session_messages", None):
            self.conversation_history = list(self.conversation_history)
        # Clear the prior turn's override before exposing the new staged input: a shutdown
        # before the worker prologue would otherwise persist old API-local text as this message.
        import contextlib
        from agent.message_metadata import stamp_message_timestamp

        persist_lock = getattr(agent, "_session_persist_lock", None)
        with persist_lock if persist_lock is not None else contextlib.nullcontext():
            agent._persist_user_message_idx = None
            agent._persist_user_message_override = None
            agent._persist_user_message_timestamp = None
            staged_user_message = stamp_message_timestamp({"role": "user", "content": message})
            agent._pending_cli_user_message = staged_user_message
            self.conversation_history.append(staged_user_message)

    def _chat_setup_turn_audio(self, turn, message, voice_input):
        """Arm the full-duplex listener and the streaming-TTS pipeline for this turn (voice mode only)."""
        from cli import _ACCENT, _RST, _STREAM_PAD, _cprint, datetime
        # Continuous voice mode: arm the mic NOW (utterance-submit), not at TTS playback —
        # it spans generation (speech interrupts the turn) and playback (speech cuts TTS)
        # and disarms itself when the turn is done. See _voice_full_duplex_listener.
        if self._voice_mode and self._voice_continuous:
            self._voice_last_tts_text = ""
            threading.Thread(target=self._voice_full_duplex_listener, daemon=True).start()

        # Streaming TTS: any working provider speaks sentence-by-sentence as tokens arrive.
        if self._voice_tts:
            try:
                from tools.tts_tool import _import_sounddevice, check_tts_requirements
                from tools.tts_tool_speaker import stream_tts_to_speaker
                _import_sounddevice()
                turn.use_streaming_tts = check_tts_requirements()
            except Exception:
                pass

        if turn.use_streaming_tts:
            turn.text_queue = queue.Queue()
            turn.stop_event = threading.Event()

            # display_callback only when token streaming is off: with streaming on,
            # _stream_delta already renders the text and this would print it twice.
            def display_callback(sentence: str):
                if not turn.box_opened:
                    turn.box_opened = True
                    label = " ⚕ Hermes "
                    if self.show_timestamps:
                        label = f"{label}{datetime.now().strftime(self.timestamp_format)} "
                    w = self._scrollback_box_width(getattr(self.console, "width", 80))
                    fill = w - 2 - self._status_bar_display_width(label)
                    _cprint(f"\n{_ACCENT}╭─{label}{'─' * max(fill - 1, 0)}╮{_RST}")
                _cprint(f"{_STREAM_PAD}{sentence.rstrip()}")

            turn.tts_thread = threading.Thread(
                target=stream_tts_to_speaker, args=(turn.text_queue, turn.stop_event, self._voice_tts_done),
                kwargs={"display_callback": None if self.streaming_enabled else display_callback},
                daemon=True,
            )
            turn.tts_thread.start()
            # Barge-in paths (voice key, full-duplex listener) cut playback via this event.
            self._voice_tts_stop = turn.stop_event

            def stream_callback(delta: str):
                turn.text_queue.put(delta)
                # Track what is being spoken so a playback-phase barge capture can be
                # checked against it (echo guard).
                self._voice_last_tts_text = (self._voice_last_tts_text or "") + delta
            turn.stream_callback = stream_callback

        # API-call-local only — run_conversation persists the original clean user message.
        if voice_input and isinstance(message, str):
            turn.voice_prefix = ("[Voice input — respond concisely and conversationally, "
                                 "2-3 sentences max. No code blocks or markdown.] ")

    def _chat_run_agent(self, turn, message):
        """Agent-thread body: bind per-thread callbacks/approval key, prepend one-shot notes, run the turn."""
        from cli import (
            _prepend_note_to_message, set_approval_callback, set_secret_capture_callback,
            set_sudo_password_callback,
        )
        # terminal_tool callbacks are thread-local: run()'s registration is invisible here.
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        set_secret_capture_callback(self._secret_capture_callback)
        # Bind the approval session key so ``is_current_session_yolo_enabled()`` resolves
        # against the same key ``/yolo`` toggles under (``enable_session_yolo(self.session_id)``).
        try:
            from tools.approval_context import reset_current_session_key, set_current_session_key
            _approval_session_token = set_current_session_key(self.session_id or "default")
        except Exception:
            reset_current_session_key = None  # type: ignore[assignment]
            _approval_session_token = None
        agent_message = turn.voice_prefix + message if turn.voice_prefix else message
        # One-shot /model and /reload-skills notes; _prepend_note_to_message also handles
        # multimodal content-part lists (string concat raised TypeError with an image).
        for _note_attr in ("_pending_model_switch_note", "_pending_skills_reload_note"):
            _note = getattr(self, _note_attr, None)
            if _note:
                agent_message = _prepend_note_to_message(agent_message, _note)
                setattr(self, _note_attr, None)
        # Barged mid-speech (VAD or record key)? Tell the model it was cut off.
        from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE, take_speech_interrupted
        if take_speech_interrupted():
            agent_message = _prepend_note_to_message(agent_message, SPEECH_INTERRUPTED_NOTE)
        _moa_cfg = getattr(self, "_pending_moa_config", None)
        self._pending_moa_config = None
        # Notes and voice prefix are API-local: the staged input stays the durable transcript
        # value so a close-path marker follows the same dict instead of a second user row.
        _persist_clean_user_message = message if (turn.voice_prefix or agent_message != message) else None
        _one_turn_model_restore = getattr(self, "_pending_one_turn_model_restore", None)
        self._pending_one_turn_model_restore = None
        conversation_kwargs = {
            "user_message": agent_message,
            "conversation_history": self.conversation_history[:-1],
            "stream_callback": turn.stream_callback,
            "task_id": self.session_id,
            "persist_user_message": _persist_clean_user_message,
            "moa_config": _moa_cfg,
        }
        if turn.origin_work_id:
            conversation_kwargs.update(
                {
                    "origin_work_id": turn.origin_work_id,
                    "work_generation": turn.work_generation,
                    "work_delivery_id": turn.work_delivery_id,
                    "work_claim_id": turn.work_claim_id,
                }
            )
        try:
            turn.result = self.agent.run_conversation(**conversation_kwargs)
            if getattr(self, "_pending_moa_disable_after_turn", False):
                _restore = getattr(self, "_pending_moa_restore_model", None) or {}
                for _key, _value in _restore.items():
                    if _value is not None:
                        setattr(self, _key, _value)
                self.agent = None
                self._pending_moa_restore_model = None
                self._pending_moa_disable_after_turn = False
        except Exception as exc:
            logging.error("run_conversation raised: %s", exc, exc_info=True)
            _summary = getattr(self.agent, '_summarize_api_error', lambda e: str(e)[:300])(exc)
            turn.result = {
                "final_response": f"Error: {_summary}", "messages": [], "api_calls": 0,
                "completed": False, "failed": True, "error": _summary,
            }
        finally:
            if _one_turn_model_restore:
                self._restore_model_runtime_snapshot(_one_turn_model_restore)
            # Credit notices paint cleanly above the prompt here, not behind streamed output.
            self._flush_credit_notices()
            # A reused thread must never hold stale references to a disposed CLI instance.
            try:
                set_sudo_password_callback(None)
                set_approval_callback(None)
                set_secret_capture_callback(None)
            except Exception:
                pass
            # Unbind the per-turn key; ``_session_yolo`` state itself persists across turns.
            if _approval_session_token is not None and reset_current_session_key is not None:
                try:
                    reset_current_session_key(_approval_session_token)
                except Exception:
                    pass

    def _chat_monitor_agent_thread(self, turn, agent_thread):
        """Poll the interrupt queue while the agent thread runs; returns the interrupting message (or None)."""
        from cli import _hermes_home, logger
        # Ambient "thinking" blips in voice mode; skipped per-blip while TTS speaks, the mic
        # records or a barge capture is live. voice.thinking_sound gates it (default on).
        if self._voice_mode:
            try:
                from tools.voice_mode import start_thinking_sound
                turn.thinking_started = start_thinking_sound(should_play=lambda: (
                    self._voice_tts_done.is_set() and not self._voice_recording
                    and not self._voice_barge_capture.is_set()))
            except Exception:
                turn.thinking_started = False

        interrupt_msg = None
        while agent_thread.is_alive():
            try:
                interrupt_msg = self._interrupt_queue.get(timeout=0.1)
            except queue.Empty:
                # Flush the StdoutProxy buffer: it otherwise only flushes on input-triggered
                # renderer passes, so on macOS the CLI looks frozen until the user types.
                # Force prompt_toolkit to flush any pending stdout output from the agent thread. (#1624)
                self._invalidate(min_interval=0.15)
                continue
            if not interrupt_msg:
                continue
            # With a clarify question active, Enter routes to the clarify queue; anything
            # landing here is a race — don't interrupt, park it as the next turn.
            if self._clarify_state or self._clarify_freetext:
                try:
                    self._pending_input.put(interrupt_msg)
                except Exception:
                    pass
                interrupt_msg = None
                continue
            print("\n⚡ New message detected, interrupting...")
            if turn.stop_event is not None:
                turn.stop_event.set()
            self.agent.interrupt(interrupt_msg)
            # Modal prompts gate input until reset — otherwise the CLI freezes after an
            # interrupt until the prompt's own timeout.
            self._clear_active_overlays_for_interrupt()
            # Debug log to file (stdout may be devnull under redirect_stdout).
            try:
                with open(_hermes_home / "interrupt_debug.log", "a", encoding="utf-8") as _f:
                    _f.write(f"{time.strftime('%H:%M:%S')} interrupt fired: msg={str(interrupt_msg)[:60]!r}, "
                             f"children={len(self.agent._active_children)}, "
                             f"parent._interrupt={self.agent._interrupt_requested}\n")
                    for _ci, _ch in enumerate(self.agent._active_children):
                        _f.write(f"  child[{_ci}]._interrupt={_ch._interrupt_requested}\n")
            except Exception:
                pass
            break

        if interrupt_msg is not None:
            # After an interrupt the agent may take seconds to clean up (kill
            # subprocess, persist). Poll instead of a blocking join so another
            # interrupt (Ctrl+C sets _should_exit) or a stuck agent can't freeze
            # us; the thread is daemon and dies on process exit regardless.
            for _ in range(50):  # 50 * 0.2s = 10s max
                agent_thread.join(timeout=0.2)
                if not agent_thread.is_alive() or self._should_exit:
                    break
            if agent_thread.is_alive():
                logger.warning(
                    "Agent thread still alive after interrupt "
                    "(thread %s). Daemon thread will be cleaned up "
                    "on exit.",
                    agent_thread.ident,
                )
        else:
            agent_thread.join(timeout=30)  # should be done already; guard edge cases
        return interrupt_msg

    def _chat_settle_turn(self, turn):
        """After the agent thread ends: freeze timers, flush streams, drain TTS, sync history/session id."""
        if self._prompt_start_time is not None:
            self._prompt_duration = max(0.0, time.time() - self._prompt_start_time)
            self._prompt_start_time = None
        self._last_turn_finished_at = time.time()  # status bar idle time
        # AsyncOpenAI clients bound to the worker's now-closed loop would crash
        # prompt_toolkit's loop from __del__ on GC.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass
        self._flush_stream()
        if turn.use_streaming_tts and turn.text_queue is not None:
            turn.text_queue.put(None)  # end-of-text sentinel
            if turn.tts_thread is not None:
                turn.tts_thread.join(timeout=120)
                # A timed-out join leaves tts_normal_exit False so the release path's
                # stop_event kills the runaway worker.
                turn.tts_normal_exit = not turn.tts_thread.is_alive()
        # Drain the StdoutProxy buffer so tool/status lines render ABOVE the response
        # box; the sleep lets the renderer paint before we draw.
        sys.stdout.flush()
        time.sleep(0.15)
        if turn.result:
            self.conversation_history = turn.result.get("messages", self.conversation_history)
        # Mid-turn auto-compression continues in a child session: sync so /status, /resume,
        # titling and the exit summary target the live child, not the ended parent.
        if (self.agent and getattr(self.agent, "session_id", None)
                and self.agent.session_id != self.session_id):
            self._transfer_session_yolo(self.session_id, self.agent.session_id)
            self.session_id = self.agent.session_id
            self._write_terminal_breadcrumb()
            self._pending_title = None

    def _chat_render_turn(self, turn, agent_thread, interrupt_msg):
        """Post-turn display: errors, interrupt marker, reasoning/response panels, bell, re-queues.

        Returns the response text.
        """
        from cli import _DIM, _RST, _cprint, _suspend_output_history
        response = turn.result.get("final_response", "") if turn.result else ""
        # "failed"/"partial" with an empty final_response: no usable answer.
        if turn.result and (turn.result.get("failed") or turn.result.get("partial")) and not response:
            response = f"Error: {turn.result.get('error', 'Unknown error')}"
            # Stop continuous voice on persistent errors (e.g. 429) — else error→record→error loops.
            if self._voice_continuous:
                self._voice_continuous = False
                _cprint(f"\n{_DIM}Continuous voice mode stopped due to error.{_RST}")

        pending_message, _show_interrupt_marker = self._chat_resolve_interrupt(
            turn, agent_thread, interrupt_msg, response)

        self._chat_print_reasoning_box(turn)
        self._chat_print_response_panel(turn, response)

        # History suppressed so the marker is never recorded in _OUTPUT_HISTORY
        # (appending it to `response` duplicated it on redraw).
        if _show_interrupt_marker:
            with _suspend_output_history():
                _cprint(f"\n{_DIM}── [Interrupted — processing new message] ──{_RST}")
        # Focus view: "⋯ N tool lines hidden" after the answer; resets the counter.
        try:
            self._emit_focus_recovery_line()
        except Exception:
            pass

        self._ring_bell(context="turn complete")  # propagates over SSH
        if turn.result and not turn.result.get("completed") and not turn.result.get("interrupted"):
            _api_calls = turn.result.get("api_calls", 0)
            _max_iter = getattr(self.agent, "max_iterations", 500)
            if _api_calls >= _max_iter:
                _cprint(
                    f"\n{_DIM}⚠ Iteration budget reached ({_api_calls}/{_max_iter}) — "
                    f"response may be incomplete{_RST}"
                )

        # Batch TTS unless streaming TTS already spoke the response.
        if self._voice_tts and response and not turn.use_streaming_tts:
            self._voice_speak_response_async(response)

        # Re-queue the interrupt message (plus any that arrived meanwhile) as the next
        # prompt. Only reached in busy_input_mode == "interrupt"; "queue" mode routes
        # Enter straight to _pending_input.
        if pending_message:
            all_parts = [pending_message]
            while not self._interrupt_queue.empty():
                try:
                    extra = self._interrupt_queue.get_nowait()
                    if extra:
                        all_parts.append(extra)
                except queue.Empty:
                    break
            combined = "\n".join(all_parts)
            preview = combined[:50] + ("..." if len(combined) > 50 else "")
            if len(all_parts) > 1:
                print(f"\n⚡ Sending {len(all_parts)} messages after interrupt: '{preview}'")
            else:
                print(f"\n⚡ Sending after interrupt: '{preview}'")
            self._pending_input.put(combined)

        # A /steer the agent finished before absorbing becomes the next user turn.
        _leftover_steer = turn.result.get("pending_steer") if turn.result else None
        if _leftover_steer:
            preview = _leftover_steer[:60] + ("..." if len(_leftover_steer) > 60 else "")
            print(f"\n⏩ Delivering leftover /steer as next turn: '{preview}'")
            self._pending_input.put(_leftover_steer)

        return response

    def _chat_resolve_interrupt(self, turn, agent_thread, interrupt_msg, response):
        """Return ``(pending_message, show_marker)``; clears a stale agent interrupt flag.

        The marker is printed separately after the response Panel (history suppressed)
        so a terminal redraw never duplicates it.
        """
        pending_message = None
        _show_interrupt_marker = False
        _interrupted_this_turn = bool(turn.result and turn.result.get("interrupted"))
        # Post-turn hooks (e.g. goal continuation) skip themselves on a user-cancelled turn.
        self._last_turn_interrupted = _interrupted_this_turn
        if _interrupted_this_turn:
            pending_message = turn.result.get("interrupt_message") or interrupt_msg
            _show_interrupt_marker = bool(response and pending_message)
        elif interrupt_msg:
            # agent.interrupt() fired but the result doesn't acknowledge it (racy): either
            # the thread had passed its last interrupt check so finalize_turn() never saw
            # the flag, or the 10s post-interrupt wait expired and `result` is None. The
            # user's message must NOT be dropped — re-queue it as the next turn.
            pending_message = interrupt_msg
            # An interrupt landing after finalize_turn()'s clear_interrupt() leaves a stale
            # flag that would abort the NEXT turn at its first check. Clear it — but ONLY if
            # the thread exited: on an abandoned thread the flag is what eventually unwinds
            # the wedged tool.
            try:
                if (not agent_thread.is_alive() and self.agent
                        and getattr(self.agent, "_interrupt_requested", False)):
                    self.agent.clear_interrupt()
            except Exception:
                pass
        return pending_message, _show_interrupt_marker

    def _chat_print_reasoning_box(self, turn):
        """Collapsed reasoning box when show_reasoning is on and streaming did not already show it."""
        from cli import _DIM, _RST, _cprint
        # _reasoning_shown_this_turn, not _reasoning_stream_started: the latter resets at
        # intermediate turn boundaries (tool loops) and re-rendered the box after the answer.
        if self.show_reasoning and turn.result and not self._reasoning_shown_this_turn:
            reasoning = turn.result.get("last_reasoning")
            if reasoning:
                w = self._scrollback_box_width()
                r_label = " Reasoning "
                r_top = f"{_DIM}┌─{r_label}{'─' * max(w - 3 - len(r_label), 0)}┐{_RST}"
                r_bot = f"{_DIM}└{'─' * (w - 2)}┘{_RST}"
                # First 10 lines unless the user opted into /reasoning full.
                lines = reasoning.strip().splitlines()
                if len(lines) > 10 and not self.reasoning_full:
                    display_reasoning = "\n".join(lines[:10])
                    display_reasoning += f"\n{_DIM}  ... ({len(lines) - 10} more lines — /reasoning full to show){_RST}"
                else:
                    display_reasoning = reasoning.strip()
                _cprint(f"\n{r_top}\n{_DIM}{display_reasoning}{_RST}\n{r_bot}")

    def _chat_print_response_panel(self, turn, response):
        """Response box (close TTS-drawn box / post-stream transform / Rich Panel), then billing CTA."""
        from cli import (
            ChatConsole, _ACCENT, _RST, _cprint, _maybe_remap_for_light_mode, _post_stream_transform_output,
            _render_final_assistant_content,
        )
        if response and not (turn.result and turn.result.get("response_previewed", False)):
            try:
                from hermes_cli.skin_engine import get_active_skin
                _skin = get_active_skin()
                label = _skin.get_branding("response_label", "⚕ Hermes")
                _resp_color = _maybe_remap_for_light_mode(_skin.get_color("response_border", "#CD7F32"))
                _resp_text = _maybe_remap_for_light_mode(_skin.get_color("banner_text", "#FFF8DC"))
            except Exception:
                label = "⚕ Hermes"
                _resp_color = _maybe_remap_for_light_mode("#CD7F32")
                _resp_text = _maybe_remap_for_light_mode("#FFF8DC")

            is_error_response = turn.result and (turn.result.get("failed") or turn.result.get("partial"))
            already_streamed = self._stream_started and self._stream_box_opened and not is_error_response
            if turn.use_streaming_tts and turn.box_opened and not is_error_response:
                # Text already printed sentence-by-sentence; just close the box.
                _cprint(f"\n{_ACCENT}╰{'─' * (self._scrollback_box_width() - 2)}╯{_RST}")
            elif already_streamed:
                # _flush_stream() already closed the streamed box; a post-stream transform
                # hook shows a suffix for append-only changes, else the full replacement.
                _post_stream_text = _post_stream_transform_output(response, turn.result)
                if _post_stream_text.strip():
                    _cprint(_post_stream_text)
            else:
                ChatConsole().print(Panel(
                    _render_final_assistant_content(response, mode=self.final_response_markdown),
                    title=f"[{_resp_color} bold]{label}[/]", title_align="left", border_style=_resp_color,
                    style=_resp_text, box=rich_box.HORIZONTALS, padding=(1, 0),
                    width=self._scrollback_box_width(),
                ))

            # Billing CTA pins the single action (Nous → /topup, others → billing page) so it
            # stays visible instead of scrolling away inside the response prose.
            if turn.result and turn.result.get("failure_reason") == "billing":
                _bb = turn.result.get("billing_block") or {}
                if _bb.get("is_nous"):
                    _cta_lines = ["Run [bold]/topup[/] to add credits, or "
                                  "[bold]/subscription[/] to change plan."]
                else:
                    _url = _bb.get("billing_url")
                    _cta_lines = [f"Add credits with {_bb.get('provider_label') or 'your provider'}"
                                  + (f": [bold]{_url}[/]" if _url else ".")]
                _cta_lines.append("Or switch providers with [bold]/model <model> --provider <provider>[/].")
                try:
                    ChatConsole().print(Panel(
                        "\n".join(_cta_lines), title="[#CD7F32 bold]⚡ Out of credits[/]",
                        title_align="left", border_style="#CD7F32", box=rich_box.HORIZONTALS,
                        padding=(1, 4), width=self._scrollback_box_width(),
                    ))
                except Exception:
                    pass
