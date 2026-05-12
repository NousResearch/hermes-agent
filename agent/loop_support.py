"""LoopSupportMixin -- iteration handling, display, and API call wrapping.

Encapsulates max-iteration handling, interruptible API calls, verbose
printing, status/warning emission, and activity tracking.  Previously
inline in AIAgent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import os
import re
import sys
import threading
import time

# Re-export utilities for mixin method access
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class LoopSupportMixin:
    """Mixin providing loop iteration support methods for AIAgent.

    All methods use `self` to access agent state and are designed to be
    mixed into the AIAgent class.
    """

    def _safe_print(self, *args, **kwargs):
        """Print that silently handles broken pipes / closed stdout.

        In headless environments (systemd, Docker, nohup) stdout may become
        unavailable mid-session.  A raw ``print()`` raises ``OSError`` which
        can crash cron jobs and lose completed work.

        Internally routes through ``self._print_fn`` (default: builtin
        ``print``) so callers such as the CLI can inject a renderer that
        handles ANSI escape sequences properly (e.g. prompt_toolkit's
        ``print_formatted_text(ANSI(...))``) without touching this method.
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass


    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed when actively streaming tokens.

        Pass ``force=True`` for error/warning messages that should always be
        shown even during streaming playback (TTS or display).

        During tool execution (``_executing_tools`` is True), printing is
        allowed even with stream consumers registered because no tokens
        are being streamed at that point.

        After the main response has been delivered and the remaining tool
        calls are post-response housekeeping (``_mute_post_response``),
        all non-forced output is suppressed.

        ``suppress_status_output`` is a stricter CLI automation mode used by
        parseable single-query flows such as ``hermes chat -q``. In that mode,
        all status/diagnostic prints routed through ``_vprint`` are suppressed
        so stdout stays machine-readable.
        """
        if getattr(self, "suppress_status_output", False):
            return
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)


    def _should_start_quiet_spinner(self) -> bool:
        """Return True when quiet-mode spinner output has a safe sink.

        In headless/stdio-protocol environments, a raw spinner with no custom
        ``_print_fn`` falls back to ``sys.stdout`` and can corrupt protocol
        streams such as ACP JSON-RPC. Allow quiet spinners only when either:
        - output is explicitly rerouted via ``_print_fn``; or
        - stdout is a real TTY.
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False


    def _should_emit_quiet_tool_messages(self) -> bool:
        """Return True when quiet-mode tool summaries should print directly.

        Quiet mode is used by both the interactive CLI and embedded/library
        callers. The CLI may still want compact progress hints when no callback
        owns rendering. Embedded/library callers, on the other hand, expect
        quiet mode to be truly silent.
        """
        return (
            self.quiet_mode
            and not self.tool_progress_callback
            and getattr(self, "platform", "") == "cli"
        )


    def _emit_status(self, message: str) -> None:
        """Emit a lifecycle status message to both CLI and gateway channels.

        CLI users see the message via ``_vprint(force=True)`` so it is always
        visible regardless of verbose/quiet mode.  Gateway consumers receive
        it through ``status_callback("lifecycle", ...)``.

        This helper never raises — exceptions are swallowed so it cannot
        interrupt the retry/fallback logic.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)


    def _emit_warning(self, message: str) -> None:
        """Emit a user-visible warning through the same status plumbing.

        Unlike debug logs, these warnings are meant for degraded side paths
        such as auxiliary compression or memory flushes where the main turn can
        continue but the user needs to know something important failed.
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("warn", message)
            except Exception:
                logger.debug("status_callback error in _emit_warning", exc_info=True)

    # Headers we capture from the dying stream's HTTP response so post-mortem
    # diagnosis can answer "which CF edge / which OpenRouter downstream
    # provider / which request id".  Lowercased; httpx returns CIMultiDict.
    _STREAM_DIAG_HEADERS = (
        "cf-ray",
        "cf-cache-status",
        "x-openrouter-provider",
        "x-openrouter-model",
        "x-openrouter-id",
        "x-request-id",
        "x-vercel-id",
        "via",
        "server",
        "x-forwarded-for",
    )

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Request a summary when max iterations are reached. Returns the final response text."""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # Build API messages, stripping internal-only fields
            # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                self._copy_reasoning_content_for_api(msg, api_msg)
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Same safety net as the main loop: repair tool-call/result
            # pairing before asking for a final summary.  Compression and
            # session resume can leave a tool result whose parent assistant
            # tool_call was summarized away; Responses API rejects that as
            # "No tool call found for function call output".
            api_messages = self._sanitize_api_messages(api_messages)

            # Same safety net as the main loop: drop thinking-only assistant
            # turns so Anthropic-family providers don't 400 the summary call.
            api_messages = self._drop_thinking_only_and_merge_users(api_messages)

            summary_extra_body = {}
            try:
                from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
            except Exception:
                _fixed_temperature_for_model = None
                _OMIT_TEMP = None
            _raw_summary_temp = (
                _fixed_temperature_for_model(self.model, self.base_url)
                if _fixed_temperature_for_model is not None
                else None
            )
            _omit_summary_temperature = _raw_summary_temp is _OMIT_TEMP
            _summary_temperature = None if _omit_summary_temperature else _raw_summary_temp
            _is_nous = "nousresearch" in self._base_url_lower
            # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
            # Mirror ChatCompletionsTransport.build_kwargs() so the summary path
            # — which calls chat.completions.create() directly without going
            # through the transport — sends the same shape the transport does.
            _is_lmstudio_summary = (
                (self.provider or "").strip().lower() == "lmstudio"
                and self._supports_reasoning_extra_body()
            )
            _lm_reasoning_effort: str | None = (
                self._resolve_lmstudio_summary_reasoning_effort()
                if _is_lmstudio_summary else None
            )
            if not _is_lmstudio_summary and self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                _ct_sum = self._get_transport()
                _cnr_sum = _ct_sum.normalize_response(summary_response)
                final_response = (_cnr_sum.content or "").strip()
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if _summary_temperature is not None:
                    summary_kwargs["temperature"] = _summary_temperature
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                if _lm_reasoning_effort is not None:
                    summary_kwargs["reasoning_effort"] = _lm_reasoning_effort

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences and (
                    (self.provider or "").strip().lower() == "openrouter"
                    or self._is_openrouter_url()
                ):
                    summary_extra_body["provider"] = provider_preferences

                # Pareto Code router plugin — model-gated. Same shape as
                # the main-loop emission so summary calls on
                # openrouter/pareto-code respect the user's coding-score floor.
                if (
                    self.model == "openrouter/pareto-code"
                    and (
                        (self.provider or "").strip().lower() == "openrouter"
                        or self._is_openrouter_url()
                    )
                    and self.openrouter_min_coding_score is not None
                    and self.openrouter_min_coding_score != ""
                ):
                    try:
                        _ps = float(self.openrouter_min_coding_score)
                    except (TypeError, ValueError):
                        _ps = None
                    if _ps is not None and 0.0 <= _ps <= 1.0:
                        summary_extra_body["plugins"] = [
                            {"id": "pareto-router", "min_coding_score": _ps}
                        ]

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    _tsum = self._get_transport()
                    _ant_kw = _tsum.build_kwargs(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _summary_result = _tsum.normalize_response(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_summary_result.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)
                    _summary_result = self._get_transport().normalize_response(summary_response)
                    final_response = (_summary_result.content or "").strip()

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # Retry summary generation
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    _ct_retry = self._get_transport()
                    _cnr_retry = _ct_retry.normalize_response(retry_response)
                    final_response = (_cnr_retry.content or "").strip()
                elif self.api_mode == "anthropic_messages":
                    _tretry = self._get_transport()
                    _ant_kw2 = _tretry.build_kwargs(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_result = _tretry.normalize_response(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_result.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if _summary_temperature is not None:
                        summary_kwargs["temperature"] = _summary_temperature
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if _lm_reasoning_effort is not None:
                        summary_kwargs["reasoning_effort"] = _lm_reasoning_effort
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)
                    _retry_result = self._get_transport().normalize_response(summary_response)
                    final_response = (_retry_result.content or "").strip()

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response


    def _interruptible_api_call(self, api_kwargs: dict):
        """
        Run the API call in a background thread so the main conversation loop
        can detect interrupts without waiting for the full HTTP round-trip.

        Each worker thread gets its own OpenAI client instance. Interrupts only
        close that worker-local client, so retries and other requests never
        inherit a closed transport.

        Includes a stale-call detector: if no response arrives within the
        configured timeout, the connection is killed and an error raised so
        the main retry loop can try again with backoff / credential rotation /
        provider fallback.
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(
                        reason="codex_stream_request",
                        api_kwargs=api_kwargs,
                    )
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                elif self.api_mode == "bedrock_converse":
                    # Bedrock uses boto3 directly — no OpenAI client needed.
                    # normalize_converse_response produces an OpenAI-compatible
                    # SimpleNamespace so the rest of the agent loop can treat
                    # bedrock responses like chat_completions responses.
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        normalize_converse_response,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise
                    result["response"] = normalize_converse_response(raw_response)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(
                        reason="chat_completion_request",
                        api_kwargs=api_kwargs,
                    )
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        # ── Stale-call timeout (mirrors streaming stale detector) ────────
        # Non-streaming calls return nothing until the full response is
        # ready.  Without this, a hung provider can block for the full
        # httpx timeout (default 1800s) with zero feedback.  The stale
        # detector kills the connection early so the main retry loop can
        # apply richer recovery (credential rotation, provider fallback).
        _stale_timeout = self._compute_non_stream_stale_timeout(
            api_kwargs.get("messages", [])
        )

        _call_start = time.time()
        self._touch_activity("waiting for non-streaming API response")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _poll_count = 0
        while t.is_alive():
            t.join(timeout=0.3)
            _poll_count += 1

            # Touch activity every ~30s so the gateway's inactivity
            # monitor knows we're alive while waiting for the response.
            if _poll_count % 100 == 0:  # 100 × 0.3s = 30s
                _elapsed = time.time() - _call_start
                self._touch_activity(
                    f"waiting for non-streaming response ({int(_elapsed)}s elapsed)"
                )

            # Stale-call detector: kill the connection if no response
            # arrives within the configured timeout.
            _elapsed = time.time() - _call_start
            if _elapsed > _stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Non-streaming API call stale for %.0fs (threshold %.0fs). "
                    "model=%s context=~%s tokens. Killing connection.",
                    _elapsed, _stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_elapsed)}s "
                    f"(non-streaming, model: {api_kwargs.get('model', 'unknown')}). "
                    f"Aborting call."
                )
                try:
                    if self.api_mode == "anthropic_messages":
                        self._anthropic_client.close()
                        self._rebuild_anthropic_client()
                    else:
                        rc = request_client_holder.get("client")
                        if rc is not None:
                            self._close_request_openai_client(rc, reason="stale_call_kill")
                except Exception:
                    pass
                self._touch_activity(
                    f"stale non-streaming call killed after {int(_elapsed)}s"
                )
                # Wait briefly for the thread to notice the closed connection.
                t.join(timeout=2.0)
                if result["error"] is None and result["response"] is None:
                    result["error"] = TimeoutError(
                        f"Non-streaming API call timed out after {int(_elapsed)}s "
                        f"with no response (threshold: {int(_stale_timeout)}s)"
                    )
                break

            if self._interrupt_requested:
                # Force-close the in-flight worker-local HTTP connection to stop
                # token generation without poisoning the shared client used to
                # seed future retries.
                try:
                    if self.api_mode == "anthropic_messages":
                        self._anthropic_client.close()
                        self._rebuild_anthropic_client()
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    @staticmethod


    def _wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
        """Word-wrap verbose tool output to fit the terminal width.

        Splits *text* on existing newlines and wraps each line individually,
        preserving intentional line breaks (e.g. pretty-printed JSON).
        Returns a ready-to-print string with *label* on the first line and
        continuation lines indented.
        """
        import shutil as _shutil
        import textwrap as _tw
        cols = _shutil.get_terminal_size((120, 24)).columns
        wrap_width = max(40, cols - len(indent))
        out_lines: list[str] = []
        for raw_line in text.split("\n"):
            if len(raw_line) <= wrap_width:
                out_lines.append(raw_line)
            else:
                wrapped = _tw.wrap(raw_line, width=wrap_width,
                                   break_long_words=True,
                                   break_on_hyphens=False)
                out_lines.extend(wrapped or [raw_line])
        body = ("\n" + indent).join(out_lines)
        return f"{indent}{label}{body}"


    def _touch_activity(self, desc: str) -> None:
        """Update the last-activity timestamp and description (thread-safe)."""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc
