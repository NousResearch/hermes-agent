"""StreamingMixin -- streaming API call methods extracted from AIAgent.

Encapsulates all streaming-related logic: delta delivery, codex streaming,
diagnostic capture, stale timeout computation, and stream interruption.

These methods were previously inline in the AIAgent class (run_agent.py).
They use `self` to access agent state and are designed to be mixed into
AIAgent via multiple inheritance.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.memory_manager import sanitize_context
from agent.model_metadata import is_local_endpoint, save_context_length
from tools.interrupt import set_interrupt as _set_interrupt

# Re-export all utilities from agent.utils so streaming methods can access
# module-level names that were originally local imports in AIAgent.
from agent.utils import *  # noqa: F401,F403

logger = logging.getLogger(__name__)


class StreamingMixin:
    """Mixin providing streaming API call methods for AIAgent.

    All methods use `self` to access agent state (log_prefix, platform,
    model, etc.) and are designed to be mixed into the AIAgent class.
    """

    @staticmethod
    def _stream_diag_init() -> Dict[str, Any]:
        """Return a fresh per-attempt diagnostic dict.

        Mutated in-place by the streaming functions and read from the retry
        block when a stream dies.  Lives on ``request_client_holder`` so it
        survives across the closure boundary.
        """
        return {
            "started_at": time.time(),
            "first_chunk_at": None,
            "chunks": 0,
            "bytes": 0,
            "headers": {},
            "http_status": None,
        }


    def _stream_diag_capture_response(
        self, diag: Dict[str, Any], http_response: Any
    ) -> None:
        """Snapshot interesting headers + HTTP status from the live stream.

        Called once at stream open (before iterating chunks) so the metadata
        survives even if the stream dies before any chunk arrives.  Failures
        are swallowed — diag is best-effort.
        """
        if http_response is None or not isinstance(diag, dict):
            return
        try:
            diag["http_status"] = getattr(http_response, "status_code", None)
        except Exception:
            pass
        try:
            headers = getattr(http_response, "headers", None) or {}
            captured: Dict[str, str] = {}
            for name in self._STREAM_DIAG_HEADERS:
                try:
                    val = headers.get(name)
                    if val:
                        # Truncate single-value to keep log lines bounded.
                        captured[name] = str(val)[:120]
                except Exception:
                    continue
            diag["headers"] = captured
        except Exception:
            pass

    @staticmethod


    def _flatten_exception_chain(error: BaseException) -> str:
        """Return a compact ``Outer(msg) <- Inner(msg) <- ...`` rendering.

        OpenAI SDK wraps httpx errors as ``APIConnectionError`` /
        ``APIError`` and only the wrapper's class is visible at the catch
        site — but the underlying ``RemoteProtocolError`` /
        ``ConnectError`` / ``ReadError`` is what tells us WHY the stream
        died.  Walks ``__cause__`` then ``__context__`` (deduped, max 4
        deep) to surface the chain in one line.
        """
        seen: List[BaseException] = []
        link: Optional[BaseException] = error
        while link is not None and len(seen) < 4:
            if link in seen:
                break
            seen.append(link)
            nxt = getattr(link, "__cause__", None) or getattr(
                link, "__context__", None
            )
            if nxt is None or nxt is link:
                break
            link = nxt
        parts: List[str] = []
        for e in seen:
            msg = str(e).strip().replace("\n", " ")
            if len(msg) > 140:
                msg = msg[:140] + "…"
            parts.append(f"{type(e).__name__}({msg})" if msg else type(e).__name__)
        return " <- ".join(parts) if parts else type(error).__name__


    def _log_stream_retry(
        self,
        *,
        kind: str,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a transient stream-drop and retry to ``agent.log``.

        Always logs a structured WARNING so users have a breadcrumb regardless
        of UI verbosity.  Subagents in particular benefit because their
        retries no longer spam the parent's terminal — but the file log keeps
        full detail (provider, error class, attempt, base_url, subagent_id).

        When *diag* is provided (the per-attempt stream-diagnostic dict from
        ``_stream_diag_init``), the WARNING also captures upstream headers
        (cf-ray, x-openrouter-provider, x-openrouter-id), HTTP status, bytes
        streamed before the drop, and elapsed time on the dying attempt.
        These are the breadcrumbs needed to answer "is one CF edge / one
        downstream provider responsible, or is it random across runs?"
        """
        try:
            try:
                _summary = self._summarize_api_error(error)
            except Exception:
                _summary = str(error)
            if _summary and len(_summary) > 240:
                _summary = _summary[:240] + "…"

            # Inner-cause chain (httpx errors hide under openai.APIError).
            try:
                _chain = self._flatten_exception_chain(error)
            except Exception:
                _chain = type(error).__name__

            # Per-attempt counters and upstream headers.
            _now = time.time()
            _bytes = 0
            _chunks = 0
            _elapsed = 0.0
            _ttfb = None
            _headers_repr = "-"
            _http_status = "-"
            if isinstance(diag, dict):
                try:
                    _bytes = int(diag.get("bytes") or 0)
                    _chunks = int(diag.get("chunks") or 0)
                    _started = float(diag.get("started_at") or _now)
                    _elapsed = max(0.0, _now - _started)
                    _first = diag.get("first_chunk_at")
                    if _first is not None:
                        _ttfb = max(0.0, float(_first) - _started)
                    headers = diag.get("headers") or {}
                    if isinstance(headers, dict) and headers:
                        _headers_repr = " ".join(
                            f"{k}={v}" for k, v in headers.items()
                        )
                    if diag.get("http_status") is not None:
                        _http_status = str(diag.get("http_status"))
                except Exception:
                    pass

            logger.warning(
                "Stream %s on attempt %s/%s — retrying. "
                "subagent_id=%s depth=%s provider=%s base_url=%s "
                "error_type=%s error=%s "
                "chain=%s "
                "http_status=%s bytes=%d chunks=%d elapsed=%.2fs ttfb=%s "
                "upstream=[%s]",
                kind,
                attempt,
                max_attempts,
                getattr(self, "_subagent_id", None) or "-",
                getattr(self, "_delegate_depth", 0),
                self.provider or "-",
                self.base_url or "-",
                type(error).__name__,
                _summary,
                _chain,
                _http_status,
                _bytes,
                _chunks,
                _elapsed,
                f"{_ttfb:.2f}s" if _ttfb is not None else "-",
                _headers_repr,
                extra={"mid_tool_call": mid_tool_call},
            )
        except Exception:
            logger.debug("stream-retry log emit failed", exc_info=True)


    def _emit_stream_drop(
        self,
        *,
        error: BaseException,
        attempt: int,
        max_attempts: int,
        mid_tool_call: bool,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a single user-visible line for a stream drop+retry.

        Both top-level agents and subagents announce drops in the UI — the
        parent prefixes subagent lines with ``[subagent-N]`` via ``log_prefix``
        so they're easy to attribute.  All cases also write a structured
        WARNING to ``agent.log`` via :meth:`_log_stream_retry` with the full
        diagnostic detail (subagent_id, provider, base_url, error_type,
        cf-ray, x-openrouter-provider, bytes/chunks, elapsed) for post-hoc
        analysis.

        The user-visible status line is intentionally compact: provider,
        error class, attempt N/M, plus ``after Xs`` when the stream dropped
        mid-flight.  Full diagnostic detail goes to ``agent.log`` only —
        ``hermes logs --level WARNING | grep "Stream drop"`` to inspect.
        """
        kind = "drop mid tool-call" if mid_tool_call else "drop"
        self._log_stream_retry(
            kind=kind,
            error=error,
            attempt=attempt,
            max_attempts=max_attempts,
            mid_tool_call=mid_tool_call,
            diag=diag,
        )
        provider = self.provider or "provider"
        # Compose a brief "after Xs" suffix when we have timing data — helps
        # the user distinguish "couldn't connect" (0s) from "died after 30s
        # of streaming" (likely upstream idle-kill or proxy timeout).
        _suffix = ""
        if isinstance(diag, dict):
            try:
                started = diag.get("started_at")
                if started is not None:
                    _suffix = f" after {max(0.0, time.time() - float(started)):.1f}s"
            except Exception:
                pass
        try:
            self._emit_status(
                f"⚠️ {provider} stream {kind} ({type(error).__name__}){_suffix} "
                f"— reconnecting, retry {attempt}/{max_attempts}"
            )
            self._touch_activity(
                f"stream retry {attempt}/{max_attempts} "
                f"after {type(error).__name__}"
            )
        except Exception:
            pass


    def _emit_auxiliary_failure(self, task: str, exc: BaseException) -> None:
        """Surface a compact warning for failed auxiliary work."""
        try:
            detail = self._summarize_api_error(exc)
        except Exception:
            detail = str(exc)
        detail = (detail or exc.__class__.__name__).strip()
        if len(detail) > 220:
            detail = detail[:217].rstrip() + "..."
        self._emit_warning(f"⚠ Auxiliary {task} failed: {detail}")


    def _compute_non_stream_stale_timeout(self, messages: list[dict[str, Any]]) -> float:
        """Compute the effective non-stream stale timeout for this request."""
        stale_base, uses_implicit_default = self._resolved_api_call_stale_timeout_base()
        base_url = getattr(self, "_base_url", None) or self.base_url or ""
        if uses_implicit_default and base_url and is_local_endpoint(base_url):
            return float("inf")

        est_tokens = sum(len(str(v)) for v in messages) // 4
        if est_tokens > 100_000:
            return max(stale_base, 600.0)
        if est_tokens > 50_000:
            return max(stale_base, 450.0)
        return stale_base


    def _reset_stream_delivery_tracking(self) -> None:
        """Reset tracking for text delivered during the current model response."""
        # Flush any benign partial-tag tail held by the think scrubber
        # first (#17924): an innocent '<' at the end of the stream that
        # turned out not to be a tag prefix should reach the UI.  Then
        # flush the context scrubber.  Order matters — the think
        # scrubber's output feeds into the context scrubber's state.
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        if think_scrubber is not None:
            think_tail = think_scrubber.flush()
            if think_tail:
                # Route the tail through the context scrubber too so a
                # memory-context span straddling the final boundary is
                # still caught.
                ctx_scrubber = getattr(self, "_stream_context_scrubber", None)
                if ctx_scrubber is not None:
                    think_tail = ctx_scrubber.feed(think_tail)
                if think_tail:
                    callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                    for cb in callbacks:
                        try:
                            cb(think_tail)
                        except Exception:
                            pass
                    self._record_streamed_assistant_text(think_tail)
        # Flush any benign partial-tag tail held by the context scrubber so it
        # reaches the UI before we clear state for the next model call.  If
        # the scrubber is mid-span, flush() drops the orphaned content.
        scrubber = getattr(self, "_stream_context_scrubber", None)
        if scrubber is not None:
            tail = scrubber.flush()
            if tail:
                callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
                for cb in callbacks:
                    try:
                        cb(tail)
                    except Exception:
                        pass
                self._record_streamed_assistant_text(tail)
        self._current_streamed_assistant_text = ""


    def _record_streamed_assistant_text(self, text: str) -> None:
        """Accumulate visible assistant text emitted through stream callbacks."""
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        return bool(streamed) and streamed == visible_content


    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # If a tool iteration set the break flag, prepend a single paragraph
        # break before the first real text delta.  This prevents the original
        # problem (text concatenation across tool boundaries) without stacking
        # blank lines when multiple tool iterations run back-to-back.
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
            prepended_break = True
        else:
            prepended_break = False
        if isinstance(text, str):
            # Suppress reasoning/thinking blocks via the stateful
            # scrubber (#17924).  Earlier versions ran _strip_think_blocks
            # per-delta here, which destroyed downstream state machines
            # when a tag was split across deltas (e.g. MiniMax-M2.7
            # sends '<think>' and its content as separate deltas —
            # regex case 2 erased the first delta, so the CLI/gateway
            # state machine never saw the open tag and leaked the
            # reasoning content as regular response text).
            think_scrubber = getattr(self, "_stream_think_scrubber", None)
            if think_scrubber is not None:
                text = think_scrubber.feed(text or "")
            else:
                # Defensive: legacy callers without the scrubber attribute.
                text = self._strip_think_blocks(text or "")
            # Then feed through the stateful context scrubber so memory-context
            # spans split across chunks cannot leak to the UI (#5719).
            scrubber = getattr(self, "_stream_context_scrubber", None)
            if scrubber is not None:
                text = scrubber.feed(text)
            else:
                # Defensive: legacy callers without the scrubber attribute.
                text = sanitize_context(text)
            # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
            if not prepended_break and not getattr(
                self, "_current_streamed_assistant_text", ""
            ):
                text = text.lstrip("\n")
        if not text:
            return
        callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
        delivered = False
        for cb in callbacks:
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        if delivered:
            self._record_streamed_assistant_text(text)


    def _has_stream_consumers(self) -> bool:
        """Return True if any streaming consumer is registered."""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )


    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """Execute one streaming Responses API request and return the final response."""
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        # Accumulate streamed text so we can recover if get_final_response()
        # returns empty output (e.g. chatgpt.com backend-api sends
        # response.incomplete instead of response.completed).
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            if self._interrupt_requested:
                raise InterruptedError("Agent interrupted before Codex stream retry")
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        self._touch_activity("receiving stream response")
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # Fire callbacks on text content deltas (suppress during tool calls)
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # Track tool calls to suppress text streaming
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # Fire reasoning callbacks
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                        # Collect completed output items — some backends
                        # (chatgpt.com/backend-api/codex) stream valid items
                        # via response.output_item.done but the SDK's
                        # get_final_response() returns an empty output list.
                        elif event_type == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None:
                                collected_output_items.append(done_item)
                        # Log non-completed terminal events for diagnostics
                        elif event_type in ("response.incomplete", "response.failed"):
                            resp_obj = getattr(event, "response", None)
                            status = getattr(resp_obj, "status", None) if resp_obj else None
                            incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                            logger.warning(
                                "Codex Responses stream received terminal event %s "
                                "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                                event_type, status, incomplete_details,
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # PATCH: ChatGPT Codex backend streams valid output items
                    # but get_final_response() can return an empty output list.
                    # Backfill from collected items or synthesize from deltas.
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise


    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Fallback path for stream completion edge cases on Codex-style Responses backends."""
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._get_transport().preflight_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # Compatibility shim for mocks or providers that still return a concrete response.
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        collected_output_items: list = []
        collected_text_deltas: list = []
        try:
            for event in stream_or_response:
                self._touch_activity("receiving stream response")
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

                # Collect output items and text deltas for backfill
                if event_type == "response.output_item.done":
                    done_item = getattr(event, "item", None)
                    if done_item is None and isinstance(event, dict):
                        done_item = event.get("item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif event_type in ("response.output_text.delta",):
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(delta)

                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    # Backfill empty output from collected stream events
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            terminal_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex fallback stream: backfilled %d output items",
                                len(collected_output_items),
                            )
                        elif collected_text_deltas:
                            assembled = "".join(collected_text_deltas)
                            terminal_response.output = [SimpleNamespace(
                                type="message", role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex fallback stream: synthesized from %d deltas (%d chars)",
                                len(collected_text_deltas), len(assembled),
                            )
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")


    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """Streaming variant of _interruptible_api_call for real-time token delivery.

        Handles all three api_modes:
        - chat_completions: stream=True on OpenAI-compatible endpoints
        - anthropic_messages: client.messages.stream() via Anthropic SDK
        - codex_responses: delegates to _run_codex_stream (already streaming)

        Fires stream_delta_callback and _stream_callback for each text token.
        Tool-call turns suppress the callback — only text-only final responses
        stream to the consumer.  Returns a SimpleNamespace that mimics the
        non-streaming response shape so the rest of the agent loop is unchanged.

        Falls back to _interruptible_api_call on provider errors indicating
        streaming is not supported.
        """
        if self._interrupt_requested:
            raise InterruptedError("Agent interrupted before streaming API call")

        if self.api_mode == "codex_responses":
            # Codex streams internally via _run_codex_stream. The main dispatch
            # in _interruptible_api_call already calls it; we just need to
            # ensure on_first_delta reaches it. Store it on the instance
            # temporarily so _run_codex_stream can pick it up.
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        # Bedrock Converse uses boto3's converse_stream() with real-time delta
        # callbacks — same UX as Anthropic and chat_completions streaming.
        if self.api_mode == "bedrock_converse":
            result = {"response": None, "error": None}
            first_delta_fired = {"done": False}
            deltas_were_sent = {"yes": False}

            def _fire_first():
                if not first_delta_fired["done"] and on_first_delta:
                    first_delta_fired["done"] = True
                    try:
                        on_first_delta()
                    except Exception:
                        pass

            def _bedrock_call():
                try:
                    from agent.bedrock_adapter import (
                        _get_bedrock_runtime_client,
                        invalidate_runtime_client,
                        is_stale_connection_error,
                        stream_converse_with_callbacks,
                    )
                    region = api_kwargs.pop("__bedrock_region__", "us-east-1")
                    api_kwargs.pop("__bedrock_converse__", None)
                    client = _get_bedrock_runtime_client(region)
                    try:
                        raw_response = client.converse_stream(**api_kwargs)
                    except Exception as _bedrock_exc:
                        # Evict the cached client on stale-connection failures
                        # so the outer retry loop builds a fresh client/pool.
                        if is_stale_connection_error(_bedrock_exc):
                            invalidate_runtime_client(region)
                        raise

                    def _on_text(text):
                        _fire_first()
                        self._fire_stream_delta(text)
                        deltas_were_sent["yes"] = True

                    def _on_tool(name):
                        _fire_first()
                        self._fire_tool_gen_started(name)

                    def _on_reasoning(text):
                        _fire_first()
                        self._fire_reasoning_delta(text)

                    result["response"] = stream_converse_with_callbacks(
                        raw_response,
                        on_text_delta=_on_text if self._has_stream_consumers() else None,
                        on_tool_start=_on_tool,
                        on_reasoning_delta=_on_reasoning if self.reasoning_callback or self.stream_delta_callback else None,
                        on_interrupt_check=lambda: self._interrupt_requested,
                    )
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_bedrock_call, daemon=True)
            t.start()
            while t.is_alive():
                t.join(timeout=0.3)
                if self._interrupt_requested:
                    raise InterruptedError("Agent interrupted during Bedrock API call")
            if result["error"] is not None:
                raise result["error"]
            return result["response"]

        result = {"response": None, "error": None, "partial_tool_names": []}
        request_client_holder = {"client": None, "diag": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # Track if any deltas were fired (for fallback)
        # Wall-clock timestamp of the last real streaming chunk.  The outer
        # poll loop uses this to detect stale connections that keep receiving
        # SSE keep-alive pings but no actual data.
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """Stream a chat completions response."""
            import httpx as _httpx
            # Per-provider / per-model request_timeout_seconds (from config.yaml)
            # wins over the HERMES_API_TIMEOUT env default if the user set it.
            _provider_timeout_cfg = get_provider_request_timeout(self.provider, self.model)
            _base_timeout = (
                _provider_timeout_cfg
                if _provider_timeout_cfg is not None
                else float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            )
            # Read timeout: config wins here too.  Otherwise use
            # HERMES_STREAM_READ_TIMEOUT (default 120s) for cloud providers.
            if _provider_timeout_cfg is not None:
                _stream_read_timeout = _provider_timeout_cfg
            else:
                _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
                # Local providers (Ollama, llama.cpp, vLLM) can take minutes for
                # prefill on large contexts before producing the first token.
                # Auto-increase the httpx read timeout unless the user explicitly
                # overrode HERMES_STREAM_READ_TIMEOUT.
                if _stream_read_timeout == 120.0 and self.base_url and is_local_endpoint(self.base_url):
                    _stream_read_timeout = _base_timeout
                    logger.debug(
                        "Local provider detected (%s) — stream read timeout raised to %.0fs",
                        self.base_url, _stream_read_timeout,
                    )
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request",
                api_kwargs=stream_kwargs,
            )
            # Reset stale-stream timer so the detector measures from this
            # attempt's start, not a previous attempt's last chunk.
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            # Initialize per-attempt stream diagnostics so the retry block can
            # reach for them after the stream dies.  Lives on
            # ``request_client_holder["diag"]`` for closure access.
            _diag = self._stream_diag_init()
            request_client_holder["diag"] = _diag
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            # Capture rate limit headers from the initial HTTP response.
            # The OpenAI SDK Stream object exposes the underlying httpx
            # response via .response before any chunks are consumed.
            self._capture_rate_limits(getattr(stream, "response", None))
            # Snapshot diagnostic headers (cf-ray, x-openrouter-provider, etc.)
            # so they survive even when the stream dies before any chunk
            # arrives.  Best-effort; never raises.
            self._stream_diag_capture_response(_diag, getattr(stream, "response", None))

            # Log OpenRouter response cache status when present.
            self._check_openrouter_cache_status(getattr(stream, "response", None))

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama-compatible endpoints reuse index 0 for every tool call
            # in a parallel batch, distinguishing them only by id.  Track
            # the last seen id per raw index so we can detect a new tool
            # call starting at the same index and redirect it to a fresh slot.
            _last_id_at_idx: dict = {}      # raw_index -> last seen non-empty id
            _active_slot_by_idx: dict = {}  # raw_index -> current slot in tool_calls_acc
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                self._touch_activity("receiving stream response")

                # Update per-attempt diagnostic counters.  Best-effort —
                # failures are swallowed so the streaming hot path is never
                # interrupted by diagnostic accounting.
                try:
                    _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                    if _diag.get("first_chunk_at") is None:
                        _diag["first_chunk_at"] = last_chunk_time["t"]
                    # Approximate byte size from the chunk's repr — exact wire
                    # bytes aren't exposed by the SDK, but len(repr(chunk)) is
                    # a stable proxy for "how much content arrived" that
                    # survives stub provider differences.
                    try:
                        _diag["bytes"] = int(_diag.get("bytes", 0)) + len(repr(chunk))
                    except Exception:
                        pass
                except Exception:
                    pass

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage comes in the final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # Accumulate reasoning content
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # Accumulate text content — fire callback only when no tool calls
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    else:
                        # Tool calls suppress regular content streaming (avoids
                        # displaying chatty "I'll use the tool..." text alongside
                        # tool calls).  But reasoning tags embedded in suppressed
                        # content should still reach the display — otherwise the
                        # reasoning box only appears as a post-response fallback,
                        # rendering it confusingly after the already-streamed
                        # response.  Route suppressed content through the stream
                        # delta callback so its tag extraction can fire the
                        # reasoning display.  Non-reasoning text is harmlessly
                        # suppressed by the CLI's _stream_delta when the stream
                        # box is already closed (tool boundary flush).
                        if self.stream_delta_callback:
                            try:
                                self.stream_delta_callback(delta.content)
                                self._record_streamed_assistant_text(delta.content)
                            except Exception:
                                pass

                # Accumulate tool call deltas — notify display on first name
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama fix: detect a new tool call reusing the same
                        # raw index (different id) and redirect to a fresh slot.
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                # Use assignment, not +=.  Function names are
                                # atomic identifiers delivered complete in the
                                # first chunk (OpenAI spec).  Some providers
                                # (MiniMax M2.7 via NVIDIA NIM) resend the full
                                # name in every chunk; concatenation would
                                # produce "read_fileread_file".  Assignment
                                # (matching the OpenAI Node SDK / LiteLLM /
                                # Vercel AI patterns) is immune to this.
                                entry["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # Fire once per tool when the full name is available
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)
                            # Record the partial tool-call name so the outer
                            # stub-builder can surface a user-visible warning
                            # if streaming dies before this tool's arguments
                            # are fully delivered.  Without this, a stall
                            # during tool-call JSON generation lets the stub
                            # at line ~6107 return `tool_calls=None`, silently
                            # discarding the attempted action.
                            result["partial_tool_names"].append(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # Build mock response matching non-streaming shape
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            has_truncated_tool_args = False
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    arguments = tc["function"]["arguments"]
                    tool_name = tc["function"]["name"] or "?"
                    if arguments and arguments.strip():
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            # Attempt repair before flagging as truncated.
                            # Models like GLM-5.1 via Ollama produce trailing
                            # commas, unclosed brackets, Python None, etc.
                            # Without repair, these hit the truncation handler
                            # and kill the session.  _repair_tool_call_arguments
                            # returns "{}" for unrepairable args, which is far
                            # better than a crashed session.
                            repaired = _repair_tool_call_arguments(arguments, tool_name)
                            if repaired != "{}":
                                # Successfully repaired — use the fixed args
                                arguments = repaired
                            else:
                                # Unrepairable — flag for truncation handling
                                has_truncated_tool_args = True
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=arguments,
                        ),
                    ))

            effective_finish_reason = finish_reason or "stop"
            if has_truncated_tool_args:
                effective_finish_reason = "length"

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=effective_finish_reason,
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """Stream an Anthropic Messages API response.

            Fires delta callbacks for real-time token delivery, but returns
            the native Anthropic Message object from get_final_message() so
            the rest of the agent loop (validation, tool extraction, etc.)
            works unchanged.
            """
            has_tool_use = False

            # Reset stale-stream timer for this attempt
            last_chunk_time["t"] = time.time()
            # Per-attempt diagnostic dict for the retry block to consume.
            _diag = self._stream_diag_init()
            request_client_holder["diag"] = _diag
            # Use the Anthropic SDK's streaming context manager
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                # The Anthropic SDK exposes the raw httpx response on
                # ``stream.response``.  Snapshot diagnostic headers
                # immediately so they survive a stream that dies before the
                # first event.
                try:
                    self._stream_diag_capture_response(
                        _diag, getattr(stream, "response", None)
                    )
                except Exception:
                    pass
                for event in stream:
                    # Update stale-stream timer on every event so the
                    # outer poll loop knows data is flowing.  Without
                    # this, the detector kills healthy long-running
                    # Opus streams after 180 s even when events are
                    # actively arriving (the chat_completions path
                    # already does this at the top of its chunk loop).
                    last_chunk_time["t"] = time.time()
                    self._touch_activity("receiving stream response")

                    # Update per-attempt diagnostic counters (best-effort).
                    try:
                        _diag["chunks"] = int(_diag.get("chunks", 0)) + 1
                        if _diag.get("first_chunk_at") is None:
                            _diag["first_chunk_at"] = last_chunk_time["t"]
                        try:
                            _diag["bytes"] = int(_diag.get("bytes", 0)) + len(repr(event))
                        except Exception:
                            pass
                    except Exception:
                        pass

                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                                    deltas_were_sent["yes"] = True
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # Return the native Anthropic Message for downstream processing
                return stream.get_final_message()

        def _call():
            import httpx as _httpx

            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                for _stream_attempt in range(_max_stream_retries + 1):
                    # Check for interrupt before each retry attempt.  Without
                    # this, /stop closes the HTTP connection (outer poll loop),
                    # but the retry loop opens a FRESH connection — negating the
                    # interrupt entirely.  On slow providers (ollama-cloud) each
                    # retry can block for the full stream-read timeout (120s+),
                    # causing multi-minute delays between /stop and response.
                    if self._interrupt_requested:
                        raise InterruptedError("Agent interrupted before stream retry")
                    try:
                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()
                        return  # success
                    except Exception as e:
                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # If the stream died AFTER some tokens were delivered:
                        # normally we don't retry (the user already saw text,
                        # retrying would duplicate it).  BUT: if a tool call
                        # was in-flight when the stream died, silently aborting
                        # discards the tool call entirely.  In that case we
                        # prefer to retry — the user sees a brief
                        # "reconnecting" marker + duplicated preamble text,
                        # which is strictly better than a failed action with
                        # a "retry manually" message.  Limit this to transient
                        # connection errors (Clawdbot-style narrow gate): no
                        # tool has executed yet within this API call, so
                        # silent retry is safe wrt side-effects.
                        if deltas_were_sent["yes"]:
                            _partial_tool_in_flight = bool(
                                result.get("partial_tool_names")
                            )
                            _is_sse_conn_err_preview = False
                            if not _is_timeout and not _is_conn_err:
                                from openai import APIError as _APIError
                                if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                    _err_lower_preview = str(e).lower()
                                    _SSE_PREVIEW_PHRASES = (
                                        "connection lost",
                                        "connection reset",
                                        "connection closed",
                                        "connection terminated",
                                        "network error",
                                        "network connection",
                                        "terminated",
                                        "peer closed",
                                        "broken pipe",
                                        "upstream connect error",
                                    )
                                    _is_sse_conn_err_preview = any(
                                        phrase in _err_lower_preview
                                        for phrase in _SSE_PREVIEW_PHRASES
                                    )
                            _is_transient = (
                                _is_timeout or _is_conn_err or _is_sse_conn_err_preview
                            )
                            _can_silent_retry = (
                                _partial_tool_in_flight
                                and _is_transient
                                and _stream_attempt < _max_stream_retries
                            )
                            if not _can_silent_retry:
                                # Either no tool call was in-flight (so the
                                # turn was a pure text response — current
                                # stub-with-recovered-text behaviour is
                                # correct), or retries are exhausted, or the
                                # error isn't transient.  Fall through to the
                                # stub path.
                                logger.warning(
                                    "Streaming failed after partial delivery, not retrying: %s", e
                                )
                                result["error"] = e
                                return
                            # Tool call was in-flight AND error is transient:
                            # retry silently.  Clear per-attempt state so the
                            # next stream starts clean.  Fire a "reconnecting"
                            # marker so the user sees why the preamble is
                            # about to be re-streamed.  Structured WARNING is
                            # emitted by ``_emit_stream_drop`` below; no
                            # additional INFO line needed.
                            try:
                                self._fire_stream_delta(
                                    "\n\n⚠ Connection dropped mid tool-call; "
                                    "reconnecting…\n\n"
                                )
                            except Exception:
                                pass
                            # Reset the streamed-text buffer so the retry's
                            # fresh preamble doesn't get double-recorded in
                            # _current_streamed_assistant_text (which would
                            # pollute the interim-visible-text comparison).
                            try:
                                self._reset_stream_delivery_tracking()
                            except Exception:
                                pass
                            # Reset in-memory accumulators so the next
                            # attempt's chunks don't concat onto the dead
                            # stream's partial JSON.
                            result["partial_tool_names"] = []
                            deltas_were_sent["yes"] = False
                            first_delta_fired["done"] = False
                            self._emit_stream_drop(
                                error=e,
                                attempt=_stream_attempt + 2,
                                max_attempts=_max_stream_retries + 1,
                                mid_tool_call=True,
                                diag=request_client_holder.get("diag"),
                            )
                            stale = request_client_holder.get("client")
                            if stale is not None:
                                self._close_request_openai_client(
                                    stale, reason="stream_mid_tool_retry_cleanup"
                                )
                                request_client_holder["client"] = None
                            try:
                                self._replace_primary_openai_client(
                                    reason="stream_mid_tool_retry_pool_cleanup"
                                )
                            except Exception:
                                pass
                            continue

                        # SSE error events from proxies (e.g. OpenRouter sends
                        # {"error":{"message":"Network connection lost."}}) are
                        # raised as APIError by the OpenAI SDK.  These are
                        # semantically identical to httpx connection drops —
                        # the upstream stream died — and should be retried with
                        # a fresh connection.  Distinguish from HTTP errors:
                        # APIError from SSE has no status_code, while
                        # APIStatusError (4xx/5xx) always has one.
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_sse = str(e).lower()
                                _SSE_CONN_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # Transient network / timeout error. Retry the
                            # streaming request with a fresh connection first.
                            if _stream_attempt < _max_stream_retries:
                                self._emit_stream_drop(
                                    error=e,
                                    attempt=_stream_attempt + 2,
                                    max_attempts=_max_stream_retries + 1,
                                    mid_tool_call=False,
                                    diag=request_client_holder.get("diag"),
                                )
                                # Close the stale request client before retry
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # Also rebuild the primary client to purge
                                # any dead connections from the pool.
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                continue
                            # Retries exhausted. Log the final failure with
                            # full diagnostic detail (chain, headers,
                            # bytes/elapsed) via the same helper used for
                            # mid-flight retries — subagent lines get the
                            # ``[subagent-N]`` log_prefix so the parent can
                            # attribute them.
                            self._log_stream_retry(
                                kind="exhausted",
                                error=e,
                                attempt=_max_stream_retries + 1,
                                max_attempts=_max_stream_retries + 1,
                                mid_tool_call=False,
                                diag=request_client_holder.get("diag"),
                            )
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                        else:
                            _err_lower = str(e).lower()
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                self._disable_streaming = True
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Switching to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery: %s",
                                e,
                            )

                        # Propagate the error to the main retry loop instead of
                        # falling back to non-streaming inline.  The main loop has
                        # richer recovery: credential rotation, provider fallback,
                        # backoff, and — for "stream not supported" — will switch
                        # to non-streaming on the next attempt via _disable_streaming.
                        result["error"] = e
                        return
            except InterruptedError as e:
                # The interrupt may be noticed inside the worker thread before
                # the polling loop sees it. Surface it through the normal result
                # channel so callers never miss a fast pre-retry interrupt.
                result["error"] = e
                return
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # Local providers (Ollama, oMLX, llama-cpp) can take 300+ seconds
        # for prefill on large contexts.  Disable the stale detector unless
        # the user explicitly set HERMES_STREAM_STALE_TIMEOUT.
        if _stream_stale_timeout_base == 180.0 and self.base_url and is_local_endpoint(self.base_url):
            _stream_stale_timeout = float("inf")
            logger.debug("Local provider detected (%s) — stale stream timeout disabled", self.base_url)
        else:
            # Scale the stale timeout for large contexts: slow models (like Opus)
            # can legitimately think for minutes before producing the first token
            # when the context is large.  Without this, the stale detector kills
            # healthy connections during the model's thinking phase, producing
            # spurious RemoteProtocolError ("peer closed connection").
            _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            if _est_tokens > 100_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
            elif _est_tokens > 50_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
            else:
                _stream_stale_timeout = _stream_stale_timeout_base

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        _last_heartbeat = time.time()
        _HEARTBEAT_INTERVAL = 30.0  # seconds between gateway activity touches
        while t.is_alive():
            t.join(timeout=0.3)

            # Periodic heartbeat: touch the agent's activity tracker so the
            # gateway's inactivity monitor knows we're alive while waiting
            # for stream chunks.  Without this, long thinking pauses (e.g.
            # reasoning models) or slow prefill on local providers (Ollama)
            # trigger false inactivity timeouts.  The _call thread touches
            # activity on each chunk, but the gap between API call start
            # and first chunk can exceed the gateway timeout — especially
            # when the stale-stream timeout is disabled (local providers).
            _hb_now = time.time()
            if _hb_now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _last_heartbeat = _hb_now
                _waiting_secs = int(_hb_now - last_chunk_time["t"])
                self._touch_activity(
                    f"waiting for stream response ({_waiting_secs}s, no chunks yet)"
                )

            # Detect stale streams: connections kept alive by SSE pings
            # but delivering no real chunks.  Kill the client so the
            # inner retry loop can start a fresh connection.
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # Rebuild the primary client too — its connection pool
                # may hold dead sockets from the same provider outage.
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # Reset the timer so we don't kill repeatedly while
                # the inner thread processes the closure.
                last_chunk_time["t"] = time.time()
                self._touch_activity(
                    f"stale stream detected after {int(_stale_elapsed)}s, reconnecting"
                )

            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        self._anthropic_client.close()
                        self._rebuild_anthropic_client()
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # Streaming failed AFTER some tokens were already delivered to
                # the platform.  Re-raising would let the outer retry loop make
                # a new API call, creating a duplicate message.  Return a
                # partial "stop" response instead so the outer loop treats this
                # turn as complete (no retry, no fallback).
                # Recover whatever content was already streamed to the user.
                # _current_streamed_assistant_text accumulates text fired
                # through _fire_stream_delta, so it has exactly what the
                # user saw before the connection died.
                _partial_text = (
                    getattr(self, "_current_streamed_assistant_text", "") or ""
                ).strip() or None

                # If the stream died while the model was emitting a tool call,
                # the stub below will silently set `tool_calls=None` and the
                # agent loop will treat the turn as complete — the attempted
                # action is lost with no user-facing signal.  Append a
                # human-visible warning to the stub content so (a) the user
                # knows something failed, and (b) the next turn's model sees
                # in conversation history what was attempted and can retry.
                _partial_names = list(result.get("partial_tool_names") or [])
                if _partial_names:
                    _name_str = ", ".join(_partial_names[:3])
                    if len(_partial_names) > 3:
                        _name_str += f", +{len(_partial_names) - 3} more"
                    _warn = (
                        f"\n\n⚠ Stream stalled mid tool-call "
                        f"({_name_str}); the action was not executed. "
                        f"Ask me to retry if you want to continue."
                    )
                    _partial_text = (_partial_text or "") + _warn
                    # Also fire as a streaming delta so the user sees it now
                    # instead of only in the persisted transcript.
                    try:
                        self._fire_stream_delta(_warn)
                    except Exception:
                        pass
                    logger.warning(
                        "Partial stream dropped tool call(s) %s after %s chars "
                        "of text; surfaced warning to user: %s",
                        _partial_names, len(_partial_text or ""), result["error"],
                    )
                else:
                    logger.warning(
                        "Partial stream delivered before error; returning stub "
                        "response with %s chars of recovered content to prevent "
                        "duplicate messages: %s",
                        len(_partial_text or ""),
                        result["error"],
                    )
                _stub_msg = SimpleNamespace(
                    role="assistant", content=_partial_text, tool_calls=None,
                    reasoning_content=None,
                )
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            raise result["error"]
        return result["response"]

    # ── Provider fallback ──────────────────────────────────────────────────

