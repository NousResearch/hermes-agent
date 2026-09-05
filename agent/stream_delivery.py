"""Streaming / interim-message delivery for ``AIAgent``.

Single-writer stream ownership, delta/reasoning hook fan-out, and interim assistant text dedup.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import re
import threading
from typing import Any, Dict, List

from agent.memory_manager import sanitize_context
from agent.message_content import flatten_message_text
from agent.redact import redact_sensitive_text

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class StreamDeliveryMixin:
    """Stream ownership, delta/reasoning hook fan-out and interim-text dedup (see module docstring)."""

    @staticmethod
    def _call_quietly(cb, *args) -> bool:
        """Call ``cb(*args)`` if set, swallowing errors; True when it ran without raising."""
        if cb is None:
            return False
        try:
            cb(*args)
            return True
        except Exception:
            return False

    def _deliver_to_stream_callbacks(self, text: str) -> bool:
        """Send ``text`` to the display + TTS delta callbacks; True if at least one accepted it."""
        results = [self._call_quietly(cb, text) for cb in (self.stream_delta_callback, self._stream_callback)]
        return any(results)

    def _enqueue_stream_hook(self, event: str, *, label: str | None = None, **fields: Any) -> None:
        """Best-effort plugin stream hook enqueue; never raises into the stream path."""
        try:
            from agent.plugin_stream_hooks import enqueue_plugin_stream_hook

            enqueue_plugin_stream_hook(event, **self._stream_hook_base_payload(), **fields)
        except Exception:
            logger.debug("%s plugin hook enqueue failed", label or event, exc_info=True)

    def _reset_stream_delivery_tracking(self) -> None:
        """Reset tracking for text delivered during the current model response.

        Flushes the think scrubber's benign tail first, routed through the context scrubber (a span
        straddling the boundary must still be caught), then the context scrubber's own tail.
        """
        think_scrubber = getattr(self, "_stream_think_scrubber", None)
        ctx_scrubber = getattr(self, "_stream_context_scrubber", None)

        def deliver(tail: str) -> None:
            if tail:
                self._deliver_to_stream_callbacks(tail)
                self._record_streamed_assistant_text(tail)

        # Flush any benign partial-tag tail held by the think scrubber first (#17924): an innocent '<' at
        # the end of the stream that turned out not to be a tag prefix should reach the UI. Then flush the
        # context scrubber. Order matters — the think scrubber's output feeds into the context scrubber's
        # state.
        # Suppress reasoning/thinking blocks via the stateful scrubber (#17924). Earlier versions ran
        # _strip_think_blocks per-delta here, which destroyed downstream state machines when a tag was split
        # across deltas (e.g. MiniMax-M2.7 sends '<think>' and its content as separate deltas — regex case 2
        # erased the first delta, so the CLI/gateway state machine never saw the open tag and leaked the
        # reasoning content as regular response text).
        if think_scrubber is not None:
            think_tail = think_scrubber.flush()
            deliver(ctx_scrubber.feed(think_tail) if think_tail and ctx_scrubber is not None else think_tail)
        if ctx_scrubber is not None:
            deliver(ctx_scrubber.flush())
        self._current_streamed_assistant_text = ""

    @property
    def _current_streamed_assistant_text(self) -> str:
        """Visible assistant text streamed so far this turn. Backed by a list of pieces: ``+=`` on a string
        attribute copies the whole reply on every delta (quadratic). Hot-path emptiness checks look at
        ``_streamed_assistant_text_parts`` so they do not join per token."""
        parts = getattr(self, "_streamed_assistant_text_parts", None)
        return "".join(parts) if parts else ""

    @_current_streamed_assistant_text.setter
    def _current_streamed_assistant_text(self, value: str) -> None:
        self._streamed_assistant_text_parts = [value] if value else []

    def _record_streamed_assistant_text(self, text: str) -> None:
        """Accumulate visible assistant text emitted through stream callbacks (superseded writers excluded)."""
        if isinstance(text, str) and text and not self._stream_writer_superseded():
            parts = getattr(self, "_streamed_assistant_text_parts", None)
            if parts is None:
                parts = self._streamed_assistant_text_parts = []
            parts.append(text)

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(self._strip_think_blocks(content or ""))
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        # Prefix match, not equality: the final may be streamed text plus a trailing delta. The
        # reverse (streamed longer) is NOT matched — it could suppress a needed resend.
        return bool(visible_content and streamed) and visible_content.startswith(streamed)

    def _extract_codex_interim_visible_parts(self, assistant_msg: Dict[str, Any]) -> List[str]:
        """Visible Codex commentary (``phase=commentary`` items), one string per message item.

        ``phase=analysis`` stays hidden (scratchpad); with ``display.show_commentary=false``
        commentary stays on the reasoning channel.
        """
        items = assistant_msg.get("codex_message_items") if getattr(self, "show_commentary", True) else None
        messages: List[str] = []
        for item in items if isinstance(items, list) else ():
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            phase, content_parts = item.get("phase"), item.get("content")
            if not isinstance(phase, str) or phase.strip().lower() != "commentary" or not isinstance(content_parts, list):
                continue
            visible = "".join(
                part["text"] for part in content_parts
                if isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str) and part["text"].strip()
            ).strip()
            visible = self._visible_commentary(visible)
            if visible:
                messages.append(visible)
        return messages

    def _visible_commentary(self, text: str) -> str:
        """Think-stripped, redacted commentary text ("" when nothing visible remains)."""
        visible = self._strip_think_blocks(text).strip()
        return redact_sensitive_text(visible) if visible else visible

    def _extract_codex_interim_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """All visible Codex commentary joined, for comparison/fallback."""
        return "\n\n".join(self._extract_codex_interim_visible_parts(assistant_msg)).strip()

    def _interim_assistant_visible_text(self, assistant_msg: Dict[str, Any]) -> str:
        """Assistant text eligible for interim delivery: structured Codex commentary first — a response can
        hold commentary AND a partial final answer while tools are pending, and treating content as
        progress leaks the answer early — else top-level content (may be a parts list)."""
        return (
            self._extract_codex_interim_visible_text(assistant_msg)
            or self._strip_think_blocks(flatten_message_text(assistant_msg.get("content"))).strip()
        )

    def _interim_text_was_delivered(self, text: str) -> bool:
        normalized = self._normalize_interim_visible_text(text)
        return bool(normalized) and normalized in getattr(self, "_delivered_interim_texts", set())

    def _record_delivered_interim_text(self, text: str) -> None:
        normalized = self._normalize_interim_visible_text(text)
        if normalized:
            if not isinstance(getattr(self, "_delivered_interim_texts", None), set):
                self._delivered_interim_texts = set()
            self._delivered_interim_texts.add(normalized)

    def _deliver_interim(self, visible: str, *, already_streamed: bool, record: List[str]) -> None:
        """Hand ``visible`` to ``interim_assistant_callback`` and mark ``record`` delivered; swallows callback errors."""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None:
            return
        try:
            cb(visible, already_streamed=already_streamed)
            for part in record:
                self._record_delivered_interim_text(part)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _fire_streamed_codex_commentary(self, text: str) -> None:
        """Deliver a completed live Codex commentary message immediately."""
        if getattr(self, "interim_assistant_callback", None) is None or not isinstance(text, str):
            return
        visible = self._visible_commentary(text)
        if not visible or visible == "(empty)" or self._interim_text_was_delivered(visible):
            return
        self._deliver_interim(visible, already_streamed=False, record=[visible])

    def _emit_interim_assistant_message(self, assistant_msg: Dict[str, Any]) -> None:
        """Surface a real mid-turn assistant commentary message to the UI layer. Does NOT set
        ``_response_was_previewed`` ("the final response was shown") — the CLI would then suppress a
        different final summary."""
        if not isinstance(assistant_msg, dict):
            return
        commentary_parts = self._extract_codex_interim_visible_parts(assistant_msg)
        # Dedup within this message and against earlier deliveries, first occurrence wins.
        pending: dict[str, str] = {}
        for part in commentary_parts:
            key = self._normalize_interim_visible_text(part)
            if key and key not in pending and not self._interim_text_was_delivered(part):
                pending[key] = part
        undelivered_parts = list(pending.values())
        visible = "\n\n".join(undelivered_parts).strip() if commentary_parts else self._interim_assistant_visible_text(assistant_msg)
        if not visible or visible == "(empty)" or self._interim_text_was_delivered(visible):
            return
        already_streamed = self._interim_content_was_streamed(visible)
        self._enqueue_stream_hook("on_interim_message", text=visible, already_streamed=already_streamed)
        self._deliver_interim(visible, already_streamed=already_streamed, record=undelivered_parts or [visible])

    def _ensure_stream_writer_state(self) -> None:
        """Lazily create the single-writer guard fields (#65991).

        The fields are normally set unconditionally in ``agent_init`` (``_STREAM_STATE``), so every
        ``__init__``-built agent shares ONE lock from birth and the lazy path below is never taken by
        two threads. Only agents constructed via ``AIAgent.__new__`` (test doubles, legacy/partially-
        initialized instances) skip that path; claiming/checking the writer must not crash those, so
        initialize the fields on first use.
        """
        if getattr(self, "_stream_writer_lock", None) is None:
            self._stream_writer_lock = threading.Lock()
        if getattr(self, "_stream_writer_tls", None) is None:
            self._stream_writer_tls = threading.local()
        for attr in ("_stream_writer_token", "_stream_writer_dropped"):
            if not hasattr(self, attr):
                setattr(self, attr, 0)

    def _claim_stream_writer(self) -> int:
        """Claim exclusive ownership of the delta sink for this stream attempt; returns its writer token.

        Every attempt (each provider path, each retry) claims right before consuming; claiming bumps
        the shared token, so an earlier attempt still alive on another thread is superseded and its
        late chunks fenced out. Stored per-thread: a thread that never claimed can never be fenced.

        See #65991.
        """
        self._ensure_stream_writer_state()
        with self._stream_writer_lock:
            self._stream_writer_token = token = self._stream_writer_token + 1
        self._stream_writer_tls.token = token
        return token

    def _stream_writer_is_current(self, token: int) -> bool:
        """True when ``token`` is still the active writer, so a stream loop can bail the instant it is superseded.

        active writer — i.e. no newer stream attempt has claimed the sink since (#65991).
        """
        return token == getattr(self, "_stream_writer_token", token)

    def _stream_writer_superseded(self) -> bool:
        """True when this thread claimed the sink but a newer attempt has since claimed it (never for a non-claimer).

        stream attempt has since claimed it — i.e. this thread is a stale writer whose chunks must be
        dropped (#65991).
        """
        token = getattr(getattr(self, "_stream_writer_tls", None), "token", None)
        return token is not None and token != getattr(self, "_stream_writer_token", token)

    def _note_dropped_stream_writer(self, where: str) -> None:
        """Record + log that a superseded stream's delta was discarded."""
        try:
            self._stream_writer_dropped = int(getattr(self, "_stream_writer_dropped", 0)) + 1
        except Exception:
            self._stream_writer_dropped = 1
        # Log sparsely (first drop, then powers of two) so a chatty superseded stream can't flood the log.
        _n = self._stream_writer_dropped
        if _n == 1 or (_n & (_n - 1)) == 0:
            logger.warning(
                "Dropped delta from a superseded stream writer at %s "
                "(discarded=%d this turn) — a stale stream tried to write into "
                "the turn after a retry superseded it.",
                where, _n,
            )

    def _stream_hook_base_payload(self) -> Dict[str, Any]:
        return {
            "turn_id": getattr(self, "_current_turn_id", "") or "",
            "iteration": int(getattr(self, "_api_call_count", 0) or 0),
            "session_id": self.session_id or "",
            "model": self.model or "",
            "provider": self.provider or "",
            "surface": self.platform or "cli",
        }

    def _emit_stream_start(self) -> None:
        self._enqueue_stream_hook("on_stream_start")

    def _emit_stream_end(self, *, final_text: str, finished: bool, error: str | None) -> None:
        self._enqueue_stream_hook("on_stream_end", final_text=final_text, finished=finished, error=error)

    def _fire_stream_delta(self, text: str) -> None:
        """Fire all registered stream delta callbacks (display + TTS)."""
        # A superseded stream must not interleave its tokens alongside the retry that replaced it.
        if self._stream_writer_superseded():
            # See #65991.
            self._note_dropped_stream_writer("_fire_stream_delta")
            return
        # One paragraph break before the first text delta after a tool iteration, without
        # stacking blank lines across back-to-back tool iterations.
        prepended_break = bool(getattr(self, "_stream_needs_break", False) and text and text.strip())
        if prepended_break:
            self._stream_needs_break = False
            text = "\n\n" + text
        if isinstance(text, str):
            # Stateful scrubbers: per-delta regex stripping destroyed downstream state machines when a
            # tag was split across deltas; memory-context spans split across chunks must not leak to
            # the UI. Legacy callers lack the scrubber attributes and get the whole-string fallbacks.
            think_scrubber = getattr(self, "_stream_think_scrubber", None)
            # See #5719.
            scrubber = getattr(self, "_stream_context_scrubber", None)
            text = think_scrubber.feed(text) if think_scrubber is not None else self._strip_think_blocks(text)
            text = scrubber.feed(text) if scrubber is not None else sanitize_context(text)
            # Only strip leading newlines on the first delta — mid-stream "\n" is legitimate markdown.
            # Check the parts list, not the joined property (joining per token copies the whole reply).
            if not prepended_break and not getattr(self, "_streamed_assistant_text_parts", None):
                text = text.lstrip("\n")
        if not text:
            return
        delivered = self._deliver_to_stream_callbacks(text)
        self._enqueue_stream_hook("on_stream_delta", delta=text, kind="text")
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered; superseded writers are fenced like content deltas."""
        if self._stream_writer_superseded():
            # Single-writer guard (#65991): fence out a superseded stream's reasoning deltas the same way as
            # content deltas.
            self._note_dropped_stream_writer("_fire_reasoning_delta")
            return
        self._call_quietly(self.reasoning_callback, text)
        try:
            from agent.plugin_stream_hooks import stream_reasoning_deltas_enabled

            enabled = stream_reasoning_deltas_enabled()
        except Exception:
            logger.debug("reasoning on_stream_delta plugin hook enqueue failed", exc_info=True)
            return
        if enabled:
            self._enqueue_stream_hook("on_stream_delta", label="reasoning on_stream_delta", delta=text, kind="reasoning")

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """Notify the display layer that the model is generating tool call arguments (spinner for large payloads)."""
        self._call_quietly(self.tool_gen_callback, tool_name)

    def _has_stream_consumers(self) -> bool:
        """Return True if any streaming consumer is registered."""
        try:
            from agent.plugin_stream_hooks import has_stream_observer_hooks

            if has_stream_observer_hooks():
                return True
        except Exception:
            logger.debug("plugin stream hook consumer check failed", exc_info=True)
        return self.stream_delta_callback is not None or getattr(self, "_stream_callback", None) is not None
