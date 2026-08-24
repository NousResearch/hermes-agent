"""User-facing status / warning / notice plumbing for ``AIAgent``.

Safe printing, quiet-mode gating, deduped context-overflow warnings, and the buffered retry
chatter that is shown only when every retry/fallback is exhausted.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
import sys

from agent.session_activity import ActivityProvenance

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


class StatusOutputMixin:
    """Status/warning/notice emission and retry-chatter buffering (see module docstring)."""

    def _safe_print(self, *args, **kwargs):
        """Print that swallows broken pipes / closed stdout (headless stdout can vanish mid-session);
        routes through ``self._print_fn`` so the CLI can inject an ANSI-aware renderer."""
        try:
            (self._print_fn or print)(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """Verbose print — suppressed while tokens are streaming (allowed during tool execution) and after
        the main response; ``force=True`` bypasses both. ``suppress_status_output`` (``hermes chat -q``) wins."""
        if getattr(self, "suppress_status_output", False):
            return
        if force or not (getattr(self, "_mute_post_response", False) or (self._has_stream_consumers() and not self._executing_tools)):
            self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """True when quiet-mode spinner output has a safe sink (``_print_fn`` or a real TTY); a raw spinner
        on a non-TTY stdout can corrupt protocol streams (ACP JSON-RPC)."""
        if self._print_fn is not None:
            return True
        try:
            return bool(sys.stdout.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """True when quiet-mode tool summaries should print directly (CLI, no callback owns rendering);
        ``suppress_status_output`` always wins so ``[tool]``/``[done]`` never land in captured stdout.

        ``suppress_status_output`` (the strict machine-readable mode used by ``hermes chat -Q``) always
        wins: those flows neutralize the rendering callbacks, and without this gate the "no callback owns
        rendering" fallback would print ``[tool]``/``[done]`` spinner lines into the captured stdout it
        exists to keep clean (#93220).
        """
        if getattr(self, "suppress_status_output", False):
            return False
        return self.quiet_mode and not self.tool_progress_callback and getattr(self, "platform", "") == "cli"

    def _call_callback(self, name: str, *args, origin: str) -> None:
        """Invoke ``self.<name>(*args)`` if set, swallowing errors — a driver callback must never break the loop."""
        cb = getattr(self, name, None)
        if cb:
            try:
                cb(*args)
            except Exception:
                logger.debug("%s error in %s", name, origin, exc_info=True)

    def _emit_status_kind(self, kind: str, message: str, *, origin: str) -> None:
        """Print to the CLI (``_vprint(force=True)``) and forward to ``status_callback(kind, message)``. Never raises."""
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        self._call_callback("status_callback", kind, message, origin=origin)

    def _emit_status(self, message: str) -> None:
        """Emit a lifecycle status message (CLI + gateway ``status_callback``)."""
        self._emit_status_kind("lifecycle", message, origin="_emit_status")

    def _emit_warning(self, message: str) -> None:
        """Emit a user-visible warning for degraded side paths where the turn continues but the user must know."""
        self._emit_status_kind("warn", message, origin="_emit_warning")

    def _warn_context_overflow_blocked(self, reason: str, preflight_tokens: int, threshold_tokens: int) -> None:
        """Warn (deduped on the block *kind* — ``cooldown`` / ``ineffective`` — not the countdown string;
        cleared by ``_clear_context_overflow_warn``) when context is over the threshold but compression is blocked."""
        _warn_kind = (reason or "unknown").split(":", 1)[0]
        _warn_key = ("ctx_overflow_blocked", _warn_kind)
        if getattr(self, "_last_ctx_overflow_warn", None) == _warn_key:
            return
        self._last_ctx_overflow_warn = _warn_key
        from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE

        # cooldown + anti-thrash (ineffective) are both "compression blocked".
        if _warn_kind in ("cooldown", "ineffective"):
            self._touch_activity(f"compression blocked ({reason})", provenance=ActivityProvenance.AGENT_COMPRESSION_COOLDOWN)
        self._emit_warning(CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(
            tokens=preflight_tokens, threshold=threshold_tokens, reason=reason,
        ))

    def _warn_uncompressed_context_overflow(self, preflight_tokens: int, context_length: int) -> None:
        """Deduped warning when uncompressed context exceeds the model limit; points the user at /compact.

        When compression is explicitly disabled (compression.enabled: false), long sessions can grow past
        the model context window with no compression to shrink them (#89297). Surface an actionable warning
        so the user knows to run /compact or enable compression.
        """
        _warn_key = ("uncompressed_ctx_overflow", context_length)
        if getattr(self, "_last_ctx_overflow_warn", None) != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            self._emit_warning(
                f"⚠️ Session context (~{preflight_tokens:,} tokens) exceeds the model "
                f"context window (~{context_length:,} tokens) with compression disabled "
                f"(compression.enabled: false). Use /compact to compress history or "
                f"enable compression in config.yaml."
            )

    def _clear_context_overflow_warn(self) -> None:
        """Reset the blocked-overflow warning dedup so it can re-fire on the next blocked turn."""
        self._last_ctx_overflow_warn = None

    def _emit_notice(self, notice) -> None:
        """Fire a structured ``AgentNotice`` to the active driver (TUI / CLI)."""
        self._call_callback("notice_callback", notice, origin="_emit_notice")

    def _emit_notice_clear(self, key: str) -> None:
        """Clear a previously-fired sticky notice by ``key`` (e.g. on recovery)."""
        self._call_callback("notice_clear_callback", key, origin="_emit_notice_clear")

    def _emit_wait_notice(self, text: str) -> None:
        """Rewrite the live status line (CLI spinner, TUI ``thinking.delta``, gateway activity)
        so long provider waits are not an anonymous spinner."""
        self._touch_activity(text)
        self._call_callback("thinking_callback", text, origin="_emit_wait_notice")

    # ── Buffered retry/fallback status: shown only when every retry/fallback is exhausted, dropped on
    # success. Backend logs are unaffected (every site still logs). ──

    def _buffer_retry_message(self, kind: str, message: str) -> None:
        """Buffer a retry/fallback line as ``(kind, text)`` until we know whether the turn recovered.

        ``kind`` is ``"status"`` (replays via ``_emit_status``), ``"vprint"`` (``_vprint(force=True)``) or
        ``"warn"`` (``_emit_warning``).
        """
        buf = getattr(self, "_retry_status_buffer", None)
        if buf is None:
            buf = self._retry_status_buffer = []
        buf.append((kind, message))

    def _buffer_status(self, message: str) -> None:
        self._buffer_retry_message("status", message)

    def _buffer_vprint(self, message: str) -> None:
        self._buffer_retry_message("vprint", message)

    def _clear_status_buffer(self) -> None:
        """Drop buffered retry messages — call on successful recovery."""
        buf = getattr(self, "_retry_status_buffer", None)
        if buf:
            buf.clear()

    def _emit_pending_fallback_notice(self) -> None:
        """Surface the one-shot fallback-switch notice on successful recovery: a provider switch is durable
        state operators must see, unlike the retry chatter ``_clear_status_buffer`` drops. Emitted once, then
        cleared; on terminal failure the buffered switch line is flushed instead (``_flush_status_buffer``)."""
        notice = getattr(self, "_pending_fallback_notice", None)
        if not notice:
            return
        # Clear before emitting so a (swallowed) callback error can't leave a stale re-emit.
        self._pending_fallback_notice = None
        for item in notice if isinstance(notice, list) else [notice]:
            try:
                self._emit_status(str(item))
            except Exception:
                # One surface failure must not hide later switches from the same chain.
                continue

    def _flush_status_buffer(self) -> None:
        """Emit buffered retry messages — call on terminal failure so the user sees what was tried."""
        # The buffered trace already carries the switch line; drop the one-shot notice.
        self._pending_fallback_notice = None
        buf = getattr(self, "_retry_status_buffer", None)
        if not buf:
            return
        # Drain first so a callback exception doesn't double-emit.
        messages = list(buf)
        buf.clear()
        replay = {"status": self._emit_status, "warn": self._emit_warning}
        for kind, msg in messages:
            try:
                if kind in replay:
                    replay[kind](msg)
                else:
                    self._vprint(f"{self.log_prefix}{msg}", force=True)
            except Exception:
                pass
