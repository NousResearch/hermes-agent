"""Buffered retry/fallback status emission helpers for AIAgent.

Extracted verbatim from ``run_agent.py`` (godfile shard plan s1, cluster
c18 — 5 methods, 20/20 move agreement). The buffered helpers defer noisy
retry chatter (rate-limit retries, fallback switches) until the turn
outcome is known; on success the buffer is dropped, on terminal failure it
is flushed. Emission is delegated back through the MRO to
``AIAgent._emit_status`` / ``_emit_warning`` / ``_vprint``; the buffer
itself lives on the instance (``self._retry_status_buffer``,
``self._pending_fallback_notice``), initialized lazily here.
"""


class RetryStatusMixin:
    # ── Buffered retry/fallback status ────────────────────────────────────
    # Retry and fallback chains were flooding the CLI/gateway with status
    # noise that users found confusing: a single transient 429 could produce
    # 10+ "Provider/Endpoint/Retrying in 5s..." lines before the request
    # eventually succeeded.  The buffered helpers below capture these
    # status messages instead of emitting them immediately.  They are
    # flushed (shown to the user) ONLY when every retry and fallback has
    # been exhausted; on success they are silently dropped.  Backend logs
    # (agent.log) are unaffected — every individual emission site still
    # writes to ``logger.warning`` / ``logger.info`` for diagnosis.

    def _buffer_status(self, message: str) -> None:
        """Buffer a retry/fallback status message.

        Stored as a (kind, text) tuple where ``kind`` is one of:
        - ``"status"``  -> replays via ``_emit_status``
        - ``"vprint"``  -> replays via ``_vprint(force=True)``
        - ``"warn"``    -> replays via ``_emit_warning``
        Used to defer noisy retry chatter until we know whether the
        turn ultimately recovered or failed.
        """
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf is None:
                buf = []
                self._retry_status_buffer = buf
            buf.append(("status", message))
        except Exception:
            # Never break the retry loop on a buffer hiccup.
            pass

    def _buffer_vprint(self, message: str) -> None:
        """Buffer a vprint(force=True) retry/fallback line."""
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf is None:
                buf = []
                self._retry_status_buffer = buf
            buf.append(("vprint", message))
        except Exception:
            pass

    def _clear_status_buffer(self) -> None:
        """Drop buffered retry messages — call on successful recovery."""
        try:
            buf = getattr(self, "_retry_status_buffer", None)
            if buf:
                buf.clear()
        except Exception:
            pass

    def _emit_pending_fallback_notice(self) -> None:
        """Surface the one-shot fallback-switch notice on successful recovery.

        A provider/model switch is a durable state change operators must see,
        unlike transient retry chatter that ``_clear_status_buffer`` drops.
        ``try_activate_fallback`` records the switch in
        ``self._pending_fallback_notice``; this emits it exactly once via
        ``_emit_status`` and then clears it, so a successful fallback still
        produces one visible notice.  On terminal failure the buffered switch
        line is flushed instead (and this notice discarded) — see
        ``_flush_status_buffer`` — so the user always sees the switch once.
        """
        try:
            notice = getattr(self, "_pending_fallback_notice", None)
            if notice:
                # Clear before emitting so a (swallowed) callback error can't
                # leave the notice set for a stale re-emit on a later turn.
                self._pending_fallback_notice = None
                self._emit_status(notice)
        except Exception:
            # Never break the conversation loop on a notice hiccup.
            pass

    def _flush_status_buffer(self) -> None:
        """Emit buffered retry messages — call on terminal failure.

        Surfaces the full retry/fallback trace so the user can see what
        was tried before the turn gave up.
        """
        try:
            # The buffered trace already carries the fallback switch line, so
            # drop any one-shot fallback notice to avoid a stale duplicate
            # leaking into a later successful turn.
            self._pending_fallback_notice = None
            buf = getattr(self, "_retry_status_buffer", None)
            if not buf:
                return
            # Drain first so a callback exception doesn't double-emit.
            messages = list(buf)
            buf.clear()
            for kind, msg in messages:
                try:
                    if kind == "status":
                        self._emit_status(msg)
                    elif kind == "warn":
                        self._emit_warning(msg)
                    else:
                        self._vprint(f"{self.log_prefix}{msg}", force=True)
                except Exception:
                    pass
        except Exception:
            pass
