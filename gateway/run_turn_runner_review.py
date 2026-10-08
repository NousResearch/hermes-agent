"""Hold background-review notices until the current turn's main response is delivered."""

from __future__ import annotations

import threading


class TurnReviewCallbacksMixin:
    """Per-turn review callback buffering shared by the agent callback wiring."""

    def _make_bg_review_callbacks(self):
        """(send, release): background-review messages ("💾 Memory updated") are held until the
        adapter's post-delivery hook releases them after the main response lands."""
        from gateway.run import _interim_metadata, _non_conversational_metadata
        ctx = self._ctx
        release_evt = threading.Event()
        pending: list[str] = []
        pending_lock = threading.Lock()

        def deliver(message: str) -> None:
            if self._status_live():
                self._send_status_text(
                    message,
                    _interim_metadata(_non_conversational_metadata(ctx._status_thread_metadata, platform=ctx.source.platform)),
                    "background_review_callback scheduling error",
                )

        def release() -> None:
            release_evt.set()
            with pending_lock:
                queued = list(pending)
                pending.clear()
            for message in queued:
                deliver(message)

        def send(message: str) -> None:
            if not self._status_live():
                return
            if not release_evt.is_set():
                with pending_lock:
                    if not release_evt.is_set():
                        pending.append(message)
                        return
            deliver(message)

        return send, release
