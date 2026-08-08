"""Frame poller — periodically captures the desktop and emits frame events.

This is the "later enhancement" from the AVA-14 findings note.  The primary
frame source observes existing ``computer_use`` multimodal results; this
poller actively captures the desktop on a configurable interval and emits
``frame`` events through the live-glass event bus.

Architecture:
  * ``FramePollerBackend`` — pluggable interface (Protocol).
  * ``computer_use_backend_factory()`` — returns the active computer_use backend.
  * ``FramePoller`` — runs a background thread that polls the backend.

Thread-safety: the poller runs in a daemon thread and calls the event bus's
thread-safe ``publish()``.  Start/stop is controlled by a threading.Event.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

_MIN_INTERVAL = 0.1   # seconds — prevent busy-loop polling
_DEFAULT_INTERVAL = 5.0


# ── Backend interface ──────────────────────────────────────────────────

class FramePollerBackend(Protocol):
    """Pluggable backend that can capture a screenshot on demand.

    Returns a frame-payload dict on success, or ``None`` when no capture
    is available (e.g. no desktop session).  Must be callable from a
    background thread.
    """

    def capture_frame(self) -> dict[str, Any] | None: ...

    def is_available(self) -> bool: ...


# ── Factory for the built-in computer_use backend ──────────────────────

def computer_use_backend_factory() -> FramePollerBackend | None:
    """Return the active ``computer_use`` backend wrapped for polling.

    Returns ``None`` when the backend is unavailable (e.g. cua-driver not
    installed, or running on an unsupported platform).
    """
    try:
        from tools.computer_use.tool import _get_backend

        backend = _get_backend()
    except Exception:
        logger.debug("frame_poller: cannot init computer_use backend", exc_info=True)
        return None

    if not _is_backend_ready(backend):
        return None

    return _ComputerUseBackendAdapter(backend)


def _is_backend_ready(backend: Any) -> bool:
    try:
        ready = getattr(backend, "is_ready", None)
        if callable(ready) and not ready():
            return False
    except Exception:
        return False
    return True


class _ComputerUseBackendAdapter:
    """Wrap the computer_use backend so it speaks the FramePollerBackend protocol."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def capture_frame(self) -> dict[str, Any] | None:
        try:
            from tools.computer_use.tool import _capture_response
            cap = self._backend.capture(mode="som")
        except Exception:
            logger.debug("frame_poller: capture failed", exc_info=True)
            return None
        return _extract_frame_from_capture(cap)

    def is_available(self) -> bool:
        return _is_backend_ready(self._backend)


def _extract_frame_from_capture(cap: Any) -> dict[str, Any] | None:
    """Pull the image_url and metadata out of a CaptureResult or dict."""
    png_b64 = getattr(cap, "png_b64", None)
    if not png_b64:
        # May be a dict (AX path has no image).
        if isinstance(cap, dict):
            return None
        return None

    mode = getattr(cap, "mode", "som")
    width = getattr(cap, "width", 0)
    height = getattr(cap, "height", 0)
    app = getattr(cap, "app", None)

    # Detect image format from base64 magic bytes.
    _b64_prefix = png_b64[:8] if isinstance(png_b64, str) else ""
    _mime = "image/jpeg" if _b64_prefix.startswith("/9j/") else "image/png"

    return {
        "image_url": f"data:{_mime};base64,{png_b64}",
        "mime_type": _mime,
        "mode": mode,
        "width": width,
        "height": height,
        "summary": f"capture mode={mode} {width}x{height}"
                   + (f" app={app}" if app else ""),
        "source": "poller",
    }


# ── Poller ─────────────────────────────────────────────────────────────

class FramePoller:
    """Periodically captures frames and emits them via the event bus.

    Usage::

        poller = FramePoller(backend, interval=5.0)
        poller.start()
        # … agent runs, doing computer_use things …
        poller.stop()

    Or as a context manager::

        with FramePoller(backend, interval=5.0):
            …
    """

    def __init__(
        self,
        backend: FramePollerBackend,
        interval: float = _DEFAULT_INTERVAL,
    ) -> None:
        self._backend = backend
        self.interval = max(interval, _MIN_INTERVAL)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="live-glass-frame-poller",
            daemon=True,
        )
        self._thread.start()
        logger.debug("frame_poller: started (interval=%.1fs)", self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.debug("frame_poller: stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _poll_loop(self) -> None:
        from plugins.observability.live_glass import publish

        while not self._stop_event.wait(self.interval):
            try:
                if not self._backend.is_available():
                    continue
                frame = self._backend.capture_frame()
                if frame is None:
                    continue
                publish("frame", frame)
            except Exception:
                logger.debug("frame_poller: poll cycle failed", exc_info=True)

    # ── context manager ────────────────────────────────────────────────

    def __enter__(self) -> "FramePoller":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()
