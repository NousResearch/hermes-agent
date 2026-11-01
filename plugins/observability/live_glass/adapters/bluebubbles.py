"""BlueBubbles/iMessage live-glass adapter with graceful degradation.

iMessage limitations: no inline buttons, no message editing.  This adapter
renders frames as images, logs as compact text, and approval requests as
reply-to-approve text prompts.  It is strictly observer-only — it never
approves or denies on its own.

Same pattern as the Telegram adapter:
  adapter = BlueBubblesLiveGlassAdapter(sender, resolve_address)
  adapter.start()
  ...
  adapter.stop()
"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

_DATA_IMAGE_RE = __import__("re").compile(r"^data:(image/[^;,]+);base64,")


class BlueBubblesSender(Protocol):
    """Minimal send interface the adapter needs from the BlueBubbles gateway."""

    def send_image(self, address: str, file_path: str, caption: str | None) -> Any: ...
    def send_message(self, address: str, text: str) -> Any: ...


AddressRouter = Callable[[str], str | None]
"""Return an iMessage address for a live-glass session_id, or None."""


class BlueBubblesLiveGlassAdapter:
    """Subscribe to the event bus and render events as iMessage messages."""

    def __init__(
        self,
        sender: BlueBubblesSender,
        resolve_address: AddressRouter,
    ) -> None:
        self._sender = sender
        self._resolve_address = resolve_address
        self._unsubscribe: Callable[[], None] | None = None

    def start(self) -> None:
        from plugins.observability.live_glass import subscribe

        self._unsubscribe = subscribe(
            self._on_event,
            event_types={"frame", "log", "approval_request"},
        )
        logger.debug("BlueBubbles live-glass adapter: started")

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.debug("BlueBubbles live-glass adapter: stopped")

    # ── event dispatch ──────────────────────────────────────────────────

    def _on_event(self, event: dict[str, Any]) -> None:
        session_id: str = event.get("session_id", "")
        address = self._resolve_address(session_id) if session_id else None
        if address is None:
            return

        event_type: str = event.get("type", "")
        if event_type == "frame":
            self._handle_frame(address, event)
        elif event_type == "log":
            self._handle_log(address, event)
        elif event_type == "approval_request":
            self._handle_approval(address, event)

    # ── frame → image attachment ────────────────────────────────────────

    def _handle_frame(self, address: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        image_url: str = payload.get("image_url", "")

        file_path = _save_data_url_to_tempfile(image_url)
        if file_path is None:
            return

        try:
            summary = payload.get("summary", "")
            self._sender.send_image(address, file_path, caption=summary[:200])
        except Exception:
            logger.debug("BlueBubbles live-glass: send_image failed", exc_info=True)
        finally:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── log → compact text ──────────────────────────────────────────────

    def _handle_log(self, address: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        tool = payload.get("tool_name", "?")
        status = payload.get("status", "?")
        duration = payload.get("duration_ms", 0)
        error = payload.get("error_message", "")

        icon = "OK" if status == "ok" else "FAIL"
        text = f"[{icon}] {tool} ({duration}ms)"
        if error:
            text += f" — {error[:120]}"

        try:
            self._sender.send_message(address, text)
        except Exception:
            logger.debug("BlueBubbles live-glass: send_message failed", exc_info=True)

    # ── approval → reply-to-approve text (NO buttons) ───────────────────

    def _handle_approval(self, address: str, event: dict[str, Any]) -> None:
        """Render approval as text instructions — iMessage has no inline buttons.

        IMPORTANT: The adapter is observer-only.  It does NOT approve or deny.
        The gateway handles the actual approval routing when the user replies.
        """
        payload: dict[str, Any] = event.get("payload") or {}
        command = payload.get("command", "approve?")
        desc = payload.get("description", "")

        text = f"⚠️ Approval needed\n{desc}\nCommand: {command[:200]}\n\nReply APPROVE or DENY to authorize."

        try:
            self._sender.send_message(address, text)
        except Exception:
            logger.debug("BlueBubbles live-glass: approval send failed", exc_info=True)


# ── Helpers ─────────────────────────────────────────────────────────────

def _save_data_url_to_tempfile(data_url: str) -> str | None:
    if not data_url:
        return None
    match = _DATA_IMAGE_RE.match(data_url)
    if not match:
        return None
    mime = match.group(1)
    b64 = data_url[match.end():]
    ext = {
        "image/png": ".png", "image/jpeg": ".jpg",
        "image/jpg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
    }.get(mime, ".png")
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        logger.debug("BlueBubbles live-glass: base64 decode failed", exc_info=True)
        return None
    try:
        fd, path = tempfile.mkstemp(suffix=ext, prefix="liveglass_bb_")
        with open(fd, "wb") as f:
            f.write(raw)
        return path
    except Exception:
        logger.debug("BlueBubbles live-glass: tempfile write failed", exc_info=True)
        return None
