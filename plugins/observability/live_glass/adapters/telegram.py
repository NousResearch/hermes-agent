"""Telegram live-glass adapter — renders event-bus events as Telegram messages.

Subscribe to the live-glass event bus and send:
  * ``frame`` → photo (from the data: URL in the payload)
  * ``log``    → compact status text
  * ``approval_request`` → message with InlineKeyboard approve/deny buttons

The adapter is transport-agnostic in that it takes a ``send_*`` interface
rather than importing the Telegram SDK directly.  The gateway's Telegram
platform adapter wires these callbacks to the real Bot instance.
"""
from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

# Matches data:image/...;base64,<b64>
_DATA_IMAGE_RE = __import__("re").compile(r"^data:(image/[^;,]+);base64,")


# ── Telegram sender interface (implemented by the gateway) ──────────────

class TelegramSender(Protocol):
    """Minimal send interface the adapter needs from the Telegram gateway."""

    def send_photo(
        self, chat_id: int, photo_path: str, caption: str | None
    ) -> Any: ...

    def send_message(
        self, chat_id: int, text: str, reply_markup: Any | None
    ) -> Any: ...


# ── Session → chat routing ──────────────────────────────────────────────

SessionRouter = Callable[[str], int | None]
"""Return a Telegram chat_id for a live-glass session_id, or None."""


# ── Adapter ─────────────────────────────────────────────────────────────

class TelegramLiveGlassAdapter:
    """Subscribe to the event bus and render events as Telegram messages."""

    def __init__(
        self,
        sender: TelegramSender,
        resolve_chat: SessionRouter,
    ) -> None:
        self._sender = sender
        self._resolve_chat = resolve_chat
        self._unsubscribe: Callable[[], None] | None = None

    def start(self) -> None:
        """Subscribe to the live-glass event bus."""
        from plugins.observability.live_glass import subscribe

        self._unsubscribe = subscribe(
            self._on_event,
            event_types={"frame", "log", "approval_request"},
        )
        logger.debug("Telegram live-glass adapter: started")

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.debug("Telegram live-glass adapter: stopped")

    # ── event dispatch ──────────────────────────────────────────────────

    def _on_event(self, event: dict[str, Any]) -> None:
        session_id: str = event.get("session_id", "")
        chat_id = self._resolve_chat(session_id) if session_id else None
        if chat_id is None:
            return

        event_type: str = event.get("type", "")
        if event_type == "frame":
            self._handle_frame(chat_id, event)
        elif event_type == "log":
            self._handle_log(chat_id, event)
        elif event_type == "approval_request":
            self._handle_approval(chat_id, event)

    # ── frame → photo ───────────────────────────────────────────────────

    def _handle_frame(self, chat_id: int, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        image_url: str = payload.get("image_url", "")

        file_path = _save_data_url_to_tempfile(image_url)
        if file_path is None:
            return

        try:
            summary = payload.get("summary", "")
            self._sender.send_photo(chat_id, file_path, caption=summary[:200])
        except Exception:
            logger.debug("Telegram live-glass: send_photo failed", exc_info=True)
        finally:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── log → text ──────────────────────────────────────────────────────

    def _handle_log(self, chat_id: int, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        tool = payload.get("tool_name", "?")
        status = payload.get("status", "?")
        duration = payload.get("duration_ms", 0)
        error = payload.get("error_message", "")

        icon = "✓" if status == "ok" else "✗"
        text = f"{icon} `{tool}` ({duration}ms)"
        if error:
            text += f" — {error[:120]}"

        try:
            self._sender.send_message(chat_id, text, reply_markup=None)
        except Exception:
            logger.debug("Telegram live-glass: send_message failed", exc_info=True)

    # ── approval → inline keyboard ──────────────────────────────────────

    def _handle_approval(self, chat_id: int, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        command = payload.get("command", "approve?")
        desc = payload.get("description", "")
        tool_call_id = event.get("tool_call_id", "")

        text = f"⚠️ **Approve?**\n{desc}\n`{command[:200]}`"

        keyboard = _build_approval_keyboard(tool_call_id)
        try:
            self._sender.send_message(chat_id, text, reply_markup=keyboard)
        except Exception:
            logger.debug("Telegram live-glass: approval send failed", exc_info=True)


# ── Helpers ─────────────────────────────────────────────────────────────

def _save_data_url_to_tempfile(data_url: str) -> str | None:
    """Decode a data: URL to a temp file, return the path or None."""
    if not data_url:
        return None
    match = _DATA_IMAGE_RE.match(data_url)
    if not match:
        return None
    mime = match.group(1)
    b64 = data_url[match.end():]
    ext = _mime_to_ext(mime)
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        logger.debug("Telegram live-glass: base64 decode failed", exc_info=True)
        return None
    try:
        fd, path = tempfile.mkstemp(suffix=ext, prefix="liveglass_")
        with open(fd, "wb") as f:
            f.write(raw)
        return path
    except Exception:
        logger.debug("Telegram live-glass: tempfile write failed", exc_info=True)
        return None


def _mime_to_ext(mime: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime, ".png")


def _build_approval_keyboard(tool_call_id: str) -> dict[str, Any] | None:
    """Build an InlineKeyboardMarkup dict for an approval prompt.

    Callback data is JSON with ``action`` and ``tool_call_id`` so the
    gateway's CallbackQueryHandler can route to the approval flow.
    """
    if not tool_call_id:
        return None

    def _cb(action: str) -> str:
        return json.dumps({"a": action, "tcid": tool_call_id})

    return {
        "inline_keyboard": [
            [
                {"text": "Approve Once", "callback_data": _cb("once")},
                {"text": "Approve Session", "callback_data": _cb("session")},
            ],
            [
                {"text": "Approve Always", "callback_data": _cb("always")},
                {"text": "Deny", "callback_data": _cb("deny")},
            ],
        ]
    }
