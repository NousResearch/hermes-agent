"""BlueBubbles live-glass adapter — render event-bus events as iMessage texts.

Subscribe to the live-glass event bus and send:
  * ``frame`` → image (from the data: URL in the payload)
  * ``log``    → compact status text
  * ``approval_request`` → text prompt ``Reply APPROVE or DENY: <command>``
    (no interactive buttons — iMessage limitation; adapter is observer-only)

The adapter is transport-agnostic in that it takes a ``send_*`` interface
rather than importing the BlueBubbles SDK directly.  The gateway's
BlueBubbles platform adapter wires these callbacks to the real HTTP client.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from plugins.observability.live_glass.adapters.telegram import (
    _save_data_url_to_tempfile,
)

logger = logging.getLogger(__name__)


# ── BlueBubbles sender interface (implemented by the gateway) ─────────────

class BlueBubblesSender(Protocol):
    """Minimal send interface the adapter needs from the BlueBubbles gateway."""

    def send_image(
        self, chat_address: str, file_path: str, caption: str | None
    ) -> Any: ...

    def send_message(
        self, chat_address: str, text: str
    ) -> Any: ...


# ── Session → chat routing ────────────────────────────────────────────────

BlueBubblesRouter = Callable[[str], str | None]
"""Return a BlueBubbles chat_address for a live-glass session_id, or None.

chat_address is typically a phone number, email, or group chat GUID
as used by the BlueBubbles API.
"""


# ── Adapter ───────────────────────────────────────────────────────────────

class BlueBubblesLiveGlassAdapter:
    """Subscribe to the event bus and render events as iMessage messages.

    Observer-only: this adapter never approves or denies commands.
    """

    def __init__(
        self,
        sender: BlueBubblesSender,
        resolve_chat: BlueBubblesRouter,
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
        logger.debug("BlueBubbles live-glass adapter: started")

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.debug("BlueBubbles live-glass adapter: stopped")

    # ── event dispatch ────────────────────────────────────────────────────

    def _on_event(self, event: dict[str, Any]) -> None:
        session_id: str = event.get("session_id", "")
        chat_address = self._resolve_chat(session_id) if session_id else None
        if chat_address is None:
            return

        event_type: str = event.get("type", "")
        if event_type == "frame":
            self._handle_frame(chat_address, event)
        elif event_type == "log":
            self._handle_log(chat_address, event)
        elif event_type == "approval_request":
            self._handle_approval(chat_address, event)

    # ── frame → image ─────────────────────────────────────────────────────

    def _handle_frame(self, chat_address: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        image_url: str = payload.get("image_url", "")

        file_path = _save_data_url_to_tempfile(image_url)
        if file_path is None:
            return

        try:
            summary = payload.get("summary", "")
            self._sender.send_image(chat_address, file_path, caption=summary[:200])
        except Exception:
            logger.debug(
                "BlueBubbles live-glass: send_image failed", exc_info=True
            )
        finally:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── log → text ────────────────────────────────────────────────────────

    def _handle_log(self, chat_address: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        tool = payload.get("tool_name", "?")
        status = payload.get("status", "?")
        duration = payload.get("duration_ms", 0)
        error = payload.get("error_message", "")

        icon = "\u2713" if status == "ok" else "\u2717"  # ✓ or ✗
        text = f"{icon} `{tool}` ({duration}ms)"
        if error:
            text += f" \u2014 {error[:120]}"  # —

        try:
            self._sender.send_message(chat_address, text)
        except Exception:
            logger.debug(
                "BlueBubbles live-glass: send_message failed", exc_info=True
            )

    # ── approval → reply-prompt text (no buttons) ─────────────────────────

    def _handle_approval(self, chat_address: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        command = payload.get("command", "approve?")
        desc = payload.get("description", "")

        text = f"\u26a0\ufe0f Approval needed\n{desc}\n\nReply APPROVE or DENY: {command[:200]}"

        try:
            self._sender.send_message(chat_address, text)
        except Exception:
            logger.debug(
                "BlueBubbles live-glass: approval send failed", exc_info=True
            )
