"""Discord and Slack live-glass adapters — render event-bus events as messages.

Subscribe to the live-glass event bus and send:
  * ``frame`` → image as file/attachment
  * ``log``    → compact status text
  * ``approval_request`` → message with interactive approve/deny buttons

The adapters are transport-agnostic in that they take a ``send_*`` interface
rather than importing the platform SDK directly.  The gateway's platform
adapters wire these callbacks to the real clients.

Discord adapter uses message components (buttons) for approvals.
Slack adapter uses Block Kit interactive buttons for approvals.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional, Protocol

from plugins.observability.live_glass.adapters.telegram import (
    _save_data_url_to_tempfile,
)

logger = logging.getLogger(__name__)

# ── Sender interfaces (implemented by the gateway) ───────────────────────


class DiscordSender(Protocol):
    """Minimal send interface the adapter needs from the Discord gateway."""

    def send_file(
        self, channel_id: int, file_path: str, caption: str | None
    ) -> Any: ...

    def send_message(
        self, channel_id: int, text: str, components: list[dict[str, Any]] | None
    ) -> Any: ...


class SlackSender(Protocol):
    """Minimal send interface the adapter needs from the Slack gateway."""

    def upload_file(
        self, channel_id: str, file_path: str, caption: str | None
    ) -> Any: ...

    def send_message(
        self, channel_id: str, text: str, blocks: list[dict[str, Any]] | None
    ) -> Any: ...


# ── Session → channel routing ────────────────────────────────────────────

DiscordRouter = Callable[[str], int | None]
"""Return a Discord channel_id for a live-glass session_id, or None."""

SlackRouter = Callable[[str], str | None]
"""Return a Slack channel_id for a live-glass session_id, or None."""


# ── Discord adapter ──────────────────────────────────────────────────────


class DiscordLiveGlassAdapter:
    """Subscribe to the event bus and render events as Discord messages."""

    def __init__(
        self,
        sender: DiscordSender,
        resolve_channel: DiscordRouter,
    ) -> None:
        self._sender = sender
        self._resolve_channel = resolve_channel
        self._unsubscribe: Callable[[], None] | None = None

    def start(self) -> None:
        """Subscribe to the live-glass event bus."""
        from plugins.observability.live_glass import subscribe

        self._unsubscribe = subscribe(
            self._on_event,
            event_types={"frame", "log", "approval_request"},
        )
        logger.debug("Discord live-glass adapter: started")

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.debug("Discord live-glass adapter: stopped")

    # ── event dispatch ──────────────────────────────────────────────────

    def _on_event(self, event: dict[str, Any]) -> None:
        session_id: str = event.get("session_id", "")
        channel_id = self._resolve_channel(session_id) if session_id else None
        if channel_id is None:
            return

        event_type: str = event.get("type", "")
        if event_type == "frame":
            self._handle_frame(channel_id, event)
        elif event_type == "log":
            self._handle_log(channel_id, event)
        elif event_type == "approval_request":
            self._handle_approval(channel_id, event)

    # ── frame → file attachment ─────────────────────────────────────────

    def _handle_frame(self, channel_id: int, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        image_url: str = payload.get("image_url", "")

        file_path = _save_data_url_to_tempfile(image_url)
        if file_path is None:
            return

        try:
            summary = payload.get("summary", "")
            self._sender.send_file(channel_id, file_path, caption=summary[:200])
        except Exception:
            logger.debug("Discord live-glass: send_file failed", exc_info=True)
        finally:
            from pathlib import Path

            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── log → text ──────────────────────────────────────────────────────

    def _handle_log(self, channel_id: int, event: dict[str, Any]) -> None:
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
            self._sender.send_message(channel_id, text, components=None)
        except Exception:
            logger.debug("Discord live-glass: send_message failed", exc_info=True)

    # ── approval → message with buttons ─────────────────────────────────

    def _handle_approval(self, channel_id: int, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        command = payload.get("command", "approve?")
        desc = payload.get("description", "")
        tool_call_id = event.get("tool_call_id", "")

        text = f"\u26a0\ufe0f **Approve?**\n{desc}\n`{command[:200]}`"

        components = _build_discord_approval_components(tool_call_id)
        try:
            self._sender.send_message(channel_id, text, components=components)
        except Exception:
            logger.debug("Discord live-glass: approval send failed", exc_info=True)


# ── Slack adapter ────────────────────────────────────────────────────────


class SlackLiveGlassAdapter:
    """Subscribe to the event bus and render events as Slack messages."""

    def __init__(
        self,
        sender: SlackSender,
        resolve_channel: SlackRouter,
    ) -> None:
        self._sender = sender
        self._resolve_channel = resolve_channel
        self._unsubscribe: Callable[[], None] | None = None

    def start(self) -> None:
        """Subscribe to the live-glass event bus."""
        from plugins.observability.live_glass import subscribe

        self._unsubscribe = subscribe(
            self._on_event,
            event_types={"frame", "log", "approval_request"},
        )
        logger.debug("Slack live-glass adapter: started")

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.debug("Slack live-glass adapter: stopped")

    # ── event dispatch ──────────────────────────────────────────────────

    def _on_event(self, event: dict[str, Any]) -> None:
        session_id: str = event.get("session_id", "")
        channel_id = self._resolve_channel(session_id) if session_id else None
        if channel_id is None:
            return

        event_type: str = event.get("type", "")
        if event_type == "frame":
            self._handle_frame(channel_id, event)
        elif event_type == "log":
            self._handle_log(channel_id, event)
        elif event_type == "approval_request":
            self._handle_approval(channel_id, event)

    # ── frame → file upload ─────────────────────────────────────────────

    def _handle_frame(self, channel_id: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        image_url: str = payload.get("image_url", "")

        file_path = _save_data_url_to_tempfile(image_url)
        if file_path is None:
            return

        try:
            summary = payload.get("summary", "")
            self._sender.upload_file(channel_id, file_path, caption=summary[:200])
        except Exception:
            logger.debug("Slack live-glass: upload_file failed", exc_info=True)
        finally:
            from pathlib import Path

            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ── log → text ──────────────────────────────────────────────────────

    def _handle_log(self, channel_id: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        tool = payload.get("tool_name", "?")
        status = payload.get("status", "?")
        duration = payload.get("duration_ms", 0)
        error = payload.get("error_message", "")

        icon = "\u2713" if status == "ok" else "\u2717"
        text = f"{icon} `{tool}` ({duration}ms)"
        if error:
            text += f" \u2014 {error[:120]}"

        try:
            self._sender.send_message(channel_id, text, blocks=None)
        except Exception:
            logger.debug("Slack live-glass: send_message failed", exc_info=True)

    # ── approval → message with Block Kit buttons ───────────────────────

    def _handle_approval(self, channel_id: str, event: dict[str, Any]) -> None:
        payload: dict[str, Any] = event.get("payload") or {}
        command = payload.get("command", "approve?")
        desc = payload.get("description", "")
        tool_call_id = event.get("tool_call_id", "")

        text = f"\u26a0\ufe0f *Approve?*\n{desc}\n`{command[:200]}`"

        blocks = _build_slack_approval_blocks(tool_call_id, text)
        try:
            self._sender.send_message(channel_id, text, blocks=blocks)
        except Exception:
            logger.debug("Slack live-glass: approval send failed", exc_info=True)


# ── Discord components builder ───────────────────────────────────────────


def _build_discord_approval_components(
    tool_call_id: str,
) -> list[dict[str, Any]] | None:
    """Build Discord message components with approve/deny buttons.

    Returns a list of ActionRow components, or None if tool_call_id is empty.
    """
    if not tool_call_id:
        return None

    def _cid(action: str) -> str:
        return json.dumps({"a": action, "tcid": tool_call_id})

    return [
        {
            "type": 1,  # ActionRow
            "components": [
                {
                    "type": 2,  # Button
                    "style": 3,  # Success (green)
                    "label": "Approve Once",
                    "custom_id": _cid("once"),
                },
                {
                    "type": 2,  # Button
                    "style": 3,  # Success (green)
                    "label": "Approve Session",
                    "custom_id": _cid("session"),
                },
            ],
        },
        {
            "type": 1,  # ActionRow
            "components": [
                {
                    "type": 2,  # Button
                    "style": 1,  # Primary (blurple)
                    "label": "Approve Always",
                    "custom_id": _cid("always"),
                },
                {
                    "type": 2,  # Button
                    "style": 4,  # Danger (red)
                    "label": "Deny",
                    "custom_id": _cid("deny"),
                },
            ],
        },
    ]


# ── Slack Block Kit builder ──────────────────────────────────────────────


def _build_slack_approval_blocks(
    tool_call_id: str,
    header_text: str,
) -> list[dict[str, Any]] | None:
    """Build Slack Block Kit blocks with approve/deny buttons.

    Returns a list of Block Kit blocks, or None if tool_call_id is empty.
    """
    if not tool_call_id:
        return None

    def _value(action: str) -> str:
        return json.dumps({"a": action, "tcid": tool_call_id})

    return [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve Once"},
                    "style": "primary",
                    "action_id": "approve_once",
                    "value": _value("once"),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve Session"},
                    "style": "primary",
                    "action_id": "approve_session",
                    "value": _value("session"),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve Always"},
                    "action_id": "approve_always",
                    "value": _value("always"),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Deny"},
                    "style": "danger",
                    "action_id": "deny",
                    "value": _value("deny"),
                },
            ],
        },
    ]
