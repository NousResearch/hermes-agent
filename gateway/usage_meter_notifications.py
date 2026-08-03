"""Opt-in Discord notifications for changed account-usage meter readings."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

_PROVIDER_LABELS = {"claude": "Claude", "codex": "Codex"}
_TRACKED_STATUSES = ("running", "ready", "todo", "blocked")


def _provider_snapshot(value: Any) -> tuple[tuple[str, int], ...] | None:
    """Return a validated provider reading, or None when the refresh failed."""
    if not isinstance(value, Mapping) or value.get("available") is not True:
        return None
    windows = value.get("windows")
    if not isinstance(windows, list) or not windows:
        return None

    normalized: list[tuple[str, int]] = []
    for window in windows:
        if not isinstance(window, Mapping):
            return None
        label = window.get("label")
        used_percent = window.get("used_percent")
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(used_percent, (int, float))
            or isinstance(used_percent, bool)
        ):
            return None
        normalized.append((label, round(max(0.0, min(100.0, float(used_percent))))))
    return tuple(normalized)


def _format_message(
    changed: Mapping[str, tuple[tuple[str, int], ...]],
    kanban_summary: Mapping[str, int],
) -> str:
    provider_lines = []
    for provider in ("claude", "codex"):
        snapshot = changed.get(provider)
        if snapshot is None:
            continue
        windows = " · ".join(f"{label} {used_percent}%" for label, used_percent in snapshot)
        provider_lines.append(f"- {_PROVIDER_LABELS[provider]}: {windows}")

    status_line = " · ".join(
        f"{status} {int(kanban_summary.get(status, 0))}" for status in _TRACKED_STATUSES
    )
    return "📊 Account usage updated\n" + "\n".join(provider_lines) + f"\nKanban: {status_line}"


class UsageMeterNotifier:
    """Suppress duplicate provider readings and send one combined status report."""

    def __init__(
        self,
        send: Callable[[str], dict[str, Any]],
        kanban_summary: Callable[[], Mapping[str, int]] | None = None,
    ) -> None:
        self._send = send
        self._kanban_summary = kanban_summary or (lambda: {})
        self._previous: dict[str, tuple[tuple[str, int], ...]] = {}
        self._lock = threading.Lock()

    def observe(self, reading: Mapping[str, Any]) -> dict[str, Any] | None:
        """Send only for successfully refreshed provider values that changed."""
        with self._lock:
            changed: dict[str, tuple[tuple[str, int], ...]] = {}
            for provider in _PROVIDER_LABELS:
                snapshot = _provider_snapshot(reading.get(provider))
                if snapshot is None:
                    continue
                previous = self._previous.get(provider)
                if previous is None:
                    self._previous[provider] = snapshot
                elif previous != snapshot:
                    changed[provider] = snapshot

            if not changed:
                return None

            try:
                summary = self._kanban_summary()
            except Exception:
                logger.debug("usage-meter notification could not read Kanban summary", exc_info=True)
                summary = {}
            message = _format_message(changed, summary)
            try:
                result = self._send(message)
            except Exception as exc:
                logger.warning("usage-meter notification delivery failed: %s", exc)
                return {"error": str(exc)}
            if isinstance(result, Mapping) and result.get("success") is True:
                self._previous.update(changed)
            return dict(result) if isinstance(result, Mapping) else {"error": "invalid delivery result"}


def _kanban_summary() -> Mapping[str, int]:
    """Read the shared board once, returning zeros when it is unavailable."""
    try:
        from hermes_cli.kanban_db import board_stats, connect_closing

        with connect_closing() as conn:
            by_status = board_stats(conn).get("by_status", {})
        return {status: int(by_status.get(status, 0)) for status in _TRACKED_STATUSES}
    except Exception:
        logger.debug("usage-meter notification could not query Kanban", exc_info=True)
        return {}


def _send_discord(target: str, message: str) -> dict[str, Any]:
    """Use the existing gateway adapter/standalone outbound path, never the TUI."""
    if not isinstance(target, str) or not target.startswith("discord:"):
        return {"error": "usage_meter_notifications.target must be discord:<channel_id>[:<thread_id>]"}
    parts = target.split(":")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts[1:]):
        return {"error": "usage_meter_notifications.target must use numeric Discord channel/thread IDs"}
    chat_id = parts[1]
    thread_id = parts[2] if len(parts) == 3 else None

    from gateway.config import Platform, load_gateway_config
    from model_tools import _run_async
    from tools.send_message_tool import _send_via_adapter

    config = load_gateway_config()
    pconfig = config.platforms.get(Platform.DISCORD)
    if not pconfig or not pconfig.enabled:
        return {"error": "Discord is not configured/enabled in the gateway"}
    return _run_async(
        _send_via_adapter(
            Platform.DISCORD,
            pconfig,
            chat_id,
            message,
            thread_id=thread_id,
        )
    )


_notifier: UsageMeterNotifier | None = None
_notifier_target: str | None = None
_notifier_lock = threading.Lock()


def observe_usage_meter_reading(reading: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Apply the opt-in config then pass a successful dual-provider refresh through."""
    gateway = config.get("gateway") if isinstance(config, Mapping) else None
    settings = gateway.get("usage_meter_notifications") if isinstance(gateway, Mapping) else None
    if not isinstance(settings, Mapping) or settings.get("enabled") is not True:
        return None
    target = settings.get("target")
    if not isinstance(target, str) or not target:
        logger.warning("usage_meter_notifications is enabled without a Discord target")
        return {"error": "usage_meter_notifications target is missing"}

    global _notifier, _notifier_target
    with _notifier_lock:
        if _notifier is None or _notifier_target != target:
            _notifier = UsageMeterNotifier(
                send=lambda message: _send_discord(target, message),
                kanban_summary=_kanban_summary,
            )
            _notifier_target = target
        return _notifier.observe(reading)


def reset_usage_meter_notification_state() -> None:
    """Test-only reset for the process-local deduplication state."""
    global _notifier, _notifier_target
    with _notifier_lock:
        _notifier = None
        _notifier_target = None
