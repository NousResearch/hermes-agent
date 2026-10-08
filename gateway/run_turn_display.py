"""Resolve per-platform display, progress and streaming-surface settings for a gateway turn.

Gateway config helpers remain late-bound through ``gateway.run``.
"""

from __future__ import annotations

import logging
import queue
from contextlib import suppress
from typing import Any, List, TYPE_CHECKING

from agent.i18n import t
from gateway.config import Platform

if TYPE_CHECKING:
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayTurnDisplayMixin:
    """Resolve the display state used when the turn context is built."""

    def _run_agent_display_settings(self, source: SessionSource) -> "GatewayRunner._RunAgentDisplay":
        """Resolve per-platform display, progress, status and streaming-surface settings for a turn."""
        from gateway.run import (
            _gateway_platform_value, _has_platform_display_override, _load_gateway_config,
            _platform_config_key,
        )
        from agent.secret_scope import get_secret
        from gateway.display_config import resolve_display_setting, resolve_tool_progress
        from gateway.status_phrases import choose_status_phrase, resolve_status_phrase_catalog
        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)
        enabled_toolsets, disabled_toolsets = self._resolve_turn_toolsets(user_config, source, platform_key)
        adapter = self._delivery_adapter_for(source)
        # Tool preview length (0 = no limit) and friendly tool labels (default on), per-platform.
        for _setter, _setting, _default, _cast in (
            ("set_tool_preview_max_len", "tool_preview_length", 0, lambda v: int(v) if v else 0),
            ("set_friendly_tool_labels", "friendly_tool_labels", True, bool),
        ):
            with suppress(Exception):
                from agent import display as _agent_display
                _val = resolve_display_setting(user_config, platform_key, _setting, _default)
                getattr(_agent_display, _setter)(_cast(_val))

        # Resolve the mode and its provenance together: null inherits, tier off is not intent.
        # A raw os.getenv here reads whichever profile's env loaded last under multiplexing
        # (#116898); get_secret resolves through the active profile's scope instead.
        progress_mode, _tool_progress_explicit = resolve_tool_progress(
            user_config, platform_key, get_secret("HERMES_TOOL_PROGRESS_MODE"),
        )
        # "accumulate" (edit one bubble) or "separate" (one msg per tool)
        progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
        _generic_status_recent: List[str] = []
        _generic_status_catalog = resolve_status_phrase_catalog(user_config, platform_key)

        def _display_surface_mode(
            setting: str, *, default: bool = False,
            require_platform_override_for: set[Any] | None = None, allow_generic: bool = False,
        ) -> str:
            """Return off|raw|generic for a gateway visibility surface."""
            if require_platform_override_for:
                current_platform = _gateway_platform_value(source.platform)
                platform_only = {_gateway_platform_value(item) for item in require_platform_override_for}
                if (
                    current_platform in platform_only
                    and not _has_platform_display_override(user_config, platform_key, setting)
                ):
                    return "off"
            value = resolve_display_setting(user_config, platform_key, setting, default)
            if isinstance(value, str) and value.strip().lower() == "generic":
                return "generic" if allow_generic else "off"
            return "raw" if bool(value) else "off"

        def _generic_status_phrase(kind: str, *, tool_name: str | None = None, preview: str | None = None, args: Any = None) -> str:
            try:
                return choose_status_phrase(
                    kind, tool_name=tool_name, preview=preview, args=args,
                    recent=_generic_status_recent, catalog=_generic_status_catalog,
                )
            except Exception as _phrase_err:
                logger.debug("generic status phrase selection failed: %s", _phrase_err)
                return (t("gateway.progress.status_fallback_long")
                        if kind in {"heartbeat", "waiting", "long_running", "status"}
                        else t("gateway.progress.status_fallback_short"))

        # Webhooks can't edit messages, so tool progress / log mode are off there.
        is_webhook = source.platform == Platform.WEBHOOK
        tool_progress_enabled = progress_mode not in {"off", "log"} and not is_webhook
        # Live status for text-rendering typing indicators (Slack); independent of tool_progress.
        _live_status_mode = resolve_display_setting(user_config, platform_key, "live_status", "full")
        _live_status_adapter = (
            adapter if getattr(adapter, "supports_status_text", False) and _live_status_mode != "off" else None
        )
        # "log" mode: tool calls go to ~/.hermes/logs/tool_calls.log instead of the chat. Gateway-only.
        log_mode_enabled = progress_mode == "log" and not is_webhook
        # Interim assistant messages and thinking_progress are independent of tool progress (same
        # queue). Mattermost requires a per-platform opt-in: scratch text leaks into public threads.
        interim_assistant_messages_mode = _display_surface_mode(
            "interim_assistant_messages", default=True, require_platform_override_for={Platform.MATTERMOST},
        )
        interim_assistant_messages_enabled = not is_webhook and interim_assistant_messages_mode != "off"
        _thinking_enabled = _display_surface_mode(
            "thinking_progress", default=False, require_platform_override_for={Platform.MATTERMOST},
        ) != "off"
        # Slack-native task cards need the progress queue even with text tool_progress off.
        # Slack-native task cards (#29483): when the Slack adapter's opt-in is set, tool progress renders as
        # native plan/task cards via chat.startStream — the progress queue is needed even though Slack keeps
        # ordinary text tool_progress off by default (requiring both flags would silently leave the native
        # feature inactive).
        # Cards are still tool progress. Slack's TIER default (``off``) only quiets the text lane so
        # cards stay on for unconfigured installs, but an operator who WRITES ``tool_progress: off``
        # (global, platform override, or legacy overrides) has asked for no tool progress at all and
        # gets no cards either. Every other explicit mode keeps the card lane.
        _native_slack_task_cards = False
        if (
            source.platform == Platform.SLACK
            and hasattr(adapter, "native_task_cards_enabled")
            and not (_tool_progress_explicit and progress_mode == "off")
        ):
            try:
                _native_slack_task_cards = bool(adapter.native_task_cards_enabled())
            except Exception:
                logger.debug("Slack native task-card config check failed", exc_info=True)
        return self._RunAgentDisplay(
            user_config=user_config, platform_key=platform_key, enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets, resolve_display_setting=resolve_display_setting,
            progress_mode=progress_mode, progress_grouping=progress_grouping,
            _display_surface_mode=_display_surface_mode,
            tool_progress_enabled=tool_progress_enabled, _live_status_mode=_live_status_mode,
            _live_status_adapter=_live_status_adapter, log_mode_enabled=log_mode_enabled,
            log_queue=queue.Queue() if log_mode_enabled else None,
            interim_assistant_messages_enabled=interim_assistant_messages_enabled,
            _thinking_enabled=_thinking_enabled, _native_slack_task_cards=_native_slack_task_cards,
            needs_progress_queue=tool_progress_enabled or _thinking_enabled or _native_slack_task_cards,
            _generic_status_phrase=_generic_status_phrase,
        )
