"""The Slack subagent panel: one in-place Block Kit message per parent turn, a row per delegated child.

The gateway (``gateway/run_turn_subagent_panel.py``) detects ``subagent_panel_enabled`` +
``send_subagent_panel`` / ``update_subagent_panel`` and hands over a panel object carrying the rows'
controls: ``authority`` (who may press the buttons), ``stop(subagent_id)`` and ``result(subagent_id)``.
Opt-in with ``platforms.slack.extra.subagent_panel: true``. Buttons carry an opaque
``<panel_id>|<subagent_id>`` value, never a session key. Stop runs the same check as an approval
button (``gateway.approval_actor``) with ``stop`` as the command it amounts to; Result is a read and
needs no command.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.approval_actor import prompt_refusal, refusal_text
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.slack.adapter")

STOP_ACTION = "hermes_subagent_stop"
RESULT_ACTION = "hermes_subagent_result"
_PANELS_MAX = 100
_ROWS_MAX = 20  # 1 header + 2 blocks per row stays under Slack's 50-block message cap
_GOAL_MAX = 120
_MODAL_TITLE_MAX = 24
_MODAL_SECTION_MAX = 2975  # 4 sections x 2975 = the first ~11.9k characters of a result
_MODAL_SECTIONS = 4
_ICONS = {"running": "⏳", "completed": "✅", "interrupted": "⏹️"}


def t(key: str, **kwargs: Any) -> str:
    """The adapter facade's ``t``: plugins reach core i18n only through the facade's baselined import."""
    from plugins.platforms.slack.adapter import t as facade_t
    return facade_t(key, **kwargs)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _row_text(row: Any) -> str:
    icon = "❌" if row.failed else _ICONS.get(row.status, "•")
    goal = " ".join(str(row.goal or "").split())
    goal = goal if len(goal) <= _GOAL_MAX else goal[: _GOAL_MAX - 1] + "…"
    parts = [f"{icon} *{_escape(goal) or row.subagent_id}*"]
    if row.running and row.current_tool:
        parts.append(f"`{_escape(row.current_tool)}`")
    parts.append(t("platform.slack.subagent_panel.tools", count=row.tool_count))
    parts.append(_elapsed(row.elapsed_s))
    if not row.running:
        parts.append(t(f"platform.slack.subagent_panel.status_{row.status}")
                     if row.status in ("completed", "interrupted") else _escape(row.status))
    return " · ".join(parts)


class SlackSubagentPanelMixin:
    """``send_subagent_panel`` / ``update_subagent_panel`` and the Stop / Result button handlers."""

    def _subagent_panels(self) -> dict:
        # panel_id -> (panel, channel_id, message_ts, team_id)
        return self._lazy_attr("_subagent_panel_messages", dict)

    def subagent_panel_enabled(self) -> bool:
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        return self._truthy_config(extra.get("subagent_panel", False))

    def _register_subagent_panel_handlers(self, app) -> None:
        app.action(STOP_ACTION)(self._handle_subagent_stop)
        app.action(RESULT_ACTION)(self._handle_subagent_result)

    def _subagent_panel_blocks(self, panel: Any, rows: tuple) -> tuple[str, list]:
        title = t("platform.slack.subagent_panel.title", count=len(rows))
        blocks: list[dict[str, Any]] = [{"type": "context", "elements": [{"type": "mrkdwn", "text": title}]}]
        for row in rows[:_ROWS_MAX]:
            value = f"{panel.panel_id}|{row.subagent_id}"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _row_text(row)[:3000]}})
            button = (self._button(t("platform.slack.subagent_panel.stop")[:75], STOP_ACTION, value, style="danger")
                      if row.running else
                      self._button(t("platform.slack.subagent_panel.result")[:75], RESULT_ACTION, value))
            blocks.append({"type": "actions", "elements": [button]})
        return title, blocks

    async def send_subagent_panel(self, chat_id: str, panel: Any, rows: tuple, *,
                                  metadata: Optional[dict] = None) -> SendResult:
        blocked = self._outbound_blocked(chat_id, "subagent panel in")
        if blocked:
            return blocked
        result = await self._send_interactive_prompt(
            chat_id, metadata, lambda: self._subagent_panel_blocks(panel, rows), "send_subagent_panel",
            sanitize=False)
        if result.success and result.message_id:
            panels = self._subagent_panels()
            channel = str((result.raw_response or {}).get("channel") or chat_id)
            panels[panel.panel_id] = (panel, channel, result.message_id, self._metadata_team_id(metadata))
            self._trim_oldest_dict_entries(panels, _PANELS_MAX)
        return result

    async def update_subagent_panel(self, chat_id: str, message_id: str, panel: Any, rows: tuple, *,
                                    metadata: Optional[dict] = None) -> SendResult:
        entry = self._subagent_panels().get(panel.panel_id)
        if entry is None:
            return SendResult(success=False, error="Subagent panel no longer tracked")
        _panel, channel, ts, team_id = entry
        text, blocks = self._subagent_panel_blocks(panel, rows)
        try:
            await self._get_client(channel, team_id=team_id or None).chat_update(
                channel=channel, ts=ts, text=text, blocks=blocks)
        except Exception as exc:  # health: allow BLE001 -- any Web API failure ends the panel, never the turn
            logger.warning("[Slack] Subagent panel update failed: %s", exc)
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=ts)

    async def _subagent_press(self, ack, body, action, kind: str, command: Optional[str]):
        """Ack, authorize, resolve the pressed row: ``(panel, subagent_id, channel_id, user_id, thread_ts,
        team_id)``, or None after telling the presser privately why not."""
        started = await self._begin_interaction(ack, body, action, kind)
        if started is None:
            return None
        team_id, _action_id, value, message, _ts, channel_id, _name, user_id = started
        panel_id, _, subagent_id = str(value).partition("|")
        entry = self._subagent_panels().get(panel_id)
        thread_ts = (message or {}).get("thread_ts")
        if entry is None or not subagent_id:
            await self._get_client(channel_id, team_id=team_id or None).chat_postEphemeral(
                channel=channel_id, user=user_id, text=t("platform.slack.subagent_panel.gone"), thread_ts=thread_ts)
            return None
        panel = entry[0]
        if await self._refuse_other_requester(
                prompt_refusal(panel.authority, user_id, extra=self.config.extra, command=command), body):
            return None
        return panel, subagent_id, channel_id, user_id, thread_ts, team_id

    async def _handle_subagent_stop(self, ack, body, action) -> None:
        pressed = await self._subagent_press(ack, body, action, "subagent stop", "stop")
        if pressed is None:
            return
        panel, subagent_id, channel_id, user_id, thread_ts, team_id = pressed
        key = "platform.slack.subagent_panel.stop_sent" if panel.stop(subagent_id) else \
            "platform.slack.subagent_panel.not_running"
        await self._get_client(channel_id, team_id=team_id or None).chat_postEphemeral(
            channel=channel_id, user=user_id, text=t(key), thread_ts=thread_ts)

    async def _handle_subagent_result(self, ack, body, action) -> None:
        pressed = await self._subagent_press(ack, body, action, "subagent result", None)
        if pressed is None:
            return
        panel, subagent_id, channel_id, user_id, thread_ts, team_id = pressed
        client = self._get_client(channel_id, team_id=team_id or None)
        text = panel.result(subagent_id)
        if text is None:
            await client.chat_postEphemeral(channel=channel_id, user=user_id,
                                            text=t("platform.slack.subagent_panel.not_running"), thread_ts=thread_ts)
            return
        text = text or t("platform.slack.subagent_panel.result_empty")
        limit = _MODAL_SECTION_MAX
        sections = [{"type": "section", "text": {"type": "plain_text", "text": text[i:i + limit], "emoji": False}}
                    for i in range(0, min(len(text), limit * _MODAL_SECTIONS), limit)]
        view = {"type": "modal",
                "title": {"type": "plain_text", "text": t("platform.slack.subagent_panel.result_title")[:_MODAL_TITLE_MAX]},
                "close": {"type": "plain_text", "text": t("platform.slack.subagent_panel.close")[:_MODAL_TITLE_MAX]},
                "blocks": sections}
        try:
            await client.views_open(trigger_id=str(body.get("trigger_id") or ""), view=view)
        except Exception:  # health: allow BLE001 -- any Web API failure leaves the panel usable for a retry
            logger.warning("[Slack] Could not open the subagent result modal", exc_info=True)
