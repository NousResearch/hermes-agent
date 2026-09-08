"""Exact, bounded reply context for the Teams adapter."""
from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)
_MAX_REPLY_CONTEXT_CHARS = 4000


def _teams_message_body_text(payload: Any) -> str:
    """Convert a Microsoft Graph chatMessage body into prompt-safe plain text."""
    if not isinstance(payload, dict):
        return ""
    body = payload.get("body")
    if not isinstance(body, dict):
        return ""
    content = str(body.get("content") or "")
    if not content:
        return ""

    if str(body.get("contentType") or "").casefold() == "html":
        content = re.sub(
            r"(?is)<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
            "",
            content,
        )
        content = re.sub(r"(?i)<li\b[^>]*>", "\n- ", content)
        content = re.sub(
            r"(?i)<br\s*/?>|</(?:p|div|li|h[1-6]|ul|ol)>",
            "\n",
            content,
        )
        content = re.sub(r"<[^>]+>", "", content)
        content = html.unescape(content).replace("\xa0", " ")

    lines = [" ".join(line.split()) for line in content.splitlines()]
    text = "\n".join(line for line in lines if line).strip()
    return text[:_MAX_REPLY_CONTEXT_CHARS]


def _record_outbound_reply_context(
    chat_id: str,
    message_id: str,
    content: str,
) -> None:
    """Best-effort index for replies to confirmed proactive Teams messages."""
    try:
        from gateway.cron_reply_context import record_reply_context

        record_reply_context(
            "teams",
            str(chat_id),
            content,
            thread_id=str(message_id),
            message_id=str(message_id),
        )
    except Exception:
        logger.debug(
            "[teams] failed to record outbound reply context",
            exc_info=True,
        )



class TeamsReplyContextMixin:
    @classmethod
    def _teams_thread_id(cls, activity: Any) -> Optional[str]:
        """Return the root Teams activity ID when this message is a reply."""
        channel_data = getattr(activity, "channel_data", None)
        candidates = (
            getattr(activity, "reply_to_id", None),
            getattr(activity, "replyToId", None),
            cls._nested_value(channel_data, "replyToId"),
            cls._nested_value(channel_data, "reply_to_id"),
            cls._nested_value(channel_data, "message", "replyToId"),
            cls._nested_value(channel_data, "message", "reply_to_id"),
            cls._nested_value(channel_data, "legacy", "replyToId"),
            cls._nested_value(channel_data, "legacy", "reply_to_id"),
        )
        for value in candidates:
            if isinstance(value, (str, int, float)) and value not in (None, ""):
                text = str(value).strip()
                if text:
                    return text
        return None

    @staticmethod
    def _nested_value(value: Any, *keys: str) -> Any:
        current = value
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @staticmethod
    def _thread_id_from_conversation_id(conversation_id: Any) -> Optional[str]:
        if not isinstance(conversation_id, str):
            return None
        marker = ";messageid="
        if marker not in conversation_id:
            return None
        message_id = conversation_id.split(marker, 1)[1].strip()
        return message_id or None

    @staticmethod
    def _cron_reply_context(conversation_id: str, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        try:
            from gateway.cron_reply_context import find_cron_reply_context

            return find_cron_reply_context(
                "teams",
                str(conversation_id),
                thread_id=str(thread_id) if thread_id else None,
            )
        except Exception:
            logger.debug("[teams] failed to load cron reply context", exc_info=True)
            return None

    async def _fetch_parent_message_text(
        self,
        activity: Any,
        thread_id: str,
    ) -> Optional[str]:
        """Fetch an uncached channel parent from Graph using its Teams message ID."""
        channel_data = getattr(activity, "channel_data", None)
        team_id = (
            self._nested_value(channel_data, "team", "aadGroupId")
            or self._nested_value(channel_data, "team", "aad_group_id")
            or self._nested_value(channel_data, "team", "id")
        )
        channel_id = self._nested_value(channel_data, "channel", "id")
        conversation_id = str(getattr(getattr(activity, "conversation", None), "id", "") or "")
        if not channel_id and "@thread.tacv2" in conversation_id:
            channel_id = conversation_id.split(";messageid=", 1)[0]

        team_id = str(team_id or "").strip()
        channel_id = str(channel_id or "").strip()
        message_id = str(thread_id or "").strip()
        if not team_id or not channel_id or not message_id:
            logger.debug(
                "[teams] cannot fetch parent context without team, channel, and message IDs"
            )
            return None

        try:
            graph_client = self._get_reply_context_graph_client()
            path = (
                f"/teams/{quote(team_id, safe='')}/channels/"
                f"{quote(channel_id, safe='')}/messages/{quote(message_id, safe='')}"
            )
            payload = await graph_client.get_json(path)
            text = _teams_message_body_text(payload)
            if text:
                logger.debug(
                    "[teams] fetched reply context from Graph for channel=%s message=%s",
                    channel_id,
                    message_id,
                )
                return text
        except Exception as exc:
            logger.warning(
                "[teams] failed to fetch parent message context from Graph: %s",
                exc,
            )
        return None

    def _get_reply_context_graph_client(self) -> Any:
        if self._reply_context_graph_client is not None:
            return self._reply_context_graph_client

        from tools.microsoft_graph_auth import GraphCredentials, MicrosoftGraphTokenProvider
        from tools.microsoft_graph_client import MicrosoftGraphClient

        credentials = GraphCredentials(
            tenant_id=str(self._tenant_id),
            client_id=str(self._client_id),
            client_secret=str(self._client_secret),
        )
        provider = MicrosoftGraphTokenProvider(credentials, timeout=10.0)
        self._reply_context_graph_client = MicrosoftGraphClient(
            provider,
            timeout=10.0,
            max_retries=0,
            user_agent="Hermes-Agent/teams-reply-context",
        )
        return self._reply_context_graph_client
