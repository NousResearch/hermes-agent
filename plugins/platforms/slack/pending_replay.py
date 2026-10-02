"""Current Slack workspace and native message evidence for pending replay."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from gateway.pending_native import PendingNativeInput
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.slack.pending_thread_context import (
    SlackThreadContext,
    thread_message_content,
)


def _native_slack_message(event: dict[str, Any]) -> dict[str, Any] | None:
    files = event.get("files") or []
    attachments = event.get("attachments") or []
    blocks = event.get("blocks") or []
    if (
        not isinstance(event.get("text", ""), str)
        or not isinstance(event.get("ts"), str)
        or not isinstance(event.get("user"), str)
        or not isinstance(files, list)
        or not all(isinstance(item, dict) for item in files)
        or not isinstance(attachments, list)
        or not all(isinstance(item, dict) for item in attachments)
        or not isinstance(blocks, list)
        or not all(isinstance(item, dict) for item in blocks)
    ):
        return None
    return {
        "text": event.get("text", ""),
        "user": event.get("user"),
        "ts": event.get("ts"),
        "thread_ts": event.get("thread_ts"),
        "subtype": event.get("subtype"),
        "blocks": event.get("blocks") or [],
        "attachments": event.get("attachments") or [],
        "files": [
            {key: file.get(key) for key in ("id", "name", "size", "mimetype")}
            for file in event.get("files") or []
        ],
    }


class SlackPendingReplayMixin(BasePlatformAdapter):
    _bot_user_id: str | None
    _team_bot_user_ids: dict[str, str]
    _get_client: Callable[..., Any]
    _drop_bot_sender: Callable[[dict], Awaitable[bool]]
    _prepare_slack_message: Callable[..., Awaitable[MessageEvent | None]]
    _format_thread_context: Callable[..., Awaitable[tuple[str, str]]]

    def pending_native_input(
        self, event: MessageEvent, *, thread_context: SlackThreadContext | None = None
    ) -> PendingNativeInput | None:
        if (
            not isinstance(event.raw_message, dict)
            or event.raw_message.get("ts") != event.message_id
        ):
            return None
        content = _native_slack_message(event.raw_message)
        if content is None:
            return None
        if event.source.thread_id and event.source.thread_id != event.message_id:
            if thread_context is None:
                return None
            content["thread_context"] = thread_context.to_payload()
        return PendingNativeInput.capture(event, content)

    async def revalidate_pending_event(
        self,
        event: MessageEvent,
        *,
        authorize: Callable[[SessionSource], bool] | None = None,
    ) -> MessageEvent | None:
        from slack_sdk.errors import SlackApiError, SlackRequestError
        from hermes_constants import get_hermes_home

        if authorize is not None and not authorize(event.source):
            return None
        from plugins.platforms.slack.adapter import _slack_response_payload

        native = event._pending_native_input
        team_id = event.source.scope_id
        if (
            native is None
            or not team_id
            or not event.message_id
            or not event.source.user_id
            or event.internal
            or event._merged_parts
            or event.source.is_bot
            or any(flag is not False for flag in event.media_text_inlined)
        ):
            return None
        try:
            client = self._get_client(event.source.chat_id, team_id=team_id)
            auth = _slack_response_payload(
                await asyncio.wait_for(client.auth_test(), timeout=10)
            )
            bot_id = self._team_bot_user_ids.get(team_id, self._bot_user_id)
            if (
                auth.get("ok") is not True
                or auth.get("team_id") != team_id
                or not bot_id
                or auth.get("user_id") != bot_id
            ):
                return None
            info = _slack_response_payload(
                await asyncio.wait_for(
                    client.conversations_info(channel=event.source.chat_id), timeout=10
                )
            )
            channel = info.get("channel")
            if (
                info.get("ok") is not True
                or not isinstance(channel, dict)
                or channel.get("id") != event.source.chat_id
                or channel.get("is_archived") is True
                or not channel.get("is_im")
                and channel.get("is_member") is not True
            ):
                return None
            thread_ts = native.content.get("thread_ts")
            kwargs = {
                "channel": event.source.chat_id,
                "oldest": event.message_id,
                "latest": event.message_id,
                "inclusive": True,
                "limit": 15,
            }
            if isinstance(thread_ts, str) and thread_ts != event.message_id:
                result = await asyncio.wait_for(
                    client.conversations_replies(ts=thread_ts, **kwargs), timeout=10
                )
            else:
                result = await asyncio.wait_for(
                    client.conversations_history(**kwargs), timeout=10
                )
            payload = _slack_response_payload(result)
            messages = payload.get("messages")
            if payload.get("ok") is not True or not isinstance(messages, list):
                return None
            matching = [
                item
                for item in messages
                if isinstance(item, dict) and item.get("ts") == event.message_id
            ]
            if len(matching) != 1:
                return None
            current = dict(matching[0])
            if (
                current.get("user") != event.source.user_id
                or _native_slack_message(current)
                != {
                    key: value
                    for key, value in native.content.items()
                    if key != "thread_context"
                }
                or current.get("subtype")
                not in {None, "", "file_share", "thread_broadcast", "me_message"}
            ):
                return None
            user_info = _slack_response_payload(
                await asyncio.wait_for(
                    client.users_info(user=event.source.user_id), timeout=10
                )
            )
            user = user_info.get("user")
            if (
                user_info.get("ok") is not True
                or not isinstance(user, dict)
                or user.get("id") != event.source.user_id
                or user.get("deleted") is True
                or user.get("is_bot") is not False
            ):
                return None
            current.update(
                channel=event.source.chat_id,
                team=team_id,
                channel_type="im"
                if channel.get("is_im")
                else "mpim"
                if channel.get("is_mpim")
                else "channel",
            )
            if await self._drop_bot_sender(current):
                return None
            thread_context = None
            if isinstance(thread_ts, str) and thread_ts != event.message_id:
                thread_context = await asyncio.wait_for(
                    self._revalidate_thread_context(event, client, thread_ts),
                    timeout=10,
                )
                if thread_context is None:
                    return None
            root_paths = (
                [file.path for file in thread_context.files]
                if thread_context is not None
                else []
            )
            if (
                len(current.get("files") or []) + len(root_paths)
                != len(event.media_urls)
                or event.media_urls[: len(root_paths)] != root_paths
            ):
                return None
            verified = await self._prepare_slack_message(
                current,
                team_id,
                event.source.chat_id,
                cached=event,
                replay_thread_context=thread_context,
                authorize=authorize,
            )
            if verified is None or verified.is_command():
                return None
            if (
                verified.source.chat_id,
                verified.source.thread_id,
                verified.source.scope_id,
            ) != (event.source.chat_id, event.source.thread_id, team_id):
                return None
            cache = get_hermes_home().resolve() / "cache"
            if event.media_urls and (
                not all(
                    Path(path).resolve().is_relative_to(cache)
                    for path in event.media_urls
                )
                or not native.attachments_available(event.media_urls)
            ):
                return None
            verified.timestamp = event.timestamp
            verified._pending_native_input = native
            return verified
        except (
            SlackApiError,
            SlackRequestError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            asyncio.TimeoutError,
        ):
            return None

    async def _revalidate_thread_context(
        self, event: MessageEvent, client: Any, thread_ts: str
    ) -> SlackThreadContext | None:
        from plugins.platforms.slack.adapter import _slack_response_payload

        native = event._pending_native_input
        if native is None:
            return None
        snapshot = SlackThreadContext.from_payload(native.content.get("thread_context"))
        current_messages = []
        unchanged = True
        for dependency in snapshot.messages:
            result = _slack_response_payload(
                await client.conversations_replies(
                    channel=event.source.chat_id,
                    ts=thread_ts,
                    oldest=dependency.message_id,
                    latest=dependency.message_id,
                    inclusive=True,
                    limit=15,
                )
            )
            values = result.get("messages")
            if result.get("ok") is not True or not isinstance(values, list):
                return None
            matching = [
                value
                for value in values
                if isinstance(value, dict) and value.get("ts") == dependency.message_id
            ]
            if len(matching) != 1:
                return None
            current = matching[0]
            content = thread_message_content(current)
            if any(
                content.get(key) != dependency.content.get(key)
                for key in ("user", "bot_id")
            ):
                return None
            if any(
                file.message_id == dependency.message_id for file in snapshot.files
            ) and content["files"] != dependency.content.get("files"):
                return None
            unchanged = unchanged and content == dependency.content
            current_messages.append(current)
        context = snapshot.text
        if not unchanged:
            context, _ = await self._format_thread_context(
                current_messages,
                thread_ts=thread_ts,
                current_ts=event.message_id,
                team_id=event.source.scope_id,
                channel_id=event.source.chat_id,
            )
        return SlackThreadContext(context, snapshot.messages, snapshot.files)
