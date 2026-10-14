"""Typed, fail-closed Discord interaction contract for plugins.

This API is opt-in. Legacy plugin slash commands continue through the existing Gateway path.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger("hermes_cli.plugins")


@dataclass(frozen=True)
class VerifiedDiscordUser:
    """Discord actor identity supplied only after the adapter's slash-auth gate succeeds."""

    id: str
    username: str
    display_name: str


@dataclass(frozen=True)
class DiscordInteractionContext:
    """Transport-authenticated interaction metadata; this is identity, not plugin authorization."""

    id: str
    user: VerifiedDiscordUser
    server_id: str | None
    channel_id: str
    command_name: str

    def __post_init__(self) -> None:
        _require_snowflake(self.id, "interaction id")
        _require_snowflake(self.user.id, "user id")
        _require_snowflake(self.channel_id, "channel id")
        if self.server_id is not None:
            _require_snowflake(self.server_id, "server id")
        if not self.command_name or len(self.command_name) > 32:
            raise ValueError("invalid Discord command name")


class DiscordInteractionVisibility(StrEnum):
    PRIVATE = "private"
    PUBLIC = "public"


@dataclass(frozen=True)
class DiscordInteractionReply:
    """Explicitly routed response. Plain strings are deliberately not accepted."""

    visibility: DiscordInteractionVisibility
    content: str

    @classmethod
    def private(cls, content: str) -> DiscordInteractionReply:
        return cls(DiscordInteractionVisibility.PRIVATE, content)

    @classmethod
    def public(cls, content: str) -> DiscordInteractionReply:
        return cls(DiscordInteractionVisibility.PUBLIC, content)


class PrivateInteractionUnavailable(RuntimeError):
    """The private interaction transport could not be established or used."""


def _require_snowflake(value: Any, label: str) -> str:
    text = str(value) if value is not None else ""
    if not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"invalid Discord {label}")
    return text


def interaction_context_from_sdk(interaction: Any, command_name: str) -> DiscordInteractionContext:
    """Validate SDK-provided IDs and construct the trusted context after adapter authorization."""
    user = getattr(interaction, "user", None)
    if user is None:
        raise ValueError("missing Discord interaction user")
    channel_id = getattr(interaction, "channel_id", None)
    if channel_id is None:
        channel_id = getattr(getattr(interaction, "channel", None), "id", None)
    server_id = getattr(interaction, "guild_id", None)
    if server_id is None:
        server_id = getattr(getattr(getattr(interaction, "channel", None), "guild", None), "id", None)
    identity = VerifiedDiscordUser(
        id=_require_snowflake(getattr(user, "id", None), "user id"),
        username=str(getattr(user, "name", "") or ""),
        display_name=str(getattr(user, "display_name", None) or getattr(user, "name", "") or ""),
    )
    return DiscordInteractionContext(
        id=_require_snowflake(getattr(interaction, "id", None), "interaction id"),
        user=identity,
        server_id=_require_snowflake(server_id, "server id") if server_id is not None else None,
        channel_id=_require_snowflake(channel_id, "channel id"),
        command_name=str(command_name or ""),
    )


def _normalize_replies(value: Any) -> list[DiscordInteractionReply]:
    if value is None:
        return []
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    replies: list[DiscordInteractionReply] = []
    for item in values:
        if isinstance(item, DiscordInteractionReply):
            reply = item
        elif isinstance(item, Mapping):
            try:
                reply = DiscordInteractionReply(
                    visibility=DiscordInteractionVisibility(item["visibility"]),
                    content=item["content"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TypeError("invalid plugin interaction response record") from exc
        else:
            raise TypeError("plugin interaction handlers must return explicitly routed replies")
        if not isinstance(reply.content, str) or not reply.content.strip():
            raise ValueError("plugin interaction responses must contain non-empty text")
        replies.append(reply)
    return replies


async def dispatch_plugin_interaction(
    *, interaction: Any,
    adapter: Any,
    command_name: str,
    raw_args: str,
    handler: Callable[[str, DiscordInteractionContext], Any],
) -> bool:
    """Run an authorized plugin interaction without ever falling back to public delivery.

    The adapter must run its server-side slash authorization gate before calling this function.
    An ephemeral defer is mandatory before plugin execution; failure rejects the invocation. Private
    replies are fully delivered before any explicitly public narrative is sent, so a private-send
    failure aborts the remaining delivery rather than falling through to the public channel.
    """
    try:
        context = interaction_context_from_sdk(interaction, command_name)
    except (TypeError, ValueError):
        await _respond_invalid_context(interaction)
        return False

    try:
        await interaction.response.defer(ephemeral=True)
    except Exception as exc:  # health: allow BLE001 -- Discord SDK transport exceptions are provider-defined.
        logger.warning("Discord interaction %s: private acknowledgement unavailable; command rejected", context.id)
        raise PrivateInteractionUnavailable("private interaction acknowledgement failed") from exc

    try:
        if inspect.iscoroutinefunction(handler):
            result = await handler(raw_args, context)
        else:
            result = await asyncio.to_thread(handler, raw_args, context)
            if inspect.isawaitable(result):
                result = await result
        replies = _normalize_replies(result)
    except Exception:
        logger.exception("Discord plugin interaction handler failed (%s)", context.command_name)
        try:
            await interaction.edit_original_response(content="The command failed; no public response was sent.")
        except Exception:
            logger.exception("Discord interaction %s: private failure notice could not be delivered", context.id)
        return False

    private_replies = [r for r in replies if r.visibility is DiscordInteractionVisibility.PRIVATE]
    public_replies = [r for r in replies if r.visibility is DiscordInteractionVisibility.PUBLIC]

    # Deliver all sensitive output on the ephemeral interaction token before any public output.
    try:
        for index, reply in enumerate(private_replies):
            if index == 0:
                await interaction.edit_original_response(content=reply.content)
            else:
                await interaction.followup.send(content=reply.content, ephemeral=True)
    except Exception as exc:  # health: allow BLE001 -- Discord SDK transport exceptions are provider-defined.
        logger.warning("Discord interaction %s: private reply failed; public delivery suppressed", context.id)
        raise PrivateInteractionUnavailable("private interaction reply failed; public delivery suppressed") from exc

    if not private_replies:
        try:
            await interaction.delete_original_response()
        except Exception as exc:  # health: allow BLE001 -- cleanup errors from the Discord SDK are provider-defined.
            # The original response is itself ephemeral; cleanup failure cannot make it public.
            logger.debug("Discord interaction %s: ephemeral acknowledgement cleanup failed: %s", context.id, exc)

    for reply in public_replies:
        result = await adapter.send(context.channel_id, reply.content)
        if not getattr(result, "success", False):
            logger.warning("Discord interaction %s: explicit public reply failed", context.id)
            return False
    return True


async def _respond_invalid_context(interaction: Any) -> None:
    """Reject malformed interaction metadata privately when the SDK still permits a response."""
    response = getattr(interaction, "response", None)
    send_message = getattr(response, "send_message", None)
    is_done = getattr(response, "is_done", None)
    if not callable(send_message) or (callable(is_done) and is_done()):
        return
    try:
        result = send_message("Invalid interaction context.", ephemeral=True)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("Discord interaction context invalid; private rejection unavailable")
