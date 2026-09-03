"""Slack slash-command intake and invoker context."""

import contextvars
import time
from typing import Any, Optional


# ContextVar carrying the user_id of the slash-command invoker.
# Set in handle_slash_command, read in SlackAdapter.send() to match the correct
# stashed response_url when multiple users issue commands on the same channel
# concurrently. ContextVars propagate to child asyncio tasks.
_slash_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_slash_user_id",
    default=None,
)


async def handle_slash_command(
    self: Any,
    command: dict,
    *,
    logger: Any,
    MessageEvent: Any,
    MessageType: Any,
) -> None:
    """Build and dispatch one Slack slash-command event."""
    slash_name = (command.get("command") or "").lstrip("/").strip()
    raw_text = str(command.get("text") or "")
    text = raw_text
    user_id = command.get("user_id", "")
    channel_id = command.get("channel_id", "")
    team_id = command.get("team_id", "")

    # Track which workspace owns this channel
    if team_id and channel_id:
        self._remember_channel_team(channel_id, team_id)

    if slash_name in {"hermes", ""}:
        # Legacy /hermes <subcommand> [args] routing + free-form questions.
        # Empty slash_name falls into this branch for backward compat
        # with any caller that didn't populate command["command"].
        legacy_text = raw_text.strip()
        from hermes_cli.commands import slack_subcommand_map

        subcommand_map = slack_subcommand_map()
        subcommand_map["compact"] = "/compress"
        # Guard against whitespace-only text where ``text`` is truthy but
        # ``text.split()`` returns ``[]`` (e.g. user sends ``/hermes   ``).
        parts = legacy_text.split() if legacy_text else []
        first_word = parts[0] if parts else ""
        if first_word in subcommand_map:
            rest = legacy_text[len(first_word) :].strip()
            text = (
                f"{subcommand_map[first_word]} {rest}".strip()
                if rest
                else subcommand_map[first_word]
            )
        elif legacy_text:
            text = legacy_text  # Treat as a regular question
        else:
            text = "/help"
    else:
        # Native slash — /<slash_name> [args]. Route directly through the
        # gateway command dispatcher by prepending the slash. Only the command
        # delimiter is nonsemantic: preserve Slack's raw argument payload,
        # including meaningful internal/trailing spacing.
        text = f"/{slash_name}" if not raw_text else f"/{slash_name} {raw_text}"

    # Slack slash commands can originate from DMs or shared channels.
    # Preserve DM semantics only for DM channel IDs; shared channels must
    # keep group semantics so different users do not collide into one
    # session key.
    #
    # If Slack includes thread context in the slash payload, preserve it so
    # session-scoped commands like `/model <name>` affect exactly the same
    # Slack thread/session that normal messages in that thread use. Without
    # this, `/model` from a thread is keyed only by channel+user, so the
    # next threaded message misses the override and appears to require
    # --global. Slack's native slash-command payloads vary by surface, so
    # accept a few known shapes (top-level and nested, preferring a real
    # parent-thread anchor over a fallback message timestamp) and otherwise
    # leave thread_id unset; users can always use the message-based
    # ``!model ...`` thread command path, which carries event.thread_ts.
    thread_id = None
    _thread_candidates = [command]
    for _nested_key in ("message", "container"):
        _nested = command.get(_nested_key)
        if isinstance(_nested, dict):
            _thread_candidates.append(_nested)
    for _ts_key in ("thread_ts", "message_ts"):
        for _payload in _thread_candidates:
            _value = _payload.get(_ts_key)
            if _value:
                thread_id = str(_value)
                break
        if thread_id:
            break
    is_dm = str(channel_id).startswith("D")
    if is_dm and self._slack_disable_dms():
        logger.info(
            "[Slack] Ignoring slash command from DM because Slack DMs are disabled: channel=%s user=%s",
            channel_id,
            user_id,
        )
        return
    source = self.build_source(
        chat_id=channel_id,
        chat_type="dm" if is_dm else "group",
        user_id=user_id,
        thread_id=thread_id,
        scope_id=team_id or None,
    )
    source.is_one_to_one = is_dm
    source.message_is_edit = False

    event = MessageEvent(
        text=text,
        message_type=(
            MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        ),
        source=source,
        raw_message=command,
    )

    # Stash the Slack response_url so the first reply for this
    # channel+user can be routed ephemerally (replaces the initial
    # "Running /cmd…" ack shown by handle_hermes_command).
    # Only stash for COMMAND events (text starts with "/") — free-form
    # questions via "/hermes <question>" must produce public replies so
    # the whole channel can see the agent's answer.
    response_url = command.get("response_url", "")
    if response_url and user_id and channel_id and text.startswith("/"):
        context_key = (
            (str(team_id), str(channel_id), str(user_id))
            if team_id
            else (str(channel_id), str(user_id))
        )
        self._slash_command_contexts[context_key] = {
            "response_url": response_url,
            # Kept for the chat.postEphemeral fallback when response_url
            # delivery fails — postEphemeral needs an explicit user.
            "user_id": user_id,
            "ts": time.monotonic(),
        }
        if len(self._slash_command_contexts) > self._SLASH_CTX_MAX:
            # TTL cleanup normally runs on lookup, but contexts stashed
            # for replies that never happen (agent error, ephemeral-only
            # command) are never looked up — purge expired entries, then
            # fall back to oldest-stash-first eviction if still over cap.
            now_ts = time.monotonic()
            for stale_key in [
                k
                for k, v in self._slash_command_contexts.items()
                if now_ts - v["ts"] > self._SLASH_CTX_TTL
            ]:
                del self._slash_command_contexts[stale_key]
            if len(self._slash_command_contexts) > self._SLASH_CTX_MAX:
                excess = len(self._slash_command_contexts) - self._SLASH_CTX_MAX // 2
                for old_key in sorted(
                    self._slash_command_contexts,
                    key=lambda k: self._slash_command_contexts[k]["ts"],
                )[:excess]:
                    del self._slash_command_contexts[old_key]

    # Set the ContextVar so send() can match the correct stashed
    # response_url even when multiple users slash concurrently.
    _slash_user_id_token = _slash_user_id.set(user_id or None)
    try:
        await self.handle_message(event)
    finally:
        _slash_user_id.reset(_slash_user_id_token)
