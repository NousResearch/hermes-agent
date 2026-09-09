"""Slack commands methods; SDK and mutable dependencies remain on the facade."""

from typing import Any, Optional
try:
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncApp = AsyncWebClient = Any


class SlackCommandsMixin:
    async def _handle_slash_command(self, command: dict) -> None:
        """Slash commands: native ``/<command> [args]`` for every COMMAND_REGISTRY entry, or
        ``/hermes <subcommand> [args]``; other text after ``/hermes`` is a regular message."""
        from . import adapter as _adapter

        user_id = command.get("user_id", "")
        channel_id = command.get("channel_id", "")
        team_id = command.get("team_id", "")
        if team_id and channel_id:
            self._remember_channel_team(channel_id, team_id)
        text = self._slash_command_text(command)
        thread_id = self._slash_thread_id(command)
        is_dm = str(channel_id).startswith("D")
        if is_dm and self._slack_disable_dms():
            _adapter.logger.info(
                "[Slack] Ignoring slash command from DM because Slack DMs are disabled: channel=%s user=%s",
                channel_id, user_id)
            return
        source = self.build_source(
            chat_id=channel_id, chat_type="dm" if is_dm else "group", user_id=user_id,
            thread_id=thread_id, scope_id=team_id or None)
        source.is_one_to_one = is_dm
        source.message_is_edit = False
        event = _adapter.MessageEvent(
            text=text,
            message_type=(_adapter.MessageType.COMMAND if text.startswith("/") else _adapter.MessageType.TEXT),
            source=source, raw_message=command)
        # Stash response_url so the first reply for this channel+user goes ephemeral. COMMAND
        # events only: free-form "/hermes <question>" replies must stay public.
        response_url = command.get("response_url", "")
        if response_url and user_id and channel_id and text.startswith("/"):
            self._stash_slash_context(team_id, channel_id, user_id, response_url)
        # ContextVar lets send() match the right response_url under
        # concurrent slashes from multiple users.
        _slash_user_id_token = _adapter._slash_user_id.set(user_id or None)
        try:
            await self.handle_message(event)
        finally:
            _adapter._slash_user_id.reset(_slash_user_id_token)

    @staticmethod
    def _slash_command_text(command: dict) -> str:
        """Gateway message text for a slash payload. Native slashes keep Slack's raw argument
        payload verbatim (internal/trailing spacing). ``/hermes`` (or a missing ``command``) maps
        ``<subcommand> [args]`` via the registry, else free-form text is a regular question."""
        slash_name = (command.get("command") or "").lstrip("/").strip()
        raw_text = str(command.get("text") or "")
        if slash_name not in {"hermes", ""}:
            return f"/{slash_name}" if not raw_text else f"/{slash_name} {raw_text}"
        legacy_text = raw_text.strip()
        from hermes_cli.commands_platforms import slack_subcommand_map
        subcommand_map = slack_subcommand_map()
        subcommand_map["compact"] = "/compress"
        first_word = legacy_text.split()[0] if legacy_text.split() else ""
        if first_word in subcommand_map:
            rest = legacy_text[len(first_word) :].strip()
            mapped = subcommand_map[first_word]
            return f"{mapped} {rest}".strip() if rest else mapped
        return legacy_text or "/help"

    @staticmethod
    def _slash_thread_id(command: dict) -> Optional[str]:
        """Thread anchor for a slash payload so session-scoped commands (``/model``)
        hit the same thread session. Shape varies by surface: top-level or nested
        ``message``/``container``; ``thread_ts`` preferred over ``message_ts``."""
        nested = (command.get(k) for k in ("message", "container"))
        candidates = [command] + [n for n in nested if isinstance(n, dict)]
        for ts_key in ("thread_ts", "message_ts"):
            for payload in candidates:
                value = payload.get(ts_key)
                if value:
                    return str(value)
        return None

    def _stash_slash_context(
        self, team_id: str, channel_id: str, user_id: str, response_url: str) -> None:
        """Remember a slash ``response_url`` (+ user for the postEphemeral fallback),
        bounded: TTL-purge then oldest-first eviction, since contexts whose reply
        never happens are otherwise never looked up."""
        from . import adapter as _adapter

        context_key = (
            (str(team_id), str(channel_id), str(user_id))
            if team_id
            else (str(channel_id), str(user_id)))
        self._slash_command_contexts[context_key] = {
            "response_url": response_url, "user_id": user_id, "ts": _adapter.time.monotonic()}
        if len(self._slash_command_contexts) <= self._SLASH_CTX_MAX:
            return
        self._purge_stale_slash_contexts()
        if len(self._slash_command_contexts) > self._SLASH_CTX_MAX:
            excess = len(self._slash_command_contexts) - self._SLASH_CTX_MAX // 2
            for old_key in sorted(
                self._slash_command_contexts, key=lambda k: self._slash_command_contexts[k]["ts"]
            )[:excess]:
                del self._slash_command_contexts[old_key]
