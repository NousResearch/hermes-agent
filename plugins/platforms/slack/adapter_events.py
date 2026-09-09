"""Slack events methods; SDK and mutable dependencies remain on the facade."""

from typing import Any, Dict, List, Optional, Tuple
from gateway.platforms.base import MessageEvent, MessageType
try:
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncApp = AsyncWebClient = Any


class SlackEventsMixin:
    def _agent_view_event_fields(self, event: dict, body: Optional[dict]) -> Dict[str, str]:
        """``{context_channel_id, user_id, team_id}`` (str, "" when absent) from an Agent-view
        lifecycle event."""
        from . import adapter as _adapter

        context = event.get("context") or event.get("app_context") or {}
        user_id = event.get("user") or event.get("user_id") or ""
        team_id = self._event_team_id(event, body)
        return {
            "context_channel_id": self._context_channel_id(context),
            "user_id": _adapter._str_or_empty(user_id), "team_id": _adapter._str_or_empty(team_id)}

    async def _handle_app_home_opened(self, event: dict, body: Optional[dict] = None) -> None:
        """Handle Slack Agent DM-open lifecycle events without producing replies."""
        from . import adapter as _adapter

        if event.get("tab") != "messages":
            return
        channel_id = event.get("channel") or event.get("channel_id") or ""
        fields = self._agent_view_event_fields(event, body)
        if fields["team_id"] and channel_id:
            self._remember_channel_team(channel_id, fields["team_id"])
        metadata = {
            "channel_id": _adapter._str_or_empty(channel_id), "user_id": fields["user_id"],
            "team_id": fields["team_id"], "context_channel_id": fields["context_channel_id"]}
        self._cache_agent_view_context(metadata)
        # ``app_home_opened`` (tab == "messages") replaces ``assistant_thread_started`` in
        # Slack's Agent experience; lifecycle only (no welcome message, no agent loop).
        self._seed_dm_session(
            metadata,
            thread_ts=None,
            fail_log=(
                "[Slack] Failed to seed agent DM session for %s", metadata.get("channel_id", "")))
        await self._set_assistant_suggested_prompts(
            metadata["channel_id"], team_id=metadata["team_id"])

    async def _handle_slack_reaction(self, event: dict, removed: bool = False) -> None:
        """Forward reactions as a synthetic ``reaction:<added|removed>:<emoji>`` message
        (Feishu/Photon convention) from the reactor in the reacted-to thread, so the normal auth
        gate applies. Hooks fire for every non-self reaction; agent routing is opt-in via
        ``reaction_triggers`` and, without an explicit allowlist, only on the bot's own messages."""
        from . import adapter as _adapter

        item = event.get("item") or {}
        if item.get("type") != "message":
            return
        channel_id = item.get("channel")
        msg_ts = item.get("ts")
        reaction_name = event.get("reaction") or ""
        user_id = event.get("user")
        if not channel_id or not msg_ts or not user_id or not reaction_name:
            return
        # Self-reactions (e.g. :eyes: lifecycle marker) would feed back.
        if self._bot_user_id and user_id == self._bot_user_id:
            return
        team_id = self._channel_team.get(channel_id) or ""
        if not team_id and self._team_clients:
            team_id = next(iter(self._team_clients))
        client = self._team_clients.get(team_id) if team_id else None
        action = "removed" if removed else "added"
        # Hooks fire before the opt-in gate so consumers see every human
        # reaction. getattr: tests build adapters via object.__new__.
        reaction_handler = getattr(self, "_reaction_handler", None)
        if reaction_handler is not None:
            try:
                await reaction_handler({
                    "platform": "slack", "event_name": f"reaction:{action}",
                    "reaction": reaction_name, "user_id": user_id,
                    "item_user_id": event.get("item_user"), "item_type": item.get("type"),
                    "channel_id": channel_id, "message_ts": msg_ts, "team_id": team_id,
                    "event_ts": event.get("event_ts"), "raw_event": event})
            except Exception:  # pragma: no cover - hook contract is non-blocking
                _adapter.logger.debug("[Slack] reaction hook forwarding failed", exc_info=True)
        # None → routing disabled; empty set → all emoji; non-empty → allowlist.
        triggers = self._slack_reaction_triggers()
        if triggers is None:
            return
        explicit_allowlist = bool(triggers)
        if explicit_allowlist and reaction_name.strip(":") not in triggers:
            return
        thread_ts = await self._reaction_thread_ts(
            client, channel_id, msg_ts, event, team_id, explicit_allowlist)
        if thread_ts is None:
            return
        await self._handle_slack_message(
            self._synthetic_reaction_event(event, action, thread_ts, team_id))

    def _synthetic_reaction_event(
        self, event: dict, action: str, thread_ts: str, team_id: str) -> dict:
        """Message-shaped event for a reaction. The reaction's own event_ts keeps the deduplicator
        from conflating it with the reacted-to message; ``_hermes_force_process`` skips the mention
        requirement (user auth and allowed_channels still apply); ``_hermes_reaction`` is
        informational. An optional handoff target channel replaces the reacted-to channel; a
        channel-only target is a handoff, not a reply — respond top-level there."""
        item = event.get("item") or {}
        channel_id, msg_ts = item.get("channel"), item.get("ts")
        reaction_name, user_id = event.get("reaction") or "", event.get("user")
        emoji_text = self._REACTION_EMOJI_MAP.get(reaction_name, reaction_name)
        synthetic: dict = {
            "type": "message",
            "user": user_id,
            "text": f"reaction:{action}:{emoji_text}",
            "channel": channel_id,
            "ts": event.get("event_ts") or f"reaction-{msg_ts}-{reaction_name}-{user_id}",
            "thread_ts": thread_ts,
            "_hermes_force_process": True,
            "_hermes_reaction": {
                "name": reaction_name, "action": action, "reacted_to_ts": msg_ts,
                "event_ts": event.get("event_ts")}}
        if team_id:
            synthetic["team"] = team_id
        # Optional handoff target (#45265): route the reaction-triggered turn into a configured channel (and
        # optionally thread) instead of the source thread. A channel-only target is a handoff, not a reply —
        # respond top-level there.
        target_channel, target_thread = self._slack_reaction_trigger_target()
        if target_channel:
            synthetic["channel"] = target_channel
            synthetic["channel_type"] = "im" if target_channel.startswith("D") else "channel"
            synthetic["_hermes_reaction_source_channel"] = channel_id
            if target_thread:
                synthetic["thread_ts"] = target_thread
            else:
                synthetic.pop("thread_ts", None)
                synthetic["_hermes_no_thread_response"] = True
        return synthetic

    async def _reaction_thread_ts(
        self, client, channel_id: str, msg_ts: str, event: dict, team_id: str,
        explicit_allowlist: bool) -> Optional[str]:
        """Thread to route a reaction into, or None to drop. Looks up the reacted-to message for
        thread + author; on failure the message itself is the parent (right top-level, loses
        linkage in-thread). Without an explicit allowlist only the bot's own messages route."""
        from . import adapter as _adapter

        thread_ts: str = msg_ts
        item_user = event.get("item_user") or ""
        if client is not None:
            try:
                history = await client.conversations_replies(
                    channel=channel_id, ts=msg_ts, limit=1, inclusive=True)
                messages = (history or {}).get("messages") or []
                if messages:
                    first = messages[0]
                    thread_ts = first.get("thread_ts") or first.get("ts") or msg_ts
                    item_user = item_user or first.get("user") or ""
                else:
                    return thread_ts
            except Exception as e:  # pragma: no cover - network path
                _adapter.logger.debug("[Slack] reaction thread_ts lookup failed for %s: %s", msg_ts, e)
                return thread_ts
        if not explicit_allowlist:
            bot_uid = self._team_bot_user_ids.get(team_id) or self._bot_user_id
            if item_user and bot_uid and item_user != bot_uid:
                return None
        return thread_ts

    def _slack_reaction_triggers(self) -> Optional[set]:
        """Reaction-routing opt-in: None = disabled (default, events acked+dropped);
        empty set = all emoji, bot's own messages only; non-empty = these emoji on
        any message. From ``slack.reaction_triggers`` or ``SLACK_REACTION_TRIGGERS``."""
        from . import adapter as _adapter

        raw = self.config.extra.get("reaction_triggers")
        if raw is None:
            raw = _adapter.os.getenv("SLACK_REACTION_TRIGGERS") or None
        if raw is None:
            return None
        if isinstance(raw, bool):
            return set() if raw else None
        if isinstance(raw, (list, tuple, set)):
            return {str(p).strip().strip(":") for p in raw if str(p).strip().strip(":")}
        text = str(raw or "").strip()
        if not text or text.lower() in {"false", "0", "no", "off"}:
            return None
        if text.lower() in {"true", "1", "yes", "on", "all", "*"}:
            return set()
        return {p.strip().strip(":") for p in _adapter.re.split(r"[,\s]+", text) if p.strip().strip(":")}

    def _slack_reaction_trigger_target(self) -> Tuple[str, str]:
        """Optional (channel, thread) reaction handoff target: ``C123`` or ``C123:<ts>``.
        Empty (default) routes into the reacted-to message's thread."""
        from . import adapter as _adapter

        raw = self.config.extra.get("reaction_trigger_target")
        if raw is None:
            raw = _adapter.os.getenv("SLACK_REACTION_TRIGGER_TARGET", "")
        channel, _, thread = str(raw or "").strip().partition(":")
        return channel.strip(), thread.strip()

    @staticmethod
    def _first_file_share(file_obj: Dict[str, Any], channel_id: str) -> Dict[str, Any]:
        """First share entry for ``channel_id`` (else the first share anywhere), or ``{}``.
        ``shares`` is ``{public|private: {channel_id: [entries]}}``; the channel match wins
        in the first bucket that has it, otherwise the first non-empty list already seen."""
        share = None
        for bucket in (file_obj.get("shares") or {}).values():
            if not isinstance(bucket, dict):
                continue
            channel_shares = bucket.get(channel_id)
            if channel_shares:
                return channel_shares[0] or {}
            if share is None:
                share = next((shares[0] for shares in bucket.values() if shares), None)
        return share or {}

    async def _handle_slack_file_shared(self, event: dict, body: Optional[dict] = None) -> None:
        """Fallback for file shares never delivered as message.files (``file_shared`` has only a
        file ID → ``files.info``). Video only: other uploads arrive on the message event."""
        from . import adapter as _adapter

        channel_id = event.get("channel_id") or event.get("channel") or ""
        if self._is_ignored_channel(channel_id):
            _adapter.logger.info(
                "[Slack] Ignoring file_shared event in configured ignored channel %s", channel_id)
            return
        file_id = event.get("file_id") or (event.get("file") or {}).get("id") or ""
        if not channel_id or not file_id:
            return
        team_id = self._event_team_id(event, body)
        try:
            client = self._team_clients.get(team_id) if team_id else None
            info_resp = await (client or self._get_client(channel_id)).files_info(file=file_id)
        except Exception as exc:
            detail = self._describe_slack_api_error(
                getattr(exc, "response", None), file_obj={"id": file_id})
            _adapter.logger.warning("[Slack] files.info error for file_shared %s: %s", file_id, detail or exc)
            return
        if not info_resp.get("ok"):
            detail = self._describe_slack_api_error(info_resp, file_obj={"id": file_id})
            _adapter.logger.warning(
                "[Slack] files.info failed for file_shared %s: %s", file_id,
                detail or info_resp.get("error"))
            return
        file_obj = info_resp.get("file") or {}
        if not str(file_obj.get("mimetype", "")).startswith("video/"):
            return
        share = self._first_file_share(file_obj, channel_id)
        ts = share.get("ts") or event.get("event_ts") or ""
        thread_ts = share.get("thread_ts") or ""
        # Let the normal message.file_share event arrive first; if it did,
        # its share ts is already recorded and this fallback skips.
        await _adapter.asyncio.sleep(0.75)
        if ts and self._dedup.is_duplicate(self._workspace_event_id(team_id, ts)):
            return
        fallback_event = {
            "type": "message",
            "subtype": "file_share",
            "text": "",
            "user": event.get("user_id") or file_obj.get("user", ""),
            "channel": channel_id,
            "channel_type": "im" if channel_id.startswith("D") else "channel",
            "team": team_id,
            "ts": "",  # already recorded above; avoid tripping our own dedup guard
            "files": [file_obj]}
        if thread_ts and thread_ts != ts:
            fallback_event["thread_ts"] = thread_ts
        await self._handle_slack_message(fallback_event)

    def _register_mentioned_thread(self, thread_ts: str, team_id: str = "") -> None:
        """Record a thread as bot-mentioned so future replies auto-trigger.
        Markers are workspace-scoped when team_id is known so identical thread ts values in two
        workspaces never wake each other's bot."""
        if not thread_ts:
            return
        self._mentioned_threads.add(self._workspace_message_marker(team_id, thread_ts))
        self._trim_mentioned_threads()

    async def _bot_authored_thread_root(
        self, channel_id: str, thread_ts: str, team_id: str = "") -> bool:
        """True when this bot authored the thread root — catches roots posted via direct
        chat.postMessage (not in _bot_message_ts) and survives restarts. Cache first, then a
        TTL-bounded fetch on a miss.

        Used by the wake-decision to detect threads where the bot posted the root via direct
        chat.postMessage (outside the gateway's send() path) — see #63530. Without this, human replies in
        bot-initiated threads were silently dropped when there was no active session and no @mention.
        Root-authorship is derived from the Slack API, so unlike the in-memory _bot_message_ts set it also
        survives gateway restarts.
        """
        if not thread_ts:
            return False
        bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id) or ""
        if not bot_uid:
            return False

        # team_id may be empty here, so match on the channel+thread key prefix; on a miss the
        # (TTL-cached) fetch populates parent_user_id, then re-check.
        for attempt in range(2):
            for cached_key, cached_entry in self._thread_context_cache.items():
                if cached_key.startswith(f"{channel_id}:{thread_ts}:"):
                    return bool(
                        cached_entry.parent_user_id and cached_entry.parent_user_id == bot_uid)
            if attempt == 0:
                await self._fetch_thread_context(
                    channel_id=channel_id, thread_ts=thread_ts, current_ts="", team_id=team_id)
        return False

    async def _should_wake_on_unmentioned_message(
        self, event_thread_ts, channel_id: str, user_id: str, is_thread_reply: bool,
        team_id: str = "", chat_type: str = "group") -> bool:
        """Return True if the bot should wake on an un-mentioned message. Checks, in order: root
        sent via send() (_bot_message_ts); thread previously @-mentioned; active session;
        bot-authored root via raw chat.postMessage; thread parent @-mentioned the bot.

        1. 2. _mentioned_threads        (someone @-mentioned us earlier) 3. _has_active_session... (there's
        already an agent session) 4. _bot_authored_thread_root (#63530: the bot posted the thread root via
        direct chat.postMessage, outside the gateway send() path — derived from the Slack API, so it also
        survives restarts).
        """
        if not event_thread_ts:
            return False
        thread_marker = self._workspace_message_marker(team_id, event_thread_ts)
        # Check scoped marker AND bare ts: entries recorded before team_id was
        # known are bare, and a scoped-vs-bare mismatch must not silence the bot.
        if is_thread_reply and (
            thread_marker in self._bot_message_ts or event_thread_ts in self._bot_message_ts):
            return True
        if thread_marker in self._mentioned_threads or event_thread_ts in self._mentioned_threads:
            return True
        if is_thread_reply and self._has_active_session_for_thread(
            channel_id=channel_id, thread_ts=event_thread_ts, user_id=user_id, team_id=team_id,
            chat_type=chat_type):
            return True
        if is_thread_reply and await self._bot_authored_thread_root(
            channel_id=channel_id, thread_ts=event_thread_ts, team_id=team_id):
            return True
        # Thread PARENT @-mentioned the bot before this process (restart): a bare "run" is for us.
        # 5th check (#24848): the thread PARENT @-mentioned the bot, but the mention event predates this
        # process (restart) or the parent asked the bot to wait for a follow-up (e.g. A plain reply like
        # "run" in that thread is addressed to the bot even though the reply itself carries no mention.
        if is_thread_reply:
            bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id)
            if bot_uid:
                parent_text = await self._fetch_thread_parent_text(
                    channel_id=channel_id, thread_ts=event_thread_ts, team_id=team_id,
                    strip_bot_mention=False)
                if parent_text and f"<@{bot_uid}>" in parent_text:
                    # Remember so later replies skip the fetch.
                    if not self._slack_strict_mention():
                        self._register_mentioned_thread(event_thread_ts)
                    return True
        return False

    @staticmethod
    def _append_block_text(text: str, blocks: list, bot_uid: str) -> str:
        """Merge Block Kit rich text not already in ``text`` plus the redacted block payload."""
        from . import adapter as _adapter

        blocks_text = _adapter._extract_additional_text_from_slack_blocks(blocks, text, bot_uid=bot_uid)
        stripped_blocks = blocks_text.strip() if blocks_text else ""
        if stripped_blocks:
            _adapter.logger.debug(
                "Slack: extracted additional text from blocks "
                "(likely quoted/forwarded content; chars=%d)", len(stripped_blocks))
            text = (text.strip() + "\n" + stripped_blocks).strip()
        blocks_payload = _adapter._serialize_slack_blocks_for_agent(blocks)
        if blocks_payload:
            text = (text.strip() + "\n\n" + blocks_payload).strip()
        return text

    def _runner_auth_fn(self) -> Any:
        """The gateway runner's ``_is_user_authorized`` (via the bound message handler), or None
        (multiplexed closures have no ``__self__``; object.__new__ doubles have no handler)."""
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        return getattr(runner, "_is_user_authorized", None)

    def _early_reject_unauthorized(self, user_id: str, channel_id: str, is_dm: bool) -> bool:
        """True (logged) when the sender is definitively unauthorized. Injected profile-bound check
        first (works under multiplex, where the handler has no ``__self__``), then runner
        introspection. Unknown (None) is NOT a rejection."""
        from . import adapter as _adapter

        chat_type = "dm" if is_dm else "group"
        decision = (
            self._is_sender_authorized(user_id, chat_type, channel_id)
            if user_id and getattr(self, "_authorization_check", None) is not None
            else None)
        auth_fn = self._runner_auth_fn()
        if decision is None and user_id and callable(auth_fn):
            source = self.build_source(
                chat_id=channel_id, chat_name="", chat_type=chat_type, user_id=user_id, user_name=""
            )
            decision = bool(auth_fn(source))
        if decision is False:
            _adapter.logger.warning(
                "[Slack] Early reject of unauthorized user %s in channel %s", user_id, channel_id)
        return decision is False

    async def _channel_gate_allows(
        self, *, channel_id: str, routing_text: str, bot_uid: str, is_mentioned: bool,
        is_thread_reply: bool, event_thread_ts, user_id: str, team_id: str, is_dm: bool,
        force_process: bool) -> bool:
        """Channel/MPIM gate: respond in a free-response channel (still gated by
        ``thread_require_mention``), when @mentioned, or when a wake check passes. Always silent
        outside ``allowed_channels`` or when addressed to another user; ``force_process`` skips only
        the mention rule."""
        from . import adapter as _adapter

        allowed_channels = self._slack_allowed_channels()
        if allowed_channels and channel_id not in allowed_channels:
            _adapter.logger.debug("[Slack] Ignoring message in non-allowed channel: %s", channel_id)
            return False
        self_uids = {u for u in (bot_uid, self._bot_user_id) if u}
        if (
            self._slack_ignore_other_user_mentions() and not is_mentioned
            and not self._slack_message_mentions_self(routing_text, self_uids)
            and self._slack_message_addressed_to_other_user(routing_text, self_uids)):
            _adapter.logger.debug(
                "[Slack] Ignoring message addressed to another user in channel %s", channel_id)
            return False
        thread_gated = self._slack_thread_require_mention() and is_thread_reply and not is_mentioned
        if force_process:
            return True
        free_channel = channel_id not in self._slack_require_mention_channels() and (
            channel_id in self._slack_free_response_channels() or not self._slack_require_mention())
        if not free_channel and self._slack_strict_mention() and not is_mentioned:
            return False  # Strict mode: ignore until @-mentioned again
        if thread_gated:
            _adapter.logger.debug(
                "[Slack] Ignoring thread reply without mention "
                "(thread_require_mention=true): channel=%s thread_ts=%s", channel_id,
                event_thread_ts)
            return False
        if free_channel:
            return True
        if not is_mentioned:
            return await self._should_wake_on_unmentioned_message(
                event_thread_ts=event_thread_ts, channel_id=channel_id, user_id=user_id,
                team_id=team_id, is_thread_reply=is_thread_reply,
                chat_type="dm" if is_dm else "group")
        return True

    def _normalize_changed_message(self, event: dict) -> Optional[dict]:
        """Turn a ``message_changed`` envelope into a plain message event.
        None if malformed or the original was already routed to the agent. The edit's own ts rides
        along as ``_slack_changed_event_ts`` for dedup."""
        updated_message = event.get("message")
        if not isinstance(updated_message, dict):
            return None
        original_message_ts = str(updated_message.get("ts") or "")
        if original_message_ts and original_message_ts in self._processed_message_ts:
            return None
        edited = updated_message.get("edited")
        edited_ts = str(edited.get("ts") or "") if isinstance(edited, dict) else ""
        outer_event_ts = str(event.get("ts") or "")
        changed_event_ts = (
            str(event.get("event_ts") or edited_ts or "")
            or (outer_event_ts if outer_event_ts != original_message_ts else "")
            or (f"{original_message_ts}:changed" if original_message_ts else ""))
        normalized_event = dict(updated_message)
        for key in ("channel", "channel_type", "team", "team_id"):
            if not normalized_event.get(key) and event.get(key):
                normalized_event[key] = event.get(key)
        if changed_event_ts:
            normalized_event["_slack_changed_event_ts"] = changed_event_ts
        return normalized_event

    @staticmethod
    def _append_link_unfurls(text: str, slack_attachments: list) -> str:
        """Append link-unfurl previews (``attachments``) to ``text``; ``is_msg_unfurl`` echoes our
        own content and is skipped. Dedup matches the rendered section, not the bare URL (which is
        usually already in the user's text while the preview body is not)."""
        from . import adapter as _adapter

        att_parts: list[str] = []
        for att in slack_attachments:
            att_title = att.get("title", "")
            att_url = att.get("title_link", "") or att.get("from_url", "")
            att_text = att.get("text", "")
            att_footer = att.get("footer", "")
            att_fallback = att.get("fallback", "")
            if att.get("is_msg_unfurl"):
                continue
            if att_title and att_url:
                header = f"📎 [{att_title}]({att_url})"
            else:
                header = f"📎 {att_title or att_url}" if (att_title or att_url) else None
            body = (att_text or att_fallback or "").strip()
            if len(body) > 500:
                body = body[:497] + "..."
            if header:
                section = f"{header}\n   {body}" if body else header
            elif body:
                section = f"📎 {body}"
            else:
                continue
            if section in text:
                continue
            if att_footer:
                section = f"{section}\n   _{att_footer}_"
            att_parts.append(section)
        if att_parts:
            text = (text.strip() + "\n\n" + "\n\n".join(att_parts)).strip()
            _adapter.logger.debug("Slack: appended %d link unfurl(s) to message text", len(att_parts))
        return text

    def _session_thread_ts(
        self, event: dict, ts: str, is_dm: bool, assistant_meta: Dict[str, str]) -> Optional[str]:
        """thread_ts for session keying. DMs: each top-level thread is its own session unless
        ``dm_top_level_threads_as_sessions: false``. Reaction handoffs reply top-level, never under
        the synthetic reaction ts. Channels: real reply → per-thread; top-level with
        ``reply_in_thread`` → ts as synthetic root; else None (``thread_ts == ts`` is no reply)."""
        if is_dm:
            thread_ts = event.get("thread_ts") or assistant_meta.get("thread_ts")
            if not thread_ts and self._dm_top_level_threads_as_sessions():
                thread_ts = ts
            return thread_ts
        if event.get("_hermes_no_thread_response"):
            return event.get("thread_ts") or None
        # Reaction handoff into a configured target channel (#45265): the response should be a new top-level
        # message in the target channel, never a thread under the synthetic ts (which is the reaction's
        # event_ts — not a real message there).
        # Channel message session scoping. Three cases: (a) genuine thread reply   → scope session per
        # thread (b) top-level, reply_in_thread=true (the default)  → legacy behaviour: each top-level
        # message becomes its own thread, so the UX still "replies in a thread" and sessions are keyed per
        # thread root (c) top-level, reply_in_thread=false → scope one session across the whole channel so
        # context accumulates across messages (#15421 bug 1)
        event_thread_ts_raw = event.get("thread_ts")
        # Align with ``is_thread_reply`` below — a ``thread_ts == ts`` payload (some thread-root shapes) is
        # not a real reply and must not prevent the shared-session path from taking effect. Matching the
        # same invariant here keeps the two branches in sync even if Slack introduces new payload variants
        # (Copilot on #15464).
        if event_thread_ts_raw and event_thread_ts_raw != ts:
            return event_thread_ts_raw
        if self.config.extra.get("reply_in_thread", True):
            return ts
        return None

    async def _hydrate_thread_context(
        self, *, channel_id: str, event_thread_ts, ts: str, user_id: str, team_id: str,
        is_thread_reply: bool, is_mentioned: bool, is_dm: bool,
    ) -> Tuple[Optional[str], List[str], List[str]]:
        """``(channel_context, root_media_urls, root_media_types)`` for a thread reply. No session:
        full thread + root images once, set watermark. Session + @mention: delta past watermark
        (cache bypassed). Session, first plain reply this process: restart rehydration; later
        replies only advance the watermark. Context goes into the NEW turn only (prompt caching)."""
        # - Active thread + explicit @mention: refresh with only the delta since the last hydrate/refresh
        #   (#23918), bypassing the TTL cache. The delta is injected as part of the NEW turn (via
        #   ``channel_context``) — prior conversation history is never rewritten, so prompt caching is
        #   preserved. Keep recovered history separate from ``text``. Prepending it here moves a recognized
        #   command away from character zero, so downstream command routing can misclassify it as
        #   conversational text. ``channel_context`` is prepended only after command dispatch.
        from . import adapter as _adapter

        channel_context = None
        # Thread-root images recovered on the cold-start hydrate: when the bot is mentioned mid-thread for
        # the first time, the thread root is very often the artifact the mention is about ("@bot what's in
        # this chart?" replying under an image post) — deliver its images with this first turn. One-time by
        # construction: the cold-start path is guarded by _has_active_session_for_thread, so subsequent
        # turns in the same session never re-deliver (adapted from #69185).
        thread_root_media_urls: _adapter.List[str] = []
        thread_root_media_types: _adapter.List[str] = []
        if not is_thread_reply:
            return channel_context, thread_root_media_urls, thread_root_media_types
        has_active_thread_session = self._has_active_session_for_thread(
            channel_id=channel_id, thread_ts=event_thread_ts, user_id=user_id, team_id=team_id,
            chat_type="dm" if is_dm else "group")

        async def _fetch(**kw) -> None:
            nonlocal channel_context
            thread_context = await self._fetch_thread_context(
                channel_id=channel_id, thread_ts=event_thread_ts, current_ts=ts, team_id=team_id,
                **kw)
            if thread_context:
                channel_context = thread_context

        watermark_args = dict(
            channel_id=channel_id, thread_ts=event_thread_ts, user_id=user_id, team_id=team_id)
        if not has_active_thread_session:
            await _fetch()
            (
                thread_root_media_urls, thread_root_media_types,
            ) = await self._collect_thread_root_images(
                channel_id=channel_id, thread_ts=event_thread_ts, team_id=team_id)
        elif is_mentioned:
            await _fetch(after_ts=self._get_thread_watermark(**watermark_args), force_refresh=True)
        else:
            # Restart rehydration (#63530 restart gap / #33215): persistent sessions survive gateway
            # restarts, but thread replies posted while the gateway was down never reached the session. On
            # the FIRST ordinary reply per thread in this process, fetch the delta past the persisted
            # watermark and inject anything missed as part of this new turn. Checked at most once per thread
            # per process; a non-empty watermark plus an empty delta costs one cached conversations.replies
            # call.
            rehydration_key = self._thread_rehydration_key(
                channel_id, event_thread_ts, user_id, team_id)
            if rehydration_key in self._thread_rehydration_checked:
                self._set_thread_watermark(watermark_ts=ts, **watermark_args)
                return channel_context, thread_root_media_urls, thread_root_media_types
            watermark_ts = self._get_thread_watermark(**watermark_args)
            if watermark_ts:
                await _fetch(after_ts=watermark_ts, force_refresh=True)
        self._set_thread_watermark(watermark_ts=ts, **watermark_args)
        self._mark_thread_rehydration_checked(channel_id, event_thread_ts, user_id, team_id)
        return channel_context, thread_root_media_urls, thread_root_media_types

    @staticmethod
    def _media_message_type(media_types: List[str]) -> MessageType:
        """PHOTO/VIDEO/VOICE/DOCUMENT by the first matching media prefix; TEXT when none."""
        from . import adapter as _adapter

        if not media_types:
            return _adapter.MessageType.TEXT
        for prefix, kind in (
            ("image/", _adapter.MessageType.PHOTO), ("video/", _adapter.MessageType.VIDEO),
            ("audio/", _adapter.MessageType.VOICE)):
            if any(m.startswith(prefix) for m in media_types):
                return kind
        return _adapter.MessageType.DOCUMENT

    def _channel_prompt_with_identity(self, channel_id: str, team_id: str) -> Optional[str]:
        """Channel prompt with the bot's Slack identity prepended (ephemeral, never persisted,
        so prompt caching holds) so it won't read a human's mention as a self-mention."""
        from gateway.platforms.base import resolve_channel_prompt
        channel_prompt = resolve_channel_prompt(self.config.extra, channel_id, None)
        identity_prompt = self._build_identity_prompt(team_id)
        if identity_prompt:
            channel_prompt = (
                f"{identity_prompt}\n\n{channel_prompt}".strip()
                if channel_prompt
                else identity_prompt)
        return channel_prompt

    def _track_reacting_message(self, team_id: str, ts: str) -> None:
        """Mark ``ts`` for the reaction lifecycle, evicting oldest-ts-first past the cap."""
        self._reacting_message_ids.add(self._workspace_message_marker(team_id, ts))
        self._evict_oldest_by_ts(self._reacting_message_ids, self._REACTING_MESSAGE_IDS_MAX)

    async def _handle_slack_message(self, event: dict, payload: Optional[dict] = None) -> None:
        """Guard around :meth:`_handle_slack_message_impl`: the impl claims the ts early (no second
        turn from a mid-flight unfurl); if THIS call newly claimed it and raises, release the claim
        so a retry/edit can re-drive it. Pre-existing claims stay."""
        from . import adapter as _adapter

        _ts = str((event or {}).get("ts") or "")
        # getattr: bare test doubles (object.__new__) may lack the map.
        _claims = getattr(self, "_processed_message_ts", None)
        _was_claimed = bool(_ts) and _claims is not None and _ts in _claims
        try:
            return await self._handle_slack_message_impl(event, payload)
        except BaseException:
            _claims = getattr(self, "_processed_message_ts", None)
            if _ts and not _was_claimed and _claims is not None and _ts in _claims:
                _claims.pop(_ts, None)
                _adapter.logger.warning(
                    "[%s] handler failed after claiming ts=%s; claim released "
                    "so a retry or edit can re-drive the turn", self.name, _ts)
            raise

    async def _drop_bot_sender(self, event: dict) -> bool:
        """allow_bots gate: ``none`` drops all bot posts (default), ``mentions`` those not
        @mentioning us, ``all`` accepts — own posts always drop (echo loops). Unlabeled events
        without ``client_msg_id`` are probed via users.info (humans carry it, stray bots don't)."""
        from . import adapter as _adapter

        msg_user = event.get("user", "")
        sender_is_bot = self._event_declares_bot_sender(event)
        if not sender_is_bot and msg_user and not event.get("client_msg_id"):
            sender_is_bot = await self._resolve_user_is_bot(
                msg_user, chat_id=event.get("channel", ""),
                team_id=str(event.get("team") or event.get("team_id") or ""))
        if not sender_is_bot:
            return False
        allow_bots = self._slack_allow_bots()
        if allow_bots == "none":
            return True
        if allow_bots == "mentions":
            # Mentions may live only in Block Kit, not the flat text.
            # See #52387.
            text_check = _adapter._slack_mention_detection_text(event)
            if self._bot_user_id and f"<@{self._bot_user_id}>" not in text_check:
                _adapter.logger.debug(
                    "[Slack] Dropping bot message under allow_bots=mentions: "
                    "no <@%s> mention in flat text or blocks", self._bot_user_id)
                return True
        return bool(msg_user and self._bot_user_id and msg_user == self._bot_user_id)

    async def _prefilter_inbound(
        self, event: dict, payload: Optional[dict]) -> Optional[Tuple[dict, str, str]]:
        """Normalize edits, then drop replays / ignored channels / bot posts / deletions.
        Returns ``(event, team_id, channel_id)`` for messages the handler should consider."""
        # Entry log BEFORE any filtering so operators can tell "dropped here"
        # from "never subscribed in the manifest". Metadata only, never text.
        # DEBUG entry log — fires BEFORE any filtering so users debugging bot-to-bot interop, allow_bots
        # config, or SLACK_ALLOWED_USERS drops can confirm whether the event actually arrived from Slack
        # (vs. being silently filtered upstream by the app's event subscriptions — Socket Mode will not
        # deliver events the app manifest hasn't subscribed to). See #30091.
        from . import adapter as _adapter

        if _adapter.logger.isEnabledFor(_adapter.logging.DEBUG):
            _bot_profile = event.get("bot_profile") or {}
            _adapter.logger.debug(
                "[Slack] event received type=%s subtype=%s user=%s bot_id=%s bot_name=%s "
                "channel=%s ts=%s thread_ts=%s", event.get("type"), event.get("subtype"),
                event.get("user", "") or "", event.get("bot_id", "") or "",
                (_bot_profile.get("name") if isinstance(_bot_profile, dict) else "") or "",
                event.get("channel", ""), event.get("ts", ""), event.get("thread_ts", ""))
        if event.get("subtype") == "message_changed":
            event = self._normalize_changed_message(event)
            if event is None:
                return None
        # Socket Mode redelivers after reconnects. Scope by workspace: ts is only unique per team.
        # Dedup: Slack Socket Mode can redeliver events after reconnects (#4777) Scope the dedup id by
        # workspace: Slack event ts values are only unique within one workspace, so two teams' events with
        # the same ts must not suppress each other.
        event_ts = event.get("_slack_changed_event_ts") or event.get("ts", "")
        dedup_team_id = self._event_team_id(event, payload)
        if event_ts and self._dedup.is_duplicate(self._workspace_event_id(dedup_team_id, event_ts)):
            return None
        channel_id = event.get("channel", "")
        if self._is_ignored_channel(channel_id):
            _adapter.logger.info("[Slack] Ignoring message in configured ignored channel %s", channel_id)
            return None
        if await self._drop_bot_sender(event):
            return None
        # Edits were normalized above so an @mention added by edit can wake the bot once.
        if event.get("subtype") == "message_deleted":
            return None
        return event, dedup_team_id, channel_id

    async def _peer_bot_drop(
        self, event: dict, user_id: str, bot_uid: Optional[str], channel_id: str, team_id: str,
        is_mentioned: bool) -> bool:
        """True when a bot *user* post (peer agent: no bot_id/subtype) must be dropped.
        Such posts would otherwise re-trigger via old thread mentions or active sessions and cause
        agent-agent loops. Under ``mentions`` only the current text counts as a summons."""
        if not user_id or user_id == bot_uid:
            return False
        sender_is_bot_user = self._event_declares_bot_sender(event)
        if not sender_is_bot_user:
            sender_is_bot_user = await self._resolve_user_is_bot(
                user_id, chat_id=channel_id, team_id=team_id)
        if not sender_is_bot_user:
            return False
        allow_bots = self._slack_allow_bots()
        return allow_bots == "none" or (allow_bots == "mentions" and not is_mentioned)

    def _apply_bot_mention(
        self, text: str, original_text: str, command_probe_text: str, is_command_text: bool,
        bot_uid: str, thread_ts: Optional[str], team_id: str) -> Tuple[str, str, str, bool]:
        """Strip our mention, re-probe for a command hidden behind it, remember the thread.
        Returns updated ``(text, original_text, command_probe_text, is_command_text)``."""
        from . import adapter as _adapter

        text = text.replace(f"<@{bot_uid}>", "").strip()
        # Re-probe commands on the canonical text (block-augmented text would leak quoted text
        # into arguments): handles ``@bot !cmd`` / ``@bot /cmd``.
        mention_stripped = original_text.replace(f"<@{bot_uid}>", "").strip()
        command_text = (
            mention_stripped
            if mention_stripped.startswith("/")
            else _adapter._rewrite_known_bang_command(mention_stripped))
        if command_text.startswith("/"):
            original_text = text = command_probe_text = command_text
            is_command_text = True
        # Remember the thread so follow-ups auto-trigger (skipped under strict_mention /
        # thread_require_mention, which it would defeat). Session-scoped ``thread_ts`` because a
        # top-level @mention STARTS a thread whose replies must trigger too.
        if (
            thread_ts and not self._slack_strict_mention()
            and not self._slack_thread_require_mention()):
            self._register_mentioned_thread(thread_ts, team_id=team_id)
        return text, original_text, command_probe_text, is_command_text

    async def _handle_slack_message_impl(self, event: dict, payload: Optional[dict] = None) -> None:
        """Handle an incoming Slack message event."""
        from . import adapter as _adapter

        is_message_edit = event.get("subtype") == "message_changed"
        accepted = await self._prefilter_inbound(event, payload)
        if accepted is None:
            return
        event, dedup_team_id, channel_id = accepted
        sender_is_bot = self._event_declares_bot_sender(event)
        if not sender_is_bot and event.get("user") and not event.get("client_msg_id"):
            sender_is_bot = await self._resolve_user_is_bot(
                event["user"], chat_id=event.get("channel", ""),
                team_id=str(event.get("team") or event.get("team_id") or ""))
        original_text = event.get("text", "")
        # Slack rejects slash commands inside threads, so a leading ``!`` is rewritten to ``/``
        # — only for known gateway commands, so "!nice work" passes through.
        command_probe_text = _adapter._rewrite_known_bang_command(original_text.lstrip())
        if command_probe_text != original_text.lstrip():
            original_text = command_probe_text
        is_command_text = command_probe_text.startswith("/")
        text = original_text
        # Quoted/forwarded block text is absent from flat ``text``. Skipped for commands: after
        # the ``!``→``/`` rewrite it no longer dedupes and would become bogus arguments.
        blocks = event.get("blocks")
        if blocks and not is_command_text:
            text = self._append_block_text(
                text, blocks, self._team_bot_user_ids.get(dedup_team_id, self._bot_user_id) or "")
        text = self._append_link_unfurls(text, event.get("attachments") or [])
        ts = event.get("ts", "")
        outer_team_id = dedup_team_id
        assistant_meta = self._lookup_assistant_thread_metadata(
            event, channel_id=channel_id, thread_ts=event.get("thread_ts", ""),
            team_id=outer_team_id, body=payload)
        user_id = event.get("user") or assistant_meta.get("user_id", "")
        if not channel_id:
            channel_id = assistant_meta.get("channel_id", "")
        # File-upload events may omit team_id; recover it for multi-workspace token lookup.
        team_id = (
            outer_team_id or assistant_meta.get("team_id", "") or self._channel_team.get(channel_id, "")
        )
        agent_context = self._agent_view_context_for_event(
            event, str(team_id or ""), str(user_id or ""))
        if team_id and channel_id:
            self._remember_channel_team(channel_id, team_id)
        channel_type = event.get("channel_type", "") or ("im" if channel_id.startswith("D") else "")
        is_dm = channel_type in {"im", "mpim"}  # Both 1:1 and group DMs
        if is_dm and self._slack_disable_dms():
            _adapter.logger.info(
                "[Slack] Ignoring DM because Slack DMs are disabled: channel=%s user=%s",
                channel_id, user_id)
            return
        # Only a 1:1 IM earns DM exemptions (no mention needed, free reactions); an MPIM obeys
        # channel gating, though session/thread scoping treats both as DM-style.
        is_one_to_one_dm = channel_type == "im"
        # Reject unauthorized users before the expensive lookups/downloads;
        # the runner's own auth check only runs after MessageEvent is built.
        if self._early_reject_unauthorized(user_id, channel_id, is_dm):
            return
        thread_ts = self._session_thread_ts(event, ts, is_dm, assistant_meta)
        bot_uid = self._team_bot_user_ids.get(team_id, self._bot_user_id)
        # Mentions may live only in Block Kit blocks.
        # See #52387.
        routing_text = _adapter._slack_mention_detection_text(event) or original_text or ""
        is_mentioned = bool(
            (bot_uid and f"<@{bot_uid}>" in routing_text)
            or self._slack_message_matches_mention_patterns(routing_text))
        event_thread_ts = event.get("thread_ts")
        is_thread_reply = bool(event_thread_ts and event_thread_ts != ts)
        # Internal triggers (reactions) skip the mention requirement but NOT
        # allowed_channels or user authorization.
        force_process = bool(event.get("_hermes_force_process"))
        if await self._peer_bot_drop(event, user_id, bot_uid, channel_id, team_id, is_mentioned):
            return
        if (
            not is_one_to_one_dm and bot_uid and not await self._channel_gate_allows(
            channel_id=channel_id, routing_text=routing_text, bot_uid=bot_uid,
            is_mentioned=is_mentioned, is_thread_reply=is_thread_reply,
            event_thread_ts=event_thread_ts, user_id=user_id, team_id=team_id, is_dm=is_dm,
            force_process=force_process)):
            return
        # Claim the message ts HERE: a link unfurl emits `message_changed` with a different event
        # ts, so only the `_processed_message_ts` guard stops a duplicate turn, and it must be set
        # before the slow enrichment awaits. Claiming before the filters would let an ignored
        # original block a later "@bot" edit from summoning the bot.
        _claim_ts = str(event.get("ts") or "")
        if _claim_ts:
            self._remember_processed_message_ts(_claim_ts)
        if is_mentioned:
            text, original_text, command_probe_text, is_command_text = self._apply_bot_mention(
                text, original_text, command_probe_text, is_command_text, bot_uid, thread_ts,
                team_id)
        # Thread history stays out of ``text``: prepending would push a command off char zero.
        (
            channel_context, thread_root_media_urls, thread_root_media_types,
        ) = await self._hydrate_thread_context(
            channel_id=channel_id, event_thread_ts=event_thread_ts, ts=ts, user_id=user_id,
            team_id=team_id, is_thread_reply=is_thread_reply, is_mentioned=is_mentioned,
            is_dm=is_dm)
        # Thread-root media is delivered ahead of the trigger message's own files.
        media_urls, media_types, text = await self._collect_inbound_media(
            event, channel_id, team_id, text, thread_root_media_urls, thread_root_media_types)
        msg_event = await self._build_message_event(
            event, text=text, original_text=original_text, command_probe_text=command_probe_text,
            is_command_text=is_command_text, channel_id=channel_id, team_id=team_id, ts=ts,
            user_id=user_id, thread_ts=thread_ts, is_dm=is_dm, media_urls=media_urls,
            media_types=media_types, channel_context=channel_context)
        msg_event.source.is_bot = sender_is_bot
        msg_event.source.is_one_to_one = is_one_to_one_dm
        msg_event.source.message_is_edit = is_message_edit
        msg_event.metadata["message_is_edit"] = is_message_edit
        # React only when directly addressed; MPIMs are shared, so they need a
        # mention like any channel.
        if (is_one_to_one_dm or is_mentioned) and self._reactions_enabled():
            self._track_reacting_message(team_id, ts)
        # App-context is per-turn UI state: in the user message, not SessionSource (would rebuild
        # the agent per view switch and leak stale context). Inert label, never a channel body.
        context_channel_id = agent_context.get("context_channel_id", "")
        if context_channel_id and context_channel_id != channel_id and not is_command_text:
            msg_event.text = (
                f"[Slack app context: user is viewing channel {context_channel_id}]\n\n"
                f"{msg_event.text}")
        if ts:
            self._remember_processed_message_ts(ts)
        await self.handle_message(msg_event)

    async def _build_message_event(
        self, event: dict, *, text: str, original_text: str, command_probe_text: str,
        is_command_text: bool, channel_id: str, team_id: str, ts: str, user_id: str,
        thread_ts: Optional[str], is_dm: bool, media_urls: List[str], media_types: List[str],
        channel_context: Optional[str]) -> MessageEvent:
        """Resolve names, title the DM thread, and build the ``MessageEvent``. Commands are restored
        from canonical input: the parser needs the token at char zero and enrichment (blocks,
        unfurls, file text, history) must never mutate arguments."""
        from . import adapter as _adapter

        if is_command_text:
            text = command_probe_text
        msg_type = _adapter.MessageType.COMMAND if is_command_text else self._media_message_type(media_types)
        user_name = await self._resolve_user_name(user_id, chat_id=channel_id, team_id=team_id)
        channel_name = await self._resolve_channel_name(channel_id, team_id=team_id)
        # Best-effort: title the DM thread from the prompt for Slack's AI Agent Messages tab.
        if is_dm and thread_ts and msg_type != _adapter.MessageType.COMMAND:
            await self._set_assistant_thread_title(
                channel_id, thread_ts, original_text or text, team_id=team_id)
        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel_name,
            chat_type="dm" if is_dm else "group",
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_ts,
            scope_id=str(team_id) if team_id else None,
            # Workflow/app posts have user=None; flag them so the SLACK_ALLOW_BOTS bypass can
            # authorize them. Same predicate as the drop gate (api_human_users stay human).
            is_bot=self._event_declares_bot_sender(event))
        from gateway.platforms.base import resolve_channel_skills
        # Remaining ``<@UID>`` are OTHER participants (own mention stripped
        # above); render as ``@DisplayName`` so the agent knows who is addressed.
        text = await self._humanize_user_mentions(text, chat_id=channel_id, team_id=team_id)
        return _adapter.MessageEvent(
            text=(command_probe_text if is_command_text else text),
            message_type=msg_type,
            source=source,
            raw_message=event,
            message_id=ts,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=thread_ts if thread_ts != ts else None,
            channel_prompt=self._channel_prompt_with_identity(channel_id, team_id),
            channel_context=channel_context,
            # thread_ts is the thread root, not an explicit reply (root is in channel_context).
            reply_to_text=None,
            auto_skill=resolve_channel_skills(self.config.extra, channel_id, None),
            metadata={
                "slack_team_id": team_id, "slack_channel_id": channel_id,
                "slack_thread_ts": thread_ts})

    def _note_attachment_failure(
        self, notices: List[str], detail: Optional[str], fallback_msg: str, *fallback_args: Any,
        exc_info: bool = False) -> None:
        """Record a user-facing attachment diagnostic, else log the raw failure."""
        from . import adapter as _adapter

        if detail:
            notices.append(detail)
            _adapter.logger.warning("[Slack] %s", detail)
        else:
            _adapter.logger.warning(fallback_msg, *fallback_args, exc_info=exc_info)

    async def _resolve_file_stub(
        self, f: Dict[str, Any], channel_id: str, team_id: str, notices: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Resolve a Slack Connect ``file_access="check_file_info"`` stub (no URL
        fields) via ``files.info``; None when unresolvable. ``notices=None`` fails silently."""
        file_id = f.get("id")
        if not file_id:
            return None
        try:
            info_resp = await self._get_client(channel_id, team_id=team_id).files_info(file=file_id)
        except Exception as e:
            if notices is not None:
                detail = self._describe_slack_api_error(getattr(e, "response", None), file_obj=f)
                self._note_attachment_failure(
                    notices, detail, "[Slack] files.info error for %s: %s", file_id, e, exc_info=True
                )
            return None
        if info_resp.get("ok"):
            return info_resp["file"]
        if notices is not None:
            detail = self._describe_slack_api_error(info_resp, file_obj=f)
            self._note_attachment_failure(
                notices, detail, "[Slack] files.info failed for %s: %s", file_id,
                info_resp.get("error"))
        return None

    @staticmethod
    def _slack_file_kind(f: Dict[str, Any], mimetype: str) -> str:
        """image / audio / voice clip / video / document, from mimetype (+ voice-clip heuristics)."""
        from . import adapter as _adapter

        for prefix in ("image", "audio"):
            if mimetype.startswith(prefix + "/"):
                return prefix
        if mimetype.startswith("video/"):
            return "voice clip" if _adapter._is_slack_voice_clip(f) else "video"
        return "document"

    async def _cache_slack_file(
        self, kind: str, f: Dict[str, Any], url: str, mimetype: str, team_id: str
    ) -> Optional[Tuple[str, str, str]]:
        """Download+cache one inbound file; ``(cached_path, media_type, text_injection)``
        or None when skipped (oversized/unknown-size document)."""
        from . import adapter as _adapter

        if kind == "image":
            ext = "." + mimetype.split("/")[-1].split(";")[0]
            if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                ext = ".jpg"
            return await self._download_slack_file(url, ext, team_id=team_id), mimetype, ""
        if kind in ("audio", "voice clip"):
            ext = _adapter._resolve_slack_audio_ext(f, mimetype)
            cached = await self._download_slack_file(url, ext, audio=True, team_id=team_id)
            if kind == "audio":
                return cached, mimetype, ""
            # Voice clips are audio-only MP4 Slack may label video/mp4; cache
            # as audio/* so the gateway routes to STT, not video understanding.
            _adapter.logger.debug("[Slack] Cached voice clip (mislabeled %s) as audio: %s", mimetype, cached)
            return cached, _adapter._SLACK_EXT_TO_AUDIO_MIME.get(ext, "audio/mp4"), ""
        if kind == "video":
            ext = _adapter.os.path.splitext(f.get("name", ""))[1].lower()
            if ext not in _adapter.SUPPORTED_VIDEO_TYPES:
                mime_to_ext = {v: k for k, v in _adapter.SUPPORTED_VIDEO_TYPES.items()}
                ext = mime_to_ext.get(mimetype.split(";", 1)[0].lower(), ".mp4")
            raw_bytes = await self._download_slack_file_bytes(url, team_id=team_id)
            cached_path = await _adapter.cache_video_from_bytes_async(raw_bytes, ext=ext)
            _adapter.logger.debug("[Slack] Cached user video: %s", cached_path)
            return cached_path, _adapter.SUPPORTED_VIDEO_TYPES.get(ext, mimetype or "video/mp4"), ""
        return await self._cache_slack_document(f, url, mimetype, team_id)

    async def _cache_slack_document(
        self, f: Dict[str, Any], url: str, mimetype: str, team_id: str
    ) -> Optional[Tuple[str, str, str]]:
        """Document branch of :meth:`_cache_slack_file`: any extension is accepted (authorization
        is the gate); Slack's bot upload cap is 20 MB; small text-like files are injected."""
        from . import adapter as _adapter

        original_filename = f.get("name", "")
        ext = _adapter.os.path.splitext(original_filename)[1].lower() if original_filename else ""
        if not ext and mimetype:
            mime_to_ext = {v: k for k, v in _adapter.SUPPORTED_DOCUMENT_TYPES.items()}
            ext = mime_to_ext.get(mimetype, "")
        # Any extension accepted (authorization is the gate); Slack bot upload cap is 20 MB.
        file_size = f.get("size", 0)
        if not file_size or file_size > 20 * 1024 * 1024:
            _adapter.logger.warning("[Slack] Document too large or unknown size: %s", file_size)
            return None
        raw_bytes = await self._download_slack_file_bytes(url, team_id=team_id)
        cached_path = await _adapter.cache_document_from_bytes_async(
            raw_bytes, original_filename or f"document{ext or '.bin'}")
        doc_mime = _adapter.SUPPORTED_DOCUMENT_TYPES.get(ext, mimetype or "application/octet-stream")
        _adapter.logger.debug("[Slack] Cached user document: %s (%s)", cached_path, doc_mime)
        injection = ""
        _is_text = ext in _adapter._TEXT_INJECT_EXTENSIONS or (mimetype or "").startswith("text/")
        if _is_text and len(raw_bytes) <= 100 * 1024:
            try:
                text_content = raw_bytes.decode("utf-8")
                display_name = original_filename or f"document{ext or '.txt'}"
                display_name = _adapter.re.sub(r"[^\w.\- ]", "_", display_name)
                injection = f"[Content of {display_name}]:\n{text_content}"
            except UnicodeDecodeError:
                pass  # Binary content, skip injection
        return cached_path, doc_mime, injection

    async def _collect_inbound_media(
        self, event: dict, channel_id: str, team_id: str, text: str,
        thread_root_media_urls: List[str], thread_root_media_types: List[str],
    ) -> Tuple[List[str], List[str], str]:
        """Download/cache ``event["files"]`` → ``(media_urls, media_types, text)``; root images
        lead. Small text-like docs are injected into ``text`` (gated on ext/MIME, not blind UTF-8
        decode — PDF/zip headers decode). Failures are prepended as an attachment notice."""
        from . import adapter as _adapter

        media_urls = list(thread_root_media_urls)
        media_types = list(thread_root_media_types)
        notices: _adapter.List[str] = []
        for f in event.get("files", []):
            if f.get("file_access") == "check_file_info":
                f = await self._resolve_file_stub(f, channel_id, team_id, notices)
                if f is None:
                    continue
            mimetype = f.get("mimetype", "unknown")
            url = f.get("url_private_download") or f.get("url_private", "")
            if not url:
                continue
            kind = self._slack_file_kind(f, mimetype)
            try:
                cached = await self._cache_slack_file(kind, f, url, mimetype, team_id)
                if cached is None:
                    continue
                cached_path, media_type, injection = cached
                media_urls.append(cached_path)
                media_types.append(media_type)
                if injection:
                    text = f"{injection}\n\n{text}" if text else injection
            except Exception as e:  # pragma: no cover - defensive logging
                self._note_attachment_failure(
                    notices, self._describe_slack_download_failure(e, file_obj=f),
                    f"[Slack] Failed to cache {kind} from %s: %s", url, e, exc_info=True)
        if notices:
            notice_block = "[Slack attachment notice]\n" + "\n".join(f"- {n}" for n in notices)
            text = f"{notice_block}\n\n{text}" if text else notice_block
        return media_urls, media_types, text
