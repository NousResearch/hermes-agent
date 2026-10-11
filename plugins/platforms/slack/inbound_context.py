"""Inbound text and context preparation for the SlackAdapter adapter."""

from __future__ import annotations

from gateway.platforms.event import attributed_context

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.slack.pending_thread_context import SlackThreadContext

if TYPE_CHECKING:
    from plugins.platforms.slack.adapter import SlackAdapter

logger = logging.getLogger("plugins.platforms.slack.adapter")


class SlackInboundContextMixin:
    @classmethod
    def _append_link_unfurls(cls, text: str, slack_attachments: list) -> str:
        """Append the rendered ``attachments`` to ``text``."""
        att_parts = cls._link_unfurl_sections(text, slack_attachments)
        if att_parts:
            text = (text.strip() + "\n\n" + "\n\n".join(att_parts)).strip()
            logger.debug("Slack: appended %d link unfurl(s) to message text", len(att_parts))
        return text

    async def _handle_slack_message_impl(self: SlackAdapter, event: dict, payload: Optional[dict] = None) -> None:
        """Handle an incoming Slack message event."""
        accepted = await self._prefilter_inbound(event, payload)
        if accepted is None:
            return
        event, dedup_team_id, channel_id = accepted
        prepared = await self._prepare_slack_message(event, dedup_team_id, channel_id, payload)
        if prepared is not None:
            await self.handle_message(prepared)

    async def _prepare_slack_message(
        self: SlackAdapter, event: dict, dedup_team_id: str, channel_id: str, payload: Optional[dict] = None,
        *, cached: MessageEvent | None = None, replay_thread_context: SlackThreadContext | None = None, authorize: Callable[[SessionSource], bool] | None = None,
    ) -> MessageEvent | None:
        """Prepare accepted input under the current workspace and channel policy."""
        from plugins.platforms.slack.adapter import (
            _rewrite_known_bang_command,
            _slack_mention_detection_text,
        )
        if self._is_ignored_channel(channel_id):
            return None
        original_text = event.get("text", "")
        # Slack rejects slash commands inside threads, so a leading ``!`` is rewritten to ``/``
        # — only for known gateway commands, so "!nice work" passes through.
        command_probe_text = _rewrite_known_bang_command(original_text.lstrip())
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
        attachments = event.get("attachments") or []
        text = self._append_link_unfurls(text, [att for att in attachments if not _is_shared_slack_attachment(att)])
        shared_sections = self._link_unfurl_sections(text, [att for att in attachments if _is_shared_slack_attachment(att)])
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
            logger.info(
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
        routing_text = _slack_mention_detection_text(event) or original_text or ""
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
        if authorize is not None:
            current_source = self.build_source(
                chat_id=channel_id, chat_type="dm" if is_dm else "group", user_id=user_id,
                thread_id=thread_ts, scope_id=team_id or None, message_id=ts,
                is_bot=self._event_declares_bot_sender(event))
            if self._canonicalize(current_source) is None or not authorize(current_source):
                return None
        _claim_ts = str(event.get("ts") or "")
        if _claim_ts and cached is None:
            self._remember_processed_message_ts(_claim_ts)
        if is_mentioned:
            text, original_text, command_probe_text, is_command_text = self._apply_bot_mention(
                text, original_text, command_probe_text, is_command_text, bot_uid, thread_ts,
                team_id)
        # Thread history stays out of ``text``: prepending would push a command off char zero.
        thread_snapshot = replay_thread_context
        if cached is None:
            (
                channel_context, thread_root_media_urls, thread_root_media_types, thread_snapshot,
            ) = await self._hydrate_thread_context(
                channel_id=channel_id, event_thread_ts=event_thread_ts, ts=ts, user_id=user_id,
                team_id=team_id, is_thread_reply=is_thread_reply, is_mentioned=is_mentioned, is_dm=is_dm)
        else:
            channel_context = thread_snapshot.text if thread_snapshot is not None else None
            thread_root_media_urls, thread_root_media_types = [], []
        # Thread-root media is delivered ahead of the trigger message's own files.
        if cached is None:
            media_urls, media_types, media_text_inlined, text = await self._collect_inbound_media(
                event, channel_id, team_id, text, thread_root_media_urls, thread_root_media_types)
        else:
            if thread_root_media_urls:
                return None
            media_urls = list(cached.media_urls)
            media_types = list(cached.media_types)
            media_text_inlined = list(cached.media_text_inlined)
        msg_event = await self._build_message_event(
            event, text=text, original_text=original_text, command_probe_text=command_probe_text,
            is_command_text=is_command_text, channel_id=channel_id, team_id=team_id, ts=ts,
            user_id=user_id, thread_ts=thread_ts, is_dm=is_dm, media_urls=media_urls,
            media_types=media_types, media_text_inlined=media_text_inlined, channel_context=channel_context,
            reply_expected=self._slack_reply_expected(
                routing_text, bot_uid, channel_id=channel_id, opens_own_session=thread_ts == ts,
                addressed=is_one_to_one_dm or is_mentioned or is_command_text or force_process),
            restored=cached is not None)
        if shared_sections:
            msg_event.add_channel_context(attributed_context("Shared links and messages", "\n\n".join(shared_sections)))
        # React only when directly addressed; MPIMs are shared, so they need a
        # mention like any channel.
        if cached is None and (is_one_to_one_dm or is_mentioned) and self._reactions_enabled():
            self._track_reacting_message(team_id, ts)
        # App-context is per-turn UI state: in the user message, not SessionSource (would rebuild
        # the agent per view switch and leak stale context). Inert label, never a channel body.
        context_channel_id = agent_context.get("context_channel_id", "")
        if context_channel_id and context_channel_id != channel_id and not is_command_text:
            msg_event.text = (
                f"[Slack app context: user is viewing channel {context_channel_id}]\n\n"
                f"{msg_event.text}")
        if ts and cached is None:
            self._remember_processed_message_ts(ts)
        if cached is None:
            msg_event._pending_native_input = self.pending_native_input(msg_event, thread_context=thread_snapshot)
        return msg_event


    async def _build_message_event(
        self: SlackAdapter, event: dict, *, text: str, original_text: str, command_probe_text: str,
        is_command_text: bool, channel_id: str, team_id: str, ts: str, user_id: str,
        thread_ts: Optional[str], is_dm: bool, media_urls: list[str], media_types: list[str],
        media_text_inlined: Sequence[bool | None], channel_context: Optional[str], reply_expected: Optional[bool] = None,
        restored: bool = False) -> MessageEvent:
        """Resolve names, title the DM thread, and build the ``MessageEvent``. Commands are restored
        from canonical input: the parser needs the token at char zero and enrichment (blocks,
        unfurls, file text, history) must never mutate arguments."""
        if is_command_text:
            text = command_probe_text
        msg_type = MessageType.COMMAND if is_command_text else self._media_message_type(media_types)
        user_name = await self._resolve_user_name(user_id, chat_id=channel_id, team_id=team_id)
        channel_name = await self._resolve_channel_name(channel_id, team_id=team_id)
        # Best-effort: title the DM thread from the prompt for Slack's AI Agent Messages tab.
        if not restored and is_dm and thread_ts and msg_type != MessageType.COMMAND:
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
            message_id=ts,
            # Workflow/app posts have user=None; flag them so the SLACK_ALLOW_BOTS bypass can
            # authorize them. Same predicate as the drop gate (api_human_users stay human).
            is_bot=self._event_declares_bot_sender(event))
        from gateway.platforms.base import resolve_channel_skills
        # Remaining ``<@UID>`` are OTHER participants (own mention stripped
        # above); render as ``@DisplayName`` so the agent knows who is addressed.
        text = await self._humanize_user_mentions(text, chat_id=channel_id, team_id=team_id)
        return MessageEvent(
            text=(command_probe_text if is_command_text else text),
            message_type=msg_type,
            source=source,
            raw_message=event,
            message_id=ts,
            media_urls=media_urls,
            media_types=media_types,
            media_text_inlined=list(media_text_inlined),
            reply_to_message_id=thread_ts if thread_ts != ts else None,
            channel_prompt=self._channel_prompt_with_identity(channel_id, team_id),
            channel_context=channel_context,
            reply_expected=reply_expected,
            # thread_ts is the thread root, not an explicit reply (root is in channel_context).
            reply_to_text=None,
            auto_skill=resolve_channel_skills(self.config.extra, channel_id, None),
            metadata={
                "slack_team_id": team_id, "slack_channel_id": channel_id,
                "slack_thread_ts": thread_ts})


    @staticmethod
    def _link_unfurl_sections(text: str, slack_attachments: list) -> list[str]:
        """Render link-unfurl previews (``attachments``) not already in ``text``; ``is_msg_unfurl``
        echoes our own content and is skipped. Dedup matches the rendered section, not the bare URL
        (which is usually already in the user's text while the preview body is not)."""
        from plugins.platforms.slack.adapter import (
            ELISION_MARKER_MAX_LEN,
            _SLACK_UNFURL_BLOCKS_MAX_CHARS,
            _extract_text_from_slack_blocks,
            elide,
        )

        att_parts: list[str] = []
        blocks_budget = _SLACK_UNFURL_BLOCKS_MAX_CHARS
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
            # Pasted tables arrive as ``table`` blocks in ``attachments[].blocks[]``, absent from
            # ``text``/``fallback``/files; without this the agent sees only the sentence before them.
            # The budget is shared across the whole array: a 20-attachment alert must not project
            # 20x what a single one does, and a spent budget still leaves the header visible.
            nested_text = ""
            if blocks_budget > 0:
                nested_text = _extract_text_from_slack_blocks(att.get("blocks") or [])
                if len(nested_text) > blocks_budget and blocks_budget <= ELISION_MARKER_MAX_LEN:
                    nested_text = ""  # leftover budget cannot hold marker + content: skip, don't overshoot
                nested_text = elide(nested_text, blocks_budget)
            if nested_text and nested_text not in body:
                blocks_budget -= len(nested_text)
                body = f"{body}\n{nested_text}".strip() if body else nested_text
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
        return att_parts

    async def _handle_slack_message(self: SlackAdapter, event: dict, payload: Optional[dict] = None) -> None:
        """Guard around :meth:`_handle_slack_message_impl`: the impl claims the ts early (no second
        turn from a mid-flight unfurl); if THIS call newly claimed it and raises, release the claim
        so a retry/edit can re-drive it. Pre-existing claims stay."""
        if event.get("subtype") == "message_deleted":
            self._withdraw_slack_message(event, payload)
            return
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
                logger.warning(
                    "[%s] handler failed after claiming ts=%s; claim released "
                    "so a retry or edit can re-drive the turn", self.name, _ts)
            raise

    async def _drop_bot_sender(self: SlackAdapter, event: dict) -> bool:
        """allow_bots gate: ``none`` drops all bot posts (default), ``mentions`` those not
        @mentioning us, ``all`` accepts — own posts always drop (echo loops). Unlabeled events
        without ``client_msg_id`` are probed via users.info (humans carry it, stray bots don't)."""
        from plugins.platforms.slack.adapter import _slack_mention_detection_text
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
            text_check = _slack_mention_detection_text(event)
            if self._bot_user_id and f"<@{self._bot_user_id}>" not in text_check:
                logger.debug(
                    "[Slack] Dropping bot message under allow_bots=mentions: "
                    "no <@%s> mention in flat text or blocks", self._bot_user_id)
                return True
        return bool(msg_user and self._bot_user_id and msg_user == self._bot_user_id)

    async def _prefilter_inbound(
        self: SlackAdapter, event: dict, payload: Optional[dict]) -> Optional[tuple[dict, str, str]]:
        """Normalize edits, then drop replays / ignored channels / bot posts / deletions.
        Returns ``(event, team_id, channel_id)`` for messages the handler should consider."""
        # Entry log BEFORE any filtering so operators can tell "dropped here"
        # from "never subscribed in the manifest". Metadata only, never text.
        # DEBUG entry log — fires BEFORE any filtering so users debugging bot-to-bot interop, allow_bots
        # config, or SLACK_ALLOWED_USERS drops can confirm whether the event actually arrived from Slack
        # (vs. being silently filtered upstream by the app's event subscriptions — Socket Mode will not
        # deliver events the app manifest hasn't subscribed to). See #30091.
        if logger.isEnabledFor(logging.DEBUG):
            _bot_profile = event.get("bot_profile") or {}
            logger.debug(
                "[Slack] event received type=%s subtype=%s user=%s bot_id=%s bot_name=%s "
                "channel=%s ts=%s thread_ts=%s", event.get("type"), event.get("subtype"),
                event.get("user", "") or "", event.get("bot_id", "") or "",
                (_bot_profile.get("name") if isinstance(_bot_profile, dict) else "") or "",
                event.get("channel", ""), event.get("ts", ""), event.get("thread_ts", ""))
        if event.get("subtype") == "message_changed":
            normalized_event = self._normalize_changed_message(event)
            if normalized_event is None:
                return None
            event = normalized_event
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
            logger.info("[Slack] Ignoring message in configured ignored channel %s", channel_id)
            return None
        if await self._drop_bot_sender(event):
            return None
        # Edits were normalized above so an @mention added by edit can wake the bot once;
        # the normalized event retains the edited message's own subtype.
        # Housekeeping subtypes (joins/leaves, topic/name/purpose changes, convert_to_private/
        # public, pins, deletions, file comments...) are not a person speaking, so they must
        # not start a turn in free-response channels (#110778). Allowlist rather than denylist
        # so subtypes Slack adds later are dropped instead of silently readmitted.
        # ``file_share`` passes: a human attaching a file is a person speaking, and the
        # ``file_shared`` fallback synthesizes exactly this subtype. ``thread_broadcast``
        # passes: a human sharing a threaded reply into the channel carries user/text.
        # ``me_message`` passes: ``/me`` is a person speaking.
        # ``bot_message`` already passed allow_bots above; document_mention is an
        # explicit app mention from a Slack canvas, not a lifecycle notification.
        # ``file_comment`` stays in the drop set deliberately (triage decision on
        # #110778): a comment left on a file is not the owner talking to the bot.
        subtype = event.get("subtype")
        if subtype not in (
            None, "", "file_share", "thread_broadcast", "me_message",
            "bot_message", "document_mention",
        ):
            logger.debug(
                "[Slack] Dropping non-conversational message subtype=%s in channel %s",
                subtype, channel_id)
            return None
        return event, dedup_team_id, channel_id

    def _withdraw_slack_message(self: SlackAdapter, event: dict, payload: Optional[dict]) -> None:
        from gateway.native_message_deletion import NativeMessageDeletion

        team_id = self._event_team_id({}, payload)
        event_team_id = self._event_team_id(event)
        if not team_id or (event_team_id and event_team_id != team_id):
            return
        if team_id not in self._team_clients:
            return
        channel_id = event.get("channel")
        message_id = event.get("deleted_ts")
        if not isinstance(channel_id, str) or not channel_id or not isinstance(message_id, str) or not message_id:
            return
        previous = event.get("previous_message")
        thread_id = previous.get("thread_ts") if isinstance(previous, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            thread_id = None
        self.withdraw_native_messages(NativeMessageDeletion(
            platform=self.platform, scope_id=team_id, chat_id=channel_id,
            message_ids=(message_id,), thread_id=thread_id))

    async def _hydrate_thread_context(
        self: SlackAdapter, *, channel_id: str, event_thread_ts, ts: str, user_id: str, team_id: str,
        is_thread_reply: bool, is_mentioned: bool, is_dm: bool,
    ) -> tuple[Optional[str], list[str], list[str], SlackThreadContext | None]:
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
        channel_context = None
        context_messages: list[dict] = []
        # Thread-root images recovered on the cold-start hydrate: when the bot is mentioned mid-thread for
        # the first time, the thread root is very often the artifact the mention is about ("@bot what's in
        # this chart?" replying under an image post) — deliver its images with this first turn. One-time by
        # construction: the cold-start path is guarded by _has_active_session_for_thread, so subsequent
        # turns in the same session never re-deliver (adapted from #69185).
        thread_root_media_urls: list[str] = []
        thread_root_media_types: list[str] = []
        if not is_thread_reply:
            return channel_context, thread_root_media_urls, thread_root_media_types, None
        has_active_thread_session = self._has_active_session_for_thread(
            channel_id=channel_id, thread_ts=event_thread_ts, user_id=user_id, team_id=team_id,
            chat_type="dm" if is_dm else "group")

        async def _fetch(**kw) -> None:
            nonlocal channel_context, context_messages
            thread_context = await self._fetch_thread_context(
                channel_id=channel_id, thread_ts=event_thread_ts, current_ts=ts, team_id=team_id,
                **kw)
            if thread_context:
                channel_context = thread_context
                cached_context = self._thread_context_cache.get(self._thread_cache_key(channel_id, event_thread_ts, team_id))
                after_ts = kw.get("after_ts", "")
                context_messages = [message for message in cached_context.messages
                                    if message.get("ts") != ts and (not after_ts or message.get("ts", "") > after_ts)] if cached_context else []

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
                return channel_context, thread_root_media_urls, thread_root_media_types, SlackThreadContext(None, (), ())
            watermark_ts = self._get_thread_watermark(**watermark_args)
            if watermark_ts:
                await _fetch(after_ts=watermark_ts, force_refresh=True)
        self._set_thread_watermark(watermark_ts=ts, **watermark_args)
        self._mark_thread_rehydration_checked(channel_id, event_thread_ts, user_id, team_id)
        try:
            snapshot = SlackThreadContext.capture(channel_context, context_messages, thread_root_media_urls, event_thread_ts)
        except (ValueError, KeyError, TypeError):
            snapshot = None
        return channel_context, thread_root_media_urls, thread_root_media_types, snapshot

def _is_shared_slack_attachment(att: dict) -> bool:
    """Whether an attachment shows someone else's content: a link preview or a shared message.
    Other attachments are content that the sender's app composed itself, such as an alert."""
    return bool(att.get("is_share") or att.get("from_url") or att.get("original_url"))
