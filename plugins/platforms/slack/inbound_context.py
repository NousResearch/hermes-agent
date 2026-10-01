"""Inbound text and context preparation for the SlackAdapter adapter."""

from __future__ import annotations

from gateway.platforms.event import attributed_context

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from gateway.platforms.event import MessageEvent, MessageType

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
        from plugins.platforms.slack.adapter import (
            _rewrite_known_bang_command,
            _slack_mention_detection_text,
        )

        accepted = await self._prefilter_inbound(event, payload)
        if accepted is None:
            return
        event, dedup_team_id, channel_id = accepted
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
        media_urls, media_types, media_text_inlined, text = await self._collect_inbound_media(
            event, channel_id, team_id, text, thread_root_media_urls, thread_root_media_types)
        msg_event = await self._build_message_event(
            event, text=text, original_text=original_text, command_probe_text=command_probe_text,
            is_command_text=is_command_text, channel_id=channel_id, team_id=team_id, ts=ts,
            user_id=user_id, thread_ts=thread_ts, is_dm=is_dm, media_urls=media_urls,
            media_types=media_types, media_text_inlined=media_text_inlined, channel_context=channel_context,
            reply_expected=self._slack_reply_expected(
                routing_text, bot_uid, channel_id=channel_id, opens_own_session=thread_ts == ts,
                addressed=is_one_to_one_dm or is_mentioned or is_command_text or force_process))
        if shared_sections:
            msg_event.add_channel_context(attributed_context("Shared links and messages", "\n\n".join(shared_sections)))
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




def _is_shared_slack_attachment(att: dict) -> bool:
    """Whether an attachment shows someone else's content: a link preview or a shared message.
    Other attachments are content that the sender's app composed itself, such as an alert."""
    return bool(att.get("is_share") or att.get("from_url") or att.get("original_url"))
