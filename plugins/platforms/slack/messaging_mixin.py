"""Messaging family for the Slack platform adapter (extracted from the adapter god-file).

SlackMessagingMixin carries the message send/edit/delete and typing-indicator
methods that previously lived inline in ``SlackAdapter``.  The adapter inherits
this mixin FIRST (``class SlackAdapter(SlackMessagingMixin, BasePlatformAdapter)``)
so these overrides win the MRO over the base hooks they shadow.

Extraction slice R2-S1 of the adapter god-file kill (epic #78647, target
#78638).  Bodies are byte-verbatim moves; only imports and the class statement
were added.
"""

import logging
import time
from typing import Any, Dict, Optional

import aiohttp

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)


class SlackMessagingMixin:
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message to a Slack channel or DM."""
        if self._is_ignored_channel(chat_id):
            logger.warning(
                "[Slack] Suppressed outbound generic send to configured ignored channel %s",
                chat_id,
            )
            return SendResult(success=False, error="ignored_channel")
        if not self._app:
            return SendResult(success=False, error="Not connected")

        chat_id = await self._ensure_dm_conversation(
            chat_id, team_id=self._metadata_team_id(metadata)
        )
        thread_ts = None
        try:
            team_id = self._metadata_team_id(metadata)
            # Check for a pending slash-command context.  When the user ran a
            # native slash command (e.g. /q, /stop, /model), the initial ack
            # already showed an ephemeral "Running /cmd…" message.  If we have
            # a stashed response_url for this channel, replace that ack with
            # the actual command reply ephemerally instead of posting publicly.
            slash_ctx = self._pop_slash_context(chat_id, team_id)
            if slash_ctx:
                ephemeral_result = await self._send_slash_ephemeral(
                    slash_ctx,
                    content,
                )
                if ephemeral_result.success:
                    # Ephemeral replies do not count as thread replies, so
                    # Slack never auto-clears the Assistant status for them.
                    # Clear it explicitly or a command run inside an
                    # assistant thread leaves "is thinking..." forever.
                    await self._clear_thread_status_quietly(chat_id, metadata)
                    return ephemeral_result
                # response_url delivery failed (#19688): fall back to
                # chat.postEphemeral — an independent API path that keeps
                # the reply private ("Only visible to you"). We do NOT fall
                # back to a public channel post: a slash reply the user
                # expects to be ephemeral must never surface to the whole
                # channel just because a delivery path failed.
                logger.warning(
                    "[Slack] response_url slash reply failed (%s); retrying "
                    "via chat.postEphemeral",
                    ephemeral_result.error,
                )
                fallback_result = await self._post_ephemeral_fallback(
                    chat_id,
                    slash_ctx,
                    content,
                )
                if fallback_result.success:
                    await self._clear_thread_status_quietly(chat_id, metadata)
                    return fallback_result
                # Both ephemeral paths failed — surface the failure instead
                # of leaking the reply publicly. The user still has the
                # "Running /cmd…" ack; the error is logged and returned so
                # the gateway can react (retry surfacing happens upstream).
                logger.error(
                    "[Slack] Ephemeral slash reply failed on both "
                    "response_url and chat.postEphemeral (%s); dropping "
                    "rather than posting publicly",
                    fallback_result.error,
                )
                return fallback_result

            # Convert standard markdown → Slack mrkdwn
            formatted = self.format_message(content)

            # Guard against empty/whitespace-only messages — Slack API
            # returns ``no_text`` for chat.postMessage with blank text.
            if not formatted or not formatted.strip():
                # This is still the end of a delivery attempt: if the turn
                # produced no visible text (e.g. "(empty)" final responses
                # are filtered upstream), the assistant thread status must
                # not stay stuck on "is thinking..." (#24117).
                await self._clear_thread_status_quietly(chat_id, metadata)
                return SendResult(success=True)

            # Split long messages, preserving code block boundaries
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            last_result = None

            # reply_broadcast: also post thread replies to the main channel.
            # Controlled via platform config: gateway.slack.reply_broadcast
            broadcast = self.config.extra.get("reply_broadcast", False)

            # Block Kit (opt-in): render the primary message as structured
            # blocks. Only applied to a single-chunk message — a >39k response
            # that had to be split is pathological for Block Kit's 50-block /
            # 3000-char limits, so those fall back to plain text. The ``text``
            # field is always kept as the notification/accessibility fallback.
            blocks = self._maybe_blocks(content) if len(chunks) == 1 else None

            for i, chunk in enumerate(chunks):
                kwargs = {
                    "channel": chat_id,
                    "text": chunk,
                    "mrkdwn": True,
                }
                if blocks and i == 0:
                    kwargs["blocks"] = blocks
                if thread_ts:
                    kwargs["thread_ts"] = thread_ts
                    # Only broadcast the first chunk of the first reply
                    if broadcast and i == 0:
                        kwargs["reply_broadcast"] = True

                try:
                    last_result = await self._get_client(
                        chat_id, team_id=team_id
                    ).chat_postMessage(**kwargs)
                except Exception as e:
                    if kwargs.get("blocks") and self._is_block_payload_rejection(e):
                        retry_kwargs = dict(kwargs)
                        retry_kwargs.pop("blocks", None)
                        logger.info(
                            "[Slack] Block Kit payload rejected; retrying send without blocks: %s",
                            e,
                        )
                        last_result = await self._get_client(
                            chat_id, team_id=team_id
                        ).chat_postMessage(**retry_kwargs)
                    else:
                        raise

            # Clear Slack Assistant status as soon as the final message is posted.
            if thread_ts:
                await self.stop_typing(chat_id, metadata=metadata)

            # Track the sent message ts so we can auto-respond to thread
            # replies without requiring @mention.
            sent_ts = last_result.get("ts") if last_result else None
            if sent_ts:
                self._bot_message_ts.add(
                    self._workspace_message_marker(team_id, sent_ts)
                )
                # Also register the thread root so replies-to-my-replies work
                if thread_ts:
                    self._bot_message_ts.add(
                        self._workspace_message_marker(team_id, thread_ts)
                    )
                self._trim_bot_message_timestamps()

            return SendResult(
                success=True,
                message_id=sent_ts,
                raw_response=last_result,
            )

        except Exception as e:  # pragma: no cover - defensive logging
            # Clear the assistant status even when the failure happened
            # BEFORE thread_ts was resolved (formatting, slash-context, DM
            # resolution): stop_typing falls back to metadata / the uniquely
            # tracked status for this channel, so a failed turn cannot leave
            # "is thinking..." visible (#24117).
            await self._clear_thread_status_quietly(chat_id, metadata)
            logger.error("[Slack] Send error: %s", e, exc_info=True)
            _retryable = self._is_retryable_upload_error(e)
            _retry_after = None
            if _retryable:
                _resp = getattr(e, "response", None)
                if _resp is not None:
                    try:
                        _ra = getattr(_resp, "headers", {}).get("Retry-After")
                        if _ra is not None:
                            _retry_after = float(_ra)
                    except (TypeError, ValueError, AttributeError):
                        pass
            return SendResult(
                success=False,
                error=str(e),
                retryable=_retryable,
                retry_after=_retry_after,
            )

    async def send_private_notice(
        self,
        chat_id: str,
        user_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a Slack ephemeral message visible only to one user."""
        if self._is_ignored_channel(chat_id):
            logger.warning(
                "[Slack] Suppressed outbound generic ephemeral notice to configured ignored channel %s",
                chat_id,
            )
            return SendResult(success=False, error="ignored_channel")
        if not self._app:
            return SendResult(success=False, error="Not connected")
        if not chat_id or not user_id:
            return SendResult(success=False, error="chat_id and user_id are required")

        try:
            formatted = self.format_message(content)
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            kwargs = {
                "channel": chat_id,
                "user": user_id,
                "text": formatted,
                "mrkdwn": True,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

            result = await self._get_client(
                chat_id, team_id=self._metadata_team_id(metadata)
            ).chat_postEphemeral(**kwargs)
            return SendResult(
                success=True,
                message_id=result.get("message_ts") or result.get("ts"),
                raw_response=result,
            )
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[Slack] Ephemeral send error: %s", e, exc_info=True)
            return SendResult(success=False, error=str(e))

    async def send_or_update_status(
        self,
        chat_id: str,
        status_key: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a status message, or edit the previous one with the same key.

        Issue #30045 (Telegram) extended to Slack: progress/status callbacks
        (context-pressure, compression retries, model fallback, lifecycle)
        used to append a fresh bubble on every call, spamming threads during
        long retry loops. The first call posts and the message ts is
        remembered; subsequent calls with the same (channel, thread,
        status_key) edit that message in place via ``chat.update``. If the
        edit fails (message deleted, too old, ...) the cached ts is dropped
        and a fresh message is sent.
        """
        thread_ts = self._resolve_thread_ts(None, metadata) or ""
        key = (str(chat_id), str(thread_ts), str(status_key))
        cached_id = self._status_message_ids.get(key)
        if cached_id is not None:
            result = await self.edit_message(
                chat_id, cached_id, content, finalize=False, metadata=metadata,
            )
            if result.success:
                if result.message_id:
                    self._status_message_ids[key] = str(result.message_id)
                return result
            # Edit failed — clear the cached ts and fall through to a fresh send.
            self._status_message_ids.pop(key, None)
        result = await self.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            if len(self._status_message_ids) >= self._STATUS_MESSAGE_IDS_MAX:
                # Simple FIFO trim: drop the oldest half to bound memory.
                for stale in list(self._status_message_ids)[
                    : self._STATUS_MESSAGE_IDS_MAX // 2
                ]:
                    self._status_message_ids.pop(stale, None)
            self._status_message_ids[key] = str(result.message_id)
        return result

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a previously sent Slack message."""
        if self._is_ignored_channel(chat_id):
            logger.warning(
                "[Slack] Suppressed message edit in configured ignored channel %s",
                chat_id,
            )
            return SendResult(success=False, error="ignored_channel")
        if not self._app:
            return SendResult(success=False, error="Not connected")
        try:
            formatted = self.format_message(content)
            # Slack's chat.update has the same ~40k char limit as postMessage.
            # Unlike send() we can't split into multiple messages (we're
            # editing an existing one), so truncate to fit — an oversized
            # payload fails the whole edit with ``msg_too_long``.
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
            formatted = chunks[0] if chunks else formatted
            update_kwargs: Dict[str, Any] = {
                "channel": chat_id,
                "ts": message_id,
                "text": formatted,
            }
            # Only render Block Kit on the FINAL edit. Intermediate streaming
            # edits stay plain mrkdwn — re-deriving a full block layout on every
            # progressive flush would be wasteful and jittery. ``text`` is kept
            # as the fallback either way.
            if finalize:
                blocks = self._maybe_blocks(content)
                if blocks:
                    update_kwargs["blocks"] = blocks
            try:
                await self._get_client(
                    chat_id, team_id=self._metadata_team_id(metadata)
                ).chat_update(**update_kwargs)
            except Exception as e:
                if update_kwargs.get("blocks") and self._is_block_payload_rejection(e):
                    retry_kwargs = dict(update_kwargs)
                    # Explicitly clear any stale blocks when falling back to the
                    # flat text update path; otherwise Slack can preserve the
                    # prior block layout for an edited message.
                    retry_kwargs["blocks"] = []
                    logger.info(
                        "[Slack] Block Kit payload rejected; retrying edit without blocks: %s",
                        e,
                    )
                    await self._get_client(
                        chat_id, team_id=self._metadata_team_id(metadata)
                    ).chat_update(**retry_kwargs)
                else:
                    raise
            if finalize:
                await self._clear_thread_status_quietly(chat_id, metadata)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:  # pragma: no cover - defensive logging
            if finalize:
                await self._clear_thread_status_quietly(chat_id, metadata)
            aiohttp_module = globals().get("aiohttp")
            connection_error_type = getattr(
                aiohttp_module, "ClientConnectionError", None
            )
            permanent_tls_error_types = tuple(
                error_type
                for error_type in (
                    getattr(aiohttp_module, "ClientSSLError", None),
                    getattr(aiohttp_module, "ServerFingerprintMismatch", None),
                )
                if isinstance(error_type, type)
            )
            is_permanent_tls_error = bool(permanent_tls_error_types) and isinstance(
                e, permanent_tls_error_types
            )
            is_transient_transport_error = isinstance(e, TimeoutError) or (
                isinstance(connection_error_type, type)
                and isinstance(e, connection_error_type)
                and not is_permanent_tls_error
            )
            if is_transient_transport_error:
                # chat.update is idempotent: keep this message ID after a
                # transport failure so a later edit can catch up. Treating the
                # failure as permanent makes every later tool update a new post.
                logger.error(
                    "[Slack] transient chat.update failure on message %s in channel %s: %s",
                    message_id,
                    chat_id,
                    e,
                    exc_info=True,
                )
                return SendResult(
                    success=False,
                    error=str(e),
                    retryable=True,
                    error_kind="transient",
                )
            logger.error(
                "[Slack] Failed to edit message %s in channel %s: %s",
                message_id,
                chat_id,
                e,
                exc_info=True,
            )
            return SendResult(success=False, error=str(e))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a Slack message previously sent by this bot.

        Used by gateway progress cleanup so temporary "Working"/tool-progress
        bubbles do not remain after a successful final response.
        """
        if not self._app:
            return False
        try:
            response = await self._get_client(chat_id).chat_delete(channel=chat_id, ts=message_id)
            if hasattr(response, "get") and response.get("ok") is False:
                logger.debug(
                    "[Slack] chat.delete returned ok=false for message %s in channel %s: %s",
                    message_id,
                    chat_id,
                    response.get("error", "unknown"),
                )
                return False
            return True
        except Exception as e:  # pragma: no cover - best-effort cleanup
            logger.debug(
                "[Slack] Failed to delete message %s in channel %s: %s",
                message_id,
                chat_id,
                e,
            )
            return False

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show a typing/status indicator using assistant.threads.setStatus.

        Displays "is thinking..." next to the bot name in a thread, or the
        platform's ``typing_status_text`` config value when set.
        Requires the assistant:write or chat:write scope.
        Auto-clears when the bot sends a reply to the thread.
        """
        if self._is_ignored_channel(chat_id):
            logger.debug("[Slack] Suppressed typing/status in configured ignored channel %s", chat_id)
            return
        if not self._app:
            return

        thread_ts = None
        if metadata:
            # Reuse the same synthetic-thread guard as message sending. When
            # reply_in_thread=false, top-level channel events carry their own
            # message ts as metadata.thread_id for session keying. Calling
            # assistant_threads_setStatus on that ts activates a Slack assistant
            # thread before the actual response is sent.
            thread_ts = self._resolve_thread_ts(
                reply_to=metadata.get("message_id"),
                metadata=metadata,
            )

        if not thread_ts:
            return  # Can only set status in a thread context

        team_id = self._metadata_team_id(metadata)
        if not team_id:
            team_id = self._channel_team.get(chat_id, "")

        status_key = self._workspace_thread_key(team_id, chat_id, str(thread_ts))
        _status_started: Optional[float] = None
        if status_key:
            # Heartbeat (#45702): preserve the first refresh's start time
            # across _keep_typing refreshes so a long turn surfaces elapsed
            # time ("still working… (2m03s)") instead of a static
            # "is thinking..." that reads as stuck — which is what provokes
            # mid-turn "you there?" pings. Stored inside the tracked status
            # entry so it shares the existing bounds/eviction and is dropped
            # by stop_typing with the rest of the status state.
            _prev_entry = self._active_status_threads.get(status_key)
            if isinstance(_prev_entry, dict):
                _status_started = _prev_entry.get("started")
            if not isinstance(_status_started, (int, float)):
                _status_started = time.monotonic()
            self._active_status_threads[status_key] = {
                "thread_ts": str(thread_ts),
                "team_id": str(team_id) if team_id else "",
                "started": _status_started,
            }
            if len(self._active_status_threads) > self._ACTIVE_STATUS_THREADS_MAX:
                # Evict abandoned statuses oldest-thread-first (key[2] is the
                # thread ts) so an eviction never clears the newest status.
                excess = (
                    len(self._active_status_threads)
                    - self._ACTIVE_STATUS_THREADS_MAX // 2
                )
                oldest = sorted(
                    self._active_status_threads,
                    key=lambda k: self._slack_timestamp_sort_key(k[2]),
                )[:excess]
                for old_key in oldest:
                    self._active_status_threads.pop(old_key, None)
        try:
            _status = (
                getattr(self, "_status_text", {}).get(str(chat_id))
                or getattr(self.config, "typing_status_text", None)
            )
            if not _status:
                # Heartbeat (#45702): once a turn has run for 30s+, replace
                # the static default with visible elapsed progress. Only the
                # fallback label changes — explicit live-status phrases and
                # configured typing_status_text always win.
                _elapsed = (
                    int(time.monotonic() - _status_started)
                    if _status_started is not None
                    else 0
                )
                if _elapsed >= 30:
                    _mins, _secs = divmod(_elapsed, 60)
                    _human = f"{_mins}m{_secs:02d}s" if _mins else f"{_secs}s"
                    _status = f"still working… ({_human})"
                else:
                    _status = "is thinking..."
            await self._get_client(chat_id, team_id=team_id).assistant_threads_setStatus(
                channel_id=chat_id,
                thread_ts=thread_ts,
                status=_status,
            )
        except Exception as e:
            # Silently ignore — may lack assistant:write scope or not be
            # in an assistant-enabled context. Falls back to reactions.
            logger.debug("[Slack] assistant.threads.setStatus failed: %s", e)

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        """Clear the assistant thread status indicator."""
        if self._is_ignored_channel(chat_id):
            logger.debug("[Slack] Suppressed status clear in configured ignored channel %s", chat_id)
            self._active_status_threads.pop(chat_id, None)
            return
        if not self._app:
            return
        requested_thread_ts = ""
        if metadata:
            requested_thread_ts = str(
                metadata.get("thread_id") or metadata.get("thread_ts") or ""
            )
        requested_team_id = self._metadata_team_id(metadata)
        active = None
        ambiguous_tracked = False
        if requested_thread_ts:
            if requested_team_id:
                active_key = self._workspace_thread_key(
                    requested_team_id, chat_id, requested_thread_ts
                )
                if active_key:
                    active = self._active_status_threads.pop(active_key, None)
            else:
                # Do not trust the mutable channel-only workspace fallback for
                # a thread-specific cleanup: Slack Connect workspaces can share
                # a channel ID. Clear the uniquely matching tracked status and
                # let its stored team choose the correct client.
                matching_keys = [
                    key
                    for key in self._active_status_threads
                    if key[1] == str(chat_id) and key[2] == requested_thread_ts
                ]
                if len(matching_keys) == 1:
                    active = self._active_status_threads.pop(matching_keys[0], None)
                ambiguous_tracked = len(matching_keys) > 1
        else:
            # Metadata-free cleanup is safe only if exactly one status exists
            # for this channel; otherwise it may clear another Slack Connect
            # workspace's Assistant status.
            matching_keys = [
                key
                for key in self._active_status_threads
                if key[1] == str(chat_id)
            ]
            if len(matching_keys) == 1:
                active = self._active_status_threads.pop(matching_keys[0], None)
        if isinstance(active, str):
            thread_ts = active
            team_id = ""
        else:
            active = active or {}
            thread_ts = active.get("thread_ts", "")
            team_id = active.get("team_id", "")
        if metadata:
            team_id = self._metadata_team_id(metadata) or team_id
        if not thread_ts and requested_thread_ts and not ambiguous_tracked:
            # No tracked entry (gateway restart, eviction, or a status set
            # before this process started) but the caller identified the exact
            # thread to clear. Issue the clear anyway so a stuck "is
            # thinking..." can always be dismissed — clearing an unset status
            # is a harmless no-op on Slack's side. Skipped when MULTIPLE
            # workspaces track this channel+thread (ambiguous_tracked): a
            # team-less clear there could hit the wrong Slack Connect
            # workspace. Client routing uses the caller's team when given,
            # else the channel→team fallback.
            thread_ts = requested_thread_ts
            team_id = requested_team_id or team_id
        if not thread_ts:
            return
        try:
            await self._get_client(chat_id, team_id=team_id).assistant_threads_setStatus(
                channel_id=chat_id,
                thread_ts=thread_ts,
                status="",
            )
        except Exception as e:
            logger.debug("[Slack] assistant.threads.setStatus clear failed: %s", e)
