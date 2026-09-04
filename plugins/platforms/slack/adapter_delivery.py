"""Slack delivery methods; SDK and mutable dependencies remain on the facade."""

import logging
from gateway.native_document_guard import check_document_fallback, mark_native_document_guard

from typing import Any, Callable, Dict, List, Optional, Tuple
from gateway.platforms.base import SendResult
try:
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncApp = AsyncWebClient = Any


class SlackDeliveryMixin:
    def _pop_slash_context(self, chat_id: str, team_id: str = "") -> Optional[Dict[str, Any]]:
        """Pop the fresh slash context for *chat_id*, matched on the exact ``(team_id, channel_id,
        user_id)`` key via the ``_slash_user_id`` ContextVar so a concurrent slash from another
        user/workspace can't steal it. ContextVar unset (non-slash send) matches nothing, else
        normal sends would steal a pending slash reply."""
        from . import adapter as _adapter

        self._purge_stale_slash_contexts()  # dict is small; purge on every lookup
        team_id = str(team_id or "")
        uid = _adapter._slash_user_id.get()
        if uid:
            key = (team_id, chat_id, uid) if team_id else (chat_id, uid)
            return self._slash_command_contexts.pop(key, None)
        return None

    def _purge_stale_slash_contexts(self) -> None:
        from . import adapter as _adapter

        now = _adapter.time.monotonic()
        for k in [
            k for k, v in self._slash_command_contexts.items()
            if now - v["ts"] > self._SLASH_CTX_TTL]:
            self._slash_command_contexts.pop(k, None)

    def _format_chunks(self, content: str) -> List[str]:
        """mrkdwn-format ``content`` and split to ``MAX_MESSAGE_LENGTH`` (never empty)."""
        formatted = self.format_message(content)
        return self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH) or [formatted]

    async def _send_slash_ephemeral(self, ctx: Dict[str, Any], content: str) -> "SendResult":
        """Replace the ephemeral ack via ``response_url`` (``replace_original`` valid 30 min). First
        chunk replaces the ack, the rest post as new ephemerals; Slack caps a response_url at 5
        POSTs so overflow gets a truncation notice. ``success=False`` lets ``send()`` fall back.

        Long replies are chunked: the first chunk replaces the ack, the rest are posted as additional
        ephemeral messages. Slack allows at most 5 POSTs to a response_url, so anything beyond that is
        closed with an explicit truncation notice instead of being silently dropped (#19688).
        Returns ``success=False`` on delivery failure so the caller (``send()``) can fall back to normal
        channel delivery — the reply must never be silently dropped just because the ephemeral swap failed
        (#19688).
        """
        # Slack's response_url has the same ~40k char limit as chat_postMessage.
        from . import adapter as _adapter

        chunks = self._format_chunks(content)
        # 5-POST cap per response_url: 1 replace + 4 follow-ups; announce the rest.
        if len(chunks) > 5:
            dropped = len(chunks) - 5
            chunks = chunks[:5]
            chunks[-1] = (
                chunks[-1].rstrip() + f"\n\n_[Reply truncated: {dropped} more part(s) exceeded "
                "Slack's ephemeral reply limit.]_")
        try:
            async with _adapter.aiohttp.ClientSession(trust_env=_adapter.gateway_trust_env()) as session:
                for idx, chunk in enumerate(chunks):
                    # Only the first chunk replaces the ack.
                    payload = {"response_type": "ephemeral", "replace_original": idx == 0, "text": chunk}
                    async with session.post(
                        ctx["response_url"], json=payload, timeout=_adapter.aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status != 200:
                            body = await _adapter._read_error_text_limited(resp)
                            _adapter.logger.warning(
                                "[Slack] response_url POST returned %s: %s", resp.status, body[:200]
                            )
                            return _adapter.SendResult(
                                success=False, error=f"response_url POST returned {resp.status}")
            return _adapter.SendResult(success=True, message_id=None)
        except Exception as e:
            _adapter.logger.warning("[Slack] response_url POST failed: %s", e)
            return _adapter.SendResult(success=False, error=str(e))

    async def _post_ephemeral_fallback(
        self, chat_id: str, ctx: Dict[str, Any], content: str) -> "SendResult":
        """Deliver a slash reply via ``chat.postEphemeral`` when ``response_url`` fails.
        Keeps the reply private (a public channel post must never happen for an ephemeral reply).
        Cannot ``replace_original``, so the ack stays; no 5-POST cap applies here.

        See #19688.
        """
        from . import adapter as _adapter

        user_id = ctx.get("user_id", "")
        if not user_id:
            return _adapter.SendResult(success=False, error="no user_id in slash context for postEphemeral")
        chunks = self._format_chunks(content)
        try:
            client = self._get_client(chat_id)
            for chunk in chunks:
                result = await client.chat_postEphemeral(channel=chat_id, user=user_id, text=chunk)
                payload = _adapter._slack_response_payload(result)
                if not payload.get("ok"):
                    err = payload.get("error", "unknown_error") if payload else "unexpected_response"
                    return _adapter.SendResult(success=False, error=f"chat.postEphemeral failed: {err}")
            return _adapter.SendResult(success=True, message_id=None)
        except Exception as e:
            return _adapter.SendResult(success=False, error=str(e))

    @staticmethod
    def _metadata_team_id(metadata: Optional[Dict[str, Any]]) -> str:
        """Return Slack workspace id from generic or Slack-specific metadata."""
        from . import adapter as _adapter

        if not metadata:
            return ""
        found = _adapter._first_truthy(
            metadata, ("scope_id", "slack_team_id", "team_id", "team", "guild_id", "workspace_id"))
        if found:
            return str(found)
        source = metadata.get("source")
        if isinstance(source, dict):
            found = _adapter._first_truthy(source, ("scope_id", "slack_team_id", "team_id", "guild_id"))
            if found:
                return str(found)
        elif source is not None:
            value = getattr(source, "scope_id", None) or getattr(source, "guild_id", None)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _workspace_event_id(team_id: str, event_id: str) -> str:
        """Scope Slack's workspace-local event/message ids for deduplication."""
        return f"{team_id}:{event_id}" if team_id else str(event_id)

    @staticmethod
    def _workspace_message_marker(team_id: str, message_id: str) -> Any:
        """Return an in-memory routing marker without changing legacy no-team tests."""
        return (str(team_id), str(message_id)) if team_id else str(message_id)

    def scope_id_for_chat(self, chat_id: str) -> Optional[str]:
        """Return the workspace id owning ``chat_id``.
        ``None`` for unknown channels and for channels claimed by several workspaces (dropped from
        the map) — no scope beats a wrong one."""
        team_id = chat_id and (getattr(self, "_channel_team", None) or {}).get(str(chat_id))
        return str(team_id) if team_id else None

    def _get_client(self, chat_id: str, team_id: Optional[str] = None) -> Any:
        """Return the workspace-specific WebClient for a channel."""
        if team_id and team_id in self._team_clients:
            return self._team_clients[team_id]
        team_id = self._channel_team.get(chat_id)
        if team_id and team_id in self._team_clients:
            return self._team_clients[team_id]
        return self._app.client  # fallback to primary

    def _client_for(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> Any:
        """WebClient for ``chat_id``, workspace-scoped by outbound ``metadata``."""
        return self._get_client(chat_id, team_id=self._metadata_team_id(metadata))

    async def _dm_target(self, chat_id: str, metadata: Optional[Dict[str, Any]]) -> str:
        """``_ensure_dm_conversation`` scoped by outbound ``metadata``."""
        return await self._ensure_dm_conversation(chat_id, team_id=self._metadata_team_id(metadata))

    async def _ensure_dm_conversation(self, chat_id: str, team_id: Optional[str] = None) -> str:
        """Resolve a bare user ID (U/W...) to a DM conversation ID via ``conversations.open``
        (``chat.postMessage``/``files_upload_v2`` reject user IDs); cached per (team, user). Returns
        ``chat_id`` unchanged when not applicable or on failure (downstream surfaces the error).

        Resolution goes through the workspace-scoped client so multi-workspace installs open the DM with the
        right bot token, and results are cached per (team, user) so repeated sends don't re-open. See
        #17261, #19236.
        """
        from . import adapter as _adapter

        cid = str(chat_id or "")
        if not cid or cid[0] not in ("U", "W"):
            return chat_id
        cache_key = f"{team_id or ''}:{cid}"
        cached = self._dm_conversation_cache.get(cache_key)
        if cached:
            return cached
        try:
            response = await self._get_client(cid, team_id=team_id).conversations_open(users=cid)
            dm_id = ((response or {}).get("channel") or {}).get("id")
            if dm_id:
                self._dm_conversation_cache[cache_key] = dm_id
                self._trim_oldest_dict_entries(
                    self._dm_conversation_cache, self._DM_CONVERSATION_CACHE_MAX)
                if team_id:
                    self._remember_channel_team(dm_id, team_id)
                return dm_id
        except Exception as e:
            _adapter.logger.warning(
                "[Slack] conversations.open failed for user target %s: %s "
                "(check the bot's im:write scope)", cid, e)
        return chat_id

    async def _clear_thread_status_quietly(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort status clear for send() paths that skip the normal clear (empty responses,
        ephemeral slash replies, exceptions before ``thread_ts`` resolved) so the thread doesn't
        stay "is thinking...". Errors must not mask the SendResult.

        Issue #24117: the assistant thread can stay stuck "is thinking..." when a turn ends through a path
        that never reaches the regular ``if thread_ts: stop_typing`` clear — an empty final response, a
        slash-command ephemeral reply, or an exception raised before ``thread_ts`` was resolved.
        ``stop_typing`` is already idempotent (clearing an unset status is a no-op on Slack's side), so this
        just guarantees it runs without letting a cleanup error mask the caller's SendResult.
        """
        from . import adapter as _adapter

        try:
            await self.stop_typing(chat_id, metadata=metadata)
        except Exception as e:  # pragma: no cover - defensive cleanup
            _adapter.logger.debug("[Slack] status cleanup failed: %s", e)

    def _is_ignored_channel(self, channel_id: str) -> bool:
        """Return True when the generic gateway must stay silent in this channel.
        Some paths carry thread-scoped ids (``C123:1712345678.000001``); matching is channel-level,
        so strip the suffix first."""
        if not channel_id:
            return False
        ignored = self._slack_ignored_channels()
        return "*" in ignored or str(channel_id).split(":", 1)[0] in ignored

    @staticmethod
    def _truthy_config(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def native_task_cards_enabled(self) -> bool:
        """Return whether Slack-native tool progress is explicitly enabled."""
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        streaming = extra.get("streaming")
        progress = streaming.get("progress") if isinstance(streaming, dict) else None
        for scope in (extra, progress if isinstance(progress, dict) else {}):
            value = scope.get("native_task_cards", scope.get("nativeTaskCards"))
            if value is not None:
                return self._truthy_config(value)
        return False

    def _native_task_card_key(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[str, str, str]]:
        thread_ts = self._resolve_thread_ts(reply_to, metadata)
        if not thread_ts:
            return None
        return self._workspace_thread_key(
            self._metadata_team_id(metadata), chat_id, str(thread_ts))

    async def send_native_task_card_progress(
        self, chat_id: str, tasks: List[Dict[str, str]], *, title: str = "Hermes is working",
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        fallback_text: Optional[str] = None) -> SendResult:
        """Start or update a Slack-native plan/task progress stream."""
        from . import adapter as _adapter

        if not self._app:
            return _adapter.SendResult(success=False, error="Not connected")
        if not tasks:
            return _adapter.SendResult(success=False, error="No tasks")
        key = self._native_task_card_key(chat_id, reply_to, metadata)
        if key is None:
            return _adapter.SendResult(success=False, error="No Slack thread target")
        stream = self._native_task_card_streams.get(key)
        if stream is None or stream.stopped:
            stream = _adapter._NativeTaskCardStream(team_id=key[0], channel=chat_id, thread_ts=key[2])
            # No await between lookup and assignment, so racers share this lock.
            self._native_task_card_streams[key] = stream
        async with stream.lock:
            if stream.stopped:
                return _adapter.SendResult(success=False, error="Progress stream already stopped")
            try:
                client = self._get_client(chat_id, team_id=stream.team_id)
                if not stream.stream_ts:
                    start_payload: _adapter.Dict[str, _adapter.Any] = {
                        "channel": chat_id, "thread_ts": stream.thread_ts,
                        "task_display_mode": "plan"}
                    md = metadata or {}
                    recipients = (
                        ("recipient_team_id", ("recipient_team_id", "team_id", "slack_team_id")),
                        ("recipient_user_id", ("recipient_user_id", "user_id")))
                    for key, sources in recipients:
                        value = _adapter._first_truthy(md, sources)
                        if value:
                            start_payload[key] = value
                    result = await client.api_call("chat.startStream", json=start_payload)
                    if hasattr(result, "get"):
                        stream.stream_ts = str(result.get("ts") or result.get("message_ts") or "")
                    if not stream.stream_ts:
                        raise RuntimeError("Slack startStream returned no stream timestamp")
                chunks: _adapter.List[_adapter.Dict[str, _adapter.Any]] = [{"type": "plan_update", "title": str(title)[:256]}]
                chunks.extend(self._task_update_chunk(task) for task in tasks)
                append_payload: _adapter.Dict[str, _adapter.Any] = {
                    "channel": chat_id, "ts": stream.stream_ts, "chunks": chunks}
                if fallback_text:
                    append_payload["markdown_text"] = fallback_text
                await client.api_call("chat.appendStream", json=append_payload)
                return _adapter.SendResult(success=True, message_id=stream.stream_ts)
            except Exception as exc:  # pragma: no cover - defensive logging
                _adapter.logger.error("[Slack] Native task-card progress error: %s", exc, exc_info=True)
                return _adapter.SendResult(success=False, error=str(exc), retryable=True)

    @staticmethod
    def _task_update_chunk(task: Dict[str, str]) -> Dict[str, Any]:
        """One ``task_update`` stream chunk; unknown statuses coerce to ``in_progress``."""
        status = str(task.get("status") or "in_progress")
        status = status if status in {"in_progress", "complete", "error"} else "in_progress"
        task_id = str(task.get("id") or task.get("task_id") or "task")
        return {
            "type": "task_update", "id": task_id, "title": str(task.get("title") or task_id)[:256],
            "status": status}

    async def stop_native_task_card_progress(
        self, chat_id: str, *, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Finalize an active Slack-native progress stream exactly once."""
        key = self._native_task_card_key(chat_id, reply_to, metadata)
        if key is None:
            return
        stream = self._native_task_card_streams.get(key)
        if stream is not None:
            await self._stop_native_task_card_stream(key, stream)

    def _suppressed_ignored(self, chat_id: str, what: str, *, level: int = logging.WARNING) -> bool:
        """True (after logging) when ``chat_id`` is a configured ignored channel."""
        from . import adapter as _adapter

        if self._is_ignored_channel(chat_id):
            _adapter.logger.log(level, "[Slack] Suppressed %s configured ignored channel %s", what, chat_id)
            return True
        return False

    def _outbound_blocked(self, chat_id: str, what: str) -> Optional[SendResult]:
        """Failed SendResult when ``chat_id`` is ignored or the app is not connected, else None."""
        from . import adapter as _adapter

        if self._suppressed_ignored(chat_id, what):
            return _adapter.SendResult(success=False, error="ignored_channel")
        if not self._app:
            return _adapter.SendResult(success=False, error="Not connected")
        return None

    async def _call_with_block_fallback(
        self, client_fn: Callable[[], Any], method: str, kwargs: Dict[str, Any], verb: str) -> Any:
        """``client_fn().<method>(**kwargs)``; on a Block Kit rejection retry once without
        ``blocks`` (an edit sends ``blocks=[]`` so the message drops its stale layout). The client
        is re-resolved for the retry."""
        from . import adapter as _adapter

        try:
            return await getattr(client_fn(), method)(**kwargs)
        except Exception as e:
            if kwargs.get("blocks") and self._is_block_payload_rejection(e):
                retry_kwargs = dict(kwargs)
                if verb == "edit":
                    retry_kwargs["blocks"] = []
                else:
                    retry_kwargs.pop("blocks", None)
                _adapter.logger.info(
                    "[Slack] Block Kit payload rejected; retrying %s without blocks: %s", verb, e)
                return await getattr(client_fn(), method)(**retry_kwargs)
            raise

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a message to a Slack channel or DM."""
        from . import adapter as _adapter

        blocked = self._outbound_blocked(chat_id, "outbound generic send to")
        if blocked:
            return blocked
        chat_id = await self._dm_target(chat_id, metadata)
        thread_ts = None
        try:
            team_id = self._metadata_team_id(metadata)
            slash_ctx = self._pop_slash_context(chat_id, team_id)
            if slash_ctx:
                return await self._send_slash_reply(chat_id, slash_ctx, content, metadata)
            # An active native stream that this content finalizes IS the final
            # message: seal it instead of posting a duplicate.
            stream_result = await self._try_finalize_stream(chat_id, content)
            if stream_result is not None:
                return stream_result
            formatted = self.format_message(content)
            if not formatted or not formatted.strip():
                # Slack returns ``no_text`` for blank posts; still the end of a
                # delivery attempt, so the "is thinking..." status must clear.
                await self._clear_thread_status_quietly(chat_id, metadata)
                # This is still the end of a delivery attempt: if the turn produced no visible text (e.g.
                # "(empty)" final responses are filtered upstream), the assistant thread status must not
                # stay stuck on "is thinking..." (#24117).
                return _adapter.SendResult(success=True)
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            last_result = await self._post_chunks(chat_id, team_id, content, formatted, thread_ts)
            # Clear Slack Assistant status as soon as the final message is posted.
            if thread_ts:
                await self.stop_typing(chat_id, metadata=metadata)
            # Track sent ts (and the thread root) so thread replies get answered
            # without an @mention.
            sent_ts = last_result.get("ts") if last_result else None
            if sent_ts:
                self._bot_message_ts.add(self._workspace_message_marker(team_id, sent_ts))
                if thread_ts:
                    self._bot_message_ts.add(self._workspace_message_marker(team_id, thread_ts))
                self._trim_bot_message_timestamps()
            return _adapter.SendResult(success=True, message_id=sent_ts, raw_response=last_result)
        except Exception as e:  # pragma: no cover - defensive logging
            # Clear the status even when the failure preceded thread_ts resolution:
            # stop_typing falls back to metadata / the uniquely tracked status.
            await self._clear_thread_status_quietly(chat_id, metadata)
            # Clear the assistant status even when the failure happened BEFORE thread_ts was resolved
            # (formatting, slash-context, DM resolution): stop_typing falls back to metadata / the uniquely
            # tracked status for this channel, so a failed turn cannot leave "is thinking..." visible
            # (#24117).
            _adapter.logger.error("[Slack] Send error: %s", e, exc_info=True)
            _retryable = self._is_retryable_upload_error(e)
            return _adapter.SendResult(
                success=False, error=str(e), retryable=_retryable,
                retry_after=self._retry_after_from_exc(e) if _retryable else None)

    async def _post_chunks(
        self, chat_id: str, team_id: str, content: str, formatted: str, thread_ts: Optional[str]
    ) -> Any:
        """``chat.postMessage`` each ``MAX_MESSAGE_LENGTH`` chunk; returns the last response.
        Block Kit only for single-chunk messages (a >39k response is pathological for the 50-block /
        3000-char limits); ``text`` stays the notification/accessibility fallback. With
        ``reply_broadcast`` only the first chunk is also posted to the main channel."""
        from . import adapter as _adapter

        chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
        broadcast = self.config.extra.get("reply_broadcast", False)
        blocks = self._maybe_blocks(content) if len(chunks) == 1 else None
        last_result = None
        for i, chunk in enumerate(chunks):
            kwargs = {
                "channel": chat_id, "text": chunk,
                "mrkdwn": True, **_adapter._slack_unfurl_kwargs(self.config.extra)}
            if blocks and i == 0:
                kwargs["blocks"] = blocks
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
                if broadcast and i == 0:
                    kwargs["reply_broadcast"] = True
            client_fn = lambda: self._get_client(chat_id, team_id=team_id)  # noqa: E731
            last_result = await self._call_with_block_fallback(
                client_fn, "chat_postMessage", kwargs, "send")
        return last_result

    @staticmethod
    def _retry_after_from_exc(e: BaseException) -> Optional[float]:
        """``Retry-After`` header (seconds) from an SDK error response, else None."""
        _resp = getattr(e, "response", None)
        if _resp is None:
            return None
        try:
            _ra = getattr(_resp, "headers", {}).get("Retry-After")
            return float(_ra) if _ra is not None else None
        except (TypeError, ValueError, AttributeError):
            return None

    async def _send_slash_reply(
        self, chat_id: str, slash_ctx: Dict[str, Any], content: str,
        metadata: Optional[Dict[str, Any]]) -> SendResult:
        """Ephemeral slash reply replacing the "Running /cmd…" ack: response_url, then
        chat.postEphemeral, NEVER a public post (a private reply must not leak because a path
        failed). Ephemerals don't auto-clear the Assistant status, so clear it here."""
        from . import adapter as _adapter

        ephemeral_result = await self._send_slash_ephemeral(slash_ctx, content)
        if ephemeral_result.success:
            await self._clear_thread_status_quietly(chat_id, metadata)
            return ephemeral_result
        _adapter.logger.warning(
            "[Slack] response_url slash reply failed (%s); retrying via chat.postEphemeral",
            ephemeral_result.error)
        fallback_result = await self._post_ephemeral_fallback(chat_id, slash_ctx, content)
        if fallback_result.success:
            await self._clear_thread_status_quietly(chat_id, metadata)
            return fallback_result
        # The user still has the ack; the error is returned so the gateway can react.
        _adapter.logger.error(
            "[Slack] Ephemeral slash reply failed on both response_url and chat.postEphemeral "
            "(%s); dropping rather than posting publicly", fallback_result.error)
        return fallback_result

    async def send_private_notice(
        self, chat_id: str, user_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a Slack ephemeral message visible only to one user."""
        from . import adapter as _adapter

        blocked = self._outbound_blocked(chat_id, "outbound generic ephemeral notice to")
        if blocked:
            return blocked
        if not chat_id or not user_id:
            return _adapter.SendResult(success=False, error="chat_id and user_id are required")
        try:
            formatted = self.format_message(content)
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            kwargs = {"channel": chat_id, "user": user_id, "text": formatted, "mrkdwn": True}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            result = await self._client_for(chat_id, metadata).chat_postEphemeral(**kwargs)
            return _adapter.SendResult(
                success=True, message_id=result.get("message_ts") or result.get("ts"),
                raw_response=result)
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[Slack] Ephemeral send error: %s", e, exc_info=True)
            return _adapter.SendResult(success=False, error=str(e))

    async def send_or_update_status(
        self, chat_id: str, status_key: str, content: str, *,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a status message or edit the previous one with the same (channel, thread, key) so
        progress callbacks edit one bubble. If the edit fails (deleted, too old) the cached ts is
        dropped and a fresh message is sent.

        Issue #30045 (Telegram) extended to Slack: progress/status callbacks (context-pressure, compression
        retries, model fallback, lifecycle) used to append a fresh bubble on every call, spamming threads
        during long retry loops. The first call posts and the message ts is remembered; subsequent calls
        with the same (channel, thread, status_key) edit that message in place via ``chat.update``.
        """
        thread_ts = self._resolve_thread_ts(None, metadata) or ""
        key = (str(chat_id), str(thread_ts), str(status_key))
        cached_id = self._status_message_ids.get(key)
        if cached_id is not None:
            result = await self.edit_message(
                chat_id, cached_id, content, finalize=False, metadata=metadata)
            if result.success:
                if result.message_id:
                    self._status_message_ids[key] = str(result.message_id)
                return result
            # Edit failed: drop cached ts, fall through to a fresh send.
            self._status_message_ids.pop(key, None)
        result = await self.send(chat_id, content, metadata=metadata)
        if result.success and result.message_id:
            if len(self._status_message_ids) >= self._STATUS_MESSAGE_IDS_MAX:
                # FIFO trim: drop the oldest half to bound memory.
                for stale in list(self._status_message_ids)[: self._STATUS_MESSAGE_IDS_MAX // 2]:
                    self._status_message_ids.pop(stale, None)
            self._status_message_ids[key] = str(result.message_id)
        return result

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Edit a previously sent Slack message."""
        from . import adapter as _adapter

        blocked = self._outbound_blocked(chat_id, "message edit in")
        if blocked:
            return blocked
        try:
            formatted = self.format_message(content)
            # chat.update has postMessage's ~40k limit but cannot split, so truncate to fit
            # (an oversized payload fails the whole edit with ``msg_too_long``).
            chunks = self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH)
            formatted = chunks[0] if chunks else formatted
            update_kwargs: _adapter.Dict[str, _adapter.Any] = {
                "channel": chat_id, "ts": message_id, "text": formatted}
            # Block Kit only on the FINAL edit: re-deriving a layout on every streaming flush
            # would be wasteful and jittery. ``text`` is the fallback either way.
            if finalize:
                blocks = self._maybe_blocks(content)
                if blocks:
                    update_kwargs["blocks"] = blocks
            await self._call_with_block_fallback(
                lambda: self._client_for(chat_id, metadata), "chat_update", update_kwargs, "edit")
            if finalize:
                await self._clear_thread_status_quietly(chat_id, metadata)
            return _adapter.SendResult(success=True, message_id=message_id)
        except Exception as e:  # pragma: no cover - defensive logging
            if finalize:
                await self._clear_thread_status_quietly(chat_id, metadata)
            if _adapter._is_transient_transport_error(e):
                # chat.update is idempotent: keep the message ID after a transport failure so a
                # later edit can catch up, else every later tool update becomes a new post.
                _adapter.logger.error(
                    "[Slack] transient chat.update failure on message %s in channel %s: %s",
                    message_id, chat_id, e, exc_info=True)
                return _adapter.SendResult(
                    success=False, error=str(e), retryable=True, error_kind="transient")
            _adapter.logger.error(
                "[Slack] Failed to edit message %s in channel %s: %s", message_id, chat_id, e,
                exc_info=True)
            return _adapter.SendResult(success=False, error=str(e))

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a bot message (used to clean up temporary progress bubbles)."""
        from . import adapter as _adapter

        if not self._app:
            return False
        try:
            response = await self._get_client(chat_id).chat_delete(channel=chat_id, ts=message_id)
            if not (hasattr(response, "get") and response.get("ok") is False):
                return True
            _adapter.logger.debug(
                "[Slack] chat.delete returned ok=false for message %s in channel %s: %s",
                message_id, chat_id, response.get("error", "unknown"))
            return False
        except Exception as e:  # pragma: no cover - best-effort cleanup
            _adapter.logger.debug(
                "[Slack] Failed to delete message %s in channel %s: %s", message_id, chat_id, e)
            return False

    def supports_draft_streaming(
        self, chat_type: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Return whether Slack's native stream can preserve configured behavior."""
        from . import adapter as _adapter

        if self._native_stream_unsupported:
            return False
        # chat.*Stream has no unfurl controls; configured unfurl behavior needs
        # the edit-based transport whose chat.postMessage carries them.
        if _adapter._slack_unfurl_kwargs(self.config.extra):
            return False
        return self._app is not None

    def _strip_stream_cursor(self, text: str) -> str:
        """Strip the consumer's trailing cursor glyph from a frame."""
        stripped = text.rstrip()
        glyph = next((g for g in self._STREAM_CURSOR_GLYPHS if stripped.endswith(g)), None)
        return stripped[: -len(glyph)].rstrip() if glyph else text

    async def send_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Stream a frame via Slack's native streaming APIs.
        First frame for a (chat, draft_id) starts the stream; later frames append the delta.
        ``content`` is the full accumulated text (append-only within one text segment)."""
        from . import adapter as _adapter

        if not self._app:
            return _adapter.SendResult(success=False, error="Not connected")
        if self._native_stream_unsupported:
            return _adapter.SendResult(success=False, error="native streaming unsupported")
        text = self._strip_stream_cursor(content)
        client = self._get_client(chat_id)
        stream = self._active_streams.get(chat_id)
        try:
            if stream is not None and stream.get("draft_id") != draft_id:
                # New segment while a prior stream is open: seal the old one so
                # it doesn't hang with a live-typing indicator.
                await self._seal_stream(chat_id, stream)
                stream = None
            if stream is None:
                return await self._start_stream(client, chat_id, draft_id, text, metadata)
            sent = stream.get("sent", "")
            if text == sent:
                return _adapter.SendResult(success=True, message_id=stream["ts"])
            if not text.startswith(sent):
                # Text was rewritten mid-segment: seal the stream, then fail
                # the frame so the consumer falls back to the edit path.
                await self._seal_stream(chat_id, stream)
                self._active_streams.pop(chat_id, None)
                return _adapter.SendResult(success=False, error="stream prefix mismatch")
            delta = text[len(sent) :]
            await client.chat_appendStream(channel=chat_id, ts=stream["ts"], markdown_text=delta)
            stream["sent"] = text
            return _adapter.SendResult(success=True, message_id=stream["ts"])
        except Exception as e:  # pragma: no cover - network/API errors
            self._active_streams.pop(chat_id, None)
            err = str(e)
            # Feature-gate errors: remember unsupported so later responses
            # skip the native attempt instead of erroring each time.
            if any(marker in err for marker in self._NATIVE_STREAM_UNSUPPORTED_MARKERS):
                self._native_stream_unsupported = True
                _adapter.logger.warning(
                    "[Slack] Native streaming unavailable (%s). Falling back to edit-based "
                    "streaming. To enable native streaming, turn on the Agents & AI Apps feature "
                    "for this Slack app (and ensure the assistant:write scope).", err)
            else:
                _adapter.logger.debug("[Slack] Native stream frame failed: %s", err)
            return _adapter.SendResult(success=False, error=err)

    async def _start_stream(
        self, client: Any, chat_id: str, draft_id: int, text: str,
        metadata: Optional[Dict[str, Any]]) -> SendResult:
        """``chat.startStream`` for the first frame and register the stream. Streams must anchor to
        a thread_ts (the gateway sets metadata.thread_id even for top-level messages, so a miss is
        rare). Channels require recipient team/user; harmless for DMs."""
        from . import adapter as _adapter

        thread_ts = self._resolve_thread_ts(None, metadata)
        if not thread_ts:
            return _adapter.SendResult(success=False, error="no thread_ts for native stream")
        start_kwargs: _adapter.Dict[str, _adapter.Any] = {"channel": chat_id, "thread_ts": thread_ts}
        md = metadata or {}
        user_id = md.get("user_id") or md.get("sender_id")
        team_id = self._channel_team.get(chat_id)
        if user_id:
            start_kwargs["recipient_user_id"] = str(user_id)
        if team_id:
            start_kwargs["recipient_team_id"] = str(team_id)
        if text:
            start_kwargs["markdown_text"] = text
        response = await client.chat_startStream(**start_kwargs)
        ts = response.get("ts") if response else None
        if not ts:
            raise RuntimeError("chat.startStream returned no ts")
        self._active_streams[chat_id] = {
            "ts": str(ts), "draft_id": draft_id, "sent": text, "started": _adapter.time.time()}
        self._bot_message_ts.add(str(ts))
        return _adapter.SendResult(success=True, message_id=str(ts))

    async def _seal_stream(
        self, chat_id: str, stream: Dict[str, Any], final_text: Optional[str] = None,
        blocks: Optional[list] = None) -> bool:
        """Best-effort chat.stopStream for an open stream.
        ``final_text`` is the complete final content; only the unsent delta is passed to stopStream
        (append-only API). Returns True on success."""
        from . import adapter as _adapter

        try:
            kwargs: _adapter.Dict[str, _adapter.Any] = {"channel": chat_id, "ts": stream["ts"]}
            if final_text is not None:
                sent = stream.get("sent", "")
                if final_text.startswith(sent) and len(final_text) > len(sent):
                    kwargs["markdown_text"] = final_text[len(sent) :]
            if blocks:
                kwargs["blocks"] = blocks
            await self._get_client(chat_id).chat_stopStream(**kwargs)
            return True
        except Exception as e:  # pragma: no cover - defensive
            _adapter.logger.debug(
                "[Slack] chat.stopStream failed for %s/%s: %s", chat_id, stream.get("ts"), e)
            return False

    async def _try_finalize_stream(self, chat_id: str, content: str) -> Optional[SendResult]:
        """Seal the active native stream if ``content`` is its final text: SendResult when the
        stream IS the final message; None when unrelated (interim commentary), leaving it open."""
        from . import adapter as _adapter

        stream = self._active_streams.get(chat_id)
        if stream is None:
            return None
        sent = stream.get("sent", "")
        text = self._strip_stream_cursor(content)
        # Only claim sends that extend what was streamed; an empty ``sent``
        # prefix would match everything.
        if not sent or not text.startswith(sent):
            return None
        self._active_streams.pop(chat_id, None)
        ts = stream["ts"]
        ok = await self._seal_stream(chat_id, stream, final_text=text)
        if not ok:
            # Stop failed — post normally; the dangling stream times out on Slack's side.
            return None
        # Streams render markdown natively; rich blocks are applied via
        # chat_update on the sealed message (mirrors edit_message finalize).
        blocks = self._maybe_blocks(text)
        if blocks:
            try:
                await self._get_client(chat_id).chat_update(
                    channel=chat_id, ts=ts, text=self.format_message(text), blocks=blocks)
            except Exception as e:
                _adapter.logger.debug(
                    "[Slack] Post-stream Block Kit update failed (markdown fallback stands): %s", e)
        await self.stop_typing(chat_id)
        return _adapter.SendResult(success=True, message_id=ts)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Show a thread status via assistant.threads.setStatus.
        Needs assistant:write or chat:write scope; auto-clears on reply."""
        from . import adapter as _adapter

        if self._suppressed_ignored(chat_id, "typing/status in", level=_adapter.logging.DEBUG):
            return
        if not self._app:
            return
        thread_ts = None
        if metadata:
            # Same synthetic-thread guard as sending: with reply_in_thread=false thread_id is the
            # message's own ts, and setStatus on it would open an assistant thread prematurely.
            thread_ts = self._resolve_thread_ts(
                reply_to=metadata.get("message_id"), metadata=metadata)
        if not thread_ts:
            return  # Can only set status in a thread context
        team_id = self._metadata_team_id(metadata) or self._channel_team.get(chat_id, "")
        status_key = self._workspace_thread_key(team_id, chat_id, str(thread_ts))
        _status_started: _adapter.Optional[float] = None
        if status_key:
            # Keep the first start time across _keep_typing refreshes so long turns show elapsed
            # time; stored in the status entry so it shares eviction/stop_typing cleanup.
            # Heartbeat (#45702): preserve the first refresh's start time across _keep_typing refreshes so a
            # long turn surfaces elapsed time ("still working… (2m03s)") instead of a static "is
            # thinking..." that reads as stuck — which is what provokes mid-turn "you there?" pings. Stored
            # inside the tracked status entry so it shares the existing bounds/eviction and is dropped by
            # stop_typing with the rest of the status state.
            _prev_entry = self._active_status_threads.get(status_key)
            if isinstance(_prev_entry, dict):
                _status_started = _prev_entry.get("started")
            if not isinstance(_status_started, (int, float)):
                _status_started = _adapter.time.monotonic()
            self._active_status_threads[status_key] = {
                "thread_ts": str(thread_ts), "team_id": str(team_id) if team_id else "",
                "started": _status_started}
            # Evict oldest-thread-first (key[2] is the thread ts) so the newest survives.
            self._evict_oldest_by_ts(
                self._active_status_threads, self._ACTIVE_STATUS_THREADS_MAX, lambda k: k[2])
        # May lack assistant:write scope or assistant context; reactions still work.
        _status = getattr(self, "_status_text", {}).get(str(chat_id)) or getattr(
            self.config, "typing_status_text", None)
        _status = _status or self._default_status_text(_status_started)
        await self._set_thread_status(chat_id, team_id, thread_ts, _status, "failed")

    async def _set_thread_status(
        self, chat_id: str, team_id: str, thread_ts: str, status: str, fail_label: str) -> None:
        """``assistant.threads.setStatus`` (empty ``status`` clears); failures are debug-logged."""
        from . import adapter as _adapter

        try:
            await self._get_client(chat_id, team_id=team_id).assistant_threads_setStatus(
                channel_id=chat_id, thread_ts=thread_ts, status=status)
        except Exception as e:
            _adapter.logger.debug("[Slack] assistant.threads.setStatus %s: %s", fail_label, e)

    @staticmethod
    def _default_status_text(started: Optional[float]) -> str:
        """Fallback status label: after 30s show elapsed progress so long turns don't read
        as stuck (live-status phrases and ``typing_status_text`` always win over this)."""
        from . import adapter as _adapter

        elapsed = int(_adapter.time.monotonic() - started) if started is not None else 0
        if elapsed < 30:
            return "is thinking..."
        mins, secs = divmod(elapsed, 60)
        return f"still working… ({f'{mins}m{secs:02d}s' if mins else f'{secs}s'})"

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        """Clear the assistant thread status indicator."""
        from . import adapter as _adapter

        if self._suppressed_ignored(chat_id, "status clear in", level=_adapter.logging.DEBUG):
            self._active_status_threads.pop(chat_id, None)
            return
        if not self._app:
            return
        requested_thread_ts = ""
        if metadata:
            requested_thread_ts = str(metadata.get("thread_id") or metadata.get("thread_ts") or "")
        requested_team_id = self._metadata_team_id(metadata)
        active = None
        ambiguous_tracked = False
        if requested_thread_ts and requested_team_id:
            active_key = self._workspace_thread_key(requested_team_id, chat_id, requested_thread_ts)
            if active_key:
                active = self._active_status_threads.pop(active_key, None)
        else:
            # Slack Connect workspaces can share a channel ID, so a team-less clear
            # only pops a UNIQUE tracked match for this channel (+ thread when given).
            matching_keys = [
                key
                for key in self._active_status_threads
                if key[1] == str(chat_id)
                and (not requested_thread_ts or key[2] == requested_thread_ts)]
            if len(matching_keys) == 1:
                active = self._active_status_threads.pop(matching_keys[0], None)
            ambiguous_tracked = bool(requested_thread_ts) and len(matching_keys) > 1
        active = active or {}
        thread_ts = active.get("thread_ts", "")
        team_id = requested_team_id or active.get("team_id", "")
        if not thread_ts and requested_thread_ts and not ambiguous_tracked:
            # Untracked (restart/eviction) but the caller named the exact thread: clear anyway so
            # a stuck status is always dismissable; skipped when several workspaces track it.
            thread_ts = requested_thread_ts
        if not thread_ts:
            return
        await self._set_thread_status(chat_id, team_id, thread_ts, "", "clear failed")

    def _resolve_thread_ts(
        self, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """thread_ts for an API call: metadata thread_id (parent ts) over reply_to (may be a child
        ts). With ``reply_in_thread: false`` top-level messages get flat replies."""
        # Inbound sets metadata.thread_id to the message's own ts for top-level messages
        # (session keying), so thread_id == reply_to means a synthetic thread → reply flat.
        if not self.config.extra.get("reply_in_thread", True):
            md = metadata or {}
            existing_thread = md.get("thread_id") or md.get("thread_ts")
            if existing_thread and reply_to and existing_thread == reply_to:
                existing_thread = None
            return existing_thread or None
        if metadata:
            if metadata.get("thread_id"):
                return metadata["thread_id"]
            if metadata.get("thread_ts"):
                return metadata["thread_ts"]
        return reply_to

    async def _upload_with_retry(
        self, chat_id: str, file_path: Optional[str], filename: str, caption: Optional[str],
        thread_ts: Optional[str], metadata: Optional[Dict[str, Any]], label: str = "Upload", *,
        content: Optional[bytes] = None, attempts: int = 3) -> SendResult:
        """``files_upload_v2`` of a local path (or in-memory ``content``) with up to
        ``attempts`` tries on transient errors; re-raises otherwise."""
        from . import adapter as _adapter

        source = {"file": file_path} if content is None else {"content": content}
        for attempt in range(attempts):
            try:
                result = await self._client_for(chat_id, metadata).files_upload_v2(
                    channel=chat_id, **source, filename=filename, initial_comment=caption or "",
                    thread_ts=thread_ts)
                self._record_uploaded_file_thread(chat_id, thread_ts, metadata)
                return _adapter.SendResult(success=True, raw_response=result)
            except Exception as exc:
                if not self._is_retryable_upload_error(exc) or attempt >= attempts - 1:
                    raise
                _adapter.logger.debug("[Slack] %s retry %d/2 for %s: %s", label, attempt + 1, file_path, exc)
                await _adapter.asyncio.sleep(1.5 * (attempt + 1))

    async def _upload_file(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Upload a local file to Slack (raises FileNotFoundError when missing)."""
        from . import adapter as _adapter

        blocked = self._outbound_blocked(chat_id, "file upload in")
        if blocked:
            return blocked
        if not _adapter.os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        chat_id = await self._dm_target(chat_id, metadata)
        thread_ts = self._resolve_thread_ts(reply_to, metadata)
        return await self._upload_with_retry(
            chat_id, file_path, _adapter.os.path.basename(file_path), caption, thread_ts, metadata)

    async def _send_local_file(
        self, chat_id: str, file_path: str, caption: Optional[str], reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]], kind: str, filename: str, not_found_error: str,
        failure_notice: str) -> SendResult:
        """Shared body of ``send_video``/``send_document``: upload with retry, notice on failure."""
        from . import adapter as _adapter

        if not self._app:
            return _adapter.SendResult(success=False, error="Not connected")
        if not _adapter.os.path.exists(file_path):
            return _adapter.SendResult(success=False, error=not_found_error)
        chat_id = await self._dm_target(chat_id, metadata)
        try:
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            label = f"{kind.capitalize()} upload"
            return await self._upload_with_retry(
                chat_id, file_path, filename, caption, thread_ts, metadata, label)
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error(
                "[%s] Failed to send %s %s: %s", self.name, kind, file_path, e, exc_info=True)
            if kind == "document":
                check_document_fallback()
            return await self._send_failure_notice(
                chat_id, caption, failure_notice, reply_to, metadata)

    async def send_multiple_images(
        self, chat_id: str, images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> None:
        """Send a batch of images as one message via ``files_upload_v2(file_uploads=...)`` (10 per
        call, Slack cap) instead of N posts; falls back to the base per-image loop on failure."""
        from . import adapter as _adapter

        if self._suppressed_ignored(chat_id, "multi-image upload in"):
            return
        if not self._app:
            return
        if not images:
            return
        chat_id = await self._dm_target(chat_id, metadata)
        try:
            from urllib.parse import unquote as _unquote
            from tools.url_safety import create_ssrf_safe_async_client, is_safe_url as _is_safe_url
        except Exception:
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
            return
        thread_ts = self._resolve_thread_ts(None, metadata)
        CHUNK = 10
        chunks = [images[i : i + CHUNK] for i in range(0, len(images), CHUNK)]
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await _adapter.asyncio.sleep(human_delay)
            try:
                file_uploads, initial_comment_parts = await self._collect_image_uploads(
                    chunk, _unquote, _is_safe_url, create_ssrf_safe_async_client)
                if not file_uploads:
                    continue
                initial_comment = "\n".join(initial_comment_parts) if initial_comment_parts else ""
                _adapter.logger.info(
                    "[Slack] Sending %d image(s) in single files_upload_v2 (chunk %d/%d)",
                    len(file_uploads), chunk_idx + 1, len(chunks))
                await self._client_for(chat_id, metadata).files_upload_v2(
                    channel=chat_id, file_uploads=file_uploads, initial_comment=initial_comment,
                    thread_ts=thread_ts)
                self._record_uploaded_file_thread(chat_id, thread_ts, metadata)
            except Exception as e:
                _adapter.logger.warning(
                    "[Slack] Multi-image files_upload_v2 failed (chunk %d/%d), falling back to per-image: %s",
                    chunk_idx + 1, len(chunks), e, exc_info=True)
                await super().send_multiple_images(
                    chat_id, chunk, metadata, human_delay=human_delay)

    @staticmethod
    async def _collect_image_uploads(
        chunk: List[Tuple[str, str]], unquote_fn, is_safe_url_fn, client_factory
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """``files_upload_v2`` entries for one batch: ``file://`` by path, remote via the SSRF-safe
        client (unsafe/failed skipped). Returns ``(file_uploads, alt_texts)``."""
        from . import adapter as _adapter

        file_uploads: _adapter.List[_adapter.Dict[str, _adapter.Any]] = []
        initial_comment_parts: _adapter.List[str] = []
        async with client_factory(
            timeout=30.0, follow_redirects=True, event_hooks={"response": [_adapter._ssrf_redirect_guard]}
        ) as http_client:
            for image_url, alt_text in chunk:
                if alt_text:
                    initial_comment_parts.append(alt_text)
                if image_url.startswith("file://"):
                    local_path = unquote_fn(image_url[7:])
                    if not _adapter.os.path.exists(local_path):
                        _adapter.logger.warning("[Slack] Skipping missing image: %s", local_path)
                        continue
                    file_uploads.append(
                        {"file": local_path, "filename": _adapter.os.path.basename(local_path)})
                    continue
                if not is_safe_url_fn(image_url):
                    _adapter.logger.warning("[Slack] Blocked unsafe image URL in batch")
                    continue
                try:
                    response = await http_client.get(image_url)
                    response.raise_for_status()
                    ct = response.headers.get("content-type", "")
                    ext = next((e for k, e in _adapter._IMAGE_CT_EXTS if k in ct), "png")
                    file_uploads.append({
                        "content": response.content, "filename": f"image_{len(file_uploads)}.{ext}"
                    })
                except Exception as dl_err:
                    _adapter.logger.warning(
                        "[Slack] Download failed for %s: %s", _adapter.safe_url_for_log(image_url), dl_err)
        return file_uploads, initial_comment_parts

    def _record_uploaded_file_thread(
        self, chat_id: str, thread_ts: Optional[str], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Treat successful file uploads as bot participation in a thread."""
        if not thread_ts:
            return
        team_id = self._metadata_team_id(metadata)
        self._bot_message_ts.add(self._workspace_message_marker(team_id, thread_ts))
        self._trim_bot_message_timestamps()

    def _is_retryable_upload_error(self, exc: Exception) -> bool:
        """Best-effort detection for transient Slack upload failures."""
        from . import adapter as _adapter

        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None:
            return status_code == 429 or status_code >= 500
        body = " ".join(
            str(part)
            for part in (exc, getattr(exc, "message", ""), getattr(exc, "response", None))
            if part).lower()
        if any(m in body for m in _adapter._TRANSIENT_UPLOAD_MARKERS):
            return True
        return self._is_retryable_error(body)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a local image file to Slack by uploading it."""
        from . import adapter as _adapter

        try:
            return await self._upload_file(chat_id, image_path, caption, reply_to, metadata)
        except FileNotFoundError:
            return _adapter.SendResult(success=False, error=f"Image file not found: {image_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error(
                "[%s] Failed to send local Slack image %s: %s", self.name, image_path, e, exc_info=True
            )
            return await self._send_failure_notice(
                chat_id, caption, "⚠️ Couldn't deliver the image attachment.", reply_to, metadata)

    async def _send_failure_notice(
        self, chat_id: str, caption: Optional[str], notice: str, reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]]) -> SendResult:
        """Post ``notice`` (prefixed by the caption) in place of a failed media delivery; the
        host-local path is never echoed into chat."""
        text = f"{caption}\n{notice}" if caption else notice
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image to Slack by uploading the URL as a file."""
        from . import adapter as _adapter

        if not self._app:
            return _adapter.SendResult(success=False, error="Not connected")
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
        if not is_safe_url(image_url):
            _adapter.logger.warning("[Slack] Blocked unsafe image URL (SSRF protection)")
            return await super().send_image(
                chat_id, image_url, caption, reply_to, metadata=metadata)
        try:

            async def _ssrf_redirect_guard(response):
                """Re-check redirect targets so public URLs cannot bounce into private IPs."""
                from tools.url_safety import redirect_target_from_response
                redirect_url = redirect_target_from_response(response)
                if redirect_url and not is_safe_url(redirect_url):
                    raise ValueError("Blocked redirect to private/internal address")

            # Download the image first
            async with create_ssrf_safe_async_client(
                timeout=30.0, follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]}) as client:
                response = await client.get(image_url)
                response.raise_for_status()
            thread_ts = self._resolve_thread_ts(reply_to, metadata)
            chat_id = await self._dm_target(chat_id, metadata)
            return await self._upload_with_retry(
                chat_id, None, "image.png", caption, thread_ts, metadata, content=response.content,
                attempts=1)
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.warning(
                "[Slack] Failed to upload image from URL %s, falling back to text: %s",
                _adapter.safe_url_for_log(image_url), e, exc_info=True)
            # Fall back to sending the URL as text
            text = f"{caption}\n{image_url}" if caption else image_url
            return await self.send(
                chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ) -> SendResult:
        """Send an audio file to Slack."""
        from . import adapter as _adapter

        try:
            return await self._upload_file(chat_id, audio_path, caption, reply_to, metadata)
        except FileNotFoundError:
            return _adapter.SendResult(success=False, error=f"Audio file not found: {audio_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[Slack] Failed to send audio file %s: %s", audio_path, e, exc_info=True)
            return _adapter.SendResult(success=False, error=str(e))

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a video file to Slack."""
        from . import adapter as _adapter

        return await self._send_local_file(
            chat_id, video_path, caption, reply_to, metadata, "video", _adapter.os.path.basename(video_path),
            f"Video file not found: {video_path}", "⚠️ Couldn't deliver the video attachment.")

    @mark_native_document_guard
    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send a document/file attachment to Slack.
        Only ``display_name`` (never the host-local path) goes in the failure notice."""
        from . import adapter as _adapter

        display_name = file_name or _adapter.os.path.basename(file_path)
        return await self._send_local_file(
            chat_id, file_path, caption, reply_to, metadata, "document", display_name,
            f"File not found: {file_path}", f"⚠️ Couldn't deliver the file attachment ({display_name}).",
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Slack channel."""
        from . import adapter as _adapter

        if not self._app:
            return {"name": chat_id, "type": "unknown"}
        try:
            result = await self._get_client(chat_id).conversations_info(channel=chat_id)
            channel = result.get("channel", {})
            is_dm = channel.get("is_im", False)
            return {"name": channel.get("name", chat_id), "type": "dm" if is_dm else "group"}
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[Slack] Failed to fetch chat info for %s: %s", chat_id, e, exc_info=True)
            return {"name": chat_id, "type": "unknown"}

    @classmethod
    def _is_slack_cdn_url(cls, url: str) -> bool:
        """Return True when *url* is an https URL on a Slack CDN host."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        return bool(host) and parsed.scheme == "https" and (
            host in cls._SLACK_CDN_EXACT_HOSTS or host.endswith(cls._SLACK_CDN_HOST_SUFFIXES))

    def _resolve_download_token(self, url: str, team_id: str = "") -> str:
        """Download token: explicit team_id, else the team parsed from ``files-pri/<TEAM>-<FILE>/``
        (events may lack team info; the wrong token yields an HTML login page), else primary."""
        from . import adapter as _adapter

        if team_id and team_id in self._team_clients:
            return self._team_clients[team_id].token
        try:
            m = _adapter.re.search(r"/files-pri/(T[A-Z0-9]+)-", url or "")
            if m and m.group(1) in self._team_clients:
                return self._team_clients[m.group(1)].token
        except Exception:  # pragma: no cover - defensive
            pass
        return self.config.token or ""

    async def _download_slack_file_bytes(
        self, url: str, team_id: str = "", *, html_label: str = "file bytes") -> bytes:
        """Download a Slack file with the bot token (3 attempts on 429/5xx/timeout). URL must pass
        ``is_safe_url`` AND the Slack-CDN allowlist (token exfiltration); redirects are
        re-validated; an HTML body (sign-in page) is rejected so bogus bytes are never cached."""
        from . import adapter as _adapter

        import httpx
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url
        if not is_safe_url(url):
            raise ValueError(
                f"Blocked unsafe Slack file URL (SSRF protection): {_adapter.safe_url_for_log(url)}")
        if not self._is_slack_cdn_url(url):
            raise ValueError(
                "Blocked non-Slack-CDN file URL (token-exfiltration protection): "
                f"{_adapter.safe_url_for_log(url)}")
        bot_token = self._resolve_download_token(url, team_id)
        async with create_ssrf_safe_async_client(
            timeout=30.0, follow_redirects=True, event_hooks={"response": [_adapter._ssrf_redirect_guard]}
        ) as client:
            for attempt in range(3):
                try:
                    response = await client.get(
                        url, headers={"Authorization": f"Bearer {bot_token}"})
                    response.raise_for_status()
                    ct = response.headers.get("content-type", "")
                    if "text/html" in ct:
                        raise ValueError(
                            f"Slack returned HTML instead of {html_label} (content-type: {ct}); "
                            "check bot token scopes and file permissions")
                    return response.content
                except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                        raise
                    if attempt < 2:
                        _adapter.logger.debug(
                            "Slack file download retry %d/2 for %s: %s", attempt + 1, url[:80], exc)
                        await _adapter.asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise

    async def _download_slack_file(
        self, url: str, ext: str, audio: bool = False, team_id: str = "") -> str:
        """Download a Slack image/audio file and cache it; returns the cached path."""
        from gateway.platforms.base import cache_audio_from_bytes_async, cache_image_from_bytes_async
        data = await self._download_slack_file_bytes(url, team_id=team_id, html_label="media")
        return await (cache_audio_from_bytes_async if audio else cache_image_from_bytes_async)(data, ext)
