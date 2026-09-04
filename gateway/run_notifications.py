"""Process/completion/update notifications, media delivery and async-delegation delivery for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional, cast

from gateway.config import Platform, _BUILTIN_PLATFORM_VALUES
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource
from gateway.run_shutdown import _log_suppressed, _notice_target_key, _send_error, _send_failed

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
# Routing fields copied verbatim from a process watcher onto its synthetic completion event.
_WATCHER_ROUTE_FIELDS = ("session_key", "platform", "chat_type", "chat_id", "thread_id", "user_id", "user_name")
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

# Durable async-delegation claim transitions: kind -> (tools.async_delegation function, failure log).
_DURABLE_CLAIM_OPS = {
    "drop": ("drop_completion_delivery", "Could not drop durable completion claim"),
    "release": ("release_completion_delivery", "Could not release durable completion claim"),
}


class GatewayNotificationsMixin:
    """Process/completion/update notifications, media delivery and async-delegation delivery for GatewayRunner."""

    # Coalescing keys: process completions (short-window fan-in) and async delegations (+ parent session).
    _COMPLETION_BATCH_KEY_FIELDS = ("session_key", "platform", "chat_type", "chat_id", "thread_id", "user_id")
    _ASYNC_GROUP_KEY_FIELDS = ("session_key", "parent_session_id", *_COMPLETION_BATCH_KEY_FIELDS[1:])

    @dataclasses.dataclass
    class _UpdatePaths:
        """Marker files ``hermes update --gateway`` and its watcher exchange under HERMES_HOME."""

        pending: Path
        claimed: Path
        output: Path
        exit_code: Path
        prompt: Path
        response: Path

        def any_pending(self) -> bool:
            return self.pending.exists() or self.claimed.exists()

        def unlink_all(self) -> None:
            for p in (self.pending, self.claimed, self.output, self.exit_code, self.prompt, self.response):
                p.unlink(missing_ok=True)

    @dataclasses.dataclass
    class _UpdateTarget:
        """Resolved delivery target for update watcher messages."""

        adapter: Any
        chat_id: Any
        session_key: Optional[str]
        metadata: Any
        platform: Any

        def send_metadata(self):
            from gateway.run import _non_conversational_metadata
            return _non_conversational_metadata(self.metadata, platform=self.platform)

        async def send(self, text: str):
            return await self.adapter.send(self.chat_id, text, metadata=self.send_metadata())

    @dataclasses.dataclass
    class _CompletionClaim:
        """Pre-flight outcome for one completion delivery."""

        delegation_id: str = ""
        claim_id: str = ""
        proceed: bool = True
        early_result: Optional[bool] = None

    async def _deliver_platform_notice(self, source, content: str) -> None:
        """Deliver a setup/operational notice using platform-specific privacy rules."""
        from gateway.run import _is_slack_ignored_channel
        adapter = self._adapter_for_source(source)
        if not adapter:
            return
        config = getattr(self, "config", None)
        chat_id = getattr(source, "chat_id", None)
        if config and getattr(source, "platform", None) == Platform.SLACK and _is_slack_ignored_channel(config, chat_id):
            logger.info("Skipping Slack platform notice for configured ignored channel %s", chat_id)
            return
        notice_delivery = (
            config.get_notice_delivery(source.platform) if config and hasattr(config, "get_notice_delivery")
            else "public"
        )
        metadata = self._thread_metadata_for_source(source)
        if notice_delivery == "private" and getattr(source, "user_id", None):
            with _log_suppressed(
                logging.DEBUG, "[%s] send_private_notice failed, falling back to public",
                getattr(source, "platform", "?"), exc_info=True,
            ):
                result = await adapter.send_private_notice(source.chat_id, source.user_id, content, metadata=metadata)
                if getattr(result, "success", False):
                    return
        await adapter.send(source.chat_id, content, metadata=metadata)

    async def _resolve_compression_lineage_target(
        self, session_db: Any, session_entry: SessionEntry, pinned_session_id: str,
    ) -> Optional[str]:
        """Return the live compression tip of ``pinned_session_id`` if the route owns that lineage, else None."""
        try:
            target_session_id = await session_db.get_compression_tip(pinned_session_id)
        except Exception:
            logger.debug("Async-delegation compression-tip lookup failed for %s", pinned_session_id, exc_info=True)
            target_session_id = None
        if not target_session_id or target_session_id == pinned_session_id:
            logger.warning(
                "Async-delegation completion pinned to compressed session %s "
                "without a continuation; dropping injection.", pinned_session_id,
            )
            return None
        try:
            tip_row = await session_db.get_session(target_session_id)
        except Exception:
            tip_row = None
        if tip_row is None or tip_row.get("ended_at"):
            logger.warning(
                "Async-delegation compression continuation %s is %s; dropping injection.",
                target_session_id, "unknown" if tip_row is None else "ended",
            )
            return None
        route_owns_lineage = session_entry.session_id in {pinned_session_id, target_session_id}
        if not route_owns_lineage:
            # Across several rotations, accept a stale route only when its own tip is the same live target.
            try:
                route_row = await session_db.get_session(session_entry.session_id)
                route_tip = (
                    await session_db.get_compression_tip(session_entry.session_id)
                    if route_row is not None
                    and route_row.get("ended_at")
                    and route_row.get("end_reason") == "compression"
                    else None
                )
            except Exception:
                route_tip = None
            route_owns_lineage = route_tip == target_session_id
        if not route_owns_lineage:
            logger.warning(
                "Async-delegation completion for compression lineage %s -> %s "
                "does not own current route %s; dropping injection.",
                pinned_session_id, target_session_id, session_entry.session_id,
            )
            return None
        return target_session_id

    async def _resolve_async_delegation_session(
        self, session_entry: SessionEntry, pinned_session_id: str,
    ) -> Optional[SessionEntry]:
        """Resolve an async completion to its verified owning gateway session.

        Follow compression-rotation lineage (parent row ended, child continues), but never let a
        late completion override an unrelated /new or restored route. Unknown ownership fails
        closed; the result stays in the delegation records.
        """
        from gateway.run import _USER_BOUNDARY_END_REASONS
        session_db = cast(Any, self._session_db)
        if session_db is None:
            logger.warning(
                "Async-delegation completion has no session database; "
                "dropping injection (#55578 fail-closed)."
            )
            return None
        pinned_row = None
        try:
            pinned_row = await session_db.get_session(pinned_session_id)
        except Exception:
            logger.debug("Async-delegation parent lookup failed for %s", pinned_session_id, exc_info=True)
        if pinned_row is None:
            logger.warning(
                "Async-delegation completion has unknown spawning session %s; "
                "dropping injection (#55578 fail-closed).", pinned_session_id,
            )
            return None
        target_session_id = pinned_session_id
        follows_compression = False
        if pinned_row.get("ended_at"):
            _end_reason = str(pinned_row.get("end_reason") or "")
            if _end_reason in _USER_BOUNDARY_END_REASONS:
                logger.warning(
                    "Async-delegation completion pinned to user-closed session %s "
                    "(end_reason=%r); dropping injection instead of resurrecting it "
                    "(#55578 fail-closed).", pinned_session_id, _end_reason,
                )
                return None
            if _end_reason != "compression":
                # Idle/timeout end (scale-to-zero norm): the chat route is still valid, so deliver to its
                # current session rather than drop (the row would be acked then silently lost).
                logger.info(
                    "Async-delegation completion pinned to %s-ended session %s; "
                    "retargeting to the chat's current session %s.",
                    _end_reason or "idle", pinned_session_id, session_entry.session_id,
                )
                return session_entry
            follows_compression = True
            target_session_id = await self._resolve_compression_lineage_target(
                session_db, session_entry, pinned_session_id,
            )
            if target_session_id is None:
                return None
        if target_session_id == session_entry.session_id:
            return session_entry
        prior_session_id = session_entry.session_id
        if follows_compression:
            switched = await self.async_session_store.advance_compression_session(
                session_entry.session_key, prior_session_id, target_session_id,
            )
        else:
            switched = await self.async_session_store.switch_session(session_entry.session_key, target_session_id)
        if switched is None:
            logger.warning(
                "Async-delegation completion could not bind routing key %s to "
                "owning session %s; dropping injection.", session_entry.session_key, target_session_id,
            )
            return None
        logger.info(
            "Pinned async-delegation completion to owning session %s (was %s) for routing key %s (#57498)",
            target_session_id, prior_session_id, session_entry.session_key,
        )
        return switched

    async def _deliver_media_from_response(
        self, response: str, event: MessageEvent, adapter, thread_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Deliver explicit MEDIA: tags from an already-streamed response (text already delivered).
        EXPLICIT-ONLY, unlike the non-streaming path in ``gateway/platforms/base.py``: a bare local
        path in a streamed reply is shown text or stale inspected content, and promoting it sent
        files the model never asked for. MEDIA tags are NOT deduped against prior turns (a final-reply
        directive is a deliberate attach); stale auto-appended tags are deduped upstream.

        Only ``MEDIA:`` directives — the explicit attachment contract — trigger post-stream uploads. See
        #20834.
        """
        from urllib.parse import quote as _quote
        with _log_suppressed(logging.WARNING, "Post-stream media extraction failed: %s"):
            # Capture [[as_document]] before extract_media strips it: images then go via send_document.
            force_document_attachments = "[[as_document]]" in response
            from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio
            media_files, cleaned = adapter.extract_media(response)
            media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
            # Strip image URLs (parity with the non-streaming chain); no extract_local_files here.
            # Do NOT deduplicate explicit MEDIA tags against prior turns here (#73771). This rescan is
            # already EXPLICIT-ONLY (see docstring): a MEDIA: directive in the final streamed reply is the
            # model deliberately attaching a file — including a user-requested resend. Stale auto-appended
            # tags are deduped upstream in _collect_auto_append_media_tags with history_media_paths. Mirrors
            # the same filter removal on the non-streaming path in gateway/platforms/base.py. Bare local
            # paths in an already-streamed reply are text the user has seen (or stale inspected content),
            # not an attachment request.
            adapter.extract_images(cleaned)
            _thread_meta = (
                dict(thread_metadata)
                if thread_metadata is not None
                else self._thread_metadata_for_source(event.source, self._reply_anchor_for_event(event))
            )
            chat_id = event.source.chat_id
            # Images go out as one batch (e.g. Signal's multi-attachment RPC) unless [[as_document]].
            def _is_photo(media_path: str, is_voice: bool) -> bool:
                ext = Path(media_path).suffix.lower()
                return ext in _IMAGE_EXTS and not is_voice and not force_document_attachments

            image_paths = [p for p, v in media_files if _is_photo(p, v)]
            non_image_media = [(p, v) for p, v in media_files if not _is_photo(p, v)]
            if image_paths:
                try:
                    images = [(f"file://{_quote(p)}", "") for p in image_paths]
                    await adapter.send_multiple_images(chat_id=chat_id, images=images, metadata=_thread_meta)
                except Exception as e:
                    logger.warning("[%s] Post-stream image batch delivery failed: %s", adapter.name, e)
            for media_path, is_voice in non_image_media:
                try:
                    ext = Path(media_path).suffix.lower()
                    if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                        await adapter.send_voice(
                            chat_id=chat_id, audio_path=media_path, metadata=_thread_meta, is_voice=is_voice,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=_thread_meta)
                    else:
                        await adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=_thread_meta)
                except Exception as e:
                    logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)


    async def _deliver_queued_first_response(
        self, response: str, source: SessionSource, adapter,
        metadata: Optional[Dict[str, Any]] = None, event_message_id: Optional[str] = None,
        text_already_delivered: bool = False, deliver_media: bool = True, stream_consumer=None,
    ) -> None:
        """Deliver a queued response using the normal text+attachment split."""
        from gateway.run import _strip_response_attachments_for_direct_send
        if not text_already_delivered:
            text_content = _strip_response_attachments_for_direct_send(response, adapter)
            if text_content:
                # Reconcile-by-edit first: a stream-sealed message already carries most of the answer;
                # a plain send here would duplicate it.
                _reconciled = False
                _sc_msg_id = getattr(stream_consumer, "message_id", None)
                if (
                    _sc_msg_id
                    and _sc_msg_id != "__no_edit__"
                    and not getattr(stream_consumer, "_turn_split_delivery", False)
                ):
                    try:
                        _edit_res = await adapter.edit_message(
                            chat_id=source.chat_id, message_id=_sc_msg_id, content=text_content, finalize=True,
                        )
                        if getattr(_edit_res, "success", False):
                            _reconciled = True
                            logger.info(
                                "Queued-lane final reconciled by editing message %s in place (no duplicate send).",
                                _sc_msg_id,
                            )
                    except Exception as _qe:
                        logger.debug("Queued-lane reconcile edit failed (%s); falling back to send.", _qe)
                if not _reconciled:
                    await adapter.send(source.chat_id, text_content, metadata=metadata)
        # Failed turns deliver their (normalized failure) text but must not upload attachments as if
        # they succeeded — mirrors the ``not agent_result.get("failed")`` completed-turn guard.
        if not deliver_media:
            return
        await self._deliver_media_from_response(
            response, MessageEvent(text="", source=source, message_id=event_message_id), adapter,
            thread_metadata=metadata,
        )

    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return
        try:
            self._update_notification_task = asyncio.create_task(self._watch_update_progress())
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    @classmethod
    def _update_paths(cls) -> "GatewayNotificationsMixin._UpdatePaths":
        from gateway.run import _hermes_home
        return cls._UpdatePaths(
            pending=_hermes_home / ".update_pending.json",
            claimed=_hermes_home / ".update_pending.claimed.json", output=_hermes_home / ".update_output.txt",
            exit_code=_hermes_home / ".update_exit_code",
            prompt=_hermes_home / ".update_prompt.json", response=_hermes_home / ".update_response",
        )

    def _resolve_update_target(self, paths: "_UpdatePaths") -> Optional["_UpdateTarget"]:
        """Resolve adapter/chat/session for update watcher messages from the pending marker."""
        for path in (paths.claimed, paths.pending):
            if not path.exists():
                continue
            with suppress(Exception):
                pending = json.loads(path.read_text(encoding="utf-8"))
                platform_str = pending.get("platform")
                chat_id = pending.get("chat_id")
                session_key = pending.get("session_key")
                if not (platform_str and chat_id):
                    continue  # BASE: an incomplete marker falls through to the next path, not "unresolved"
                platform = Platform(platform_str)
                adapter = self.adapters.get(platform)
                if not adapter:
                    return None
                metadata = self._pending_marker_metadata(platform, chat_id, pending, adapter)
                # Fallback session key if not stored (old pending files)
                return self._UpdateTarget(
                    adapter, chat_id, session_key or f"{platform_str}:{chat_id}", metadata, platform,
                )
        return None

    def _pending_marker_metadata(self, platform, chat_id, data: dict, adapter):
        """Thread metadata for a persisted update/restart marker (thread_id/chat_type/message_id keys)."""
        return self._thread_metadata_for_target(
            platform, chat_id, data.get("thread_id"), chat_type=data.get("chat_type"),
            reply_to_message_id=data.get("message_id"), adapter=adapter,
        )

    async def _watch_update_completion_only(self, paths: "_UpdatePaths", deadline: float, poll_interval: float) -> None:
        """Fallback when no adapter/chat can be resolved: wait for the exit code, then notify."""
        logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
        # Poll until _send_update_notification delivers (it returns False while the platform reconnects).
        loop = asyncio.get_running_loop()
        while paths.any_pending() and loop.time() < deadline:
            if paths.exit_code.exists() and await self._send_update_notification():
                return
            await asyncio.sleep(poll_interval)
        if paths.any_pending() and not paths.exit_code.exists():
            paths.exit_code.write_text("124", encoding="utf-8")
            await self._send_update_notification()

    @staticmethod
    def _update_exit_code(paths: "_UpdatePaths") -> int:
        return int(paths.exit_code.read_text(encoding="utf-8").strip() or "1")

    @staticmethod
    def _read_update_output_since(path: Path, offset: int) -> tuple[str, int]:
        """Read update output defensively; logs may contain invalid UTF-8."""
        try:
            data = path.read_bytes()
        except OSError:
            return "", offset
        if len(data) <= offset:
            return "", len(data)
        return data[offset:].decode("utf-8", errors="replace"), len(data)

    async def _send_update_output(self, target: "_UpdateTarget", text: str) -> None:
        """Send buffered update output as fenced chunks that fit message limits (Telegram: 4096)."""
        from tools.ansi_strip import strip_ansi
        clean = strip_ansi(text).strip()
        if not clean:
            return
        max_chunk = 3500
        for i in range(0, len(clean), max_chunk):
            with _log_suppressed(logging.DEBUG, "Update stream send failed: %s"):
                await target.send(f"```\n{clean[i:i + max_chunk]}\n```")

    async def _forward_update_prompt(self, target: "_UpdateTarget", prompt_text: str, default: str) -> None:
        """Forward an update prompt: platform-native buttons first (Discord, Telegram), else text."""
        sent_buttons = False
        adapter = target.adapter
        if getattr(type(adapter), "send_update_prompt", None) is not None:
            with _log_suppressed(logging.DEBUG, "Button-based update prompt failed: %s"):
                await adapter.send_update_prompt(
                    chat_id=target.chat_id, prompt=prompt_text, default=default,
                    session_key=target.session_key, metadata=target.send_metadata(),
                )
                sent_buttons = True
        if not sent_buttons:
            default_hint = f" (default: {default})" if default else ""
            _p = getattr(adapter, "typed_command_prefix", "/")
            await target.send(
                f"⚕ **Update needs your input:**\n\n{prompt_text}{default_hint}\n\n"
                f"Reply `{_p}approve` (yes) or `{_p}deny` (no), or type your answer directly."
            )
        # Keep the prompt marker on disk until answered so a restarted watcher can re-forward it.
        self._session_state(target.session_key).persistent.update_prompt_pending = True
        logger.info("Forwarded update prompt to %s: %s", target.session_key, prompt_text[:80])

    def _clear_update_markers(self, paths: "_UpdatePaths", session_key: Optional[str]) -> None:
        paths.unlink_all()
        state = self._peek_session_state(session_key)
        if state is not None:
            state.persistent.update_prompt_pending = False

    async def _watch_update_progress(
        self, poll_interval: float = 2.0, stream_interval: float = 4.0, timeout: float = 1800.0
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the user periodically;
        detects ``.update_prompt.json`` (written when the update process needs input) and forwards it.
        """
        paths = self._update_paths()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        target = self._resolve_update_target(paths)
        if target is None:
            await self._watch_update_completion_only(paths, deadline, poll_interval)
            return
        session_key = target.session_key
        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            nonlocal buffer, last_stream_time
            text, buffer = buffer, ""
            if text.strip():
                last_stream_time = loop.time()
                await self._send_update_output(target, text)

        def _read_new_output() -> None:
            nonlocal buffer, bytes_sent
            if paths.output.exists():
                with suppress(OSError):
                    chunk, bytes_sent = self._read_update_output_since(paths.output, bytes_sent)
                    buffer += chunk

        while loop.time() < deadline:
            if paths.exit_code.exists():
                _read_new_output()
                await _flush_buffer()
                with _log_suppressed(logging.WARNING, "Update final notification failed: %s"):
                    exit_code = self._update_exit_code(paths)
                    await target.send(
                        "✅ Hermes update finished." if exit_code == 0
                        else "❌ Hermes update failed (exit code {}).".format(exit_code)
                    )
                    logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                self._clear_update_markers(paths, session_key)
                return
            _read_new_output()
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()
            # Forward a prompt only when none is pending, else every poll re-forwards the same prompt.
            _pending_state = self._peek_session_state(session_key) if session_key else None
            if paths.prompt.exists() and session_key and not getattr(
                getattr(_pending_state, "persistent", None), "update_prompt_pending", False
            ):
                try:
                    prompt_data = json.loads(paths.prompt.read_text(encoding="utf-8"))
                    prompt_text = prompt_data.get("prompt", "")
                    if prompt_text:
                        await _flush_buffer()  # user sees context before the prompt
                        await self._forward_update_prompt(target, prompt_text, prompt_data.get("default", ""))
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)
            await asyncio.sleep(poll_interval)
        if not paths.exit_code.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            paths.exit_code.write_text("124", encoding="utf-8")
            await _flush_buffer()
            with suppress(Exception):
                await target.send("❌ Hermes update timed out after 30 minutes.")
            self._clear_update_markers(paths, session_key)

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        False while the update is still running (caller may retry); True after a definitive send/skip.
        """
        from gateway.run import _non_conversational_metadata
        paths = self._update_paths()
        if not paths.any_pending():
            return False
        cleanup = True
        active_pending_path = paths.claimed

        def _defer(reason: str, *args) -> bool:
            nonlocal cleanup, active_pending_path
            logger.info(reason, *args)
            cleanup = False
            active_pending_path = paths.pending
            paths.claimed.replace(paths.pending)
            return False

        try:
            if paths.pending.exists():
                try:
                    paths.pending.replace(paths.claimed)
                except FileNotFoundError:
                    if not paths.claimed.exists():
                        return True
            elif not paths.claimed.exists():
                return True
            pending = json.loads(paths.claimed.read_text(encoding="utf-8"))
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")
            if not paths.exit_code.exists():
                return _defer("Update notification deferred: update still running")
            exit_code = self._update_exit_code(paths)
            output = paths.output.read_bytes().decode("utf-8", errors="replace") if paths.output.exists() else ""
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)
            if chat_id and not adapter:
                # Target platform not reconnected yet (common right after the update's restart): keep the
                # markers for a later retry instead of silently losing the notification.
                return _defer("Update notification deferred: %s adapter not connected yet", platform_str)
            if chat_id:
                metadata = self._pending_marker_metadata(platform, chat_id, pending, adapter)
                from tools.ansi_strip import strip_ansi
                output = strip_ansi(output).strip()
                if output:
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    status = "✅ Hermes update finished." if exit_code == 0 else "❌ Hermes update failed."
                    msg = f"{status}\n\n```\n{output}\n```"
                else:
                    msg = (
                        "✅ Hermes update finished successfully." if exit_code == 0 else
                        "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
                    )
                await adapter.send(chat_id, msg, metadata=_non_conversational_metadata(metadata, platform=platform))
                logger.info("Sent post-update notification to %s:%s (exit=%s)", platform_str, chat_id, exit_code)
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                for p in (active_pending_path, paths.claimed, paths.output, paths.exit_code):
                    p.unlink(missing_ok=True)
        return True

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back."""
        from gateway.delivery import resolve_delivery_transport
        from gateway.run import _hermes_home, _non_conversational_metadata
        notify_path = _hermes_home / ".restart_notify.json"
        if not notify_path.exists():
            return None
        try:
            data = json.loads(notify_path.read_text(encoding="utf-8"))
            platform_str = data.get("platform")
            chat_id = data.get("chat_id")
            thread_id = data.get("thread_id")
            if not platform_str or not chat_id:
                return None
            platform = Platform(platform_str)
            transport = resolve_delivery_transport(platform, self.config, self.adapters)
            if transport is None:
                logger.debug("Restart notification skipped: no live transport for %s", platform_str)
                return None
            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Restart notification suppressed: %s has gateway_restart_notification=false", platform_str
                )
                return None
            metadata = self._pending_marker_metadata(platform, chat_id, data, transport.adapter)
            if data.get("delivered_via_upstream_relay") is True:
                metadata = dict(metadata or {})
                for field in ("user_id", "scope_id"):
                    if data.get(field):
                        metadata[field] = str(data[field])
            result = await transport.send(
                platform, str(chat_id), "♻ Gateway restarted successfully. Your session continues.",
                metadata=_non_conversational_metadata(metadata, platform=platform),
            )
            # adapter.send() catches provider errors (e.g. "Chat not found") and returns
            # SendResult(success=False) rather than raising, so inspect the result before claiming success.
            if _send_failed(result):
                logger.warning(
                    "Restart notification to %s:%s was not delivered: %s", platform_str, chat_id, _send_error(result),
                )
                return None
            logger.info("Sent restart notification to %s:%s", platform_str, chat_id)
            return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
        except Exception as e:
            logger.warning("Restart notification failed: %s", e)
            return None
        finally:
            notify_path.unlink(missing_ok=True)

    def _home_channel_transports(self):
        """Yield ``(platform, platform_cfg, home, transport)`` for every home channel with a live transport."""
        from gateway.delivery import resolve_delivery_transport
        for platform, platform_cfg in self.config.platforms.items():
            home = platform_cfg.home_channel
            if not home or not home.chat_id:
                continue
            transport = resolve_delivery_transport(platform, self.config, self.adapters)
            if transport is None:
                continue
            yield platform, platform_cfg, home, transport

    async def _send_home_channel_message(self, platform, home, transport, message: str, failure_fmt: str) -> bool:
        """Best-effort send to one home channel; True on success, failures logged with ``failure_fmt``."""
        from gateway.run import _non_conversational_metadata
        try:
            metadata = self._thread_metadata_for_target(platform, home.chat_id, home.thread_id, adapter=transport.adapter)
            if transport.is_relay:
                metadata = dict(metadata or {})
                if home.user_id:
                    metadata["user_id"] = home.user_id
                if home.scope_id:
                    metadata["scope_id"] = home.scope_id
            send_metadata = _non_conversational_metadata(metadata, platform=platform)
            if send_metadata is not None or transport.is_relay:
                result = await transport.send(platform, str(home.chat_id), message, metadata=send_metadata)
            else:
                result = await transport.adapter.send(str(home.chat_id), message)
            if _send_failed(result):
                logger.warning(failure_fmt, platform.value, home.chat_id, _send_error(result))
                return False
            return True
        except Exception as exc:
            logger.warning(failure_fmt, platform.value, home.chat_id, exc)
            return False

    async def _send_home_channel_startup_notifications(
        self, *, skip_targets: Optional[set[tuple[str, str, Optional[str]]]] = None
    ) -> set[tuple[str, str, Optional[str]]]:
        """Notify configured home channels that the gateway is back online.

        Best-effort, once per connected platform home channel. ``skip_targets`` lets startup avoid
        duplicate messages when a more specific restart notification is queued for the same chat.
        """
        delivered: set[tuple[str, str, Optional[str]]] = set()
        skipped = skip_targets or set()
        message = "♻️ Gateway online — Hermes is back and ready."
        for platform, platform_cfg, home, transport in self._home_channel_transports():
            if not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Home-channel startup notification suppressed: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue
            target = _notice_target_key(platform.value, home.chat_id, home.thread_id)
            if target in skipped or target in delivered:
                continue
            if await self._send_home_channel_message(
                platform, home, transport, message, "Home-channel startup notification failed for %s:%s: %s",
            ):
                delivered.add(target)
                logger.info("Sent home-channel startup notification to %s:%s", platform.value, home.chat_id)
        return delivered

    async def _send_session_db_warning_notifications(self) -> None:
        """Broadcast a state.db failure warning to all home channels.

        When SessionDB init fails at gateway startup, messages may flow but nothing is persisted
        — /resume, /history, and session_search all silently break. Best-effort: failures are
        logged, not raised.

        See #88235.
        """
        error = getattr(self, "_session_db_init_error", None)
        if not error:
            return
        from hermes_state import _default_db_path, classify_persistence_error, format_session_db_unavailable
        if classify_persistence_error(error) == "corrupt":
            # Copy-pasteable, so name the real store (profiles / HERMES_HOME do not live under ~/.hermes).
            db_path = _default_db_path()
            message = (
                "⚠️ Session database corruption detected. Messages may not be "
                "persisted. Recovery options:\n"
                "1. Run `hermes doctor --fix`\n"
                "2. Stop the gateway, then recover with:\n"
                f"   hermes sessions recover --source {db_path} "
                "--inspect-only\n"
                "   (if it reports recoverable) hermes sessions recover "
                f"--source {db_path} --output recovered-state.db\n"
                "   — recovery snapshots the damaged file first; do NOT run "
                "`sqlite3 ... \".recover\"` against the live state.db, a "
                "vulnerable sqlite3 CLI can corrupt it further\n"
                "3. Restore from a backup in ~/.hermes/backups/\n"
                "Run `hermes doctor` for sanitized diagnostics."
            )
        else:
            message = (
                f"⚠️ Session database unavailable — messages may not be persisted. "
                f"{format_session_db_unavailable()}\nRun `hermes doctor` for diagnostics."
            )
        logger.warning("Broadcasting state.db failure warning to home channels: %s", error)
        for platform, _platform_cfg, home, transport in self._home_channel_transports():
            await self._send_home_channel_message(
                platform, home, transport, message, "state.db warning notification failed for %s:%s: %s",
            )

    def _build_process_event_source(self, evt: dict):
        """Resolve the canonical source for a synthetic background-process event.

        Prefer the persisted session-store origin; the active foreground event causes cross-topic bleed.
        """
        from gateway.run import _parse_session_key
        session_key = str(evt.get("session_key") or "").strip()
        derived = {}
        if session_key:
            try:
                self.session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                if entry and getattr(entry, "origin", None):
                    return entry.origin
            except Exception as exc:
                logger.debug("Synthetic process-event session-store lookup failed for %s: %s", session_key, exc)
            cached_source = self._get_cached_session_source(session_key)
            if cached_source is not None:
                return cached_source
            derived = _parse_session_key(session_key) or {}
        platform_name = str(evt.get("platform") or derived.get("platform") or "").strip().lower()
        chat_type = str(evt.get("chat_type") or derived.get("chat_type") or "").strip().lower()
        chat_id = str(evt.get("chat_id") or derived.get("chat_id") or "").strip()
        if not platform_name or not chat_type or not chat_id:
            logger.warning(
                "Synthetic event source unresolvable: "
                "session_key=%r platform=%r chat_type=%r chat_id=%r evt_type=%s",
                session_key, platform_name, chat_type, chat_id, evt.get("type", "?"),
            )
            return None
        try:
            platform = Platform(platform_name)
            # Reject dynamic pseudo-members: plugin platforms must be registered.
            if platform.value not in _BUILTIN_PLATFORM_VALUES:
                try:
                    from gateway.platform_registry import platform_registry
                    if not platform_registry.is_registered(platform.value):
                        raise ValueError(platform_name)
                except Exception:
                    raise ValueError(platform_name)
        except Exception:
            logger.warning("Synthetic process event has invalid platform metadata: %r", platform_name)
            return None

        def _opt(field: str) -> Optional[str]:
            return str(evt.get(field) or "").strip() or None

        scope_id = _opt("scope_id")
        if scope_id is None and chat_type not in ("dm", "thread"):
            # Reconstructed scoped-chat source without scope_id: a relay connector's tenant guard may
            # decline the reply. Warn, don't fail (native adapters need no scope_id).
            logger.warning(
                "Synthetic event source for %s chat=%s (%s) reconstructed "
                "without scope_id; scoped relay egress may be declined by "
                "the connector's tenant guard (user_id fallback only).", platform_name, chat_id, chat_type,
            )
        return SessionSource(
            platform=platform, chat_id=chat_id, chat_type=chat_type, thread_id=_opt("thread_id"),
            user_id=_opt("user_id"), user_name=_opt("user_name"), scope_id=scope_id,
        )

    async def _drain_watch_notifications(self, completion_queue) -> None:
        """Consume queued watch events and inject them when notifications are enabled.

        The queue is ALWAYS drained (so watch events don't rot or requeue-spin) but injection is
        skipped entirely when ``display.background_process_notifications`` is ``off``.

        See #9290.
        """
        from gateway.run import _drain_gateway_watch_events, _format_gateway_process_notification
        watch_events = _drain_gateway_watch_events(completion_queue)
        if self._load_background_notifications_mode() == "off":
            return
        for evt in watch_events:
            synth_text = _format_gateway_process_notification(evt)
            if not synth_text:
                continue
            with _log_suppressed(logging.ERROR, "Watch notification injection error: %s"):
                await self._inject_watch_notification(synth_text, evt)

    def _adapter_by_platform_value(self, platform_name: str):
        """Literal ``p.value == platform_name`` scan over connected adapters (native adapters only)."""
        for p, a in self.adapters.items():
            if p.value == platform_name:
                return a
        return None

    async def _self_post_api_server(self, adapter, synth_text: str, raw_sid: str, evt: dict) -> bool:
        """Deliver to a non-push (api_server) session by raw session id.

        Async-delegation completions are persisted as a durable delivery row — after the parent
        turn's event.complete the CLIENT owns the next turn on this stateless surface, so never
        self-post them as a new role=user prompt. Other watch events wake the session via self-post.
        """
        from gateway.wake import deliver_wake, persist_delegation_delivery
        if evt.get("type") == "async_delegation":
            info = "Async delegation completion — persisting delivery row for api_server session %s (no wake turn)"
            fail = "Async delegation delivery persist failed for session %s: %s"
            deliver = lambda: persist_delegation_delivery(adapter, text=synth_text, session_id=raw_sid, evt=evt)  # noqa: E731
        else:
            info = "Watch pattern notification — waking api_server session %s via self-post"
            fail = "Watch notification self-post wake failed for session %s: %s"
            deliver = lambda: deliver_wake(adapter, text=synth_text, session_id=raw_sid)  # noqa: E731
        try:
            logger.info(info, raw_sid)
            await deliver()
            return True
        except Exception as e:
            logger.warning(fail, raw_sid, e)
            return False

    def _resolve_injection_adapter(self, platform_name: str):
        """Adapter for a synthetic-event platform: alias-aware transport resolver first (one
        Platform.RELAY adapter fronts N logical platforms; native wins), literal ``p.value`` scan as
        fallback for minimal runner stubs / exotic platform strings when the resolver can't run."""
        from gateway.delivery import resolve_delivery_transport
        try:
            _transport = resolve_delivery_transport(Platform(platform_name), self.config, self.adapters)
        except Exception:
            _transport = None
        if _transport is not None:
            return _transport.adapter
        return self._adapter_by_platform_value(platform_name)

    async def _inject_watch_notification(self, synth_text: str, evt: dict) -> Optional[bool]:
        """Inject a watch/completion notification as a synthetic message event.

        Routing comes from the queued event, never the active foreground message. Returns
        ``True`` on adapter acceptance, ``False`` on retryable adapter failure, ``None`` with no
        gateway route. Not transactional: a crash after acceptance can replay (at-least-once).
        """
        from gateway.run import _parse_session_key
        from gateway.wake import adapter_supports_push
        source = await asyncio.to_thread(self._build_process_event_source, evt)
        if not source:
            # API-server sessions bind the RAW X-Hermes-Session-Id key, not a structured ``agent:...`` key.
            raw_sid = str(evt.get("origin_session_id") or "").strip()
            _sk = str(evt.get("session_key") or "").strip()
            if not raw_sid and _sk and _parse_session_key(_sk) is None:
                raw_sid = _sk
            if raw_sid:
                adapter = self.adapters.get(Platform.API_SERVER)
                if adapter is not None and not adapter_supports_push(adapter):
                    return await self._self_post_api_server(adapter, synth_text, raw_sid, evt)
                logger.warning(
                    "Dropping watch notification for raw session %s: no api_server adapter to self-post through",
                    raw_sid,
                )
                return None
            logger.warning(
                "Dropping watch notification with no routing metadata for process %s",
                evt.get("session_id", "unknown"),
            )
            return None
        platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        adapter = self._resolve_injection_adapter(platform_name)
        if not adapter:
            return None
        if not adapter_supports_push(adapter):
            # Non-push adapter (api_server): its chat_id IS the raw session id, so handle_message would
            # key the wake under a build_session_key() that never matches — self-post instead.
            raw_sid = str(evt.get("origin_session_id") or "").strip() or str(source.chat_id or "")
            return await self._self_post_api_server(adapter, synth_text, raw_sid, evt)
        try:
            metadata = {}
            parent_session_id = str(evt.get("parent_session_id") or "").strip()
            if parent_session_id:
                metadata["gateway_session_id"] = parent_session_id
            synth_event = MessageEvent(
                text=synth_text, message_type=MessageType.TEXT, source=source, internal=True,
                message_id=str(evt.get("message_id") or "").strip() or None, metadata=metadata,
            )
            logger.info(
                "Watch pattern notification — injecting for %s chat=%s thread=%s",
                platform_name, source.chat_id, source.thread_id,
            )
            # Relay egress priming: post-restart routing caches are cold (they warm only on inbound), so
            # replies would egress without tenant discriminators and be declined by the connector.
            _prime = getattr(adapter, "prime_routing_cache", None)
            if callable(_prime):
                _prime(synth_event)
            await adapter.handle_message(synth_event)
            return True
        except Exception as e:
            logger.error("Watch notification injection error: %s", e)
            return False

    @staticmethod
    def _completion_delivery_identity(evt: dict) -> Optional[tuple[str, str, object]]:
        """Return a producer-stable identity when one is available.

        Delegation UUIDs identify one producer completion. Process session IDs include the
        persisted spawn epoch so a reused ID is a distinct incarnation; legacy events without
        ``started_at`` are delivered undeduplicated rather than risk suppressing a real completion.
        """
        evt_type = str(evt.get("type") or "")
        if evt_type == "async_delegation":
            producer_id = str(evt.get("delegation_id") or "")
            return (evt_type, producer_id, "") if producer_id else None
        if evt_type == "completion":
            producer_id = str(evt.get("session_id") or "")
            started_at = evt.get("started_at")
            if producer_id and started_at is not None:
                return (evt_type, producer_id, started_at)
        return None

    def _mark_completions_delivered_locked(self, identities) -> None:
        """Move identities inflight -> delivered and trim retention. Caller holds ``_completion_delivery_lock``."""
        for identity in identities:
            self._completion_deliveries_inflight.discard(identity)
            self._completion_deliveries_delivered[identity] = None
        while len(self._completion_deliveries_delivered) > self._completion_delivery_retention:
            self._completion_deliveries_delivered.popitem(last=False)

    def _completion_identity_seen(self, identity, *, claim: bool = False) -> bool:
        """True when ``identity`` is inflight or already delivered this gateway lifecycle.

        With ``claim`` an unseen identity is atomically marked inflight (same lock hold).
        """
        with self._completion_delivery_lock:
            seen = (
                identity in self._completion_deliveries_inflight
                or identity in self._completion_deliveries_delivered
            )
            if claim and not seen:
                self._completion_deliveries_inflight.add(identity)
            return seen

    async def _classify_completion_target(self, parent_session_id: str) -> str:
        """Classify an async-completion target before adapter acceptance: ``"deliver"`` (spawning
        session live or compression-rotated with a live continuation; the resolver still retargets),
        ``"terminal"`` (parent gone for good — unknown / user boundary like /new; drop the durable row
        rather than falsely ack), ``"retry"`` (DB unavailable / rotation mid-flight; release the claim)."""
        from gateway.run import _USER_BOUNDARY_END_REASONS
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return "retry"
        try:
            parent = await session_db.get_session(parent_session_id)
        except Exception:
            logger.debug("Async-completion pre-flight parent lookup failed for %s", parent_session_id, exc_info=True)
            return "retry"
        if parent is None:
            return "terminal"
        if not parent.get("ended_at"):
            return "deliver"
        end_reason = str(parent.get("end_reason") or "")
        if end_reason != "compression":
            # Only a USER-closed session (/new, user_exit, session_switch) is unreachable; idle/timeout
            # ends stay routable and the resolver retargets. Boundary set shared with the resolver.
            return "terminal" if end_reason in _USER_BOUNDARY_END_REASONS else "deliver"
        try:
            tip_session_id = await session_db.get_compression_tip(parent_session_id)
            if not tip_session_id or tip_session_id == parent_session_id:
                # Rotation mid-flight: continuation not visible yet. Retry, don't drop.
                return "retry"
            tip = await session_db.get_session(tip_session_id)
        except Exception:
            logger.debug("Async-completion pre-flight tip lookup failed for %s", parent_session_id, exc_info=True)
            return "retry"
        if tip is None or tip.get("ended_at"):
            return "retry"
        return "deliver"

    @staticmethod
    def _settle_durable_claim(kind: str, delegation_id: str, claim_id: str) -> None:
        """Best-effort ``drop``/``release`` of a durable completion claim."""
        fn_name, fail_msg = _DURABLE_CLAIM_OPS[kind]
        try:
            import tools.async_delegation as _ad
            getattr(_ad, fn_name)(delegation_id, claim_id)
        except Exception:
            logger.debug(fail_msg, exc_info=True)

    async def _preflight_completion_delivery(self, evt: dict) -> "_CompletionClaim":
        """Claim the durable row (async delegations) and verify the target before adapter acceptance.

        Adapter acceptance is not proof of delivery: the inner resolver can still fail closed inside
        the pipeline after acceptance, falsely acking the durable row. Verifying first gives drops an
        honest durable disposition.
        """
        claim = self._CompletionClaim()
        evt_type = evt.get("type")
        if evt_type == "async_delegation":
            claim.delegation_id = str(evt.get("delegation_id") or "")
            if claim.delegation_id:
                try:
                    from tools.async_delegation import claim_completion_delivery
                    claim.claim_id = f"gateway:{id(self)}:{__import__('uuid').uuid4().hex}"
                    if not claim_completion_delivery(claim.delegation_id, claim.claim_id):
                        claim.proceed = False
                        return claim
                except Exception as exc:
                    logger.warning("Could not claim durable async completion %s: %s", claim.delegation_id, exc)
                    claim.proceed, claim.early_result = False, False
                    return claim
        elif evt_type != "completion":
            return claim
        # Background completions carry only session_key, so after /new the OLD session's notification
        # would land in the NEW one. Stamped events get the async-delegation pre-flight; unstamped deliver.
        parent_session_id = str(evt.get("parent_session_id") or "").strip()
        if not parent_session_id:
            return claim
        # Pre-flight (#65838-class): adapter acceptance is NOT proof of delivery — the inner #55578 resolver
        # can still fail closed inside the message pipeline AFTER the adapter accepted, which would falsely
        # acknowledge the durable row as delivered. Verify the target here, before acceptance, and give
        # drops an honest durable disposition.
        verdict = await self._classify_completion_target(parent_session_id)
        if verdict == "terminal":
            if evt_type == "async_delegation":
                logger.warning(
                    "Async delegation %s targets permanently-gone session %s; "
                    "terminally dropping delivery (result remains in the delegation records).",
                    claim.delegation_id or "<legacy>", parent_session_id,
                )
                if claim.claim_id:
                    self._settle_durable_claim("drop", claim.delegation_id, claim.claim_id)
            else:
                logger.warning(
                    "Background process %s completion targets "
                    "permanently-gone session %s (user boundary such as "
                    "/new); dropping notification (output remains available via process(action='log')).",
                    evt.get("session_id") or "<unknown>", parent_session_id,
                )
            claim.proceed = False
        elif verdict == "retry":
            # Transient uncertainty: tell the watcher to re-poll rather than drop or misroute.
            if claim.claim_id:
                self._settle_durable_claim("release", claim.delegation_id, claim.claim_id)
            claim.proceed, claim.early_result = False, False
        return claim

    async def _deliver_completion_notification(self, synth_text: str, evt: dict) -> Optional[bool]:
        """Deliver once per live gateway, or return False for a retry.

        ``True``: adapter accepted; ``False``: injection failed, claim released for retry; ``None``:
        another same-lifecycle caller owns/delivered it, or no route. No cross-process exactly-once.
        """
        identity = self._completion_delivery_identity(evt)
        claim = await self._preflight_completion_delivery(evt)
        if not claim.proceed:
            return claim.early_result
        if identity is not None and self._completion_identity_seen(identity, claim=True):
            return None
        accepted = False
        try:
            injection_result = await self._inject_watch_notification(synth_text, evt)
            if injection_result is not True:
                return injection_result
            accepted = True
            if identity is not None:
                with self._completion_delivery_lock:
                    self._mark_completions_delivered_locked((identity,))
            # The durable row is the authoritative replay state — ack it after adapter acceptance.
            if claim.claim_id:
                try:
                    from tools.async_delegation import complete_completion_delivery
                    complete_completion_delivery(claim.delegation_id, claim.claim_id)
                except Exception as exc:
                    logger.warning("Could not acknowledge durable async completion %s: %s", claim.delegation_id, exc)
            return True
        finally:
            if identity is not None and not accepted:
                with self._completion_delivery_lock:
                    self._completion_deliveries_inflight.discard(identity)
            if claim.claim_id and not accepted:
                self._settle_durable_claim("release", claim.delegation_id, claim.claim_id)

    @staticmethod
    def _event_route_key(evt: dict, fields: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(evt.get(field) or "") for field in fields)

    @staticmethod
    def _format_coalesced_process_completions(entries: list[tuple[str, dict, asyncio.Future]]) -> str:
        """Build one bounded synthetic event from several redacted completions."""
        from gateway.run import _redact_gateway_user_facing_secrets
        lines = [
            f"[IMPORTANT: {len(entries)} background processes completed for this session.",
            "Treat these results as one completion batch and send at most one "
            "consolidated user-facing response.",
        ]
        shown = entries[:10]
        for _text, evt, _future in shown:
            session_id = str(evt.get("session_id") or "unknown")
            exit_code = evt.get("exit_code")
            reason = str(evt.get("completion_reason") or "exited")
            # Unconditional gateway redaction floor (the producer-seam redactor is configurable). Redact
            # BEFORE slicing: truncating first can leave a credential fragment the patterns miss.
            output = _redact_gateway_user_facing_secrets(str(evt.get("output") or "")).strip()
            if len(output) > 800:
                output = f"[… truncated …]\n{output[-800:]}"
            lines.append(f"\n- {session_id}: exit_code={exit_code}, reason={reason}")
            if output:
                lines.append(output)
        omitted = len(entries) - len(shown)
        if omitted:
            lines.append(
                f"\n- … and {omitted} more completion(s); inspect them with "
                "the process tool if they affect the conclusion."
            )
        lines.append("If a result does not change the current conclusion, absorb it silently.]")
        return "\n".join(lines)

    def _record_coalesced_completion_siblings(self, events: list[dict]) -> None:
        """Extend a successful primary delivery claim to its batched siblings."""
        identities = [i for i in map(self._completion_delivery_identity, events) if i is not None]
        with self._completion_delivery_lock:
            self._mark_completions_delivered_locked(identities)

    async def _flush_process_completion_batch(self, key: tuple[str, ...]) -> None:
        """Deliver one short-window completion batch and resolve its waiters."""
        current_task = asyncio.current_task()
        entries: list[tuple[str, dict, asyncio.Future]] = []
        delivered: Optional[bool] = False
        try:
            await asyncio.sleep(self._completion_notification_batch_window)
            entries = self._completion_notification_batches.pop(key, [])
            # Detach before delivery so a completion arriving mid-flight can schedule the next flush.
            if self._completion_notification_batch_tasks.get(key) is current_task:
                self._completion_notification_batch_tasks.pop(key, None)
            if not entries:
                return
            synth_text = entries[0][0] if len(entries) == 1 else self._format_coalesced_process_completions(entries)
            # A duplicate primary returns None from the dedupe seam; try the next identity so a fresh
            # sibling is never discarded with it.
            delivered = None
            for _text, candidate_evt, _future in entries:
                delivered = await self._deliver_completion_notification(synth_text, candidate_evt)
                if delivered is not None:
                    break
            if delivered is True and len(entries) > 1:
                self._record_coalesced_completion_siblings([evt for _text, evt, _future in entries])
        except asyncio.CancelledError:
            # Shutdown cancellation: recover undetached entries and resolve every waiter as retryable.
            delivered = False
            if not entries:
                entries = self._completion_notification_batches.pop(key, [])
            raise
        except Exception:
            logger.exception("Coalesced process completion delivery failed")
            delivered = False
        finally:
            # Never strand watcher futures: False = watcher retry path; None = ordinary dedupe result.
            self._settle_batch_waiters(entries, delivered)
            # Do not remove a newer flush task that reused the same route key.
            if self._completion_notification_batch_tasks.get(key) is current_task:
                self._completion_notification_batch_tasks.pop(key, None)

    @staticmethod
    def _settle_batch_waiters(entries, result) -> None:
        for _text, _evt, future in entries:
            if not future.done():
                future.set_result(result)

    async def _cancel_process_completion_batch_tasks(self) -> None:
        """Settle pending completion batches before adapter teardown."""
        self._completion_notification_batches_stopping = True
        tasks = {
            task
            for task in getattr(self, "_completion_notification_batch_flush_tasks", set())
            if not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Defensive cleanup for an orphaned queue with no live flush task.
        batches = getattr(self, "_completion_notification_batches", {})
        for entries in batches.values():
            self._settle_batch_waiters(entries, False)
        batches.clear()
        getattr(self, "_completion_notification_batch_tasks", {}).clear()
        getattr(self, "_completion_notification_batch_flush_tasks", set()).clear()

    async def _enqueue_process_completion_notification(self, synth_text: str, evt: dict) -> Optional[bool]:
        """Fan in concurrent process completions that share one conversation."""
        # Lazy defaults: lifecycle tests build GatewayRunner via object.__new__.
        for attr, default in (
            ("_completion_notification_batches", dict), ("_completion_notification_batch_tasks", dict),
            ("_completion_notification_batch_flush_tasks", set),
            ("_completion_notification_batch_window", lambda: 0.1),
            ("_completion_notification_batches_stopping", lambda: False), ("_background_tasks", set),
        ):
            if not hasattr(self, attr):
                setattr(self, attr, default())
        if self._completion_notification_batches_stopping:
            return False
        key = self._event_route_key(evt, self._COMPLETION_BATCH_KEY_FIELDS)
        future = asyncio.get_running_loop().create_future()
        self._completion_notification_batches.setdefault(key, []).append((synth_text, evt, future))
        if key not in self._completion_notification_batch_tasks:
            task = asyncio.create_task(self._flush_process_completion_batch(key))
            self._completion_notification_batch_tasks[key] = task
            # Keep the flush alive under the gateway's normal lifecycle accounting.
            self._retain_background_task(task)
            self._track_task_in(self._completion_notification_batch_flush_tasks, task)
        return await future

    def _enrich_async_delegation_routing(self, evt: dict) -> None:
        """Fill platform/chat_id/thread_id/chat_type on an async-delegation event.

        Such events only carry ``session_key`` (the daemon worker lacks per-message routing
        metadata). Best-effort: a CLI-origin event (empty session_key) is left as-is and won't route.
        """
        from gateway.run import _parse_session_key
        if evt.get("platform"):
            return  # already enriched
        parsed = _parse_session_key(evt.get("session_key", "") or "")
        if not parsed:
            return
        evt["platform"] = parsed.get("platform", "")
        evt["chat_type"] = parsed.get("chat_type", "")
        evt["chat_id"] = parsed.get("chat_id", "")
        if parsed.get("thread_id"):
            evt["thread_id"] = parsed["thread_id"]

    @staticmethod
    def _settle_sibling_claims(siblings: list[tuple[dict, str]], fn, fail_msg: str) -> None:
        for evt, claim_id in siblings:
            try:
                fn(evt, claim_id)
            except Exception:
                logger.debug(fail_msg, exc_info=True)

    async def _deliver_async_delegation_group(self, group: list[dict]) -> Optional[bool]:
        """Deliver a same-session batch of async completions as ONE turn: the primary carries the
        consolidated text of every sibling THIS runner claimed (siblings owned elsewhere are excluded;
        their claims are acked only after adapter acceptance). True after acceptance, False to requeue
        the group, None when nothing is deliverable here (retry siblings requeued)."""
        from gateway.run import _format_gateway_process_notification
        from tools.process_registry import process_registry as _pr
        deliverable: list[tuple[dict, str]] = []
        for evt in group:
            synth_text = _format_gateway_process_notification(evt)
            if not synth_text:
                continue
            identity = self._completion_delivery_identity(evt)
            if identity is not None and self._completion_identity_seen(identity):
                continue
            deliverable.append((evt, synth_text))
        if not deliverable:
            return None
        if len(deliverable) == 1:
            evt, synth_text = deliverable[0]
            return await self._deliver_completion_notification(synth_text, evt)
        from tools.async_delegation import claim_event_delivery, complete_event_delivery, release_event_delivery
        primary_evt, primary_text = deliverable[0]
        blocks = [primary_text]
        siblings: list[tuple[dict, str]] = []
        for evt, synth_text in deliverable[1:]:
            claim_id = claim_event_delivery(evt, f"gateway-batch:{id(self)}")
            if claim_id is None:
                # Another consumer owns this row: keep it out of our text so it is never double-injected.
                continue
            siblings.append((evt, claim_id))
            blocks.append(synth_text)
        if not siblings:
            return await self._deliver_completion_notification(primary_text, primary_evt)
        header = (
            f"[IMPORTANT: {len(blocks)} background subagent delegations "
            "completed for this session. Treat these results as one "
            "completion batch and send at most one consolidated user-facing "
            "response. If a result does not change the current conclusion, absorb it silently.]"
        )
        consolidated = "\n\n".join([header, *blocks])
        delivered: Optional[bool] = False
        try:
            delivered = await self._deliver_completion_notification(consolidated, primary_evt)
        finally:
            if delivered is True:
                self._settle_sibling_claims(
                    siblings, complete_event_delivery, "Could not acknowledge coalesced durable completion",
                )
                self._record_coalesced_completion_siblings([evt for evt, _claim_id in siblings])
            else:
                # Not delivered: release every sibling claim so a retry or another consumer can take it.
                self._settle_sibling_claims(siblings, release_event_delivery, "Could not release coalesced durable claim")
                if delivered is None:
                    # Primary dropped/owned elsewhere: requeue just the siblings for the next tick.
                    for evt, _claim_id in siblings:
                        _pr.completion_queue.put(evt)
        return delivered

    async def _async_delegation_watcher(self, interval: float = 2.0) -> None:
        """Drain async-delegation completions and inject them as new turns (IDLE case).

        Background subagents run on the daemon executor with no per-process watcher, so their
        completions would otherwise only be seen by the post-turn drain. Ignores non-async events.
        """
        await asyncio.sleep(3)  # let platforms finish connecting
        from tools.process_registry import process_registry as _pr
        while self._running:
            with _log_suppressed(logging.DEBUG, "Async delegation watcher error: %s"):
                # Take async-delegation events only; requeue watch/completion events for their own drains.
                requeue = []
                async_events = []
                while not _pr.completion_queue.empty():
                    try:
                        evt = _pr.completion_queue.get_nowait()
                    except Exception:
                        break
                    (async_events if evt.get("type") == "async_delegation" else requeue).append(evt)
                for evt in requeue:
                    _pr.completion_queue.put(evt)
                # A fan-out finishing together yields N completions for one session; group by full route +
                # parent session so each group becomes ONE consolidated turn.
                # A same-tick drain often carries several completions for the SAME originating session (a
                # fan-out of background subagents finishing together). Events for different sessions never
                # coalesce. See #70300.
                groups: dict[tuple[str, ...], list[dict]] = {}
                for evt in async_events:
                    self._enrich_async_delegation_routing(evt)
                    groups.setdefault(self._event_route_key(evt, self._ASYNC_GROUP_KEY_FIELDS), []).append(evt)
                for group in groups.values():
                    try:
                        delivered = await self._deliver_async_delegation_group(group)
                        if delivered is False:
                            for evt in group:
                                _pr.completion_queue.put(evt)
                    except Exception as e:
                        for evt in group:
                            _pr.completion_queue.put(evt)
                        logger.error("Async delegation injection error: %s", e)
            await asyncio.sleep(interval)

    @staticmethod
    def _redacted_output_tail(session, limit: int) -> str:
        """Last ``limit`` chars of process output through the secret redactors (unconditional floor)."""
        from gateway.run import _redact_gateway_user_facing_secrets
        new_output = session.output_buffer[-limit:] if session.output_buffer else ""
        if new_output:
            from agent.redact import redact_terminal_output
            new_output = redact_terminal_output(new_output, getattr(session, "command", "") or "")
            # redact_terminal_output() is unforced (raw when security.redact_secrets is off); this goes
            # straight to the adapter, so apply the same unconditional floor as agent-notify.
            new_output = _redact_gateway_user_facing_secrets(new_output)
        return new_output

    async def _send_watcher_message(self, platform_name: str, chat_id, thread_id, message_text: str) -> None:
        from gateway.run import _non_conversational_metadata
        adapter = self._adapter_by_platform_value(platform_name)
        if adapter and chat_id:
            with _log_suppressed(logging.ERROR, "Watcher delivery error: %s"):
                send_meta = {"thread_id": thread_id} if thread_id else None
                await adapter.send(
                    chat_id, message_text, metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                )

    @staticmethod
    def _build_process_completion_event(watcher: dict, session, session_id: str) -> dict:
        """Build the synthetic ``completion`` event for an agent-notify watcher."""
        from gateway.run import _redact_gateway_user_facing_secrets
        from agent.redact import redact_terminal_output
        from tools.ansi_strip import strip_ansi
        _command = getattr(session, "command", "") or ""
        _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
        _raw = redact_terminal_output(_raw, _command)
        # Keep the last ~2000 chars snapped to a line boundary, with a marker when cut.
        _LIMIT = 2000
        # Truncate at line boundaries so notifications never start mid-line (fixes #23284). Keep the last
        # ~2000 chars but snap to the nearest preceding newline, then prepend a truncation marker when
        # output was cut.
        if len(_raw) > _LIMIT:
            _tail = _raw[-_LIMIT:]
            _nl = _tail.find("\n")
            _tail = _tail[_nl + 1:] if _nl != -1 else _tail
            _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
        else:
            _out = _raw
        return {
            "type": "completion",
            "session_id": session_id,
            **{k: watcher.get(k, "") for k in _WATCHER_ROUTE_FIELDS},
            "message_id": str(watcher.get("message_id") or "").strip() or None,
            "started_at": getattr(session, "started_at", None),
            "command": _redact_gateway_user_facing_secrets(_command),
            "exit_code": session.exit_code,
            "completion_reason": getattr(session, "completion_reason", "exited"),
            "termination_source": getattr(session, "termination_source", ""),
            "output": _redact_gateway_user_facing_secrets(_out),
            # Spawning session-db id: lets pre-flight drop this completion if the user /new'd first.
            "parent_session_id": (
                watcher.get("parent_session_id") or getattr(session, "parent_session_id", "") or ""
            ),
        }

    def _format_process_final_message(self, session_id: str, session, notify_mode: str) -> str:
        from gateway.run import _format_concise_process_notification, _redact_gateway_user_facing_secrets
        new_output = self._redacted_output_tail(session, 1000)
        if notify_mode != "concise":
            return (
                f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                f"Here's the final output:\n{new_output}]"
            )
        _started = getattr(session, "started_at", None)
        _dur = max(0.0, time.time() - _started) if isinstance(_started, (int, float)) else None
        return _format_concise_process_notification(
            session_id, _redact_gateway_user_facing_secrets(getattr(session, "command", "") or ""),
            session.exit_code, new_output, duration_seconds=_dur,
        )

    async def _run_process_watcher(self, watcher: dict) -> None:
        """Poll a background process and push updates until it exits. Mode
        (``display.background_process_notifications``): concise (default one-liner; failures append
        the output tail) / all (running updates + final raw) / result (final raw) / error (final raw
        if exit != 0) / off."""
        from tools.process_registry import process_registry
        from tools.process_registry_notifications import format_process_notification
        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = self._load_background_notifications_mode()
        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)
        silent = notify_mode == "off" and not agent_notify
        last_output_len = 0
        while True:
            await asyncio.sleep(interval)
            session = process_registry.get(session_id)
            if session is None:
                break
            if silent:
                # Still wait for the process to exit so we can log it, but don't push any messages.
                if session.exited:
                    break
                continue
            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len
            if session.exited:
                # Agent-notify: inject a synthetic message unless the agent already consumed the result via
                # wait/log (poll() is read-only and deliberately does NOT mark consumed).
                if agent_notify and not process_registry.is_completion_consumed(session_id):
                    completion_evt = self._build_process_completion_event(watcher, session, session_id)
                    synth_text = format_process_notification(completion_evt)
                    if not synth_text:
                        break
                    delivered = await self._enqueue_process_completion_notification(synth_text, completion_evt)
                    if delivered is False:
                        # The process remains terminal; retry after failed adapter injection instead
                        # of suppressing the result.
                        continue
                    break
                # Text-only notification; skip when already consumed via wait/log (the agent_notify branch
                # FALLS THROUGH here, hence the re-check).
                if process_registry.is_completion_consumed(session_id):
                    logger.debug(
                        "Process watcher: completion for %s already consumed "
                        "via wait/log — skipping raw notification (#65379)", session_id,
                    )
                    break
                if notify_mode in {"concise", "all", "result"} or (
                    notify_mode == "error" and session.exit_code not in {0, None}
                ):
                    message_text = self._format_process_final_message(session_id, session, notify_mode)
                    await self._send_watcher_message(platform_name, chat_id, thread_id, message_text)
                break
            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output — deliver a status update (only in "all" mode; agent_notify watchers
                # only care about completion).
                new_output = self._redacted_output_tail(session, 500)
                await self._send_watcher_message(
                    platform_name, chat_id, thread_id,
                    f"[Background process {session_id} is still running~ New output:\n{new_output}]",
                )
        logger.debug("Process watcher ended%s: %s", " (silent)" if silent else "", session_id)
