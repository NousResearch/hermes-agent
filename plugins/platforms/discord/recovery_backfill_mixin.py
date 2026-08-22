"""Recovery backfill + processing reaction hooks for the Discord adapter.

Extracted VERBATIM from ``plugins/platforms/discord/adapter.py`` (god-file
kill slice R2-S1, epic #78647, target #78634): recovery backfill ledger
(C1, lines 2096-2745 at pin) plus reaction hooks (C2, lines 2942-2996).

``DiscordAdapter`` composes this mixin FIRST in its bases tuple so the
``on_processing_start`` / ``on_processing_complete`` overrides win MRO over
the ``BasePlatformAdapter`` no-op stubs; appending it last would silently
kill the reaction hooks.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import time
from contextlib import suppress
from pathlib import Path as _Path
from typing import Any, Optional

try:
    import discord
    from discord import Message as DiscordMessage, Intents
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None
    DiscordMessage = Any
    Intents = Any
    commands = None

from gateway.platforms.base import MessageEvent, ProcessingOutcome, SendResult

# Keep log records under the adapter's logger name (pre-extraction behavior).
logger = logging.getLogger("plugins.platforms.discord.adapter")


class DiscordRecoveryBackfillMixin:
    def _missed_message_backfill_enabled(self) -> bool:
        """Whether to reconcile Discord messages missed while the gateway was down."""
        configured = self.config.extra.get("missed_message_backfill")
        if isinstance(configured, dict) and "enabled" in configured:
            value = configured["enabled"]
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes", "on")
            return bool(value)
        raw = os.getenv("DISCORD_MISSED_MESSAGE_BACKFILL", "false")
        return str(raw).strip().lower() in ("true", "1", "yes", "on")

    def _missed_message_backfill_channels(self) -> set[str]:
        """Channels to scan for missed messages after Discord reconnects.

        Defaults to the union of allowed and free-response channels so both
        mention-gated requests and mention-free work can be recovered.
        Operators can set ``channels: "*"`` to scan every reachable text
        channel, but the safe default is scoped.
        """
        configured = self.config.extra.get("missed_message_backfill")
        if isinstance(configured, dict) and "channels" in configured:
            raw = configured.get("channels")
            if isinstance(raw, list):
                return {str(item).strip() for item in raw if str(item).strip()}
            raw = str(raw or "")
            if raw.strip():
                return {item.strip() for item in raw.split(",") if item.strip()}
        raw = self._gate_env("DISCORD_MISSED_MESSAGE_BACKFILL_CHANNELS")
        if not raw.strip():
            allowed = self._get_allowed_channels()
            return allowed | self._discord_free_response_channels()
        return {item.strip() for item in raw.split(",") if item.strip()}

    def _missed_message_backfill_window_seconds(self) -> float:
        configured = self.config.extra.get("missed_message_backfill")
        raw = (
            configured.get("window_seconds", 21600)
            if isinstance(configured, dict)
            else os.getenv("DISCORD_MISSED_MESSAGE_BACKFILL_WINDOW_SECONDS", "21600")
        )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 21600.0
        return max(60.0, value)

    def _missed_message_backfill_limit(self) -> int:
        configured = self.config.extra.get("missed_message_backfill")
        raw = (
            configured.get("limit", 100)
            if isinstance(configured, dict)
            else os.getenv("DISCORD_MISSED_MESSAGE_BACKFILL_LIMIT", "100")
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 100
        return max(1, min(value, 500))

    def _missed_message_backfill_max_dispatches(self) -> int:
        configured = self.config.extra.get("missed_message_backfill")
        raw = (
            configured.get("max_dispatches", 10)
            if isinstance(configured, dict)
            else os.getenv("DISCORD_MISSED_MESSAGE_BACKFILL_MAX_DISPATCHES", "10")
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 10
        return max(1, min(value, 100))

    def _ensure_missed_message_backfill_task(self) -> asyncio.Task:
        """Return the active recovery task, or start one when none is running."""
        task = self._missed_message_backfill_task
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(self._run_missed_message_backfill())
        self._missed_message_backfill_task = task
        runner = getattr(self, "gateway_runner", None)
        if runner is not None and getattr(runner, "_startup_restore_in_progress", False):
            tasks = getattr(runner, "_startup_restore_tasks", None)
            if tasks is None:
                tasks = []
                runner._startup_restore_tasks = tasks
            tasks.append(task)
        return task

    async def _run_missed_message_backfill(self) -> None:
        """Find and enqueue recent Discord messages missed while the bot was down.

        Discord gateway events are not replayed for messages sent while the bot
        is offline. Normal startup resume only handles sessions already marked
        resume_pending; this pass scans recent channel/thread history, records
        what it saw durably, and reuses the normal message handler for messages
        that lack a substantive non-outage Hermes response. Emoji-only acks are
        deliberately not sufficient completion evidence.
        """
        if not self._client:
            return
        channels = self._missed_message_backfill_channels()
        ledger_ok = await self._with_discord_recovery_db_async(
            lambda conn: conn.execute("SELECT 1").fetchone() is not None,
            False,
        )
        if not ledger_ok:
            logger.error(
                "[%s] Missed-message recovery aborted: durable ledger unavailable",
                self.name,
            )
            return
        scan_id = await asyncio.to_thread(
            self._record_recovery_scan_start,
            channels,
        )
        if not channels:
            logger.info("[%s] Missed-message backfill enabled but no channels configured", self.name)
            await asyncio.to_thread(
                self._record_recovery_scan_complete,
                scan_id,
                status="skipped",
                scanned=0,
                missed=0,
                dispatched=0,
            )
            return

        max_dispatches = self._missed_message_backfill_max_dispatches()
        dispatched = 0
        scanned = 0
        missed = 0
        try:
            async for message in self._iter_missed_message_backfill_candidates(channels):
                scanned += 1
                message_id = str(getattr(message, "id", ""))
                self._record_discord_message_seen(message, status="discovered")
                # A live gateway event may race this REST scan. Check without
                # claiming the ID; the shared ingress helper owns the dedup
                # write immediately before normal auth/filter dispatch.
                if self._dedup.contains(message_id):
                    continue
                if not await self._should_backfill_discord_message(message):
                    continue
                missed += 1
                logger.info(
                    "[%s] Backfilling missed Discord message %s in channel %s",
                    self.name,
                    getattr(message, "id", "unknown"),
                    getattr(getattr(message, "channel", None), "id", "unknown"),
                )
                self._record_recovery_attempt(message, status="queued")
                try:
                    admitted = await self._dispatch_recovered_message(message)
                    if admitted:
                        dispatched += 1
                except asyncio.CancelledError:
                    self._dedup.discard(message_id)
                    self._record_recovery_attempt(message, status="cancelled")
                    raise
                except Exception as exc:
                    self._dedup.discard(message_id)
                    self._record_recovery_attempt(message, status="failed", error=str(exc))
                    raise
                if dispatched >= max_dispatches:
                    break
            await asyncio.to_thread(
                self._record_recovery_scan_complete,
                scan_id,
                status="success",
                scanned=scanned,
                missed=missed,
                dispatched=dispatched,
            )
            logger.info(
                "[%s] Missed-message backfill complete: scanned=%d missed=%d dispatched=%d",
                self.name,
                scanned,
                missed,
                dispatched,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(
                self._record_recovery_scan_complete,
                scan_id,
                status="cancelled",
                scanned=scanned,
                missed=missed,
                dispatched=dispatched,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            await asyncio.to_thread(
                self._record_recovery_scan_complete,
                scan_id,
                status="failed",
                scanned=scanned,
                missed=missed,
                dispatched=dispatched,
                error=str(exc),
            )
            logger.warning("[%s] Missed-message backfill failed: %s", self.name, exc, exc_info=True)

    async def _dispatch_recovered_message(self, message: Any) -> bool:
        """Run one recovered message through the live Discord ingress gates."""
        if not isinstance(message.channel, discord.DMChannel):
            parent_id = self._get_parent_channel_id(message.channel)
            channel_keys = self._discord_channel_keys(message, parent_id)
            free_channels = self._discord_free_response_channels()
            in_bot_thread = (
                isinstance(message.channel, discord.Thread)
                and str(message.channel.id) in self._threads
                and not self._discord_thread_require_mention()
            )
            if (
                self._discord_require_mention()
                and "*" not in free_channels
                and not (channel_keys & free_channels)
                and not in_bot_thread
                and not self._self_is_explicitly_mentioned(message)
            ):
                return False
        admitted, role_authorized = self._discord_message_admission(
            message, claim=False,
        )
        if not admitted:
            return False
        return await self._handle_message(
            message,
            role_authorized=role_authorized,
            recovered=True,
        )

    async def _iter_missed_message_backfill_candidates(self, channel_ids: set[str]):
        if not self._client:
            return
        after = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            seconds=self._missed_message_backfill_window_seconds()
        )
        limit = self._missed_message_backfill_limit()
        seen: set[str] = set()

        candidate_channels = []
        if "*" in channel_ids:
            for guild in getattr(self._client, "guilds", []) or []:
                candidate_channels.extend(getattr(guild, "text_channels", []) or [])
        else:
            for channel_id in sorted(channel_ids):
                channel = None
                try:
                    channel = self._client.get_channel(int(channel_id))
                except Exception:
                    channel = None
                if channel is None:
                    try:
                        channel = await self._client.fetch_channel(int(channel_id))
                    except Exception as exc:
                        logger.debug("[%s] Cannot fetch backfill channel %s: %s", self.name, channel_id, exc)
                        continue
                candidate_channels.append(channel)

        iterators = [
            self._iter_channel_and_thread_messages(
                channel,
                limit=limit,
                after=after,
                seen_channels=seen,
            ).__aiter__()
            for channel in candidate_channels
        ]
        yielded = 0
        while iterators and yielded < limit:
            next_round = []
            for iterator in iterators:
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    continue
                yield item
                yielded += 1
                next_round.append(iterator)
                if yielded >= limit:
                    return
            iterators = next_round

    async def _iter_channel_and_thread_messages(self, channel: Any, *, limit: int, after: Any, seen_channels: set[str]):
        """Yield history from a channel plus active/recent archived child threads."""
        channel_key = str(getattr(channel, "id", ""))
        if not channel_key or channel_key in seen_channels:
            return
        seen_channels.add(channel_key)

        cursor = self._discord_recovery_cursor(channel_key)
        if cursor:
            with suppress(ValueError, TypeError):
                after = discord.Object(id=int(cursor))
        history = getattr(channel, "history", None)
        if callable(history):
            try:
                # Fetch the latest N messages in the window, then restore
                # chronological dispatch order. With oldest_first=True the API
                # returns the earliest N and can permanently starve newer work.
                history_iter = history(
                    limit=limit,
                    after=after,
                    oldest_first=False,
                )
                messages = []
                async for message in history_iter:  # type: ignore[attr-defined]
                    messages.append(message)
                for message in reversed(messages):
                    yield message
            except Exception as exc:
                logger.debug("[%s] Cannot read history for %s: %s", self.name, channel_key, exc)

        child_threads = list(getattr(channel, "threads", []) or [])
        archived_threads = getattr(channel, "archived_threads", None)
        if callable(archived_threads):
            try:
                async for thread in archived_threads(limit=limit):
                    child_threads.append(thread)
            except Exception as exc:
                logger.debug("[%s] Cannot list archived threads for %s: %s", self.name, channel_key, exc)

        for thread in child_threads:
            thread_key = str(getattr(thread, "id", ""))
            if not thread_key or thread_key in seen_channels:
                continue
            async for message in self._iter_channel_and_thread_messages(thread, limit=limit, after=after, seen_channels=seen_channels):
                yield message

    def _discord_recovery_cursor(self, channel_id: str) -> Optional[str]:
        if not channel_id:
            return None

        def _op(conn):
            row = conn.execute(
                "SELECT last_message_id FROM discord_recovery_cursors WHERE channel_id=?",
                (channel_id,),
            ).fetchone()
            return str(row[0]) if row else None

        return self._with_discord_recovery_db(_op)

    def _advance_discord_recovery_cursor(self, channel_id: str, message_id: str) -> None:
        if not channel_id or not message_id:
            return
        now = self._utc_now_iso()

        def _op(conn):
            conn.execute(
                """
                INSERT INTO discord_recovery_cursors (channel_id, last_message_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_message_id=excluded.last_message_id,
                    updated_at=excluded.updated_at
                """,
                (channel_id, message_id, now),
            )

        self._with_discord_recovery_db(_op)

    async def _should_backfill_discord_message(self, message: Any) -> bool:
        """Return True when a recent Discord message still needs Hermes work."""
        if not self._client or not getattr(self._client, "user", None):
            return False
        if getattr(getattr(message, "author", None), "id", None) == getattr(self._client.user, "id", None):
            return False
        if self._discord_message_is_persistently_complete(str(getattr(message, "id", ""))):
            return False
        if self._discord_message_has_active_claim(str(getattr(message, "id", ""))):
            return False
        # A success reaction alone is only an acknowledgement.  It is not
        # enough evidence that the substantive response/action completed.
        if await self._message_has_non_down_bot_response(message):
            return False
        return True

    def _is_down_notice_content(self, content: str) -> bool:
        """Recognize only explicit Hermes/gateway outage notices."""
        text = (content or "").lower()
        subject = r"(?:hermes|the agent|agent|the gateway|gateway|bmo)"
        state = r"(?:is|was|appears to be|is currently|was currently)"
        condition = r"(?:down|offline|unavailable|not running)"
        return re.search(rf"\b{subject}\s+{state}\s+{condition}\b", text) is not None

    async def _message_has_non_down_bot_response(self, message: Any) -> bool:
        """Detect an already-addressed message without trusting down notices."""
        bot_user = getattr(self._client, "user", None) if self._client else None
        bot_id = getattr(bot_user, "id", None)
        if bot_id is None:
            return False

        async def _scan_history(channel: Any) -> bool:
            history = getattr(channel, "history", None)
            if not callable(history):
                return False
            try:
                async for candidate in history(limit=25, after=getattr(message, "created_at", None), oldest_first=True):
                    author = getattr(candidate, "author", None)
                    if getattr(author, "id", None) != bot_id:
                        continue
                    if self._is_down_notice_content(getattr(candidate, "content", "")):
                        continue
                    reference = getattr(candidate, "reference", None)
                    ref_id = str(getattr(reference, "message_id", "") or "")
                    if ref_id == str(getattr(message, "id", "")):
                        return True
            except Exception:
                return False
            return False

        message_channel = getattr(message, "channel", None)
        # Only an explicit reply reference proves which input a bot response
        # completed. An arbitrary later bot post can otherwise mask multiple
        # unanswered requests in the same parent channel or thread.
        if await _scan_history(message_channel):
            return True

        thread = getattr(message, "thread", None)
        if thread is not None and await _scan_history(thread):
            return True
        return False

    def _discord_recovery_db_path(self) -> _Path:
        return self._discord_recovery_store.path()

    def _with_discord_recovery_db(self, fn, default=None):
        return self._discord_recovery_store.call(fn, default)

    async def _with_discord_recovery_db_async(self, fn, default=None):
        return await asyncio.to_thread(
            self._discord_recovery_store.call,
            fn,
            default,
        )

    @staticmethod
    def _utc_now_iso() -> str:
        import datetime as _dt
        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    def _message_channel_ids(self, message: Any) -> tuple[str, Optional[str], Optional[str]]:
        channel = getattr(message, "channel", None)
        channel_id = str(getattr(channel, "id", "") or "")
        parent_id = str(getattr(channel, "parent_id", "") or "") or None
        thread_id = channel_id if parent_id else None
        return channel_id, thread_id, parent_id

    def _record_discord_message_seen(self, message: Any, *, status: str) -> None:
        if not self._missed_message_backfill_enabled():
            return
        message_id = str(getattr(message, "id", "") or "")
        if not message_id:
            return
        channel_id, thread_id, parent_id = self._message_channel_ids(message)
        author_id = str(getattr(getattr(message, "author", None), "id", "") or "")
        created_at = getattr(message, "created_at", None)
        created_text = created_at.isoformat() if hasattr(created_at, "isoformat") else None
        now = self._utc_now_iso()

        def _op(conn):
            existing = conn.execute("SELECT status FROM discord_messages WHERE message_id=?", (message_id,)).fetchone()
            final_status = existing[0] if existing and existing[0] == "responded" else status
            conn.execute(
                """
                INSERT INTO discord_messages (message_id, channel_id, thread_id, parent_channel_id, author_id, created_at, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    channel_id=excluded.channel_id,
                    thread_id=excluded.thread_id,
                    parent_channel_id=excluded.parent_channel_id,
                    author_id=excluded.author_id,
                    created_at=COALESCE(discord_messages.created_at, excluded.created_at),
                    status=?,
                    updated_at=excluded.updated_at
                """,
                (message_id, channel_id, thread_id, parent_id, author_id, created_text, final_status, now, final_status),
            )

        self._with_discord_recovery_db(_op)

    def _record_recovery_attempt(self, message: Any, *, status: str, error: Optional[str] = None) -> None:
        if not self._missed_message_backfill_enabled():
            return
        self._record_discord_message_seen(message, status=status)
        message_id = str(getattr(message, "id", "") or "")
        if not message_id:
            return
        now = self._utc_now_iso()

        def _op(conn):
            conn.execute(
                """
                UPDATE discord_messages
                   SET status=?, attempts=attempts+1, last_attempt_at=?, last_error=?, updated_at=?
                 WHERE message_id=?
                """,
                (status, now, error, now, message_id),
            )

        self._with_discord_recovery_db(_op)

    def _record_discord_processing_start(self, event: MessageEvent, *, emoji_ack: bool) -> None:
        if not self._missed_message_backfill_enabled():
            return
        message = event.raw_message
        self._record_discord_message_seen(message, status="processing")
        message_id = str(getattr(message, "id", "") or getattr(event, "message_id", "") or "")
        if not message_id:
            return
        now = self._utc_now_iso()

        def _op(conn):
            conn.execute(
                "UPDATE discord_messages SET status='processing', emoji_ack=?, updated_at=? WHERE message_id=?",
                (1 if emoji_ack else 0, now, message_id),
            )

        self._with_discord_recovery_db(_op)

    def _record_discord_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        if not self._missed_message_backfill_enabled():
            return
        message_id = str(getattr(getattr(event, "raw_message", None), "id", "") or getattr(event, "message_id", "") or "")
        if not message_id:
            return
        status = "processed" if outcome == ProcessingOutcome.SUCCESS else ("cancelled" if outcome == ProcessingOutcome.CANCELLED else "failed")
        now = self._utc_now_iso()

        def _op(conn):
            conn.execute(
                "UPDATE discord_messages "
                "SET status=CASE WHEN status='responded' THEN status ELSE ? END, "
                "updated_at=? WHERE message_id=?",
                (status, now, message_id),
            )

        self._with_discord_recovery_db(_op)

    def _record_discord_response(
        self,
        *,
        reply_to: Optional[str],
        result: SendResult,
        content: str,
        final: bool,
    ) -> None:
        if not self._missed_message_backfill_enabled() or not reply_to:
            return
        now = self._utc_now_iso()
        completed = bool(final and result.success)
        status = "responded" if completed else "failed"

        def _op(conn):
            conn.execute(
                """
                INSERT INTO discord_messages (message_id, status, replied, outage_response, response_message_id, updated_at)
                VALUES (?, ?, ?, 0, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    status=CASE WHEN ? THEN 'responded' ELSE discord_messages.status END,
                    replied=CASE WHEN ? THEN 1 ELSE discord_messages.replied END,
                    outage_response=CASE WHEN ? THEN 0 ELSE discord_messages.outage_response END,
                    response_message_id=COALESCE(?, response_message_id),
                    updated_at=?
                """,
                (
                    reply_to,
                    status,
                    1 if completed else 0,
                    result.message_id,
                    now,
                    1 if completed else 0,
                    1 if completed else 0,
                    1 if completed else 0,
                    result.message_id,
                    now,
                ),
            )

        self._with_discord_recovery_db(_op)
        if completed:
            def _channel_for_message(conn):
                row = conn.execute(
                    "SELECT COALESCE(thread_id, channel_id) FROM discord_messages "
                    "WHERE message_id=?",
                    (reply_to,),
                ).fetchone()
                return str(row[0]) if row and row[0] else None

            channel_id = self._with_discord_recovery_db(_channel_for_message)
            if channel_id:
                self._advance_discord_recovery_cursor(channel_id, reply_to)

    def _discord_message_is_persistently_complete(self, message_id: str) -> bool:
        if not message_id:
            return False

        def _op(conn):
            row = conn.execute("SELECT status, replied, outage_response FROM discord_messages WHERE message_id=?", (message_id,)).fetchone()
            if not row:
                return False
            status, replied, outage = row
            return status == "responded" and bool(replied) and not bool(outage)

        return bool(self._with_discord_recovery_db(_op, default=False))

    def _discord_message_has_active_claim(self, message_id: str) -> bool:
        if not message_id:
            return False
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)
        ).isoformat()

        def _op(conn):
            row = conn.execute(
                "SELECT status, updated_at FROM discord_messages WHERE message_id=?",
                (message_id,),
            ).fetchone()
            return bool(
                row
                and row[0] in {"queued", "processing"}
                and row[1] >= cutoff
            )

        return bool(self._with_discord_recovery_db(_op, default=True))

    def _record_recovery_scan_start(self, channels: set[str]) -> str:
        scan_id = f"{int(time.time() * 1000)}-{os.getpid()}"
        now = self._utc_now_iso()

        def _op(conn):
            conn.execute(
                "INSERT OR REPLACE INTO discord_recovery_scans (scan_id, started_at, status, channels, window_seconds, limit_count) VALUES (?, ?, ?, ?, ?, ?)",
                (scan_id, now, "running", json.dumps(sorted(channels)), self._missed_message_backfill_window_seconds(), self._missed_message_backfill_limit()),
            )

        self._with_discord_recovery_db(_op)
        return scan_id

    def _record_recovery_scan_complete(self, scan_id: str, *, status: str, scanned: int, missed: int, dispatched: int, error: Optional[str] = None) -> None:
        now = self._utc_now_iso()

        def _op(conn):
            conn.execute(
                "UPDATE discord_recovery_scans SET completed_at=?, status=?, scanned=?, missed=?, dispatched=?, error=? WHERE scan_id=?",
                (now, status, scanned, missed, dispatched, error, scan_id),
            )

        self._with_discord_recovery_db(_op)

    async def _add_reaction(self, message: Any, emoji: str) -> bool:
        """Add an emoji reaction to a Discord message."""
        if not message or not hasattr(message, "add_reaction"):
            return False
        try:
            await message.add_reaction(emoji)
            return True
        except Exception as e:
            logger.debug("[%s] add_reaction failed (%s): %s", self.name, emoji, e)
            return False

    async def _remove_reaction(self, message: Any, emoji: str) -> bool:
        """Remove the bot's own emoji reaction from a Discord message."""
        if not message or not hasattr(message, "remove_reaction") or not self._client or not self._client.user:
            return False
        try:
            await message.remove_reaction(emoji, self._client.user)
            return True
        except Exception as e:
            logger.debug("[%s] remove_reaction failed (%s): %s", self.name, emoji, e)
            return False

    def _reactions_enabled(self) -> bool:
        """Check if message reactions are enabled via config/env."""
        return os.getenv("DISCORD_REACTIONS", "true").lower() not in {"false", "0", "no"}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Add an in-progress reaction and record durable handling state."""
        message = event.raw_message
        acked = False
        if self._reactions_enabled() and hasattr(message, "add_reaction"):
            acked = await self._add_reaction(message, "👀")
        await asyncio.to_thread(
            self._record_discord_processing_start,
            event,
            emoji_ack=acked,
        )

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Swap the in-progress reaction for final reaction and durable state."""
        await asyncio.to_thread(
            self._record_discord_processing_complete,
            event,
            outcome,
        )
        if not self._reactions_enabled():
            return
        message = event.raw_message
        if hasattr(message, "add_reaction"):
            await self._remove_reaction(message, "👀")
            if outcome == ProcessingOutcome.SUCCESS:
                await self._add_reaction(message, "✅")
            elif outcome == ProcessingOutcome.FAILURE:
                await self._add_reaction(message, "❌")
