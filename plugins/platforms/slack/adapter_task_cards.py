"""Slack-native task cards (``platforms.slack.extra.native_task_cards``) for ``SlackAdapter``.

One card per turn and thread, streamed with ``chat.startStream`` / ``appendStream`` / ``stopStream``.
The card is closed with a terminal frame (title + settled rows) and rotated before Slack's
documented 10-minute stream lifetime, so no card in the thread is left saying the agent is still
working (#141643). The thinking card (``adapter_thinking_card.py``) shares the stream machinery
under its own slot.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.i18n import t
from gateway.platforms.base import SendResult

try:  # sibling module; support both package and flat plugin-dir import
    from .block_kit import _inline_elements
except ImportError:  # pragma: no cover - plugin loaded outside package context
    from block_kit import _inline_elements  # type: ignore

logger = logging.getLogger("plugins.platforms.slack.adapter")

TOOLS_SLOT = "tools"
# Slack documents a stream as open for up to 10 minutes after chat.startStream.
DEFAULT_ROTATE_SECONDS = 540.0


@dataclass
class _NativeTaskCardStream:
    """Serialized state for one workspace-scoped Slack progress stream."""

    team_id: str
    channel: str
    thread_ts: str
    stream_ts: str = ""
    stopped: bool = False
    started_at: float = 0.0
    # task id -> last task_update chunk on THIS stream message (the rotation's closing frame).
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The turn's terminal frame, written with chat.stopStream (set_native_task_card_final_frame).
    final_chunks: list[dict[str, Any]] = field(default_factory=list)
    thinking_rows_sent: int = 0
    thinking_rows: list[dict[str, Any]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _first_truthy(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """First truthy ``mapping[key]`` in ``keys`` order, else None."""
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def _slack_error_is(exc: BaseException, code: str) -> bool:
    """True when a Slack API exception carries ``code`` as its structured ``error`` field.

    Reads ``exc.response["error"]`` (slack_sdk's SlackApiError shape); never matches on the
    message text, which embeds the whole response body and would match on unrelated codes.
    """
    response = getattr(exc, "response", None)
    try:
        return bool(response) and response.get("error") == code
    except Exception:
        return False


def _rotate_seconds(extra: Any) -> float:
    """``native_task_card_rotate_seconds`` (0 disables proactive rotation)."""
    value = extra.get("native_task_card_rotate_seconds") if isinstance(extra, dict) else None
    if value is None:
        return DEFAULT_ROTATE_SECONDS
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return DEFAULT_ROTATE_SECONDS


class SlackTaskCardMixin:
    """The tool card's Slack stream lifecycle, mixed into ``SlackAdapter``."""

    @staticmethod
    def rich_text_inline_elements(text: str) -> list[dict[str, Any]]:
        """Slack ``rich_text`` inline elements for card details (the rich publisher's renderer)."""
        return _inline_elements(text)

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

    def task_card_rollup_enabled(self) -> bool:
        """Whether the tool card folds consecutive steps into one row per run (opt-in)."""
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        return self._truthy_config(extra.get("task_card_rollup", False))

    def _native_task_card_key(
        self, chat_id: str, reply_to: Optional[str], metadata: Optional[dict[str, Any]],
        slot: str = "tools",
    ) -> Optional[tuple[str, ...]]:
        thread_ts = self._resolve_thread_ts(reply_to, metadata)
        if not thread_ts:
            return None
        key = self._workspace_thread_key(
            self._metadata_team_id(metadata), chat_id, str(thread_ts))
        if not key:
            return None
        # The tool card keeps the workspace-thread key; other cards in the thread add their slot.
        return key if slot == TOOLS_SLOT else (*key, slot)

    def native_task_card_destination_supported(
        self, chat_id: str, *, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Placement eligibility independent of connection/stream availability."""
        return self._native_task_card_key(chat_id, reply_to, metadata) is not None

    async def send_native_task_card_progress(
        self, chat_id: str, tasks: list[dict[str, str]], *, title: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
        fallback_text: Optional[str] = None) -> SendResult:
        """Start or update a Slack-native plan/task progress stream."""
        title = title or t("platform.slack.task_card.title")
        if not self._app:
            return SendResult(success=False, error="Not connected")
        if not tasks:
            return SendResult(success=False, error="No tasks")
        key = self._native_task_card_key(chat_id, reply_to, metadata)
        if key is None:
            return SendResult(success=False, error="No Slack thread target")
        stream = self._native_task_card_streams.get(key)
        if stream is None or stream.stopped:
            stream = _NativeTaskCardStream(team_id=key[0], channel=chat_id, thread_ts=key[2])
            # No await between lookup and assignment, so racers share this lock.
            self._native_task_card_streams[key] = stream
        async with stream.lock:
            if stream.stopped:
                return SendResult(success=False, error="Progress stream already stopped")
            try:
                client = self._get_client(chat_id, team_id=stream.team_id)
                if stream.stream_ts and self._native_task_card_due_for_rotation(stream):
                    await self._rotate_native_task_card_stream(client, stream)
                if not stream.stream_ts:
                    await self._start_native_task_card_stream(client, stream, metadata)
                rows = [self._task_update_chunk(task) for task in tasks]
                chunks: list[dict[str, Any]] = [{"type": "plan_update", "title": str(title)[:256]}, *rows]
                # chunks-only: Slack rejects markdown_text alongside chunks
                # (cannot_provide_both_markdown_text_and_chunks, #87743); the gateway owns
                # the editable-text fallback rail that fallback_text feeds when this call fails.
                await self._append_stream_chunks(client, stream, chunks, chat_id, metadata)
                stream.rows.update((row["id"], {k: v for k, v in row.items() if k != "details"}) for row in rows)
                return SendResult(success=True, message_id=stream.stream_ts)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("[Slack] Native task-card progress error: %s", exc, exc_info=True)
                return SendResult(success=False, error=str(exc), retryable=True)

    async def _append_stream_chunks(
        self, client, stream: _NativeTaskCardStream, chunks: list[dict[str, Any]],
        chat_id: str, metadata: Optional[dict[str, Any]],
        reopen_chunks: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """chat.appendStream chunks, reopening a sealed stream once.

        Slack seals a native stream server-side once it outlives its lifetime
        (documented as 10 minutes after chat.startStream; the proactive rotation
        in ``_rotate_native_task_card_stream`` normally closes the card first).
        The sealed message is a regular message now, so reopen a fresh card in
        the same thread. ``reopen_chunks`` is the FULL visible
        projection (required for delta-frame senders: a fresh stream carries no
        history, so it must be seeded with every row). One reopen per update:
        a second rejection is a real failure.
        """
        try:
            await client.api_call(
                "chat.appendStream", json={"channel": chat_id, "ts": stream.stream_ts, "chunks": chunks})
        except Exception as exc:
            if not _slack_error_is(exc, "message_not_in_streaming_state"):
                raise
            logger.info(
                "[Slack] Native task-card stream %s expired (message_not_in_streaming_state); "
                "reopening a fresh card in thread %s", stream.stream_ts, stream.thread_ts)
            stream.stream_ts = ""
            await self._start_native_task_card_stream(client, stream, metadata)
            await client.api_call(
                "chat.appendStream", json={
                    "channel": chat_id, "ts": stream.stream_ts,
                    "chunks": reopen_chunks if reopen_chunks is not None else chunks})

    @staticmethod
    async def _start_native_task_card_stream(
        client, stream: _NativeTaskCardStream, metadata: Optional[dict[str, Any]],
        task_display_mode: str = "plan",
    ) -> None:
        """chat.startStream a card in ``stream``'s thread; sets ``stream.stream_ts``."""
        start_payload: dict[str, Any] = {
            "channel": stream.channel, "thread_ts": stream.thread_ts,
            "task_display_mode": task_display_mode}
        md = metadata or {}
        recipients = (
            ("recipient_team_id", ("recipient_team_id", "team_id", "slack_team_id")),
            ("recipient_user_id", ("recipient_user_id", "user_id")))
        for key, sources in recipients:
            value = _first_truthy(md, sources)
            if value:
                start_payload[key] = value
        result = await client.api_call("chat.startStream", json=start_payload)
        if hasattr(result, "get"):
            stream.stream_ts = str(result.get("ts") or result.get("message_ts") or "")
        if not stream.stream_ts:
            raise RuntimeError("Slack startStream returned no stream timestamp")
        stream.started_at = time.monotonic()
        stream.rows = {}

    @staticmethod
    def _task_update_chunk(task: dict[str, str]) -> dict[str, Any]:
        """One ``task_update`` stream chunk; unknown statuses coerce to ``in_progress``."""
        status = str(task.get("status") or "in_progress")
        status = status if status in {"in_progress", "complete", "error"} else "in_progress"
        task_id = str(task.get("id") or task.get("task_id") or "task")
        chunk = {
            "type": "task_update", "id": task_id, "title": str(task.get("title") or task_id)[:256],
            "status": status}
        if task.get("details"):
            # Slack APPENDS details on every re-send of a row: callers send them once.
            chunk["details"] = str(task["details"])[:2000]
        return chunk

    async def _stop_native_task_card_stream(
        self, key: tuple[str, ...], stream: _NativeTaskCardStream,
        chunks: Optional[list[dict[str, Any]]] = None) -> None:
        async with stream.lock:
            if stream.stopped:
                return
            stream.stopped = True
            try:
                if self._app and stream.stream_ts:
                    payload: dict[str, Any] = {"channel": stream.channel, "ts": stream.stream_ts}
                    if chunks:
                        # chat.stopStream takes chunks: the terminal frame lands with the close.
                        payload["chunks"] = chunks
                    await self._get_client(stream.channel, team_id=stream.team_id).api_call(
                        "chat.stopStream", json=payload)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("[Slack] Native task-card stopStream failed: %s", exc, exc_info=True)
            finally:
                if self._native_task_card_streams.get(key) is stream:
                    self._native_task_card_streams.pop(key, None)

    def set_native_task_card_final_frame(
        self, chat_id: str, *, title: str, tasks: list[dict[str, str]],
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> bool:
        """Stage the turn's terminal frame; the next stop writes it with ``chat.stopStream``.

        Without it the closed card keeps its live title and any row Slack still sees running
        (#141643). Returns whether a live card took it.
        """
        key = self._native_task_card_key(chat_id, reply_to, metadata)
        stream = self._native_task_card_streams.get(key) if key is not None else None
        if stream is None or stream.stopped or not stream.stream_ts:
            return False
        stream.final_chunks = [{"type": "plan_update", "title": str(title)[:256]},
                               *(self._task_update_chunk(task) for task in tasks)]
        return True

    async def stop_native_task_card_progress(
        self, chat_id: str, *, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None) -> None:
        """Finalize an active Slack-native progress stream exactly once."""
        key = self._native_task_card_key(chat_id, reply_to, metadata)
        if key is None:
            return
        stream = self._native_task_card_streams.get(key)
        if stream is not None:
            await self._stop_native_task_card_stream(key, stream, stream.final_chunks or None)

    def _native_task_card_due_for_rotation(self, stream: _NativeTaskCardStream) -> bool:
        limit = _rotate_seconds(self.config.extra)
        return limit > 0 and time.monotonic() - stream.started_at >= limit

    async def _rotate_native_task_card_stream(self, client, stream: _NativeTaskCardStream) -> None:
        """Close the card before Slack's stream lifetime seals it; the caller opens the next one.

        A card Slack seals keeps its last frame forever, running rows included (#141643).
        Closing it ourselves first leaves a terminal frame: a "continued below" title and no
        running row (a running row is settled as complete with a ⤵ marker; its real outcome
        lands on the next card, which receives the full current frame).
        """
        closing: list[dict[str, Any]] = [
            {"type": "plan_update", "title": t("platform.slack.task_card.continued")[:256]}]
        for row in stream.rows.values():
            if row.get("status") == "in_progress":
                row = {**row, "status": "complete", "title": f"{str(row.get('title', ''))[:250]} ⤵"}
            closing.append(row)
        try:
            await client.api_call(
                "chat.stopStream", json={"channel": stream.channel, "ts": stream.stream_ts, "chunks": closing})
        except Exception as exc:
            # Already sealed (or gone): the fresh card still continues the turn.
            logger.info("[Slack] Native task-card rotation could not close %s: %s", stream.stream_ts, exc,
                        exc_info=True)
        logger.info("[Slack] Native task-card stream %s rotated before Slack's stream lifetime", stream.stream_ts)
        stream.stream_ts = ""
