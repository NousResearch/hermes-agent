"""Voice server room gateway adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_SPOKEN_RECONCILE_RETRY_DELAYS = (0.25, 1.0, 2.0)


def check_voice_server_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
        return True
    except Exception:
        return False


class VoiceServerAdapter(BasePlatformAdapter):
    """Subscribe to a Voice server room event stream and dispatch voice turns."""

    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.VOICE_SERVER)
        self.url = str(config.extra.get("url") or "").strip()
        self.room_id = str(config.extra.get("room_id") or "default").strip() or "default"
        self.room_name = str(config.extra.get("room_name") or self.room_id).strip() or self.room_id
        self.event_path = str(config.extra.get("event_path") or "/events").strip() or "/events"
        self.multi_room = is_truthy_value(config.extra.get("multi_room"), default=False)
        self.reconnect_delay_seconds = float(config.extra.get("reconnect_delay_seconds") or 2.0)
        self._session: Any = None
        self._ws: Any = None
        self._listen_task: asyncio.Task | None = None
        self._turns: dict[str, dict[str, Any]] = {}
        self._stream_turns: dict[str, dict[str, Any]] = {}
        self._pending_stream_spoken: dict[str, dict[str, Any]] = {}
        self._spoken_reconcile_tasks: dict[str, asyncio.Task] = {}
        self._next_reply_turn_ids: dict[str, str] = {}
        self._held_pending_spoken_turns: set[str] = set()
        # (room_id, call_id) pairs of outbound calls Hermes initiated.
        # Each entry pre-authorizes the matching ``call_started``
        # confirmation (the outbound call was already authorized by the
        # agent turn that requested it). The stored value is the
        # ``time.monotonic()`` timestamp at registration so entries can
        # be expired after a short confirmation window, preventing a
        # never-confirmed outbound id from pre-authorizing an unrelated
        # later ``call_started``. Entries are also consumed on first
        # matching binding and the structure is LRU-bounded so a
        # long-lived adapter cannot grow without bound.
        self._pending_outbound_calls: "OrderedDict[tuple[str, str], float]" = OrderedDict()
        self._pending_outbound_calls_limit = 1024
        self._pending_outbound_ttl_seconds = 300.0

    def _target_room_or_error(self, chat_id: str) -> tuple[str, SendResult | None]:
        target_room = str(chat_id or self.room_id)
        if not self.multi_room and target_room != self.room_id:
            return target_room, SendResult(
                success=False,
                error=f"Voice server room '{target_room}' is not configured for this adapter",
            )
        return target_room, None

    def _turn_id_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        if metadata_dict.get("turn_id"):
            return str(metadata_dict["turn_id"])
        if metadata_dict.get("_voice_server_turn_id"):
            return str(metadata_dict["_voice_server_turn_id"])
        turn_id = f"voice_server-{uuid.uuid4().hex}"
        if isinstance(metadata, dict):
            metadata["_voice_server_turn_id"] = turn_id
        return turn_id

    def _call_id_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        return str(metadata_dict.get("call_id") or metadata_dict.get("thread_id") or "").strip()

    def set_next_reply_turn_id(self, session_key: str, turn_id: str) -> None:
        session_key = str(session_key or "").strip()
        turn_id = str(turn_id or "").strip()
        if session_key and turn_id:
            self._next_reply_turn_ids[session_key] = turn_id

    def clear_next_reply_turn_id(self, session_key: str) -> None:
        session_key = str(session_key or "").strip()
        if session_key:
            self._next_reply_turn_ids.pop(session_key, None)

    def _metadata_with_next_reply_turn_id(self, metadata: Any) -> Any:
        if not isinstance(metadata, dict):
            return metadata
        if metadata.get("turn_id") or metadata.get("_voice_server_turn_id"):
            return metadata
        session_key = str(metadata.get("_hermes_session_key") or "").strip()
        turn_id = self._next_reply_turn_ids.pop(session_key, "") if session_key else ""
        if not turn_id:
            return metadata
        enriched = dict(metadata)
        enriched["turn_id"] = turn_id
        return enriched

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Any = None,
        max_retries: int = 2,
        base_delay: float = 2.0,
    ) -> SendResult:
        return await super()._send_with_retry(
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            metadata=self._metadata_with_next_reply_turn_id(metadata),
            max_retries=max_retries,
            base_delay=base_delay,
        )

    async def _send_room_payload(self, payload: dict[str, Any], failure_prefix: str) -> SendResult:
        ws = self._ws
        if ws is None or getattr(ws, "closed", False):
            return SendResult(success=False, error="Voice server room websocket is not connected")
        try:
            await ws.send_json(payload)
        except Exception as exc:
            return SendResult(
                success=False,
                error=f"{failure_prefix}: {exc}",
                retryable=True,
            )
        return SendResult(success=True)

    async def start_assistant_stream(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        target_room, error = self._target_room_or_error(chat_id)
        if error:
            return error

        metadata_dict = metadata if isinstance(metadata, dict) else {}
        turn_id = self._turn_id_from_metadata(metadata)
        payload: dict[str, Any] = {
            "type": "assistant_llm_start",
            "room_id": target_room,
            "turn_id": turn_id,
            "seq": 0,
        }
        if metadata_dict.get("participant_id"):
            payload["participant_id"] = str(metadata_dict["participant_id"])
        call_id = self._call_id_from_metadata(metadata_dict)
        if call_id:
            payload["call_id"] = call_id

        result = await self._send_room_payload(payload, "Voice server assistant stream start failed")
        if not result.success:
            return result

        stream_turn = {
            "room_id": target_room,
            "planned_parts": [],
            "seq": 0,
            "session_key": str(metadata_dict.get("_hermes_session_key") or ""),
            "session_id": str(metadata_dict.get("_hermes_session_id") or ""),
            "participant_id": str(metadata_dict.get("participant_id") or ""),
            "call_id": call_id,
        }
        self._stream_turns[turn_id] = stream_turn
        return SendResult(success=True)

    async def push_assistant_delta(
        self,
        chat_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not text:
            return SendResult(success=True)
        target_room, error = self._target_room_or_error(chat_id)
        if error:
            return error

        if not isinstance(metadata, dict):
            metadata = {}
        metadata_dict = metadata
        turn_id = self._turn_id_from_metadata(metadata)
        stream_turn = self._stream_turns.get(turn_id)
        if stream_turn is None:
            start = await self.start_assistant_stream(chat_id, metadata=metadata)
            if not start.success:
                return start
            stream_turn = self._stream_turns.get(turn_id)
        if stream_turn is None:
            return SendResult(success=False, error="Voice server assistant stream did not start")

        stream_turn["seq"] = int(stream_turn.get("seq") or 0) + 1
        payload: dict[str, Any] = {
            "type": "assistant_llm_text",
            "room_id": target_room,
            "turn_id": turn_id,
            "seq": stream_turn["seq"],
            "text": text,
        }
        participant_id = metadata_dict.get("participant_id") or stream_turn.get("participant_id")
        if participant_id:
            payload["participant_id"] = str(participant_id)
        call_id = self._call_id_from_metadata(metadata_dict) or str(stream_turn.get("call_id") or "")
        if call_id:
            payload["call_id"] = call_id

        result = await self._send_room_payload(payload, "Voice server assistant stream text failed")
        if result.success:
            stream_turn.setdefault("planned_parts", []).append(text)
        return result

    async def end_assistant_stream(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        target_room, error = self._target_room_or_error(chat_id)
        if error:
            return error

        metadata_dict = metadata if isinstance(metadata, dict) else {}
        turn_id = self._turn_id_from_metadata(metadata)
        stream_turn = self._stream_turns.get(turn_id)
        seq = int(stream_turn.get("seq") or 0) + 1 if stream_turn else 0
        payload: dict[str, Any] = {
            "type": "assistant_llm_end",
            "room_id": target_room,
            "turn_id": turn_id,
            "seq": seq,
        }
        participant_id = metadata_dict.get("participant_id") or (stream_turn or {}).get("participant_id")
        if participant_id:
            payload["participant_id"] = str(participant_id)
        call_id = self._call_id_from_metadata(metadata_dict) or str((stream_turn or {}).get("call_id") or "")
        if call_id:
            payload["call_id"] = call_id

        result = await self._send_room_payload(payload, "Voice server assistant stream end failed")
        if not result.success:
            self._preserve_or_drop_stream_turn(turn_id, target_room, reconcile_pending=True)
            return result
        if result.success and stream_turn is not None:
            planned_text = "".join(stream_turn.get("planned_parts") or [])
            self._stream_turns.pop(turn_id, None)
            self._turns[turn_id] = {
                "room_id": target_room,
                "planned_text": planned_text,
                "session_key": str(stream_turn.get("session_key") or ""),
                "session_id": str(stream_turn.get("session_id") or ""),
                "call_id": str(stream_turn.get("call_id") or ""),
            }
            self._cap_turn_cache()
            pending_spoken = self._pending_stream_spoken.get(turn_id)
            if pending_spoken is not None:
                self._handle_spoken_turn(pending_spoken)
        return result

    async def abort_assistant_stream(
        self,
        chat_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        target_room, error = self._target_room_or_error(chat_id)
        if error:
            return error

        metadata_dict = metadata if isinstance(metadata, dict) else {}
        turn_id = self._turn_id_from_metadata(metadata)
        stream_turn = self._stream_turns.get(turn_id)
        seq = int(stream_turn.get("seq") or 0) + 1 if stream_turn else 0
        payload: dict[str, Any] = {
            "type": "assistant_llm_abort",
            "room_id": target_room,
            "turn_id": turn_id,
            "seq": seq,
        }
        participant_id = metadata_dict.get("participant_id") or (stream_turn or {}).get("participant_id")
        if participant_id:
            payload["participant_id"] = str(participant_id)
        call_id = self._call_id_from_metadata(metadata_dict) or str((stream_turn or {}).get("call_id") or "")
        if call_id:
            payload["call_id"] = call_id

        result = await self._send_room_payload(payload, "Voice server assistant stream abort failed")
        self._preserve_or_drop_stream_turn(turn_id, target_room, reconcile_pending=False)
        return result

    def cleanup_assistant_stream(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        turn_id = self._turn_id_from_metadata(metadata)
        if turn_id:
            self._preserve_or_drop_stream_turn(turn_id, "")

    def _preserve_or_drop_stream_turn(
        self,
        turn_id: str,
        room_id: str,
        *,
        reconcile_pending: bool = False,
    ) -> None:
        stream_turn = self._stream_turns.pop(turn_id, None)
        if not stream_turn:
            self._pending_stream_spoken.pop(turn_id, None)
            return
        planned_parts = stream_turn.get("planned_parts") or []
        if not planned_parts:
            self._pending_stream_spoken.pop(turn_id, None)
            return
        self._turns[turn_id] = {
            "room_id": room_id or str(stream_turn.get("room_id") or ""),
            "planned_text": "".join(planned_parts),
            "session_key": str(stream_turn.get("session_key") or ""),
            "session_id": str(stream_turn.get("session_id") or ""),
            "call_id": str(stream_turn.get("call_id") or ""),
        }
        self._cap_turn_cache()
        if not reconcile_pending:
            # Synchronous cleanup may run from the agent worker thread. Leave
            # any pending spoken receipt queued; the gateway drains it after
            # transcript persistence from the event-loop path.
            return
        pending_spoken = self._pending_stream_spoken.get(turn_id)
        if pending_spoken is not None:
            self._handle_spoken_turn(pending_spoken)

    def _events_url(self) -> str:
        if self.url.startswith(("ws://", "wss://")):
            return self.url
        parsed = urlsplit(self.url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = self.event_path if self.event_path.startswith("/") else f"/{self.event_path}"
        return urlunsplit((scheme, parsed.netloc, path, "", ""))

    async def connect(self) -> bool:
        if not self.url:
            self._set_fatal_error("missing_url", "Voice server room URL is not configured", retryable=False)
            return False
        if not check_voice_server_requirements():
            self._set_fatal_error("missing_dependency", "aiohttp is required for Voice server gateway", retryable=False)
            return False

        import aiohttp

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(self._events_url())
            await self._ws.send_json({"type": "start_bot", "room_id": self.room_id})
        except Exception as exc:
            await self.disconnect()
            self._set_fatal_error("connect_failed", f"Voice server room connection failed: {exc}", retryable=True)
            return False

        self._listen_task = asyncio.create_task(self._listen_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        task = self._listen_task
        self._listen_task = None
        if task and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        reconcile_tasks = list(self._spoken_reconcile_tasks.values())
        self._spoken_reconcile_tasks.clear()
        for reconcile_task in reconcile_tasks:
            reconcile_task.cancel()
        if reconcile_tasks:
            await asyncio.gather(*reconcile_tasks, return_exceptions=True)
        self._stream_turns.clear()
        self._pending_stream_spoken.clear()
        self._held_pending_spoken_turns.clear()
        ws = self._ws
        self._ws = None
        if ws is not None and not getattr(ws, "closed", False):
            await ws.close()
        session = self._session
        self._session = None
        if session is not None and not getattr(session, "closed", False):
            await session.close()
        self._mark_disconnected()

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        """Voice server owns TTS; gateway auto-TTS would double-speak room replies."""
        return False

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        return SendResult(success=True)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return SendResult(success=True)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        ws = self._ws
        if ws is None or getattr(ws, "closed", False):
            return SendResult(success=False, error="Voice server room websocket is not connected")
        target_room = str(chat_id or self.room_id)
        if not self.multi_room and target_room != self.room_id:
            return SendResult(
                success=False,
                error=f"Voice server room '{target_room}' is not configured for this adapter",
            )

        payload: dict[str, Any] = {
            "type": "assistant_reply",
            "room_id": target_room,
            "text": content,
        }
        if reply_to:
            payload["reply_to"] = str(reply_to)
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        if metadata_dict:
            if metadata_dict.get("turn_id"):
                payload["turn_id"] = str(metadata_dict["turn_id"])
            elif metadata_dict.get("_voice_server_turn_id"):
                payload["turn_id"] = str(metadata_dict["_voice_server_turn_id"])
            if metadata_dict.get("participant_id"):
                payload["participant_id"] = str(metadata_dict["participant_id"])
            call_id = self._call_id_from_metadata(metadata_dict)
            if call_id:
                payload["call_id"] = call_id
        if not payload.get("turn_id"):
            payload["turn_id"] = f"voice_server-{uuid.uuid4().hex}"
            if isinstance(metadata, dict):
                metadata_dict["_voice_server_turn_id"] = payload["turn_id"]
        turn_id = str(payload.get("turn_id") or "")
        if turn_id:
            self._turns[turn_id] = {
                "room_id": target_room,
                "planned_text": content,
                "session_key": str(metadata_dict.get("_hermes_session_key") or ""),
                "session_id": str(metadata_dict.get("_hermes_session_id") or ""),
            }
            self._cap_turn_cache()
        try:
            await ws.send_json(payload)
        except Exception as exc:
            if turn_id:
                self._turns.pop(turn_id, None)
            return SendResult(
                success=False,
                error=f"Voice server room websocket send failed: {exc}",
                retryable=True,
            )
        return SendResult(success=True)

    async def start_outbound_call(
        self,
        *,
        target: str | dict[str, Any],
        room_id: str | None = None,
        context: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> SendResult:
        ws = self._ws
        if ws is None or getattr(ws, "closed", False):
            return SendResult(success=False, error="Voice server room websocket is not connected")
        target_room = str(room_id or self.room_id)
        if not self.multi_room and target_room != self.room_id:
            return SendResult(
                success=False,
                error=f"Voice server room '{target_room}' is not configured for this adapter",
            )

        resolved_call_id = call_id or f"voice-call-{uuid.uuid4().hex}"
        payload: dict[str, Any] = {
            "type": "start_outbound_call",
            "room_id": target_room,
            "call_id": resolved_call_id,
            "target": target,
        }
        if context:
            payload["context"] = context
        if metadata:
            payload["metadata"] = metadata
        # Register BEFORE awaiting send_json: a fast voice runtime can deliver
        # ``call_started`` while we are still suspended in send_json, and the
        # confirmation must see this entry to bypass the inbound allowlist.
        self._register_pending_outbound_call(target_room, resolved_call_id)
        try:
            await ws.send_json(payload)
        except Exception as exc:
            # Discard the pre-registered entry so a stale id cannot bypass
            # auth on some later spurious call_started.
            self._consume_pending_outbound_call(target_room, resolved_call_id)
            return SendResult(
                success=False,
                error=f"Voice server outbound call command failed: {exc}",
                retryable=True,
            )
        return SendResult(success=True)

    def _register_pending_outbound_call(self, room_id: str, call_id: str) -> None:
        if not room_id or not call_id:
            return
        key = (str(room_id), str(call_id))
        now = time.monotonic()
        self._prune_expired_pending_outbound_calls(now)
        self._pending_outbound_calls[key] = now
        self._pending_outbound_calls.move_to_end(key)
        while len(self._pending_outbound_calls) > self._pending_outbound_calls_limit:
            self._pending_outbound_calls.popitem(last=False)

    def _consume_pending_outbound_call(self, room_id: str, call_id: str) -> bool:
        if not room_id or not call_id:
            return False
        key = (str(room_id), str(call_id))
        if key not in self._pending_outbound_calls:
            return False
        registered_at = self._pending_outbound_calls.pop(key)
        # Expired entries do not pre-authorize: the outbound call sat too
        # long without a confirmation, so any later call_started reusing
        # that id must go through the authorizer like an inbound event.
        if time.monotonic() - registered_at > self._pending_outbound_ttl_seconds:
            return False
        return True

    def _prune_expired_pending_outbound_calls(self, now: float) -> None:
        threshold = now - self._pending_outbound_ttl_seconds
        # OrderedDict keeps insertion order; oldest entries are at the front,
        # so we can stop as soon as we find a non-expired entry.
        while self._pending_outbound_calls:
            oldest_key, oldest_ts = next(iter(self._pending_outbound_calls.items()))
            if oldest_ts < threshold:
                del self._pending_outbound_calls[oldest_key]
            else:
                break

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        room_id = str(chat_id or self.room_id)
        return {"name": self.room_name if room_id == self.room_id else room_id, "type": "channel"}

    async def _listen_loop(self) -> None:
        try:
            async for message in self._ws:
                data = getattr(message, "data", None)
                if data is None:
                    continue
                try:
                    payload = json.loads(data) if isinstance(data, str) else data
                except Exception:
                    logger.debug("[%s] Ignoring non-JSON Voice server event", self.name)
                    continue
                if isinstance(payload, dict):
                    await self.handle_room_event(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[%s] Voice server room listener failed: %s", self.name, exc, exc_info=True)
            await self._handle_listener_disconnect(
                "listener_failed",
                f"Voice server listener failed: {exc}",
            )
        else:
            if self._running:
                await self._handle_listener_disconnect(
                    "connection_closed",
                    "Voice server room websocket closed",
                )

    async def _handle_listener_disconnect(self, code: str, message: str) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None and not getattr(ws, "closed", False):
            try:
                await ws.close()
            except Exception:
                logger.debug("[%s] Failed to close Voice server websocket", self.name, exc_info=True)
        session = self._session
        self._session = None
        if session is not None and not getattr(session, "closed", False):
            try:
                await session.close()
            except Exception:
                logger.debug("[%s] Failed to close Voice server HTTP session", self.name, exc_info=True)
        self._set_fatal_error(code, message, retryable=True)
        await self._notify_fatal_error()

    def _call_id_from_payload(self, payload: dict[str, Any]) -> str:
        return str(
            payload.get("call_id")
            or payload.get("callId")
            or payload.get("session_call_id")
            or ""
        ).strip()

    async def _create_fresh_call_session(self, payload: dict[str, Any], direction: str) -> None:
        source = self.event_to_session_source(payload)
        call_id = self._call_id_from_payload(payload)
        # Outbound calls were initiated by Hermes via start_outbound_call(),
        # which is invoked by an already-authorized agent turn. The matching
        # call_started confirmation may omit caller identity entirely, so
        # gate the authorizer on inbound traffic only and on outbound traffic
        # whose (room_id, call_id) is not in the pending-outbound ledger.
        # ``_consume_pending_outbound_call`` removes the entry as part of the
        # check, so the same id cannot bypass the allowlist twice and cannot
        # be reused after the call binds.
        is_known_outbound = direction == "outbound" and self._consume_pending_outbound_call(
            source.chat_id, call_id
        )
        if not is_known_outbound:
            authorizer = getattr(self, "_authorizer", None)
            if callable(authorizer) and not authorizer(source):
                logger.info(
                    "[%s] Ignoring %s call lifecycle event from unauthorized caller user_id=%s room=%s",
                    self.name,
                    direction,
                    source.user_id,
                    source.chat_id,
                )
                return
        session_store = getattr(self, "_session_store", None)
        if session_store is None:
            return
        try:
            entry = session_store.get_or_create_session(source, force_new=not bool(call_id))
        except Exception as exc:
            logger.debug("[%s] Failed creating %s voice call session: %s", self.name, direction, exc)
            return

        session_id = str(getattr(entry, "session_id", "") or "").strip()
        session_key = str(getattr(entry, "session_key", "") or "").strip()
        if not session_key:
            session_key = self._session_key_for_source(source)
        if not session_id and not session_key:
            return

        binding: dict[str, Any] = {
            "type": "session_bound",
            "room_id": source.chat_id,
            "direction": direction,
        }
        if call_id:
            binding["session_call_id"] = call_id
            binding["call_id"] = call_id
        if session_id:
            binding["session_id"] = session_id
        if session_key:
            binding["session_key"] = session_key
        caller = payload.get("caller") if isinstance(payload.get("caller"), dict) else {}
        binding["caller"] = {
            "id": source.user_id,
            "name": source.user_name or source.user_id,
        }
        if caller:
            binding["caller"].update(
                {
                    "id": str(caller.get("id") or caller.get("caller_id") or source.user_id),
                    "name": str(caller.get("name") or caller.get("display_name") or source.user_name or source.user_id),
                }
            )
        result = await self._send_room_payload(binding, "Voice server session binding failed")
        if not result.success:
            logger.debug("[%s] %s", self.name, result.error)

    async def handle_room_event(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type") or payload.get("event") or "").strip().lower()
        room_id = str(payload.get("room_id") or self.room_id).strip() or self.room_id
        if not self.multi_room and room_id != self.room_id:
            logger.warning(
                "[%s] Ignoring Voice server event for unconfigured room %s",
                self.name,
                room_id,
            )
            return
        if event_type in {"inbound_call", "start_inbound_call"}:
            await self._create_fresh_call_session(payload, "inbound")
            return
        if event_type in {"call_started", "outbound_call_started"}:
            await self._create_fresh_call_session(payload, "outbound")
            return
        if event_type in {"transcript", "user_transcript"}:
            event = self.event_to_message_event(payload)
            if event is not None:
                await self.handle_message(event)
            return
        if event_type == "assistant_spoken":
            logger.info(
                "[%s] assistant_spoken room=%s turn=%s interrupted=%s",
                self.name,
                payload.get("room_id") or self.room_id,
                payload.get("turn_id") or "",
                is_truthy_value(payload.get("interrupted"), default=False),
            )
            self._handle_spoken_turn(payload)
            return
        if event_type == "error":
            logger.warning("[%s] Voice server room error: %s", self.name, payload.get("message") or payload)

    def event_to_message_event(self, payload: dict[str, Any]) -> MessageEvent | None:
        text = str(payload.get("text") or payload.get("transcript") or "").strip()
        if not text:
            return None

        source = self.event_to_session_source(payload)
        return MessageEvent(
            text=text,
            message_type=MessageType.VOICE,
            source=source,
            raw_message=payload,
            message_id=str(payload.get("turn_id") or payload.get("message_id") or "") or None,
            auto_skill="talk",
        )

    def event_to_session_source(self, payload: dict[str, Any]):
        room_id = str(payload.get("room_id") or self.room_id).strip() or self.room_id
        room_name = str(payload.get("room_name") or self.room_name or room_id).strip() or room_id
        caller = payload.get("caller") if isinstance(payload.get("caller"), dict) else {}
        participant_id = str(
            payload.get("participant_id")
            or payload.get("caller_id")
            or caller.get("id")
            or caller.get("caller_id")
            or payload.get("user_id")
            or "caller"
        ).strip() or "caller"
        participant_name = str(
            payload.get("participant_name")
            or payload.get("caller_name")
            or caller.get("name")
            or caller.get("display_name")
            or payload.get("user_name")
            or participant_id
        ).strip() or participant_id
        call_id = str(
            payload.get("call_id")
            or payload.get("callId")
            or payload.get("session_call_id")
            or ""
        ).strip()

        return self.build_source(
            chat_id=room_id,
            chat_name=room_name,
            chat_type="channel",
            user_id=participant_id,
            user_name=participant_name,
            thread_id=call_id or None,
            message_id=str(payload.get("turn_id") or payload.get("message_id") or "") or None,
        )

    def _cap_turn_cache(self) -> None:
        while len(self._turns) > 256:
            self._turns.pop(next(iter(self._turns)), None)

    def _session_id_for_turn(self, turn: dict[str, Any]) -> str:
        candidates = self._candidate_session_ids_for_turn(turn)
        return candidates[0] if candidates else ""

    def _candidate_session_ids_for_turn(self, turn: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        session_store = getattr(self, "_session_store", None)
        session_key = str(turn.get("session_key") or "").strip()
        if session_store is not None and session_key:
            try:
                entry = getattr(session_store, "_entries", {}).get(session_key)
                current_session_id = str(getattr(entry, "session_id", "") or "").strip()
                if current_session_id:
                    candidates.append(current_session_id)
            except Exception:
                pass
        captured_session_id = str(turn.get("session_id") or "").strip()
        if captured_session_id and captured_session_id not in candidates:
            candidates.append(captured_session_id)
        return candidates

    def _load_transcript_for_voice_turn(
        self,
        session_store: Any,
        session_id: str,
        turn_id: str,
    ) -> list[dict[str, Any]]:
        messages = session_store.load_transcript(session_id)
        if any(str(msg.get("voice_turn_id") or "") == turn_id for msg in messages):
            return messages

        get_transcript_path = getattr(session_store, "get_transcript_path", None)
        if not callable(get_transcript_path):
            return messages
        try:
            transcript_path = get_transcript_path(session_id)
            jsonl_messages: list[dict[str, Any]] = []
            if transcript_path.exists():
                with open(transcript_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            jsonl_messages.append(json.loads(line))
            if any(str(msg.get("voice_turn_id") or "") == turn_id for msg in jsonl_messages):
                return jsonl_messages
        except Exception as exc:
            logger.debug("[%s] Failed loading JSONL transcript for voice turn reconciliation: %s", self.name, exc)
        return messages

    def _handle_spoken_turn(self, payload: dict[str, Any]) -> None:
        turn_id = str(payload.get("turn_id") or "").strip()
        if not turn_id:
            return
        if turn_id in self._stream_turns:
            if is_truthy_value(payload.get("interrupted"), default=False):
                self._pending_stream_spoken[turn_id] = dict(payload)
            return
        if not is_truthy_value(payload.get("interrupted"), default=False):
            self._turns.pop(turn_id, None)
            self._pending_stream_spoken.pop(turn_id, None)
            return

        self._pending_stream_spoken[turn_id] = dict(payload)
        self._reconcile_spoken_turn(payload, consume=False)
        existing = self._spoken_reconcile_tasks.pop(turn_id, None)
        if existing is not None:
            existing.cancel()
        self._spoken_reconcile_tasks[turn_id] = asyncio.create_task(
            self._retry_spoken_turn_reconciliation(dict(payload))
        )

    async def _retry_spoken_turn_reconciliation(self, payload: dict[str, Any]) -> None:
        turn_id = str(payload.get("turn_id") or "").strip()
        reconciled = False
        cancelled = False
        try:
            for delay in _SPOKEN_RECONCILE_RETRY_DELAYS:
                await asyncio.sleep(delay)
                if self._reconcile_spoken_turn(payload, consume=False):
                    reconciled = True
                    break
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            current_task = asyncio.current_task()
            if turn_id and self._spoken_reconcile_tasks.get(turn_id) is current_task:
                self._spoken_reconcile_tasks.pop(turn_id, None)
                if not reconciled and not cancelled:
                    if turn_id in self._held_pending_spoken_turns:
                        return
                    self._turns.pop(turn_id, None)
                    self._pending_stream_spoken.pop(turn_id, None)

    def hold_pending_spoken_turn(self, turn_id: str | None) -> None:
        turn_id = str(turn_id or "").strip()
        if turn_id:
            self._held_pending_spoken_turns.add(turn_id)

    def release_pending_spoken_turn(self, turn_id: str | None) -> None:
        turn_id = str(turn_id or "").strip()
        if not turn_id:
            return
        was_held = turn_id in self._held_pending_spoken_turns
        self._held_pending_spoken_turns.discard(turn_id)
        if (
            was_held
            and turn_id in self._pending_stream_spoken
            and turn_id not in self._spoken_reconcile_tasks
        ):
            self._spoken_reconcile_tasks[turn_id] = asyncio.create_task(
                self._retry_spoken_turn_reconciliation(dict(self._pending_stream_spoken[turn_id]))
            )

    def drain_pending_spoken(
        self,
        *,
        turn_id: str | None = None,
        session_key: str | None = None,
        session_id: str | None = None,
    ) -> None:
        target_turn_id = str(turn_id or "").strip()
        target_session_key = str(session_key or "").strip()
        target_session_id = str(session_id or "").strip()
        for pending_turn_id, payload in list(self._pending_stream_spoken.items()):
            if target_turn_id and pending_turn_id != target_turn_id:
                continue
            turn = self._turns.get(pending_turn_id)
            if not turn:
                continue
            if target_session_key and str(turn.get("session_key") or "").strip() != target_session_key:
                continue
            if target_session_id and target_session_id not in self._candidate_session_ids_for_turn(turn):
                continue
            if self._reconcile_spoken_turn(payload, consume=False):
                task = self._spoken_reconcile_tasks.pop(pending_turn_id, None)
                if task is not None:
                    task.cancel()

    def _reconcile_spoken_turn(self, payload: dict[str, Any], *, consume: bool = True) -> bool:
        turn_id = str(payload.get("turn_id") or "").strip()
        if not turn_id:
            return False
        turn = self._turns.pop(turn_id, None) if consume else self._turns.get(turn_id)
        if not turn or not is_truthy_value(payload.get("interrupted"), default=False):
            return False

        session_store = getattr(self, "_session_store", None)
        if session_store is None:
            return False
        candidate_session_ids = self._candidate_session_ids_for_turn(turn)
        if not candidate_session_ids:
            return False

        spoken_text = str(payload.get("spoken_text") or "").strip()
        session_id = ""
        messages: list[dict[str, Any]] | None = None
        for candidate_session_id in candidate_session_ids:
            try:
                candidate_messages = self._load_transcript_for_voice_turn(
                    session_store,
                    candidate_session_id,
                    turn_id,
                )
            except Exception as exc:
                logger.debug("[%s] Failed loading transcript for spoken turn reconciliation: %s", self.name, exc)
                continue
            if any(
                msg.get("role") == "assistant"
                and str(msg.get("voice_turn_id") or "") == turn_id
                for msg in candidate_messages
            ):
                session_id = candidate_session_id
                messages = candidate_messages
                break
        if messages is None:
            logger.debug(
                "[%s] Skipping spoken turn reconciliation; turn id did not match transcript",
                self.name,
            )
            return False

        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") != "assistant":
                continue
            if str(messages[idx].get("voice_turn_id") or "") == turn_id:
                assistant_idx = idx
                break
        if assistant_idx is None:
            logger.debug(
                "[%s] Skipping spoken turn reconciliation; turn id did not match transcript",
                self.name,
            )
            return False

        original = messages[assistant_idx]
        if spoken_text:
            messages[assistant_idx] = {
                **original,
                "content": spoken_text,
                "voice_turn_id": turn_id,
                "voice_interrupted": True,
                "voice_planned_content": original.get("voice_planned_content") or original.get("content"),
                "voice_spoken_content": spoken_text,
            }
        else:
            del messages[assistant_idx]

        try:
            session_store.rewrite_transcript(session_id, messages)
        except Exception as exc:
            logger.debug("[%s] Failed rewriting interrupted spoken turn: %s", self.name, exc)
            return False
        self._turns.pop(turn_id, None)
        self._pending_stream_spoken.pop(turn_id, None)
        return True
