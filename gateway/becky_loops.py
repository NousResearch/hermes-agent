"""Private, loopback-only JSON-RPC bridge for Becky Telegram loops.

The bridge deliberately lives at the gateway edge.  It exposes a small,
closed contract to the Becky dashboard without exposing Hermes sessions,
transcript identifiers, prompts, or arbitrary gateway RPCs.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.becky_loop_summarizer import (
    AsyncAuxiliarySummaryProvider,
    LoopSummary,
    LoopSummarizer,
    SummaryUnavailable,
    _TOOL_ENVELOPE_PATTERN,
    _ConversationTooLarge,
    _SummaryValidationError,
    _is_tool_result_json,
    _is_visible_entry,
    _parse_timestamp,
    _remove_embedded_tool_result_json,
)
from gateway.becky_loop_reply import (
    AsyncAuxiliaryReplyProvider,
    LoopReplyGenerator,
    ReplyGenerator,
)
from gateway.telegram_mtproto import MtprotoTopicControlError
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.http11 import Headers, Request, Response

logger = logging.getLogger(__name__)
_WEBSOCKET_LOGGER = logging.getLogger("gateway.becky_loops.websocket")
_WEBSOCKET_LOGGER.setLevel(logging.WARNING)
_WEBSOCKET_LOGGER.propagate = False

try:
    from agent.redact import redact_sensitive_text as _force_redact
except Exception:  # pragma: no cover - only applies to an incomplete Hermes install
    _force_redact = None

_READY = {
    "jsonrpc": "2.0",
    "method": "event",
    "params": {"type": "gateway.ready", "payload": {"skin": {}}},
}
_METHODS = ["list", "summarize", "close", "reopen", "reply", "reply_retry"]
_SAFE_REMOTE_CODES = frozenset({
    "conversation_too_large",
    "idempotency_conflict",
    "revision_conflict",
    "reply_retry_unavailable",
    "reply_send_failed",
    "same_topic_reopen_unsupported",
    "source_not_found",
    "successor_already_exists",
    "successor_creation_failed",
    "successor_creation_incomplete",
    "summary_invalid",
    "summary_timeout",
    "topic_already_closed",
    "topic_already_open",
    "topic_control_unavailable",
    "topic_control_unsupported",
    "topic_reply_unavailable",
    "topic_not_found",
    "topic_state_read_failed",
    "topic_state_write_failed",
})
_SOURCE_REF_RE = re.compile(r"^loop_[A-Za-z0-9_-]{43}$")
_REVISION_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
_MAX_REQUEST_BYTES = 65_536
_MAX_RESPONSE_BYTES = 262_144
_SUMMARY_DEADLINE_SECONDS = 30.0
_REPLY_DEADLINE_SECONDS = 30.0
_REPLY_ATTEMPT_TTL_SECONDS = 15 * 60.0
_MAX_REPLY_ATTEMPTS = 256
_MAX_REPLY_TEXT_CHARS = 2_000
_MAX_REPLY_COMMENT_CHARS = 5_000
_MAX_NEW_TOPIC_ANSWERS = 256
_SESSION_PAGE_SIZE = 200
_MAX_SESSION_SCAN = 10_000
_TELEGRAM_ID_RE = re.compile(r"^-?\d+$")
_POSITIVE_TELEGRAM_ID_RE = re.compile(r"^[1-9]\d{0,19}$")


@dataclass(frozen=True)
class BeckyLoopsConfig:
    enabled: bool
    chat_id: str
    token: str
    port: int = 9_120
    topic_control: str = "unavailable"
    topic_reply: str = "unavailable"


@dataclass(frozen=True)
class TopicSendReceipt:
    message_id: str


class TopicSender(Protocol):
    async def send_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        text: str,
        reply_to_message_id: str | None,
    ) -> TopicSendReceipt: ...


class TopicController(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def supports_close(self) -> bool: ...

    async def close_topic(self, *, chat_id: str, thread_id: str) -> datetime: ...


class TelegramTopicSender:
    """Guard the existing Telegram adapter behind one private-topic seam."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    @property
    def is_connected(self) -> bool:
        """Reflect the live adapter state without exposing the adapter."""
        state = getattr(self._adapter, "is_connected", None)
        if state is None:
            # Test seams and lightweight adapters may not expose a state
            # property. Their presence in GatewayRunner.adapters is the
            # connected proof, so preserve that contract here.
            return True
        try:
            return bool(state() if callable(state) else state)
        except Exception:
            return False

    async def send_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        text: str,
        reply_to_message_id: str | None,
    ) -> TopicSendReceipt:
        if not self.is_connected:
            raise _TopicSendFailure()
        label = "Becky:" if reply_to_message_id is not None else "Cory via Becky:"
        metadata = {
            "thread_id": thread_id,
            "notify": True,
        }
        # Telegram forum/supergroup topics require message_thread_id, which
        # Hermes derives from metadata.thread_id. direct_messages_topic_id is
        # only valid for positive private-chat topic lanes; including it for a
        # -100... forum chat causes the adapter to omit message_thread_id and
        # silently deliver into General.
        try:
            if int(chat_id) > 0:
                metadata["direct_messages_topic_id"] = thread_id
        except (TypeError, ValueError):
            pass
        try:
            result = await self._adapter.send(
                chat_id=chat_id,
                content=f"{label} {text}",
                reply_to=reply_to_message_id,
                metadata=metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise _TopicSendFailure() from None
        raw_response = getattr(result, "raw_response", None)
        message_id = str(getattr(result, "message_id", "") or "").strip()
        if isinstance(raw_response, dict):
            raw_message_ids = raw_response.get("message_ids")
            if isinstance(raw_message_ids, list):
                message_ids = [
                    item.strip()
                    for item in raw_message_ids
                    if isinstance(item, str) and item.strip()
                ]
                if message_ids:
                    # Telegram splits long comments into multiple messages;
                    # anchor Becky’s answer to the final chunk.
                    message_id = message_ids[-1]
        if (
            not bool(getattr(result, "success", False))
            or not message_id
            or isinstance(raw_response, dict)
            and bool(raw_response.get("thread_fallback"))
        ):
            raise _TopicSendFailure()
        return TopicSendReceipt(message_id=message_id)


class TelegramTopicController:
    """Small, Bot API-only seam for closing a proven Telegram topic."""

    method = "bot_api_private_topic"

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    @property
    def is_connected(self) -> bool:
        state = getattr(self._adapter, "is_connected", None)
        if state is None:
            return False
        try:
            return bool(state() if callable(state) else state)
        except Exception:
            return False

    @property
    def supports_close(self) -> bool:
        bot = getattr(self._adapter, "_bot", None)
        return callable(getattr(bot, "close_forum_topic", None))

    async def close_topic(self, *, chat_id: str, thread_id: str) -> datetime:
        if not self.is_connected:
            raise _TopicControlFailure("topic_control_unavailable")
        bot = getattr(self._adapter, "_bot", None)
        method = getattr(bot, "close_forum_topic", None)
        if not callable(method):
            raise _TopicControlFailure("topic_control_unsupported")
        try:
            result = await method(
                chat_id=int(chat_id), message_thread_id=int(thread_id)
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = _classify_topic_control_error(error)
            if code == "topic_already_closed":
                return datetime.now(UTC)
            raise _TopicControlFailure(code) from None
        if result is False:
            raise _TopicControlFailure("topic_control_unavailable")
        return datetime.now(UTC)


@dataclass
class _ReplyAttempt:
    source_ref: str
    expected_revision: str
    comment: str
    thread_id: str
    expires_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    comment_message_id: str | None = None
    comment_sent_at: str | None = None
    answer: str | None = None
    answer_generated_at: str | None = None
    answer_sent_at: str | None = None
    state: str = "comment_pending"
    in_progress: bool = True


class BeckyLoopsStore(Protocol):
    def list_topics(self, chat_id: str) -> list[dict[str, Any]]: ...

    def get_topic(self, source_ref: str) -> dict[str, Any] | None: ...

    def transcript(self, session_id: str) -> list[dict[str, Any]]: ...

    def revision_for_topic(
        self, row: dict[str, Any], transcript: list[dict[str, Any]]
    ) -> str: ...


def _utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.fromtimestamp(0, tz=UTC)


def _bounded_text(value: Any, limit: int = 500) -> str:
    text = _redact(str(value or ""))
    text = " ".join(text.split()).strip()
    return text[:limit].rstrip()


def _safe_public_text(value: Any, hidden_values: set[str], limit: int = 500) -> str:
    """Bound text while removing known Telegram/session identifiers."""
    text = _redact(str(value or ""))
    if _force_redact is None:
        return "[REDACTED]" if text else ""
    try:
        text = _force_redact(text, force=True)
    except Exception:
        return "[REDACTED]" if text else ""
    for hidden in sorted(
        (item for item in hidden_values if len(item) >= 3), key=len, reverse=True
    ):
        if hidden.isdigit():
            pattern = rf"(?<!\d){re.escape(hidden)}(?!\d)"
        else:
            pattern = rf"(?<![A-Za-z0-9_-]){re.escape(hidden)}(?![A-Za-z0-9_-])"
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip()
    return text[:limit].rstrip()


def _redact(value: str) -> str:
    text = value
    text = re.sub(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b", "[REDACTED]", text)
    text = re.sub(r"\b(?:gh[pousr]|glpat)-[A-Za-z0-9_-]{16,}\b", "[REDACTED]", text)
    text = re.sub(r"\b\d{8,12}:[A-Za-z0-9_-]{20,}\b", "[REDACTED]", text)
    text = re.sub(
        r"(?i)\b(?:bearer|token|api[_ -]?key|password)\s*[:=]\s*[^\s,;]+",
        "[REDACTED]",
        text,
    )
    return text


def _latest_public_becky_response(
    transcript: list[dict[str, Any]], hidden_values: set[str]
) -> tuple[str | None, datetime | None]:
    """Return the newest safe assistant turn without exposing tool output."""
    for message in reversed(transcript):
        if not _is_visible_entry(message) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        cleaned = _TOOL_ENVELOPE_PATTERN.sub(" ", content)
        cleaned = _remove_embedded_tool_result_json(cleaned).strip()
        if not cleaned or _is_tool_result_json(cleaned):
            continue
        response = _safe_public_text(cleaned, hidden_values, 2_000)
        if not response or response == "[REDACTED]":
            continue
        return response, _parse_timestamp(message.get("timestamp"))
    return None, None


def source_ref_for(*, chat_id: str, thread_id: str) -> str:
    digest = hashlib.sha256(f"telegram\0{chat_id}\0{thread_id}".encode()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return f"loop_{encoded}"


def revision_for(row: dict[str, Any], transcript: list[dict[str, Any]]) -> str:
    stable = {
        "source_ref": row.get("source_ref"),
        "session_id": row.get("session_id"),
        "message_count": row.get("message_count", 0),
        "updated_at": str(row.get("updated_at", "")),
        "messages": [
            {
                "role": message.get("role"),
                "content": _redact(str(message.get("content") or "")),
                "timestamp": message.get("timestamp"),
            }
            for message in transcript
        ],
    }
    encoded = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class SessionDBBeckyLoopsStore:
    """Read-only projection over Hermes's existing SessionDB."""

    def __init__(self, db: Any, *, session_store: Any | None = None) -> None:
        self._db = db
        self._session_store = session_store
        self._source_rows: dict[str, dict[str, Any]] = {}
        self._chat_id = ""

    def list_topics(self, chat_id: str) -> list[dict[str, Any]]:
        self._chat_id = str(chat_id)
        self._source_rows = {}
        rows: list[dict[str, Any]] = []
        offset = 0
        while len(rows) < _MAX_SESSION_SCAN:
            page = self._db.list_sessions_rich(
                source="telegram",
                include_children=False,
                include_archived=False,
                project_compression_tips=True,
                order_by_last_active=True,
                limit=_SESSION_PAGE_SIZE,
                offset=offset,
            )
            rows.extend(page)
            if len(page) < _SESSION_PAGE_SIZE:
                break
            offset += len(page)
        result: list[dict[str, Any]] = []
        for raw in rows:
            if self._hidden_child(raw):
                continue
            if str(raw.get("chat_id") or "") != str(chat_id):
                continue
            thread_id = str(raw.get("thread_id") or "").strip()
            if not thread_id:
                continue
            session_id = str(raw.get("id") or "")
            if not session_id:
                continue
            ref = source_ref_for(chat_id=str(chat_id), thread_id=thread_id)
            transcript = self.transcript(session_id)
            hidden_values = {
                str(chat_id),
                session_id,
                thread_id,
                ref,
            }
            for message in transcript:
                for key in (
                    "id",
                    "platform_message_id",
                    "telegram_message_id",
                    "chat_id",
                    "thread_id",
                    "session_id",
                    "user_id",
                ):
                    value = message.get(key)
                    if value not in (None, ""):
                        hidden_values.add(str(value))
            last_response, last_response_at = _latest_public_becky_response(
                transcript, hidden_values
            )
            revision_input = {**raw, "source_ref": ref}
            source_state = "active" if raw.get("ended_at") is None else "closed"
            item = {
                "source_ref": ref,
                "title": _bounded_text(
                    raw.get("title") or raw.get("preview") or "Telegram loop", 128
                ),
                "source_state": source_state,
                "revision": revision_for(revision_input, transcript),
                "message_count": max(
                    0, int(raw.get("message_count") or len(transcript))
                ),
                "created_at": _utc_datetime(raw.get("started_at")),
                "updated_at": _utc_datetime(
                    raw.get("last_active") or raw.get("started_at")
                ),
                "telegram_url": None,
                "session_id": session_id,
                "thread_id": thread_id,
                "last_becky_response": last_response,
                "last_becky_response_at": last_response_at,
                "_revision_input": revision_input,
            }
            if ref in self._source_rows:
                continue
            self._source_rows[ref] = item
            result.append(item)
        return result

    def get_topic(self, source_ref: str) -> dict[str, Any] | None:
        row = self._source_rows.get(source_ref)
        if row is not None:
            return row
        # Refresh the bounded projection so a call made without list first is
        # still authoritative, while never accepting a caller-supplied ID.
        for item in self.list_topics(self._chat_id):
            if item["source_ref"] == source_ref:
                return item
        return None

    def transcript(self, session_id: str) -> list[dict[str, Any]]:
        return self._db.get_messages(session_id, include_inactive=False)

    def _shortcut_session_id(self, topic_id: str) -> str:
        digest = hashlib.sha256(
            f"telegram\0{self._chat_id}\0{topic_id}".encode()
        ).hexdigest()[:32]
        return f"telegram_shortcut_{digest}"

    def _shortcut_source(self, topic_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=self._chat_id,
            chat_type="group",
            thread_id=topic_id,
        )

    def _set_shortcut_title(self, session_id: str, title: str) -> None:
        set_title = getattr(self._db, "set_session_title", None)
        if not callable(set_title):
            return
        try:
            set_title(session_id, title)
        except ValueError:
            for suffix in range(2, 100):
                try:
                    if set_title(session_id, f"{title} ({suffix})"):
                        break
                except ValueError:
                    continue

    def _shortcut_has_message(self, session_id: str, message_id: str) -> bool:
        messages = self._db.get_messages(session_id, include_inactive=False)
        return any(
            str(message.get("platform_message_id") or "") == message_id
            for message in messages
        )

    def _append_shortcut_message(
        self, *, session_id: str, role: str, text: str, message_id: str
    ) -> None:
        timestamp = datetime.now(UTC).timestamp()
        append = getattr(self._session_store, "append_to_transcript", None)
        if callable(append):
            append(
                session_id,
                {
                    "role": role,
                    "content": text,
                    "message_id": message_id,
                    "observed": True,
                    "timestamp": timestamp,
                },
            )
            return
        self._db.append_message(
            session_id,
            role,
            content=text,
            platform_message_id=message_id,
            observed=True,
            timestamp=timestamp,
        )

    def record_shortcut_topic(
        self, *, title: str, text: str, topic_id: str, message_id: str
    ) -> str | None:
        """Persist a Shortcut-created topic so it enters the normal projection."""
        legacy_session_id = self._shortcut_session_id(topic_id)
        session_id: str | None = None
        if self._session_store is not None:
            try:
                entry = self._session_store.get_or_create_session(
                    self._shortcut_source(topic_id)
                )
                session_id = str(getattr(entry, "session_id", "")) or None
                legacy = self._db.get_session(legacy_session_id)
                switch = getattr(self._session_store, "switch_session", None)
                if (
                    legacy is not None
                    and session_id != legacy_session_id
                    and callable(switch)
                ):
                    switched = switch(
                        str(getattr(entry, "session_key", "")), legacy_session_id
                    )
                    if switched is not None:
                        session_id = legacy_session_id
            except Exception:
                logger.debug(
                    "Unable to route Shortcut topic through gateway session store",
                    exc_info=True,
                )
                session_id = None

        if session_id is None:
            if not all(
                callable(getattr(self._db, name, None))
                for name in (
                    "create_session",
                    "get_session",
                    "get_messages",
                    "append_message",
                )
            ):
                return None
            session_id = legacy_session_id
            if self._db.get_session(session_id) is None:
                self._db.create_session(
                    session_id=session_id,
                    source="telegram",
                    chat_id=self._chat_id,
                    chat_type="group",
                    thread_id=topic_id,
                )

        self._set_shortcut_title(session_id, title)
        if not self._shortcut_has_message(session_id, message_id):
            self._append_shortcut_message(
                session_id=session_id,
                role="user",
                text=text,
                message_id=message_id,
            )
        return session_id

    def record_shortcut_answer(
        self, *, session_id: str, text: str, message_id: str
    ) -> None:
        if self._shortcut_has_message(session_id, message_id):
            return
        self._append_shortcut_message(
            session_id=session_id,
            role="assistant",
            text=text,
            message_id=message_id,
        )

    @staticmethod
    def _hidden_child(row: dict[str, Any]) -> bool:
        if row.get("parent_session_id") not in (None, ""):
            return True
        model_config = row.get("model_config")
        if isinstance(model_config, str):
            try:
                model_config = json.loads(model_config)
            except (TypeError, ValueError, json.JSONDecodeError):
                model_config = None
        if isinstance(model_config, dict) and (
            model_config.get("_branched_from") is not None
            or model_config.get("_delegate_from") is not None
        ):
            return True
        return str(row.get("source") or "") == "tool"

    @staticmethod
    def revision_for_topic(
        row: dict[str, Any], transcript: list[dict[str, Any]]
    ) -> str:
        revision_input = row.get("_revision_input")
        if not isinstance(revision_input, dict):
            revision_input = row
        return revision_for(revision_input, transcript)


class BeckyLoopsBridgeServer:
    """A single-process, authenticated WebSocket server for Becky."""

    def __init__(
        self,
        *,
        config: BeckyLoopsConfig,
        store: BeckyLoopsStore,
        summarizer: LoopSummarizer,
        topic_sender: TopicSender | None = None,
        topic_controller: TopicController | None = None,
        reply_generator: ReplyGenerator | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("Becky loops bridge is disabled")
        if len(config.token) < 32:
            raise ValueError("Becky loops bridge token is invalid")
        if not config.chat_id.strip():
            raise ValueError("Becky loops bridge chat ID is invalid")
        if not 0 <= config.port <= 65_535:
            raise ValueError("Becky loops bridge port is invalid")
        self.config = config
        self.store = store
        self.summarizer = summarizer
        self.topic_sender = topic_sender
        self.topic_controller = topic_controller
        self.reply_generator = reply_generator
        self._reply_attempts: dict[str, _ReplyAttempt] = {}
        self._reply_attempts_lock = asyncio.Lock()
        self._new_topic_answers: dict[
            str, tuple[str, asyncio.Future[dict[str, Any]]]
        ] = {}
        self._new_topic_answers_lock = asyncio.Lock()
        self._close_results: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self._close_results_lock = asyncio.Lock()
        self._server: Server | None = None

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.config.port
        return int(self._server.sockets[0].getsockname()[1])

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handle_connection,
            host="127.0.0.1",
            port=self.config.port,
            process_request=self._process_request,
            max_size=_MAX_RESPONSE_BYTES,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=1,
            compression=None,
            server_header="Hermes-Becky-Loops",
            logger=_WEBSOCKET_LOGGER,
        )
        logger.info("Becky loop bridge listening on 127.0.0.1:%d", self.bound_port)

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        await server.wait_closed()

    async def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        del connection
        if request.path.split("?", 1)[0] != "/api/ws":
            return _http_response(404, "Not found")
        query = parse_qs(urlsplit(request.path).query, keep_blank_values=True)
        supplied = query.get("token", [""])[0]
        if set(query) != {"token"} or len(query["token"]) != 1:
            return _http_response(401, "Unauthorized")
        if supplied != self.config.token:
            return _http_response(401, "Unauthorized")
        return None

    async def _handle_connection(self, connection: ServerConnection) -> None:
        try:
            await connection.send(json.dumps(_READY, separators=(",", ":")))
            async for frame in connection:
                if (
                    not isinstance(frame, str)
                    or len(frame.encode()) > _MAX_REQUEST_BYTES
                ):
                    await connection.send(
                        json.dumps(self._error(None, "protocol"), separators=(",", ":"))
                    )
                    await connection.close(code=1009, reason="request too large")
                    return
                response = await self._dispatch(frame)
                encoded = json.dumps(
                    response, ensure_ascii=False, separators=(",", ":")
                )
                if len(encoded.encode()) > _MAX_RESPONSE_BYTES:
                    await connection.send(
                        json.dumps(self._error(None, "protocol"), separators=(",", ":"))
                    )
                    await connection.close(code=1009, reason="response too large")
                    return
                await connection.send(encoded)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Becky bridge connection ended", exc_info=True)

    async def _dispatch(self, frame: str) -> dict[str, Any]:
        try:
            request = json.loads(frame)
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._error(None, "protocol")
        if not isinstance(request, dict):
            return self._error(None, "protocol")
        request_id = request.get("id")
        if (
            set(request) != {"jsonrpc", "id", "method", "params"}
            or request.get("jsonrpc") != "2.0"
            or not isinstance(request_id, int)
            or isinstance(request_id, bool)
            or not isinstance(request.get("method"), str)
            or not isinstance(request.get("params"), dict)
        ):
            return self._error(
                request_id
                if isinstance(request_id, int) and not isinstance(request_id, bool)
                else None,
                "protocol",
            )
        method = request["method"]
        params = request["params"]
        try:
            result = await self._method(method, params)
        except _RemoteFailure as failure:
            return self._remote_error(request_id, failure.code)
        except Exception:
            logger.warning("Becky bridge method failed: %s", method)
            return self._error(request_id, "protocol")
        if result is _PROTOCOL_FAILURE:
            return self._error(request_id, "protocol")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def _method(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | list[Any] | object:
        if method == "becky.loops.capabilities":
            if params:
                return _PROTOCOL_FAILURE
            return {
                "schema_version": "2",
                "summary_schema_version": "1",
                "methods": _METHODS,
                "topic_control": self._topic_control_method(),
                "topic_reply": (
                    "bot_api_private_topic"
                    if self._topic_reply_available()
                    else "unavailable"
                ),
                "same_topic_reopen": False,
                "new_session_fallback": True,
                "max_request_bytes": _MAX_REQUEST_BYTES,
                "max_response_bytes": _MAX_RESPONSE_BYTES,
            }
        if method == "becky.loops.list":
            if params:
                return _PROTOCOL_FAILURE
            rows = self.store.list_topics(self.config.chat_id)
            return {"loops": [self._public_index(row) for row in rows]}
        if method == "becky.loops.summarize":
            if set(params) != {"source_ref", "expected_revision", "force"}:
                return _PROTOCOL_FAILURE
            source_ref = params.get("source_ref")
            expected_revision = params.get("expected_revision")
            force = params.get("force")
            if (
                not isinstance(source_ref, str)
                or not _SOURCE_REF_RE.fullmatch(source_ref)
                or not isinstance(expected_revision, str)
                or not _REVISION_RE.fullmatch(expected_revision)
                or not isinstance(force, bool)
            ):
                return _PROTOCOL_FAILURE
            row = self._find_topic(source_ref)
            if row is None:
                raise _RemoteFailure("source_not_found")
            transcript = self.store.transcript(str(row["session_id"]))
            # ``list_topics`` computes the revision from the same authoritative
            # snapshot used to resolve this source.  Reusing that value keeps
            # test/store implementations that supply an explicit revision
            # contract-compatible while the SessionDB store still recomputes it
            # on every list call.
            revision_fn = getattr(self.store, "revision_for_topic", None)
            current_revision = (
                revision_fn(row, transcript)
                if callable(revision_fn)
                else str(row.get("revision") or "")
            )
            if current_revision != expected_revision:
                raise _RemoteFailure("revision_conflict")
            try:
                summary = await self.summarizer.summarize(
                    row=row,
                    transcript=transcript,
                    deadline=(
                        asyncio.get_running_loop().time() + _SUMMARY_DEADLINE_SECONDS
                    ),
                )
            except _ConversationTooLarge:
                raise _RemoteFailure("conversation_too_large") from None
            except _SummaryValidationError:
                raise _RemoteFailure("summary_invalid") from None
            except SummaryUnavailable:
                raise _RemoteFailure("summary_timeout") from None
            return self._public_summary(
                row=row,
                revision=current_revision,
                transcript=transcript,
                summary=summary,
            )
        if method == "becky.loops.reply":
            if not self._valid_reply_params(params):
                return _PROTOCOL_FAILURE
            if not self._topic_reply_available():
                raise _RemoteFailure("topic_reply_unavailable")
            source_ref = params["source_ref"]
            expected_revision = params["expected_revision"]
            comment = params["text"].strip()
            idempotency_key = params["idempotency_key"].lower()
            existing = await self._get_reply_attempt(idempotency_key)
            if existing is not None:
                if (
                    existing.source_ref != source_ref
                    or existing.expected_revision != expected_revision
                    or existing.comment != comment
                ):
                    raise _RemoteFailure("idempotency_conflict")
                return await self._replay_reply_attempt(existing)
            row, _, _ = self._current_topic(source_ref, expected_revision)
            attempt, created = await self._get_or_create_reply_attempt(
                idempotency_key=idempotency_key,
                source_ref=source_ref,
                expected_revision=expected_revision,
                comment=comment,
                thread_id=str(row.get("thread_id") or ""),
            )
            if not created:
                if (
                    attempt.source_ref != source_ref
                    or attempt.expected_revision != expected_revision
                    or attempt.comment != comment
                ):
                    raise _RemoteFailure("idempotency_conflict")
                return await self._replay_reply_attempt(attempt)
            return await self._continue_reply_attempt(attempt=attempt)
        if method == "becky.loops.reply_retry":
            if not self._valid_reply_retry_params(params):
                return _PROTOCOL_FAILURE
            if not self._topic_reply_available():
                raise _RemoteFailure("topic_reply_unavailable")
            idempotency_key = params["idempotency_key"].lower()
            attempt = await self._get_reply_attempt(idempotency_key)
            if attempt is None:
                raise _RemoteFailure("reply_retry_unavailable")
            source_ref = params["source_ref"]
            expected_revision = params["expected_revision"]
            if (
                attempt.source_ref != source_ref
                or attempt.expected_revision != expected_revision
            ):
                raise _RemoteFailure("idempotency_conflict")
            self._current_topic(source_ref, expected_revision)
            return await self._continue_reply_attempt(attempt=attempt)
        if method == "becky.loops.answer_new_topic":
            if not self._valid_new_topic_reply_params(params):
                return _PROTOCOL_FAILURE
            if not self._topic_reply_available():
                raise _RemoteFailure("topic_reply_unavailable")
            return await self._answer_new_topic(
                title=params["title"],
                text=params["text"],
                topic_id=params["topic_id"],
                message_id=params["message_id"],
                idempotency_key=params["idempotency_key"].lower(),
            )
        if method == "becky.loops.close":
            if not self._valid_close_params(params):
                return _PROTOCOL_FAILURE
            if not self._topic_control_available():
                raise _RemoteFailure("topic_control_unavailable")
            return await self._close_topic(
                source_ref=params["source_ref"],
                expected_revision=params["expected_revision"],
                idempotency_key=params["idempotency_key"].lower(),
            )
        if method == "becky.loops.reopen":
            if not self._valid_reopen_params(params):
                return _PROTOCOL_FAILURE
            raise _RemoteFailure("topic_control_unavailable")
        return _PROTOCOL_FAILURE

    async def _answer_new_topic(
        self,
        *,
        title: str,
        text: str,
        topic_id: str,
        message_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        fingerprint = json.dumps(
            {
                "title": title,
                "text": text,
                "topic_id": topic_id,
                "message_id": message_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        loop = asyncio.get_running_loop()
        async with self._new_topic_answers_lock:
            existing = self._new_topic_answers.get(idempotency_key)
            if existing is not None:
                if existing[0] != fingerprint:
                    raise _RemoteFailure("idempotency_conflict")
                future = existing[1]
                owner = False
            else:
                if len(self._new_topic_answers) >= _MAX_NEW_TOPIC_ANSWERS:
                    oldest_key = next(iter(self._new_topic_answers))
                    oldest = self._new_topic_answers[oldest_key][1]
                    if not oldest.done():
                        raise _RemoteFailure("topic_reply_unavailable")
                    self._new_topic_answers.pop(oldest_key, None)
                future = loop.create_future()
                self._new_topic_answers[idempotency_key] = (fingerprint, future)
                owner = True
        if not owner:
            return await asyncio.shield(future)

        result: dict[str, Any]
        shortcut_session_id = self._record_shortcut_topic(
            title=title,
            text=text,
            topic_id=topic_id,
            message_id=message_id,
        )
        try:
            answer = await self._generate_reply(
                row={
                    "title": title,
                    "chat_id": self.config.chat_id,
                    "thread_id": topic_id,
                    "source_ref": f"shortcut_{idempotency_key}",
                },
                transcript=[
                    {
                        "role": "user",
                        "content": text,
                        "timestamp": datetime.now(UTC).timestamp(),
                    }
                ],
                comment="Answer the user's opening message.",
            )
            answer = answer.strip()
            if not 1 <= len(answer) <= _MAX_REPLY_TEXT_CHARS:
                raise ValueError("reply unavailable")
            receipt = await self._send_topic(
                thread_id=topic_id,
                text=answer,
                reply_to_message_id=message_id,
            )
            if shortcut_session_id is not None:
                self._record_shortcut_answer(
                    session_id=shortcut_session_id,
                    text=answer,
                    message_id=receipt.message_id,
                )
            result = {"schema_version": "1", "answer_state": "answered"}
        except asyncio.CancelledError:
            result = {"schema_version": "1", "answer_state": "answer_unavailable"}
            if not future.done():
                future.set_result(result)
            raise
        except Exception:
            result = {"schema_version": "1", "answer_state": "answer_unavailable"}
        if not future.done():
            future.set_result(result)
        return result

    def _record_shortcut_topic(
        self, *, title: str, text: str, topic_id: str, message_id: str
    ) -> str | None:
        recorder = getattr(self.store, "record_shortcut_topic", None)
        if not callable(recorder):
            return None
        try:
            session_id = recorder(
                title=title,
                text=text,
                topic_id=topic_id,
                message_id=message_id,
            )
        except Exception:
            logger.debug(
                "Unable to persist Shortcut-created Telegram topic", exc_info=True
            )
            return None
        return session_id if isinstance(session_id, str) and session_id else None

    def _record_shortcut_answer(
        self, *, session_id: str, text: str, message_id: str
    ) -> None:
        recorder = getattr(self.store, "record_shortcut_answer", None)
        if not callable(recorder):
            return
        try:
            recorder(session_id=session_id, text=text, message_id=message_id)
        except Exception:
            logger.debug(
                "Unable to persist Shortcut-created Telegram answer", exc_info=True
            )

    def _find_topic(self, source_ref: str) -> dict[str, Any] | None:
        rows = self.store.list_topics(self.config.chat_id)
        return next((row for row in rows if row.get("source_ref") == source_ref), None)

    def _topic_reply_available(self) -> bool:
        if (
            self.config.topic_reply != "bot_api_private_topic"
            or self.topic_sender is None
            or self.reply_generator is None
        ):
            return False
        sender_state = getattr(self.topic_sender, "is_connected", None)
        if sender_state is None:
            return True
        try:
            return bool(sender_state() if callable(sender_state) else sender_state)
        except Exception:
            return False

    def _topic_control_available(self) -> bool:
        if (
            self.config.topic_control
            not in {"bot_api_private_topic", "mtproto_private_topic"}
            or self.topic_controller is None
        ):
            return False
        for attribute in ("is_connected", "supports_close"):
            state = getattr(self.topic_controller, attribute, False)
            try:
                state = state() if callable(state) else state
            except Exception:
                return False
            if not bool(state):
                return False
        return (
            getattr(self.topic_controller, "method", None) == self.config.topic_control
        )

    def _topic_control_method(self) -> str:
        if not self._topic_control_available():
            return "unavailable"
        method = getattr(self.topic_controller, "method", None)
        if method != self.config.topic_control:
            return "unavailable"
        return method

    async def _close_topic(
        self, *, source_ref: str, expected_revision: str, idempotency_key: str
    ) -> dict[str, Any]:
        async with self._close_results_lock:
            existing = self._close_results.get(idempotency_key)
            if existing is not None:
                if existing[0] != source_ref or existing[1] != expected_revision:
                    raise _RemoteFailure("idempotency_conflict")
                return dict(existing[2])
            row, _, _ = self._current_topic(source_ref, expected_revision)
            source_state = str(row.get("source_state") or "active")
            if source_state == "deleted":
                raise _RemoteFailure("source_not_found")
            if source_state == "closed":
                closed_at = _utc_datetime(row.get("updated_at"))
            elif source_state == "active":
                if self.topic_controller is None:
                    raise _RemoteFailure("topic_control_unavailable")
                try:
                    closed_at = await self.topic_controller.close_topic(
                        chat_id=self.config.chat_id,
                        thread_id=str(row.get("thread_id") or ""),
                    )
                except _TopicControlFailure as failure:
                    raise _RemoteFailure(failure.code) from None
                except MtprotoTopicControlError as failure:
                    code = (
                        "topic_control_unavailable"
                        if failure.code == "topic_control_forbidden"
                        else failure.code
                    )
                    raise _RemoteFailure(code) from None
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise _RemoteFailure("topic_control_unavailable") from None
            else:
                raise _RemoteFailure("topic_state_read_failed")
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=UTC)
            control_method = getattr(self.topic_controller, "method", None)
            if control_method not in {
                "bot_api_private_topic",
                "mtproto_private_topic",
            }:
                control_method = self.config.topic_control
            result = {
                "source_ref": source_ref,
                "source_state": "closed",
                "closed_at": closed_at.isoformat(),
                "control_method": control_method,
                "idempotency_key": idempotency_key,
            }
            self._close_results[idempotency_key] = (
                source_ref,
                expected_revision,
                result,
            )
            return dict(result)

    def _current_topic(
        self, source_ref: str, expected_revision: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        row = self._find_topic(source_ref)
        if row is None:
            raise _RemoteFailure("source_not_found")
        transcript = self.store.transcript(str(row["session_id"]))
        revision_fn = getattr(self.store, "revision_for_topic", None)
        current_revision = (
            revision_fn(row, transcript)
            if callable(revision_fn)
            else str(row.get("revision") or "")
        )
        if current_revision != expected_revision:
            raise _RemoteFailure("revision_conflict")
        return row, transcript, current_revision

    async def _get_or_create_reply_attempt(
        self,
        *,
        idempotency_key: str,
        source_ref: str,
        expected_revision: str,
        comment: str,
        thread_id: str,
    ) -> tuple[_ReplyAttempt, bool]:
        now = asyncio.get_running_loop().time()
        async with self._reply_attempts_lock:
            self._purge_reply_attempts(now)
            existing = self._reply_attempts.get(idempotency_key)
            if existing is not None:
                return existing, False
            if not thread_id:
                raise _RemoteFailure("source_not_found")
            if len(self._reply_attempts) >= _MAX_REPLY_ATTEMPTS:
                raise _RemoteFailure("topic_reply_unavailable")
            attempt = _ReplyAttempt(
                source_ref=source_ref,
                expected_revision=expected_revision,
                comment=comment,
                thread_id=thread_id,
                expires_at=now + _REPLY_ATTEMPT_TTL_SECONDS,
            )
            self._reply_attempts[idempotency_key] = attempt
            return attempt, True

    async def _get_reply_attempt(self, idempotency_key: str) -> _ReplyAttempt | None:
        now = asyncio.get_running_loop().time()
        async with self._reply_attempts_lock:
            self._purge_reply_attempts(now)
            return self._reply_attempts.get(idempotency_key)

    async def _discard_reply_attempt(self, attempt: _ReplyAttempt) -> None:
        async with self._reply_attempts_lock:
            for key, stored_attempt in self._reply_attempts.items():
                if stored_attempt is attempt:
                    self._reply_attempts.pop(key)
                    return

    def _purge_reply_attempts(self, now: float) -> None:
        expired = [
            key
            for key, attempt in self._reply_attempts.items()
            if attempt.expires_at <= now and not attempt.in_progress
        ]
        for key in expired:
            self._reply_attempts.pop(key, None)

    async def _replay_reply_attempt(self, attempt: _ReplyAttempt) -> dict[str, Any]:
        async with attempt.lock:
            if attempt.state == "send_failed":
                raise _RemoteFailure("reply_send_failed")
            if attempt.comment_sent_at is None:
                raise _RemoteFailure("reply_retry_unavailable")
            return self._reply_attempt_result(attempt)

    async def _continue_reply_attempt(
        self,
        *,
        attempt: _ReplyAttempt,
    ) -> dict[str, Any]:
        async with attempt.lock:
            attempt.in_progress = True
            try:
                if attempt.state == "answered":
                    return self._reply_attempt_result(attempt)
                if attempt.state == "send_failed":
                    raise _RemoteFailure("reply_retry_unavailable")
                if attempt.comment_message_id is None:
                    try:
                        self._current_topic(
                            attempt.source_ref, attempt.expected_revision
                        )
                    except Exception:
                        await self._discard_reply_attempt(attempt)
                        raise
                    try:
                        receipt = await self._send_topic(
                            thread_id=attempt.thread_id,
                            text=attempt.comment,
                            reply_to_message_id=None,
                        )
                    except asyncio.CancelledError:
                        attempt.state = "send_failed"
                        raise
                    except Exception:
                        attempt.state = "send_failed"
                        raise _RemoteFailure("reply_send_failed") from None
                    attempt.comment_message_id = receipt.message_id
                    attempt.comment_sent_at = datetime.now(UTC).isoformat()
                if attempt.answer is None:
                    try:
                        row, transcript, _ = self._current_topic(
                            attempt.source_ref, attempt.expected_revision
                        )
                    except Exception:
                        attempt.state = "answer_unavailable"
                        raise
                    try:
                        answer = await self._generate_reply(
                            row=row,
                            transcript=transcript,
                            comment=attempt.comment,
                        )
                        answer = answer.strip()
                        if not 1 <= len(answer) <= _MAX_REPLY_TEXT_CHARS:
                            raise ValueError("reply unavailable")
                    except asyncio.CancelledError:
                        attempt.state = "answer_unavailable"
                        raise
                    except Exception:
                        attempt.state = "answer_unavailable"
                        return self._reply_attempt_result(attempt)
                    attempt.answer = answer
                    attempt.answer_generated_at = datetime.now(UTC).isoformat()
                    attempt.state = "answer_pending"
                try:
                    _, _, current_revision = self._current_topic(
                        attempt.source_ref, attempt.expected_revision
                    )
                except _RemoteFailure:
                    attempt.state = "answer_pending"
                    raise
                try:
                    await self._send_topic(
                        thread_id=attempt.thread_id,
                        text=attempt.answer,
                        reply_to_message_id=attempt.comment_message_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    attempt.state = "answer_pending"
                    return self._reply_attempt_result(attempt)
                attempt.answer_sent_at = datetime.now(UTC).isoformat()
                attempt.state = "answered"
                result = self._reply_attempt_result(attempt)
                result["revision"] = current_revision
                return result
            finally:
                attempt.in_progress = False

    async def _send_topic(
        self, *, thread_id: str, text: str, reply_to_message_id: str | None
    ) -> TopicSendReceipt:
        if self.topic_sender is None:
            raise _TopicSendFailure()
        receipt = await self.topic_sender.send_topic(
            chat_id=self.config.chat_id,
            thread_id=thread_id,
            text=text,
            reply_to_message_id=reply_to_message_id,
        )
        if not isinstance(receipt, TopicSendReceipt) or not receipt.message_id.strip():
            raise _TopicSendFailure()
        return receipt

    async def _generate_reply(
        self,
        *,
        row: dict[str, Any],
        transcript: list[dict[str, Any]],
        comment: str,
    ) -> str:
        if self.reply_generator is None:
            raise RuntimeError("reply unavailable")
        return await self.reply_generator.generate(
            row=row,
            transcript=transcript,
            comment=comment,
            deadline=asyncio.get_running_loop().time() + _REPLY_DEADLINE_SECONDS,
        )

    @staticmethod
    def _reply_attempt_result(attempt: _ReplyAttempt) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "source_ref": attempt.source_ref,
            "revision": attempt.expected_revision,
            "comment_sent_at": attempt.comment_sent_at,
            "answer": attempt.answer,
            "answer_generated_at": attempt.answer_generated_at,
            "answer_sent_at": attempt.answer_sent_at,
            "answer_state": attempt.state,
        }

    def _public_index(self, row: dict[str, Any]) -> dict[str, Any]:
        hidden_values = {
            self.config.chat_id,
            str(row.get("session_id") or ""),
            str(row.get("thread_id") or ""),
            str(row.get("source_ref") or ""),
        }
        if "last_becky_response" in row:
            last_response = row.get("last_becky_response")
            last_response_at = row.get("last_becky_response_at")
        else:
            session_id = str(row.get("session_id") or "")
            transcript = self.store.transcript(session_id) if session_id else []
            for message in transcript:
                for key in (
                    "id",
                    "platform_message_id",
                    "telegram_message_id",
                    "chat_id",
                    "thread_id",
                    "session_id",
                    "user_id",
                ):
                    value = message.get(key)
                    if value not in (None, ""):
                        hidden_values.add(str(value))
            last_response, last_response_at = _latest_public_becky_response(
                transcript, hidden_values
            )
        title = (
            _safe_public_text(row.get("title") or "Telegram loop", hidden_values, 128)
            or "Telegram loop"
        )
        if title == "[REDACTED]" or title.startswith("«redacted"):
            title = "Telegram loop"
        if isinstance(last_response, str):
            last_response = _safe_public_text(last_response, hidden_values, 2_000)
            if not last_response or last_response == "[REDACTED]":
                last_response = None
        else:
            last_response = None
        if isinstance(last_response_at, datetime):
            response_at = last_response_at.isoformat()
        else:
            response_at = None
        return {
            "source_ref": row["source_ref"],
            "title": title,
            "source_state": row.get("source_state", "active"),
            "revision": row["revision"],
            "message_count": max(0, int(row.get("message_count") or 0)),
            "created_at": _utc_datetime(row.get("created_at")).isoformat(),
            "updated_at": _utc_datetime(row.get("updated_at")).isoformat(),
            "telegram_url": None,
            "last_becky_response": last_response,
            "last_becky_response_at": response_at,
        }

    def _public_summary(
        self,
        *,
        row: dict[str, Any],
        revision: str,
        transcript: list[dict[str, Any]],
        summary: LoopSummary,
    ) -> dict[str, Any]:
        hidden_values = {
            self.config.chat_id,
            str(row.get("session_id") or ""),
            str(row.get("thread_id") or ""),
            str(row.get("source_ref") or ""),
        }
        for message in transcript:
            for key in (
                "id",
                "platform_message_id",
                "telegram_message_id",
                "chat_id",
                "thread_id",
                "session_id",
                "user_id",
            ):
                value = message.get(key)
                if value not in (None, ""):
                    hidden_values.add(str(value))
        return {
            "schema_version": "1",
            "source_ref": row["source_ref"],
            "revision": revision,
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": _safe_public_text(summary.summary, hidden_values, 2_000),
            "decisions": [
                _safe_public_text(item, hidden_values, 500)
                for item in summary.decisions
            ],
            "unresolved_items": [
                _safe_public_text(item, hidden_values, 500)
                for item in summary.unresolved_items
            ],
            "next_action": (
                _safe_public_text(summary.next_action, hidden_values, 500)
                if summary.next_action is not None
                else None
            ),
            "waiting_on": summary.waiting_on,
            "key_events": [
                {
                    "occurred_at": event["occurred_at"],
                    "text": _safe_public_text(event["text"], hidden_values, 500),
                }
                for event in summary.key_events
            ],
            "final_outcome": (
                _safe_public_text(summary.final_outcome, hidden_values, 1_000)
                if summary.final_outcome is not None
                else None
            ),
        }

    @staticmethod
    def _valid_reply_params(params: dict[str, Any]) -> bool:
        text = params.get("text")
        return (
            set(params)
            == {"source_ref", "expected_revision", "text", "idempotency_key"}
            and isinstance(params.get("source_ref"), str)
            and _SOURCE_REF_RE.fullmatch(params["source_ref"]) is not None
            and isinstance(params.get("expected_revision"), str)
            and _REVISION_RE.fullmatch(params["expected_revision"]) is not None
            and isinstance(text, str)
            and 1 <= len(text.strip()) <= _MAX_REPLY_COMMENT_CHARS
            and isinstance(params.get("idempotency_key"), str)
            and _UUID_RE.fullmatch(params["idempotency_key"]) is not None
        )

    @staticmethod
    def _valid_reply_retry_params(params: dict[str, Any]) -> bool:
        return (
            set(params) == {"source_ref", "expected_revision", "idempotency_key"}
            and isinstance(params.get("source_ref"), str)
            and _SOURCE_REF_RE.fullmatch(params["source_ref"]) is not None
            and isinstance(params.get("expected_revision"), str)
            and _REVISION_RE.fullmatch(params["expected_revision"]) is not None
            and isinstance(params.get("idempotency_key"), str)
            and _UUID_RE.fullmatch(params["idempotency_key"]) is not None
        )

    @staticmethod
    def _valid_new_topic_reply_params(params: dict[str, Any]) -> bool:
        title = params.get("title")
        text = params.get("text")
        return (
            set(params)
            == {"title", "text", "topic_id", "message_id", "idempotency_key"}
            and isinstance(title, str)
            and 1 <= len(title.strip()) <= 128
            and isinstance(text, str)
            and 1 <= len(text.strip()) <= _MAX_REPLY_COMMENT_CHARS
            and isinstance(params.get("topic_id"), str)
            and _POSITIVE_TELEGRAM_ID_RE.fullmatch(params["topic_id"]) is not None
            and isinstance(params.get("message_id"), str)
            and _POSITIVE_TELEGRAM_ID_RE.fullmatch(params["message_id"]) is not None
            and isinstance(params.get("idempotency_key"), str)
            and _UUID_RE.fullmatch(params["idempotency_key"]) is not None
        )

    @staticmethod
    def _valid_close_params(params: dict[str, Any]) -> bool:
        return (
            set(params) == {"source_ref", "expected_revision", "idempotency_key"}
            and isinstance(params.get("source_ref"), str)
            and _SOURCE_REF_RE.fullmatch(params["source_ref"]) is not None
            and isinstance(params.get("expected_revision"), str)
            and _REVISION_RE.fullmatch(params["expected_revision"]) is not None
            and isinstance(params.get("idempotency_key"), str)
            and _UUID_RE.fullmatch(params["idempotency_key"]) is not None
        )

    @staticmethod
    def _valid_reopen_params(params: dict[str, Any]) -> bool:
        context = params.get("context")
        if not isinstance(context, dict):
            return False
        decisions = context.get("decisions")
        unresolved = context.get("unresolved_items")
        final_outcome = context.get("final_outcome")
        return (
            set(params) == {"source_ref", "idempotency_key", "context"}
            and isinstance(params.get("source_ref"), str)
            and _SOURCE_REF_RE.fullmatch(params["source_ref"]) is not None
            and isinstance(params.get("idempotency_key"), str)
            and _UUID_RE.fullmatch(params["idempotency_key"]) is not None
            and set(context)
            == {"title", "summary", "decisions", "unresolved_items", "final_outcome"}
            and isinstance(context.get("title"), str)
            and 1 <= len(context["title"]) <= 128
            and isinstance(context.get("summary"), str)
            and 1 <= len(context["summary"]) <= 2000
            and isinstance(decisions, list)
            and len(decisions) <= 12
            and all(
                isinstance(item, str) and 1 <= len(item) <= 500 for item in decisions
            )
            and isinstance(unresolved, list)
            and len(unresolved) <= 12
            and all(
                isinstance(item, str) and 1 <= len(item) <= 500 for item in unresolved
            )
            and (
                final_outcome is None
                or isinstance(final_outcome, str)
                and 1 <= len(final_outcome) <= 1000
            )
        )

    @staticmethod
    def _error(request_id: int | None, code: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32600, "message": code},
        }

    @staticmethod
    def _remote_error(request_id: int | None, code: str) -> dict[str, Any]:
        if code not in _SAFE_REMOTE_CODES:
            code = "protocol"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": code},
        }


class _RemoteFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


class _TopicSendFailure(RuntimeError):
    def __init__(self) -> None:
        super().__init__("topic_send_failed")


class _TopicControlFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = (
            code
            if code
            in {
                "topic_already_closed",
                "topic_control_unavailable",
                "topic_control_unsupported",
                "topic_not_found",
                "topic_state_write_failed",
            }
            else "topic_control_unavailable"
        )
        super().__init__(self.code)


def _classify_topic_control_error(error: BaseException) -> str:
    """Map provider diagnostics to the closed public topic-control set."""
    text = str(error).lower()
    if any(
        marker in text
        for marker in (
            "topic_not_modified",
            "topic is already closed",
            "already closed",
        )
    ):
        return "topic_already_closed"
    if any(
        marker in text
        for marker in ("message thread not found", "topic not found", "chat not found")
    ):
        return "topic_not_found"
    if any(
        marker in text
        for marker in (
            "not a forum",
            "forums_disabled",
            "forum topics are disabled",
            "method not found",
        )
    ):
        return "topic_control_unsupported"
    return "topic_control_unavailable"


class _ProtocolFailure:
    pass


_PROTOCOL_FAILURE = _ProtocolFailure()


def _http_response(status: int, body: str) -> Response:
    body_bytes = body.encode("utf-8")
    headers = Headers([
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body_bytes))),
    ])
    return Response(
        status, "Unauthorized" if status == 401 else "Not Found", headers, body_bytes
    )


def load_becky_loops_config(config_path: Path | None = None) -> BeckyLoopsConfig | None:
    """Load the opt-in bridge settings without exposing their values."""
    if config_path is not None:
        path = config_path
    else:
        try:
            from hermes_constants import get_hermes_home

            path = get_hermes_home() / "config.yaml"
        except Exception:
            path = (
                Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser() / "config.yaml"
            )
    raw: dict[str, Any] = {}
    try:
        import yaml

        if path.exists():
            parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(parsed, dict):
                raw = parsed
    except Exception:
        logger.warning(
            "Unable to read Hermes config for Becky loop bridge", exc_info=True
        )
    section = (
        raw.get("gateway", {}).get("becky_loops", {})
        if isinstance(raw.get("gateway"), dict)
        else {}
    )
    if not isinstance(section, dict) or section.get("enabled") is not True:
        return None
    token = os.getenv("HERMES_BECKY_LOOPS_TOKEN", "").strip()
    chat_id = str(section.get("telegram_chat_id", "")).strip()
    try:
        port = int(section.get("port", 9_120))
    except (TypeError, ValueError):
        port = 9_120
    proven_topic_reply = (
        section.get("proven_topic_reply") is True
        and os.getenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "") == "1"
        and bool(token)
        and _valid_telegram_id(chat_id)
        and _has_configured_loop_topic(raw, section, chat_id)
    )
    raw_requested_control = section.get("proven_topic_control")
    requested_control: str | None
    if raw_requested_control is True:
        # Backward-compatible configuration for the original Bot API proof.
        requested_control = "bot_api_private_topic"
    elif isinstance(raw_requested_control, str) and raw_requested_control in {
        "bot_api_private_topic",
        "mtproto_private_topic",
    }:
        requested_control = raw_requested_control
    else:
        requested_control = None
    mtproto_credentials = (
        requested_control != "mtproto_private_topic"
        or _mtproto_credentials_configured()
    )
    proven_topic_control = (
        requested_control is not None
        and os.getenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_CONTROL", "") == "1"
        and bool(token)
        and bool(mtproto_credentials)
        and _valid_telegram_id(chat_id)
        and _has_configured_loop_topic(raw, section, chat_id)
    )
    return BeckyLoopsConfig(
        enabled=True,
        chat_id=chat_id,
        token=token,
        port=port,
        # Topic control is advertised only after the separate proof flag,
        # same-chat topic binding, and (for MTProto) credentials are present.
        # GatewayRunner additionally checks the live controller before
        # injection.
        topic_control=(requested_control if proven_topic_control else "unavailable"),
        # Topic sends require the profile proof, the gateway-start environment
        # proof, a bridge token, and a configured Telegram chat/topic. The
        # connected adapter is checked separately by GatewayRunner before it
        # is injected into the bridge.
        topic_reply=("bot_api_private_topic" if proven_topic_reply else "unavailable"),
    )


def _mtproto_credentials_configured() -> bool:
    raw_api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not raw_api_id or not api_hash or not bot_token:
        return False
    try:
        return int(raw_api_id) > 0 and bool(re.fullmatch(r"[0-9a-fA-F]{32}", api_hash))
    except ValueError:
        return False


def _valid_telegram_id(value: Any) -> bool:
    """Accept only the numeric Telegram chat/topic identifiers used internally."""
    if isinstance(value, bool):
        return False
    return bool(_TELEGRAM_ID_RE.fullmatch(str(value).strip()))


def _has_configured_loop_topic(
    raw: dict[str, Any], section: dict[str, Any], chat_id: str
) -> bool:
    """Check for an explicit topic or a matching configured DM topic.

    The topic identifier never leaves the gateway. It is only used here as a
    local proof that the opt-in capability points at an existing private-topic
    configuration rather than an arbitrary chat.
    """
    platforms = raw.get("platforms")
    if not isinstance(platforms, dict):
        return False
    telegram = platforms.get("telegram")
    if not isinstance(telegram, dict):
        return False
    extra = telegram.get("extra")
    if not isinstance(extra, dict):
        return False
    dm_topics = extra.get("dm_topics")
    if not isinstance(dm_topics, list):
        return False
    configured_thread_ids: set[str] = set()
    for chat_entry in dm_topics:
        if not isinstance(chat_entry, dict):
            continue
        if str(chat_entry.get("chat_id", "")).strip() != chat_id:
            continue
        topics = chat_entry.get("topics")
        if not isinstance(topics, list):
            continue
        configured_thread_ids.update(
            str(topic.get("thread_id")).strip()
            for topic in topics
            if isinstance(topic, dict) and _valid_telegram_id(topic.get("thread_id"))
        )
    if not configured_thread_ids:
        return False
    explicit_keys = (
        "telegram_topic_id",
        "telegram_thread_id",
        "topic_id",
        "thread_id",
    )
    present_aliases = [key for key in explicit_keys if key in section]
    if any(not _valid_telegram_id(section[key]) for key in present_aliases):
        return False
    explicit_ids = {str(section[key]).strip() for key in present_aliases}
    # Multiple aliases are accepted only when they agree on one configured
    # topic. An intersection is not enough: conflicting aliases must not turn
    # a partial proof into an enabled capability.
    return not explicit_ids or (
        len(explicit_ids) == 1 and explicit_ids <= configured_thread_ids
    )


async def start_becky_loops_bridge(
    *,
    config: BeckyLoopsConfig | None,
    db: Any,
    session_store: Any | None = None,
    summarizer: LoopSummarizer | None = None,
    topic_sender: TopicSender | None = None,
    topic_controller: TopicController | None = None,
    reply_generator: ReplyGenerator | None = None,
) -> BeckyLoopsBridgeServer | None:
    """Start the opt-in bridge and return its lifecycle handle."""
    if config is None or not config.enabled:
        return None
    try:
        store = SessionDBBeckyLoopsStore(db, session_store=session_store)
        store._chat_id = config.chat_id
        server = BeckyLoopsBridgeServer(
            config=config,
            store=store,
            summarizer=(
                summarizer
                if summarizer is not None
                else LoopSummarizer(AsyncAuxiliarySummaryProvider())
            ),
            topic_sender=topic_sender,
            topic_controller=topic_controller,
            reply_generator=(
                reply_generator
                if reply_generator is not None
                else (
                    LoopReplyGenerator(AsyncAuxiliaryReplyProvider())
                    if topic_sender is not None
                    and config.topic_reply == "bot_api_private_topic"
                    else None
                )
            ),
        )
        await server.start()
        return server
    except Exception:
        logger.error("Becky loop bridge failed to start", exc_info=True)
        return None


async def stop_becky_loops_bridge(server: BeckyLoopsBridgeServer | None) -> None:
    """Stop the bridge without allowing it to block gateway teardown."""
    if server is None:
        return
    try:
        await asyncio.wait_for(server.stop(), timeout=2.0)
    except Exception:
        logger.debug("Becky loop bridge failed to stop", exc_info=True)
