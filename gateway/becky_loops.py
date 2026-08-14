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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.http11 import Headers, Request, Response

logger = logging.getLogger(__name__)

_READY = {
    "jsonrpc": "2.0",
    "method": "event",
    "params": {"type": "gateway.ready", "payload": {"skin": {}}},
}
_METHODS = ["list", "summarize", "close", "reopen"]
_SAFE_REMOTE_CODES = frozenset({
    "conversation_too_large",
    "idempotency_conflict",
    "revision_conflict",
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
_MAX_SUMMARY_CHARS = 2_000
_MAX_LIST_ITEMS = 12


@dataclass(frozen=True)
class BeckyLoopsConfig:
    enabled: bool
    chat_id: str
    token: str
    port: int = 9_120
    topic_control: str = "unavailable"


class BeckyLoopsStore(Protocol):
    def list_topics(self, chat_id: str) -> list[dict[str, Any]]: ...

    def get_topic(self, source_ref: str) -> dict[str, Any] | None: ...

    def transcript(self, session_id: str) -> list[dict[str, Any]]: ...


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
    for hidden in sorted(
        (item for item in hidden_values if item), key=len, reverse=True
    ):
        text = re.sub(re.escape(hidden), "[REDACTED]", text, flags=re.IGNORECASE)
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

    def __init__(self, db: Any) -> None:
        self._db = db
        self._source_rows: dict[str, dict[str, Any]] = {}
        self._chat_id = ""

    def list_topics(self, chat_id: str) -> list[dict[str, Any]]:
        self._chat_id = str(chat_id)
        self._source_rows = {}
        rows = self._db.list_sessions_rich(
            source="telegram",
            include_children=False,
            include_archived=False,
            project_compression_tips=True,
            order_by_last_active=True,
            limit=200,
        )
        result: list[dict[str, Any]] = []
        for raw in rows:
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
            source_state = "active" if raw.get("ended_at") is None else "closed"
            item = {
                "source_ref": ref,
                "title": _bounded_text(
                    raw.get("title") or raw.get("preview") or "Telegram loop", 128
                ),
                "source_state": source_state,
                "revision": revision_for({**raw, "source_ref": ref}, transcript),
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
            }
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


class BeckyLoopsBridgeServer:
    """A single-process, authenticated WebSocket server for Becky."""

    def __init__(self, *, config: BeckyLoopsConfig, store: BeckyLoopsStore) -> None:
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
                    await connection.send(self._error(None, "protocol"))
                    await connection.close(code=1009, reason="request too large")
                    return
                response = await self._dispatch(frame)
                encoded = json.dumps(
                    response, ensure_ascii=False, separators=(",", ":")
                )
                if len(encoded.encode()) > _MAX_RESPONSE_BYTES:
                    await connection.send(self._error(None, "protocol"))
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
                request_id if isinstance(request_id, int) else None, "protocol"
            )
        method = request["method"]
        params = request["params"]
        try:
            result = await self._method(method, params)
        except _RemoteFailure as failure:
            return self._remote_error(request_id, failure.code)
        except Exception:
            logger.warning("Becky bridge method failed: %s", method, exc_info=True)
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
                "schema_version": "1",
                "summary_schema_version": "1",
                "methods": _METHODS,
                "topic_control": "unavailable",
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
            current_revision = str(row.get("revision") or "")
            if current_revision != expected_revision:
                raise _RemoteFailure("revision_conflict")
            return self._summary(row, transcript)
        if method == "becky.loops.close":
            if not self._valid_close_params(params):
                return _PROTOCOL_FAILURE
            raise _RemoteFailure("topic_control_unavailable")
        if method == "becky.loops.reopen":
            if not self._valid_reopen_params(params):
                return _PROTOCOL_FAILURE
            raise _RemoteFailure("topic_control_unavailable")
        return _PROTOCOL_FAILURE

    def _find_topic(self, source_ref: str) -> dict[str, Any] | None:
        rows = self.store.list_topics(self.config.chat_id)
        return next((row for row in rows if row.get("source_ref") == source_ref), None)

    def _public_index(self, row: dict[str, Any]) -> dict[str, Any]:
        hidden_values = {
            self.config.chat_id,
            str(row.get("session_id") or ""),
            str(row.get("thread_id") or ""),
            str(row.get("source_ref") or ""),
        }
        return {
            "source_ref": row["source_ref"],
            "title": _safe_public_text(
                row.get("title") or "Telegram loop", hidden_values, 128
            ),
            "source_state": row.get("source_state", "active"),
            "revision": row["revision"],
            "message_count": max(0, int(row.get("message_count") or 0)),
            "created_at": _utc_datetime(row.get("created_at")).isoformat(),
            "updated_at": _utc_datetime(row.get("updated_at")).isoformat(),
            "telegram_url": None,
        }

    def _summary(
        self, row: dict[str, Any], transcript: list[dict[str, Any]]
    ) -> dict[str, Any]:
        hidden_values = {
            self.config.chat_id,
            str(row.get("session_id") or ""),
            str(row.get("thread_id") or ""),
            str(row.get("source_ref") or ""),
        }
        texts = [
            (_safe_public_text(message.get("content"), hidden_values, 900), message)
            for message in transcript
            if isinstance(message.get("content"), str)
            and _bounded_text(message.get("content"), 900)
        ]
        user_texts = [text for text, message in texts if message.get("role") == "user"]
        assistant_texts = [
            text for text, message in texts if message.get("role") == "assistant"
        ]
        summary_source = " ".join(text for text, _ in texts[-4:]).strip()
        summary = _safe_public_text(
            summary_source or "No summary text was available.",
            hidden_values,
            _MAX_SUMMARY_CHARS,
        )
        decisions = [
            text[:500]
            for text in user_texts + assistant_texts
            if re.search(
                r"\b(decid|agreed|choose|selected|will use|approved)\w*\b", text, re.I
            )
        ][-(_MAX_LIST_ITEMS):]
        unresolved = [text[:500] for text in user_texts if "?" in text][
            -(_MAX_LIST_ITEMS):
        ]
        last_message = texts[-1][1] if texts else {}
        waiting_on = "user" if last_message.get("role") == "assistant" else "becky"
        key_events = [
            {
                "occurred_at": _utc_datetime(message.get("timestamp")).isoformat(),
                "text": text[:500],
            }
            for text, message in texts[-_MAX_LIST_ITEMS:]
        ]
        return {
            "schema_version": "1",
            "source_ref": row["source_ref"],
            "revision": row["revision"],
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": summary,
            "decisions": decisions,
            "unresolved_items": unresolved,
            "next_action": user_texts[-1][:500]
            if user_texts and last_message.get("role") == "user"
            else None,
            "waiting_on": waiting_on,
            "key_events": key_events,
            "final_outcome": assistant_texts[-1][:1000] if assistant_texts else None,
        }

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
        return (
            set(params) == {"source_ref", "idempotency_key", "context"}
            and isinstance(params.get("source_ref"), str)
            and _SOURCE_REF_RE.fullmatch(params["source_ref"]) is not None
            and isinstance(params.get("idempotency_key"), str)
            and _UUID_RE.fullmatch(params["idempotency_key"]) is not None
            and isinstance(context, dict)
            and set(context)
            == {"title", "summary", "decisions", "unresolved_items", "final_outcome"}
            and isinstance(context.get("title"), str)
            and 1 <= len(context["title"]) <= 128
            and isinstance(context.get("summary"), str)
            and 1 <= len(context["summary"]) <= 2000
            and isinstance(context.get("decisions"), list)
            and len(context["decisions"]) <= 12
            and isinstance(context.get("unresolved_items"), list)
            and len(context["unresolved_items"]) <= 12
            and (
                context.get("final_outcome") is None
                or isinstance(context.get("final_outcome"), str)
            )
        )

    @staticmethod
    def _error(request_id: int | None, code: str) -> str:
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32600, "message": code},
            },
            separators=(",", ":"),
        )

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
    if not isinstance(section, dict) or not section.get("enabled"):
        return None
    token = os.getenv("HERMES_BECKY_LOOPS_TOKEN", "").strip()
    chat_id = str(section.get("telegram_chat_id", "")).strip()
    try:
        port = int(section.get("port", 9_120))
    except (TypeError, ValueError):
        port = 9_120
    return BeckyLoopsConfig(
        enabled=True,
        chat_id=chat_id,
        token=token,
        port=port,
        # No mutation adapter is installed in this deployment.  Never
        # advertise a configured-but-unimplemented control method.
        topic_control="unavailable",
    )


async def start_becky_loops_bridge(
    *, config: BeckyLoopsConfig | None, db: Any
) -> BeckyLoopsBridgeServer | None:
    """Start the opt-in bridge and return its lifecycle handle."""
    if config is None or not config.enabled:
        return None
    try:
        store = SessionDBBeckyLoopsStore(db)
        store._chat_id = config.chat_id
        server = BeckyLoopsBridgeServer(config=config, store=store)
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
