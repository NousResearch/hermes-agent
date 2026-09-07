"""Internal Responses WebSocket driver for the direct GPT-6 Astra route.

The driver is deliberately small and synchronous.  Hermes owns the receive loop and
the existing Responses assembler; the socket is only a transport for the provider's
``response.create``/``response.steer`` events.  No other Responses route is eligible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlsplit
from typing import Any, Callable

logger = logging.getLogger(__name__)


_STEERING_MODEL_CONFIG_KEY = "_astra_steering"
_STEERING_JOURNAL_VERSION = 1
_STEERING_JOURNAL_LIMIT = 16
_STEERING_UNRESOLVED_STATES = frozenset({"prepared", "sent", "failed", "accepted", "ambiguous", "fallback_queued"})
_STEERING_STATE_RANK = {
    "prepared": 10, "sent": 20, "failed": 25, "accepted": 30,
    "ambiguous": 40, "successor_created": 50, "fallback_queued": 60,
    "fallback_delivered": 70,
}


class AstraSteeringPersistenceError(RuntimeError):
    """The durable steering admission record could not be written before dispatch."""


class AstraPreDispatchError(RuntimeError):
    """The WebSocket lane failed before the initial request could be sent."""


class AstraDeliveryUncertainError(RuntimeError):
    """The provider may own bytes already sent; retrying would be unsafe."""

    delivery_uncertain = True


class AstraProtocolError(RuntimeError):
    """The provider returned an explicit protocol/error condition."""


def _base_url_is_official(base_url: Any) -> bool:
    try:
        parsed = urlsplit(str(base_url or "").strip())
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.hostname == "api.openai.com" and parsed.path.rstrip("/") == "/v1"


def is_astra_websocket_eligible(agent: Any, request: dict[str, Any] | None = None) -> bool:
    """Exact API-key Astra gate; all unsupported routes retain the SSE path."""
    model = str(getattr(agent, "model", "") or "").strip().lower().rsplit("/", 1)[-1]
    if getattr(agent, "api_mode", None) != "codex_responses" or model != "gpt-6-astra":
        return False
    if not _base_url_is_official(getattr(agent, "base_url", "")):
        return False
    if not isinstance(getattr(agent, "api_key", None), str) or not agent.api_key.strip():
        return False
    auth_mode = str(getattr(agent, "auth_mode", "api_key") or "api_key").strip().lower()
    if auth_mode not in {"", "api_key", "apikey"}:
        return False
    if getattr(agent, "provider", None) in {"openai-codex", "xai-oauth", "azure", "azure-foundry"}:
        return False
    if callable(getattr(agent, "_is_codex_backend", None)) and agent._is_codex_backend():
        return False
    if getattr(agent, "is_subagent", False) or getattr(agent, "compression_checkpoint_required", False):
        return False
    request = request or {}
    if (request.get("context_management") or request.get("conversation") or request.get("conversation_id")
            or request.get("previous_response_id")):
        return False
    return True


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    value = event.get(name, default) if isinstance(event, dict) else getattr(event, name, default)
    return default if value is None else value


def _event_response_id(event: Any) -> str | None:
    response = _event_field(event, "response")
    raw = _event_field(event, "response_id") or _event_field(response, "id")
    return str(raw).strip() if raw else None


def _event_item_id(event: Any) -> str | None:
    item = _event_field(event, "item")
    raw = _event_field(event, "item_id") or _event_field(item, "id")
    return str(raw).strip() if raw else None


def _ws_url(base_url: str) -> str:
    return "wss://api.openai.com/v1/responses"


def _default_connect(url: str, api_key: str, timeout: float):
    from websockets.sync.client import connect

    return connect(
        url,
        additional_headers={
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "responses_websockets=2026-02-06",
        },
        open_timeout=timeout,
    )


def _append_pending(agent: Any, text: str) -> None:
    lock = getattr(agent, "_pending_steer_lock", None)
    context = lock if lock is not None else threading.Lock()
    with context:
        existing = getattr(agent, "_pending_steer", None)
        agent._pending_steer = f"{existing}\n{text}" if existing else text


def _remove_pending(agent: Any, text: str) -> None:
    lock = getattr(agent, "_pending_steer_lock", None)
    context = lock if lock is not None else threading.Lock()
    with context:
        existing = getattr(agent, "_pending_steer", None)
        if not existing:
            return
        if existing == text:
            agent._pending_steer = None
        elif existing.startswith(text + "\n"):
            agent._pending_steer = existing[len(text) + 1:] or None
        else:
            parts = existing.splitlines()
            try:
                parts.remove(text)
            except ValueError:
                return
            agent._pending_steer = "\n".join(parts) or None


def _parse_model_config(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _queue_failed_redirects(agent: Any, entries: list[dict[str, Any]]) -> None:
    """Keep exact admission identities on Hermes' internal redirect queue until persistence."""
    lock = getattr(agent, "_pending_redirect_lock", None) or threading.RLock()
    with lock:
        queued = getattr(agent, "_astra_pending_redirect_receipts", None) or []
        drained = getattr(agent, "_astra_drained_redirect_receipts", None) or []
        known = {item["admission_id"] for item in [*queued, *drained]}
        for entry in entries:
            if entry.get("state") not in {"failed", "fallback_queued"}:
                continue
            admission_id, text = entry.get("admission_id"), entry.get("text")
            if not admission_id or not isinstance(text, str) or not text or admission_id in known:
                continue
            existing = getattr(agent, "_pending_redirect", None)
            agent._pending_redirect = f"{existing}\n\n[Additional user correction]\n{text}" if existing else text
            queued.append({"admission_id": admission_id, "input_sha256": entry.get("input_sha256", "")})
            known.add(admission_id)
        agent._astra_pending_redirect_receipts = queued


def restore_astra_fallback_redirects(agent: Any) -> None:
    """Restore before iteration preparation, including after a crash with no tool result."""
    if not is_astra_websocket_eligible(agent):
        return
    session_id = str(getattr(agent, "session_id", "") or "")
    if getattr(agent, "_astra_fallback_restore_session", None) == session_id:
        return
    getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
    if not session_id or not callable(getter):
        return
    raw = getter(session_id, _STEERING_MODEL_CONFIG_KEY, {})
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    _queue_failed_redirects(agent, [entry for entry in entries if isinstance(entry, dict)])
    agent._astra_fallback_restore_session = session_id


def confirm_astra_redirect_persisted(agent: Any, receipts: list[dict[str, Any]]) -> None:
    """Do not send another request until the user row and its acknowledgement are durable."""
    if not receipts:
        return
    getter = getattr(getattr(agent, "_session_db", None), "get_session_model_config_value", None)
    if not callable(getter):
        raise AstraSteeringPersistenceError("Astra fallback redirect has no durable session store")
    raw = getter(agent.session_id, _STEERING_MODEL_CONFIG_KEY, {})
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    saved = {entry.get("admission_id") for entry in entries
             if isinstance(entry, dict) and entry.get("state") == "fallback_delivered"}
    if any(receipt["admission_id"] not in saved for receipt in receipts):
        raise AstraSteeringPersistenceError("Astra fallback redirect was not durably delivered")
    agent._astra_drained_redirect_receipts = []


def _safe_steering_text(text: str) -> str:
    """Keep the journal useful without copying credential-like text into metadata."""
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text)
    except Exception:
        return "[steering input redacted]"


@dataclass
class _Steer:
    sequence: int
    text: str
    previous_response_id: str
    accepted: bool = False
    failed: bool = False


class AstraWebSocketSession:
    """One owner-thread receive loop plus lock-protected cross-thread steering."""

    def __init__(self, agent: Any, *, connect: Callable[..., Any] | None = None, timeout: float = 30.0) -> None:
        self.agent = agent
        self._connect = connect or _default_connect
        self.timeout = timeout
        self._state = "IDLE"
        self._state_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._interrupt = threading.Event()
        self._socket: Any = None
        self._request: dict[str, Any] = {}
        self._response_id: str | None = None
        self._assemblers: dict[str, Any] = {}
        self._steers: dict[int, _Steer] = {}
        self._next_sequence = 0
        self._await_successor = False
        self._continuations: set[str] = set()
        self._seen_events: set[str] = set()
        self._seen_sequences: set[str] = set()
        self._seen_items: set[tuple[str, str, str]] = set()
        self.delivery_uncertain = False
        self.last_error: Exception | None = None
        self._steering_receipts: list[dict[str, Any]] = []
        self._steering_receipt_lock = threading.RLock()
        self._steering_persistence_error: Exception | None = None
        self._steering_load_failed = False
        self._load_steering_receipts()

    def _session_db(self) -> Any:
        db = getattr(self.agent, "_session_db", None)
        if db is None:
            acquire = getattr(self.agent, "_get_session_db_for_recall", None)
            if callable(acquire):
                try:
                    db = acquire()
                except Exception as exc:
                    self._steering_persistence_error = exc
        return db

    @staticmethod
    def _steering_config_value(db: Any, session_id: str) -> Any:
        getter = getattr(db, "get_session_model_config_value", None)
        if callable(getter):
            return getter(session_id, _STEERING_MODEL_CONFIG_KEY, {})
        get_session = getattr(db, "get_session", None)
        if not callable(get_session):
            raise AstraSteeringPersistenceError("Astra session store lacks model metadata reads")
        row = get_session(session_id) or {}
        return _parse_model_config(row.get("model_config")).get(_STEERING_MODEL_CONFIG_KEY, {})

    def _load_steering_receipts(self) -> None:
        """Load only the bounded ownership journal; never reconstruct an in-flight send."""
        db = self._session_db()
        session_id = str(getattr(self.agent, "session_id", "") or "")
        if db is None or not session_id:
            return
        try:
            raw = self._steering_config_value(db, session_id)
            entries = raw.get("entries", []) if isinstance(raw, dict) else []
            if isinstance(entries, list):
                loaded = [dict(entry) for entry in entries if isinstance(entry, dict)]
                unresolved = any(entry.get("state") in _STEERING_UNRESOLVED_STATES for entry in loaded)
                self._steering_receipts = (
                    loaded if len(loaded) <= _STEERING_JOURNAL_LIMIT or unresolved
                    else loaded[-_STEERING_JOURNAL_LIMIT:]
                )
                self._next_sequence = max(
                    (int(entry.get("generation", 0)) for entry in self._steering_receipts
                     if isinstance(entry.get("generation", 0), int)),
                    default=0,
                )
            _queue_failed_redirects(self.agent, self._steering_receipts)
        except Exception as exc:
            self._steering_persistence_error = exc
            self._steering_load_failed = True
            logger.warning("Astra steering ownership journal read failed; native steering is disabled")

    def _persist_steering_receipts(self) -> None:
        """Atomically replace the small journal in the existing session model metadata."""
        db = self._session_db()
        session_id = str(getattr(self.agent, "session_id", "") or "")
        if self._steering_load_failed:
            raise AstraSteeringPersistenceError("Astra steering ownership journal could not be read") from self._steering_persistence_error
        if db is None or not session_id:
            raise AstraSteeringPersistenceError("Astra steering ownership journal is unavailable") from self._steering_persistence_error
        ensure = getattr(self.agent, "_ensure_db_session", None)
        if callable(ensure) and not bool(getattr(self.agent, "_session_db_created", True)):
            ensure()
            if not bool(getattr(self.agent, "_session_db_created", False)):
                raise AstraSteeringPersistenceError("Astra session row is not durable")
        patch = getattr(db, "patch_session_model_config", None)
        if not callable(patch):
            raise AstraSteeringPersistenceError("Astra session store lacks model metadata patching")
        with self._steering_receipt_lock:
            if len(self._steering_receipts) > _STEERING_JOURNAL_LIMIT:
                raise AstraSteeringPersistenceError("Astra steering ownership journal exceeds its bounded limit")
            payload = {
                "version": _STEERING_JOURNAL_VERSION,
                "entries": [dict(entry) for entry in self._steering_receipts[-_STEERING_JOURNAL_LIMIT:]],
            }
        try:
            patch(session_id, {_STEERING_MODEL_CONFIG_KEY: payload})
            stored = self._steering_config_value(db, session_id)
            if not isinstance(stored, dict) or stored.get("version") != _STEERING_JOURNAL_VERSION \
                    or stored.get("entries") != payload["entries"]:
                raise AstraSteeringPersistenceError("Astra session metadata write was not durable")
            self._steering_persistence_error = None
        except Exception as exc:
            self._steering_persistence_error = exc
            raise AstraSteeringPersistenceError("Astra steering ownership admission could not be persisted") from exc

    def _record_steering_state(self, sequence: int, state: str) -> bool:
        with self._steering_receipt_lock:
            for entry in reversed(self._steering_receipts):
                if entry.get("generation") == sequence:
                    old_state = str(entry.get("state") or "prepared")
                    if _STEERING_STATE_RANK.get(state, 0) >= _STEERING_STATE_RANK.get(old_state, 0):
                        entry["state"] = state
                    break
            try:
                self._persist_steering_receipts()
                return True
            except Exception:
                # A pre-dispatch record already exists. Never turn an accepted/ambiguous
                # wire outcome into a retry merely because its later state stamp failed.
                logger.warning("Astra steering ownership state update failed", exc_info=True)
                return False

    def _append_steering_receipt(self, sequence: int, text: str, previous_id: str) -> None:
        safe_text = _safe_steering_text(text)
        if safe_text != text:
            raise AstraSteeringPersistenceError("Astra steering input would be changed by redaction")
        wire_input = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
        input_digest = hashlib.sha256(
            json.dumps(
                [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        session_id = str(getattr(self.agent, "session_id", "") or "")
        admission_id = hashlib.sha256(
            f"{session_id}\0{sequence}\0{previous_id}".encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:24]
        with self._steering_receipt_lock:
            if (
                len(self._steering_receipts) >= _STEERING_JOURNAL_LIMIT
                and any(entry.get("state") in _STEERING_UNRESOLVED_STATES for entry in self._steering_receipts)
            ):
                raise AstraSteeringPersistenceError("Astra steering ownership journal is full of unresolved inputs")
            self._steering_receipts.append({
                "version": _STEERING_JOURNAL_VERSION,
                "admission_id": admission_id,
                "generation": sequence,
                "response_id": previous_id,
                "input": wire_input,
                "input_sha256": input_digest,
                "text": safe_text,
                "state": "prepared",
            })
            del self._steering_receipts[:-_STEERING_JOURNAL_LIMIT]
        self._persist_steering_receipts()

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def response_id(self) -> str | None:
        with self._state_lock:
            return self._response_id

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state

    def _send(self, payload: dict[str, Any]) -> None:
        if self._socket is None:
            raise AstraDeliveryUncertainError("Astra WebSocket is not open")
        wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._send_lock:
            try:
                self._socket.send(wire)
            except Exception as exc:
                self.delivery_uncertain = True
                self.last_error = exc
                self._set_state("POST_DISPATCH_AMBIGUOUS")
                raise AstraDeliveryUncertainError("Astra WebSocket send outcome is uncertain") from exc

    def _append_steer(self, text: str) -> int:
        with self._state_lock:
            self._next_sequence += 1
            sequence = self._next_sequence
            self._steers[sequence] = _Steer(sequence, text, self._response_id or "")
            self._await_successor = True
            _append_pending(self.agent, text)
            self._set_state("STEER_ADMITTED")
            return sequence

    def request_steer(self, text: str) -> bool:
        """Admit and send one steer, returning False when no active response exists."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        with self._state_lock:
            if self._socket is None or self._state not in {"ACTIVE", "SUCCESSOR_CREATED", "ACCEPTED"} or not self._response_id:
                return False
            previous_id = self._response_id
            previous_state = self._state
            previous_await_successor = self._await_successor
            sequence = self._append_steer(cleaned)
            try:
                # Ownership is durable before any provider bytes leave the socket.
                self._append_steering_receipt(sequence, cleaned, previous_id)
            except Exception:
                with self._steering_receipt_lock:
                    self._steering_receipts = [
                        entry for entry in self._steering_receipts
                        if entry.get("generation") != sequence
                    ]
                self._steers.pop(sequence, None)
                self._await_successor = previous_await_successor
                self._set_state(previous_state)
                _remove_pending(self.agent, cleaned)
                raise
            wire_input = [{"role": "user", "content": [{"type": "input_text", "text": cleaned}]}]
        try:
            # Responses steering deliberately has a tiny wire shape. In particular, stream_id is never sent.
            self._send({
                "type": "response.steer", "previous_response_id": previous_id,
                "input": wire_input,
            })
        except AstraDeliveryUncertainError:
            # The provider may have accepted the input; leaving it in Hermes' fallback queue could duplicate it.
            _remove_pending(self.agent, cleaned)
            self._record_steering_state(sequence, "ambiguous")
            raise
        self._record_steering_state(sequence, "sent")
        return bool(sequence)

    def request_interrupt(self) -> None:
        self._interrupt.set()
        self.close()

    def close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            with self._send_lock:
                try:
                    socket.close()
                except Exception:
                    logger.debug("Astra WebSocket close failed", exc_info=True)

    def _new_assembler(self, response_id: str):
        from agent.codex_runtime import _CodexResponseAssembler

        return _CodexResponseAssembler(
            model=self._request.get("model", getattr(self.agent, "model", "")),
            on_text_delta=self._on_text_delta,
            on_reasoning_delta=self._on_reasoning_delta,
            on_commentary_message=self._on_commentary_message,
            on_first_delta=self._on_first_delta,
            on_async_tool_call=self._async_tool_call,
            on_async_tool_announcement=self._async_tool_announcement,
        )

    def _on_text_delta(self, text: str) -> None:
        if not text:
            return
        self.agent._codex_streamed_text_parts.append(text)
        callback = getattr(self.agent, "_fire_stream_delta", None)
        if callable(callback):
            callback(text)

    def _on_reasoning_delta(self, text: str) -> None:
        callback = getattr(self.agent, "_fire_reasoning_delta", None)
        if callable(callback):
            callback(text)

    def _on_commentary_message(self, text: str) -> None:
        callback = getattr(self.agent, "_fire_streamed_codex_commentary", None)
        if callable(callback):
            callback(text)

    def _on_first_delta(self) -> None:
        callback = getattr(self, "_first_delta", None)
        if callable(callback):
            callback()

    @property
    def _first_delta(self):
        return getattr(self, "_on_first_delta_callback", None)

    def _async_tool_call(self, call: Any) -> None:
        callback = getattr(getattr(self.agent, "_astra_async_executor", None), "admit", None)
        if callable(callback):
            callback(call)

    def _async_tool_announcement(self, call: Any) -> None:
        callback = getattr(getattr(self.agent, "_astra_async_executor", None), "reserve", None)
        if callable(callback):
            callback(call)

    def _saved_tool_result(self, call_id: str) -> Any:
        messages = getattr(self.agent, "_session_messages", None) or []
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            stored_id = str(message.get("tool_call_id") or "").split("|", 1)[0]
            if stored_id == call_id:
                return message.get("content", "")
        return None

    def _fill_required_input(self, required: Any) -> list[Any]:
        if not isinstance(required, list):
            raise AstraProtocolError("Astra steer.pending did not provide required_input")
        filled = []
        for stub in required:
            if not isinstance(stub, dict):
                raise AstraProtocolError("Astra required_input item is not an object")
            item = dict(stub)
            call_id = str(item.get("call_id") or item.get("id") or "").strip()
            result = self._saved_tool_result(call_id) if call_id else None
            if result is None:
                raise AstraProtocolError(f"No saved tool result for required call {call_id or '<missing>'}")
            if "result" in item:
                item["result"] = result
            else:
                item["output"] = result
            filled.append(item)
        return filled

    def _send_required_continuation(self, event: Any) -> None:
        parent = _event_response_id(event) or self._response_id
        if not parent or parent in self._continuations:
            return
        required = _event_field(event, "required_input")
        if required is None:
            response = _event_field(event, "response")
            required = _event_field(response, "required_input")
        executor = getattr(self.agent, "_astra_async_executor", None)
        settle_required = getattr(executor, "settle_required", None)
        if callable(settle_required):
            required_items = required if isinstance(required, list) else ()
            call_ids = [
                str(item.get("call_id") or item.get("id") or "").strip()
                for item in required_items if isinstance(item, dict)
            ]
            if isinstance(required, list) and not settle_required(call_ids):
                raise AstraProtocolError("Astra async tool results were not durably settled")
        input_items = self._fill_required_input(required)
        settings = {k: v for k, v in self._request.items() if k not in {"type", "input", "stream", "previous_response_id"}}
        self._send({"type": "response.create", **settings, "previous_response_id": parent, "input": input_items})
        self._continuations.add(parent)
        self._await_successor = True

    def _is_duplicate(self, event: Any, event_type: str, response_id: str | None) -> bool:
        sequence_number = _event_field(event, "sequence_number")
        if sequence_number is not None:
            # Sequence numbers are monotonic only within one response generation; automatic successors may
            # restart at zero/one on the same WebSocket. Scope the key before deciding whether to drop a frame.
            sequence_scope = response_id or self._response_id or "session"
            sequence_key = f"{sequence_scope}:{sequence_number}"
            if sequence_key in self._seen_sequences:
                return True
            self._seen_sequences.add(sequence_key)
        event_id = _event_field(event, "event_id") or _event_field(event, "id")
        if event_id:
            key = str(event_id)
            if key in self._seen_events:
                return True
            self._seen_events.add(key)
        if event_type.endswith("output_item.done"):
            item_id = _event_item_id(event)
            if item_id and response_id:
                key = (response_id, item_id, event_type)
                if key in self._seen_items:
                    return True
                self._seen_items.add(key)
        return False

    def _event_belongs_to_active_response(self, event: Any, response_id: str | None, event_type: str) -> bool:
        if not response_id or not self._response_id or response_id == self._response_id:
            return True
        return event_type == "response.created" and self._await_successor

    def _handle_created(self, event: Any, response_id: str | None) -> None:
        if not response_id:
            raise AstraProtocolError("Astra response.created omitted response id")
        with self._state_lock:
            if self._response_id is None:
                self._response_id = response_id
                self._assemblers[response_id] = self._new_assembler(response_id)
                self._set_state("ACTIVE")
                return
            if response_id == self._response_id:
                return
            if not self._await_successor:
                return
            predecessor_id = self._response_id
            successor_sequences = [
                item.sequence for item in self._steers.values()
                if item.previous_response_id == predecessor_id and not item.failed
            ]
            self._response_id = response_id
            self._assemblers[response_id] = self._new_assembler(response_id)
            self._await_successor = False
            self._set_state("SUCCESSOR_CREATED")
        # A successor is the provider's durable ownership receipt. Once observed,
        # reconstruction must never put these steer inputs back in the fallback queue.
        for sequence in successor_sequences:
            self._record_steering_state(sequence, "successor_created")

    def _steer_event(self, event: Any, event_type: str) -> None:
        if event_type == "response.steer.accepted":
            sequence = None
            with self._state_lock:
                pending = next((item for item in self._steers.values() if not item.accepted and not item.failed), None)
                if pending is not None:
                    pending.accepted = True
                    sequence = pending.sequence
                    _remove_pending(self.agent, pending.text)
                self._set_state("ACCEPTED")
            if sequence is not None:
                self._record_steering_state(sequence, "accepted")
        elif event_type == "response.steer.failed":
            sequence = None
            with self._state_lock:
                pending = next((item for item in self._steers.values() if not item.accepted and not item.failed), None)
                if pending is not None:
                    pending.failed = True
                    sequence = pending.sequence
                    _remove_pending(self.agent, pending.text)
                self._await_successor = any(not item.failed and not item.accepted for item in self._steers.values())
                self._set_state("ACTIVE")
            if sequence is not None:
                # Redirect stays internal to the agent loop even when there is no tool
                # result. Its user row acknowledges this receipt in one DB transaction.
                if not self._record_steering_state(sequence, "failed"):
                    raise AstraSteeringPersistenceError("Astra rejection could not be persisted")
                _queue_failed_redirects(self.agent, self._steering_receipts)
        elif event_type == "response.steer.pending":
            self._set_state("PENDING_REQUIRED_INPUT")
            self._send_required_continuation(event)

    def _terminal_reason(self, event: Any) -> str:
        response = _event_field(event, "response")
        details = _event_field(response, "incomplete_details") or _event_field(event, "incomplete_details") or {}
        return str(_event_field(details, "reason", "") or "").strip().lower()

    def _settle_async_executor(self, final: Any) -> None:
        """Use the PR2 persist-before-execute settlement boundary for the final WS response."""
        executor = getattr(self.agent, "_astra_async_executor", None)
        if executor is None:
            return
        if getattr(executor, "has_admitted", False) or getattr(executor, "has_pending", False) or getattr(executor, "failed", False):
            if not executor.finish_stream(
                assistant_content=getattr(final, "output_text", "") or "",
                settled_calls=getattr(final, "output", None),
            ):
                raise RuntimeError("Astra async tool execution did not reach a durable result boundary")
        elif executor.retire_empty():
            self.agent._astra_async_executor = None

    def run(self, request: dict[str, Any], *, on_first_delta: Callable[[], None] | None = None) -> Any:
        from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current

        self._request = dict(request)
        self._request.pop("stream", None)
        self._on_first_delta_callback = on_first_delta
        self.agent._codex_streamed_text_parts = []
        writer_token = claim_stream_writer(self.agent)
        try:
            self._set_state("DISPATCHING")
            try:
                self._socket = self._connect(_ws_url(str(getattr(self.agent, "base_url", ""))), self.agent.api_key, self.timeout)
            except Exception as exc:
                self.last_error = exc
                self._set_state("PRE_DISPATCH_FAILURE")
                raise AstraPreDispatchError("Astra WebSocket connection failed before dispatch") from exc
            initial = {"type": "response.create", **{k: v for k, v in self._request.items() if k != "type"}}
            try:
                self._send(initial)
            except AstraDeliveryUncertainError:
                raise
            while True:
                if self._interrupt.is_set() or getattr(self.agent, "_interrupt_requested", False):
                    raise InterruptedError("Astra WebSocket turn interrupted")
                try:
                    raw = self._socket.recv()
                except Exception as exc:
                    if self._interrupt.is_set() or getattr(self.agent, "_interrupt_requested", False):
                        raise InterruptedError("Astra WebSocket turn interrupted") from exc
                    self.delivery_uncertain = True
                    self.last_error = exc
                    self._set_state("POST_DISPATCH_AMBIGUOUS")
                    raise AstraDeliveryUncertainError("Astra WebSocket response delivery is uncertain") from exc
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    event = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError) as exc:
                    raise AstraProtocolError("Astra WebSocket returned invalid JSON") from exc
                if not isinstance(event, dict):
                    raise AstraProtocolError("Astra WebSocket returned a non-object event")
                event_type = str(event.get("type") or "")
                response_id = _event_response_id(event)
                if self._is_duplicate(event, event_type, response_id):
                    continue
                if event_type == "response.created":
                    self._handle_created(event, response_id)
                    continue
                if event_type.startswith("response.steer."):
                    self._steer_event(event, event_type)
                    continue
                if not self._event_belongs_to_active_response(event, response_id, event_type):
                    continue
                assembler = self._assemblers.get(self._response_id or "")
                if assembler is None:
                    raise AstraProtocolError("Astra event arrived before response.created")
                self.agent._codex_stream_last_event_ts = time.time()
                touch = getattr(self.agent, "_touch_activity", None)
                if callable(touch):
                    touch("receiving Astra WebSocket response")
                if not stream_writer_is_current(self.agent, writer_token):
                    raise TimeoutError("Astra WebSocket stream was superseded")
                # Claim terminal ownership before feeding the event.  A concurrent steer that observes
                # terminal processing has begun must return False rather than dispatching after completion.
                terminal_event = event_type in {"response.completed", "response.incomplete", "response.failed"}
                terminal_waits_for_successor = False
                if terminal_event:
                    # The wait decision and terminal claim are one linearization point. A steer can either
                    # acquire this lock first (and force successor wait) or observe TERMINAL_PROCESSING and
                    # return False; it cannot be admitted against a stale pre-lock snapshot.
                    with self._state_lock:
                        terminal_waits_for_successor = self._await_successor or self._terminal_reason(event) == "steered"
                        if terminal_waits_for_successor:
                            pass
                        elif self._state in {"ACTIVE", "SUCCESSOR_CREATED", "ACCEPTED", "STEER_ADMITTED"}:
                            self._set_state("TERMINAL_PROCESSING")
                terminal = assembler.feed(event)
                if terminal:
                    if terminal_waits_for_successor:
                        continue
                    with self._state_lock:
                        self._set_state("COMPLETED" if event_type == "response.completed" else "EXPLICIT_FAILURE")
                    final = assembler.result()
                    self._settle_async_executor(final)
                    return final
        finally:
            self.close()


def run_astra_websocket_stream(agent: Any, request: dict[str, Any], *, on_first_delta=None, connect=None):
    """Attach one turn-owned session so concurrent Hermes control calls can steer it."""
    session = AstraWebSocketSession(agent, connect=connect)
    previous = getattr(agent, "_astra_websocket_session", None)
    if previous is not None:
        previous.close()
    agent._astra_websocket_session = session
    try:
        return session.run(request, on_first_delta=on_first_delta)
    finally:
        if getattr(agent, "_astra_websocket_session", None) is session:
            agent._astra_websocket_session = None


__all__ = [
    "AstraDeliveryUncertainError", "AstraPreDispatchError", "AstraProtocolError", "AstraWebSocketSession",
    "is_astra_websocket_eligible", "run_astra_websocket_stream",
]
