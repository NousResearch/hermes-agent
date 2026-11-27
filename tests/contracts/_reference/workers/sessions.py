"""Reference verbs of the worker session family (``persistence.workers.sessions.WorkerSessionVerbs``).

Mixed into ``InMemoryWorkerExecutionRepository``. Each verb builds the wire payload and runs its handler inside the receipt transaction of :mod:`.receipt`; the handler writes through the sibling repositories' own verbs on the staged tables, so the semantics have one implementation and the write publishes with its receipt.

Reference limit: a transcript row's persistence-only keys that :class:`persistence.conversation.messages.NewMessage` does not carry (``message_id``, ``_canonical_content``, ``_canonical_row``, ``_submit_row_session_id``, the alternation repair's counts, ``_absorbed_row_ids``, ``_compressed_summary_has_user_turn``) are accepted and digested but not stored.
"""

from __future__ import annotations

import json
import time
import math
import re
from typing import Any, Mapping, Optional, Sequence, Unpack

from persistence.conversation.messages import CONTENT_JSON_PREFIX, NewMessage
from persistence.conversation.sessions import NewSession, SessionLifecycleStatus, TitleSource
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.context import ProfileScope
from persistence.core.identity import JsonValue, MessageId, SessionId
from persistence.workers.executions import WorkerExecutionId
from persistence.workers.sessions import (
    RowAnnotation,
    ToolPin,
    TranscriptAppendOptions,
    WorkerActivity,
    WorkerAppendResult,
    WorkerBillingRoute,
    WorkerMessage,
    WorkerSessionCreate,
    WorkerTitleResult,
)
from persistence.workers.vocabulary import SIDECAR_KEYS, TRANSCRIPT_BATCH_MAX, WORKER_CONSTRUCTOR_CONFIG_KEYS, WORKER_MESSAGE_FIELDS

from .receipt import Handler, Step, denied, refuse, text

__all__ = ["InMemoryWorkerSessionVerbs", "append_rows", "check_rows", "new_message"]

_TITLE_SOURCES = ("user", "derived", "llm")
_SURROGATE = re.compile("[\ud800-\udfff]")
_TITLE_REFUSED = "Title unavailable or invalid"
#: The identity fields a worker create must match, and every field it may send.
_IDENTITY_FIELDS = ("user_id", "session_key", "chat_id", "chat_type", "thread_id", "parent_session_id", "cwd", "profile_name", "git_repo_root", "origin_json", "display_name")
_CREATE_FIELDS = frozenset(_IDENTITY_FIELDS) | {"source", "model", "model_config", "system_prompt"}
_ROLES = ("user", "assistant", "system", "tool")
#: Wire keys stored under another ``NewMessage`` field name.
_RENAMED = {"_compressed_summary": "compressed_summary", "_row_id": "repair_target", "_db_row_snapshot": "db_row_snapshot",
            "_absorbed_message_uids": "absorbed_message_uids", "_tool_call_uids": "tool_call_uids", "_tool_call_uid": "tool_call_uid"}
_DIRECT = ("role", "content", "timestamp", "tool_name", "tool_call_id", "display_kind", "api_content", "display_metadata", "platform_message_id",
           "observed", "effect_disposition", "finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
           "codex_message_items", "message_uid", "token_count", "data_use")
_TUPLES = ("tool_calls", "data_use", "absorbed_message_uids")


def new_message(wire: Mapping[str, Any]) -> NewMessage:
    """One wire message dict as the row ``append_batch`` writes."""
    values: dict[str, Any] = {key: wire[key] for key in _DIRECT if key in wire}
    values.update({field: wire[key] for key, field in _RENAMED.items() if key in wire})
    values["restored_from_store"] = wire.get("_restored_from_store") is True
    if wire.get("tool_calls"):
        values["tool_calls"] = wire["tool_calls"]
    for name in _TUPLES:
        if values.get(name) is not None:
            values[name] = tuple(values[name])
        else:
            values.pop(name, None)
    for name in ("compressed_summary", "observed"):
        if name in values:
            values[name] = bool(values[name])
    if values.get("repair_target") is not None:
        values["repair_target"] = MessageId(values["repair_target"])
    return NewMessage(**values)


def check_rows(step: Step, messages: Any, fields: frozenset[str]) -> None:
    """The row checks: a list of at most :data:`TRANSCRIPT_BATCH_MAX` dicts within ``fields`` with a known role, each ``_row_id`` naming a row of this transcript."""
    if not isinstance(messages, list) or len(messages) > TRANSCRIPT_BATCH_MAX:
        refuse("invalid_params")
    transcript = step.tables.get("messages", {}).get("transcripts", {}).get(step.session_id)
    ids = {row.id for row in transcript.rows} if transcript is not None else set()
    for message in messages:
        if not isinstance(message, dict) or set(message) - fields or message.get("role") not in _ROLES:
            refuse("invalid_params")
        if "_row_id" in message and message["_row_id"] not in ids:
            raise denied("message", message["_row_id"])


def append_rows(step: Step, messages: list[dict[str, Any]], *, turn_lease_holder: Optional[str], ttl: float = 300.0) -> dict[str, Any]:
    """``_append_messages_in_transaction`` on the worker's session: the batch append with its write guards; the count and one annotation per message."""
    outcome = step.store.messages.append_batch(SessionId(step.session_id), [new_message(m) for m in messages], turn_lease_holder=turn_lease_holder, turn_lease_ttl_seconds=ttl)
    return {"count": outcome.inserted, "annotations": [o.as_row_annotation() for o in outcome.outcomes]}


def append_result(result: Mapping[str, Any]) -> WorkerAppendResult:
    return WorkerAppendResult(count=int(result["count"]), annotations=tuple(RowAnnotation.from_wire(a) for a in result["annotations"]))


def _stored_origin_json(origin: Any) -> Optional[str]:
    """The ``origin_json`` text a row created with this origin holds; it is compared as exact text."""
    return json.dumps(dict(origin)) if origin else None


def append_guarded(step: Step, messages: list[dict[str, Any]], holder: Optional[str], ttl: float = 300.0) -> dict[str, Any]:
    """The guarded append: its write guards run even for an empty batch."""
    if not messages:
        step.store.messages._write_guards(step.scope, SessionId(step.session_id), turn_lease_holder=holder, ttl=ttl)
        return {"count": 0, "annotations": []}
    return append_rows(step, messages, turn_lease_holder=holder, ttl=ttl)


def _title(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"title", "source"} or payload["source"] not in _TITLE_SOURCES or not (payload["title"] is None or isinstance(payload["title"], str)):
        refuse("invalid_params")
    try:
        value = bool(step.store.sessions._write_title(step.scope, SessionId(step.session_id), payload["title"], TitleSource(payload["source"])))
    except ValueError:
        return {"value": False, "error": _TITLE_REFUSED, "title_candidate": payload["title"]}
    step.wrote = value
    return {"value": value}


def _title_source(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"source"} or payload["source"] not in _TITLE_SOURCES:
        refuse("invalid_params")
    value = step.store.sessions.set_title_source(SessionId(step.session_id), TitleSource(payload["source"]))
    step.wrote = value
    return {"value": value}


def _next_title(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"base_title"} or not isinstance(payload["base_title"], str):
        refuse("invalid_params")
    candidate = payload["base_title"]
    workers = step.authority["workers"]
    attempted = any(
        isinstance(stored["result"], dict) and stored["result"].get("title_candidate") == candidate
        and (workers.get(eid) or {}).get("session_id") == step.session_id and workers[eid]["status"] == "running"
        for (eid, _sequence), stored in step.authority["receipts"].items()
    )
    if candidate != step.sessions[step.session_id]["title"] and not attempted:
        raise denied("session_title", candidate)
    return {"value": step.store.sessions.next_title_in_lineage(candidate)}


def _activity(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"ts", "description", "provenance"} or any(payload[k] is not None and not isinstance(payload[k], str) for k in ("description", "provenance")):
        refuse("invalid_params")
    when = payload["ts"]
    if when is not None and (type(when) not in (float, int) or not math.isfinite(when)):
        refuse("invalid_params")
    step.store.sessions.touch_activity(SessionId(step.session_id), time.time() if when is None else float(when),
                                       description=payload["description"], provenance=payload["provenance"])
    return {"value": None}


def _activity_clear(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        refuse("invalid_params")
    step.store.sessions.clear_activity_labels(SessionId(step.session_id))
    step.wrote = True
    return {"value": None}


def _billing_route(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"provider", "base_url", "billing_mode"} or any(v is not None and not isinstance(v, str) for v in payload.values()):
        refuse("invalid_params")
    step.store.usage.set_billing_route(SessionId(step.session_id), provider=payload["provider"], base_url=payload["base_url"], billing_mode=payload["billing_mode"])
    step.wrote = True
    return {"value": None}


def _row_id(payload: Mapping[str, Any]) -> MessageId:
    row_id = payload.get("row_id")
    if type(row_id) is not int or row_id <= 0:
        refuse("invalid_params")
    return MessageId(row_id)


def _changed(step: Step, value: Any) -> dict[str, Any]:
    step.wrote = bool(value)
    return {"value": int(bool(value))}


class _StagedView:
    """The backend as seen from inside one authority transaction: ``scope_store`` of the transaction's scope is the staged copy, everything else the real backend. Consumer: ``_reference/authority/bindings_resets.py`` ``reset_target``."""

    def __init__(self, backend: Any, scope: ProfileScope, store: dict[str, Any]) -> None:
        self._backend = backend
        self._key = scope.hermes_home_key
        self._store = store

    def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
        if scope.hermes_home_key == self._key:
            return self._store
        store: dict[str, Any] = self._backend.scope_store(scope)
        return store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _encoded(content: Any) -> Any:
    """The value the store binds, which ``content IS ?`` compares."""
    if isinstance(content, str):
        content = _SURROGATE.sub("\ufffd", content)
        if not content.startswith(CONTENT_JSON_PREFIX):
            return content
    elif content is None or isinstance(content, (bytes, int, float)):
        return content
    try:
        return CONTENT_JSON_PREFIX + json.dumps(content)
    except (TypeError, ValueError):
        return _SURROGATE.sub("\ufffd", str(content))


def _user_row(step: Step, row_id: Optional[int], content: Any) -> Any:
    """The active user row of this session (the newest, or ``row_id``) whose stored content encodes as ``content`` does, else ``None``."""
    users = [row for row in step.store.messages.read_transcript(SessionId(step.session_id)) if row.role == "user"]
    row = (users[-1] if users else None) if row_id is None else next((r for r in users if r.id == row_id), None)
    return row if row is not None and _encoded(row.content) == _encoded(content) else None


def _api_content(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"content", "api_content"} or not isinstance(payload["api_content"], str):
        refuse("invalid_params")
    row = _user_row(step, None, payload["content"])
    return _changed(step, row is not None and step.store.messages.set_latest_user_api_content(SessionId(step.session_id), row.content, payload["api_content"]))


def _message_api_content(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"row_id", "content", "api_content"} or not isinstance(payload["api_content"], str):
        refuse("invalid_params")
    row = _user_row(step, _row_id(payload), payload["content"])
    return _changed(step, row is not None and step.store.messages.set_message_api_content(SessionId(step.session_id), row.id, row.content, payload["api_content"]))


def _user_content(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) - {"display_metadata"} != {"row_id", "content"}:
        refuse("invalid_params")
    # A sent key merges into the row's metadata; an empty or null one leaves it as it is, as the store encodes it.
    extra = {"display_metadata": payload["display_metadata"] or {}} if "display_metadata" in payload else {}
    return _changed(step, step.store.messages.set_user_message_content(SessionId(step.session_id), _row_id(payload), payload["content"], **extra))


def _create(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """Create the worker's session: the source and every sent identity field must match the stored row, then the session is created with the sent fields and model config."""
    if set(payload) - _CREATE_FIELDS or "source" not in payload:
        refuse("invalid_params")
    row = step.sessions.get(step.session_id)
    if row is None or payload["source"] != row["source"]:
        raise denied("session", step.session_id)
    for key in _IDENTITY_FIELDS:
        stored = _stored_origin_json(row.get("origin")) if key == "origin_json" else row.get(key)
        if payload.get(key) is not None and payload[key] != stored:
            raise denied("session", step.session_id)
    if any(payload.get(key) is not None and not isinstance(payload[key], str) for key in _CREATE_FIELDS - {"model_config"}):
        refuse("invalid_params")
    config = payload.get("model_config")
    if config is not None and (not isinstance(config, dict) or set(config) - WORKER_CONSTRUCTOR_CONFIG_KEYS):
        refuse("invalid_params")
    origin = payload.get("origin_json")
    fields = {key: payload.get(key) for key in _CREATE_FIELDS - {"origin_json", "source", "model_config"}}
    step.store.sessions.create(NewSession(session_id=SessionId(step.session_id), source=payload["source"], origin=json.loads(origin) if origin else None,
                                          model_config=config or None, **fields))
    if config and row["model_config"].get("_delegate_from") is not None:
        step.store.sessions.patch_model_config(SessionId(step.session_id), config)
    step.wrote = True
    return {"value": step.session_id}


def _end(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"end_reason"} or not isinstance(payload["end_reason"], str) or not payload["end_reason"]:
        refuse("invalid_params")
    step.store.sessions.end(SessionId(step.session_id), payload["end_reason"])
    return {"value": None}


def _lifecycle(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        refuse("invalid_params")
    statuses = step.store.sessions.lifecycle_statuses([SessionId(step.session_id)])
    return {"value": statuses.get(SessionId(step.session_id), SessionLifecycleStatus.EMPTY).value}


def _prompt(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"system_prompt"} or not (payload["system_prompt"] is None or isinstance(payload["system_prompt"], str)):
        refuse("invalid_params")
    step.store.sessions.update_system_prompt(SessionId(step.session_id), payload["system_prompt"])
    step.wrote = True
    return {"value": None}


def _sidecars(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"patch"} or not isinstance(payload["patch"], dict) or set(payload["patch"]) - SIDECAR_KEYS:
        refuse("invalid_params")
    step.store.sessions.patch_model_config(SessionId(step.session_id), payload["patch"])
    step.wrote = True
    return {"value": None}


def _tools(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and store the session's tool pin: a list of tool names or a versioned pin."""
    if set(payload) != {"tool_names"}:
        refuse("invalid_params")
    pin = payload["tool_names"]
    if pin is not None:
        if not isinstance(pin, (list, dict)):
            refuse("invalid_params")
        if isinstance(pin, list) and not all(isinstance(n, str) and 0 < len(n) <= 256 for n in pin):
            refuse("invalid_params")
        backend = pin.get("delegation_backend", "") if isinstance(pin, dict) else ""
        if isinstance(pin, dict) and (not {"version", "tools"} <= set(pin) <= {"version", "tools", "delegation_backend"}
                                      or not isinstance(pin["version"], str) or not isinstance(pin["tools"], list)
                                      or not isinstance(backend, str) or len(backend) > 64):
            refuse("invalid_params")
    if pin is not None and len(json.dumps(pin)) > 4 * 1024 * 1024:
        refuse("invalid_params")
    step.store.sessions.update_tool_pin(SessionId(step.session_id), pin)
    step.wrote = True
    return {"value": None}


def _tail_role(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        refuse("invalid_params")
    return {"value": step.store.messages.latest_conversation_role(SessionId(step.session_id))}


def _row_role(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"row_id"} or type(payload["row_id"]) is not int:
        refuse("invalid_params")
    return {"value": step.store.messages.message_role(SessionId(step.session_id), MessageId(payload["row_id"]))}


def _append(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """The worker append: the payload checks, then the guarded append."""
    if set(payload) - {"messages", "turn_lease_holder"} or "messages" not in payload:
        refuse("invalid_params")
    check_rows(step, payload["messages"], WORKER_MESSAGE_FIELDS)
    holder = payload.get("turn_lease_holder")
    if holder is not None:
        text(holder)
    return append_guarded(step, payload["messages"], holder)


class InMemoryWorkerSessionVerbs:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_apply``)."""

    def _session_op(self, operation: str, payload: Any, handler: Handler, epoch: Any, execution_id: Any, session_id: Any, generation: Any, sequence: Any) -> Any:
        return self._apply(operation, payload, handler, epoch=epoch, execution_id=execution_id, session_id=session_id,  # type: ignore[attr-defined]
                           generation=generation, sequence=sequence)["value"]

    def set_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, title: str | None, source: TitleSource) -> WorkerTitleResult:
        result = self._apply("session.title", {"title": title, "source": getattr(source, "value", source)}, _title, epoch=epoch,  # type: ignore[attr-defined]
                             execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)
        return WorkerTitleResult(value=bool(result["value"]), error=result.get("error"), title_candidate=result.get("title_candidate"))

    def set_session_title_source(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, source: TitleSource) -> bool:
        return bool(self._session_op("session.title_source", {"source": getattr(source, "value", source)}, _title_source, epoch, execution_id, session_id, generation, sequence))

    def next_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, base_title: str) -> str:
        return str(self._session_op("session.next_title", {"base_title": base_title}, _next_title, epoch, execution_id, session_id, generation, sequence))

    def touch_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, activity: WorkerActivity) -> None:
        payload = {"ts": activity.ts, "description": activity.description, "provenance": activity.provenance}
        self._session_op("session.activity", payload, _activity, epoch, execution_id, session_id, generation, sequence)

    def clear_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._session_op("session.activity_clear", {}, _activity_clear, epoch, execution_id, session_id, generation, sequence)

    def set_session_billing_route(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, route: WorkerBillingRoute) -> None:
        payload = {"provider": route.provider, "base_url": route.base_url, "billing_mode": route.billing_mode}
        self._session_op("session.billing_route", payload, _billing_route, epoch, execution_id, session_id, generation, sequence)

    def set_latest_user_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, content: JsonValue, api_content: str) -> int:
        return int(self._session_op("session.api_content", {"content": content, "api_content": api_content}, _api_content, epoch, execution_id, session_id, generation, sequence))

    def set_message_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, api_content: str) -> int:
        payload = {"row_id": row_id, "content": content, "api_content": api_content}
        return int(self._session_op("session.message_api_content", payload, _message_api_content, epoch, execution_id, session_id, generation, sequence))

    def set_user_message_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> int:
        payload: dict[str, Any] = {"row_id": row_id, "content": content}
        if display_metadata is not None:
            payload["display_metadata"] = dict(display_metadata)
        return int(self._session_op("session.user_content", payload, _user_content, epoch, execution_id, session_id, generation, sequence))

    def create_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **fields: Unpack[WorkerSessionCreate]) -> SessionId:
        return SessionId(self._session_op("session.create", dict(fields), _create, epoch, execution_id, session_id, generation, sequence))

    def end_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, end_reason: str) -> None:
        self._session_op("session.end", {"end_reason": end_reason}, _end, epoch, execution_id, session_id, generation, sequence)

    def session_lifecycle(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> SessionLifecycleStatus:
        return SessionLifecycleStatus(self._session_op("session.lifecycle", {}, _lifecycle, epoch, execution_id, session_id, generation, sequence))

    def set_session_system_prompt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, system_prompt: str | None) -> None:
        self._session_op("session.prompt", {"system_prompt": system_prompt}, _prompt, epoch, execution_id, session_id, generation, sequence)

    def patch_session_sidecars(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, patch: Mapping[str, JsonValue]) -> None:
        self._session_op("session.sidecars", {"patch": dict(patch) if isinstance(patch, Mapping) else patch}, _sidecars, epoch, execution_id, session_id, generation, sequence)

    def set_session_tool_pin(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, tool_pin: ToolPin | None) -> None:
        self._session_op("session.tools", {"tool_names": dict(tool_pin) if isinstance(tool_pin, Mapping) else tool_pin}, _tools, epoch, execution_id, session_id, generation, sequence)

    def latest_conversation_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        return self._session_op("session.tail_role", {}, _tail_role, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def message_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: int) -> str | None:
        return self._session_op("session.row_role", {"row_id": row_id}, _row_role, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def append_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], **options: Unpack[TranscriptAppendOptions]) -> WorkerAppendResult:
        payload = {"messages": [dict(m) if isinstance(m, Mapping) else m for m in messages], **options}
        result = self._apply("transcript.append", payload, _append, epoch=epoch, execution_id=execution_id,  # type: ignore[attr-defined]
                             session_id=session_id, generation=generation, sequence=sequence)
        return append_result(result)
