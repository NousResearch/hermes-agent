"""Private history-safe continuation for one already-started main turn."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping
import uuid

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.execution_router import ExecutionKind


@dataclass(frozen=True)
class _FrozenMap:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _FrozenList:
    items: tuple[Any, ...]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMap(tuple((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, _FrozenMap):
        return {key: _thaw(item) for key, item in value.items}
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value.items]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


@dataclass(frozen=True)
class _MainTurnContinuationRecord:
    owner_profile: str
    session_id: str
    execution_kind: ExecutionKind
    old_request_id: str
    old_attempt_id: str
    turn_id: str
    current_turn_user_idx: int
    transcript: _FrozenList
    persistence_watermark: int
    persistence_binding: _FrozenMap
    old_route_signature: tuple[Any, ...]
    router_generation: int
    router_contract_generation: str
    terminal_reason: str
    fallback_receipt: Any
    consumed_fallback_slot: int
    next_fallback_cursor: int


def _validate_boundary(messages: list[Any], turn_id: str, current_turn_user_idx: Any) -> int:
    if not turn_id:
        raise ValueError("continuation requires the old turn_id")
    if type(current_turn_user_idx) is not int or not 0 <= current_turn_user_idx < len(messages):
        raise ValueError("continuation has an invalid current-turn boundary")
    row = messages[current_turn_user_idx]
    if not isinstance(row, dict) or row.get("role") != "user":
        raise ValueError("continuation boundary does not address a user row")
    return current_turn_user_idx


def _seal_main_turn_continuation(agent: Any, prepared: Any, result: Any) -> _MainTurnContinuationRecord:
    """Persist and recursively freeze the exact transcript of a terminal routed attempt."""
    if not isinstance(result, dict) or not isinstance(result.get("messages"), list):
        raise ValueError("continuation requires an exact transcript")
    receipt = getattr(agent, "_routed_restart_required", None)
    consumed = getattr(receipt, "consumed_fallback_slot", None)
    if type(consumed) is not int or consumed < 0:
        raise ValueError("continuation requires a valid fallback receipt")
    messages = result["messages"]
    turn_id = str(result.get("turn_id") or "")
    boundary = _validate_boundary(messages, turn_id, result.get("current_turn_user_idx"))

    lock = getattr(agent, "_session_persist_lock", None)
    with lock if lock is not None else nullcontext():
        persist = getattr(agent, "_persist_session", None)
        if callable(persist):
            persist(messages)
        for row in messages:
            if isinstance(row, dict) and not row.get(_DB_PERSISTED_MARKER):
                raise RuntimeError("continuation transcript is not durably sealed")
        watermark = int(getattr(agent, "_last_flushed_db_idx", len(messages)))
        if watermark < len(messages):
            raise RuntimeError("continuation persistence watermark contradicts transcript")
        snapshot = _freeze(messages)

    request = prepared.request
    session_id = str(getattr(agent, "session_id", "") or "")
    if not session_id:
        raise ValueError("continuation requires an owning session")
    try:
        from hermes_cli.profiles import get_active_profile_name

        owner_profile = get_active_profile_name()
    except Exception:
        owner_profile = "default"
    persistence_binding = _freeze({
        name: getattr(agent, name)
        for name in (
            "_last_flushed_db_idx",
            "_persist_user_message_idx",
            "_persist_user_message_id",
            "_persist_user_turn_id",
            "_session_persist_user_committed",
        )
        if hasattr(agent, name)
    })
    return _MainTurnContinuationRecord(
        owner_profile=owner_profile,
        session_id=session_id,
        execution_kind=ExecutionKind.MAIN_TURN,
        old_request_id=str(request.request_id),
        old_attempt_id=str(request.attempt_id),
        turn_id=turn_id,
        current_turn_user_idx=boundary,
        transcript=snapshot,
        persistence_watermark=watermark,
        persistence_binding=persistence_binding,
        old_route_signature=tuple(prepared.route_signature or ()),
        router_generation=(
            int(prepared.route_signature[-2])
            if prepared.route_signature and prepared.route_signature[-2] is not None
            else 0
        ),
        router_contract_generation=(
            str(prepared.route_signature[-1])
            if prepared.route_signature and prepared.route_signature[-1] is not None
            else ""
        ),
        terminal_reason=str(getattr(receipt, "reason_code", "fallback"))[:128],
        fallback_receipt=receipt,
        consumed_fallback_slot=consumed,
        next_fallback_cursor=consumed + 1,
    )


def _continue_main_turn_attempt(
    agent: Any,
    continuation_record: _MainTurnContinuationRecord,
    *,
    existing_surface_callbacks: Mapping[str, Any],
) -> Any:
    """Continue after the sealed tail without running normal new-user admission."""
    if str(getattr(agent, "session_id", "") or "") != continuation_record.session_id:
        raise ValueError("continuation agent does not own the sealed session")
    try:
        from hermes_cli.profiles import get_active_profile_name

        active_profile = get_active_profile_name()
    except Exception:
        active_profile = "default"
    if active_profile != continuation_record.owner_profile:
        raise ValueError("continuation agent does not own the sealed profile")
    for name, callback in existing_surface_callbacks.items():
        setattr(agent, name, callback)
    agent._fallback_index = continuation_record.next_fallback_cursor
    agent._current_turn_id = continuation_record.turn_id
    agent._persist_user_message_idx = continuation_record.current_turn_user_idx
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._persist_user_message_platform_id = None
    agent._last_flushed_db_idx = continuation_record.persistence_watermark
    for name, value in _thaw(continuation_record.persistence_binding).items():
        setattr(agent, name, value)
    transcript = _thaw(continuation_record.transcript)
    from agent.conversation_loop import _continue_main_turn_from_transcript

    return _continue_main_turn_from_transcript(
        agent,
        transcript,
        turn_id=continuation_record.turn_id,
        current_turn_user_idx=continuation_record.current_turn_user_idx,
    )


def _bind_routed_main_turn_callbacks(
    owner: Any, callbacks: Mapping[str, Any]
) -> dict[str, Any]:
    """Return attempt-scoped callback wrappers that ignore late old-attempt emissions."""
    token = uuid.uuid4().hex
    setattr(owner, "_main_turn_callback_token", token)
    bound: dict[str, Any] = {}
    for name, callback in callbacks.items():
        if not callable(callback):
            bound[name] = callback
            continue

        def guarded(*args, __callback=callback, __token=token, **kwargs):
            if getattr(owner, "_main_turn_callback_token", None) != __token:
                return None
            return __callback(*args, **kwargs)

        bound[name] = guarded
    return bound


def _invalidate_routed_main_turn_callbacks(owner: Any) -> None:
    setattr(owner, "_main_turn_callback_token", None)
