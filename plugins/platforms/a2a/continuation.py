"""Explicit task references and atomic admission of a new execution attempt."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
import hashlib
from functools import partial
from typing import Any, cast

from . import protocol, security


def validate_identifiers(params: dict[str, object]) -> dict[str, int | str] | None:
    message = params.get("message")
    if not isinstance(message, dict):
        return {"code": protocol.ERR_INVALID_PARAMS, "message": "message must be an object"}
    for key in ("taskId", "contextId", "messageId"):
        if key in message and (not isinstance(message[key], str) or not message[key]):
            return {"code": protocol.ERR_INVALID_PARAMS, "message": f"{key} must be a nonempty string"}
    if "contextId" in params and (not isinstance(params["contextId"], str) or not params["contextId"]):
        return {"code": protocol.ERR_INVALID_PARAMS, "message": "contextId must be a nonempty string"}
    return None


def prepare_continuation(adapter, params: dict[str, object], peer: str,
                         agent: dict[str, object], generation: int):
    message = cast(dict[str, object], params["message"])
    task_id = cast(str, message["taskId"])
    scope = adapter._scope_for_agent(agent)
    adapter._expire_waits()
    rec = adapter.tasks.get(task_id, *scope)
    if rec is None:
        return _error(protocol.ERR_TASK_NOT_FOUND, "task not found")
    context_id = cast(str | None, message.get("contextId", params.get("contextId")))
    if context_id is not None and context_id != rec["context_id"]:
        return _error(protocol.ERR_INVALID_PARAMS, "contextId does not match task")
    text = protocol.extract_text(params)
    if not text:
        return _error(protocol.ERR_INVALID_PARAMS, "continuation requires input")
    rec, future, error, retry = _admit(adapter, task_id, context_id, scope, generation,
                                       cast(str, message.get("messageId", "")))
    if error:
        return _error(error, "task cannot accept continuation")
    rec = cast(dict[str, Any], rec)
    if retry:
        return None, {"task_id": task_id, "context_id": rec["context_id"], "future": future,
                      "created_iso": rec["created_iso"]}
    input_id = "a2a-msg:" + hashlib.sha256(
        f'{rec["context_id"]}\0{message.get("messageId", rec["attempt_id"])}'.encode()).hexdigest()
    security.audit("inbound", peer, task_id, text)
    protocol.persist_message(rec["context_id"], "user", text, task_id, input_id=input_id)
    protocol.metrics.inbound_total += 1
    adapter._register_inline_push(task_id, params, agent=agent)
    return adapter._dispatch_prepared(rec, params, peer, agent, security.wrap_inbound(peer, text),
                                      generation, input_id, future)


def _admit(adapter, task_id: str, context_id: str | None,
           scope: tuple[str, str], generation: int, message_id: str
           ) -> tuple[dict[str, Any] | None, Future[tuple[str, str]] | None, int | None, bool]:
    with adapter._pending_lock:
        if adapter._admission_closed or adapter._generation != generation:
            return None, None, protocol.ERR_UNSUPPORTED_OPERATION, False
        rec, future, error, retry = adapter.tasks.resume(
            task_id, context_id, *scope, partial(_reserve_attempt, adapter), message_id)
        if error or retry:
            return rec, future, error, retry
        rec = cast(dict[str, Any], rec)
        future = cast(Future[tuple[str, str]], future)
        attempt_id = rec["attempt_id"]
        previous = adapter._pending_attempts.get(task_id)
        adapter._attempt_tasks.pop(previous, None)
        adapter._pending[task_id] = (rec["context_id"], future)
        adapter._pending_attempts[task_id] = attempt_id
        adapter._attempt_tasks[attempt_id] = task_id
        adapter._owned_tasks.add(task_id)
        order = adapter._pending_order.setdefault(rec["context_id"], deque())
        if task_id not in order:
            order.append(task_id)
        return rec, future, None, False


def _reserve_attempt(adapter, attempt_id: str, context_id: str) -> int | None:
    if not adapter._completion_io.reserve(attempt_id):
        return protocol.ERR_RATE_LIMITED
    if not adapter._turns.admit(context_id, protocol.max_pingpong_turns()):
        adapter._completion_io.release_reservation(attempt_id)
        protocol.metrics.anti_loop_triggers += 1
        return protocol.ERR_UNSUPPORTED_OPERATION
    return None


def _error(code: int, message: str) -> tuple[dict[str, dict[str, int | str]], None]:
    return {"error": {"code": code, "message": message}}, None
