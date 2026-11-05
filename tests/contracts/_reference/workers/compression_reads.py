"""Reference worker compression reads (``persistence.workers.compression_reads.WorkerCompressionReads``).

Mixed into ``InMemoryWorkerExecutionRepository``. Each read runs its handler inside the receipt transaction of :mod:`.receipt` and answers through the sibling references that own the state (sessions, lineage, messages), so a result equals what those verbs answer for the same target. The context is the housekeeping reference's stored-row dict of the target (``_export_row``).
"""

from __future__ import annotations

from typing import Any, Mapping

from persistence.conversation.messages import ConversationRequest, TranscriptView
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, SessionId
from persistence.workers.compression_reads import WorkerLineageView
from persistence.workers.executions import WorkerExecutionId

from .receipt import Handler, Step, denied, refuse, require_fields, text

__all__ = ["InMemoryWorkerCompressionReads", "authorize", "readable_ids"]

_HISTORY_FLAGS = ("include_ancestors", "include_inactive", "repair_alternation", "include_row_ids", "include_compacted")


def readable_ids(store: Any, session_id: str) -> list[str]:
    """The session's ancestors root first, cut at the nearest explicit branch, then its compression chain."""
    return [*reversed(store.usage._lineage_chain(store.scope, session_id)), *store.lineage.chain(SessionId(session_id))]


def authorize(step: Step, target: Any) -> None:
    """Authorize a compression read: the target within the text bound and readable by the worker's session."""
    text(target)
    if target not in readable_ids(step.store, step.session_id):
        raise denied("session", target)


def _context(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    from ..housekeeping.maintenance import _export_row

    require_fields(payload, ("target",))
    authorize(step, payload["target"])
    record = step.store.sessions.get(SessionId(payload["target"]))
    return {"session": None if record is None else _export_row(record).as_dict()}


def _lineage(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ("target",))
    target = payload["target"]
    authorize(step, target)
    lineage, sid = step.store.lineage, SessionId(target)
    record = step.store.sessions.get(sid)
    identity = [False, ""] if record is None else [lineage.is_fork_child(sid), str(record.source or "").strip()]
    return {"readable_ids": readable_ids(step.store, step.session_id), "lineage": list(lineage.lineage(sid)), "root": lineage.conversation_root(sid),
            "tip": lineage.tip(sid), "resume": lineage.resolve_resume_target(sid), "identity": identity}


def _history(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    require_fields(payload, ("target", *_HISTORY_FLAGS))
    if any(type(payload[flag]) is not bool for flag in _HISTORY_FLAGS):
        refuse("invalid_params")
    authorize(step, payload["target"])
    view = TranscriptView.ALL if payload["include_inactive"] else TranscriptView.DISPLAY if payload["include_compacted"] else TranscriptView.ACTIVE
    request = ConversationRequest(view=view, include_lineage=payload["include_ancestors"], repair_alternation=payload["repair_alternation"],
                                  include_row_ids=payload["include_row_ids"])
    return {"messages": [m.as_openai_message() for m in step.store.messages.read_conversation(SessionId(payload["target"]), request)]}


class InMemoryWorkerCompressionReads:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_apply``)."""

    def _read(self, operation: str, payload: dict[str, Any], handler: Handler, epoch: Any, execution_id: Any, session_id: Any, generation: Any, sequence: Any) -> dict[str, Any]:
        return self._apply(operation, payload, handler, epoch=epoch, execution_id=execution_id, session_id=session_id,  # type: ignore[attr-defined, no-any-return]
                           generation=generation, sequence=sequence)

    def compression_session_context(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> Mapping[str, JsonValue]:
        return self._read("compression.context", {"target": target_session_id}, _context, epoch, execution_id, session_id, generation, sequence)["session"]  # type: ignore[no-any-return]

    def compression_lineage_view(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> WorkerLineageView:
        view = self._read("compression.lineage", {"target": target_session_id}, _lineage, epoch, execution_id, session_id, generation, sequence)
        return WorkerLineageView(readable_ids=tuple(SessionId(s) for s in view["readable_ids"]), lineage=tuple(SessionId(s) for s in view["lineage"]),
                                 root=SessionId(view["root"]), tip=SessionId(view["tip"]), resume=SessionId(view["resume"]),
                                 identity=(bool(view["identity"][0]), str(view["identity"][1])))

    def compression_history(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, include_ancestors: bool, include_inactive: bool, repair_alternation: bool, include_row_ids: bool, include_compacted: bool) -> tuple[Mapping[str, JsonValue], ...]:
        payload = {"target": target_session_id, "include_ancestors": include_ancestors, "include_inactive": include_inactive,
                   "repair_alternation": repair_alternation, "include_row_ids": include_row_ids, "include_compacted": include_compacted}
        return tuple(self._read("compression.history", payload, _history, epoch, execution_id, session_id, generation, sequence)["messages"])
