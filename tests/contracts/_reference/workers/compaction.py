"""Reference verbs of the worker's row-carrying compression writes (``persistence.workers.compaction.WorkerCompactionVerbs``).

Mixed into ``InMemoryWorkerExecutionRepository``. Each verb runs its handler inside the receipt transaction of :mod:`.receipt`, over the owner's own bodies on the staged tables: ``MessageRepository.compact`` for the archive, ``LineageRepository.publish_compression_child`` with the binding reference's ``advance_local_target`` for the publication, ``append_batch`` for the append.

Reference limits: an append's compression lock holder is accepted and checked but fences nothing: the append guard does not read the lock.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import DB_PERSISTED_MARKER_KEY, HeldCoverage, HeldMessage, NewMessage, Watermark
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import MessageId, SessionId
from persistence.workers.compaction import WorkerArchiveResult, WorkerCompaction, WorkerCompressionPublication, WorkerPublishResult
from persistence.workers.executions import WorkerExecutionId
from persistence.workers.sessions import RowAnnotation, WorkerAppendResult, WorkerMessage
from persistence.workers.vocabulary import HANDOFF_MESSAGE_FIELDS, SIDECAR_KEYS
from session_rules.automation import GoalState, HeartbeatState, LoopState

from .._authority import session_runtime
from ..authority.bindings_lineage import advance_local_target
from .receipt import Step, denied, number, refuse, require_fields, stale_generation, text
from .sessions import append_guarded, append_result, check_rows, new_message

__all__ = ["InMemoryWorkerCompactionVerbs"]

_ARCHIVE_FIELDS = ("messages", "model_config_patch", "watermark", "lock_holder", "tail_count")
_ARCHIVE_OPTIONAL = ("carried_messages", "covered_ids", "unresolved_held", "held_for_cas")
_PUBLISH_FIELDS = ("child_session_id", "source", "messages", "model", "model_config", "system_prompt", "cwd", "profile_name",
                   "compression_lock_holder", "require_compression_lease", "require_lease_refresh", "lease_ttl_seconds", "watermark", "watermark_ceiling")
_MARKERS = ("_branched_from", "_delegate_from", "_reset_from")
_AUTOMATION = (("goal", GoalState), ("heartbeat", HeartbeatState), ("loop", LoopState))


def _dict_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(m, dict) for m in value)


def _watermarks(payload: Mapping[str, Any]) -> None:
    for name in ("watermark", "watermark_ceiling"):
        value = payload.get(name)
        if value is not None and (type(value) is not int or value < 0):
            refuse("invalid_params")


def _held(message: Mapping[str, Any]) -> HeldMessage:
    """A held message dict as the row compare and the coverage proof read it (role, content, tool-call identity, the stored epoch, ``_row_id``, the persist marker, the ownership version and uids, and the alternation repair's markers)."""
    row_id = message.get("_row_id")
    return HeldMessage(
        role=str(message.get("role") or "unknown"), content=message.get("content"), tool_call_id=message.get("tool_call_id"),
        tool_calls=tuple(message.get("tool_calls") or ()), timestamp=message.get("timestamp"),
        row_id=MessageId(row_id) if type(row_id) is int and row_id > 0 else None, persisted=message.get(DB_PERSISTED_MARKER_KEY) is True,
        db_row_snapshot=message.get("_db_row_snapshot"), message_uid=message.get("message_uid"),
        absorbed_row_ids=tuple(MessageId(i) for i in message.get("_absorbed_row_ids") or ()),
        merged_durable_rows=message.get("_merged_durable_rows") or 0,
        retired_durable_rows=tuple(_held(row) for row in message.get("_retired_durable_rows") or ()),
        tool_call_ids=tuple(message.get("tool_call_ids") or ()), own_row=bool(message.get("_own_row")),
        retired_row=bool(message.get("_retired_row")),
    )


def _replacement(message: Mapping[str, Any]) -> NewMessage:
    """A compaction row as ``compact`` takes it: the written row with the alternation repair's markers."""
    markers = NewMessage.from_live(message)
    return replace(new_message(message), merged_durable_rows=markers.merged_durable_rows,
                   unnamed_durable_rows=markers.unnamed_durable_rows, absorbed_row_ids=markers.absorbed_row_ids)


def _check_archive(step: Step, payload: dict[str, Any]) -> None:
    """The archive payload checks, in order."""
    require_fields(payload, _ARCHIVE_FIELDS, _ARCHIVE_OPTIONAL)
    check_rows(step, payload["messages"], HANDOFF_MESSAGE_FIELDS)
    if not _dict_list(payload.get("carried_messages") or []):
        refuse("invalid_params")
    covered = payload.get("covered_ids")
    if covered is not None and (not isinstance(covered, list) or any(type(i) is not int or i <= 0 for i in covered)):
        refuse("invalid_params")
    for name in ("unresolved_held", "held_for_cas"):
        if payload.get(name) is not None and not _dict_list(payload[name]):
            refuse("invalid_params")
    _watermarks(payload)
    if payload["lock_holder"] is not None:
        text(payload["lock_holder"])
    patch = payload["model_config_patch"]
    if patch is not None and (not isinstance(patch, dict) or set(patch) - SIDECAR_KEYS):
        refuse("invalid_params")
    if type(payload["tail_count"]) is not int or payload["tail_count"] < 0:
        refuse("invalid_params")


def _archive(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    _check_archive(step, payload)
    sid, holder, watermark = SessionId(step.session_id), payload["lock_holder"], payload["watermark"]
    covered, unresolved, cas = payload.get("covered_ids"), payload.get("unresolved_held"), payload.get("held_for_cas")
    coverage = None if covered is None and unresolved is None else HeldCoverage(
        covered_ids=tuple(MessageId(i) for i in covered or ()), unresolved=tuple(_held(m) for m in unresolved or ()))
    compacted = step.store.messages.compact(
        sid, [_replacement(m) for m in payload["messages"]], watermark=None if watermark is None else Watermark(MessageId(watermark) if watermark else None),
        carried_tail_count=payload["tail_count"], lock_holder=holder,
        model_config_patch=payload["model_config_patch"], carried=tuple(_held(m) for m in payload.get("carried_messages") or ()),
        coverage=coverage, held_for_cas=None if cas is None else tuple(_held(m) for m in cas))
    annotations = [{"_row_id": int(stamp.id), "timestamp": stamp.timestamp, "_db_row_snapshot": stamp.db_row_snapshot, "_canonical_row": None,
                    "message_uid": stamp.message_uid, "_tool_call_uids": None if stamp.tool_call_uids is None else dict(stamp.tool_call_uids),
                    "_tool_call_uid": stamp.tool_call_uid} for stamp in compacted.rows]
    return {"value": compacted.active_count, "annotations": annotations}


def _check_publication(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """The publish payload checks, in order; returns the child's model config with the parent's lineage markers."""
    require_fields(payload, _PUBLISH_FIELDS)
    text(payload["child_session_id"])
    text(payload["compression_lock_holder"])
    _watermarks(payload)
    if payload["require_compression_lease"] is not True or type(payload["require_lease_refresh"]) is not bool:
        refuse("invalid_params")
    if not 0.1 <= number(payload["lease_ttl_seconds"]) <= 3600:
        refuse("invalid_params")
    check_rows(step, payload["messages"], HANDOFF_MESSAGE_FIELDS)
    parent = step.sessions[step.session_id]
    for key in ("source", "cwd", "profile_name"):
        if payload[key] is not None and payload[key] != parent.get(key):
            raise denied("session", step.session_id)
    for key in ("model", "system_prompt"):
        if payload[key] is not None and not isinstance(payload[key], str):
            refuse("invalid_params")
    config = payload["model_config"]
    if config is not None and not isinstance(config, dict):
        refuse("invalid_params")
    config, prior = dict(config or {}), parent.get("model_config") or {}
    for key in _MARKERS:
        if key in config and config[key] != prior.get(key):
            raise denied("session", step.session_id)
        if key in prior:
            config[key] = prior[key]
    if payload["child_session_id"] in step.sessions:
        raise denied("session", payload["child_session_id"])
    return config


def _move_automation(meta: dict[str, str], parent: str, child: str) -> None:
    """Live state moves to the child unless the child has its own (a cleared heartbeat there may be replaced); the parent's copy is marked cleared. Both are written through the state class, so each holds every field of its ``to_json``."""
    for family, state_type in _AUTOMATION:
        old_key, new_key = f"{family}:{parent}", f"{family}:{child}"
        if old_key not in meta:
            continue
        state = state_type.from_json(meta[old_key])
        if state.status == "cleared":
            continue
        if new_key in meta and (family != "heartbeat" or state_type.from_json(meta[new_key]).status != "cleared"):
            continue
        meta[new_key] = state.to_json()
        state.status = "cleared"
        meta[old_key] = state.to_json()


def _publish(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    config = _check_publication(step, payload)
    sid, child = step.session_id, payload["child_session_id"]
    worker = next((row for row in step.authority["workers"].values() if row["session_id"] == sid and row["status"] in ("registered", "running")), None)
    if worker is None:
        raise stale_generation("session", sid)
    watermark, ceiling = payload["watermark"], payload["watermark_ceiling"]
    stamps = step.store.lineage.publish_compression_child(
        SessionId(sid), CompressionChild(session_id=SessionId(child), source=payload["source"], model=payload["model"], model_config=config,
                                         system_prompt=payload["system_prompt"], cwd=payload["cwd"], profile_name=payload["profile_name"]),
        [new_message(m) for m in payload["messages"]], compression_lock_holder=payload["compression_lock_holder"],
        require_compression_lease=payload["require_compression_lease"], require_lease_refresh=payload["require_lease_refresh"],
        lease_ttl_seconds=payload["lease_ttl_seconds"], watermark=None if watermark is None else MessageId(watermark),
        watermark_ceiling=None if ceiling is None else MessageId(ceiling))
    advance_local_target(step.store, step.scope, step.authority, step.tables, sid, child)
    _move_automation(step.tables.setdefault("state_meta", {}), sid, child)
    session_runtime(step.authority, child)["generation"] = worker["generation"]
    worker["session_id"] = child
    for admission in step.authority["admissions"].values():
        lineage = admission.get("lineage") or []
        if (lineage and lineage[-1] == sid and admission["generation"] == worker["generation"]
                and admission["owner_epoch"] == worker["owner_epoch"] and admission["status"] in ("started", "unknown")):
            admission["lineage"] = [*lineage, child]
    annotations = [{"_row_id": int(stamp.id), "timestamp": stamp.timestamp, "_db_row_snapshot": stamp.db_row_snapshot, "_canonical_row": None,
                    "message_uid": stamp.message_uid, "_tool_call_uids": None if stamp.tool_call_uids is None else dict(stamp.tool_call_uids),
                    "_tool_call_uid": stamp.tool_call_uid} for stamp in stamps]
    return {"value": None, "worker_assignment": {"parent": sid, "session_id": child}, "annotations": annotations}


def _append(step: Step, payload: dict[str, Any]) -> dict[str, Any]:
    """The compression append: the payload checks, then the append guarded by the turn lease."""
    require_fields(payload, ("messages", "compression_lock_holder", "turn_lease_holder", "turn_lease_ttl_seconds"))
    check_rows(step, payload["messages"], HANDOFF_MESSAGE_FIELDS)
    for key in ("compression_lock_holder", "turn_lease_holder"):
        if payload[key] is not None:
            text(payload[key])
    ttl = number(payload["turn_lease_ttl_seconds"])
    if not 0.1 <= ttl <= 3600:
        refuse("invalid_params")
    return append_guarded(step, payload["messages"], payload["turn_lease_holder"], ttl)


def _annotations(result: Mapping[str, Any]) -> tuple[RowAnnotation, ...]:
    return tuple(RowAnnotation.from_wire(a) for a in result["annotations"])


class InMemoryWorkerCompactionVerbs:
    """Host: ``InMemoryWorkerExecutionRepository`` (``_apply``)."""

    def compact_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, compaction: WorkerCompaction) -> WorkerArchiveResult:
        result = self._apply("compression.archive", compaction.payload(), _archive, epoch=epoch, execution_id=execution_id,  # type: ignore[attr-defined]
                             session_id=session_id, generation=generation, sequence=sequence)
        return WorkerArchiveResult(value=int(result["value"]), annotations=_annotations(result))

    def publish_compression_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, publication: WorkerCompressionPublication) -> WorkerPublishResult:
        result = self._apply("compression.publish", publication.payload(), _publish, epoch=epoch, execution_id=execution_id,  # type: ignore[attr-defined]
                             session_id=session_id, generation=generation, sequence=sequence)
        moved = result["worker_assignment"]
        return WorkerPublishResult(parent=SessionId(moved["parent"]), session_id=SessionId(moved["session_id"]), annotations=_annotations(result))

    def append_compression_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], compression_lock_holder: str | None, turn_lease_holder: str | None, turn_lease_ttl_seconds: float) -> WorkerAppendResult:
        payload = {"messages": [dict(m) if isinstance(m, Mapping) else m for m in messages], "compression_lock_holder": compression_lock_holder,
                   "turn_lease_holder": turn_lease_holder, "turn_lease_ttl_seconds": turn_lease_ttl_seconds}
        result = self._apply("compression.append", payload, _append, epoch=epoch, execution_id=execution_id,  # type: ignore[attr-defined]
                             session_id=session_id, generation=generation, sequence=sequence)
        return append_result(result)
