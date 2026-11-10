"""The worker's receipted reads: ``compression.context``, ``compression.lineage`` and ``compression.history``.

A managed worker reads its session row, lineage and transcript through the owner (``agent/runtime_session_compression.py``); each verb is one operation of the receipt transaction documented in :mod:`persistence.workers.executions`. The payload names a ``target`` the worker must be able to read: a session on the worker session's lineage root to tip, or on its compression chain (``permission_denied`` otherwise, ``invalid_params`` for a target outside the text bound). A read changes no row, so it leaves the runtime revision alone; its receipt replays the answer recorded at the time until the execution ends, when the context and history receipts are compacted and a late replay is ``stale_generation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol

from ..core.fences import AuthorityEpoch, ExecutionGeneration
from ..core.identity import JsonValue, SessionId

if TYPE_CHECKING:
    from .executions import WorkerExecutionId

__all__ = ["WorkerCompressionReads", "WorkerLineageView"]


@dataclass(frozen=True, slots=True)
class WorkerLineageView:
    """The ``compression.lineage`` answer: ``readable_ids`` the sessions the worker may read (computed for its own session, repeats possible), and for the target its compression ``lineage``, conversation ``root``, compression ``tip``, ``resume`` target, ``identity`` ``(is_fork_child, source)`` and ``tool_root`` (``LineageRepository.tool_rotation_root``)."""

    readable_ids: tuple[SessionId, ...]
    lineage: tuple[SessionId, ...]
    root: SessionId
    tip: SessionId
    resume: SessionId
    identity: tuple[bool, str]
    tool_root: SessionId


class WorkerCompressionReads(Protocol):
    """The worker's receipted reads of a readable target."""

    def compression_session_context(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> Mapping[str, JsonValue]:
        """``compression.context`` with ``{"target"}``: the target's full ``sessions`` row as a row dict (``id``, every column, ``model_config`` and ``origin_json`` as stored text, ``system_prompt`` and ``tool_names`` resolved, ``_reset_from`` / ``_branched_from`` lifted from the markers)."""
        ...

    def compression_lineage_view(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> WorkerLineageView:
        """``compression.lineage`` with ``{"target"}``: the target's :class:`WorkerLineageView`."""
        ...

    def compression_history(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, include_ancestors: bool, include_inactive: bool, repair_alternation: bool, include_row_ids: bool, include_compacted: bool) -> tuple[Mapping[str, JsonValue], ...]:
        """``compression.history`` with the target and the five flags as sent (each a bool, else ``invalid_params``): the target's conversation dicts as ``get_messages_as_conversation`` loads them (the resume lineage with ``include_ancestors``, every row with ``include_inactive``, the deduped display generations with ``include_compacted``)."""
        ...
