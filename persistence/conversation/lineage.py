"""Session lineage: compression chains, resume targets, the compression write side and the durable compressor counters.

Sessions form a tree through ``parent_session_id``. A compression continuation is a child of a session ended with ``end_reason == "compression"`` that is not a branch, delegate or reset fork and not ``tool``-sourced; it is recognized when read, never from a marker on the child. Every read answers for an unknown id the way the store does: the projection of an unrecorded session is the session itself. Times are epoch-second floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

from ..core.identity import JsonValue, MessageId, SessionId
from .messages import ConversationMessage, NewMessage, RowStamp
from .sessions import SessionRecord

__all__ = [
    "CompressionChild",
    "LineageAdoption",
    "LineageRepository",
]


@dataclass(frozen=True, slots=True)
class CompressionChild:
    """What the compressor decides about the continuation it publishes: its id, source, model, model config, system prompt, cwd (the parent's when empty) and profile (the parent's, else the store owner's, when ``None``). Routing keys, user, git metadata, display name and origin are copied from the parent."""

    session_id: SessionId
    source: str
    model: Optional[str] = None
    model_config: Optional[Mapping[str, JsonValue]] = None
    system_prompt: Optional[str] = None
    cwd: Optional[str] = None
    profile_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("CompressionChild.session_id must be non-empty")
        if not self.source:
            raise ValueError("CompressionChild.source must be non-empty")


@dataclass(frozen=True, slots=True)
class LineageAdoption:
    """Outcome of adopting a compression lineage from a donor store: whether the import ran (``ok``), whether every segment is now present (``adopted``), whether every donor segment was retired, the segments imported and skipped (root first), the imported children whose parent edge was dropped, the import errors, and the id the donor did not know."""

    ok: bool
    adopted: bool
    donor_retired: bool
    imported: tuple[SessionId, ...] = ()
    skipped: tuple[SessionId, ...] = ()
    detached: int = 0
    errors: tuple[str, ...] = ()
    missing: Optional[SessionId] = None


@runtime_checkable
class LineageRepository(Protocol):
    """Lineage walks and the compression write side over one scope's session tree."""

    def publish_compression_child(self, parent_session_id: SessionId, child: CompressionChild, handoff: Sequence[NewMessage], *, compression_lock_holder: Optional[str] = None, require_compression_lease: bool = True, require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0, watermark: Optional[MessageId] = None, watermark_ceiling: Optional[MessageId] = None) -> tuple[RowStamp, ...]:
        """End the parent by compression and publish ``child`` carrying ``handoff`` as its transcript, plus the parent's active rows in ``(watermark, watermark_ceiling]`` cloned after it, in one transaction; returns one ``RowStamp`` per handoff row. A lease not held by ``compression_lock_holder`` (refreshed first with ``require_lease_refresh``) raises ``CompressionBusy``; a missing parent, a parent ended deliberately (an automatic end stamp is cleared instead), an empty handoff or a parent closed concurrently raises ``CompressionPublishRefused``; a child id that already exists raises ``Conflict`` on the session."""
        ...

    def reopen_orphaned_compression(self, session_id: SessionId) -> bool:
        """Clear the compression end stamp of a parent that has no continuation child and no live compression lease (an expired lease row is deleted), in one transaction; False with nothing written otherwise."""
        ...

    def finalize_orphaned_compression_children(self) -> int:
        """End with ``orphaned_compression`` every open continuation older than seven days that holds rows and never made an API call; returns the rows ended."""
        ...

    def stamp_legacy_compression_chains(self) -> int:
        """Stamp the unset creation surface of compression-chain ancestors from their stamped continuations, in one idempotent transaction; returns the rows stamped."""
        ...

    def set_compression_fallback_streak(self, session_id: SessionId, value: int) -> None:
        """Store the deterministic-fallback streak, clamped at 0."""
        ...

    def set_compression_ineffective_count(self, session_id: SessionId, value: int) -> None:
        """Store the ineffective-compaction strike count, clamped at 0."""
        ...

    def set_compression_overload_streak(self, session_id: SessionId, value: int) -> None:
        """Store the sustained-overload abort streak, clamped at 0."""
        ...

    def increment_compression_overload_streak(self, session_id: SessionId) -> Optional[int]:
        """Add one to the overload streak in one statement and return the new value; ``None`` when the session has no row."""
        ...

    def set_compression_recovery_deadline(self, session_id: SessionId, deadline: Optional[float]) -> None:
        """Arm the anti-thrash recovery clock at ``deadline``; 0, ``None`` or an unparsable value disarms it."""
        ...

    def tip(self, session_id: SessionId) -> SessionId:
        """The live end of the session's compression chain, preferring at each step a continuation that compressed again, then an open one, then the most recently active; the input when nothing continued it."""
        ...

    def lineage(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """The compression lineage holding the session, root first through its oldest continuations; ``()`` for an unknown id, ``(session_id,)`` for a fork child or when the session is off the walked chain."""
        ...

    def chain(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """The compression chain walked forward from the session to its tip (``tip`` is the last element), without ancestors."""
        ...

    def conversation_root(self, session_id: SessionId) -> SessionId:
        """The root of the session's whole parent chain across every edge kind; the input when it has no recorded parent."""
        ...

    def tool_rotation_root(self, session_id: SessionId) -> SessionId:
        """The root of a ``tool``-sourced run's compression chain: climb from the session while it and its parent are both ``tool``-sourced and it would be the parent's compression continuation were it not ``tool``-sourced (no branch, delegate or reset marker naming the parent, parent ended by compression). The input for any other session and for an unknown id."""
        ...

    def is_fork_child(self, session_id: SessionId) -> bool:
        """Whether the session is a branch or delegate child of its parent (its config marker names the parent) or ``tool``-sourced; False for a missing row."""
        ...

    def is_compression_child(self, child: SessionRecord) -> bool:
        """Whether ``child``, a row the caller already read, continues its parent: it has a parent, is neither ``tool``-sourced nor a branch, delegate or reset child of it (a config marker naming the parent), and the parent row ended by compression. Reads only the parent row; a missing parent is False."""
        ...

    def resolve_resume_target(self, session_id: SessionId) -> SessionId:
        """The session a resume loads: the tip, then the deepest session below it through non-fork children that holds rows. Never raises: a failure returns the id resolved so far."""
        ...

    def resume_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """The segments a display resume of the session reads, root first: its verified compression lineage (each hop a compression continuation of a parent ended by compression), or the session alone for a fork, reset or tool child, a plain session and an unknown id."""
        ...

    def delegated_child_lineage_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """The segments a delegated child's transcript reads: ``resume_lineage_ids``, or the session alone while its row carries no ``_delegate_from`` marker (a queued child's row is reserved before its first turn writes the marker, and a delegator that rotated meanwhile would read as the child's own earlier segment)."""
        ...

    def delegate_descendant_ids(self, session_id: SessionId) -> tuple[SessionId, ...]:
        """The marked delegate subagents the session owns, in no defined order: walking up its parent chain from the session (stopping before a row with a ``_delegate_from`` marker and after one with ``_branched_from``), every delegate below a walked session (a ``_delegate_from`` marker naming it, or such a marker on a child of it, walked recursively), each with its compression chain, and the delegates below those segments in turn. ``()`` for a delegate session itself."""
        ...

    def current_segments(self, session_ids: Sequence[SessionId]) -> Mapping[SessionId, SessionId]:
        """``{session id: its current physical segment}`` in one read: the segment a local receipt's entry names, else the compression tip of a session ended by compression, else the session itself. Ids with no session row are absent."""
        ...

    def ancestor_display_prefix(self, session_id: SessionId) -> tuple[ConversationMessage, ...]:
        """The deduped display messages of the session's resume lineage that live in ancestors, without row ids; ``()`` when the lineage is the session alone."""
        ...

    def compression_fallback_streak(self, session_id: SessionId) -> int:
        """The stored deterministic-fallback streak; 0 for a missing row or unreadable value."""
        ...

    def compression_ineffective_count(self, session_id: SessionId) -> int:
        """The stored ineffective-compaction strike count; 0 for a missing row or unreadable value."""
        ...

    def compression_overload_streak(self, session_id: SessionId) -> int:
        """The stored sustained-overload abort streak; 0 for a missing row or unreadable value."""
        ...

    def compression_recovery_deadline(self, session_id: SessionId) -> float:
        """The stored recovery deadline; 0.0 when not armed, for a missing row and for an unreadable or negative value."""
        ...
