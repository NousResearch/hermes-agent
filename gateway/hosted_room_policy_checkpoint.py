"""Durable bounded policy projection for hosted Group Chat preparation.

The append-only room log remains the user-visible source of truth. This module materializes only the state
needed to choose and reconstruct the next active discussion, so a busy room does not replay its complete
history every poll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from persistence.gateway.hosted_rooms import HostedPlacement, PolicySnapshot

if TYPE_CHECKING:
    from persistence_assembly import Persistence


MAX_ACTIVE_POLICY_EVENTS = 64


class HostedRoomPolicyCheckpoint:
    """Incrementally index room policy without compacting visible history."""
    def __init__(
        self, store: Persistence, *, placement: HostedPlacement, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS
    ) -> None:
        if isinstance(max_active_events, bool) or not isinstance(max_active_events, int) or max_active_events < 1:
            raise ValueError("max_active_events must be a positive integer")
        self.store, self.placement, self.max_active_events = store, placement, max_active_events

    def sync(self, *, room_id: str, latest_seq: int) -> int:
        """Materialize each unseen event exactly once by durable cursor; returns the cursor the snapshot read."""
        return self.snapshot(room_id=room_id, latest_seq=latest_seq).through_seq

    def snapshot(self, *, room_id: str, latest_seq: int) -> PolicySnapshot:
        """Return only the oldest active discussion and its watermark set."""
        return self.store.hosted_room_policy.snapshot(
            placement=self.placement, room_id=room_id, latest_seq=latest_seq, max_active_events=self.max_active_events)

    def publication_exists(self, *, room_id: str, task_id: str, status: str, execution_generation: int) -> bool:
        """Return whether one exact driver outcome is already in the room log."""
        return self.store.hosted_room_policy.publication_exists(
            placement=self.placement, room_id=room_id, task_id=task_id, status=status,
            execution_generation=execution_generation, max_active_events=self.max_active_events)

    def events_for_task(self, *, room_id: str, source_event_seq: int,
                        input_context: Mapping[str, Any] | None = None,
                        task_id: str | None = None) -> list[dict[str, Any]]:
        """Load one bounded discussion projection for terminal reconstruction."""
        return list(self.store.hosted_room_policy.events_for_task(
            placement=self.placement, room_id=room_id, source_event_seq=source_event_seq,
            input_context=input_context, task_id=task_id, max_active_events=self.max_active_events))

    def compact_completed(self, *, room_id: str) -> None:
        """Drop any completed projections left by an interrupted sync."""
        self.store.hosted_room_policy.compact_completed(
            placement=self.placement, room_id=room_id, max_active_events=self.max_active_events)
