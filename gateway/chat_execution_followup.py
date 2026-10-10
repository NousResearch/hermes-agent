"""Fresh physical execution identities within an unchanged logical queued chat chain."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueuedWorkerReservation:
    home: str
    session_key: str
    generation: int
    run_generation: int
    session_id: str
    predecessor_generation: int


def worker_execution_identity(reservation, session_key, run_generation, session_id):
    if isinstance(reservation, QueuedWorkerReservation):
        home, key, generation = reservation.home, reservation.session_key, reservation.generation
        if (reservation.run_generation != run_generation or reservation.session_id != session_id
                or not generation > reservation.predecessor_generation >= run_generation):
            raise RuntimeError("Durable queued worker identity mismatch")
    else:
        home, key, generation = reservation
        if generation != run_generation:
            raise RuntimeError("Durable worker admission is stale")
    if key != session_key:
        raise RuntimeError("Durable worker admission is stale")
    return home, key, generation


def reserve_followup_execution(runner, turn_ctx, source, session_key):
    previous = turn_ctx.chat_execution_reservation
    if previous is None:
        return None  # Legacy/unadmitted entry points retain their existing behavior.
    home, key, generation = worker_execution_identity(
        previous, turn_ctx.session_key, turn_ctx.run_generation, turn_ctx.session_id,
    )
    if (key != session_key or Path(runner._resolve_profile_home_for_source(source)).resolve() != Path(home)
            or not runner._is_session_run_current(key, turn_ctx.run_generation)):
        raise RuntimeError("Durable queued worker admission is stale")
    from gateway.chat_execution_recovery import ChatExecutionStore
    next_generation = ChatExecutionStore(home).reserve(
        key, turn_ctx.run_generation, generation, predecessor=(generation, turn_ctx.session_id),
    )
    if next_generation is None:
        raise RuntimeError("Durable queued worker reservation refused")
    # Routing/leases/final delivery belong to the logical chain; the journal counts physical targets.
    return QueuedWorkerReservation(
        home, key, next_generation, turn_ctx.run_generation, turn_ctx.session_id, generation,
    )
