"""Slice 9: launch-protocol wiring for the live dispatcher spawn path.

Bridges the admission-kernel launch ledger (kanban_launch_protocol) around
the live ``dispatch_once`` spawn step: the wrapper returned by
:func:`wrap_spawn_with_launch_protocol` is handed to ``dispatch_once`` as
``spawn_fn``. Every launch flows

    acquire owner lease
    -> claim_not_spawned
    -> record_spawn_intent
    -> spawn
    -> record_spawned

with :func:`release_claim` on every abort path, and best-effort termination
of an already-spawned worker when post-spawn ledger recording fails, so a
real process can never outlive an unrecorded launch.

Ordering rationale (Atomic round-1 finding 2): the owner lease is acquired
BEFORE the claim is persisted. If lease acquisition fails, nothing has been
written; if any later step fails, the lease is in hand and ``release_claim``
can retire the claim.

Integration (Atomic round-1 finding 1): :func:`configure_launch_wiring`
installs a module-global wiring; :func:`apply_launch_wiring` is called by
the dispatcher daemon path so the live ``spawn_fn`` is decorated when (and
only when) wiring is explicitly configured. Default is fully pass-through.
"""

from __future__ import annotations

import os
import signal
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Optional

from hermes_cli.kanban_launch_protocol import (
    BoardIdentity,
    DispatcherOwnerLease,
    LaunchProtocolError,
    acquire_dispatcher_owner,
    canonical_board_identity,
    claim_not_spawned,
    record_spawn_intent,
    record_spawned,
    release_claim,
)

__all__ = [
    "LaunchWiringConfig",
    "WrappedSpawn",
    "wrap_spawn_with_launch_protocol",
    "configure_launch_wiring",
    "clear_launch_wiring",
    "apply_launch_wiring",
]


@dataclass(frozen=True)
class LaunchWiringConfig:
    """Immutable inputs for wiring the ledger to one board's spawn path."""

    board_database: Any  # path-like; resolved via canonical_board_identity
    board_uuid: str
    owner_home: Any  # path-like; dispatcher owner home directory
    owner_generation: int
    policy_generation: int
    route_generation: int
    current_claim_token_fn: Callable[[str], str]
    next_run_generation_fn: Callable[[str], int]

    def __post_init__(self) -> None:
        for name in (
            "owner_generation",
            "policy_generation",
            "route_generation",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name} must be a positive integer")


class WrappedSpawn:
    """spawn_fn replacement that records every launch in the kernel ledger."""

    def __init__(
        self,
        inner: Callable[..., Optional[int]],
        launch_conn: sqlite3.Connection,
        config: LaunchWiringConfig,
    ) -> None:
        self._inner = inner
        self._conn = launch_conn
        self._cfg = config
        os.makedirs(config.owner_home, mode=0o700, exist_ok=True)
        self.board: BoardIdentity = canonical_board_identity(
            config.board_database, config.board_uuid
        )

    def __call__(
        self, task: Any, workspace: str, board: Any = None
    ) -> Optional[int]:
        task_id = task.id
        run_generation = self._cfg.next_run_generation_fn(task_id)
        claim_token = self._cfg.current_claim_token_fn(task_id)
        common = dict(
            board=self.board,
            task_id=task_id,
            run_generation=run_generation,
            dispatcher_owner_generation=self._cfg.owner_generation,
            policy_generation=self._cfg.policy_generation,
            route_generation=self._cfg.route_generation,
            claim_token=claim_token,
        )
        # Lease FIRST: if acquisition fails nothing has been persisted and
        # there is no claim to leak (Atomic round-1 finding 2).
        lease = acquire_dispatcher_owner(self._cfg.owner_home, self._cfg.owner_generation)
        claim_written = False

        def _cleanup(claimed: bool) -> None:
            """Best-effort claim retirement that never masks a kernel error.

            Atomic round-2 finding 2: release_claim itself raises when the
            lease was lost or the policy generation moved — exactly the
            failures we are cleaning up after. A claim that cannot be
            released stays behind as freeze-ack residue (terminal-state
            machinery: record_run_outcome / a future reaper), never a
            silently-dropped or double-raised error.
            """
            if not claimed:
                return
            try:
                release_claim(self._conn, owner_lease=lease, **common)
            except Exception:
                pass

        try:
            claim_not_spawned(self._conn, **common)
            claim_written = True
            try:
                record_spawn_intent(self._conn, owner_lease=lease, **common)
            except Exception:
                # Intent failed: retire the claim we just wrote before
                # propagating (Atomic round-1 finding 2).
                _cleanup(claim_written)
                claim_written = False
                raise
            try:
                pid = self._inner(task, workspace, board=board) if _accepts_board(
                    self._inner
                ) else self._inner(task, workspace)
            except Exception as exc:
                _cleanup(claim_written)
                claim_written = False
                raise LaunchProtocolError(f"launch aborted: {exc}") from exc
            if (
                pid is None
                or pid is False
                or isinstance(pid, bool)
                or not (isinstance(pid, int) and 0 < pid <= 0x7FFFFFFF)
            ):
                # Strict execution identity (Atomic round-2 finding 3 +
                # round-3 finding 2): only a positive real int within
                # int32 is representable in the ledger (SQLite/os.kill
                # choke on larger values; pid_t is int32). Strings
                # ("123"), floats, bools, and >int32 ints are rejected.
                _cleanup(claim_written)
                claim_written = False
                raise LaunchProtocolError(
                    f"launch aborted: spawn_fn returned invalid pid {pid!r}"
                )
            spawned_pid: int = pid
            try:
                record_spawned(
                    self._conn,
                    pid=spawned_pid,
                    owner_lease=lease,
                    **common,
                )
                claim_written = False
            except Exception:
                # A real worker exists but the ledger cannot record it
                # (Atomic round-1 finding 3): terminate the worker
                # best-effort, retire the claim, fail the launch.
                _best_effort_terminate(spawned_pid)
                _cleanup(claim_written)
                claim_written = False
                raise
            return pid
        finally:
            _cleanup(claim_written)
            # Atomic round-3 finding 2: an unguarded lease.release() here
            # could flip a fully-recorded successful spawn into a
            # dispatcher-level failure (worker alive, ledger says
            # spawned, yet the board claim is retired). Guard it: the
            # worst case is a stale on-disk lease file, which
            # acquire_dispatcher_owner already treats as lost-owner
            # evidence and the next acquisition cycle reclaims.
            try:
                lease.release()
            except Exception:
                pass


def _best_effort_terminate(pid: int) -> None:
    # Atomic round-3 finding 2: never let the terminator mask the ledger
    # failure it follows — any failure (OSError, OverflowError from a
    # pid outside os.kill's range, TypeError, ...) is swallowed.
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _accepts_board(fn: Callable[..., Any]) -> bool:
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "board" in sig.parameters


def wrap_spawn_with_launch_protocol(
    inner: Callable[..., Optional[int]],
    launch_conn: sqlite3.Connection,
    config: LaunchWiringConfig,
) -> WrappedSpawn:
    return WrappedSpawn(inner, launch_conn, config)


# ---------------------------------------------------------------------------
# Daemon-path integration (Atomic round-1 finding 1)
# ---------------------------------------------------------------------------

_WIRING: Optional[tuple[sqlite3.Connection, LaunchWiringConfig]] = None


def configure_launch_wiring(
    launch_conn: sqlite3.Connection, config: LaunchWiringConfig
) -> None:
    """Install module-global wiring for the dispatcher daemon path."""
    global _WIRING
    _WIRING = (launch_conn, config)


def clear_launch_wiring() -> None:
    global _WIRING
    _WIRING = None


def apply_launch_wiring(
    spawn_fn: Optional[Callable[..., Optional[int]]] = None,
) -> Optional[Callable[..., Optional[int]]]:
    """Decorate ``spawn_fn`` (or the default spawn) when wiring is configured.

    Called from the dispatcher daemon path before handing ``spawn_fn`` to
    ``dispatch_once``. With no wiring configured this is an exact
    pass-through of ``spawn_fn`` (possibly ``None`` for the default), so the
    live behavior is unchanged until activation explicitly configures it.
    """
    if _WIRING is None:
        return spawn_fn
    # Atomic round-3 finding 3: wrapping an already-wrapped callable
    # would nest owner-lease acquisition (the inner wrapper fails closed
    # against its own outer lease and nothing ever spawns). Wiring is
    # applied exactly once.
    if isinstance(spawn_fn, WrappedSpawn):
        return spawn_fn
    conn, config = _WIRING
    if spawn_fn is None:
        from hermes_cli.kanban_db import _default_spawn

        spawn_fn = _default_spawn
    assert spawn_fn is not None  # for type-checkers
    return wrap_spawn_with_launch_protocol(spawn_fn, conn, config)
