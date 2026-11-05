"""The physical admin capability: file-level operations outside the contracts.

The assembly opens a handle with ``open_store_admin(target, *, read_only=True)`` (a profile's store or one database file) or ``open_profile_repair(scope, *, read_only)`` (crossed-profile repair on one profile's store); each handle acts on the one profile or file it was opened for. A backend without the capability answers either door with :class:`AdminUnsupported`, and ``PersistenceCapabilities.physical_admin`` says in advance which answer a backend gives: callers branch on the flag or on the result, never on a backend's name. Path-targeted operations (``verify_file``, ``recover_transcripts``, ``has_runtime_ledger_rows``, ``has_runtime_ledger_table``) address a file with no live store behind it. Returns are plain dicts and tuples; errors are the typed family (``CorruptStoreError``, ``StoreReplaced``, ``StoreContention``, ``DeclarativeDrift``, ``SessionLedgered``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Literal, Mapping, Optional, Protocol, runtime_checkable

__all__ = [
    "BROWSE_RECONCILE_BUDGET_S", "BROWSE_RECONCILE_PURPOSE", "PHYSICAL_ADMIN", "WRITE_PROBE_SKIPPED", "AdminLockHandle", "AdminUnsupported", "HealthCheckUnavailable", "ProfileRepairAdmin", "RepairGuard",
    "SessionRecoveryError", "SessionRecoverySafetyError", "SessionRecoverySourceError", "StoreAdmin",
]

# The capability both admin doors answer for.
PHYSICAL_ADMIN = "physical_admin"

# What ``StoreAdmin.write_health_reason`` answers when it skipped the probe: the notice the doctor prints for it.
WRITE_PROBE_SKIPPED = ("state.db write-health probe skipped: store is held by a live writer and larger than 1 GB "
                       "(run 'hermes doctor --fix' to probe it)")

# How long ``StoreAdmin.reconcile_drift_for_browse`` may hold a profile's maintenance lock for its schema catch-up,
# and the purpose that lock records; a gateway booting meanwhile waits the budget out.
BROWSE_RECONCILE_BUDGET_S = 3.0
BROWSE_RECONCILE_PURPOSE = "browse_schema_reconcile"


@dataclass(frozen=True, slots=True)
class AdminUnsupported:
    """A door's answer on a backend without the admin capability: ``capability`` names it (:data:`PHYSICAL_ADMIN`), ``backend_kind`` is the backend's provider name."""

    capability: str
    backend_kind: str


class HealthCheckUnavailable(str):
    """A health-probe reason meaning the probe could not run, not that the store is damaged: callers must neither report the store healthy nor route it to repair."""


class SessionRecoveryError(RuntimeError):
    """Base error for offline session recovery."""


class SessionRecoverySafetyError(SessionRecoveryError):
    """Raised before recovery when a path or overwrite guard fails."""


class SessionRecoverySourceError(SessionRecoveryError):
    """Raised when the source cannot provide the required canonical tables."""


class AdminLockHandle(Protocol):
    """An acquired auto-maintenance lock, opaque to the caller; it goes back to the handle that acquired it."""


class RepairGuard(Protocol):
    """The operations a caller runs on the exclusive repair connection while it holds the guard."""

    def wal_checkpoint_passive(self) -> None:
        """Copy the WAL's committed frames into the store file without waiting on anyone."""
        ...


@runtime_checkable
class StoreAdmin(Protocol):
    """Physical operations on one store file."""

    def stats(self) -> dict[str, Any]:
        """Row counts and file facts of the store."""
        ...

    def holder_count(self) -> Optional[int]:
        """How many processes hold the store file open; None when it cannot be told."""
        ...

    def deleted_sidecar_holders(self) -> list[tuple[int, str]]:
        """``(pid, path)`` of each process holding a deleted WAL or shared-memory sidecar of the store."""
        ...

    def foreign_holders(self) -> list[tuple[int, str]]:
        """``(pid, path)`` of each other process holding the store or a sidecar."""
        ...

    def held_refusal(self, *, command: str, force_hint: Optional[str] = "--force") -> Optional[str]:
        """The refusal text for running ``command`` while other processes hold the store, or None when none does."""
        ...

    def live_writer_holds(self) -> bool:
        """Whether a live writer process holds the store."""
        ...

    def opens_cleanly(self) -> Optional[str]:
        """None when the store opens and reads; else the failure text, a :class:`HealthCheckUnavailable` when the probe itself could not run (a probe copy that cannot be made, written, checked or reopened)."""
        ...

    def has_structural_damage(self) -> bool:
        """Whether the store file reports structural damage."""
        ...

    def exclusive_repair_guard(self) -> ContextManager[RepairGuard]:
        """Hold one connection that excludes every other opener of the file for the block, yielding the :class:`RepairGuard` operations that run on it; ``StoreContention`` when exclusion cannot be taken."""
        ...

    def write_health_reason(self, *, should_fix: bool) -> Optional[str]:
        """Why a rolled-back write probe of the store fails, or None when it passes, or a :class:`HealthCheckUnavailable` when the probe could not run; a store a live writer holds is probed on a snapshot. Answers :data:`WRITE_PROBE_SKIPPED` without probing when such a held store is over 1 GB and ``should_fix`` is False. A failed snapshot copy raises the typed store errors (``CorruptStoreError`` for a malformed file)."""
        ...

    def integrity_check(self, *, deadline_s: Optional[float] = None) -> str:
        """The integrity check's answer, ``"ok"`` for a sound file."""
        ...

    def journal_mode(self) -> Optional[str]:
        """The store's journal mode on disk."""
        ...

    def set_journal_mode(self, mode: str, *, force: bool = False) -> tuple[str, str]:
        """Switch the file to journal mode ``mode`` (``delete`` or ``wal``) while no other process holds it; returns the mode its header reported before and the one it reports after, the same mode twice when it already was ``mode``. ``force`` waives only an incomplete holder scan, never a holder the scan found. A holder, or an opener that appears before the switch, raises ``StoreContention``; every other refusal raises ``PermanentStoreError`` with the reason."""
        ...

    def repair_schema(self, *, backup: bool = True) -> dict[str, Any]:
        """Rebuild a malformed schema, with a backup first by default."""
        ...

    def schema_drift_reason(self) -> Optional[str]:
        """The read error naming a declared table or column the stored schema lacks, or None; read on a connection that never creates or migrates the file. Any other read error raises its typed store error."""
        ...

    def repair_declarative_schema(self, *, backup: bool = True) -> dict[str, Any]:
        """Bring the stored schema back to the declared one."""
        ...

    def reconcile_drift_for_browse(self) -> bool:
        """Reconcile schema drift so a read-only browse can proceed; whether it did."""
        ...

    def fts_optimize_available(self) -> bool:
        """Whether ``optimize_fts_storage`` has work: an older search index layout, an interrupted optimize, or an empty index beside stored messages. False when full-text search is unavailable or the handle is read-only; read errors raise the typed store errors."""
        ...

    def optimize_fts_storage(self, *, progress_cb: Optional[Callable[[dict[str, Any]], None]] = None, vacuum: bool = True) -> dict[str, Any]:
        """Rebuild the search index storage; ``progress_cb`` is called between its transactions, never inside one."""
        ...

    def fts_layout_drift(self) -> list[str]:
        """The search index objects whose layout differs from the current one."""
        ...

    def fts_storage_notice(self) -> dict[str, Any]:
        """What the optimize-storage notice reads, on one read-only connection that never creates or migrates the file: ``drift`` (as :meth:`fts_layout_drift`), ``legacy_inline`` (whether the index still stores message text inline) and ``interrupted`` (whether an optimize or its trash tables were left unfinished). Read errors raise the typed store errors."""
        ...

    def vacuum(self) -> int:
        """Rewrite the file to reclaim free pages; returns the number of search indexes optimized first."""
        ...

    def freelist_ratio(self) -> Optional[float]:
        """Fraction of pages a vacuum would give back; None when it cannot be read."""
        ...

    def logical_size_bytes(self) -> Optional[int]:
        """The file's logical size (page count times page size); None when it cannot be read."""
        ...

    def vacuum_blockers(self) -> tuple[tuple[int, str], ...]:
        """``(pid, path)`` of each other holder that makes a vacuum unsafe now."""
        ...

    def try_acquire_auto_maintenance_lock(self) -> Optional[AdminLockHandle]:
        """The auto-maintenance lock, or None when another process holds it."""
        ...

    def release_auto_maintenance_lock(self, handle: AdminLockHandle) -> None:
        """Release a lock :meth:`try_acquire_auto_maintenance_lock` returned."""
        ...

    def backup_to(self, dst: Path, *, timeout_seconds: float = 10.0) -> bool:
        """Write a consistent copy of the store to ``dst``; whether it completed."""
        ...

    def preflight_backup(self) -> dict:
        """Publish an emergency snapshot beside the store and keep the newest two; returns ``{path, bytes}``, or ``{path: None, message}`` when the store file does not exist. Raises ``RuntimeError`` when the copy or its integrity check fails, keeping the earlier snapshots."""
        ...

    @staticmethod
    def verify_file(path: Path, *, max_bytes: int = 2 << 30) -> Any:
        """``{valid, message, size}`` for a database file with no live store behind it: its size, its header, then a full integrity check, or a cheap structural probe instead for a file over ``max_bytes`` (2 GiB by default)."""
        ...

    def restore_from(self, snapshot: Path) -> bool:
        """Replace the store's pages with ``snapshot``'s; the snapshot is never modified."""
        ...

    def restore_epoch_floor(self) -> int:
        """The runtime epoch a restore over the file must keep, read on a read-only connection: 0 when the file or its epoch row is absent. Raises ``OSError`` naming the transcript salvage when the epoch cannot be read."""
        ...

    def session_row_counts(self) -> Optional[tuple[int, int]]:
        """``(sessions, messages)`` row counts, or None when they cannot be read."""
        ...

    def session_row_count(self) -> int:
        """Every session row of the store, archived and hidden ones included, read on a read-only connection; read errors raise the typed store errors with the database's message as their text."""
        ...

    @staticmethod
    def recover_transcripts(
        source: Path, output: Path, *, inspect_only: bool = False, work_dir: Optional[Path] = None,
        chunk_size: int = 1_000, progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Recover the readable transcripts of a damaged file at ``source`` into the new file ``output`` without opening ``source`` itself: the snapshot lies under ``work_dir`` (a temporary folder by default), rows are copied ``chunk_size`` at a time with ``progress_cb`` called between chunks, and ``allow_partial`` keeps what can be salvaged instead of refusing a source that is not fully recoverable. ``inspect_only`` reports which tables are readable and writes nothing. Raises ``SessionRecoveryError`` (a ``RuntimeError``; ``SessionRecoverySafetyError`` when a path or overwrite guard fails, ``SessionRecoverySourceError`` when the source lacks a readable table recovery needs) or ``OSError``."""
        ...

    @staticmethod
    def has_runtime_ledger_rows(path: Path) -> bool:
        """Whether the file at ``path`` holds runtime ledger rows."""
        ...

    @staticmethod
    def has_runtime_ledger_table(path: Path) -> bool:
        """Whether the file at ``path`` holds the runtime admission ledger table, rows or not, read on a read-only connection that never modifies the file; read errors raise the typed store errors (``CorruptStoreError`` for a file that is not a database)."""
        ...

    def change_stamp(self) -> Optional[int]:
        """A value that changes whenever the store's content changes; None when it cannot be read."""
        ...


@runtime_checkable
class ProfileRepairAdmin(Protocol):
    """Crossed-profile repair on the one profile's store the handle was opened for."""

    def find_crossed_profile_sessions(self, owner: str) -> dict[str, list[dict[str, Any]]]:
        """The rows whose key or label names a profile other than ``owner``, by kind."""
        ...

    def list_gateway_routing_rows(self) -> list[dict[str, Any]]:
        """Every gateway routing row."""
        ...

    def find_profile_less_telegram_topic_rows(self) -> list[dict[str, Any]]:
        """Telegram topic rows that carry no profile."""
        ...

    def count_messages_all(self, session_id: str) -> int:
        """Every message row of the session, whatever its state."""
        ...

    def relabel_sessions_to_key_profile(self, ids: Iterable[str]) -> int:
        """Set each session's profile label to the profile its key names; returns the rows changed."""
        ...

    def sever_crossed_parents(self, ids: Iterable[str]) -> int:
        """Clear the parent link of each session whose parent belongs to another profile; returns the rows changed."""
        ...

    def export_session_for_move(self, session_id: str) -> dict[str, Any] | None:
        """The session with its messages, for import into another profile's store; None when absent."""
        ...

    def import_moved_session(self, payload: Mapping[str, Any], *, profile_name: str) -> Literal["present", "imported"]:
        """Import a moved session; ``present`` when it is already here."""
        ...

    def delete_moved_session(self, session_id: str) -> bool:
        """Delete a session after its move; whether it was removed."""
        ...

    def rekey_legacy_main_sessions(self, ids: Iterable[str], profile: str) -> int:
        """Rewrite the legacy main-profile keys of the sessions to ``profile``; returns the rows changed."""
        ...

    def delete_gateway_routing_rows(self, rows: Iterable[tuple[str, str]]) -> int:
        """Delete the routing rows; returns how many went."""
        ...

    def insert_gateway_routing_rows_if_absent(self, rows: Iterable[tuple[str, str, str, float]]) -> int:
        """Insert the routing rows not already present; returns how many were added."""
        ...

    def relabel_telegram_topic_rows(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Relabel the topic rows to their profiles; returns counts by outcome."""
        ...

    def key_profiles_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """The profiles that route keys for the chat."""
        ...
