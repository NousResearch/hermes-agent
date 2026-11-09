"""The local session storage backend: today's per-profile ``state.db`` behind the storage ports.

:class:`LocalSessionStore` is ``SessionDB``; :class:`LocalTurnAdmission` is the ledger in
``hermes_state_runtime`` plus the ``gateway.lock`` owner lock (``gateway.runtime_ownership``). Each
operation is the same call the code makes today, after one check of the pin: the bound scope
(``hermes_scope.current_scope()``) must be a local one whose agent is the store's own profile home.
A local store holds one agent and only agent-level records, so any other pin (another tenant,
another profile) is refused with ``StoragePinRefused`` before anything is read or written. With no
scope bound, the pin is the ambient profile (``get_hermes_home()``), as every local caller has it
today.

The store's identity is frozen when the ports open: the database file's canonical path and the
profile home that holds it, taken from the absolute path the handle recorded when it opened, never
recomputed against a later working directory (a handle with a relative path is refused). A
scope's agent is compared as the canonical absolute directory it names when the operation runs. A
directory alias of the profile (a symlinked home) is the same profile; a database file that is a
link into another directory is refused at open, since its rows belong to whichever profile owns
that directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from hermes_scope import LOCAL_TENANT, current_scope
from hermes_state_common import stat_db_file_identity
from hermes_state_port import (
    ANY_HEAD, ClaimPreconditions, OwnerHandle, SessionStorage, SessionStorageProvider, SessionStore,
    StoragePinRefused, TurnAdmission,
)


def store_identity(db: Any) -> tuple[Path, Path]:
    """Threat model: the pin guards against Hermes code binding the wrong scope to a store, not
    against the machine's owner rearranging their own files under a running process. The local
    backend serves only the local tenant, whose person owns every profile on this machine, so
    moving a directory alias, renaming or replacing a profile directory, or hard-linking one
    profile's database into another after it was opened is out of scope: the identity is fixed
    once, here, at wrap time, from a path that must still name the file the handle opened.
    A hosted backend never derives a tenant's identity from a filesystem path.

    ``(canonical database file, owning profile home)`` of the file handle *db* opened: its
    ``db_path``, which must already be absolute (``SessionDB`` keeps the path it is given), never
    re-read against a later working directory. Refuses (``StoragePinRefused``) a handle whose path is
    relative (which file it opened cannot be established now) and a database file that resolves
    outside the directory it is named in (a file link into another profile)."""
    opened = getattr(db, "db_path", None)
    if opened is None or not os.path.isabs(os.fspath(opened)):
        raise StoragePinRefused(f"session store handle {opened!r} does not record which file it opened")
    named = Path(os.fspath(opened))
    home = named.parent.resolve()
    canonical = named.resolve()
    if canonical.parent != home:
        raise StoragePinRefused(
            f"session store {named} is a link into {canonical.parent}, not a file of its profile {home}")
    # The path is re-read now; the handle's connection is to the file it opened then. They must be
    # the same file, or a directory alias moved in between would name another profile's home for
    # this connection's rows.
    recorded = getattr(db, "_db_file_identity", None)
    if recorded is None or stat_db_file_identity(canonical) != recorded:
        raise StoragePinRefused(
            f"session store {named} no longer names the file this handle opened")
    return canonical, home


class _Pinned:
    """A port over one ``SessionDB`` that refuses any pin but its own profile home."""

    def __init__(self, db: Any) -> None:
        from hermes_constants import hermes_home_key
        self.db = db
        self.db_file, self.home = store_identity(db)
        self._home_key = hermes_home_key(self.home)

    def _pin(self) -> None:
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise StoragePinRefused(f"a local session store serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise StoragePinRefused("the bound agent is not this session store's profile")


class LocalSessionStore(_Pinned, SessionStore):
    def get(self, session_id):
        self._pin()
        return self.db.get_session(session_id)

    def create(self, session_id, source, **fields):
        self._pin()
        return self.db.create_session(session_id, source, **fields)

    def end(self, session_id, reason):
        self._pin()
        self.db.end_session(session_id, reason)

    def append(self, session_id, messages):
        self._pin()
        return self.db.append_messages_batch(session_id, messages)

    def append_turn(self, session_id, messages):
        self._pin()
        from hermes_state_errors import TurnFenceRequiredError
        from hermes_state_fence import store_turn
        mark = store_turn(self.db)
        if mark is None:
            raise TurnFenceRequiredError(
                f"a turn write to session {session_id!r} needs that session's claimed turn bound")
        if session_id == mark.session:
            return self.db.append_messages_batch(session_id, messages)

        # A mid-turn compression moves the turn's transcript to a child; the claim stays on its
        # session, and the continuation is that claim's alone. Both checks run inside the write's
        # own transaction (the claim's own lease was checked at its start): a claim that lands on
        # the destination while this write waits for the writer cannot slip between them.
        def append_continuation(conn):
            from hermes_state_compression import _CHAIN_STEP_SQL, _walk_compression_chain
            if session_id not in _walk_compression_chain(
                    mark.session, lambda current: conn.execute(_CHAIN_STEP_SQL, (current,)).fetchone()):
                raise TurnFenceRequiredError(
                    f"a turn write to session {session_id!r} needs that session's claimed turn bound")
            own_claim = conn.execute(
                "SELECT 1 FROM session_admissions WHERE target_session_id=? "
                "AND status IN ('started','unknown') LIMIT 1", (session_id,)).fetchone()
            if own_claim is not None:
                raise TurnFenceRequiredError(
                    f"session {session_id!r} holds its own claimed turn; the claim it continues "
                    f"cannot write it")
            return self.db._append_messages_in_transaction(conn, session_id, messages)

        return self.db._execute_transcript_write(append_continuation, messages,
                                                 patience_s=self.db._TRANSCRIPT_WRITE_PATIENCE_S)

    def read_conversation(self, session_id, *, include_row_ids=False):
        self._pin()
        return self.db.get_messages_as_conversation(session_id, include_row_ids=include_row_ids)

    def get_title(self, session_id):
        self._pin()
        return self.db.get_session_title(session_id)

    def set_title(self, session_id, title):
        self._pin()
        return self.db.set_session_title(session_id, title)

    def list_sessions(self, *, limit=20, offset=0):
        self._pin()
        return self.db.list_sessions_rich(limit=limit, offset=offset)

    def delete(self, session_id):
        self._pin()
        return bool(self.db.delete_session(session_id))

    def replace_messages(self, session_id, messages, *, active_only=False, archive_dropped=False,
                         reject_active_turn_lease=False):
        self._pin()
        self.db.replace_messages(session_id, messages, active_only=active_only, archive_dropped=archive_dropped,
                                 reject_active_turn_lease=reject_active_turn_lease)

    def rewind_to_message(self, session_id, target_message_id, *, preserve_compaction_handoff=False,
                          expected_active_ids=None, expected_target_content=None):
        self._pin()
        return self.db.rewind_to_message(session_id, target_message_id,
                                         preserve_compaction_handoff=preserve_compaction_handoff,
                                         expected_active_ids=expected_active_ids,
                                         expected_target_content=expected_target_content)

    def rewind_user_turn(self, session_id, user_ordinal, **options):
        self._pin()
        return self.db.rewind_user_turn(session_id, user_ordinal, **options)

    def set_user_message_content(self, session_id, row_id, content, display_metadata=None):
        self._pin()
        return self.db.set_user_message_content(session_id, row_id, content, display_metadata)

    def deactivate_message(self, session_id, row_id):
        self._pin()
        return self.db.deactivate_message(session_id, row_id)

    def set_message_reaction(self, session_id, row_id, emoji, *, author="user"):
        self._pin()
        return self.db.set_message_reaction(session_id, row_id, emoji, author=author)

    def publish_compression_child(self, *, parent_session_id, child_session_id, source, messages, **options):
        self._pin()
        self.db.publish_compression_child(parent_session_id=parent_session_id, child_session_id=child_session_id,
                                          source=source, messages=messages, **options)

    def archive_and_compact(self, session_id, compacted_messages, **options):
        self._pin()
        return self.db.archive_and_compact(session_id, compacted_messages, **options)

    def record_usage(self, session_id, *, queued=True, **counts):
        self._pin()
        write = self.db.queue_token_counts if queued else self.db.update_token_counts
        write(session_id, **counts)

    def record_auxiliary_usage(self, session_id, task, **counts):
        self._pin()
        self.db.record_auxiliary_usage(session_id, task, **counts)

    def read_auxiliary_usage(self, session_id):
        self._pin()
        return self.db.auxiliary_usage_by_task(session_id)

    def flush_usage(self, timeout=5.0):
        self._pin()
        return self.db.flush_token_counts(timeout)


class _OwnerLock(OwnerHandle):
    def __init__(self, ownership: Any, home: Path) -> None:
        self._ownership, self._home = ownership, home

    def release(self) -> None:
        self._ownership.release(self._home)


class LocalTurnAdmission(_Pinned, TurnAdmission):
    def acquire_owner(self, *, purpose=None):
        self._pin()
        from gateway.runtime_ownership import ProfileOwnership
        ownership = ProfileOwnership()
        ownership.reserve([self.home], purpose=purpose)
        return _OwnerLock(ownership, self.home)

    def begin_epoch(self, *, instance_id):
        self._pin()
        from hermes_state_runtime import begin_runtime_epoch
        return begin_runtime_epoch(self.db, instance_id=instance_id)

    def recover(self, *, epoch):
        self._pin()
        from hermes_state_runtime import recover_session_inputs
        return recover_session_inputs(self.db, epoch=epoch)

    def admit(self, *, epoch, principal_id, session_id, request_id, payload, intent="queue"):
        self._pin()
        from hermes_state_runtime import admit_session_input
        return admit_session_input(self.db, epoch=epoch, principal_id=principal_id, session_id=session_id,
                                   request_id=request_id, payload=payload, intent=intent)

    def get(self, admission_id):
        self._pin()
        from hermes_state_runtime import get_session_admission
        return get_session_admission(self.db, admission_id=admission_id)

    def find(self, *, principal_id, session_id, request_id):
        self._pin()
        from hermes_state_runtime import find_session_admission
        return find_session_admission(self.db, principal_id=principal_id, session_id=session_id,
                                      request_id=request_id)

    def list(self, session_id, *, pending_only=True):
        self._pin()
        from hermes_state_runtime import list_session_admissions
        return list_session_admissions(self.db, session_id=session_id, pending_only=pending_only)

    def claim_next(self, *, epoch, session_id, preconditions: Optional[ClaimPreconditions] = None):
        self._pin()
        from hermes_state_runtime import claim_session_input
        pre = preconditions or ClaimPreconditions()
        gate = pre.gate
        # The gate runs inside the claim's write transaction, after any wait for the writer.
        guard = None if gate is None else (lambda _conn: gate.require_open())
        if pre.head is ANY_HEAD:
            return claim_session_input(self.db, epoch=epoch, session_id=session_id, _guard=guard)
        return claim_session_input(self.db, epoch=epoch, session_id=session_id, _guard=guard, head=pre.head)

    def steer(self, *, epoch, admission_id, generation):
        self._pin()
        from hermes_state_runtime import steer_session_input
        return steer_session_input(self.db, epoch=epoch, admission_id=admission_id, generation=generation)

    def revert_steer(self, *, epoch, admission_id, generation):
        self._pin()
        from hermes_state_runtime import revert_steer_input
        return revert_steer_input(self.db, epoch=epoch, admission_id=admission_id, generation=generation)

    def settle(self, *, epoch, admission_id, generation, outcome, result=None, transfers=None):
        self._pin()
        from hermes_state_runtime import settle_session_input
        return settle_session_input(self.db, epoch=epoch, admission_id=admission_id, generation=generation,
                                    outcome=outcome, result=result, _transfers=transfers)

    def cancel(self, *, epoch, admission_id):
        self._pin()
        from hermes_state_runtime import cancel_session_input
        return cancel_session_input(self.db, epoch=epoch, admission_id=admission_id)

    def resolve_unknown(self, *, epoch, admission_id, generation):
        self._pin()
        from hermes_state_runtime import resolve_unknown_session_input
        return resolve_unknown_session_input(self.db, epoch=epoch, admission_id=admission_id,
                                             generation=generation)

    def generation(self, session_id):
        self._pin()
        row = self.db.get_session(session_id)
        return None if row is None else row["runtime_generation"]

    def mutate_session(self, *, epoch, principal_id, session_id, request_id, expected_revision, operation,
                       payload, expected_generation=None):
        self._pin()
        from hermes_state_runtime import mutate_runtime_session
        return mutate_runtime_session(self.db, epoch=epoch, principal_id=principal_id, session_id=session_id,
                                      request_id=request_id, expected_revision=expected_revision,
                                      operation=operation, payload=payload,
                                      expected_generation=expected_generation)

    def register_worker(self, *, epoch, execution_id, session_id, generation, kind, adoption_secret,
                        require_idle=False):
        self._pin()
        from hermes_state_runtime import register_worker_execution
        return register_worker_execution(self.db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                         generation=generation, kind=kind, adoption_secret=adoption_secret,
                                         require_idle=require_idle)

    def adopt_worker(self, *, epoch, execution_id, session_id, generation, adoption_secret):
        self._pin()
        from hermes_state_runtime import adopt_worker_execution
        return adopt_worker_execution(self.db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                      generation=generation, adoption_secret=adoption_secret)

    def apply_worker_op(self, *, epoch, execution_id, session_id, generation, sequence, operation, payload):
        self._pin()
        from hermes_state_runtime import mutate_worker_execution
        return mutate_worker_execution(self.db, epoch=epoch, execution_id=execution_id, session_id=session_id,
                                       generation=generation, sequence=sequence, operation=operation,
                                       payload=payload)


class LocalSessionStorageProvider(SessionStorageProvider):
    """Today's storage: one ``state.db`` per profile home, holding that agent's records only."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self, db: Any = None) -> SessionStorage:
        """Ports over *db*, or over the bound profile's shared ``state.db`` (released on
        ``close``). The store's file and profile are frozen here (``store_identity``; a file link
        into another directory is refused now); the pin is checked by every operation."""
        if db is not None:
            return SessionStorage(LocalSessionStore(db), LocalTurnAdmission(db))
        from hermes_constants import get_hermes_home
        from hermes_state_registry import acquire, release
        named = get_hermes_home() / "state.db"
        if named.resolve().parent != named.parent.resolve():  # the registry resolves links itself
            raise StoragePinRefused(f"session store {named} is a link into {named.resolve().parent}")
        shared = acquire(named)
        try:
            sessions, admission = LocalSessionStore(shared), LocalTurnAdmission(shared)
        except BaseException:
            release(shared)
            raise

        def close() -> None:
            release(shared)
        return SessionStorage(sessions, admission, close)

    def assert_isolating(self) -> None:
        """A local store is one file per profile and refuses every pin but its own (``_Pinned``);
        there is no shared role that could bypass that."""
        return
