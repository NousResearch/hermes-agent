"""Broken local backends the storage conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``preflight_fence`` checks the turn fence before the
write's transaction instead of inside it; ``unfenced_create``, ``auxiliary_committed_before_refusal``
and ``preflight_gate`` are the create, accounting and admission-gate holes; the reads and the owner
lock leak through a tenant-blind pin (``reads_ignore_tenant``, ``unpinned_admission_reads``,
``tenant_blind_owner``); ``epoch_committed_before_refusal`` commits state before refusing, and
``resolve_unknown_current_epoch`` ignores the caller's epoch. Each mutant is applied
by the ``backend`` fixture when ``HERMES_STORAGE_MUTANT`` names it.
"""

import contextvars
from types import SimpleNamespace


def _foreign_metadata(monkeypatch):
    """set_title and delete skip the pin check."""
    from hermes_state_port_local import LocalSessionStore
    monkeypatch.setattr(LocalSessionStore, "set_title", lambda self, sid, title: self.db.set_session_title(sid, title))
    monkeypatch.setattr(LocalSessionStore, "delete", lambda self, sid: bool(self.db.delete_session(sid)))


def _ignored_claim_preconditions(monkeypatch):
    """claim_next discards the validated head and the admission gate."""
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.claim_next
    monkeypatch.setattr(LocalTurnAdmission, "claim_next",
                        lambda self, *, epoch, session_id, preconditions=None: original(
                            self, epoch=epoch, session_id=session_id))


def _unfenced_ordinary_append(monkeypatch):
    """append writes outside the bound turn, so a superseded turn's ordinary append lands."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_state_port_local import LocalSessionStore

    def append(self, sid, messages):
        self._pin()

        def write():
            with bind_profile_scope(self.home):
                return self.db.append_messages_batch(sid, messages)
        return contextvars.Context().run(write)
    monkeypatch.setattr(LocalSessionStore, "append", append)


def _no_owner_exclusion(monkeypatch):
    """acquire_owner always succeeds with a handle that holds nothing."""
    from hermes_state_port_local import LocalTurnAdmission
    monkeypatch.setattr(LocalTurnAdmission, "acquire_owner",
                        lambda self, **kw: (self._pin(), SimpleNamespace(release=lambda: None))[1])


def _producer_partitioned_fifo(monkeypatch):
    """claim_next schedules by producer before admission order."""
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.claim_next

    def claim_next(self, *, epoch, session_id, preconditions=None):
        def reorder(conn):
            rows = conn.execute("SELECT admission_id FROM session_admissions WHERE target_session_id=? "
                                "AND status='queued' ORDER BY principal_id, seq", (session_id,)).fetchall()
            seqs = sorted(r[0] for r in conn.execute("SELECT seq FROM session_admissions WHERE target_session_id=? "
                                                     "AND status='queued'", (session_id,)).fetchall())
            for (admission_id,), seq in zip(rows, seqs):
                conn.execute("UPDATE session_admissions SET seq=? WHERE admission_id=?", (-seq, admission_id))
            conn.execute("UPDATE session_admissions SET seq=-seq WHERE target_session_id=? AND seq<0", (session_id,))
        from hermes_state_fence import writes_as
        with writes_as(None):
            self.db._execute_write(reorder)
        return original(self, epoch=epoch, session_id=session_id, preconditions=preconditions)
    monkeypatch.setattr(LocalTurnAdmission, "claim_next", claim_next)


def _preflight_fence(monkeypatch):
    """The turn fence is checked by a read before the write, and the write itself runs unfenced:
    a takeover that commits in between is missed."""
    import hermes_state
    from hermes_state_fence import check_turn_lease, store_turn
    original = hermes_state.SessionDB._execute_write

    def execute_write(self, fn, patience_s=None):
        mark = store_turn(self)
        if mark is not None:
            with self._read_ctx() as conn:
                check_turn_lease(conn, mark)
        from hermes_state_fence import writes_as
        with writes_as(None):
            return original(self, fn, patience_s)
    monkeypatch.setattr(hermes_state.SessionDB, "_execute_write", execute_write)


def _unfenced_create(monkeypatch):
    """create keeps the pin check but writes outside the bound turn, so a superseded turn's
    create lands."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_state_port_local import LocalSessionStore
    original = LocalSessionStore.create

    def create(self, *args, **kwargs):
        self._pin()

        def write():
            with bind_profile_scope(self.home):
                return original(self, *args, **kwargs)
        return contextvars.Context().run(write)
    monkeypatch.setattr(LocalSessionStore, "create", create)


def _auxiliary_committed_before_refusal(monkeypatch):
    """A refused auxiliary usage write commits the usage outside the turn, then re-raises the
    refusal: the caller sees the refusal, the accounting changed anyway."""
    from agent.runtime_scope import bind_profile_scope
    from hermes_state_errors import SessionTurnLeaseLostError
    from hermes_state_port_local import LocalSessionStore
    original = LocalSessionStore.record_auxiliary_usage

    def record(self, session_id, task, **counts):
        try:
            return original(self, session_id, task, **counts)
        except SessionTurnLeaseLostError:
            def leak():
                with bind_profile_scope(self.home):
                    return original(self, session_id, task, **counts)
            contextvars.Context().run(leak)
            raise
    monkeypatch.setattr(LocalSessionStore, "record_auxiliary_usage", record)


def _preflight_gate(monkeypatch):
    """claim_next asks the admission gate before the claim's transaction, not inside it: a drain
    that begins while the claim waits for the writer is missed."""
    import dataclasses
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.claim_next

    def claim_next(self, *, epoch, session_id, preconditions=None):
        if preconditions is not None and preconditions.gate is not None:
            preconditions.gate.require_open()
            preconditions = dataclasses.replace(preconditions, gate=None)
        return original(self, epoch=epoch, session_id=session_id, preconditions=preconditions)
    monkeypatch.setattr(LocalTurnAdmission, "claim_next", claim_next)


def _agent_pin(self):
    """The pin's agent half alone: another tenant naming the store's home passes."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import current_scope
    from hermes_state_port import StoragePinRefused
    scope = current_scope()
    agent = scope.agent if scope is not None else get_hermes_home()
    if hermes_home_key(agent) != self._home_key:
        raise StoragePinRefused("the bound agent is not this session store's profile")


def _reads_ignore_tenant(monkeypatch):
    """Every read checks the bound agent but not the tenant: another tenant naming the home
    reads its rows and payload-bearing admissions."""
    from hermes_state_port_local import LocalSessionStore, LocalTurnAdmission

    def store_read(name):
        db_call = {"get": lambda self, sid: self.db.get_session(sid),
                   "get_title": lambda self, sid: self.db.get_session_title(sid),
                   "read_conversation": lambda self, sid, include_row_ids=False: self.db.get_messages_as_conversation(
                       sid, include_row_ids=include_row_ids),
                   "list_sessions": lambda self, limit=20, offset=0: self.db.list_sessions_rich(
                       limit=limit, offset=offset),
                   "read_auxiliary_usage": lambda self, sid: self.db.auxiliary_usage_by_task(sid)}[name]

        def method(self, *args, **kwargs):
            _agent_pin(self)
            return db_call[name](self, *args, **kwargs)
        return method

    def admission_read(name):
        def get(self, admission_id):
            from hermes_state_runtime import get_session_admission
            return get_session_admission(self.db, admission_id=admission_id)

        def find(self, *, principal_id, session_id, request_id):
            from hermes_state_runtime import find_session_admission
            return find_session_admission(self.db, principal_id=principal_id, session_id=session_id,
                                          request_id=request_id)

        def list_(self, session_id, *, pending_only=True):
            from hermes_state_runtime import list_session_admissions
            return list_session_admissions(self.db, session_id=session_id, pending_only=pending_only)

        def generation(self, session_id):
            row = self.db.get_session(session_id)
            return None if row is None else row["runtime_generation"]
        return {"get": get, "find": find, "list": list_, "generation": generation}[name]

    for name in ("get", "get_title", "read_conversation", "list_sessions", "read_auxiliary_usage"):
        monkeypatch.setattr(LocalSessionStore, name, store_read(name))
    for name in ("get", "find", "list", "generation"):
        monkeypatch.setattr(LocalTurnAdmission, name, admission_read(name))


def _unpinned_admission_reads(monkeypatch):
    """admission get, find and generation skip the pin check entirely."""
    from hermes_state_runtime import find_session_admission, get_session_admission
    from hermes_state_port_local import LocalTurnAdmission

    def generation(self, session_id):
        row = self.db.get_session(session_id)
        return None if row is None else row["runtime_generation"]
    monkeypatch.setattr(LocalTurnAdmission, "get",
                        lambda self, admission_id: get_session_admission(self.db, admission_id=admission_id))
    monkeypatch.setattr(LocalTurnAdmission, "find",
                        lambda self, *, principal_id, session_id, request_id: find_session_admission(
                            self.db, principal_id=principal_id, session_id=session_id, request_id=request_id))
    monkeypatch.setattr(LocalTurnAdmission, "generation", generation)


def _tenant_blind_owner(monkeypatch):
    """acquire_owner checks the bound agent but not the tenant."""
    from hermes_state_port_local import LocalTurnAdmission, _OwnerLock

    def acquire_owner(self, *, purpose=None):
        _agent_pin(self)
        from gateway.runtime_ownership import ProfileOwnership
        ownership = ProfileOwnership()
        ownership.reserve([self.home], purpose=purpose)
        return _OwnerLock(ownership, self.home)
    monkeypatch.setattr(LocalTurnAdmission, "acquire_owner", acquire_owner)


def _epoch_committed_before_refusal(monkeypatch):
    """begin_epoch commits a new owner epoch even under a foreign pin, then raises the refusal."""
    from hermes_state_port import StoragePinRefused
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.begin_epoch

    def begin_epoch(self, *, instance_id):
        try:
            self._pin()
        except StoragePinRefused:
            from hermes_state_runtime import begin_runtime_epoch
            begin_runtime_epoch(self.db, instance_id=instance_id)
            raise
        return original(self, instance_id=instance_id)
    monkeypatch.setattr(LocalTurnAdmission, "begin_epoch", begin_epoch)


def _resolve_unknown_current_epoch(monkeypatch):
    """resolve_unknown ignores the caller's epoch and resolves under the store's current one, so
    a stale owner resolves its row as long as the row's own status allows it."""
    from hermes_state_port_local import LocalTurnAdmission
    original = LocalTurnAdmission.resolve_unknown

    def resolve_unknown(self, *, epoch, admission_id, generation):
        self._pin()
        with self.db._read_ctx() as conn:
            current = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()[0]
        return original(self, epoch=current, admission_id=admission_id, generation=generation)
    monkeypatch.setattr(LocalTurnAdmission, "resolve_unknown", resolve_unknown)


MUTANTS = {
    "foreign_metadata": _foreign_metadata,
    "ignored_claim_preconditions": _ignored_claim_preconditions,
    "unfenced_ordinary_append": _unfenced_ordinary_append,
    "no_owner_exclusion": _no_owner_exclusion,
    "producer_partitioned_fifo": _producer_partitioned_fifo,
    "preflight_fence": _preflight_fence,
    "unfenced_create": _unfenced_create,
    "auxiliary_committed_before_refusal": _auxiliary_committed_before_refusal,
    "preflight_gate": _preflight_gate,
    "reads_ignore_tenant": _reads_ignore_tenant,
    "unpinned_admission_reads": _unpinned_admission_reads,
    "tenant_blind_owner": _tenant_blind_owner,
    "epoch_committed_before_refusal": _epoch_committed_before_refusal,
    "resolve_unknown_current_epoch": _resolve_unknown_current_epoch,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_STORAGE_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)
