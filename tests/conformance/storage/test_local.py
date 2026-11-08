"""The storage conformance suite against the local backend (today's per-profile ``state.db``).

Agents are profile homes under ``tmp_path``; a replica is a second ``SessionDB`` on the same file
(another process's connection). The suite's list of invariants it does NOT cover applies here
too (``hermes_conformance.storage.suite``): the local store holds the agent level only, and the
fixture cannot vary participant, actor or owner.
"""

import sqlite3
import threading
from contextlib import closing, contextmanager

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state, bind_turn
from hermes_conformance.storage import AdmissionConformance, Backend, FenceConformance
from hermes_scope import Group, Principal, Scope
from hermes_state import SessionDB
from hermes_state_port_local import LocalSessionStorageProvider


def local_backend(tmp_path, monkeypatch):
    """The local backend's :class:`Backend`, and the SessionDB handles it opened (to close)."""
    handles = []
    homes = {name: tmp_path / name for name in ("a", "b", "unpinned")}
    for home in homes.values():
        home.mkdir(exist_ok=True)

    def connect(agent):
        db = SessionDB(homes[agent] / "state.db")
        handles.append(db)
        return LocalSessionStorageProvider().open(db)

    @contextmanager
    def unbound():
        # No scope: a local store's pin is the ambient profile, here one that owns no store.
        monkeypatch.setenv("HERMES_HOME", str(homes["unpinned"]))
        yield

    @contextmanager
    def turn(storage, session_id, claimed):
        db = storage.sessions.db
        with bind_turn(session_id, claimed["admission_id"], storage.admission.fence(claimed),
                       store=db.db_path):
            yield

    def foreign_tenant(agent):
        return bind_state(Scope("acme", str(homes[agent]), Principal("u"), owner=Group("g"), role="member"))

    @contextmanager
    def contend(agent, takeover):
        """Another replica holds the writer; once a write waits behind it, that replica's
        transaction takes the profile over (a new owner epoch) and commits, or with no
        *takeover* releases the writer having changed nothing."""
        waiting = threading.Event()
        for db in handles:
            real = db._sleep_before_write_retry

            def sleep(*args, _real=real):
                waiting.set()
                return _real(*args)
            db._sleep_before_write_retry = sleep
        holder = sqlite3.connect(homes[agent] / "state.db", timeout=0, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            yield waiting
            if takeover:
                holder.execute("UPDATE runtime_epoch SET epoch=epoch+1, instance_id='contender' WHERE singleton=1")
            holder.execute("COMMIT")
        except BaseException:
            holder.execute("ROLLBACK")
            raise
        finally:
            holder.close()

    backend = Backend(open=connect, replica=connect, bind=lambda agent: bind_profile_scope(homes[agent]),
                      foreign_tenant=foreign_tenant, unbound=unbound, turn=turn,
                      contend_takeover=lambda agent: contend(agent, True),
                      hold_writer=lambda agent: contend(agent, False))
    return backend, handles


@pytest.fixture
def backend(tmp_path, monkeypatch):
    from tests.conformance.storage.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    backend, handles = local_backend(tmp_path, monkeypatch)
    yield backend
    for db in handles:
        with closing(db):
            pass


# The suite's three parts run as separate modules (pinning in test_local_pinning.py) so each file
# stays well inside the per-file runner budget.
class TestLocalFences(FenceConformance):
    pass


class TestLocalAdmission(AdmissionConformance):
    pass
