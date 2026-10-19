"""Shared fixtures for tests/hermes_state."""
from pathlib import Path

import pytest


@pytest.fixture
def worker(tmp_path):
    """A registered compute worker execution on session ``owned``: ``(db, RuntimeSessionStore)``."""
    # Imported here, not at module level: this conftest loads for every hermes_state test file.
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution

    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'cli', system_prompt='prefix')
    db.create_session('foreign', 'cli', system_prompt='secret')
    epoch = begin_runtime_epoch(db, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
    store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / 'outbox')
    yield db, store
    store.failure = None
    store.journal['pending'] = []
    store.close()
    db.close()


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


@pytest.fixture
def private_db_no_foreign_holders(tmp_path, monkeypatch):
    """Explicit premise for private-DB policy tests, never real foreign-holder tests.

    In-process holders remain real. Scanning unrelated host processes would add
    an uncontrolled input and can hit the suite's real-home I/O tripwire.
    """
    import hermes_state_holders

    def scan_owned_db(path):
        assert Path(path).resolve().parent == tmp_path.resolve()
        return []

    monkeypatch.setattr(hermes_state_holders, "foreign_state_db_holders", scan_owned_db)
