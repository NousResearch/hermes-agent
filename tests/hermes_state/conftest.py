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


@pytest.fixture()
def no_foreign_db_holders(tmp_path, monkeypatch):
    """These policy/repair tests own their temp DB; host processes are unrelated.

    Keep the in-process holder and SQLite lock checks real. Foreign-holder
    detection and refusal have dedicated maintenance/holder contract suites.
    """
    def isolated_holders(db_path):
        assert Path(db_path).is_relative_to(tmp_path)
        return []

    monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", isolated_holders)
