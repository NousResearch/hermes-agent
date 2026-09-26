"""Differential parity: durable session slash-control state — SQLite oracle vs PostgreSQL.

Both backends expose the same ``get_meta``/``set_meta``/``list_meta_prefix``
control-store surface (SQLite via ``state_meta`` JSON, PostgreSQL via the
compatibility facade over the focused control table).  Read-back payloads and
listing order must be identical.  Revision CAS and atomic transfer are
PostgreSQL-specific capabilities asserted against its focused API.
"""
from __future__ import annotations

import json
import uuid

import pytest

from hermes_state import SessionDB
from session_control_store import PostgreSQLSessionControlStore
from state_store import PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)

pytestmark = pytest.mark.integration

_PAYLOADS = {
    "goal": {"goal": "ship", "subgoals": ["test", "verify"], "status": "active"},
    "heartbeat": {"prompt": "check", "interval_seconds": 60, "status": "active"},
    "loop": {"prompt": "watch", "status": "active"},
}


@pytest.fixture(params=["sqlite", "postgresql"])
def backend(request, tmp_path, monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget):
    if request.param == "sqlite":
        store = SessionDB(db_path=tmp_path / "state.db")
    else:
        monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
        store = PostgreSQLSessionControlStore(
            PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema))
    try:
        yield request.param, store
    finally:
        store.close()


def test_control_state_readback_and_listing_parity(backend):
    kind, store = backend
    sid = f"ctl-diff-{uuid.uuid4()}"
    for ctrl, payload in _PAYLOADS.items():
        store.set_meta(f"{ctrl}:{sid}", json.dumps(payload))
        assert json.loads(store.get_meta(f"{ctrl}:{sid}")) == payload
    for ctrl in ("goal", "heartbeat", "loop"):
        rows = store.list_meta_prefix(f"{ctrl}:")
        assert [key for key, _ in rows] == [f"{ctrl}:{sid}"]
        assert json.loads(dict(rows)[f"{ctrl}:{sid}"]) == _PAYLOADS[ctrl]


def test_postgresql_control_revision_cas_and_atomic_transfer(
    monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget,
):
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    store = PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema)
    parent, child = f"ctl-{uuid.uuid4()}", f"ctl-{uuid.uuid4()}"
    try:
        store.ensure_session(parent, source="integration")
        store.ensure_session(child, source="integration", metadata={"parent_session_id": parent})
        goal = {"goal": "ship", "subgoals": ["test"], "status": "active"}
        assert store.put_session_control_state(parent, "goal", "active", goal) == 1
        first = store.get_session_control_state(parent, "goal")
        assert first and first["payload"] == goal and first["revision"] == 1
        amended = {**goal, "subgoals": ["test", "verify"]}
        assert store.put_session_control_state(parent, "goal", "active", amended, expected_revision=1) == 2
        # Stale CAS is rejected without mutating.
        assert store.put_session_control_state(parent, "goal", "active", goal, expected_revision=1) is None
        assert store.get_session_control_state(parent, "goal")["payload"] == amended
        assert store.transfer_session_control_states(parent, child) is True
        assert store.get_session_control_state(parent, "goal")["status"] == "cleared"
        assert store.get_session_control_state(child, "goal")["payload"] == amended
    finally:
        store.close()
