"""Real PostgreSQL evidence for durable session slash-control state."""
from __future__ import annotations

import uuid

import pytest

from hermes_cli.goals import GoalManager, load_goal
from hermes_cli.heartbeat import HeartbeatManager, load_heartbeat
from hermes_cli.loops import LoopManager, load_loop
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from session_control_store import clear_session_control_store_cache
from state_store_runtime_readiness import trap_state_db_opens
from state_store import PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)

pytestmark = pytest.mark.integration


def test_postgresql_session_control_state_roundtrip_cas_transfer_and_cascade(
    monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget,
):
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    store = PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema)
    parent, child = f"control-{uuid.uuid4()}", f"control-{uuid.uuid4()}"
    try:
        store.ensure_session(parent, source="integration")
        store.ensure_session(child, source="integration", metadata={"parent_session_id": parent})
        goal = {"goal": "ship", "subgoals": ["test"], "status": "active"}
        assert store.put_session_control_state(parent, "goal", "active", goal) == 1
        first = store.get_session_control_state(parent, "goal")
        assert first and first["payload"] == goal and first["revision"] == 1
        amended = {**goal, "subgoals": ["test", "verify"]}
        assert store.put_session_control_state(parent, "goal", "active", amended, expected_revision=1) == 2
        assert store.put_session_control_state(parent, "goal", "active", goal, expected_revision=1) is None
        for kind, payload in (("heartbeat", {"prompt": "check", "interval_seconds": 60, "status": "active"}), ("loop", {"prompt": "watch", "status": "active"})):
            assert store.put_session_control_state(parent, kind, "active", payload)
        assert store.transfer_session_control_states(parent, child)
        assert store.get_session_control_state(parent, "goal")["status"] == "cleared"
        assert store.get_session_control_state(child, "goal")["payload"] == amended
        assert {row["session_id"] for row in store.list_session_control_states("loop", status="active")} == {child}
        with store._connection() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE id=%s", (child,))
        assert store.list_session_control_states("goal") == [store.get_session_control_state(parent, "goal") | {"session_id": parent}]
    finally:
        store.close()


def test_selected_postgresql_controls_roundtrip_without_state_db(
    tmp_path, monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget,
):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 5\n    pool_max_size: 2\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: postgresql_test_target.schema)
    token = set_hermes_home_override(str(home))
    try:
        with trap_state_db_opens(home) as opens:
            assert GoalManager("slash-control").state is None
        assert opens == []
        goal = GoalManager("slash-control")
        goal.set("deliver")
        goal.add_subgoal("verify")
        assert load_goal("slash-control").subgoals == ["verify"]
        heartbeat = HeartbeatManager("slash-control")
        heartbeat.set("check", 60)
        heartbeat.pause()
        assert load_heartbeat("slash-control").status == "paused"
        loop = LoopManager("slash-control")
        loop.set("watch", interval_seconds=60)
        loop.pause()
        assert load_loop("slash-control").status == "paused"
        assert not (home / "state.db").exists()
        reopened = GoalManager("slash-control")
        assert reopened.state and reopened.state.goal == "deliver" and reopened.state.subgoals == ["verify"]
        reopened.pause()
        reopened.resume()
        reopened.clear()
        heartbeat.resume()
        heartbeat.clear()
        loop.resume()
        loop.clear()
        assert load_goal("slash-control").status == "cleared"
        assert load_heartbeat("slash-control") is None
        assert load_loop("slash-control").status == "cleared"
    finally:
        clear_session_control_store_cache()
        reset_hermes_home_override(token)
