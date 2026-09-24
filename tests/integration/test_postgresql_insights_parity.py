"""Real PostgreSQL observability contract for backend-neutral insights."""
from __future__ import annotations

import time

import pytest

from agent.insights import InsightsEngine
from state_store import MessageRecord, open_state_store

pytestmark = pytest.mark.integration

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONFIG = {"state_store": {"backend": "postgresql", "postgresql": {
    "dsn_env": "HERMES_STATE_STORE_TEST_DSN", "connect_timeout_seconds": 5, "pool_max_size": 2,
}}}


def test_postgresql_insights_snapshot_flushes_and_matches_engine_contract(postgresql_test_target, monkeypatch):
    """PG returns canonical rows (including decoded JSONB) without opening SQLite."""
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: postgresql_test_target.schema)
    store = open_state_store(_CONFIG)
    try:
        session_id = "pg-insights-contract"
        store.ensure_session(session_id, "cli", metadata={"model": "test/model"})
        tool_calls = [{"function": {"name": "skill_view", "arguments": '{"name":"postgresql-observability"}'}}]
        store.append_message_record(session_id, MessageRecord(role="assistant", tool_calls=tool_calls, timestamp=time.time()))
        store.append_message_record(session_id, MessageRecord(role="tool", tool_name="skill_view", timestamp=time.time()))
        store.queue_token_counts(session_id, input_tokens=7, output_tokens=3, model="test/model", billing_provider="local", source="cli")
        report = InsightsEngine(store).generate(days=1, source="cli")
        route = store.get_recent_session_model_route(session_id)
        assert route is not None and route["model"] == "test/model"
        assert report["empty"] is False
        assert report["overview"]["total_tokens"] == 10
        assert report["tools"] == [{"tool": "skill_view", "count": 1, "percentage": 100.0}]
        assert report["skills"]["summary"]["total_skill_loads"] == 1
    finally:
        store.close()
