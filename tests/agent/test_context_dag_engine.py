from unittest.mock import MagicMock

from agent.context_dag_engine import DAGContextEngine
from agent.context_engine import ContextCompressionResult
from agent.context_dag_store import ContextDAGStore
from hermes_state import SessionDB


def _db_with_messages(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess", "cli")
    db.append_message("sess", "user", "old question")
    db.append_message("sess", "assistant", "old answer")
    db.append_message("sess", "user", "fresh question")
    return db


def test_dag_engine_assembles_projection_when_enabled(tmp_path):
    db = _db_with_messages(tmp_path)
    store = ContextDAGStore(db)
    summary = store.create_summary_node(
        session_id="sess",
        kind="leaf",
        summary_text="Old exchange summary",
        metadata={"source_span": {"start_message_id": 1, "end_message_id": 2}},
    )
    store.write_active_projection(
        session_id="sess",
        engine_version=DAGContextEngine.ENGINE_VERSION,
        projection=[{"type": "summary", "summary_id": summary.id}],
        fresh_tail_start_message_id=3,
        latest_raw_message_id=3,
        token_estimate=10,
    )
    store.write_checkpoint(session_id="sess", last_ingested_message_id=3)

    engine = DAGContextEngine(session_db=db, enabled=True, max_context_tokens=200)
    engine.on_session_start("sess", platform="cli")

    result = engine.compress([{"role": "user", "content": "legacy fallback"}])
    assembled = result.messages

    assert isinstance(result, ContextCompressionResult)
    assert result.projection_only is True
    assert result.preserves_session is True
    assert result.changed is True
    assert result.warning is None
    assert result.raw_checkpoint is not None
    assert any("Old exchange summary" in (m.get("content") or "") for m in assembled)
    assert assembled[-1]["content"] == "fresh question"
    assert all("id" not in m and "metadata" not in m for m in assembled)
    assert engine.compression_count == 1


def test_dag_engine_missing_projection_falls_back_to_raw_messages(tmp_path):
    db = _db_with_messages(tmp_path)
    engine = DAGContextEngine(session_db=db, enabled=True, max_context_tokens=200)
    engine.on_session_start("sess", platform="cli")

    result = engine.compress([{"role": "user", "content": "fallback"}])
    assembled = result.messages

    assert result.projection_only is True
    assert result.preserves_session is True
    assert result.changed is True
    assert result.warning == "missing_projection"
    assert [m["content"] for m in assembled] == ["old question", "old answer", "fresh question"]
    assert all("id" not in m and "metadata" not in m for m in assembled)
    assert engine.get_status()["fallback_reason"] == "missing_projection"


def test_dag_engine_corrupt_projection_falls_back_to_input_messages(tmp_path):
    db = _db_with_messages(tmp_path)
    engine = DAGContextEngine(session_db=db, enabled=True, max_context_tokens=200)
    engine.on_session_start("sess", platform="cli")
    engine.store.read_active_projection = MagicMock(side_effect=ValueError("bad projection"))
    input_messages = [{"role": "user", "content": "safe input", "id": 999, "metadata": {"internal": True}}]
    original_input = [dict(input_messages[0], metadata=dict(input_messages[0]["metadata"]))]

    result = engine.compress(input_messages)
    assembled = result.messages

    assert result.projection_only is True
    assert result.preserves_session is True
    assert result.changed is True
    assert "bad projection" in result.warning
    assert assembled == [{"role": "user", "content": "safe input"}]
    assert input_messages == original_input
    assert "bad projection" in engine.get_status()["fallback_reason"]


def test_dag_engine_disabled_is_passthrough(tmp_path):
    db = _db_with_messages(tmp_path)
    engine = DAGContextEngine(session_db=db, enabled=False)
    input_messages = [{"role": "user", "content": "unchanged"}]

    result = engine.compress(input_messages)

    assert isinstance(result, ContextCompressionResult)
    assert result.messages == input_messages
    assert result.projection_only is True
    assert result.preserves_session is True
    assert result.changed is False
    assert "disabled" in result.warning
    assert engine.compression_count == 0
