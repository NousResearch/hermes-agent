from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_dag_engine import DAGContextEngine
from agent.context_dag_status import dag_context_status_lines
from agent.context_dag_store import ContextDAGStore
from cli import HermesCLI
from hermes_state import SessionDB


DOCS = [
    Path("website/docs/developer-guide/context-compression-and-caching.md"),
    Path("website/docs/developer-guide/context-engine-plugin.md"),
]


def make_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-1", "test", model="test-model")
    return db


def test_default_legacy_config_has_no_dag_status_noise(tmp_path):
    db = make_db(tmp_path)
    try:
        lines = dag_context_status_lines(
            config={"context": {"engine": "compressor"}},
            session_id="sess-1",
            session_db=db,
        )
        assert lines == []
    finally:
        db.close()


def test_enabled_dag_status_reports_beta_safety_projection_checkpoint_queue_and_sidecar(tmp_path):
    db = make_db(tmp_path)
    try:
        msg1 = db.append_message("sess-1", "user", "hello")
        msg2 = db.append_message("sess-1", "tool", "x" * 20000, tool_call_id="call-1", tool_name="terminal")
        store = ContextDAGStore(db)
        store.write_active_projection(
            session_id="sess-1",
            engine_version="dag-v1",
            projection=[{"role": "user", "content": "hello"}],
            fresh_tail_start_message_id=msg1,
            latest_raw_message_id=msg2,
            token_estimate=42,
            status="active",
        )
        store.write_checkpoint(
            session_id="sess-1",
            last_ingested_message_id=msg2,
            last_projection_message_id=msg1,
            last_anchor_message_id=msg1,
            anchor_hash="abc",
        )
        store.append_mutation_log(
            session_id="sess-1",
            operation="reconcile_transcript",
            status="pending",
            idempotency_key="k1",
        )
        with db._lock:
            db._conn.execute(
                """
                INSERT INTO context_message_parts (
                    message_id, part_index, part_type, content_inline, content_ref,
                    sha256, size_bytes, token_estimate, metadata_json, created_at
                ) VALUES (?, 0, 'tool_output', 'full output', 'sidecar://message/2/part/0', 'abc', 11, 3, '{}', 1.0)
                """,
                (msg2,),
            )
            db._conn.commit()

        engine = DAGContextEngine(session_db=db, gateway_enabled=True, mutation_queue_enabled=True)
        engine.on_session_start("sess-1")
        lines = dag_context_status_lines(
            config={"context": {"engine": "dag", "dag": {"gateway_enabled": True, "mutation_queue_enabled": True}}},
            session_id="sess-1",
            session_db=db,
            engine=engine,
        )
        rendered = "\n".join(lines)
        assert "BETA" in rendered
        assert "explicit opt-in" in rendered
        assert "projection-only/no transcript rewrite" in rendered
        assert "gateway_enabled=on" in rendered
        assert "mutation_queue_enabled=on" in rendered
        assert "token_estimate=42" in rendered
        assert f"fresh_tail_start_message_id={msg1}" in rendered
        assert f"last_ingested_message_id={msg2}" in rendered
        assert "counts=pending=1" in rendered
        assert "stored_parts=1" in rendered
        assert "defaults are safe/off" in rendered
    finally:
        db.close()


def test_cli_status_default_legacy_has_no_dag_noise():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {"context": {"engine": "compressor"}}
    cli_obj.console = SimpleNamespace(print=lambda *args, **kwargs: None)
    cli_obj.agent = SimpleNamespace(session_total_tokens=0, context_compressor=SimpleNamespace(name="compressor"))
    cli_obj.session_id = "session-123"
    cli_obj.session_start = __import__("datetime").datetime(2026, 1, 1, 12, 0)
    cli_obj._agent_running = False
    cli_obj.model = "test-model"
    cli_obj.provider = "test-provider"
    cli_obj._session_db = SimpleNamespace(get_session=lambda _sid: None)
    printed = []
    cli_obj.console.print = lambda text, **kwargs: printed.append(text)

    with patch("cli.display_hermes_home", return_value="~/.hermes"):
        cli_obj._show_session_status()

    rendered = "\n".join(printed)
    assert "Hermes CLI Status" in rendered
    assert "DAG context" not in rendered


def test_docs_config_snippets_warn_beta_opt_in_and_flags():
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    assert "context:\n  engine: dag" in text
    assert "gateway_enabled" in text
    assert "mutation_queue_enabled" in text
    assert "HERMES_DAG_CONTEXT_GATEWAY_ENABLED" in text
    assert "HERMES_DAG_CONTEXT_MUTATION_QUEUE_ENABLED" in text
    assert "beta" in text.lower()
    assert "opt-in" in text.lower()
    assert "engine: compressor" in text
    assert "projection-only" in text
    assert "context_expand" in text
    assert "sidecar" in text
