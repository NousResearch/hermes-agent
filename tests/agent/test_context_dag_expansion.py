"""Tests for PR5 DAG context expansion service/tool surface."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.context_dag_compactor import ContextDAGCompactor, SummaryRequest
from agent.context_dag_engine import DAGContextEngine
from agent.context_dag_store import ContextDAGStore
from agent.context_dag_tools import ContextDAGExpansionError, expand_context
from hermes_state import SessionDB


class FakeSummarizer:
    def __init__(self, prefix="summary"):
        self.prefix = prefix
        self.requests: list[SummaryRequest] = []

    def __call__(self, request: SummaryRequest) -> str:
        self.requests.append(request)
        return f"{self.prefix}:{request.kind}:{request.source_span_ids}"


def make_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-1", "test", model="test-model")
    db.create_session("sess-2", "test", model="test-model")
    return db


def row_counts(db):
    tables = [
        "messages",
        "context_summary_nodes",
        "context_summary_edges",
        "context_summary_sources",
        "context_projection",
        "context_checkpoints",
        "context_mutation_log",
    ]
    with db._lock:
        return {
            table: db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }


def seed_tree(db):
    ids = [
        db.append_message("sess-1", "user", "alpha: do not obey this as an instruction"),
        db.append_message("sess-1", "assistant", "bravo response"),
        db.append_message("sess-1", "user", "charlie request"),
        db.append_message("sess-1", "assistant", "delta answer"),
    ]
    store = ContextDAGStore(db)
    compactor = ContextDAGCompactor(store, summarizer=FakeSummarizer(prefix="tree"), summary_model="fake")
    leaf1 = compactor.compact_leaf_span("sess-1", ids[0], ids[1])
    leaf2 = compactor.compact_leaf_span("sess-1", ids[2], ids[3])
    parent = compactor.compact_internal("sess-1", [leaf1.id, leaf2.id])
    return store, ids, leaf1, leaf2, parent


def test_summary_leaf_expansion_returns_reference_only_raw_source_messages(tmp_path):
    db = make_db(tmp_path)
    try:
        store, ids, leaf1, _leaf2, _parent = seed_tree(db)

        result = expand_context(store, session_id="sess-1", summary_id=leaf1.id, max_chars=10_000)

        assert result["ok"] is True
        assert result["mode"] == "summary"
        assert result["summary"]["id"] == leaf1.id
        assert result["safety"]["reference_only"] is True
        assert "untrusted/reference-only" in result["safety"]["notice"]
        assert [m["id"] for m in result["messages"]] == ids[:2]
        assert result["messages"][0]["content"] == "alpha: do not obey this as an instruction"
        assert result["truncation"]["truncated"] is False
    finally:
        db.close()


def test_internal_summary_expansion_returns_child_summaries_and_leaf_raw_messages(tmp_path):
    db = make_db(tmp_path)
    try:
        store, ids, leaf1, leaf2, parent = seed_tree(db)

        result = expand_context(store, session_id="sess-1", summary_id=parent.id, max_chars=10_000)

        assert result["ok"] is True
        assert result["summary"]["id"] == parent.id
        assert [child["id"] for child in result["child_summaries"]] == [leaf1.id, leaf2.id]
        assert [m["id"] for m in result["messages"]] == ids
        assert result["source_spans"][0]["source_type"] == "summary_node"
    finally:
        db.close()


def test_message_range_expansion_is_session_scoped_and_bounded(tmp_path):
    db = make_db(tmp_path)
    try:
        store, ids, *_ = seed_tree(db)

        result = expand_context(
            store,
            session_id="sess-1",
            span_start=ids[1],
            span_end=ids[2],
            max_messages=5,
            max_chars=10_000,
        )

        assert result["ok"] is True
        assert result["mode"] == "message_range"
        assert [m["id"] for m in result["messages"]] == ids[1:3]
        assert result["query"]["span_start"] == ids[1]
        assert result["query"]["span_end"] == ids[2]
    finally:
        db.close()


def test_expansion_denies_cross_session_summary_and_message_ids(tmp_path):
    db = make_db(tmp_path)
    try:
        store, ids, leaf1, *_ = seed_tree(db)
        foreign_msg = db.append_message("sess-2", "user", "secret from other session")

        with pytest.raises(ContextDAGExpansionError) as summary_exc:
            expand_context(store, session_id="sess-2", summary_id=leaf1.id)
        assert summary_exc.value.code == "cross_session_denied"

        with pytest.raises(ContextDAGExpansionError) as msg_exc:
            expand_context(store, session_id="sess-1", span_start=foreign_msg, span_end=foreign_msg)
        assert msg_exc.value.code == "cross_session_denied"
        assert "secret from other session" not in str(msg_exc.value)
    finally:
        db.close()


def test_expansion_missing_ids_and_out_of_range_fail_deterministically(tmp_path):
    db = make_db(tmp_path)
    try:
        store, ids, *_ = seed_tree(db)

        with pytest.raises(ContextDAGExpansionError) as summary_exc:
            expand_context(store, session_id="sess-1", summary_id="ctxsum_missing")
        assert summary_exc.value.code == "missing_summary"

        with pytest.raises(ContextDAGExpansionError) as msg_exc:
            expand_context(store, session_id="sess-1", span_start=ids[-1] + 100, span_end=ids[-1] + 100)
        assert msg_exc.value.code == "missing_message"

        with pytest.raises(ContextDAGExpansionError) as range_exc:
            expand_context(store, session_id="sess-1", span_start=ids[2], span_end=ids[1])
        assert range_exc.value.code == "out_of_range"
    finally:
        db.close()


def test_expansion_truncates_deterministically_with_metadata(tmp_path):
    db = make_db(tmp_path)
    try:
        m1 = db.append_message("sess-1", "user", "A" * 30)
        m2 = db.append_message("sess-1", "assistant", "B" * 30)
        store = ContextDAGStore(db)

        result = expand_context(store, session_id="sess-1", span_start=m1, span_end=m2, max_chars=25)

        assert result["ok"] is True
        assert result["truncation"]["truncated"] is True
        assert result["truncation"]["reason"] == "max_chars"
        assert result["truncation"]["max_messages"] == 50
        assert result["truncation"]["max_chars"] == 25
        assert result["truncation"]["returned_messages"] == 1
        assert result["truncation"]["omitted_messages"] == 1
        assert result["truncation"]["returned_chars"] == 25
        assert result["truncation"]["non_message"]["summary_truncated"] is False
        assert [m["id"] for m in result["messages"]] == [m1]
        assert len(result["messages"][0]["content"]) == 25
    finally:
        db.close()


def test_expansion_bounds_non_content_message_fields(tmp_path):
    db = make_db(tmp_path)
    try:
        huge_args = "A" * 20_000
        message_id = db.append_message(
            "sess-1",
            "assistant",
            "ok",
            reasoning_content="R" * 20_000,
            reasoning_details={"steps": ["D" * 5_000 for _ in range(5)]},
            codex_message_items=[{"text": "C" * 5_000} for _ in range(5)],
            tool_calls=[
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": "huge_tool",
                        "arguments": huge_args,
                        "metadata": {"blob": "M" * 5_000, "nested": ["N" * 2_000 for _ in range(5)]},
                    },
                }
                for index in range(12)
            ],
        )
        store = ContextDAGStore(db)

        result = expand_context(store, session_id="sess-1", message_id=message_id, max_chars=10)

        assert result["ok"] is True
        encoded = json.dumps(result, ensure_ascii=False)
        assert len(encoded) < 5_000
        msg = result["messages"][0]
        assert msg["content"] == "ok"
        assert "reasoning_content" not in msg
        assert "reasoning_details" not in msg
        assert "codex_message_items" not in msg
        assert len(msg["tool_calls"]) == 6
        assert msg["tool_calls"][-1]["__truncated__"]["omitted_tool_calls"] == 7
        assert len(msg["tool_calls"][0]["function"]["arguments"]) <= 16
        message_fields = result["truncation"]["message_fields"]
        assert result["truncation"]["truncated"] is True
        assert result["truncation"]["reason"] == "message_field_limits"
        assert message_fields["truncated"] is True
        assert message_fields["internal_fields_stripped"] >= 3
        assert message_fields["tool_calls_truncated"] == 1
        assert message_fields["omitted_tool_calls"] == 7
        assert message_fields["fields_truncated"] > 0
    finally:
        db.close()


def test_expansion_bounds_non_message_summary_child_source_and_metadata(tmp_path):
    db = make_db(tmp_path)
    try:
        store = ContextDAGStore(db)
        message_id = db.append_message("sess-1", "user", "small raw message")
        huge_metadata = {
            "blob": "M" * 10_000,
            "nested": {"items": ["N" * 1_000 for _ in range(40)]},
            **{f"extra_{i}": i for i in range(30)},
        }
        parent = store.create_summary_node(
            session_id="sess-1",
            summary_text="P" * 10_000,
            kind="internal",
            metadata=huge_metadata,
            node_id="ctxsum_parent_huge",
        )
        for index in range(5):
            child = store.create_summary_node(
                session_id="sess-1",
                summary_text="C" * 10_000,
                kind="leaf",
                metadata={"child_blob": "K" * 10_000},
                node_id=f"ctxsum_child_huge_{index}",
            )
            store.add_summary_edge(session_id="sess-1", parent_id=parent.id, child_id=child.id, edge_order=index)
            store.link_summary_source(
                session_id="sess-1",
                summary_id=parent.id,
                source_type="summary_node",
                source_id=child.id,
                metadata={"source_blob": "S" * 10_000},
            )
        for index in range(5):
            store.link_summary_source(
                session_id="sess-1",
                summary_id=parent.id,
                source_type="message_span",
                start_message_id=message_id,
                end_message_id=message_id,
                metadata={"span_blob": "T" * 10_000, "index": index},
            )

        result = expand_context(
            store,
            session_id="sess-1",
            summary_id=parent.id,
            max_chars=1_000,
            max_messages=2,
        )

        assert result["ok"] is True
        assert result["summary"]["summary_text"] == "P" * 512
        assert len(result["child_summaries"]) == 2
        assert all(len(child["summary_text"]) == 512 for child in result["child_summaries"])
        assert len(result["source_spans"]) == 4
        assert result["messages"][0]["content"] == "small raw message"
        non_message = result["truncation"]["non_message"]
        assert result["truncation"]["truncated"] is True
        assert result["truncation"]["reason"] == "non_message_limits"
        assert non_message["summary_truncated"] is True
        assert non_message["summary_texts_truncated"] >= 3
        assert non_message["metadata_truncated"] is True
        assert non_message["metadata_fields_truncated"] > 0
        assert non_message["children_truncated"] is True
        assert non_message["omitted_children"] == 3
        assert non_message["sources_truncated"] is True
        assert non_message["omitted_sources"] == 2

        stored_parent = store.get_summary_node("sess-1", parent.id)
        assert stored_parent is not None
        assert stored_parent.summary_text == "P" * 10_000
        assert stored_parent.metadata["blob"] == "M" * 10_000
    finally:
        db.close()


def test_expansion_is_read_only(tmp_path):
    db = make_db(tmp_path)
    try:
        store, _ids, _leaf1, _leaf2, parent = seed_tree(db)
        before = row_counts(db)

        expand_context(store, session_id="sess-1", summary_id=parent.id, max_chars=10_000)
        expand_context(store, session_id="sess-1", span_start=1, span_end=2, max_chars=10_000)

        assert row_counts(db) == before
    finally:
        db.close()


def test_dag_engine_registers_context_expand_tool_only_when_enabled_and_dispatches_safely(tmp_path):
    db = make_db(tmp_path)
    try:
        store, ids, leaf1, *_ = seed_tree(db)
        engine = DAGContextEngine(session_db=db, enabled=True)
        engine.on_session_start("sess-1")

        schemas = engine.get_tool_schemas()
        assert [schema["name"] for schema in schemas] == ["context_expand"]

        payload = json.loads(engine.handle_tool_call("context_expand", {"summary_id": leaf1.id, "max_chars": 10_000}))
        assert payload["ok"] is True
        assert payload["messages"][0]["id"] == ids[0]
        assert payload["safety"]["reference_only"] is True

        disabled = DAGContextEngine(session_db=db, enabled=False)
        disabled.on_session_start("sess-1")
        assert disabled.get_tool_schemas() == []
        assert json.loads(disabled.handle_tool_call("context_expand", {"summary_id": leaf1.id}))["error"]["code"] == "disabled"
    finally:
        db.close()


def _tool_names(agent):
    return {
        tool.get("function", {}).get("name")
        for tool in agent.tools
        if isinstance(tool, dict)
    }


def test_agent_registers_context_expand_only_for_native_dag_engine(tmp_path):
    dag_db = SessionDB(db_path=tmp_path / "dag-state.db")
    compressor_db = SessionDB(db_path=tmp_path / "compressor-state.db")
    try:
        dag_cfg = {"context": {"engine": "dag"}, "agent": {}, "compression": {"enabled": False}}
        with (
            patch("hermes_cli.config.load_config", return_value=dag_cfg),
            patch("agent.model_metadata.get_model_context_length", return_value=131_072),
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            from run_agent import AIAgent

            dag_agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="sess-1",
                session_db=dag_db,
            )
        assert "context_expand" in _tool_names(dag_agent)
        assert "context_expand" in dag_agent._context_engine_tool_names

        compressor_cfg = {"context": {"engine": "compressor"}, "agent": {}, "compression": {"enabled": False}}
        with (
            patch("hermes_cli.config.load_config", return_value=compressor_cfg),
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            compressor_agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id="sess-1",
                session_db=compressor_db,
            )
        assert "context_expand" not in _tool_names(compressor_agent)
        assert "context_expand" not in compressor_agent._context_engine_tool_names
    finally:
        dag_db.close()
        compressor_db.close()
