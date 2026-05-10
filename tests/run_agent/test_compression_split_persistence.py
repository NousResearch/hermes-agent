from pathlib import Path
from unittest.mock import MagicMock

from hermes_state import SessionDB
from run_agent import AIAgent


def _make_agent(db: SessionDB, tmp_path: Path) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "parent-session"
    agent.model = "test/model"
    agent.platform = "cli"
    agent.tools = []
    agent.logs_dir = tmp_path
    agent.session_log_file = tmp_path / f"session_{agent.session_id}.json"
    agent._session_db = db
    agent._session_db_created = True
    agent._session_init_model_config = {"model": "test/model"}
    agent._last_flushed_db_idx = 0
    agent._memory_manager = None
    agent._cached_system_prompt = "old-system"
    agent._last_compression_summary_warning = None
    agent._last_aux_fallback_warning_key = None
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent.context_compressor = MagicMock()
    agent.context_compressor.compression_count = 1
    agent.context_compressor.last_prompt_tokens = 0
    agent.context_compressor.last_completion_tokens = 0
    agent.context_compressor._last_summary_error = None
    agent.context_compressor._last_aux_model_failure_model = None
    agent.context_compressor._last_aux_model_failure_error = None
    agent._save_session_log = MagicMock()
    agent._emit_warning = MagicMock()
    agent._vprint = MagicMock()
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="new-system")
    return agent


def _source_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Initial requirements"},
        {
            "role": "assistant",
            "content": "I will inspect the repo.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "tool_name": "shell",
            "content": "existing test output",
        },
        {"role": "assistant", "content": "Found the compression split."},
    ]


def _compressed_messages() -> list[dict]:
    return [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "Ready to continue from the summary."},
        {"role": "user", "content": "recent follow-up"},
    ]


def test_compression_split_persists_parent_transcript_before_ending(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent-session", source="cli", model="test/model")
    agent = _make_agent(db, tmp_path)
    agent.context_compressor.compress.return_value = _compressed_messages()

    source = _source_messages()
    agent._compress_context(source, "system prompt", approx_tokens=10_000)

    parent = db.get_session("parent-session")
    assert parent["end_reason"] == "compression"

    rows = db.get_messages("parent-session")
    assert [row["role"] for row in rows] == [msg["role"] for msg in source]
    assert [row["content"] for row in rows] == [msg["content"] for msg in source]
    assert rows[1]["tool_calls"] == source[1]["tool_calls"]
    assert rows[2]["tool_call_id"] == "call-1"


def test_compression_split_persists_child_compressed_transcript_immediately(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent-session", source="cli", model="test/model")
    agent = _make_agent(db, tmp_path)
    compressed = _compressed_messages()
    agent.context_compressor.compress.return_value = compressed

    agent._compress_context(_source_messages(), "system prompt", approx_tokens=10_000)

    child_session_id = agent.session_id
    assert child_session_id != "parent-session"
    child = db.get_session(child_session_id)
    assert child is not None
    assert child["parent_session_id"] == "parent-session"

    rows = db.get_messages(child_session_id)
    assert [row["role"] for row in rows] == [msg["role"] for msg in compressed]
    assert [row["content"] for row in rows] == [msg["content"] for msg in compressed]


def test_compression_split_does_not_duplicate_already_flushed_parent_messages(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("parent-session", source="cli", model="test/model")
    agent = _make_agent(db, tmp_path)
    agent.context_compressor.compress.return_value = _compressed_messages()
    source = _source_messages()

    agent._persist_session(source)
    assert len(db.get_messages("parent-session")) == len(source)

    agent._compress_context(source, "system prompt", approx_tokens=10_000)

    rows = db.get_messages("parent-session")
    assert len(rows) == len(source)
    assert [row["content"] for row in rows] == [msg["content"] for msg in source]
