"""Regression for #121734: a continuing client must see every active row it did not summarize."""

from unittest.mock import patch
import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.conversation_compression import compress_context
from hermes_state import SessionDB


def _agent(db, session_id):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=session_id,
            skip_context_files=True, skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor.compress = lambda messages, **kwargs: [
        {"role": "user", "content": "[CONTEXT COMPACTION] held summary"},
        {"role": "assistant", "content": "reply"},
        *([{"role": "user", "content": messages[-1]["content"]}]
          if messages[-1].get("content") == "current unpersisted turn" else []),
    ]
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def test_continuing_client_gets_foreign_gap_and_durable_provenance(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    for i in range(8):
        db.append_message("s", role="user" if i % 2 == 0 else "assistant", content=f"held {i}")
    held = db.get_messages_as_conversation("s", repair_alternation=True, include_row_ids=True)
    db.append_message("s", role="user", content="foreign gap")
    db.append_message("s", role="assistant", content="foreign reply")
    db.append_message("s", role="user", content="held latest")
    held.extend(db.get_messages_as_conversation("s", include_row_ids=True)[-1:])
    held.append({"role": "user", "content": "current unpersisted turn"})
    agent = _agent(db, "s")
    live, _ = compress_context(agent, held, approx_tokens=100_000, system_message="sys")
    durable = db.get_messages_as_conversation("s", repair_alternation=True, include_row_ids=True)
    assert [(m["role"], m["content"]) for m in live] == [
        (m["role"], m["content"]) for m in durable
    ]
    assert any("foreign gap" in str(m["content"]) for m in live)
    assert any("current unpersisted turn" in str(m["content"]) for m in live)
    assert all(m.get(_DB_PERSISTED_MARKER) and (m.get("_row_id") or m.get("_source_row_ids")) for m in live)
    count = db.get_session("s")["message_count"]
    agent._persist_session(live, conversation_history=None)
    assert db.get_session("s")["message_count"] == count


def test_persisted_input_without_provenance_refuses_to_archive(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    for i in range(8):
        db.append_message("s", role="user" if i % 2 == 0 else "assistant", content=f"held {i}")
    held = db.get_messages_as_conversation("s", repair_alternation=True)
    db.append_message("s", role="user", content="foreign")
    agent = _agent(db, "s")
    live, _ = compress_context(agent, held, approx_tokens=100_000, system_message="sys")
    assert live == held
    assert "foreign" in [m["content"] for m in db.get_messages_as_conversation("s")]
    assert not any(m["compacted"] for m in db.get_messages("s", include_inactive=True))


def test_post_commit_read_failure_does_not_publish_partial_live_transcript(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    for i in range(8):
        db.append_message("s", role="user" if i % 2 == 0 else "assistant", content=f"held {i}")
    held = db.get_messages_as_conversation("s", repair_alternation=True, include_row_ids=True)
    db.append_message("s", role="user", content="foreign gap")
    db.append_message("s", role="assistant", content="foreign reply")
    held.append({"role": "user", "content": "current unpersisted turn"})
    agent = _agent(db, "s")
    original_read = db.get_messages_as_conversation
    with patch.object(db, "get_messages_as_conversation", side_effect=OSError("read failed")):
        with pytest.raises(RuntimeError, match="Committed compression requires a durable transcript reload"):
            compress_context(agent, held, approx_tokens=100_000, system_message="sys")

    durable = original_read("s", repair_alternation=True, include_row_ids=True)
    assert any("foreign gap" in str(m["content"]) for m in durable)
    assert any("current unpersisted turn" in str(m["content"]) for m in durable)
    assert agent._last_compaction_in_place is True
