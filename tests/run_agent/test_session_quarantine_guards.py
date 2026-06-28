"""Session-integrity guards for malformed/interrupted assistant output.

These contracts pin the failure mode that can poison a live session: provider
transport/interruption glitches may produce assistant text that should remain in
forensics, but must not be replayed to the next model call.
"""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from run_agent import AIAgent
from agent.conversation_compression import compress_context
from agent.message_quarantine import live_context_messages


def _make_agent_with_db(tmp_path):
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="test")
    agent._session_db = db
    setattr(agent, "session_id", "s1")
    agent._session_db_created = True
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = "s1"
    agent._last_flushed_db_idx = 0
    agent.client = MagicMock()
    return agent, db


def test_inactive_quarantined_message_is_preserved_but_not_live_context(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="test")
    db.append_message("s1", role="user", content="safe user")
    db.append_message(
        "s1",
        role="assistant",
        content="garbled partial assistant text",
        finish_reason="interrupted_during_api_call",
        active=False,
        compacted=False,
    )

    live = db.get_messages("s1")
    all_rows = db.get_messages("s1", include_inactive=True)

    assert [m["content"] for m in live] == ["safe user"]
    assert [m["content"] for m in all_rows] == [
        "safe user",
        "garbled partial assistant text",
    ]
    assert all_rows[1]["active"] == 0
    assert all_rows[1]["compacted"] == 0
    assert all_rows[1]["finish_reason"] == "interrupted_during_api_call"


def test_agent_flush_persists_quarantined_partial_as_inactive(tmp_path):
    agent, db = _make_agent_with_db(tmp_path)
    messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": "partial bytes from interrupted provider response",
            "finish_reason": "interrupted_during_api_call",
            "active": False,
            "compacted": False,
            "_quarantine_reason": "interrupted_during_api_call",
        },
    ]

    agent._flush_messages_to_session_db(messages, conversation_history=[])

    live = db.get_messages("s1")
    all_rows = db.get_messages("s1", include_inactive=True)

    assert [m["content"] for m in live] == ["continue"]
    assert [m["content"] for m in all_rows] == [
        "continue",
        "partial bytes from interrupted provider response",
    ]
    assert all_rows[1]["active"] == 0
    assert all_rows[1]["finish_reason"] == "interrupted_during_api_call"


def test_live_context_filter_drops_inactive_forensic_rows():
    history = [
        {"role": "user", "content": "safe"},
        {"role": "assistant", "content": "poison", "active": False},
        {"role": "assistant", "content": "also poison", "active": 0},
        {"role": "assistant", "content": "safe reply", "active": True},
    ]

    assert [m["content"] for m in live_context_messages(history)] == [
        "safe",
        "safe reply",
    ]


def test_compression_entry_excludes_quarantined_messages():
    captured = {}

    class _Compressor:
        _last_compress_aborted = True
        _last_summary_error = "test abort"

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
            captured["messages"] = list(messages)
            return list(messages)

    class _Agent:
        _compression_feasibility_checked = True
        _session_db = None
        _memory_manager = None
        _cached_system_prompt = "system"
        _last_compression_summary_warning = None
        context_compressor = _Compressor()
        session_id = "s1"
        model = "test/model"

        def _emit_status(self, _msg):
            pass

        def _emit_warning(self, _msg):
            pass

        def _build_system_prompt(self, _system_message):
            return "system"

    messages = [
        {"role": "user", "content": "safe"},
        {"role": "assistant", "content": "poison", "active": False},
        {"role": "assistant", "content": "safe reply"},
    ]

    returned, _ = compress_context(_Agent(), messages, "system", approx_tokens=123)

    assert [m["content"] for m in captured["messages"]] == ["safe", "safe reply"]
    assert [m["content"] for m in returned] == ["safe", "safe reply"]
