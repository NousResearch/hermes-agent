"""In-place compaction must store the current turn's user row in its persisted form.

The live message dict holds the agent-only API form of the turn (for an image turn, the
``[The user attached an image ...]`` hint). The turn-start flush writes the clean
``persist_user_message`` override instead. A compaction that runs mid-turn re-inserts the
turn's user row from the live dict, so the override must be applied there too, or the API form
becomes the only active row and the clean form is archived.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB

API_FORM = (
    "[The user attached an image: upload_x.jpg]\n"
    "[Examine it with the vision_analyze tool using image_url: /tmp/upload_x.jpg]\n\n"
    "Again im unable to download."
)
CLEAN_FORM = "Again im unable to download.\n@image:/tmp/upload_x.jpg"


def _agent(db, sid, compacted):
    from run_agent import AIAgent

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            platform="cli", quiet_mode=True, session_db=db, session_id=sid,
            skip_context_files=True, skip_memory=True,
        )
    agent._cached_system_prompt = "SYSTEM"
    agent.compression_enabled = True
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    compressor = MagicMock()
    compressor.protect_first_n = 0
    compressor.protect_last_n = 0
    compressor.threshold_tokens = 1
    compressor.context_length = 1000
    compressor.last_prompt_tokens = -1
    compressor.compress = lambda _messages, **_kw: compacted
    compressor.compression_count = 1
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


def test_midturn_in_place_compaction_keeps_persisted_user_form(tmp_path):
    from agent.conversation_compression import compress_context

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-midturn-image"
    try:
        live_user = {"role": "user", "content": API_FORM}
        messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            live_user,
        ]
        compacted = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
            dict(live_user),
        ]
        agent = _agent(db, sid, compacted)
        db.create_session(session_id=sid, source="cli", model="test/model")
        agent._session_db_created = True
        agent._persist_user_message_idx = 2
        agent._persist_user_message_override = CLEAN_FORM
        agent._persist_user_message_timestamp = None
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            compress_context(agent, messages, "SYSTEM")
        assert agent._last_compaction_in_place is True
        users = [m["content"] for m in db.get_messages(sid) if m["role"] == "user"]
        assert CLEAN_FORM in users
        assert API_FORM not in users
        # The wire form is kept as the sidecar, as the flush does, and the live row is untouched.
        row = [m for m in db.get_messages(sid) if m["role"] == "user"][-1]
        assert row["api_content"] == API_FORM
        assert compacted[1]["content"] == API_FORM
        assert "api_content" not in compacted[1]
    finally:
        db.close()


def test_midturn_in_place_compaction_without_override_is_unchanged(tmp_path):
    from agent.conversation_compression import compress_context

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-midturn-plain"
    try:
        live_user = {"role": "user", "content": "plain question"}
        messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            live_user,
        ]
        compacted = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
            dict(live_user),
        ]
        agent = _agent(db, sid, compacted)
        db.create_session(session_id=sid, source="cli", model="test/model")
        agent._session_db_created = True
        agent._persist_user_message_idx = 2
        agent._persist_user_message_override = None
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            compress_context(agent, messages, "SYSTEM")
        users = [m["content"] for m in db.get_messages(sid) if m["role"] == "user"]
        assert users == ["plain question"]
    finally:
        db.close()


def test_midturn_in_place_compaction_does_not_rewrite_summary_merged_turn(tmp_path):
    """A turn whose user row was merged into the summary no longer matches the live dict, so it is left alone."""
    from agent.conversation_compression import compress_context

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-midturn-merged"
    try:
        live_user = {"role": "user", "content": API_FORM}
        messages = [
            {"role": "user", "content": "earlier question " * 200},
            {"role": "assistant", "content": "earlier answer " * 200},
            live_user,
        ]
        merged = "[CONTEXT COMPACTION] summary\n\n" + API_FORM
        compacted = [{"role": "user", "content": merged}]
        agent = _agent(db, sid, compacted)
        db.create_session(session_id=sid, source="cli", model="test/model")
        agent._session_db_created = True
        agent._persist_user_message_idx = 2
        agent._persist_user_message_override = CLEAN_FORM
        agent._persist_user_message_timestamp = None
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            compress_context(agent, messages, "SYSTEM")
        users = [m["content"] for m in db.get_messages(sid) if m["role"] == "user"]
        assert users == [merged]
    finally:
        db.close()


def test_midturn_in_place_compaction_keeps_turn_user_timestamp(tmp_path):
    """The re-inserted row keeps the timestamp the turn-start flush gave the persisted row."""
    from agent.conversation_compression import compress_context

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-midturn-ts"
    try:
        db.create_session(session_id=sid, source="cli", model="test/model")
        db.append_message(sid, "user", CLEAN_FORM, timestamp=1234.5)
        live_user = {"role": "user", "content": API_FORM}
        messages = [{"role": "assistant", "content": "earlier answer"}, live_user]
        compacted = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
            dict(live_user),
        ]
        agent = _agent(db, sid, compacted)
        agent._session_db_created = True
        agent._persist_user_message_idx = 1
        agent._persist_user_message_override = CLEAN_FORM
        agent._persist_user_message_timestamp = 1234.5
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            compress_context(agent, messages, "SYSTEM")
        active = [m for m in db.get_messages(sid) if m["role"] == "user" and m["content"] == CLEAN_FORM]
        assert active and active[-1]["timestamp"] == 1234.5
    finally:
        db.close()
