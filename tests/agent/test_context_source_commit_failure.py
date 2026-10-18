"""A retained attachment failure cannot undo a committed compression child."""

from types import SimpleNamespace
import time

import pytest

from tests.agent.test_run_agent import agent as _agent_fixture

agent = _agent_fixture


def test_source_changed_at_publication_preserves_committed_child_and_blocks_sdk(
    agent, tmp_path, monkeypatch,
):
    from agent import conversation_compression as compression
    from agent.source_provenance import SourceProvenanceError
    from hermes_state import SessionDB
    from tests.agent.test_context_source_lifecycle import _dispatch, _prepare, _sdk

    source, attached, turn = _prepare(agent, tmp_path)
    messages = [{"role": "user", "content": "original parent conversation " * 20}]
    compressed = [{"role": "user", "content": "compact summary"}]
    monkeypatch.setattr(compression, "_salvage_or_refuse_grown_transcript",
                        lambda _a, _m, result, **_k: (result, None))
    agent.commit_memory_session = lambda _messages: None
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("parent", source="tui")
        agent._session_db = db
        original_publish = db.publish_compression_child
        def committed_then_changed(**kwargs):
            result = original_publish(**kwargs)
            source.write_text("my_value = changed_value + 456\n")
            return result
        monkeypatch.setattr(db, "publish_compression_child", committed_then_changed)
        outcome = compression._commit_compaction(
            agent, messages, compressed, in_place=False,
            lease=SimpleNamespace(holder=None, ttl=10, watermark=None),
            new_system_prompt="", system_message="", compressed_user_turn_outcome="already_present",
            messages_before_compression=list(messages), made_progress=True,
            attempt=SimpleNamespace(started_at=time.monotonic(), snapshot={}),
        )
        child = db.get_compression_tip("parent")
        assert child != "parent"
        assert agent.session_id == child
        assert outcome.session_commit_succeeded and outcome.split_status == "rotated_committed"
        assert outcome.compressed == compressed
        captured = []
        with _sdk(agent, captured) as sdk:
            with pytest.raises(SourceProvenanceError, match="content_mismatch"):
                _dispatch(agent, sdk, attached, turn, 1)
        assert captured == []
