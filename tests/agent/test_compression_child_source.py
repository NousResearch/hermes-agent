"""A compression child is the same conversation as its parent: it keeps the parent's persisted source
(``--source tool``, ``oneshot``, an inherited ``kanban``) instead of degrading to ``agent.platform`` (#112550)."""

import types

import pytest

from agent import conversation_compression as cc
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _agent(db, session_id):
    return types.SimpleNamespace(
        _session_db=db, session_id=session_id, platform="cli", model="m", _session_init_model_config=None,
        working_directory=None, _memory_manager=None, context_compressor=types.SimpleNamespace(),
        _flush_messages_to_session_db=lambda *a, **k: None, _persist_user_message_idx=None,
        _session_messages=None, _gateway_session_key=None, _cached_system_prompt="sys",
        _session_row_model_config=lambda: None,
    )


@pytest.mark.parametrize("parent_source", ["tool", "oneshot", "kanban"])
def test_compression_child_keeps_parent_source(db, parent_source):
    db.create_session("parent", source=parent_source)
    db.append_message("parent", "user", "hello")
    db.append_message("parent", "assistant", "world")
    agent = _agent(db, "parent")

    cc._publish_rotated_compaction(
        agent, [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}],
        [{"role": "user", "content": "[handoff]"}], new_system_prompt="sys",
        lease=types.SimpleNamespace(holder=None, ttl=60.0, watermark=None),
        old_session_id="parent", compressed_user_turn_outcome="none",
    )

    assert agent.session_id != "parent"
    assert db.get_session(agent.session_id)["source"] == parent_source
