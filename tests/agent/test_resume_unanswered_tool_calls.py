"""A turn killed mid-tool must stay visible to the resumed model.

The agent loop persists the assistant ``tool_calls`` row BEFORE the tool runs, so a SIGKILL,
crash or a tool that restarts Hermes leaves a call with no result row. Restore and the
pre-request alternation repair used to prune that call and merge the two user turns around
it: the resumed model never learned the action may have happened and could repeat it (#49201).
"""

import json

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from hermes_state import SessionDB

_CALL = {"id": "call_1", "type": "function",
         "function": {"name": "terminal", "arguments": json.dumps({"command": "systemctl restart app"})}}


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _wire(messages):
    return [(m["role"], m.get("content"), m.get("tool_call_id")) for m in messages]


@pytest.mark.parametrize("resumed_turns", [0, 1], ids=["killed-turn-is-tail", "after-a-resumed-turn"])
def test_killed_side_effect_call_survives_restore(db, resumed_turns):
    db.create_session("s1", "system prompt")
    db.append_message(session_id="s1", role="user", content="restart the app")
    db.append_message(session_id="s1", role="assistant", content="", tool_calls=[_CALL])
    for n in range(resumed_turns):
        db.append_message(session_id="s1", role="user", content=f"later ask {n}")
        db.append_message(session_id="s1", role="assistant", content=f"later reply {n}")

    restored = db.get_messages_as_conversation("s1", repair_alternation=True)

    assert [m["role"] for m in restored[:3]] == ["user", "assistant", "tool"]
    assert restored[0]["content"] == "restart the app"
    assert restored[1]["tool_calls"][0]["id"] == "call_1"
    assert restored[2]["tool_call_id"] == "call_1"
    assert restored[2]["effect_disposition"] == "unknown"
    # The pre-request pass sees the same shape and leaves it byte-identical (cache-stable), and the
    # recovered result is a projection: nothing new was written to the durable transcript.
    live = list(restored) + [{"role": "user", "content": "next"}]
    assert repair_message_sequence(None, live) == 0
    assert _wire(live[:-1]) == _wire(db.get_messages_as_conversation("s1", repair_alternation=True))
    assert len(db.get_messages_as_conversation("s1")) == 2 + 2 * resumed_turns


def test_gateway_resume_replays_killed_call_as_unknown(tmp_path, monkeypatch):
    import hermes_state
    from gateway.config import GatewayConfig
    from gateway.run import _build_gateway_agent_history
    from gateway.session import SessionStore

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db.create_session(session_id="g1", source="test")
    store.append_to_transcript("g1", {"role": "user", "content": "restart the app", "timestamp": 1.0})
    store.append_to_transcript("g1", {"role": "assistant", "content": "", "tool_calls": [_CALL], "timestamp": 2.0})

    agent_history, _ = _build_gateway_agent_history(store.load_transcript("g1"))

    assert [m["role"] for m in agent_history] == ["user", "assistant", "tool"]
    assert agent_history[-1]["tool_call_id"] == "call_1"
    assert agent_history[-1]["effect_disposition"] == "unknown"
