"""Verification-loop synthetic scaffolding must never reach durable session state.

verify_on_stop / pre_verify append a synthetic assistant "done" plus a synthetic
user nudge to keep the agent going one more turn before it can claim completion.
These messages exist only to drive the loop; persisting them poisons the resumed
transcript and breaks prompt-prefix cache reuse on later turns (#55733).

Both persistence sinks (SQLite flush + JSON snapshot) route through the single
``_is_ephemeral_scaffolding`` chokepoint, which is driven by
``_EPHEMERAL_SCAFFOLDING_FLAGS``. These tests assert that the verification-loop
flags are registered there and that both sinks drop the flagged messages while
keeping the real conversation.
"""

import json
import sys
from unittest.mock import MagicMock

import pytest


def _fresh_run_agent(hermes_home):
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    import run_agent  # noqa: F401
    return sys.modules["run_agent"]


def test_verification_flags_registered_as_ephemeral(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ra = _fresh_run_agent(tmp_path)

    assert "_verification_stop_synthetic" in ra._EPHEMERAL_SCAFFOLDING_FLAGS
    assert "_pre_verify_synthetic" in ra._EPHEMERAL_SCAFFOLDING_FLAGS

    # The central classifier drives both persistence sinks.
    assert ra._is_ephemeral_scaffolding(
        {"role": "assistant", "content": "done", "_verification_stop_synthetic": True}
    )
    assert ra._is_ephemeral_scaffolding(
        {"role": "user", "content": "[System: run tests]", "_pre_verify_synthetic": True}
    )
    # Real messages are not scaffolding.
    assert not ra._is_ephemeral_scaffolding({"role": "user", "content": "hi"})


def _make_agent(ra, session_id, tmp_path):
    agent = ra.AIAgent(
        session_id=session_id,
        api_key="test-key",
        base_url="http://127.0.0.1:8000/v1",
        provider="openai-compat",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path / "logs"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    return agent


def test_db_flush_drops_verification_scaffolding(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ra = _fresh_run_agent(tmp_path)
    agent = _make_agent(ra, "sess_db", tmp_path)

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "premature done", "_verification_stop_synthetic": True},
        {"role": "user", "content": "[System: run tests]", "_verification_stop_synthetic": True},
        {"role": "assistant", "content": "verified and clean"},
    ]

    agent._flush_messages_to_session_db(messages, conversation_history=[])

    persisted = [
        kwargs.get("content")
        for _args, kwargs in agent._session_db.append_message.call_args_list
    ]
    assert "hi" in persisted
    assert "verified and clean" in persisted
    assert "premature done" not in persisted
    assert "[System: run tests]" not in persisted


def test_json_log_drops_verification_scaffolding(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ra = _fresh_run_agent(tmp_path)
    agent = _make_agent(ra, "sess_json", tmp_path)

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "premature done", "_pre_verify_synthetic": True},
        {"role": "user", "content": "[System: run tests]", "_pre_verify_synthetic": True},
        {"role": "assistant", "content": "verified and clean"},
    ]

    agent._save_session_log(messages)

    log_file = agent.logs_dir / "session_sess_json.json"
    assert log_file.exists()
    data = json.loads(log_file.read_text(encoding="utf-8"))
    contents = [m.get("content") for m in data["messages"]]
    assert contents == ["hi", "verified and clean"]
    assert all(not m.get("_pre_verify_synthetic") for m in data["messages"])
