"""Regression tests for empty-response recovery transcript persistence."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent


class _CapturingSessionDB:
    """Minimal SessionDB stand-in that records every appended message."""

    def __init__(self):
        self.rows = []

    def append_message(self, session_id, role, content=None, **kwargs):
        self.rows.append({"role": role, "content": content})
        return len(self.rows)

    def append_messages_batch(self, session_id, messages, **kwargs):
        # Mirror the real batch writer: same rows, one call.
        for m in messages:
            self.rows.append({"role": m.get("role"), "content": m.get("content")})
        return list(range(len(self.rows) - len(messages) + 1, len(self.rows) + 1))


def _agent_with_capturing_db():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._session_db = _CapturingSessionDB()
    agent._session_db_created = True
    agent._last_flushed_db_idx = 0
    agent.session_id = "sess-test"
    return agent


def _agent_with_stubbed_persistence():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._session_db = None
    agent._session_messages = []
    agent.flushed_session_db_messages = []
    agent._flush_messages_to_session_db = lambda messages, conversation_history=None: (
        agent.flushed_session_db_messages.append([m.copy() for m in messages])
    )
    return agent


def test_persist_session_strips_trailing_empty_recovery_scaffolding():
    """Only the flagged scaffolding goes. The assistant(tool_calls) + tool pair
    already ran and was saved before execution, so it stays. The persist layer does
    not author a closing row: the exit owner closes the tool tail with its reason.
    """
    agent = _agent_with_stubbed_persistence()
    messages = [
        {"role": "user", "content": "run the task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        {
            "role": "assistant",
            "content": "(empty)",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "user",
            "content": (
                "You just executed tool calls but returned an empty response. "
                "Please process the tool results above and continue with the task."
            ),
            "_empty_recovery_synthetic": True,
        },
    ]

    AIAgent._persist_session(agent, messages, conversation_history=[])

    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["id"] == messages[2]["tool_call_id"]
    assert agent.flushed_session_db_messages[-1] == messages
    assert all(not msg.get("_empty_recovery_synthetic") for msg in messages)


def test_persist_session_keeps_unmarked_terminal_empty_response():
    agent = _agent_with_stubbed_persistence()
    messages = [
        {"role": "user", "content": "run the task"},
        {"role": "assistant", "content": "(empty)"},
    ]

    AIAgent._persist_session(agent, messages, conversation_history=[])

    assert messages == [
        {"role": "user", "content": "run the task"},
        {"role": "assistant", "content": "(empty)"},
    ]
    assert agent.flushed_session_db_messages[-1] == messages




def test_flush_never_writes_buried_empty_recovery_scaffolding():
    """When an empty-after-tools nudge is followed by a tool-calling response,
    the synthetic ``(empty)`` + nudge pair stays buried in the live message
    list (only the trailing copies are ever dropped). The append-only flush
    must skip it regardless of position, otherwise the synthetic turns land in
    the session store and pollute every resumed transcript.
    """
    agent = _agent_with_capturing_db()

    messages = [
        {"role": "user", "content": "run the task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
        # Synthetic recovery scaffolding, now buried because the model answered
        # the nudge with another tool call rather than terminating.
        {"role": "assistant", "content": "(empty)", "_empty_recovery_synthetic": True},
        {
            "role": "user",
            "content": "You just executed tool calls but returned an empty response.",
            "_empty_recovery_synthetic": True,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_2", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "{}", "tool_call_id": "call_2"},
        {"role": "assistant", "content": "All done."},
    ]

    agent._flush_messages_to_session_db(messages, conversation_history=[])

    persisted = agent._session_db.rows
    assert all(row["content"] != "(empty)" for row in persisted)
    assert all("empty response" not in (row["content"] or "") for row in persisted)
    # Only the genuine turns reach the store, in order.
    assert [r["role"] for r in persisted] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert persisted[-1]["content"] == "All done."


def test_flush_skips_thinking_prefill_scaffolding():
    agent = _agent_with_capturing_db()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "_thinking_prefill": True},
        {"role": "assistant", "content": "Hello!"},
    ]
    agent._flush_messages_to_session_db(messages, conversation_history=[])

    assert [r["content"] for r in agent._session_db.rows] == ["hi", "Hello!"]


# ── Real turn loop: a tool that already ran survives an empty-response exit ──

_DEAD_LOCAL = "http://127.0.0.1:9"


def _response(content="", finish_reason="stop", tool_calls=None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason, index=0,
    )
    return SimpleNamespace(id="chatcmpl-test", choices=[choice], model="test/model", usage=None)


def _write_file_call(path):
    return SimpleNamespace(
        id="call_write", type="function",
        function=SimpleNamespace(name="write_file", arguments=json.dumps({"path": str(path), "content": "PAYMENT #1 SENT\n"})),
    )


@pytest.fixture
def real_loop(tmp_path, monkeypatch):
    """A real AIAgent on the real ``file`` toolset and a real SessionDB; only the provider is scripted.
    Any request that would still leave the process dies on a dead local address."""
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, _DEAD_LOCAL)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    monkeypatch.setattr("agent.title_generator.start_title_upgrade", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-empty-exit"
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(
            api_key="test-key", base_url=f"{_DEAD_LOCAL}/v1", model="test/model", quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=["file"], session_db=db, session_id=sid,
        )

    def _no_real_client(*_a, **_k):
        raise AssertionError("a real provider client would be built")

    agent._create_openai_client = _no_real_client
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False

    def run(script, user_message):
        pending = list(script)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = lambda **_kw: pending.pop(0)
        return agent.run_conversation(user_message)

    yield SimpleNamespace(agent=agent, db=db, sid=sid, ledger=tmp_path / "ledger.txt", run=run)
    db.close()


def _assert_saved_tool_pairs_stay_live(result, db, sid):
    def ids(rows):
        calls = {tc["id"] for m in rows if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])}
        return calls, {m.get("tool_call_id") for m in rows if m.get("role") == "tool"}

    saved = db.get_messages_as_conversation(sid)
    saved_calls, saved_results = ids(saved)
    live_calls, live_results = ids(result["messages"])
    assert saved_calls and saved_calls == saved_results
    # The next turn replays result["messages"]: a pair missing there is a side effect the model re-runs.
    assert saved_calls <= live_calls and saved_results <= live_results
    assert saved[-1]["role"] != "tool"


def test_empty_response_give_up_keeps_the_executed_tool_call_live(real_loop):
    result = real_loop.run(
        [_response(finish_reason="tool_calls", tool_calls=[_write_file_call(real_loop.ledger)])]
        + [_response() for _ in range(8)],
        "record the payment in ledger.txt",
    )

    assert real_loop.ledger.read_text() == "PAYMENT #1 SENT\n"
    assert result["turn_exit_reason"] == "empty_response_exhausted"
    _assert_saved_tool_pairs_stay_live(result, real_loop.db, real_loop.sid)


def test_stop_during_empty_response_recovery_keeps_the_executed_tool_call_live(real_loop, monkeypatch):
    agent = real_loop.agent

    def stop_during_backoff(*_a, **_k):
        agent.interrupt("user pressed stop")
        return 5.0

    monkeypatch.setattr("agent.retry_utils.jittered_backoff", stop_during_backoff)
    script = [_response(finish_reason="tool_calls", tool_calls=[_write_file_call(real_loop.ledger)]), _response(), _response()]
    result = real_loop.run(script, "record the payment in ledger.txt")

    assert real_loop.ledger.read_text() == "PAYMENT #1 SENT\n"
    assert result["interrupted"] is True
    _assert_saved_tool_pairs_stay_live(result, real_loop.db, real_loop.sid)
    # The Stop owner strips the nudge scaffold itself and closes with its own reason.
    assert result["messages"][-1]["content"] == result["final_response"]
