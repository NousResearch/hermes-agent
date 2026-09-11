"""Acceptance-gate regressions for tool_call/tool_result pairing.

These drive the REAL persistence path (a real ``SessionDB`` on a temp
``HERMES_HOME``), a real ``run_conversation`` turn, a real reload from SQLite,
and the real outbound sanitizer used by ``turn_request_assembly``.

Nothing under test is mocked: only the provider client and the tool dispatch
surface are faked, plus a deliberate one-shot failure injected into the session
flush (the fault being reproduced).
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import sanitize_api_messages
from agent.message_sanitization import coalesce_tool_call_id
from agent.tool_executor import execute_tool_calls_segmented
from hermes_state import SessionDB
from run_agent import AIAgent


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(hermes_home: Path):
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _attach_real_session_db(agent, db_path: Path, session_id: str) -> SessionDB:
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=session_id, source="tui", model="test/model")
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    return db


def _durable_messages(db_path: Path, session_id: str) -> list[dict]:
    restarted = SessionDB(db_path=db_path)
    try:
        return restarted.get_messages_as_conversation(session_id)
    finally:
        restarted.close()


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model", usage=None,
    )


def _unanswered(messages: list[dict]) -> list[str]:
    """Declared tool_call ids in ``messages`` with no matching role=tool row."""
    answered = {
        (m.get("tool_call_id") or "").strip()
        for m in messages
        if m.get("role") == "tool"
    }
    missing: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            cid = coalesce_tool_call_id(tc)
            if cid and cid not in answered:
                missing.append(cid)
    return missing


class _FlushFailsOnce:
    """Wrap the real flush; make call #``fail_on`` behave like a transient DB failure."""

    def __init__(self, agent, fail_on: int, mode: str = "return_false"):
        self._real = agent._flush_messages_to_session_db
        self.fail_on = fail_on
        self.mode = mode
        self.calls = 0

    def __call__(self, messages, conversation_history=None):
        self.calls += 1
        if self.calls == self.fail_on:
            if self.mode == "raise":
                raise RuntimeError("injected transient session-db failure")
            return False
        return self._real(messages, conversation_history)


# --------------------------------------------------------------------------
# (a) E2E: real run_conversation, real SQLite, reload, outbound payload
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fail_on", [
    pytest.param(2, marks=pytest.mark.xfail(strict=True, reason=(
        "RESIDUAL, not covered by any executor change: flush #2 is the PRE-EXECUTION "
        "assistant tool-call flush. turn_tool_round breaks with session_persistence_failed "
        "before a single tool dispatches, yet a later successful flush still writes the "
        "assistant row — so the store keeps 3 declared ids with 0 tool rows. Only "
        "turn_tool_round can close this one; sanitize_api_messages still repairs the payload."
    ))),
    3,
    4,
])
def test_e2e_midbatch_persist_failure_leaves_no_orphan_in_store_or_payload(tmp_path, fail_on):
    """A transient flush failure mid multi-tool batch must not orphan tool_call ids.

    Drives the concurrent executor through ``run_conversation``, lets the
    turn end, reloads the transcript from SQLite in a fresh connection and
    re-assembles the next turn's outbound payload with the real sanitizer.
    """
    home = tmp_path / "home"
    agent = _make_agent(home)
    db_path = tmp_path / "state.db"
    session_id = f"e2e-midbatch-{fail_on}"
    db = _attach_real_session_db(agent, db_path, session_id)

    calls = [_mock_tool_call(call_id=f"call_{i}", arguments='{"query": "q%d"}' % i) for i in (1, 2, 3)]
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=calls),
        _mock_response(content="done", finish_reason="stop"),
    ]
    flush = _FlushFailsOnce(agent, fail_on=fail_on)
    agent._flush_messages_to_session_db = flush

    with (
        patch("agent.agent_runtime_helpers.invoke_tool", return_value="search result"),
        patch("model_tools.handle_function_call", return_value="search result"),
        patch("agent.tool_executor.maybe_persist_tool_result", side_effect=lambda **kw: kw["content"]),
        patch.object(agent, "_save_trajectory"),
    ):
        agent.run_conversation("search three things")

    db.close()

    durable = _durable_messages(db_path, session_id)
    roles = [m["role"] for m in durable]
    missing_store = _unanswered(durable)
    payload = sanitize_api_messages(durable)
    missing_payload = _unanswered(payload)

    print(f"\n[fail_on={fail_on}] flushes={flush.calls} roles={roles}")
    for _i, _m in enumerate(durable):
        print(f"[fail_on={fail_on}] durable[{_i}] role={_m.get('role')!r} "
              f"content={str(_m.get('content'))[:60]!r} "
              f"tool_calls={[coalesce_tool_call_id(t) for t in (_m.get('tool_calls') or [])]} "
              f"tcid={_m.get('tool_call_id')!r}")
    print(f"[fail_on={fail_on}] store tool ids ="
          f" {[m.get('tool_call_id') for m in durable if m['role'] == 'tool']}")
    print(f"[fail_on={fail_on}] payload tool ids ="
          f" {[m.get('tool_call_id') for m in payload if m['role'] == 'tool']}")

    assert missing_store == [], (
        f"persisted store has unanswered tool_call ids {missing_store}; roles={roles}"
    )
    assert missing_payload == [], (
        f"outbound payload has unanswered tool_call ids {missing_payload}"
    )


# --------------------------------------------------------------------------
# (b) CONCURRENT executor: drain must carry the real remaining call ids
# --------------------------------------------------------------------------
def test_concurrent_batch_persist_failure_pairs_remaining_ids():
    """``_append_batch_results`` drains the tail after a failed commit — the
    drained rows must carry the REAL remaining tool_call ids, not blanks."""
    home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    agent = _make_agent(home)
    calls = [_mock_tool_call(call_id=f"cc-{i}", arguments='{"query": "q%d"}' % i) for i in (1, 2, 3)]
    messages: list = []

    flush = _FlushFailsOnce(agent, fail_on=1)
    flush._real = lambda messages, conversation_history=None: True
    agent._flush_messages_to_session_db = flush

    with (
        patch("agent.agent_runtime_helpers.invoke_tool", return_value="ok"),
        patch("agent.tool_executor.maybe_persist_tool_result", side_effect=lambda **kw: kw["content"]),
    ):
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
        )

    got = [(m.get("role"), m.get("tool_call_id"), m.get("name")) for m in messages]
    print(f"\n[concurrent] rows={got}")
    assert [m.get("tool_call_id") for m in messages] == ["cc-1", "cc-2", "cc-3"], (
        f"concurrent drain did not pair the remaining ids: {got}"
    )


# --------------------------------------------------------------------------
# (b) SEGMENTED executor: later segments must be drained too
# --------------------------------------------------------------------------
def test_segmented_persist_failure_pairs_calls_in_later_segments():
    """A persistence failure inside segment 1 must still pair every call in
    segments 2..N — the segmented dispatcher owns those ids."""
    home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    agent = _make_agent(home)
    seg1 = [_mock_tool_call(call_id="seg1-a", arguments='{"query": "a"}')]
    seg2 = [_mock_tool_call(call_id="seg2-a", arguments='{"query": "b"}'), _mock_tool_call(call_id="seg2-b", arguments='{"query": "c"}')]
    segments = [("sequential", seg1), ("parallel", seg2)]
    assistant = SimpleNamespace(content="", tool_calls=seg1 + seg2)
    messages: list = []

    flush = _FlushFailsOnce(agent, fail_on=1)
    flush._real = lambda messages, conversation_history=None: True
    agent._flush_messages_to_session_db = flush

    with (
        patch("agent.agent_runtime_helpers.invoke_tool", return_value="ok"),
        patch("model_tools.handle_function_call", return_value="ok"),
        patch("agent.tool_executor.maybe_persist_tool_result", side_effect=lambda **kw: kw["content"]),
    ):
        execute_tool_calls_segmented(agent, assistant, messages, "task-1", segments=segments)

    got = [m.get("tool_call_id") for m in messages]
    print(f"\n[segmented] rows={got}")
    assert got == ["seg1-a", "seg2-a", "seg2-b"], (
        f"segmented drain lost the later segments: {got}"
    )


# --------------------------------------------------------------------------
# (c) Finding 2: pre-execution abort -> turn_finalizer -> repair -> sanitizer
# --------------------------------------------------------------------------
def test_e2e_preexecution_abort_orphan_reaches_store_but_not_payload(tmp_path):
    """A pre-execution abort of a pure tool-call turn.

    Runs the REAL turn_finalizer, the REAL durable flush, a reload from
    SQLite and the REAL outbound sanitizer, then reports exactly where an
    orphan survives.
    """
    home = tmp_path / "home"
    agent = _make_agent(home)
    db_path = tmp_path / "state.db"
    session_id = "e2e-preexec-abort"
    db = _attach_real_session_db(agent, db_path, session_id)

    calls = [_mock_tool_call(call_id="orphan-1", arguments='{"query": "x"}')]
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=calls),
    ]

    def _abort(*_a, **_kw):
        raise TimeoutError("tool preparation timed out before dispatch")

    with (
        patch.object(agent, "_execute_tool_calls", side_effect=_abort),
        patch.object(agent, "_save_trajectory"),
    ):
        result = agent.run_conversation("do a thing")

    db.close()
    durable = _durable_messages(db_path, session_id)
    payload = sanitize_api_messages(durable)

    print(f"\n[preexec] final_response={str(result.get('final_response'))[:70]!r}")
    for _i, _m in enumerate(durable):
        print(f"[preexec] durable[{_i}] role={_m.get('role')!r} "
              f"content={str(_m.get('content'))[:60]!r} "
              f"tool_calls={[coalesce_tool_call_id(t) for t in (_m.get('tool_calls') or [])]} "
              f"tcid={_m.get('tool_call_id')!r}")
    print(f"[preexec] store unanswered  = {_unanswered(durable)}")
    print(f"[preexec] payload roles     = {[m['role'] for m in payload]}")
    print(f"[preexec] payload unanswered= {_unanswered(payload)}")

    assert _unanswered(payload) == [], "outbound payload carries an unanswered tool_call"
    assert _unanswered(durable) == [], (
        f"persisted store carries an unanswered tool_call: {_unanswered(durable)}"
    )
