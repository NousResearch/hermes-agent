"""Behavior contracts for effect_disposition persistence on tool completion paths.

The executor computes the outcome of every tool call (``_detect_tool_failure``,
guardrail ``blocked`` flags, timeout markers) but historically persisted
``effect_disposition = NULL`` on normal completions, leaving the state.db column
filled only on anomalous paths (timeout "unknown", blocked "none", replay repair).

These tests pin the contract: the outcome the executor already classified is what
lands in ``messages.effect_disposition``:

    1. sequential success  -> "success"
    2. sequential error    -> "error"    (post-transform result classification)
    3. concurrent success  -> "success"  and error -> "error"
    4. guardrail-blocked   -> "blocked"  (call never ran)
    5. timeout             -> "timeout"  (was "unknown")
    6. interrupt skip      -> "cancelled" (was "none"; path already emits
                            post_tool_call with status="cancelled")

All tests drive the real production dispatch surface
(``agent.tool_executor`` + a real ``SessionDB`` on a temp ``HERMES_HOME``) and
read the durable row back through a fresh SessionDB handle.
"""

import copy
import io
import json
import sqlite3
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.tool_executor import execute_tool_calls_segmented, _ManagedToolResult
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _disable_background_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)


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


def _make_agent(tools=("probe_tool",)):
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs(*tools)),
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


def _durable_dispositions(db_path: Path, session_id: str) -> list:
    """Read (role, effect_disposition) rows durably persisted for the session."""
    db = SessionDB(db_path=db_path)
    try:
        msgs = db.get_messages_as_conversation(session_id)
    finally:
        db.close()
    return [(m.get("role"), m.get("effect_disposition")) for m in msgs]


def _mock_tool_call(name="probe_tool", arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _run_tool_turn(agent, tool_payload: str, *, call_id="c1", parallel=False):
    """One full run_conversation turn where the tool returns ``tool_payload``."""
    tool_call = _mock_tool_call(call_id=call_id)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls",
                       tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]
    seg = "parallel" if parallel else "sequential"
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.tool_executor.execute_tool_calls_segmented",
              side_effect=lambda am, msgs, task, api_call_count=0, **kw: (
                  execute_tool_calls_segmented(
                      agent, am, msgs, task,
                      segments=[(seg, am.tool_calls)]))),
    ):
        return agent.run_conversation("use the probe tool")


# ---------------------------------------------------------------------------
# 1+2: sequential path — success and error classification is persisted
# ---------------------------------------------------------------------------
def test_sequential_success_persists_success(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "seq-success"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        with patch("model_tools.handle_function_call", return_value='{"ok": true}'):
            _run_tool_turn(agent, '{"ok": true}')
    finally:
        db.close()

    rows = _durable_dispositions(db_path, session_id)
    tool_rows = [d for role, d in rows if role == "tool"]
    assert tool_rows == ["success"], rows


def test_sequential_error_result_persists_error(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "seq-error"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        # generic dict with error + success=false -> _detect_tool_failure is True
        with patch("model_tools.handle_function_call",
                   return_value='{"success": false, "error": "boom"}'):
            _run_tool_turn(agent, '{"success": false, "error": "boom"}')
    finally:
        db.close()

    rows = _durable_dispositions(db_path, session_id)
    tool_rows = [d for role, d in rows if role == "tool"]
    assert tool_rows == ["error"], rows


# ---------------------------------------------------------------------------
# 3: concurrent path
# ---------------------------------------------------------------------------
def test_concurrent_success_and_error_persist(tmp_path):
    agent = _make_agent(tools=("probe_a", "probe_b"))
    db_path = tmp_path / "state.db"
    session_id = "conc-mixed"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        calls = [_mock_tool_call(name="probe_a", call_id="ca"),
                 _mock_tool_call(name="probe_b", call_id="cb")]
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=calls),
            _mock_response(content="done", finish_reason="stop"),
        ]

        def _fake_handle(name, args, task_id=None, **kwargs):
            if name == "probe_b":
                return '{"success": false, "error": "nope"}'
            return '{"ok": true}'

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.tool_executor.execute_tool_calls_segmented",
                  side_effect=lambda am, msgs, task, api_call_count=0, **kw: (
                      execute_tool_calls_segmented(
                          agent, am, msgs, task,
                          segments=[("parallel", am.tool_calls)]))),
            patch("model_tools.handle_function_call", side_effect=_fake_handle),
        ):
            agent.run_conversation("run both probes")
    finally:
        db.close()

    rows = _durable_dispositions(db_path, session_id)
    tool_rows = [d for role, d in rows if role == "tool"]
    assert sorted(tool_rows, key=str) == ["error", "success"], rows


# ---------------------------------------------------------------------------
# 4: guardrail-blocked call never ran -> "blocked"
# ---------------------------------------------------------------------------
def test_blocked_call_persists_blocked(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "blocked"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        calls = [_mock_tool_call(call_id="cb")]
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=calls),
            _mock_response(content="done", finish_reason="stop"),
        ]
        blocked_result = json.dumps({
            "status": "blocked",
            "user_summary": "blocked by guardrail",
            "message": "BLOCKED: do not retry",
        })
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.tool_executor.execute_tool_calls_segmented",
                  side_effect=lambda am, msgs, task, api_call_count=0, **kw: (
                      execute_tool_calls_segmented(
                          agent, am, msgs, task,
                          segments=[("parallel", am.tool_calls)]))),
            # the worker reads the managed result returned by the execution middleware;
            # blocked=True means the guardrail layer refused before dispatch
            patch("agent.tool_executor._run_agent_tool_execution_middleware",
                  return_value=_ManagedToolResult(
                      result=blocked_result, args={}, middleware_trace=[],
                      blocked=True, dispatched=False)),
        ):
            agent.run_conversation("try the guarded tool")
    finally:
        db.close()

    rows = _durable_dispositions(db_path, session_id)
    tool_rows = [d for role, d in rows if role == "tool"]
    assert tool_rows == ["blocked"], rows


# ---------------------------------------------------------------------------
# 5: deadline exceeded -> "timeout" (truthful, was "unknown")
# ---------------------------------------------------------------------------
def test_timeout_persists_timeout_literal(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "timeout"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        calls = [_mock_tool_call(call_id="ct")]
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=calls),
            _mock_response(content="done", finish_reason="stop"),
        ]

        def _slow_tool(*_args, **_kwargs):
            time.sleep(7.0)  # outlive the 1s deadline AND the ~5s poll interval
            return SimpleNamespace(result="late", args={}, middleware_trace=None,
                                   blocked=False, dispatched=True)

        from agent.tool_executor import execute_tool_calls_concurrent

        def _force_concurrent_exec(assistant_message, messages, effective_task_id, api_call_count=0):
            # single-call batches default to the sequential executor; force concurrent
            execute_tool_calls_concurrent(agent, assistant_message, messages, effective_task_id, api_call_count)

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_execute_tool_calls", _force_concurrent_exec),
            patch("agent.tool_executor._resolve_concurrent_tool_timeout",
                  return_value=1.0),
            patch.object(agent, "_invoke_tool", side_effect=_slow_tool),
        ):
            agent.run_conversation("hang the tool")
    finally:
        db.close()

    rows = _durable_dispositions(db_path, session_id)
    tool_rows = [d for role, d in rows if role == "tool"]
    assert tool_rows and all(d == "timeout" for d in tool_rows), rows


# ---------------------------------------------------------------------------
# 6: interrupt skip -> "cancelled" (was "none"; hook already says cancelled)
# ---------------------------------------------------------------------------
def test_interrupt_skip_persists_cancelled(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "skip"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        calls = [_mock_tool_call(call_id="cs")]
        agent.client.chat.completions.create.side_effect = [
            _mock_response(content="", finish_reason="tool_calls", tool_calls=calls),
            _mock_response(content="done", finish_reason="stop"),
        ]
        from agent.tool_executor import execute_tool_calls_concurrent

        def _interrupt_then_exec(assistant_message, messages, effective_task_id, api_call_count=0):
            # set the flag HERE: after the loop accepted the tool-call response,
            # before the executor's own check — the interrupt-skip path inside
            # execute_tool_calls_concurrent is what must run
            agent._interrupt_requested = True
            execute_tool_calls_concurrent(agent, assistant_message, messages, effective_task_id, api_call_count)

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_execute_tool_calls", _interrupt_then_exec),
        ):
            agent.run_conversation("skip me")
    finally:
        db.close()

    rows = _durable_dispositions(db_path, session_id)
    tool_rows = [d for role, d in rows if role == "tool"]
    assert tool_rows == ["cancelled"], rows
