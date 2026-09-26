"""Behavior contracts for tool-result outcome persistence in the session store.

``messages`` carries TWO orthogonal axes about a tool call, and the distinction is
load-bearing:

* ``effect_disposition`` — the #61783 contract: is an effect possible? ``none`` =
  provably none (the call never ran), ``unknown`` = an interrupted effectful call
  may have acted and this cannot be observed. NULL = the call completed and its
  effect is simply not described.
* ``execution_status`` — how did the call end? ``success`` / ``error`` /
  ``blocked`` / ``timeout`` / ``cancelled``.

Conflating them loses information: a timed-out ``write_file`` is ``unknown`` +
``timeout``, a guardrail-blocked call is ``none`` + ``blocked``. Replay recovery
depends on the first axis, so the second must not be smuggled into it.

These tests drive the real production dispatch surfaces (``agent.tool_executor``
and the turn-loop recovery writers) with a real ``SessionDB`` on a temp
``HERMES_HOME`` and read the durable row back through a fresh handle.
"""

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import _ManagedToolResult, execute_tool_calls_segmented
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


def _durable_tool_outcomes(db_path: Path, session_id: str) -> list:
    """Read ``(effect_disposition, execution_status)`` per durably persisted tool row."""
    db = SessionDB(db_path=db_path)
    try:
        msgs = db.get_messages_as_conversation(session_id)
    finally:
        db.close()
    return [(m.get("effect_disposition"), m.get("execution_status"))
            for m in msgs if m.get("role") == "tool"]


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


def _queue_turn_responses(agent, calls) -> None:
    """Queue one tool-call response + one final text response on the mocked client."""
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=list(calls)),
        _mock_response(content="done", finish_reason="stop"),
    ]


def _run_tool_turn(agent, *, parallel=False, calls=None):
    """One full run_conversation turn dispatching ``calls`` through the real executor."""
    calls = calls or [_mock_tool_call()]
    _queue_turn_responses(agent, calls)
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
# Executor paths: the status axis fills in, the effect axis keeps its contract
# ---------------------------------------------------------------------------
def test_sequential_success_is_null_effect_plus_success_status(tmp_path):
    """A completed call does not claim "no effect" — only that it succeeded."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "seq-success"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        with patch("model_tools.handle_function_call", return_value='{"ok": true}'):
            _run_tool_turn(agent)
    finally:
        db.close()

    assert _durable_tool_outcomes(db_path, session_id) == [(None, "success")]


def test_sequential_error_result_persists_error_status(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "seq-error"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        # A generic payload carrying error + success=false is what _detect_tool_failure reads.
        with patch("model_tools.handle_function_call",
                   return_value='{"success": false, "error": "boom"}'):
            _run_tool_turn(agent)
    finally:
        db.close()

    assert _durable_tool_outcomes(db_path, session_id) == [(None, "error")]


def test_concurrent_outcomes_are_keyed_to_their_own_call(tmp_path):
    """No sorting: swapping the statuses between probe_a/probe_b must fail this."""
    agent = _make_agent(tools=("probe_a", "probe_b"))
    db_path = tmp_path / "state.db"
    session_id = "conc-mixed"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        calls = [_mock_tool_call(name="probe_a", call_id="ca"),
                 _mock_tool_call(name="probe_b", call_id="cb")]
        _queue_turn_responses(agent, calls)

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

    db = SessionDB(db_path=db_path)
    try:
        by_call = {
            m.get("tool_call_id"): (m.get("effect_disposition"), m.get("execution_status"))
            for m in db.get_messages_as_conversation(session_id) if m.get("role") == "tool"
        }
    finally:
        db.close()
    assert by_call == {"ca": (None, "success"), "cb": (None, "error")}


def test_blocked_call_is_none_effect_plus_blocked_status(tmp_path):
    """A call refused before dispatch provably had no effect — ``none`` + ``blocked``."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "blocked"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        blocked_result = json.dumps({
            "status": "blocked",
            "user_summary": "blocked by guardrail",
            "message": "BLOCKED: do not retry",
        })
        _queue_turn_responses(agent, [_mock_tool_call(call_id="cb")])

        from agent.tool_executor import execute_tool_calls_concurrent

        def _force_concurrent_exec(assistant_message, messages, effective_task_id, api_call_count=0):
            execute_tool_calls_concurrent(agent, assistant_message, messages, effective_task_id, api_call_count)

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_execute_tool_calls", _force_concurrent_exec),
            patch("agent.tool_executor._run_agent_tool_execution_middleware",
                  return_value=_ManagedToolResult(
                      result=blocked_result, args={}, middleware_trace=[],
                      blocked=True, dispatched=False)),
        ):
            agent.run_conversation("try the guarded tool")
    finally:
        db.close()

    assert _durable_tool_outcomes(db_path, session_id) == [("none", "blocked")]


def test_concurrent_deadline_is_unknown_effect_plus_timeout_status(tmp_path):
    """A call killed mid-flight: the effect is unobservable, the outcome is a timeout."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "timeout"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        def _slow_tool(*_args, **_kwargs):
            time.sleep(7.0)  # outlive the 1s deadline AND the ~5s poll interval
            return SimpleNamespace(result="late", args={}, middleware_trace=None,
                                   blocked=False, dispatched=True)

        from agent.tool_executor import execute_tool_calls_concurrent

        _queue_turn_responses(agent, [_mock_tool_call(call_id="ct")])

        def _force_concurrent_exec(assistant_message, messages, effective_task_id, api_call_count=0):
            execute_tool_calls_concurrent(agent, assistant_message, messages, effective_task_id, api_call_count)

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_execute_tool_calls", _force_concurrent_exec),
            patch("agent.tool_executor._resolve_concurrent_tool_timeout", return_value=1.0),
            patch.object(agent, "_invoke_tool", side_effect=_slow_tool),
        ):
            agent.run_conversation("hang the tool")
    finally:
        db.close()

    outcomes = _durable_tool_outcomes(db_path, session_id)
    assert outcomes and all(o == ("unknown", "timeout") for o in outcomes), outcomes


def test_interrupt_skip_is_none_effect_plus_cancelled_status(tmp_path):
    """Skipped-before-start: provably no effect, and the outcome is a cancellation."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "skip"
    db = _attach_real_session_db(agent, db_path, session_id)
    try:
        from agent.tool_executor import execute_tool_calls_concurrent

        _queue_turn_responses(agent, [_mock_tool_call(call_id="cs")])

        def _interrupt_then_exec(assistant_message, messages, effective_task_id, api_call_count=0):
            # Set the flag after the loop accepted the tool-call response and before the
            # executor's own check, so the interrupt-skip path is the one under test.
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

    assert _durable_tool_outcomes(db_path, session_id) == [("none", "cancelled")]


def test_invalid_arguments_agree_across_both_dispatch_paths():
    """Malformed JSON never reaches the tool: ``none`` + ``error`` on BOTH dispatch paths.

    Regression for the concurrent/sequential disagreement (concurrent used to report
    ``blocked`` because the placeholder outcome set ``blocked=True``). Driven straight at
    the two dispatchers: the turn loop's own JSON-validation retry would otherwise
    intercept malformed arguments before the executor ever sees them.
    """
    from agent.tool_executor import execute_tool_calls_concurrent, execute_tool_calls_sequential

    agent = _make_agent(tools=("probe_a", "probe_b"))
    calls = [_mock_tool_call(name="probe_a", call_id="ca", arguments="[not, an, object]"),
             _mock_tool_call(name="probe_b", call_id="cb", arguments="[also, bad]")]

    sequential_messages: list[dict] = []
    assistant = SimpleNamespace(tool_calls=list(calls))
    with patch.object(agent, "_flush_messages_to_session_db", return_value=True):
        execute_tool_calls_sequential(agent, assistant, sequential_messages, "task", finalize=False)

    concurrent_messages: list[dict] = []
    assistant = SimpleNamespace(tool_calls=list(calls))
    with patch.object(agent, "_flush_messages_to_session_db", return_value=True):
        execute_tool_calls_concurrent(agent, assistant, concurrent_messages, "task", finalize=False)

    def _outcomes(rows):
        return [(m.get("effect_disposition"), m.get("execution_status"))
                for m in rows if m.get("role") == "tool"]

    assert _outcomes(sequential_messages) == [("none", "error"), ("none", "error")]
    assert _outcomes(concurrent_messages) == [("none", "error"), ("none", "error")]


# ---------------------------------------------------------------------------
# Sibling durable writers the executor never sees
# ---------------------------------------------------------------------------
def test_unknown_tool_recovery_results_carry_error_status(tmp_path):
    """Unknown-tool / invalid-JSON recovery rows are execution errors with no effect."""
    from types import SimpleNamespace as NS

    from agent.turn_tool_validation import _append_tool_error_results

    tc = NS(function=NS(name="no_such_tool"))
    messages: list[dict] = []
    _append_tool_error_results(messages, [tc], lambda _tc: "Error: unknown tool")

    row = messages[0]
    assert row["role"] == "tool"
    assert (row["effect_disposition"], row["execution_status"]) == ("none", "error")


def test_outer_loop_error_results_split_effect_by_tool_kind():
    """A never-answered effectful call is UNKNOWN; a read-only one provably had none."""
    from agent.turn_loop_errors import handle_outer_loop_error

    agent = SimpleNamespace(
        suppress_status_output=True, max_iterations=10,
        _safe_print=lambda *a, **k: None,
    )
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "w", "function": {"name": "write_file", "arguments": "{}"}},
            {"id": "r", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
    ]
    verdict = handle_outer_loop_error(
        agent, e=RuntimeError("api exploded"), _outer_error_count=0, api_call_count=0,
        messages=messages, conversation_history=None, _turn_exit_reason="", failed=False,
        final_response="",
    )
    assert verdict.action in {"break", "fallthrough"}
    recovered = {m["tool_call_id"]: m for m in messages if m.get("role") == "tool"}
    assert recovered["w"]["effect_disposition"] == "unknown"
    assert recovered["r"]["effect_disposition"] == "none"
    assert {m["execution_status"] for m in recovered.values()} == {"error"}


def test_codex_projection_records_provider_reported_status():
    """Provider-side tool items get a status; the effect axis stays unclaimed."""
    from agent.transports.codex_event_projector import CodexEventProjector

    ok = CodexEventProjector().project({"method": "item/completed", "params": {"item": {
        "type": "commandExecution", "id": "c1", "command": "x", "cwd": "/",
        "status": "completed", "exitCode": 0, "aggregatedOutput": "done",
    }}})
    assert ok.messages[1]["execution_status"] == "success"
    assert "effect_disposition" not in ok.messages[1]

    failed = CodexEventProjector().project({"method": "item/completed", "params": {"item": {
        "type": "mcpToolCall", "id": "m1", "server": "x", "tool": "y",
        "status": "failed", "arguments": {}, "error": {"code": -1, "message": "no"},
    }}})
    assert failed.messages[1]["execution_status"] == "error"


# ---------------------------------------------------------------------------
# Storage contract
# ---------------------------------------------------------------------------
def test_execution_status_survives_the_export_import_roundtrip(tmp_path):
    """A copied session keeps the outcome axis; losing it would silently blind the feed."""
    source = SessionDB(db_path=tmp_path / "source.db")
    try:
        source.create_session(session_id="roundtrip", source="tui", model="test/model")
        source.append_message(
            "roundtrip", "tool", content="boom", tool_name="probe_tool",
            tool_call_id="c1", effect_disposition="unknown", execution_status="timeout",
        )
        payload = source.export_session("roundtrip")
    finally:
        source.close()

    target = SessionDB(db_path=tmp_path / "target.db")
    try:
        target.import_sessions([payload])
        rows = [m for m in target.get_messages_as_conversation("roundtrip")
                if m.get("role") == "tool"]
    finally:
        target.close()
    assert [(m.get("effect_disposition"), m.get("execution_status")) for m in rows] == [
        ("unknown", "timeout")]


def test_execution_status_is_stripped_from_the_provider_wire():
    """Strict OpenAI-compatible routes reject unknown message keys (HTTP 400)."""
    from agent.transports.chat_completions import _sanitize_message

    out = _sanitize_message(
        {"role": "tool", "content": "x", "tool_call_id": "c1",
         "tool_name": "probe", "effect_disposition": "unknown", "execution_status": "timeout"},
        strip_extra_content=True,
    )
    assert "execution_status" not in out
    assert "effect_disposition" not in out
    assert "tool_name" not in out
