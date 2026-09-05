"""Behavior contracts for incremental tool-call persistence (#49045).

A destructive or process-terminating tool that runs during tool execution
must not lose the just-executed assistant(tool_calls) block or the tool
results that were produced before it fired.  These tests pin the contract:

    1. run_conversation flushes the assistant tool-call turn to the session
       DB BEFORE handing control to _execute_tool_calls (so a tool that
       restarts/kills the process never orphans the tool-call block).
    2. The SEQUENTIAL tool path flushes each tool result to the session DB
       immediately after appending it — BEFORE the next tool dispatches.
    3. The CONCURRENT tool path flushes each tool result in append order.

These exercise the REAL production dispatch surfaces:

    * sequential -> ``model_tools.handle_function_call`` (tool_executor ~1256/1298)
    * concurrent -> ``agent._invoke_tool`` (tool_executor ~539)

Mocking the genuine dispatch surface keeps the tests deterministic (no real
``web_search`` / network) AND mutation-survivable: the ordering assertions
read snapshots captured at flush time, so removing any production flush call
makes the corresponding assertion fail.
"""

import copy
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
from agent.agent_runtime_helpers import sanitize_api_messages
from agent.tool_executor import execute_tool_calls_segmented
from hermes_state import SessionDB
from run_agent import AIAgent


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


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
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
    restarted_db = SessionDB(db_path=db_path)
    try:
        return restarted_db.get_messages_as_conversation(session_id)
    finally:
        restarted_db.close()


def _durable_roles(db_path: Path, session_id: str) -> list[str]:
    return [message["role"] for message in _durable_messages(db_path, session_id)]


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


# ---------------------------------------------------------------------------
# Contract 1: run_conversation persists the assistant tool-call block BEFORE
# tool execution begins.
# ---------------------------------------------------------------------------
def test_run_conversation_flushes_assistant_tool_call_before_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]

    # Record a deep snapshot of the message list at every flush so the
    # assertion does not depend on later mutations.
    flush_snapshots: list[list] = []

    def _record_flush(messages, conversation_history=None):
        flush_snapshots.append(copy.deepcopy(messages))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    # Capture observations at execute time into module-level lists rather than
    # asserting inside _execute_tool_calls — run_conversation's outer loop
    # swallows exceptions, so an in-callback assertion would never surface.
    executed = {"count": 0}
    snapshot_at_execute: list = []

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        executed["count"] += 1
        # Record the DB state observed at the moment tool execution begins.
        snapshot_at_execute.append(
            copy.deepcopy(flush_snapshots[-1]) if flush_snapshots else None
        )
        # Simulate the tool producing a result (as the real path would).
        messages.append(make_tool_result_message("web_search", "search result", "c1"))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("search something")

    assert executed["count"] == 1, "_execute_tool_calls was never reached"
    # The assistant tool-call block MUST have been flushed before execution.
    last = snapshot_at_execute[0]
    assert last is not None, "no flush occurred before tool execution"
    assert last[-1]["role"] == "assistant"
    assert last[-1]["tool_calls"][0]["id"] == "c1"
    assert result["final_response"] == "done"


def test_interim_assistant_is_durable_before_ui_projection_on_abnormal_exit(tmp_path):
    """A visible interim assistant row must survive an immediate process exit.

    ``GeneratorExit`` models an uncatchable turn interruption at the UI bridge:
    no turn finalizer or graceful shutdown persistence is allowed to rescue the
    row after the callback observes it.
    """
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "interim-abnormal-exit"
    db = _attach_real_session_db(agent, db_path, session_id)
    tool_call = _mock_tool_call(call_id="visible-call")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )

    roles_seen_by_ui: list[str] = []

    def _ui_projection(_text, *, already_streamed=False):
        roles_seen_by_ui.extend(_durable_roles(db_path, session_id))
        raise GeneratorExit("simulated process termination after UI projection")

    agent.interim_assistant_callback = _ui_projection
    try:
        with pytest.raises(GeneratorExit, match="simulated process termination"):
            agent.run_conversation("inspect the repository")
    finally:
        db.close()

    assert roles_seen_by_ui == ["user", "assistant"]
    durable = _durable_messages(db_path, session_id)
    assert [message["role"] for message in durable] == ["user", "assistant"]
    assert durable[1]["content"] == "I'll inspect the repository now."
    assert durable[1]["tool_calls"][0]["id"] == "visible-call"

    # Cold-resume reconciliation closes the interrupted call in the provider
    # payload without mutating or duplicating the canonical transcript.
    resumed = sanitize_api_messages(durable)
    assert [message["role"] for message in resumed] == [
        "user",
        "assistant",
        "tool",
    ]
    assert resumed[2]["tool_call_id"] == "visible-call"
    assert len(_durable_messages(db_path, session_id)) == 2


def test_failed_assistant_persist_blocks_ui_projection_and_tool_side_effects():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="must-not-run")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.interim_assistant_callback = MagicMock()
    agent._execute_tool_calls = MagicMock()

    result = agent.run_conversation("inspect the repository")

    agent.interim_assistant_callback.assert_not_called()
    agent._execute_tool_calls.assert_not_called()
    assert agent.client is not None
    assert agent.client.chat.completions.create.call_count == 1
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["turn_exit_reason"] == "session_persistence_failed"
    # No exception was visible (flush returned False), so the cause is
    # unknown — but the machine-readable contract fields must still be set.
    assert result["failure_reason"] == "session_persistence_failed:unknown"
    assert isinstance(result.get("error"), str) and result["error"].strip() != ""


def test_locked_flush_exception_surfaces_locked_cause_in_result_contract():
    """SQLite write-lock contention must surface as a 'locked' cause.

    Gateway contract: result['failure_reason'] is exactly
    'session_persistence_failed:locked' and result['error'] is a non-empty
    string whose wording talks about busy storage, NOT disk space.
    """
    import sqlite3

    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="must-not-run")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._flush_messages_to_session_db = MagicMock(
        side_effect=sqlite3.OperationalError("database is locked")
    )
    agent.interim_assistant_callback = MagicMock()
    agent._execute_tool_calls = MagicMock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("inspect the repository")

    agent.interim_assistant_callback.assert_not_called()
    agent._execute_tool_calls.assert_not_called()
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "session_persistence_failed"
    assert result["failure_reason"] == "session_persistence_failed:locked"
    assert isinstance(result.get("error"), str) and result["error"].strip() != ""
    assert "busy" in result["error"].lower()
    assert "disk" not in result["error"].lower()


def test_persistence_cause_resets_between_turns():
    """A locked failure on turn 1 must not leak its cause into turn 2."""
    import sqlite3

    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="must-not-run")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="I'll inspect the repository now.",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._flush_messages_to_session_db = MagicMock(
        side_effect=sqlite3.OperationalError("database is locked")
    )
    agent._execute_tool_calls = MagicMock()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("inspect the repository")
        assert first["failure_reason"] == "session_persistence_failed:locked"

        # Storage recovered but the flush function now reports a bare False
        # (no exception): the stale 'locked' cause must not be reused.
        agent.client.chat.completions.create.side_effect = None
        agent.client.chat.completions.create.return_value = _mock_response(
            content="I'll inspect the repository now.",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(call_id="must-not-run-2")],
        )
        agent._flush_messages_to_session_db = MagicMock(return_value=False)
        second = agent.run_conversation("inspect the repository again")

    assert second["turn_exit_reason"] == "session_persistence_failed"
    assert second["failure_reason"] == "session_persistence_failed:unknown"


# ---------------------------------------------------------------------------
# Contract 2: the SEQUENTIAL path flushes each tool result immediately, BEFORE
# the next tool dispatches.  Dispatch goes through model_tools.handle_function_call
# (the real production surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_sequential_flushes_each_tool_result_before_next_dispatch():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    # Ordered event log interleaving real dispatches and DB flushes.
    events: list = []

    def _fake_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # The result for call N must have been flushed before call N+1 fires.
        events.append(("dispatch", kwargs.get("tool_call_id")))
        return f"result-{kwargs.get('tool_call_id')}"

    def _record_flush(flush_messages, conversation_history=None):
        # Snapshot the tail tool result that triggered this flush.
        tail = flush_messages[-1]
        events.append(("flush", tail.get("role"), tail.get("tool_call_id")))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch("model_tools.handle_function_call", side_effect=_fake_dispatch) as disp,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # The mock proves we exercised the REAL sequential dispatch surface.
    assert disp.call_count == 2, "sequential path did not dispatch via handle_function_call"

    # Both tool results landed, in order.
    assert [m["role"] for m in messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Ordering contract: each tool result is flushed AFTER its own dispatch
    # and BEFORE the next dispatch. Expected interleaving:
    #   dispatch c1 -> flush c1 -> dispatch c2 -> flush c2
    assert events == [
        ("dispatch", "c1"),
        ("flush", "tool", "c1"),
        ("dispatch", "c2"),
        ("flush", "tool", "c2"),
    ]


def test_sequential_keyboard_interrupt_emits_results_for_all_calls():
    """A KeyboardInterrupt mid-batch must not leave dangling tool_calls.

    When a tool handler raises KeyboardInterrupt, the sequential executor
    re-raises to abort the turn — but it must first append a tool result for
    the interrupted call AND every remaining call, or the assistant tool-call
    turn is left without matching tool results (a message-role alternation
    violation that malforms the next provider request). Mirrors the
    cooperative-interrupt and concurrent paths, which already do this.
    """
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
        _mock_tool_call(name="web_search", call_id="c3"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    def _interrupt_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # First tool raises a hard interrupt mid-batch.
        raise KeyboardInterrupt()

    agent._flush_messages_to_session_db = MagicMock()

    with (
        patch("model_tools.handle_function_call", side_effect=_interrupt_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # Every call_id has a matching tool result — alternation preserved.
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_results] == ["c1", "c2", "c3"]
    # The results are marked as cancelled, not fabricated successes.
    assert all("cancelled" in m["content"].lower() for m in tool_results)


@pytest.mark.parametrize("executor_mode", ["sequential", "concurrent"])
def test_tool_result_is_durable_before_ui_completion_on_abnormal_exit(
    tmp_path,
    executor_mode,
):
    """A visible tool completion must already exist in the canonical DB."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = f"tool-result-abnormal-exit-{executor_mode}"
    db = _attach_real_session_db(agent, db_path, session_id)
    tool_call = _mock_tool_call(call_id="visible-call")
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": "I'll inspect the repository now.",
            "tool_calls": [
                {
                    "id": "visible-call",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        },
    ]
    agent._flush_messages_to_session_db(messages)

    roles_seen_by_ui: list[str] = []

    def _ui_completion(*_args):
        roles_seen_by_ui.extend(_durable_roles(db_path, session_id))
        raise GeneratorExit("simulated process termination after tool completion")

    agent.tool_complete_callback = _ui_completion
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    dispatch_patch = (
        patch("model_tools.handle_function_call", return_value="repository result")
        if executor_mode == "sequential"
        else patch.object(agent, "_invoke_tool", return_value="repository result")
    )
    try:
        with (
            dispatch_patch,
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
            pytest.raises(GeneratorExit, match="simulated process termination"),
        ):
            if executor_mode == "sequential":
                agent._execute_tool_calls_sequential(
                    assistant_message,
                    messages,
                    "task-1",
                )
            else:
                agent._execute_tool_calls_concurrent(
                    assistant_message,
                    messages,
                    "task-1",
                )
    finally:
        db.close()

    expected_roles = ["user", "assistant", "tool"]
    assert roles_seen_by_ui == expected_roles
    durable = _durable_messages(db_path, session_id)
    assert [message["role"] for message in durable] == expected_roles
    assert durable[2]["tool_call_id"] == "visible-call"
    assert durable[2]["content"] == "repository result"


@pytest.mark.parametrize("executor_mode", ["sequential", "concurrent"])
def test_failed_tool_result_persist_blocks_completion_projection(executor_mode):
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="failed-persist")
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.tool_complete_callback = MagicMock()
    dispatch_patch = (
        patch("model_tools.handle_function_call", return_value="repository result")
        if executor_mode == "sequential"
        else patch.object(agent, "_invoke_tool", return_value="repository result")
    )

    with (
        dispatch_patch,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        if executor_mode == "sequential":
            agent._execute_tool_calls_sequential(
                assistant_message,
                messages,
                "task-1",
            )
        else:
            agent._execute_tool_calls_concurrent(
                assistant_message,
                messages,
                "task-1",
            )

    agent.tool_complete_callback.assert_not_called()
    assert getattr(agent, "_incremental_persistence_failed", False) is True


def test_segmented_batch_stops_before_later_segment_after_persist_failure():
    agent = _make_agent()
    first = _mock_tool_call(call_id="first")
    second = _mock_tool_call(call_id="second")
    assistant_message = SimpleNamespace(tool_calls=[first, second])
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock(return_value=False)

    with (
        patch.object(agent, "_invoke_tool", return_value="first result") as invoke,
        patch("model_tools.handle_function_call", return_value="second result") as dispatch,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        execute_tool_calls_segmented(
            agent,
            assistant_message,
            messages,
            "task-1",
            segments=[("parallel", [first]), ("sequential", [second])],
        )

    invoke.assert_called_once()
    dispatch.assert_not_called()
    assert getattr(agent, "_incremental_persistence_failed", False) is True


# ---------------------------------------------------------------------------
# Contract 3: the CONCURRENT path flushes each collected tool result in append
# order.  Dispatch goes through agent._invoke_tool (the real concurrent
# surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_concurrent_flushes_each_tool_result_in_order():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    invoked_ids: list = []

    def _fake_invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        invoked_ids.append(tool_call_id)
        return f"result-{tool_call_id}"

    # Each flush must observe exactly one more tool result than the previous
    # flush, in append order — i.e. the tail tool_call_id sequence is c1, c2.
    flushed_tool_ids: list = []
    flush_lengths: list = []

    def _record_flush(flush_messages, conversation_history=None):
        flushed_tool_ids.append(flush_messages[-1]["tool_call_id"])
        flush_lengths.append(len([m for m in flush_messages if m.get("role") == "tool"]))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch.object(agent, "_invoke_tool", side_effect=_fake_invoke) as inv,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    # Proves the real concurrent dispatch surface was exercised.
    assert inv.call_count == 2, "concurrent path did not dispatch via _invoke_tool"
    assert sorted(invoked_ids) == ["c1", "c2"]

    # Results appended in deterministic order.
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Each tool result was flushed exactly once, in append order, with the
    # running tool count growing by one each time (1 then 2).  Removing either
    # production flush call breaks one of these assertions.
    assert flushed_tool_ids == ["c1", "c2"]
    assert flush_lengths == [1, 2]


def test_empty_final_response_updates_already_flushed_blank_assistant_row(tmp_path):
    """#95514: popping _db_persisted must UPDATE the flushed row, not INSERT.

    Incremental persist already wrote assistant(content=''). finalize_turn
    recovers the stream buffer onto that live dict. Production flush is
    append-only, so a re-insert would leave the empty row and add a second
    assistant (assistant→assistant on reload).
    """
    from agent.turn_finalizer import finalize_turn

    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "sess-empty-final-flush"
    _attach_real_session_db(agent, db_path, session_id)
    agent._current_streamed_assistant_text = "Already streamed to the user."

    messages = [
        {"role": "user", "content": "summarize"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "name": "terminal", "content": "ok"},
        {"role": "assistant", "content": ""},
    ]
    agent._flush_messages_to_session_db(messages)
    pre = _durable_messages(db_path, session_id)
    assert pre[-1]["role"] == "assistant"
    assert (pre[-1].get("content") or "") == ""

    finalize_turn(
        agent,
        final_response="",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="summarize",
        original_user_message="summarize",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    reloaded = _durable_messages(db_path, session_id)
    assistants = [m for m in reloaded if m.get("role") == "assistant"]
    assert assistants[-1]["content"] == "Already streamed to the user."
    assert len(assistants) == 2
    assert assistants[0].get("tool_calls")
    assert sum(
        1 for m in assistants
        if not (m.get("content") or "").strip() and not m.get("tool_calls")
    ) == 0


def test_flush_stale_row_id_from_other_session_still_inserts(tmp_path):
    """Copied dicts that keep a parent _row_id must still INSERT in a child session."""
    parent = _make_agent()
    child = _make_agent()
    db_path = tmp_path / "state.db"
    _attach_real_session_db(parent, db_path, "sess-parent")
    _attach_real_session_db(child, db_path, "sess-child")
    parent_msgs = [
        {"role": "user", "content": "p-user"},
        {"role": "assistant", "content": "p-answer"},
    ]
    parent._flush_messages_to_session_db(parent_msgs)
    copies = [{k: v for k, v in m.items() if k != "_db_persisted"} for m in parent_msgs]
    copies.extend(
        [
            {"role": "user", "content": "c-user"},
            {"role": "assistant", "content": "c-answer"},
        ]
    )
    child._flush_messages_to_session_db(copies)
    reloaded = _durable_messages(db_path, "sess-child")
    assert [m.get("content") for m in reloaded] == [
        "p-user",
        "p-answer",
        "c-user",
        "c-answer",
    ]


def test_flush_stale_row_id_from_other_session_does_not_fill_child_blank(tmp_path):
    """A parent `_row_id` must INSERT in the child, not steal a blank tail."""
    parent = _make_agent()
    child = _make_agent()
    db_path = tmp_path / "state.db"
    _attach_real_session_db(parent, db_path, "sess-parent")
    _attach_real_session_db(child, db_path, "sess-child")
    parent_msgs = [
        {"role": "user", "content": "p-user"},
        {"role": "assistant", "content": "p-answer"},
    ]
    parent._flush_messages_to_session_db(parent_msgs)
    child_msgs = [
        {"role": "user", "content": "c-keep"},
        {"role": "assistant", "content": ""},
    ]
    child._flush_messages_to_session_db(child_msgs)
    copies = [{k: v for k, v in m.items() if k != "_db_persisted"} for m in parent_msgs]
    copies.extend(
        [
            {"role": "user", "content": "c-user"},
            {"role": "assistant", "content": "c-answer"},
        ]
    )
    child._flush_messages_to_session_db(copies)
    reloaded = _durable_messages(db_path, "sess-child")
    assert [m.get("content") for m in reloaded] == [
        "c-keep",
        "",
        "p-user",
        "p-answer",
        "c-user",
        "c-answer",
    ]


def test_flush_archived_same_session_row_id_fills_active_clone(tmp_path):
    """Watermark compaction clones the tail; stale `_row_id` must not win.

    ``archive_and_compact(..., watermark=...)`` archives the original
    concurrent-tail row and inserts a fresh active clone with a new id.
    The live dict still holds the archived id. A rewrite keyed only on
    ``id + session_id`` updates the inactive row, stamps persistence, and
    skips INSERT — reload then still sees the blank clone (#95514 P0).
    """
    from agent.turn_finalizer import finalize_turn

    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "sess-archived-row-id"
    db = _attach_real_session_db(agent, db_path, session_id)
    recovered = "Already streamed to the user."
    agent._current_streamed_assistant_text = recovered
    messages = [
        {"role": "user", "content": "summarize"},
        {"role": "assistant", "content": ""},
    ]
    agent._flush_messages_to_session_db(messages)
    stale_id = messages[-1]["_row_id"]
    user_id = messages[0]["_row_id"]
    assert isinstance(stale_id, int)
    assert stale_id > user_id

    db.archive_and_compact(
        session_id,
        compacted_messages=[{"role": "user", "content": "prior turns summarized"}],
        watermark=user_id,
    )

    finalize_turn(
        agent,
        final_response="",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="summarize",
        original_user_message="summarize",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    reloaded = _durable_messages(db_path, session_id)
    active_assistants = [m for m in reloaded if m.get("role") == "assistant"]
    assert active_assistants, "compaction clone should keep an active assistant"
    assert any(
        (m.get("content") or "").strip() == recovered for m in active_assistants
    )
    assert not any(
        not (m.get("content") or "").strip() for m in active_assistants
    )
    inactive = db.get_messages(session_id, include_inactive=True)
    archived = next(m for m in inactive if m.get("id") == stale_id)
    assert archived.get("active") in (0, False)
    assert messages[-1].get("_row_id") != stale_id


def test_flush_atomic_mixed_repair_and_append_rollback_on_failure(tmp_path, monkeypatch):
    """Mixed rewrite + append must be atomic: failure rolls back repair and stamps no markers.

    If a turn contains both a blank assistant repair and a subsequent new message,
    failure during the batch insert must roll back the assistant update and stamp
    no in-memory markers.
    """
    from hermes_state import SessionDB

    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "sess-atomic-rollback"
    db = _attach_real_session_db(agent, db_path, session_id)

    messages = [
        {"role": "user", "content": "summarize"},
        {"role": "assistant", "content": ""},
    ]
    agent._flush_messages_to_session_db(messages)
    orig_row_id = messages[-1]["_row_id"]
    assert isinstance(orig_row_id, int)

    # Now modify assistant message in memory and add a new message to the batch
    messages[-1]["content"] = "Recovered stream answer"
    messages[-1].pop("_db_persisted", None)
    new_tool_msg = {"role": "tool", "tool_call_id": "t1", "name": "terminal", "content": "ok"}
    messages.append(new_tool_msg)

    # Force a failure inside the batch write transaction on the second message
    orig_insert = SessionDB._insert_message_rows

    def _failing_insert(self_db, conn, sid, msgs):
        for msg in msgs:
            if msg.get("role") == "tool":
                raise RuntimeError("Simulated mid-batch persistence crash")
        return orig_insert(self_db, conn, sid, msgs)

    monkeypatch.setattr(SessionDB, "_insert_message_rows", _failing_insert)

    success = agent._flush_messages_to_session_db(messages)
    assert success is False or getattr(agent, "_incremental_persistence_failed", False) is True

    # Check that in-memory markers were NOT stamped
    assert messages[-1].get("_db_persisted") is not True
    assert new_tool_msg.get("_db_persisted") is not True

    # Check that the durable SQLite row was NOT updated (rolled back)
    durable = _durable_messages(db_path, session_id)
    assert len(durable) == 2
    assert durable[-1]["role"] == "assistant"
    assert (durable[-1].get("content") or "") == ""


def test_flush_concurrent_nonblank_winner_adopts_canonical_content(tmp_path):
    """A concurrent non-blank winner must be adopted without overwrite and synced to live dict."""
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "sess-concurrent-winner"
    db = _attach_real_session_db(agent, db_path, session_id)

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": ""},
    ]
    agent._flush_messages_to_session_db(messages)
    row_id = messages[-1]["_row_id"]
    assert isinstance(row_id, int)

    # Concurrent writer updates the row in SQLite to non-blank canonical content
    def _concurrent_write(conn):
        conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            (db._encode_content("Canonical winner answer from sibling"), row_id),
        )

    db._execute_write(_concurrent_write)

    # Live agent attempts to flush divergent content
    messages[-1]["content"] = "Divergent live agent answer"
    messages[-1].pop("_db_persisted", None)

    success = agent._flush_messages_to_session_db(messages)
    assert success is True

    # SQLite must still have canonical winner content
    durable = _durable_messages(db_path, session_id)
    assert durable[-1]["content"] == "Canonical winner answer from sibling"

    # Live dict must have adopted the canonical content and be marked persisted
    assert messages[-1]["content"] == "Canonical winner answer from sibling"
    assert messages[-1]["_db_persisted"] is True
    assert messages[-1]["_row_id"] == row_id

