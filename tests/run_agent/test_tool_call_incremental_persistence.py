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

    * sequential -> ``run_agent.handle_function_call`` (tool_executor ~1256/1298)
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
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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


# ---------------------------------------------------------------------------
# Contract 2: the SEQUENTIAL path flushes each tool result immediately, BEFORE
# the next tool dispatches.  Dispatch goes through run_agent.handle_function_call
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

    with patch("run_agent.handle_function_call", side_effect=_fake_dispatch) as disp:
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
        patch("run_agent.handle_function_call", return_value="repository result")
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
        patch("run_agent.handle_function_call", return_value="repository result")
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
        patch("run_agent.handle_function_call", return_value="second result") as dispatch,
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

    with patch.object(agent, "_invoke_tool", side_effect=_fake_invoke) as inv:
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
