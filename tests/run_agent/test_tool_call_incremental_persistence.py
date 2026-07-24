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

import concurrent.futures
import copy
import hashlib
import json
import time
from types import SimpleNamespace
from pathlib import Path
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import make_tool_result_message
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
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


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

    with (
        patch("run_agent.handle_function_call", side_effect=_fake_dispatch) as disp,
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


@pytest.mark.parametrize(
    ("replacement_label", "expected_label"),
    [(None, "old-turn"), ("replacement-turn", "replacement-turn")],
)
def test_concurrent_timeout_revokes_detached_mutation_evidence(
    tmp_path,
    replacement_label,
    expected_label,
):
    """A timed-out worker cannot publish late evidence in any turn state."""

    from agent.file_mutation_verifier import FileContentFingerprint

    agent = _make_agent()
    getattr(agent, "valid_tool_names").add("terminal")
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    target_key = str(target.resolve())

    def fingerprint() -> FileContentFingerprint:
        return FileContentFingerprint(
            kind="file",
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )

    def reset_with_failure(label: str) -> None:
        agent._turn_failed_file_mutations = {
            target_key: {
                "tool": "patch",
                "error_preview": label,
                "display_path": target_key,
                "resolved_path": target_key,
                "task_id": "task-1",
                "generation": 1,
                "fingerprint": fingerprint(),
                "fingerprint_deferred": False,
            }
        }
        agent._turn_file_mutation_paths = set()
        agent._turn_file_mutation_lock = threading.Lock()
        agent._turn_file_mutation_generation = 1
        agent._turn_file_mutation_epoch = object()

    reset_with_failure("old-turn")
    release = threading.Event()
    recorded = threading.Event()
    messages: list = []
    tool_call = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "late fallback"}),
        call_id="c-timeout",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    def late_invoke(*_args, **_kwargs):
        assert release.wait(timeout=5)
        target.write_text("late-old-turn-change\n")
        return json.dumps({"output": "ok", "exit_code": 0})

    real_record = agent._record_file_mutation_result

    def recording_wrapper(*args, **kwargs):
        try:
            return real_record(*args, **kwargs)
        finally:
            recorded.set()

    agent._flush_messages_to_session_db = MagicMock()
    try:
        with (
            patch.object(agent, "_invoke_tool", side_effect=late_invoke),
            patch.object(
                agent,
                "_record_file_mutation_result",
                side_effect=recording_wrapper,
            ),
            patch(
                "agent.tool_executor._resolve_concurrent_tool_timeout",
                return_value=0.05,
            ),
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
        ):
            agent._execute_tool_calls_concurrent(
                assistant_message,
                messages,
                "task-1",
            )
            assert "timed out" in messages[0]["content"]

            if replacement_label is not None:
                # Simulate build_turn_context replacing state before the
                # detached old worker completes. Both generations intentionally
                # equal one so only call identity can distinguish them.
                reset_with_failure(replacement_label)
            release.set()
            assert recorded.wait(timeout=5)
    finally:
        release.set()

    assert agent._turn_failed_file_mutations[target_key]["error_preview"] == (
        expected_label
    )
    assert agent._turn_file_mutation_paths == set()


def test_concurrent_timeout_keeps_real_result_while_verifier_record_is_unfinished():
    agent = _make_agent()
    getattr(agent, "valid_tool_names").add("terminal")
    tool_call = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "completed fallback"}),
        call_id="c-result-before-verifier",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []
    record_entered = threading.Event()
    release_record = threading.Event()
    record_finished = threading.Event()

    def blocked_record(*_args, **_kwargs):
        record_entered.set()
        try:
            assert release_record.wait(timeout=5)
        finally:
            record_finished.set()

    try:
        with (
            patch.object(
                agent,
                "_invoke_tool",
                return_value=json.dumps({"output": "real-result", "exit_code": 0}),
            ),
            patch.object(
                agent,
                "_record_file_mutation_result",
                side_effect=blocked_record,
            ),
            patch(
                "agent.tool_executor._resolve_concurrent_tool_timeout",
                return_value=0.05,
            ),
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
        ):
            agent._execute_tool_calls_concurrent(
                assistant_message,
                messages,
                "task-1",
            )
        assert record_entered.is_set()
        assert len(messages) == 1
        assert "real-result" in messages[0]["content"]
        assert "timed out" not in messages[0]["content"]
    finally:
        release_record.set()
        assert record_finished.wait(timeout=5)


def test_concurrent_dispatch_propagates_verifier_execution_bounds():
    """Workers receive both the batch deadline and live interrupt predicate."""

    agent = _make_agent()
    getattr(agent, "valid_tool_names").add("terminal")
    tool_call = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "bounded fallback"}),
        call_id="c-bounds",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []
    captured: list[dict] = []

    def capture_prepare(*_args, **kwargs):
        captured.append(kwargs)

    with (
        patch.object(agent, "_prepare_file_mutation_verifier_call", side_effect=capture_prepare),
        patch.object(
            agent,
            "_invoke_tool",
            return_value=json.dumps({"output": "ok", "exit_code": 0}),
        ),
        patch(
            "agent.tool_executor._resolve_concurrent_tool_timeout",
            return_value=5.0,
        ),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(
            assistant_message,
            messages,
            "task-1",
        )

    assert len(captured) == 1
    assert isinstance(captured[0].get("deadline"), float)
    cancel_check = captured[0].get("cancel_check")
    assert callable(cancel_check)
    assert cancel_check() is False
    agent._interrupt_requested = True
    assert cancel_check() is True


def test_sequential_dispatch_propagates_live_interrupt_and_defers_probe_activation():
    agent = _make_agent()
    getattr(agent, "valid_tool_names").add("terminal")
    tool_call = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "bounded fallback"}),
        call_id="s-bounds",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []
    captured: list[dict] = []

    def capture_prepare(*_args, **kwargs):
        captured.append(kwargs)

    with (
        patch.object(
            agent,
            "_prepare_file_mutation_verifier_call",
            side_effect=capture_prepare,
        ),
        patch(
            "run_agent.handle_function_call",
            return_value=json.dumps({"output": "ok", "exit_code": 0}),
        ),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "task-1",
        )

    assert len(captured) == 1
    assert captured[0].get("defer_probe_activation") is True
    cancel_check = captured[0].get("cancel_check")
    assert callable(cancel_check)
    assert cancel_check() is False
    agent._interrupt_requested = True
    assert cancel_check() is True


@pytest.mark.parametrize("tool_name", ["terminal", "execute_code"])
def test_sequential_denied_fallback_does_not_read_deferred_target(
    tmp_path,
    tool_name,
):
    from agent.file_mutation_verifier import FileContentFingerprint

    agent = _make_agent()
    getattr(agent, "valid_tool_names").add(tool_name)
    target = tmp_path / "sensitive.txt"
    target.write_text("secret\n")
    target_key = str(target.resolve())
    setattr(
        agent,
        "_turn_failed_file_mutations",
        {
            target_key: {
                "tool": "write_file",
                "error_preview": "policy denied",
                "display_path": target_key,
                "resolved_path": target_key,
                "task_id": "task-1",
                "generation": 1,
                "fingerprint": None,
                "fingerprint_deferred": True,
            }
        },
    )
    setattr(agent, "_turn_file_mutation_paths", set())
    setattr(agent, "_turn_file_mutation_lock", threading.Lock())
    setattr(agent, "_turn_file_mutation_generation", 1)
    setattr(agent, "_turn_file_mutation_epoch", object())
    fingerprinted: list[str] = []

    def fingerprint_spy(path, _task_id="default"):
        fingerprinted.append(path)
        return FileContentFingerprint("sha256", "a" * 64)

    if tool_name == "terminal":
        arguments = {"command": "denied fallback"}
        denied = json.dumps(
            {"output": "", "exit_code": -1, "status": "blocked", "error": "denied"}
        )
    else:
        arguments = {"code": "print('denied fallback')"}
        denied = json.dumps(
            {"status": "error", "error": "denied", "tool_calls_made": 0}
        )
    tool_call = _mock_tool_call(
        name=tool_name,
        arguments=json.dumps(arguments),
        call_id=f"denied-{tool_name}",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []

    with (
        patch(
            "tools.file_tools._fingerprint_resolved_file_content",
            side_effect=fingerprint_spy,
        ),
        patch("run_agent.handle_function_call", return_value=denied),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "task-1",
        )

    assert fingerprinted == []
    assert getattr(agent, "_turn_failed_file_mutations")[target_key][
        "fingerprint_deferred"
    ] is True
    assert getattr(agent, "_turn_file_mutation_paths") == set()


def test_sequential_interrupt_after_probe_activation_cannot_clear_failure(tmp_path):
    from agent.file_mutation_verifier import (
        FileContentFingerprint,
        activate_pending_file_mutation_verifier_call,
    )

    agent = _make_agent()
    getattr(agent, "valid_tool_names").add("terminal")
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    target_key = str(target.resolve())

    def fingerprint() -> FileContentFingerprint:
        return FileContentFingerprint(
            kind="sha256",
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )

    setattr(
        agent,
        "_turn_failed_file_mutations",
        {
            target_key: {
                "tool": "patch",
                "error_preview": "must remain",
                "display_path": target_key,
                "resolved_path": target_key,
                "task_id": "task-1",
                "generation": 1,
                "fingerprint": fingerprint(),
                "fingerprint_deferred": False,
            }
        },
    )
    setattr(agent, "_turn_file_mutation_paths", set())
    setattr(agent, "_turn_file_mutation_lock", threading.Lock())
    setattr(agent, "_turn_file_mutation_generation", 1)
    setattr(agent, "_turn_file_mutation_epoch", object())
    tool_call = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "interrupted fallback"}),
        call_id="s-interrupt",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages: list = []

    def interrupted_dispatch(*_args, **_kwargs):
        activate_pending_file_mutation_verifier_call()
        target.write_text("after-interrupt\n")
        agent._interrupt_requested = True
        return json.dumps({"output": "ok", "exit_code": 0})

    with (
        patch("run_agent.handle_function_call", side_effect=interrupted_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "task-1",
        )

    assert getattr(agent, "_turn_failed_file_mutations")[target_key][
        "error_preview"
    ] == "must remain"
    assert getattr(agent, "_turn_file_mutation_paths") == set()


@pytest.mark.parametrize("boundary", ["deadline", "interrupt"])
def test_worker_cannot_publish_between_execution_boundary_and_main_revocation(
    tmp_path,
    boundary,
):
    """Verifier checks the boundary itself, not only main-thread revocation."""

    from agent.file_mutation_verifier import FileContentFingerprint

    agent = _make_agent()
    getattr(agent, "valid_tool_names").add("terminal")
    target = tmp_path / "app.txt"
    target.write_text("before\n")
    target_key = str(target.resolve())

    def fingerprint() -> FileContentFingerprint:
        return FileContentFingerprint(
            kind="file",
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )

    setattr(
        agent,
        "_turn_failed_file_mutations",
        {
            target_key: {
                "tool": "patch",
                "error_preview": "must remain",
                "display_path": target_key,
                "resolved_path": target_key,
                "task_id": "task-1",
                "generation": 1,
                "fingerprint": fingerprint(),
                "fingerprint_deferred": False,
            }
        },
    )
    setattr(agent, "_turn_file_mutation_paths", set())
    setattr(agent, "_turn_file_mutation_lock", threading.Lock())
    setattr(agent, "_turn_file_mutation_generation", 1)
    setattr(agent, "_turn_file_mutation_epoch", object())

    invoked = threading.Event()
    release = threading.Event()
    recorded = threading.Event()
    messages: list = []
    tool_call = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({"command": "boundary race"}),
        call_id=f"c-{boundary}",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])

    def late_invoke(*_args, **_kwargs):
        invoked.set()
        assert release.wait(timeout=5)
        target.write_text(f"after-{boundary}\n")
        return json.dumps({"output": "ok", "exit_code": 0})

    real_record = agent._record_file_mutation_result

    def recording_wrapper(*args, **kwargs):
        try:
            return real_record(*args, **kwargs)
        finally:
            recorded.set()

    real_wait = concurrent.futures.wait
    wait_calls = 0

    def boundary_losing_wait(futures, timeout=None):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            assert invoked.wait(timeout=5)
            if boundary == "deadline":
                time.sleep(0.08)
            else:
                agent._interrupt_requested = True
            release.set()
            assert recorded.wait(timeout=5)
            # Deliberately let the worker publish before the main executor gets
            # a chance to revoke its active_event.
            return set(), set(futures)
        return real_wait(futures, timeout=timeout)

    agent._flush_messages_to_session_db = MagicMock()
    try:
        with (
            patch.object(agent, "_invoke_tool", side_effect=late_invoke),
            patch.object(
                agent,
                "_record_file_mutation_result",
                side_effect=recording_wrapper,
            ),
            patch(
                "agent.tool_executor._resolve_concurrent_tool_timeout",
                return_value=0.05 if boundary == "deadline" else 5.0,
            ),
            patch(
                "agent.tool_executor.concurrent.futures.wait",
                side_effect=boundary_losing_wait,
            ),
            patch(
                "agent.tool_executor.maybe_persist_tool_result",
                side_effect=lambda **kwargs: kwargs["content"],
            ),
        ):
            agent._execute_tool_calls_concurrent(
                assistant_message,
                messages,
                "task-1",
            )
    finally:
        release.set()

    assert getattr(agent, "_turn_failed_file_mutations")[target_key][
        "error_preview"
    ] == "must remain"
    assert getattr(agent, "_turn_file_mutation_paths") == set()
