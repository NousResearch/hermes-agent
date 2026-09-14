"""Sequential tool calls recover when one dispatch never returns."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import execute_tool_calls_sequential
from run_agent import AIAgent
from tools.clarify_gateway import resolve_clarify_timeout


@pytest.fixture(autouse=True)
def _deterministic_worker_start(monkeypatch):
    """Deadline must race the TOOL, not the scheduler or the preamble.

    The sequential timeout path computes ``deadline = now + timeout_s``
    immediately after ``executor.submit()``. Two races made a sub-second
    test deadline flaky on loaded CI workers (three slice-4 reds on
    unrelated PRs, Aug 2026):

    1. Thread-start latency — the pool thread took longer than the whole
       deadline to START; the future was cancelled before the tool ever
       dispatched. Killed here by blocking submit() until the worker
       callable has begun.
    2. Middleware preamble latency — after the worker starts, argument
       parsing / hooks / cold imports run BEFORE ``handle_function_call``;
       when the deadline expired in that window, the timeout interrupt made
       the middleware return without dispatching (``first_started`` unset).
       Killed by using a 1.0s deadline: tight enough to keep the suite
       fast, ~7x the worst observed preamble.

    Together the countdown starts only once the worker is running and has
    generous headroom to reach the dispatch — which is the behavior these
    tests mean to pin.
    """
    import tools.daemon_pool as daemon_pool

    real_executor = daemon_pool.DaemonThreadPoolExecutor

    class _StartSyncedExecutor(real_executor):
        def submit(self, fn, *args, **kwargs):
            begun = threading.Event()

            def _traced(*fa, **fk):
                begun.set()
                return fn(*fa, **fk)

            future = super().submit(_traced, *args, **kwargs)
            # Bounded: a wedged pool must fail the test loudly, not hang it.
            assert begun.wait(timeout=10), "tool worker thread never started"
            return future

    monkeypatch.setattr(daemon_pool, "DaemonThreadPoolExecutor", _StartSyncedExecutor)
    yield


def _make_agent(tmp_path: Path) -> AIAgent:
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_extract",
                        "description": "test tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("run_agent._hermes_home", tmp_path),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._flush_messages_to_session_db = MagicMock(return_value=True)
    agent._append_guardrail_observation = MagicMock(
        side_effect=lambda _name, _args, result, **_kwargs: result
    )
    agent._record_file_mutation_result = MagicMock()
    agent._subdirectory_hints.check_tool_call = MagicMock(return_value="")
    agent._tool_result_content_for_active_model = MagicMock(
        side_effect=lambda _name, result: result
    )
    return agent


def _tool_call(call_id: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="web_extract", arguments="{}"),
    )


def _clarify_call(call_id: str = "clarify-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="clarify",
            arguments='{"question": "Pick one?", "choices": ["A", "B"]}',
        ),
    )


def test_sequential_tool_timeout_emits_result_and_continues(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path)
    first_started = threading.Event()
    release_first = threading.Event()
    dispatched: list[str] = []
    terminal_events: list[dict] = []

    def _dispatch(_name, _args, _task_id, *, tool_call_id, **_kwargs):
        dispatched.append(tool_call_id)
        if tool_call_id == "hung":
            first_started.set()
            release_first.wait()
            return "late result"
        return "second result"

    def _capture_terminal_event(*_args, **kwargs):
        terminal_events.append(kwargs)

    calls = [_tool_call("hung"), _tool_call("next")]
    assistant = SimpleNamespace(tool_calls=calls)
    messages: list[dict] = []
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "1.0")

    started = time.monotonic()
    try:
        with (
            patch("model_tools.handle_function_call", side_effect=_dispatch),
            patch(
                "agent.tool_executor._emit_terminal_post_tool_call",
                side_effect=_capture_terminal_event,
            ),
        ):
            execute_tool_calls_sequential(agent, assistant, messages, "task")
    finally:
        release_first.set()

    assert first_started.is_set()
    assert time.monotonic() - started < 10.0
    assert dispatched == ["hung", "next"]
    assert [message["tool_call_id"] for message in messages] == ["hung", "next"]
    assert "timed out after 1.0s" in messages[0]["content"]
    assert messages[0]["effect_disposition"] == "unknown"
    assert messages[1]["content"] == "second result"
    timeout_events = [event for event in terminal_events if event.get("error_type") == "tool_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0]["status"] == "timeout"
    agent._flush_messages_to_session_db.assert_called()


def test_sequential_tool_timeout_suppresses_late_terminal_event(tmp_path, monkeypatch):
    import hermes_cli.lifecycle as lifecycle
    import model_tools

    agent = _make_agent(tmp_path)
    release_first = threading.Event()
    first_returned = threading.Event()
    dispatch_count = 0
    terminal_events: list[dict] = []

    def _dispatch(_name, _args, **_kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            release_first.wait()
            first_returned.set()
            return "late result"
        return "second result"

    calls = [_tool_call("hung"), _tool_call("next")]
    messages: list[dict] = []
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "1.0")

    try:
        with (
            patch.object(model_tools.registry, "dispatch", side_effect=_dispatch),
            patch.object(lifecycle, "has_hook", return_value=True),
            patch.object(
                lifecycle,
                "invoke_hook",
                side_effect=lambda hook, **kwargs: (
                    terminal_events.append(kwargs) if hook == "post_tool_call" else []
                ),
            ),
        ):
            execute_tool_calls_sequential(
                agent, SimpleNamespace(tool_calls=calls), messages, "task"
            )
            release_first.set()
            assert first_returned.wait(timeout=1)
    finally:
        release_first.set()

    assert [(event["tool_call_id"], event.get("error_type")) for event in terminal_events] == [
        ("hung", "tool_timeout"),
        ("next", None),
    ]


def test_sequential_tool_interrupt_hides_lifecycle_cancel_detail(tmp_path, monkeypatch):
    from agent.subagent_lifecycle import (
        SubagentLaunchRequest,
        SubagentLifecycleService,
        SubagentState,
    )

    agent = _make_agent(tmp_path)
    agent._subagent_id = "sa-lifecycle-cancel-output"
    agent._delegate_role = "leaf"
    agent._delegate_depth = 1
    first_started = threading.Event()
    release_first = threading.Event()
    terminal_events: list[dict] = []

    def _dispatch(*_args, **_kwargs):
        first_started.set()
        release_first.wait()
        return "late result"

    messages: list[dict] = []

    def _run_child(*_args, **_kwargs):
        execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[_tool_call("hung")]),
            messages,
            "task",
        )
        return {
            "status": "interrupted",
            "summary": None,
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "30")
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: agent,
    )
    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", _run_child)
    lifecycle = SubagentLifecycleService(
        lambda: SimpleNamespace(
            session_id="parent-lifecycle-cancel-output",
            enabled_toolsets=["file"],
        )
    )

    try:
        with (
            patch("model_tools.handle_function_call", side_effect=_dispatch),
            patch(
                "agent.tool_executor._emit_terminal_post_tool_call",
                side_effect=lambda *_args, **kwargs: terminal_events.append(kwargs),
            ),
        ):
            handle = lifecycle.launch(SubagentLaunchRequest(goal="cancel output test"))
            assert first_started.wait(timeout=5)
            assert lifecycle.cancel(
                handle,
                reason="PRIVATE_LIFECYCLE_REASON_DO_NOT_COPY",
            ).accepted
            assert lifecycle.wait(handle, timeout_seconds=5).state is SubagentState.CANCELLED
    finally:
        release_first.set()

    assert "subagent cancellation requested" in messages[0]["content"]
    assert "PRIVATE_LIFECYCLE_REASON_DO_NOT_COPY" not in messages[0]["content"]
    assert "user interrupt" not in messages[0]["content"]
    assert "PRIVATE_LIFECYCLE_REASON_DO_NOT_COPY" not in terminal_events[0]["error_message"]
    assert terminal_events[0]["error_type"] == "tool_interrupted"


@pytest.mark.parametrize(
    "clarify_timeout",
    [resolve_clarify_timeout({}), 0],
    ids=["default-3600s", "unlimited"],
)
def test_sequential_timeout_does_not_cut_clarify_human_wait(
    tmp_path, monkeypatch, clarify_timeout
):
    """Clarify waits on a human; the generic sequential deadline must not fire.

    Default ``agent.clarify_timeout`` is 3600s; ``<= 0`` is unlimited. Both
    outlast ``HERMES_CONCURRENT_TOOL_TIMEOUT_S`` (default 420s).
    """
    agent = _make_agent(tmp_path)
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "1.0")
    monkeypatch.setattr(
        "tools.clarify_gateway.get_clarify_timeout",
        lambda: clarify_timeout,
    )

    def _callback(question, choices, multi_select=False):
        # Must OUTLAST the 1.0s sequential deadline — the test proves the
        # generic timeout never cuts a human clarify wait.
        time.sleep(1.3)
        return "A"

    agent.clarify_callback = _callback
    terminal_events: list[dict] = []

    def _dispatch(_name, _args, _task_id, *, tool_call_id, **_kwargs):
        return "second result"

    def _capture_terminal_event(*_args, **kwargs):
        terminal_events.append(kwargs)

    messages: list[dict] = []
    started = time.monotonic()
    with (
        patch("model_tools.handle_function_call", side_effect=_dispatch),
        patch(
            "agent.tool_executor._emit_terminal_post_tool_call",
            side_effect=_capture_terminal_event,
        ),
    ):
        execute_tool_calls_sequential(
            agent,
            SimpleNamespace(tool_calls=[_clarify_call(), _tool_call("next")]),
            messages,
            "task",
        )

    assert time.monotonic() - started < 10.0
    assert [message["tool_call_id"] for message in messages] == ["clarify-1", "next"]
    payload = json.loads(messages[0]["content"])
    assert payload["user_response"] == "A"
    assert "timed out" not in messages[0]["content"]
    assert messages[1]["content"] == "second result"
    assert not any(event.get("error_type") == "tool_timeout" for event in terminal_events)
