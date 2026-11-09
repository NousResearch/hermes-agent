"""A tool call a ``pre_tool_call`` hook stopped must still show as a tool row.

A hook-blocked call returns before ``_begin_tool_execution``, so the executor emitted no lifecycle
signal at all: the user saw a spinner and then nothing, and could not tell "the agent never tried"
from "a guard stopped it" (#151276). These tests drive the real executor paths — no private helper
is called directly — and pin the events the surfaces consume, on both the sequential and the
concurrent route.

Two boundaries are deliberate and asserted here:

* ``tool_start_callback`` fires for a blocked call — it is the ID-keyed lane TUI/Desktop open a row
  from — but no checkpoint is taken (``tests/agent/test_run_agent.py::
  test_sequential_blocked_tool_skips_checkpoints_but_announces_start`` pins both halves), because a
  call that will not run must not snapshot the workspace.
* a guardrail/scope block keeps emitting nothing at all, as its own suite pins
  (``test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution``).
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

BLOCK_MESSAGE = "blocked by test policy"


def _make_tool_defs(*names: str) -> list[dict]:
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


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_agent(*tool_names: str, config: dict | None = None) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _record(agent):
    """Attach the three callbacks a surface wires and return their recorded calls."""
    seen = {"progress": [], "start": [], "complete": []}
    agent.tool_progress_callback = lambda *a, **k: seen["progress"].append((a, k))
    agent.tool_start_callback = lambda *a, **k: seen["start"].append((a, k))
    agent.tool_complete_callback = lambda *a, **k: seen["complete"].append((a, k))
    return seen


def _events(seen, event_type):
    return [(a, kw) for a, kw in seen["progress"] if a and a[0] == event_type]


def _run_sequential(agent, call, pre_hook=None):
    assistant = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    hooks = pre_hook or (lambda name, args, **kwargs: (None, None))
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=hooks):
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")
    return messages


def _blocked_sequential(agent, call):
    return _run_sequential(agent, call, lambda name, args, **kwargs: (BLOCK_MESSAGE, None))


def test_a_hook_blocked_call_opens_and_closes_a_row_with_the_hook_message():
    """The row the guard working needs: opened by ``tool.started``, closed by ``tool.completed``."""
    agent = _make_agent("web_search")
    seen = _record(agent)
    call = _mock_tool_call("web_search", json.dumps({"query": "x"}), "c-blocked")

    messages = _blocked_sequential(agent, call)

    started = _events(seen, "tool.started")
    completed = _events(seen, "tool.completed")
    assert [a[1] for a, _ in started] == ["web_search"]
    assert len(completed) == 1
    assert completed[0][1]["is_error"] is True
    # The row carries the hook's own message, so the surface can say WHY the call did not run.
    assert BLOCK_MESSAGE in completed[0][1]["result"]
    # Both start lanes carry the same stable id the completion uses, so TUI/Desktop can pair them.
    assert [a[:2] for a, _ in seen["start"]] == [("c-blocked", "web_search")]
    assert [a[:2] for a, _ in seen["complete"]] == [("c-blocked", "web_search")]
    assert BLOCK_MESSAGE in seen["complete"][0][0][3]
    # The model-facing side is unchanged: exactly one tool result, carrying the block message.
    assert len([m for m in messages if m["role"] == "tool"]) == 1


def test_a_hook_blocked_call_announces_a_start_but_takes_no_checkpoint():
    """Both start lanes fire (TUI/Desktop open the row from the callback); no snapshot is taken."""
    agent = _make_agent("web_search")
    seen = _record(agent)
    agent._checkpoint_mgr.enabled = True
    agent._checkpoint_mgr.ensure_checkpoint = MagicMock(side_effect=AssertionError("no checkpoint"))

    _blocked_sequential(agent, _mock_tool_call("web_search", json.dumps({"query": "x"}), "c-nocheck"))

    assert [a[:2] for a, _ in seen["start"]] == [("c-nocheck", "web_search")]
    agent._checkpoint_mgr.ensure_checkpoint.assert_not_called()


def test_an_allowed_call_emits_one_start_and_one_completion():
    """Negative control: the blocked lane must not add a second row to a call that really ran."""
    agent = _make_agent("web_search")
    seen = _record(agent)
    call = _mock_tool_call("web_search", json.dumps({"query": "x"}), "c-allowed")

    assistant = SimpleNamespace(content="", tool_calls=[call])
    messages = []
    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)), patch(
        "model_tools.handle_function_call", return_value=json.dumps({"ok": True}),
    ):
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    assert len(_events(seen, "tool.started")) == 1
    assert len(_events(seen, "tool.completed")) == 1
    assert _events(seen, "tool.completed")[0][1]["is_error"] is False
    assert [a[:2] for a, _ in seen["start"]] == [("c-allowed", "web_search")]
    assert [a[:2] for a, _ in seen["complete"]] == [("c-allowed", "web_search")]


def test_a_guardrail_block_keeps_its_no_row_contract():
    """Boundary of this fix: a hard-stopped call announces no row, as its own suite pins."""
    config = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {"exact_failure": 2, "same_tool_failure": 8, "idempotent_no_progress": 5},
        }
    }
    agent = _make_agent("web_search", config=config)
    args = {"query": "same"}
    for _ in range(3):
        agent._tool_guardrails.after_call("web_search", args, json.dumps({"error": "boom"}), failed=True)
    seen = _record(agent)

    messages = _run_sequential(agent, _mock_tool_call("web_search", json.dumps(args), "c-guarded"))

    assert seen["progress"] == []
    assert seen["start"] == []
    assert seen["complete"] == []
    assert len([m for m in messages if m["role"] == "tool"]) == 1


def test_the_announced_row_is_released_once_the_call_settles():
    """The bookkeeping must not leak: the second blocked call still renders, nothing is left over."""
    agent = _make_agent("web_search")
    seen = _record(agent)

    _blocked_sequential(agent, _mock_tool_call("web_search", json.dumps({"query": "a"}), "c-one"))
    _blocked_sequential(agent, _mock_tool_call("web_search", json.dumps({"query": "b"}), "c-two"))

    assert [a[:2] for a, _ in seen["complete"]] == [("c-one", "web_search"), ("c-two", "web_search")]
    assert getattr(agent, "_visible_blocked_tool_calls", set()) == set()


def test_a_blocked_call_in_a_concurrent_batch_keeps_its_own_row():
    """The concurrent route shares the same policy seam: a blocked slot gets one row, not zero or two."""
    agent = _make_agent("web_search")
    seen = _record(agent)
    calls = [
        _mock_tool_call("web_search", json.dumps({"query": "allowed"}), "c-ok"),
        _mock_tool_call("web_search", json.dumps({"query": "blocked"}), "c-blocked"),
    ]
    assistant = SimpleNamespace(content="", tool_calls=calls)
    messages = []

    def pre_hook(name, args, **kwargs):
        if args.get("query") == "blocked":
            return BLOCK_MESSAGE, None
        return None, None

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=pre_hook), patch(
        "model_tools.handle_function_call", return_value=json.dumps({"ok": True}),
    ):
        agent._execute_tool_calls(assistant, messages, "task-1")

    # One start per call, blocked slot included: the TUI/Desktop lane must find its row to complete.
    assert sorted(a[:2] for a, _ in seen["start"]) == [("c-blocked", "web_search"), ("c-ok", "web_search")]
    assert [a[:2] for a, _ in seen["complete"]] == [("c-ok", "web_search"), ("c-blocked", "web_search")]
    completed = _events(seen, "tool.completed")
    assert len(completed) == 2
    assert [kw["is_error"] for _, kw in completed] == [False, True]
    assert len([m for m in messages if m["role"] == "tool"]) == 2
