"""Behavior contract for the shared per-tool lifecycle in ``agent.tool_executor``.

``execute_one_tool`` is the single funnel every runtime must use to run a
model tool call — the sequential dispatcher, the concurrent dispatcher, and
the ``claude_agent_sdk`` MCP bridge.  These tests pin the *invariants* of that
funnel (a wrapper is consulted, a denial short-circuits, an interrupt cancels,
the task id reaches the handler) rather than the current wording of any
message or the internal call counts of private helpers, so the extraction can
keep moving without the suite fighting it.
"""

import json
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import (
    ToolExecutionOutcome,
    execute_one_tool,
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
    finalize_tool_outcome,
)
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _tool_defs(*names: str) -> list[dict]:
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


def _make_agent(*tool_names: str, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _tool_call(name="web_search", args=None, call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(
            name=name, arguments=json.dumps(args if args is not None else {})
        ),
    )


def _assistant(*tool_calls):
    return SimpleNamespace(content="", tool_calls=list(tool_calls))


def _dispatch(mode):
    return (
        execute_tool_calls_sequential
        if mode == "sequential"
        else execute_tool_calls_concurrent
    )


BOTH_PATHS = pytest.mark.parametrize("mode", ["sequential", "concurrent"])


def _assert_role_alternation(messages: list, baseline: int = 0) -> None:
    """Alternation invariant for what tool execution appends.

    A batch legitimately produces one ``tool`` message per tool_call, so a run
    of ``tool`` rows is valid; two ``user`` or two ``assistant`` rows in a row
    is not, and no ``user`` turn may be injected mid-loop.
    """
    for message in messages[baseline:]:
        assert message.get("role") != "user", (
            "tool execution must not inject a user message"
        )
    previous = None
    for message in messages:
        role = message.get("role")
        if role in {"user", "assistant"}:
            assert role != previous, (
                f"consecutive {role!r} messages break alternation"
            )
        previous = role


class _Observed:
    """Records the observable effect of each wrapper around the handler."""

    def __init__(self):
        self.plugin_block = []
        self.guardrail_before = []
        self.approval = []
        self.tool_started = []
        self.tool_completed = []
        self.tool_start_callback = []
        self.checkpoints = []
        self.post_hook = []
        self.dispatched = []


def _instrument(agent, observed: _Observed, *, handler=None, plugin_block=None):
    """Patch every wrapper's observation point around one tool dispatch.

    Dispatch is intercepted at ``registry.dispatch`` rather than at
    ``handle_function_call`` so the real dispatcher — and therefore the real
    approval layer and the real ``post_tool_call`` emission — still runs.
    """

    def _progress(event, name, preview, args, **kwargs):
        if event == "tool.started":
            observed.tool_started.append(name)
        elif event == "tool.completed":
            observed.tool_completed.append(name)

    agent.tool_progress_callback = _progress
    agent.tool_start_callback = lambda call_id, name, args: (
        observed.tool_start_callback.append(name)
    )
    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed.checkpoints.append(
            (path, reason)
        ),
    )

    original_before_call = agent._tool_guardrails.before_call

    def _before_call(name, args):
        observed.guardrail_before.append(name)
        return original_before_call(name, args)

    def _plugin(name, args, **kwargs):
        observed.plugin_block.append(name)
        return plugin_block

    def _approval(name, args):
        observed.approval.append(name)
        return None

    def _post_hook(**kwargs):
        observed.post_hook.append(kwargs.get("function_name"))

    def _dispatch_tool(name, args, **kwargs):
        observed.dispatched.append((name, dict(args), kwargs.get("task_id")))
        if handler is not None:
            return handler(name, args, **kwargs)
        return json.dumps({"ok": True})

    patches = (
        patch("hermes_cli.plugins.resolve_pre_tool_block", side_effect=_plugin),
        patch.object(agent._tool_guardrails, "before_call", side_effect=_before_call),
        patch("model_tools._emit_post_tool_call_hook", side_effect=_post_hook),
        patch("model_tools.registry.dispatch", side_effect=_dispatch_tool),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=_approval,
        ),
    )

    stack = ExitStack()
    for one in patches:
        stack.enter_context(one)
    return stack


def _hard_stop_config() -> dict:
    return {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }


def _seed_guardrail_block(agent, tool_name: str, args: dict) -> None:
    for _ in range(2):
        agent._tool_guardrails.after_call(
            tool_name, args, json.dumps({"error": "boom"}), failed=True
        )


# ---------------------------------------------------------------------------
# Every wrapper runs, exactly once, on both dispatchers
# ---------------------------------------------------------------------------


@BOTH_PATHS
def test_tool_call_passes_through_every_wrapper_exactly_once(mode):
    agent = _make_agent("write_file")
    args = {"path": "/tmp/hermes-lifecycle.txt", "content": "hi"}
    call = _tool_call("write_file", args, "c-1")
    messages: list = []
    observed = _Observed()

    with _instrument(agent, observed):
        _dispatch(mode)(agent, _assistant(call), messages, "task-lifecycle")

    # Each wrapper is consulted for this call, and only once.
    assert observed.plugin_block == ["write_file"]
    assert observed.guardrail_before == ["write_file"]
    assert observed.tool_started == ["write_file"]
    assert observed.tool_completed == ["write_file"]
    assert observed.tool_start_callback == ["write_file"]
    assert observed.approval == ["write_file"]
    assert observed.post_hook == ["write_file"]
    assert len(observed.dispatched) == 1
    # A file-mutating tool checkpoints the path it is about to touch.
    assert [reason for _path, reason in observed.checkpoints] == [
        "before write_file"
    ]
    # Exactly one tool result, matched to the model's tool_call_id.
    assert [m["tool_call_id"] for m in messages] == ["c-1"]
    _assert_role_alternation(messages)


@BOTH_PATHS
def test_effective_task_id_reaches_the_handler(mode):
    """Environment routing (local/Docker/SSH/Modal/...) rides on the task id."""
    agent = _make_agent("web_search")
    messages: list = []
    observed = _Observed()

    with _instrument(agent, observed):
        _dispatch(mode)(
            agent, _assistant(_tool_call("web_search", {"query": "x"})),
            messages, "task-routed-env",
        )

    assert [task_id for _n, _a, task_id in observed.dispatched] == ["task-routed-env"]


# ---------------------------------------------------------------------------
# Denials short-circuit
# ---------------------------------------------------------------------------


@BOTH_PATHS
def test_approval_denial_short_circuits_execution(mode):
    agent = _make_agent("terminal")
    messages: list = []
    observed = _Observed()

    with _instrument(agent, observed, plugin_block="denied by policy"):
        _dispatch(mode)(
            agent, _assistant(_tool_call("terminal", {"command": "ls"}, "c-deny")),
            messages, "task-deny",
        )

    assert observed.dispatched == [], "a denied call must never reach the handler"
    assert observed.tool_started == []
    assert observed.checkpoints == []
    # The denial is reported back to the model on the denied call's own id.
    assert [m["tool_call_id"] for m in messages] == ["c-deny"]
    assert "denied by policy" in messages[0]["content"]
    _assert_role_alternation(messages)


@BOTH_PATHS
def test_guardrail_block_prevents_the_handler_from_running(mode):
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "looping"}
    _seed_guardrail_block(agent, "web_search", args)
    messages: list = []
    observed = _Observed()

    with _instrument(agent, observed):
        _dispatch(mode)(
            agent, _assistant(_tool_call("web_search", args, "c-blocked")),
            messages, "task-guardrail",
        )

    assert observed.guardrail_before == ["web_search"]
    assert observed.dispatched == [], "a guardrail block must pre-empt dispatch"
    assert [m["tool_call_id"] for m in messages] == ["c-blocked"]
    _assert_role_alternation(messages)


# ---------------------------------------------------------------------------
# Interrupt semantics
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_mid_execution_yields_a_cancelled_outcome():
    agent = _make_agent("terminal")
    agent.interrupt = MagicMock()

    def _boom(*_args, **_kwargs):
        raise KeyboardInterrupt

    with (
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value=None),
        patch("run_agent.handle_function_call", side_effect=_boom),
    ):
        outcome = execute_one_tool(
            agent, _tool_call("terminal", {"command": "sleep 60"}), "task-int",
        )

    assert outcome.cancelled is True
    assert outcome.is_error is True
    assert json.loads(outcome.result)["status"] == "cancelled"
    agent.interrupt.assert_called_once()


@BOTH_PATHS
def test_interrupt_mid_batch_keeps_one_result_per_call_and_valid_alternation(mode):
    agent = _make_agent("web_search")
    calls = [
        _tool_call("web_search", {"query": "first"}, "c-a"),
        _tool_call("web_search", {"query": "second"}, "c-b"),
    ]
    messages: list = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    observed = _Observed()

    def _interrupting_handler(name, args, **kwargs):
        agent._interrupt_requested = True
        return json.dumps({"ok": True})

    with _instrument(agent, observed, handler=_interrupting_handler):
        _dispatch(mode)(agent, _assistant(*calls), messages, "task-int-batch")

    tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert tool_ids == ["c-a", "c-b"], "every tool_call_id must be answered once"
    _assert_role_alternation(messages, baseline=2)


# ---------------------------------------------------------------------------
# Output budgeting
# ---------------------------------------------------------------------------


def test_oversized_result_is_budgeted_identically_through_both_callers():
    oversized = "x" * 400_000
    contents = {}

    for mode in ("sequential", "concurrent"):
        agent = _make_agent("web_search")
        messages: list = []
        with (
            patch("hermes_cli.plugins.resolve_pre_tool_block", return_value=None),
            patch("run_agent.handle_function_call", return_value=oversized),
        ):
            _dispatch(mode)(
                agent, _assistant(_tool_call("web_search", {"query": "big"}, "c-big")),
                messages, "task-budget",
            )
        assert [m["tool_call_id"] for m in messages] == ["c-big"]
        contents[mode] = messages[0]["content"]

    assert len(contents["sequential"]) < len(oversized), (
        "an oversized result must be budgeted, not passed through whole"
    )
    assert contents["sequential"] == contents["concurrent"]


# ---------------------------------------------------------------------------
# Cross-path equivalence
# ---------------------------------------------------------------------------


def test_both_callers_produce_equivalent_per_tool_observable_effects():
    args = {"path": "/tmp/hermes-equivalence.txt", "content": "same"}
    snapshots = {}

    for mode in ("sequential", "concurrent"):
        agent = _make_agent("write_file")
        messages: list = []
        observed = _Observed()
        with _instrument(agent, observed):
            _dispatch(mode)(
                agent, _assistant(_tool_call("write_file", args, "c-eq")),
                messages, "task-eq",
            )
        snapshots[mode] = {
            "plugin_block": observed.plugin_block,
            "guardrail_before": observed.guardrail_before,
            "tool_started": observed.tool_started,
            "tool_completed": observed.tool_completed,
            "tool_start_callback": observed.tool_start_callback,
            "post_hook": observed.post_hook,
            "approval": observed.approval,
            "dispatched": observed.dispatched,
            "checkpoint_reasons": [r for _p, r in observed.checkpoints],
            "tool_messages": [
                (m["tool_call_id"], m["content"])
                for m in messages
                if m["role"] == "tool"
            ],
        }

    assert snapshots["sequential"] == snapshots["concurrent"]


# ---------------------------------------------------------------------------
# The funnel is usable without a transcript (the SDK bridge's shape)
# ---------------------------------------------------------------------------


def test_execute_one_tool_applies_wrappers_without_a_message_list():
    """The claude_agent_sdk bridge owns no message list; wrappers still run."""
    agent = _make_agent("web_search")
    observed = _Observed()

    with _instrument(agent, observed):
        outcome = execute_one_tool(
            agent, _tool_call("web_search", {"query": "x"}, "c-bridge"),
            "task-bridge", messages=None,
        )
        result = finalize_tool_outcome(agent, outcome)

    assert isinstance(outcome, ToolExecutionOutcome)
    assert outcome.blocked is False and outcome.cancelled is False
    assert observed.plugin_block == ["web_search"]
    assert observed.guardrail_before == ["web_search"]
    assert observed.tool_started == ["web_search"]
    assert observed.post_hook == ["web_search"]
    assert [task_id for _n, _a, task_id in observed.dispatched] == ["task-bridge"]
    assert json.loads(result) == {"ok": True}


def test_malformed_arguments_are_reported_without_dispatching():
    agent = _make_agent("web_search")
    bad_call = SimpleNamespace(
        id="c-bad",
        type="function",
        function=SimpleNamespace(name="web_search", arguments="{not json"),
    )
    observed = _Observed()

    with _instrument(agent, observed):
        outcome = execute_one_tool(agent, bad_call, "task-bad")

    assert outcome.malformed is True
    assert observed.dispatched == []
    assert observed.tool_started == []
