"""Tests for memory-tool result -> _emit_warning surfacing (#2771).

The built-in `memory` tool is executed inline by
``execute_tool_calls_sequential`` in ``agent/tool_executor.py``. Prior to
the salvage fix for #2771, capacity overflows / write failures were
JSON-encoded into the tool result string and silently flowed back to the
model — gateway/IM users never saw them. The fix parses the result and
fans the three interesting cases out through the existing
``Agent._emit_warning`` plumbing.

These tests verify:
  - ``success: False``                          -> warning containing 'failed'
  - ``success: True`` + ``usage`` >= 80%%         -> capacity warning
  - ``success: True`` + ``truncated: True``       -> truncation warning
  - normal success (usage < 80%%, no truncate)   -> NO warning emitted

We construct a real ``AIAgent`` (cheap thanks to ``skip_memory=True``)
and monkey-patch ``tools.memory_tool.memory_tool`` to return a chosen
JSON string. We then capture ``_emit_warning`` calls and assert on them.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


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


def _mock_tool_call(name="memory", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_agent() -> AIAgent:
    """Build a minimal AIAgent that can run the sequential tool dispatcher."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("memory")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=3,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    # Capture warnings for assertion. _emit_warning is the user-visible
    # channel we are validating.
    agent._captured_warnings = []
    original = agent._emit_warning

    def _capture(msg):
        agent._captured_warnings.append(msg)
        # Don't call original — it tries to vprint which is fine but
        # noisy; we just need the list of messages.
        return original(msg) if False else None

    agent._emit_warning = _capture  # type: ignore[assignment]
    return agent


def _run_memory_tool(agent: AIAgent, memory_result: dict, args: dict | None = None) -> list:
    """Drive a single memory tool call through the sequential dispatcher.

    Returns the list of warning messages captured during the run.
    """
    args = args or {"action": "add", "target": "memory", "content": "hello"}
    tc = _mock_tool_call("memory", json.dumps(args), "c-mem")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages: list = []
    # Patch the symbol where it is imported inside the dispatcher branch.
    with patch(
        "tools.memory_tool.memory_tool",
        return_value=json.dumps(memory_result),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-mem")
    return list(agent._captured_warnings)


# =========================================================================
# Failure path: success=False -> warning containing 'failed'
# =========================================================================


def test_failure_emits_warning():
    agent = _make_agent()
    warnings = _run_memory_tool(
        agent,
        {
            "success": False,
            "error": "Memory at 480/500 chars. Adding this entry (200 chars) would exceed the limit.",
        },
    )
    assert any("failed" in w.lower() for w in warnings), warnings
    assert any("memory" in w.lower() for w in warnings), warnings


def test_failure_warning_truncates_long_error():
    # Defensive: a runaway error string must not flood the warning channel.
    agent = _make_agent()
    long_err = "boom " * 200  # 1000 chars
    warnings = _run_memory_tool(
        agent,
        {"success": False, "error": long_err},
    )
    assert warnings, "expected at least one warning"
    # The warning includes a prefix; the embedded error portion should
    # be capped at ~200 chars per the spec.
    bodies = [w for w in warnings if "failed" in w.lower()]
    assert bodies
    # The whole warning is "Memory write failed (target): <err[:200]>";
    # check the err substring is truncated.
    assert len(bodies[0]) < 400


# =========================================================================
# Capacity warning: usage >= 80%% -> capacity message
# =========================================================================


def test_full_capacity_emits_warning():
    agent = _make_agent()
    warnings = _run_memory_tool(
        agent,
        {
            "success": True,
            "target": "memory",
            "entries": ["fact"],
            "usage": "85% — 1,700/2,000 chars",
            "entry_count": 1,
        },
    )
    assert any("85%" in w and "capacity" in w.lower() for w in warnings), warnings


def test_capacity_warning_at_exactly_80():
    agent = _make_agent()
    warnings = _run_memory_tool(
        agent,
        {
            "success": True,
            "target": "memory",
            "entries": ["fact"],
            "usage": "80% — 1,600/2,000 chars",
            "entry_count": 1,
        },
    )
    assert any("capacity" in w.lower() for w in warnings), warnings


# =========================================================================
# Truncation warning: truncated=True -> kept N/M chars message
# =========================================================================


def test_truncated_emits_warning():
    agent = _make_agent()
    warnings = _run_memory_tool(
        agent,
        {
            "success": True,
            "target": "memory",
            "entries": ["x"],
            "usage": "60% — 1,200/2,000 chars",
            "entry_count": 1,
            "truncated": True,
            "original_length": 1000,
            "saved_length": 600,
        },
    )
    assert any("truncated" in w.lower() for w in warnings), warnings
    # Should mention the byte counts.
    truncation_msgs = [w for w in warnings if "truncated" in w.lower()]
    assert any("600" in w and "1000" in w for w in truncation_msgs), truncation_msgs


# =========================================================================
# Normal success: no warning at all
# =========================================================================


def test_normal_success_no_warning():
    agent = _make_agent()
    warnings = _run_memory_tool(
        agent,
        {
            "success": True,
            "target": "memory",
            "entries": ["fact"],
            "usage": "30% — 600/2,000 chars",
            "entry_count": 1,
        },
    )
    assert warnings == [], warnings


def test_malformed_result_does_not_break_path():
    # Guarantee robustness: a non-JSON result must not raise. Use a string
    # that isn't valid JSON via a patch that returns garbage.
    agent = _make_agent()
    tc = _mock_tool_call("memory", json.dumps({"action": "add", "content": "x"}), "c-mem")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages: list = []
    with patch("tools.memory_tool.memory_tool", return_value="not valid json!!"):
        # Should not raise.
        agent._execute_tool_calls_sequential(msg, messages, "task-mem")
    # And no warnings synthesised from garbage.
    assert agent._captured_warnings == []
