"""Guard D-core — budget-grace tool gate tests.

Two layers:

1. **Pure predicate** (``agent/budget_grace_gate.py``): allowlist / deny-by-default
   / mutating-overlap / block-message shape.
2. **Real dispatch integration**: drive the actual sequential AND concurrent tool
   executors with ``agent._in_budget_grace = True`` and assert side-effecting
   tools are refused (``handle_function_call`` never called, a blocked role=tool
   message is appended) while read-only tools still execute. A mixed batch
   executes the read and refuses the side effect (per-call gating).

Why integration and not just the loop end-to-end: ``_budget_grace_call`` is a
dormant hook (never armed to True today), so a "real bounded task that exhausts
budget" exits *before* any grace turn and would prove nothing. Setting
``_in_budget_grace = True`` directly exercises the exact state the loop sets on
the grace turn (``agent/conversation_loop.py``) and drives the real dispatchers.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

from agent.budget_grace_gate import (
    GRACE_READONLY_TOOLS,
    grace_block_message,
    grace_block_result,
    is_readonly_grace_tool,
)
from agent.tool_guardrails import MUTATING_TOOL_NAMES


# ──────────────────────────────────────────────────────────────────────────
# Layer 1 — pure predicate
# ──────────────────────────────────────────────────────────────────────────


def test_readonly_allowlisted_tool_is_permitted():
    assert is_readonly_grace_tool("read_file") is True
    assert is_readonly_grace_tool("search_files") is True
    assert is_readonly_grace_tool("skill_view") is True


def test_side_effecting_tool_is_refused():
    for name in ("terminal", "execute_code", "write_file", "delegate_task", "send_message"):
        assert is_readonly_grace_tool(name) is False, name


def test_unknown_or_future_tool_is_refused_deny_by_default():
    # The crux of deny-by-default: a name nobody has seen is refused.
    assert is_readonly_grace_tool("some_future_tool") is False
    assert is_readonly_grace_tool("third_party_mcp_doomsday") is False
    assert is_readonly_grace_tool("") is False
    assert is_readonly_grace_tool(None) is False  # type: ignore[arg-type]


def test_mutating_set_always_wins_over_allowlist():
    # Defense-in-depth: no allowlist entry may also be a known mutating tool.
    overlap = GRACE_READONLY_TOOLS & MUTATING_TOOL_NAMES
    assert overlap == frozenset(), f"allowlist must not contain mutating tools: {overlap}"
    # memory/todo mutate state and must be refused even though they're "cheap".
    assert is_readonly_grace_tool("memory") is False
    assert is_readonly_grace_tool("todo") is False


def test_block_message_and_result_shape():
    msg = grace_block_message("terminal")
    assert "terminal" in msg and "budget" in msg.lower()
    parsed = json.loads(grace_block_result("terminal"))
    assert parsed["error"] == msg
    assert parsed["budget_grace_block"]["tool_name"] == "terminal"
    assert parsed["budget_grace_block"]["reason"] == "budget_exhausted_grace_turn"


# ──────────────────────────────────────────────────────────────────────────
# Layer 2 — real dispatch integration
# ──────────────────────────────────────────────────────────────────────────


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


def _mock_tool_call(name="read_file", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_agent(*tool_names: str, max_iterations: int = 10) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
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


def _tool_msgs(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def test_default_in_budget_grace_is_false_after_init():
    agent = _make_agent("read_file")
    assert getattr(agent, "_in_budget_grace", None) is False


def test_grace_turn_refuses_side_effecting_tool_sequential():
    agent = _make_agent("terminal")
    agent._in_budget_grace = True
    tc = _mock_tool_call("terminal", json.dumps({"command": "rm -rf /"}), "c-term")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    tmsgs = _tool_msgs(messages)
    assert len(tmsgs) == 1
    assert tmsgs[0]["tool_call_id"] == "c-term"
    body = json.loads(tmsgs[0]["content"])
    assert "budget" in body["error"].lower()


def test_grace_turn_allows_readonly_tool_sequential():
    agent = _make_agent("read_file")
    agent._in_budget_grace = True
    tc = _mock_tool_call("read_file", json.dumps({"path": "/tmp/x"}), "c-read")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"ok": True})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    tmsgs = _tool_msgs(messages)
    assert len(tmsgs) == 1
    assert tmsgs[0]["tool_call_id"] == "c-read"
    assert "budget" not in tmsgs[0]["content"].lower()


def test_grace_turn_refuses_unknown_tool_sequential():
    agent = _make_agent("read_file")  # tool not even registered
    agent._in_budget_grace = True
    tc = _mock_tool_call("some_future_tool", "{}", "c-unknown")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    tmsgs = _tool_msgs(messages)
    assert len(tmsgs) == 1
    # Sequential path emits the shared {"error": <msg>} block shape; the message
    # names the refused tool and the budget reason.
    body = json.loads(tmsgs[0]["content"])
    assert "some_future_tool" in body["error"]
    assert "budget" in body["error"].lower()


def test_grace_turn_mixed_batch_executes_read_refuses_side_effect_sequential():
    agent = _make_agent("read_file", "terminal")
    agent._in_budget_grace = True
    calls = [
        _mock_tool_call("read_file", json.dumps({"path": "/tmp/x"}), "c-read"),
        _mock_tool_call("terminal", json.dumps({"command": "id"}), "c-term"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append(name)
        return json.dumps({"ok": name})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    # read_file executed; terminal refused (per-call gating, not all-or-nothing).
    assert executed == ["read_file"]
    by_id = {m["tool_call_id"]: m for m in _tool_msgs(messages)}
    assert "budget" not in by_id["c-read"]["content"].lower()
    term_body = json.loads(by_id["c-term"]["content"])
    assert "terminal" in term_body["error"] and "budget" in term_body["error"].lower()


def test_grace_turn_refuses_side_effecting_tool_concurrent():
    agent = _make_agent("terminal", "execute_code")
    agent._in_budget_grace = True
    calls = [
        _mock_tool_call("terminal", json.dumps({"command": "id"}), "c-term"),
        _mock_tool_call("execute_code", json.dumps({"code": "x=1"}), "c-code"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    tmsgs = _tool_msgs(messages)
    assert {m["tool_call_id"] for m in tmsgs} == {"c-term", "c-code"}
    for m in tmsgs:
        assert "budget" in m["content"].lower()


def test_grace_turn_mixed_batch_executes_read_refuses_side_effect_concurrent():
    # Concurrent counterpart to the sequential mixed-batch test: per-call gating
    # in _execute_tool_calls_concurrent — the read runs, the side effect is
    # refused (not all-or-nothing).
    agent = _make_agent("read_file", "terminal")
    agent._in_budget_grace = True
    calls = [
        _mock_tool_call("read_file", json.dumps({"path": "/tmp/x"}), "c-read"),
        _mock_tool_call("terminal", json.dumps({"command": "id"}), "c-term"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append(name)
        return json.dumps({"ok": name})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    # read_file executed; terminal refused (per-call gating in the concurrent path).
    assert executed == ["read_file"]
    by_id = {m["tool_call_id"]: m for m in _tool_msgs(messages)}
    assert "budget" not in by_id["c-read"]["content"].lower()
    term_body = json.loads(by_id["c-term"]["content"])
    assert "terminal" in term_body["error"] and "budget" in term_body["error"].lower()


def test_grace_block_result_metadata_key_present_in_real_dispatch():
    # The crux of the shared-helper fix: both dispatch paths must emit the
    # grace_block_result() shape (with the budget_grace_block metadata key the
    # unit test asserts), not a bare {"error": ...}. Proven against BOTH real
    # executors so the tested shape and the runtime-emitted shape can't diverge.
    for dispatch in ("sequential", "concurrent"):
        agent = _make_agent("terminal")
        agent._in_budget_grace = True
        tc = _mock_tool_call("terminal", json.dumps({"command": "id"}), "c-term")
        msg = SimpleNamespace(content="", tool_calls=[tc])
        messages = []
        with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN"):
            getattr(agent, f"_execute_tool_calls_{dispatch}")(msg, messages, "task-1")
        body = json.loads(_tool_msgs(messages)[0]["content"])
        assert body["budget_grace_block"]["tool_name"] == "terminal", dispatch
        assert body["budget_grace_block"]["reason"] == "budget_exhausted_grace_turn", dispatch


def test_no_grace_means_normal_execution_side_effect_runs():
    # Control: with _in_budget_grace False (the normal state), a side-effecting
    # tool runs as usual — the gate must not block outside the grace turn.
    agent = _make_agent("terminal")
    assert agent._in_budget_grace is False
    tc = _mock_tool_call("terminal", json.dumps({"command": "id"}), "c-term")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 0})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    tmsgs = _tool_msgs(messages)
    assert "budget_grace_block" not in tmsgs[0]["content"]


def test_grace_flag_is_not_re_armed_by_a_refusal():
    # The gate refuses execution but must NEVER set _budget_grace_call back to
    # True — a deny-that-loops would itself be a runaway. Refusing a call leaves
    # the grace flag cleared so the loop exits after this iteration.
    agent = _make_agent("terminal")
    agent._in_budget_grace = True
    agent._budget_grace_call = False  # loop already consumed it this turn
    tc = _mock_tool_call("terminal", json.dumps({"command": "id"}), "c-term")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN"):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert agent._budget_grace_call is False  # not re-armed
