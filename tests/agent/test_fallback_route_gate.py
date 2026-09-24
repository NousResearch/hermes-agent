"""Side-effecting tools are gated on automatic-fallback routes (issue #117495).

Drives the REAL sequential and concurrent dispatch entrypoints: with the gate
enabled and an automatic fallback route acting, a side-effecting tool call is
refused before dispatch while a read-only sibling still executes. A deliberate
model switch (``/model --once`` sets ``_fallback_activated`` but NOT
``_provider_fallback_active``) must never be restricted, and the default
config keeps the legacy unrestricted behavior.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in ("web_search", "cronjob")
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _run_batch(agent, mode, tool_calls, messages):
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    executed = []

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        executed.append(name)
        return json.dumps({"ok": name})

    post_calls = []

    def recording_post(**kwargs):
        post_calls.append(kwargs)

    with (
        patch("model_tools.handle_function_call", side_effect=fake_dispatch),
        patch.object(agent, "_invoke_tool", side_effect=fake_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
        patch(
            "agent.tool_executor._emit_terminal_post_tool_call",
            side_effect=lambda *a, **kw: recording_post(**kw),
        ),
    ):
        execute = getattr(agent, f"_execute_tool_calls_{mode}")
        execute(assistant_message, messages, "task-1")
    return executed, post_calls


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_gate_blocks_side_effecting_tool_but_executes_read_only_sibling(dispatch_mode):
    agent = _make_agent()
    agent._provider_fallback_active = True  # automatic fallback route acting
    messages = []
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"fallback": {"halt_on_side_effecting_tools": True}},
    ):
        executed, post_calls = _run_batch(
            agent, dispatch_mode,
            [
                _tool_call("call-write", "cronjob", '{"action": "create", "schedule": "1h", "prompt": "x"}'),
                _tool_call("call-read", "web_search", '{"query": "docs"}'),
            ],
            messages,
        )

    # The side-effecting call never dispatched; the read-only sibling did.
    assert executed == ["web_search"]
    assert json.loads(messages[1]["content"]) == {"ok": "web_search"}
    # The blocked result landed as the cronjob call's tool result, with the refusal text.
    blocked = next(m for m in messages if m["tool_call_id"] == "call-write")
    assert "automatic provider fallback" in blocked["content"]
    assert [m["tool_call_id"] for m in messages] == ["call-write", "call-read"]
    # The refusal is attributed to the route gate, not to a plugin block.
    assert any(
        kwargs.get("error_type") == "fallback_route_block" and kwargs.get("status") == "blocked"
        for kwargs in post_calls
    )


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_deliberate_model_switch_is_never_restricted(dispatch_mode):
    """``/model --once`` sets _fallback_activated without _provider_fallback_active;
    the gate must stay off for that user-selected route."""
    agent = _make_agent()
    agent._fallback_activated = True
    agent._provider_fallback_active = False
    messages = []
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"fallback": {"halt_on_side_effecting_tools": True}},
    ):
        executed, _ = _run_batch(
            agent, dispatch_mode,
            [
                _tool_call("call-write", "cronjob", '{"action": "create", "schedule": "1h", "prompt": "x"}'),
                _tool_call("call-read", "web_search", '{"query": "docs"}'),
            ],
            messages,
        )

    # ``cronjob`` is a legacy alias resolved to ``cronjob_manage`` at dispatch.
    assert set(executed) == {"cronjob_manage", "web_search"}
    assert all("automatic provider fallback" not in m["content"] for m in messages)


@pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
def test_default_config_keeps_legacy_unrestricted_behavior(dispatch_mode):
    agent = _make_agent()
    agent._provider_fallback_active = True
    messages = []
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},  # default: halt_on_side_effecting_tools absent → gate off
    ):
        executed, _ = _run_batch(
            agent, dispatch_mode,
            [
                _tool_call("call-write", "cronjob", '{"action": "create", "schedule": "1h", "prompt": "x"}'),
            ],
            messages,
        )

    # Legacy alias: the registry dispatches ``cronjob`` as ``cronjob_manage``.
    assert executed == ["cronjob_manage"]


def test_gate_classification_decisions():
    """Unit-level pin of the gate's decisions: effect-capable tool blocked, clarify
    allowed (a prompt for the user, not an external write), read-only allowed, and the
    acting-route predicate alone never blocks."""
    from types import SimpleNamespace as NS

    from agent.fallback_route_gate import fallback_route_block_reason

    acting = NS(_provider_fallback_active=True)
    idle = NS(_provider_fallback_active=False)
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={"fallback": {"halt_on_side_effecting_tools": True}},
    ):
        assert "automatic provider fallback" in fallback_route_block_reason(acting, "execute_code", "nous", "glm")
        assert fallback_route_block_reason(acting, "clarify", "nous", "glm") is None
        assert fallback_route_block_reason(acting, "read_file", "nous", "glm") is None
        assert fallback_route_block_reason(idle, "execute_code", "nous", "glm") is None
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert fallback_route_block_reason(acting, "execute_code", "nous", "glm") is None


def _make_agent_real_config() -> AIAgent:
    """Like _make_agent but WITHOUT patching the config readers: the real
    ``load_config_readonly`` must resolve the key from the sandboxed ``HERMES_HOME``
    (root rubric: config propagation is E2E'd with real imports, not mocks)."""
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "cronjob",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def test_real_config_file_arms_the_gate_end_to_end(tmp_path):
    """Root rubric E2E: the opt-in key set in a real config.yaml under a temp
    HERMES_HOME (the autouse sandbox) reaches the gate through the REAL
    ``load_config_readonly`` and blocks a side-effecting tool at dispatch."""
    import os

    hermes_home = os.environ["HERMES_HOME"]
    cfg = Path(hermes_home) / "config.yaml"
    cfg.write_text("fallback:\n  halt_on_side_effecting_tools: true\n")

    agent = _make_agent_real_config()
    agent._provider_fallback_active = True
    messages = []
    executed, post_calls = _run_batch(
        agent, "sequential",
        [_tool_call("call-write", "cronjob", '{"action": "create", "schedule": "1h", "prompt": "x"}')],
        messages,
    )

    assert executed == []  # never dispatched
    blocked = next(m for m in messages if m["tool_call_id"] == "call-write")
    assert "automatic provider fallback" in blocked["content"]
    assert any(kwargs.get("error_type") == "fallback_route_block" for kwargs in post_calls)
