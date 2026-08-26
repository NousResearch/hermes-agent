from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _function_source(relative_path: str, function_name: str) -> str:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            segment = ast.get_source_segment(text, node)
            assert segment is not None
            return segment
    raise AssertionError(f"function not found: {relative_path}:{function_name}")


def test_turn_context_calls_reasoning_escalation_with_clean_prompt():
    source = _function_source("agent/turn_context.py", "build_turn_context")
    assert "from agent.reasoning_escalation import apply_turn_reasoning_escalation" in source
    assert "apply_turn_reasoning_escalation(agent, reasoning_prompt)" in source
    assert "persist_user_message" in source


def test_wire_reasoning_config_reads_effective_turn_config():
    module = (ROOT / "agent/chat_completion_helpers.py").read_text(encoding="utf-8")
    wire = _function_source("agent/chat_completion_helpers.py", "_reasoning_config_for_wire")
    build = _function_source("agent/chat_completion_helpers.py", "_build_api_kwargs_for_mode")
    assert "from agent.reasoning_escalation import effective_reasoning_config" in module
    assert "effective_reasoning_config(agent)" in wire
    assert "_reasoning_config_for_wire(agent)" in build


def test_wire_reasoning_config_honours_pinned_turn_override():
    from types import SimpleNamespace

    from agent.chat_completion_helpers import _reasoning_config_for_wire

    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-6-astra",
        reasoning_config={"enabled": True, "effort": "medium"},
        _turn_reasoning_config_override={"enabled": True, "effort": "high"},
        _turn_reasoning_escalation_target=("openai-codex", "gpt-6-astra"),
    )
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "high"}

    # A fallback route (different provider/model tuple) must not inherit the escalation.
    agent.model = "gpt-5.6-luna"
    assert _reasoning_config_for_wire(agent) == {"enabled": True, "effort": "medium"}
