from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.delegation_purpose import (
    BOUNDED_IMPLEMENTATION,
    RESEARCH_EVIDENCE,
    normalize_delegation_purpose,
    purpose_prompt_block,
    purpose_tool_block_message,
)
from agent.tool_executor import _ManagedToolResult, _ToolCallRef, _dispatch_authorized_once


def test_purpose_normalization_is_closed_to_two_values():
    assert normalize_delegation_purpose(None) == RESEARCH_EVIDENCE
    assert normalize_delegation_purpose(" bounded_implementation ") == BOUNDED_IMPLEMENTATION
    try:
        normalize_delegation_purpose("unbounded")
    except ValueError as exc:
        assert "bounded_implementation" in str(exc)
        assert "research_evidence" in str(exc)
    else:
        raise AssertionError("invalid delegation purpose was accepted")


def test_parent_agent_without_purpose_is_not_restricted():
    assert purpose_tool_block_message(SimpleNamespace(), "write_file") is None


def test_research_child_blocks_direct_mutation_but_bounded_child_does_not():
    research = SimpleNamespace(_delegate_purpose=RESEARCH_EVIDENCE)
    bounded = SimpleNamespace(_delegate_purpose=BOUNDED_IMPLEMENTATION)
    assert "unavailable" in purpose_tool_block_message(research, "write_file")
    assert purpose_tool_block_message(research, "read_file") is None
    assert purpose_tool_block_message(bounded, "write_file") is None


def test_research_purpose_block_happens_before_real_dispatch():
    agent = SimpleNamespace(_delegate_purpose=RESEARCH_EVIDENCE)
    state = _ManagedToolResult(result=None, args={"path": "C:/work/x"}, middleware_trace=[], blocked=False, dispatched=False)
    ref = _ToolCallRef("write_file", {"path": "C:/work/x"}, "task", "call", [])
    execute = MagicMock()
    with patch("agent.tool_executor._blocked_tool_result", return_value="blocked") as blocked:
        result = _dispatch_authorized_once(
            agent, state, ref, execute=execute, scope_block=None,
            display_index=None, begin_execution=None, authorization_gate=None,
        )
    assert result == "blocked"
    assert state.blocked is True
    execute.assert_not_called()
    assert blocked.call_args.kwargs["block_error_type"] == "delegation_purpose_block"


def test_prompt_blocks_name_the_enforced_contract():
    research = purpose_prompt_block(RESEARCH_EVIDENCE)
    implementation = purpose_prompt_block(BOUNDED_IMPLEMENTATION)
    assert "Do not create, edit, or delete" in research
    assert "Runtime-blocked" in research
    assert "explicitly allowed" in implementation
