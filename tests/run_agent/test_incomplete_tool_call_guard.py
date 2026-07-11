"""Focused regressions for serialized tool-call false completions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import is_incomplete_tool_call_response
from agent.chat_completion_helpers import _normalize_iteration_summary
from run_agent import AIAgent


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a command.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _response(content: str, *, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


@pytest.fixture()
def agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance.client = MagicMock()
    instance._cached_system_prompt = "You are helpful."
    instance._use_prompt_caching = False
    instance.tool_delay = 0
    instance.compression_enabled = False
    instance.save_trajectories = False
    return instance


@pytest.mark.parametrize(
    "payload",
    [
        (
            '<tool_calls><invoke name="terminal"></invoke></tool_calls>\n'
            "Done; command succeeded."
        ),
        (
            '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="terminal">'
            "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>\nTask complete."
        ),
        '<tool_calls><invoke name="terminal"',
        "<｜｜DSML｜｜tool_calls",
        ('```json\n{"tool_calls":[{"name":"terminal","arguments":{}}]}\n```'),
        (
            '~~~application/json\n{"tool_calls":[{"name":"terminal",'
            '"arguments":{}}]}\n~~~'
        ),
        (
            '{"id":"resp_1","content":"Done",'
            '"tool_calls":[{"name":"terminal","arguments":{}}]}'
        ),
        (
            '{"id":"call_1","type":"function","function":'
            '{"name":"terminal","arguments":"{}"}}'
        ),
        '[{"name":"terminal","arguments":{"cmd":"pwd"}}]',
        (
            '{"id":"call_1","type":"function","function":'
            '{"name":"terminal","arguments":"{'
        ),
        '[{"name":"terminal","arguments":{"cmd":"pwd"}',
        '{"id":"call_1","type":"function","function":{',
        (
            '{"id":"call_1","type":"function","function":'
            '{"name":"terminal"'
        ),
        '[{"name":"terminal"',
        (
            '\ufeff{"id":"call_1","type":"function","function":'
            '{"name":"terminal"'
        ),
        '\u200b[{"name":"terminal"',
        (
            '{"id":"resp_1","status":"completed",'
            '"tool_calls":[{"function":{"name":"terminal"'
        ),
        '<function name="terminal">{"cmd":"pwd"}',
        '<｜｜DSML｜｜invoke name="terminal">{"cmd":"pwd"}',
        (
            "<reasoning>private planning</reasoning>"
            '<tool_calls><invoke name="terminal"></invoke></tool_calls>\nDone.'
        ),
    ],
    ids=[
        "xml_with_claim",
        "dsml_with_claim",
        "malformed_xml",
        "truncated_dsml",
        "fenced_json",
        "tilde_fenced_json",
        "json_metadata_before_tool_calls",
        "openai_function_object",
        "direct_function_object_list",
        "truncated_openai_function_object",
        "truncated_direct_function_object_list",
        "truncated_openai_before_function_body",
        "truncated_openai_before_arguments",
        "truncated_direct_before_arguments",
        "bom_prefixed_truncated_openai",
        "zero_width_prefixed_truncated_direct",
        "truncated_json_metadata_before_tool_calls",
        "truncated_gemma_function",
        "bare_dsml_invoke",
        "reasoning_prefix",
    ],
)
def test_structural_tool_envelopes_are_never_final(payload: str) -> None:
    assert is_incomplete_tool_call_response(payload) is True


@pytest.mark.parametrize(
    "prose",
    [
        "The provider emitted <tool_calls> markup, so I repaired the parser.",
        "Use `<tool_calls>` when documenting the wire protocol.",
        '{"summary":"The tool_calls field is documented here."}',
        '{"summary":"Done."} The "tool_calls": key is documented separately.',
        (
            '{"type":"function","function":{"name":"terminal",'
            '"description":"Run a command","parameters":{"type":"object"}}}'
        ),
        "<function> is a term used in programming documentation.",
        "```python\nprint('<tool_calls>')\n```",
    ],
)
def test_prose_that_only_discusses_markup_remains_valid(prose: str) -> None:
    assert is_incomplete_tool_call_response(prose) is False


def test_iteration_summary_classifies_raw_xml_before_scrubbing() -> None:
    scrubber = MagicMock(return_value="Done; command succeeded.")
    fake_agent = SimpleNamespace(_strip_think_blocks=scrubber)
    payload = (
        '<tool_calls><invoke name="terminal"></invoke></tool_calls>\n'
        "Done; command succeeded."
    )

    assert _normalize_iteration_summary(fake_agent, payload) == ""
    scrubber.assert_not_called()


def test_normal_final_with_tool_envelope_and_claim_is_failed(agent: AIAgent) -> None:
    payload = (
        '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="terminal">'
        "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>\n"
        "Done; command succeeded."
    )
    agent.client.chat.completions.create.side_effect = [
        _response(payload),
        _response(payload),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Run the command and report the result.")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "incomplete_tool_response_exhausted"
    assert "finalDisposition: blocked" in result["final_response"]
    assert payload not in result["final_response"]


def test_length_continuation_is_reassembled_before_classification(
    agent: AIAgent,
) -> None:
    agent.client.chat.completions.create.side_effect = [
        _response(
            '<tool_calls><invoke name="terminal">',
            finish_reason="length",
        ),
        _response(
            "</invoke></tool_calls>\nDone; command succeeded.",
        ),
        _response("Recovered final answer."),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Run the command and report the result.")

    assert result["completed"] is True
    assert result["failed"] is False
    assert result["api_calls"] == 3
    assert result["final_response"] == "Recovered final answer."
    assert all(
        "<tool_calls>" not in str(message.get("content") or "")
        for message in result["messages"]
    )
