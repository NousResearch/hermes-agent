"""Regression tests for DeepSeek V4 textual DSML tool compatibility.

DeepSeek V4-compatible endpoints may emit textual DSML instead of
OpenAI-native tool_calls. Hermes converts those calls to its canonical
internal representation for execution, then rewrites only the synthetic
history back to DSML + <tool_result> at the provider boundary.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import (
    _prepare_deepseek_v4_synthetic_tool_history,
)


def _chunk(content=None, *, finish_reason=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        choices=[choice],
        model="deepseek-v4-flash",
        usage=None,
    )


def _run_deepseek_stream(content: str):
    from run_agent import AIAgent

    chunks = [
        _chunk(content),
        _chunk(finish_reason="stop"),
    ]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)

    with (
        patch(
            "run_agent.AIAgent._create_request_openai_client",
            return_value=mock_client,
        ),
        patch("run_agent.AIAgent._close_request_openai_client"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://deepseekv4.network/api/v1",
            model="deepseek-v4-flash",
            provider="deepseek",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False

        return agent._interruptible_streaming_api_call({})


def test_dsml_read_file_normalizes_file_to_path():
    response = _run_deepseek_stream(
        """<｜DSML｜tool_calls>
<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="file" string="true">/tmp/test.txt</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
    )

    calls = response.choices[0].message.tool_calls

    assert calls is not None
    assert len(calls) == 1
    assert calls[0].id.startswith("dsml_")
    assert calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments) == {
        "path": "/tmp/test.txt",
    }


def test_dsml_decodes_typed_parameters():
    response = _run_deepseek_stream(
        """<｜DSML｜tool_calls>
<｜DSML｜invoke name="search_files">
<｜DSML｜parameter name="query" string="true">alpha</｜DSML｜parameter>
<｜DSML｜parameter name="limit" string="false">2</｜DSML｜parameter>
<｜DSML｜parameter name="recursive" string="false">true</｜DSML｜parameter>
<｜DSML｜parameter name="paths" string="false">["a","b"]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
    )

    calls = response.choices[0].message.tool_calls
    assert calls is not None
    assert len(calls) == 1

    args = json.loads(calls[0].function.arguments)

    assert args == {
        "query": "alpha",
        "limit": 2,
        "recursive": True,
        "paths": ["a", "b"],
    }


def test_dsml_multiple_invokes_become_multiple_tool_calls():
    response = _run_deepseek_stream(
        """<｜DSML｜tool_calls>
<｜DSML｜invoke name="read_file">
<｜DSML｜parameter name="path" string="true">/tmp/a.txt</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="search_files">
<｜DSML｜parameter name="query" string="true">needle</｜DSML｜parameter>
<｜DSML｜parameter name="limit" string="false">3</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""
    )

    calls = response.choices[0].message.tool_calls

    assert calls is not None
    assert len(calls) == 2
    assert [call.function.name for call in calls] == [
        "read_file",
        "search_files",
    ]

    assert json.loads(calls[0].function.arguments) == {
        "path": "/tmp/a.txt",
    }
    assert json.loads(calls[1].function.arguments) == {
        "query": "needle",
        "limit": 3,
    }


def test_simple_xml_read_file_fallback():
    response = _run_deepseek_stream(
        "<read_file><path>/tmp/xml.txt</path></read_file>"
    )

    calls = response.choices[0].message.tool_calls

    assert calls is not None
    assert len(calls) == 1
    assert calls[0].id.startswith("deepseek_xml_")
    assert calls[0].function.name == "read_file"
    assert json.loads(calls[0].function.arguments) == {
        "path": "/tmp/xml.txt",
    }


def test_synthetic_history_rewrites_to_dsml_and_tool_result():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "dsml_001",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"/tmp/a.txt"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "dsml_001",
            "content": "VALUE_A",
        },
    ]

    out = _prepare_deepseek_v4_synthetic_tool_history(messages)

    assert len(out) == 2

    assert out[0]["role"] == "assistant"
    assert "tool_calls" not in out[0]
    assert "<｜DSML｜tool_calls>" in out[0]["content"]
    assert '<｜DSML｜invoke name="read_file">' in out[0]["content"]
    assert (
        '<｜DSML｜parameter name="path" string="true">'
        '/tmp/a.txt'
        '</｜DSML｜parameter>'
    ) in out[0]["content"]

    assert out[1] == {
        "role": "user",
        "content": "<tool_result>VALUE_A</tool_result>",
    }


def test_synthetic_batch_preserves_tool_call_result_order():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "dsml_a",
                    "type": "function",
                    "function": {
                        "name": "search_files",
                        "arguments": '{"query":"alpha","limit":2}',
                    },
                },
                {
                    "id": "dsml_b",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/tmp/b.txt"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dsml_b",
            "content": "RESULT_B",
        },
        {
            "role": "tool",
            "tool_call_id": "dsml_a",
            "content": "RESULT_A",
        },
    ]

    out = _prepare_deepseek_v4_synthetic_tool_history(messages)

    assert len(out) == 2
    assert out[0]["content"].count("<｜DSML｜invoke") == 2

    assert out[1]["content"] == (
        "<tool_result>RESULT_A</tool_result>\n"
        "<tool_result>RESULT_B</tool_result>"
    )


def test_native_tool_history_is_not_rewritten():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_native_001",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"/tmp/native.txt"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_native_001",
            "content": "NATIVE_RESULT",
        },
    ]

    out = _prepare_deepseek_v4_synthetic_tool_history(messages)

    assert out == messages
