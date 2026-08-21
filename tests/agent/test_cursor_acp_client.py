"""Unit tests for the Cursor ACP OpenAI-compatible shim.

These cover the contract Hermes needs to drive `cursor-agent acp` as a
tool-using seat: tool-schema injection, <tool_call> extraction (including
malformed / truncated / multiple blocks), plain-prose responses, and
CLI-name → ACP model-id mapping.
"""

from __future__ import annotations

import json

import pytest

from agent.cursor_acp_client import (
    ACP_MARKER_BASE_URL,
    CursorACPClient,
    CursorACPToolCallParseError,
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
    resolve_cursor_acp_model_id,
)


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

WRITE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
}


def _tool_block(call_id: str, name: str, arguments: dict) -> str:
    payload = {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }
    return f"<tool_call>{json.dumps(payload, separators=(',', ':'))}</tool_call>"


class TestFormatMessagesAsPrompt:
    def test_injects_tool_schema_and_tool_call_contract(self) -> None:
        prompt = _format_messages_as_prompt(
            [{"role": "user", "content": "Read /etc/hostname"}],
            model="gpt-5.6-sol-high",
            tools=[READ_FILE_TOOL],
            tool_choice="auto",
        )
        assert "You are being used as the active ACP agent backend for Hermes." in prompt
        assert "<tool_call>" in prompt
        assert "</tool_call>" in prompt
        assert "read_file" in prompt
        assert "Read a file from disk" in prompt
        assert "Hermes requested model hint: gpt-5.6-sol-high" in prompt
        assert "User:" in prompt
        assert "Read /etc/hostname" in prompt
        assert "Available tools (OpenAI function schema)" in prompt

    def test_renders_tool_role_transcript_for_multi_turn(self) -> None:
        prompt = _format_messages_as_prompt(
            [
                {"role": "user", "content": "Read /etc/hostname then summarize it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"/etc/hostname"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "maximus-host",
                },
            ],
            tools=[READ_FILE_TOOL],
        )
        assert "Tool:" in prompt
        assert "maximus-host" in prompt
        assert "User:" in prompt


class TestExtractToolCallsFromText:
    def test_parses_single_tool_call_block(self) -> None:
        raw = _tool_block("call_1", "read_file", {"path": "/etc/hostname"})
        calls, cleaned = _extract_tool_calls_from_text(raw)
        assert len(calls) == 1
        assert calls[0].id == "call_1"
        assert calls[0].type == "function"
        assert calls[0].function.name == "read_file"
        assert json.loads(calls[0].function.arguments) == {"path": "/etc/hostname"}
        assert cleaned == ""

    def test_parses_multiple_tool_call_blocks(self) -> None:
        raw = (
            _tool_block("call_1", "read_file", {"path": "/etc/hostname"})
            + "\n"
            + _tool_block("call_2", "write_file", {"path": "/tmp/out", "content": "x"})
        )
        calls, cleaned = _extract_tool_calls_from_text(raw)
        assert [c.function.name for c in calls] == ["read_file", "write_file"]
        assert [c.id for c in calls] == ["call_1", "call_2"]
        assert json.loads(calls[1].function.arguments)["path"] == "/tmp/out"
        assert cleaned == ""

    def test_plain_prose_returns_no_tool_calls(self) -> None:
        prose = "The hostname is maximus-host. No further action needed."
        calls, cleaned = _extract_tool_calls_from_text(prose)
        assert calls == []
        assert cleaned == prose

    def test_malformed_json_inside_block_does_not_crash_or_drop_silently(self) -> None:
        raw = "<tool_call>{not-json}</tool_call>"
        with pytest.raises(CursorACPToolCallParseError, match="malformed"):
            _extract_tool_calls_from_text(raw)

    def test_truncated_tool_call_block_does_not_crash_or_drop_silently(self) -> None:
        raw = (
            '<tool_call>{"id":"call_1","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"/etc/hostname\\"}"'
        )
        with pytest.raises(CursorACPToolCallParseError, match="truncated"):
            _extract_tool_calls_from_text(raw)

    def test_mixed_valid_and_malformed_does_not_silently_drop(self) -> None:
        raw = (
            _tool_block("call_1", "read_file", {"path": "/etc/hostname"})
            + "\n<tool_call>{broken}</tool_call>"
        )
        with pytest.raises(CursorACPToolCallParseError, match="malformed"):
            _extract_tool_calls_from_text(raw)

    def test_object_arguments_are_serialized(self) -> None:
        raw = (
            '<tool_call>{"id":"call_obj","type":"function",'
            '"function":{"name":"read_file","arguments":{"path":"/etc/hostname"}}}'
            "</tool_call>"
        )
        calls, _cleaned = _extract_tool_calls_from_text(raw)
        assert len(calls) == 1
        assert json.loads(calls[0].function.arguments) == {"path": "/etc/hostname"}


class TestResolveCursorAcpModelId:
    AVAILABLE = [
        {"modelId": "default[]", "name": "Auto"},
        {
            "modelId": "gpt-5.6-sol[context=272k,reasoning=medium,fast=false]",
            "name": "gpt-5.6-sol",
        },
        {"modelId": "gemini-3.7-flash[effort=high]", "name": "gemini-3.7-flash"},
    ]

    def test_maps_cli_high_alias_to_available_gemini_id(self) -> None:
        assert (
            resolve_cursor_acp_model_id("gemini-3.7-flash-high", self.AVAILABLE)
            == "gemini-3.7-flash[effort=high]"
        )

    def test_maps_cli_high_alias_to_available_sol_family(self) -> None:
        assert (
            resolve_cursor_acp_model_id("gpt-5.6-sol-high", self.AVAILABLE)
            == "gpt-5.6-sol[context=272k,reasoning=medium,fast=false]"
        )

    def test_exact_model_id_passthrough(self) -> None:
        assert (
            resolve_cursor_acp_model_id(
                "gemini-3.7-flash[effort=high]", self.AVAILABLE
            )
            == "gemini-3.7-flash[effort=high]"
        )

    def test_unknown_model_returns_none(self) -> None:
        assert resolve_cursor_acp_model_id("not-a-real-model", self.AVAILABLE) is None


class TestCursorACPClientShape:
    def test_marker_base_url(self) -> None:
        client = CursorACPClient()
        assert ACP_MARKER_BASE_URL == "acp://cursor"
        assert client.base_url == "acp://cursor"
        assert client.api_key == "cursor-acp"

    def test_default_launch_is_cursor_agent_acp(self) -> None:
        client = CursorACPClient()
        assert client._acp_command == "cursor-agent"
        assert client._acp_args == ["acp"]

    def test_create_completion_parses_tool_call(self) -> None:
        client = CursorACPClient()
        raw = _tool_block("call_1", "read_file", {"path": "/etc/hostname"})
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(client, "_run_prompt", lambda *_a, **_k: (raw, ""))
            completion = client.chat.completions.create(
                model="gpt-5.6-sol-high",
                messages=[{"role": "user", "content": "read hostname"}],
                tools=[READ_FILE_TOOL],
            )
        assert completion.choices[0].finish_reason == "tool_calls"
        tool_calls = completion.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "read_file"
        assert completion.choices[0].message.content == ""

    def test_create_completion_plain_prose(self) -> None:
        client = CursorACPClient()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                client,
                "_run_prompt",
                lambda *_a, **_k: ("hostname is maximus-host", ""),
            )
            completion = client.chat.completions.create(
                model="gemini-3.7-flash-high",
                messages=[{"role": "user", "content": "what is the hostname?"}],
            )
        assert completion.choices[0].finish_reason == "stop"
        assert completion.choices[0].message.tool_calls == []
        assert "maximus-host" in completion.choices[0].message.content

    def test_missing_cli_is_actionable(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "agent.cursor_acp_client.subprocess.Popen",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("missing")),
        )
        client = CursorACPClient(
            acp_command="cursor-agent-missing",
            acp_args=["acp"],
            acp_cwd=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="cursor-agent login"):
            client._run_prompt("hello", timeout_seconds=1)

    def test_authenticate_failure_is_actionable(self, tmp_path) -> None:
        """A live ACP process that rejects cursor_login must not fall back to prose."""
        server = tmp_path / "fake_unauth_acp.py"
        server.write_text(
            "import json, sys\n"
            "\n"
            "def send(obj):\n"
            "    sys.stdout.write(json.dumps(obj) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "\n"
            "for line in sys.stdin:\n"
            "    msg = json.loads(line)\n"
            "    method = msg.get('method')\n"
            "    mid = msg.get('id')\n"
            "    if method == 'initialize':\n"
            "        send({\n"
            "            'jsonrpc': '2.0',\n"
            "            'id': mid,\n"
            "            'result': {\n"
            "                'protocolVersion': 1,\n"
            "                'authMethods': [{'id': 'cursor_login'}],\n"
            "            },\n"
            "        })\n"
            "    elif method == 'authenticate':\n"
            "        send({\n"
            "            'jsonrpc': '2.0',\n"
            "            'id': mid,\n"
            "            'error': {'code': -32000, 'message': 'Not logged in'},\n"
            "        })\n"
            "    else:\n"
            "        send({\n"
            "            'jsonrpc': '2.0',\n"
            "            'id': mid,\n"
            "            'error': {'code': -32601, 'message': method},\n"
            "        })\n",
            encoding="utf-8",
        )
        import sys

        client = CursorACPClient(
            acp_command=sys.executable,
            acp_args=[str(server)],
            acp_cwd=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match="authentication failed"):
            client._run_prompt("hello", timeout_seconds=5)
