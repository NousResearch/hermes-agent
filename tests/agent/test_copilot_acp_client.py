"""Focused regressions for the Copilot ACP shim safety layer and prompt formatting."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.copilot_acp_client import (
    CopilotACPClient,
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
)


def test_prompt_includes_assistant_tool_calls_without_content():
    prompt = _format_messages_as_prompt(
        [
            {"role": "user", "content": "Investigate this task."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "delegate_task",
                            "arguments": "{\"goal\":\"Inspect the flow\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "tool_name": "delegate_task",
                "content": "{\"results\":[{\"status\":\"ok\"}]}",
            },
        ]
    )

    assert "Assistant:\n<tool_call>" in prompt
    assert '"name": "delegate_task"' in prompt
    assert '"arguments": "{\\"goal\\":\\"Inspect the flow\\"}"' in prompt
    assert "Tool:\n[tool_call_id=call_1, tool_name=delegate_task]" in prompt


def test_prompt_renders_simplenamespace_tool_calls():
    tool_call = SimpleNamespace(
        id="call_ns",
        function=SimpleNamespace(
            name="search_files",
            arguments="{\"pattern\":\"*.md\"}",
        ),
    )

    prompt = _format_messages_as_prompt(
        [
            {
                "role": "assistant",
                "content": "I'll inspect the notes first.",
                "tool_calls": [tool_call],
            }
        ]
    )

    assert "I'll inspect the notes first." in prompt
    assert "<tool_call>" in prompt
    assert '"id": "call_ns"' in prompt
    assert '"name": "search_files"' in prompt


def test_extract_tool_calls_recovers_multiline_terminal_arguments_as_valid_json():
    raw = """<tool_call>{
      "id":"call_1",
      "type":"function",
      "function":{
        "name":"terminal",
        "arguments":"{\\"command\\":\\"python3 - <<'PY'\\\\nprint('hello')\\\\nPY\\",\\"timeout\\":60}"
      }
    }</tool_call>"""

    tool_calls, cleaned = _extract_tool_calls_from_text(raw)

    assert cleaned == ""
    assert len(tool_calls) == 1
    parsed = json.loads(tool_calls[0].function.arguments)
    assert parsed["timeout"] == 60
    assert parsed["command"] == "python3 - <<'PY'\nprint('hello')\nPY"


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CopilotACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CopilotACPClient(acp_cwd="/tmp")

    def _dispatch(self, message: dict, *, cwd: str) -> dict:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd=cwd,
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        payload = process.stdin.getvalue().strip()
        self.assertTrue(payload)
        return json.loads(payload)

    def test_request_permission_is_not_auto_allowed(self) -> None:
        response = self._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/request_permission",
                "params": {},
            },
            cwd="/tmp",
        )

        outcome = (((response.get("result") or {}).get("outcome") or {}).get("outcome"))
        self.assertEqual(outcome, "cancelled")

    def test_read_text_file_blocks_internal_hermes_hub_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            blocked = home / ".hermes" / "skills" / ".hub" / "index-cache" / "entry.json"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.write_text('{"token": "abc123def456"}')

            with patch.dict(
                os.environ,
                {"HOME": str(home), "HERMES_HOME": str(home / ".hermes")},
                clear=False,
            ):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "fs/read_text_file",
                        "params": {"path": str(blocked)},
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)

    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret_file = root / "config.env"
            secret_file.write_text("OPENAI_API_KEY=abc123def456\n")

            response = self._dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "fs/read_text_file",
                    "params": {"path": str(secret_file)},
                },
                cwd=str(root),
            )

        content = ((response.get("result") or {}).get("content") or "")
        self.assertNotIn("abc123def456", content)
        self.assertIn("OPENAI_API_KEY=***", content)

    def test_write_text_file_reuses_write_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            target = home / ".ssh" / "id_rsa"
            target.parent.mkdir(parents=True, exist_ok=True)

            with patch.dict(os.environ, {"HOME": str(home)}, clear=False):
                with patch("agent.copilot_acp_client.is_write_denied", return_value=True, create=True):
                    response = self._dispatch(
                        {
                            "jsonrpc": "2.0",
                            "id": 4,
                            "method": "fs/write_text_file",
                            "params": {
                                "path": str(target),
                                "content": "fake-private-key",
                            },
                        },
                        cwd=str(home),
                    )

        self.assertIn("error", response)
        self.assertFalse(target.exists())

    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"

            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(outside),
                            "content": "should-not-write",
                        },
                    },
                    cwd=str(root),
                )

        self.assertIn("error", response)
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
