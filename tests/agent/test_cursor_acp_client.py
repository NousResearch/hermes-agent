"""Focused regressions for the Cursor ACP shim."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.cursor_acp_client import CursorACPClient, _format_messages_as_prompt


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CursorACPClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CursorACPClient(acp_cwd="/tmp")

    def test_completion_ignores_hermes_tools_and_returns_stop(self) -> None:
        with patch.object(self.client, "_run_prompt", return_value=("Done.", "thinking")):
            response = self.client._create_chat_completion(
                model="agent",
                messages=[{"role": "user", "content": "fix the bug"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ],
            )

        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertIsNone(choice.message.tool_calls)
        self.assertEqual(choice.message.content, "Done.")
        self.assertEqual(choice.message.reasoning, "thinking")

    def test_prompt_format_excludes_tool_schemas(self) -> None:
        prompt = _format_messages_as_prompt(
            [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Refactor auth"},
            ],
            model="agent",
        )
        self.assertIn("Refactor auth", prompt)
        self.assertNotIn("Available tools", prompt)
        self.assertNotIn("<tool_call>", prompt)

    def test_permission_requests_are_allowed_once(self) -> None:
        proc = _FakeProcess()
        handled = self.client._handle_server_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "session/request_permission",
                "params": {},
            },
            process=proc,
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        reply = json.loads(proc.stdin.getvalue().strip())
        self.assertEqual(reply["id"], 7)
        self.assertEqual(reply["result"]["outcome"]["outcome"], "selected")
        self.assertEqual(reply["result"]["outcome"]["optionId"], "allow-once")

    def test_cursor_ask_question_is_skipped(self) -> None:
        proc = _FakeProcess()
        handled = self.client._handle_server_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "cursor/ask_question",
                "params": {"questions": []},
            },
            process=proc,
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        reply = json.loads(proc.stdin.getvalue().strip())
        self.assertEqual(reply["result"]["outcome"]["outcome"], "skipped")

    def test_cursor_create_plan_is_accepted(self) -> None:
        proc = _FakeProcess()
        handled = self.client._handle_server_message(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "cursor/create_plan",
                "params": {"plan": "do stuff"},
            },
            process=proc,
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        reply = json.loads(proc.stdin.getvalue().strip())
        self.assertEqual(reply["result"]["outcome"]["outcome"], "accepted")

    def test_agent_message_chunks_are_collected(self) -> None:
        text_parts: list[str] = []
        handled = self.client._handle_server_message(
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "hello "},
                    }
                },
            },
            process=_FakeProcess(),
            cwd="/tmp",
            text_parts=text_parts,
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        self.assertEqual(text_parts, ["hello "])

    def test_fs_write_stays_within_cwd(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as cwd_str:
            cwd = Path(cwd_str)
            client = CursorACPClient(acp_cwd=str(cwd))
            target = cwd / "out.txt"
            proc = _FakeProcess()
            handled = client._handle_server_message(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "fs/write_text_file",
                    "params": {"path": str(target), "content": "ok"},
                },
                process=proc,
                cwd=str(cwd),
                text_parts=[],
                reasoning_parts=[],
            )
            self.assertTrue(handled)
            self.assertEqual(target.read_text(), "ok")


if __name__ == "__main__":
    unittest.main()
