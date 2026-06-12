"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient


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
            blocked.write_text('{"token":"sk-test-secret-1234567890"}')

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
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")

            # agent.redact snapshots HERMES_REDACT_SECRETS at import time into
            # _REDACT_ENABLED, so patching os.environ is a no-op. Flip the
            # module-level constant directly for the duration of the call.
            with patch("agent.redact._REDACT_ENABLED", True):
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
        self.assertIn("OPENAI_API_KEY=", content)

    def test_write_text_file_reuses_write_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            target = home / ".ssh" / "id_rsa"
            target.parent.mkdir(parents=True, exist_ok=True)

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


# ── HOME env propagation tests (from PR #11285) ─────────────────────

from unittest.mock import patch as _patch
import pytest


def _make_home_client(tmp_path):
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )


def _fake_popen_capture(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise FileNotFoundError("copilot not found")
    return _fake


def test_run_prompt_prefers_profile_home_when_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    profile_home = hermes_home / "home"
    profile_home.mkdir(parents=True)

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(profile_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]

# Tests for <tool_use> XML parsing (fix for issue #45139)

from agent.copilot_acp_client import _extract_tool_calls_from_text, _TOOL_USE_XML_RE


class ToolUseXMLParsingTests(unittest.TestCase):
    """Tests for parsing Anthropic/Claude native <tool_use> XML tags."""

    def test_self_closing_tool_use_tag_with_json_arguments(self):
        text = '<tool_use name="terminal" arguments="{\"command\": \"ls\"}" id="call_123" />'
        tool_calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "terminal")
        self.assertEqual(tool_calls[0].function.arguments, '{"command": "ls"}')
        self.assertEqual(tool_calls[0].id, "call_123")
        self.assertEqual(cleaned, "")

    def test_self_closing_tool_use_tag_with_simple_arguments(self):
        text = '<tool_use name="terminal" arguments="ls -la" id="call_456" />'
        tool_calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "terminal")
        self.assertEqual(tool_calls[0].function.arguments, "ls -la")
        self.assertEqual(tool_calls[0].id, "call_456")
        self.assertEqual(cleaned, "")

    def test_self_closing_tool_use_tag_with_input_attribute(self):
        text = '<tool_use name="read_file" input="path=/tmp/test.txt" id="call_789" />'
        tool_calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "read_file")
        self.assertEqual(tool_calls[0].function.arguments, "path=/tmp/test.txt")
        self.assertEqual(tool_calls[0].id, "call_789")
        self.assertEqual(cleaned, "")

    def test_opening_closing_tool_use_tag_with_json(self):
        # The arguments field contains a JSON string, so inner quotes must be DOUBLE-escaped
        # for the outer JSON to be valid: "{\"query\": \"test\"}" -> arguments="{\"query\": \"test\"}"
        json_content = '{"id": "call_999", "type": "function", "function": {"name": "web_search", "arguments": "{\\\"query\\\": \\\"test\\\"}"}}'
        text = f"<tool_use>{json_content}</tool_use>"
        tool_calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "web_search")
        self.assertEqual(tool_calls[0].function.arguments, '{"query": "test"}')
        self.assertEqual(tool_calls[0].id, "call_999")
        self.assertEqual(cleaned, "")

    def test_mixed_tool_use_and_hermes_format(self):
        text = 'Some text before. <tool_use name="terminal" arguments="ls" id="call_1" /> More text.'
        tool_calls, cleaned = _extract_tool_calls_from_text(text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "terminal")
        self.assertIn("Some text before", cleaned)
        self.assertIn("More text", cleaned)

if __name__ == "__main__":
    unittest.main()
