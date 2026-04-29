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


# ── Permission-denied / reasoning-only response tests (PR fix) ─────────────

import threading
import queue


def _make_acp_server(messages: list[dict]) -> io.StringIO:
    """Build a fake ACP server stdin buffer that yields `messages` as JSON lines."""
    buf = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n")
    return buf


class ReasoningOnlyResponseTests(unittest.TestCase):
    """Tests for the fix: return reasoning content when no text is produced."""

    def setUp(self) -> None:
        self.client = CopilotACPClient(acp_cwd="/tmp")

    def _make_fake_proc(self, stdout_lines: list[str]) -> "_FakeProc":
        class _FakeProc:
            def __init__(self, lines):
                self.stdin = io.StringIO()
                self._lines = iter(lines)
                self.stdout = self
                self.stderr = io.StringIO()
                self._rc = None

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._lines)

            def poll(self):
                return self._rc

            def terminate(self):
                self._rc = -15

            def wait(self, timeout=None):
                return self._rc

            def kill(self):
                self._rc = -9

        return _FakeProc(stdout_lines)

    def test_end_turn_sets_prompt_completed(self) -> None:
        """_handle_server_message should set prompt_completed on end_turn."""
        completed = threading.Event()
        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "end_turn",
                    "content": {},
                }
            },
        }
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            msg,
            process=process,
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=[],
            prompt_completed=completed,
        )
        self.assertTrue(handled)
        self.assertTrue(completed.is_set(), "prompt_completed should be set after end_turn")

    def test_end_turn_without_completed_event_is_safe(self) -> None:
        """_handle_server_message should not raise if prompt_completed is None."""
        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "end_turn",
                    "content": {},
                }
            },
        }
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            msg,
            process=process,
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=[],
            prompt_completed=None,
        )
        self.assertTrue(handled)

    def test_thought_chunks_collected(self) -> None:
        """agent_thought_chunk messages should be collected into reasoning_parts."""
        reasoning_parts: list[str] = []
        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"text": "I am thinking..."},
                }
            },
        }
        process = _FakeProcess()
        self.client._handle_server_message(
            msg,
            process=process,
            cwd="/tmp",
            text_parts=[],
            reasoning_parts=reasoning_parts,
        )
        self.assertEqual(reasoning_parts, ["I am thinking..."])

    def test_reasoning_fallback_when_no_text(self) -> None:
        """If text_result is empty but reasoning_result has content, reasoning is returned as text."""
        import subprocess
        from unittest.mock import patch as _patch2, MagicMock

        # Simulate a sequence: initialize response, session/new response, thought chunks, end_turn
        server_messages = [
            # initialize response
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}}),
            # session/new response
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}}),
            # session/request_permission (triggers cancellation)
            json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "method": "session/request_permission",
                "params": {"permission": "fs"},
            }),
            # reasoning-only response (no agent_message_chunk)
            json.dumps({
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "agent_thought_chunk", "content": {"text": "I cannot access that file."}}},
            }),
            json.dumps({
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "end_turn", "content": {}}},
            }),
            # session/prompt response
            json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}),
        ]

        fake_stdout = io.StringIO("\n".join(server_messages) + "\n")
        fake_stdin = io.StringIO()

        mock_proc = MagicMock()
        mock_proc.stdin = fake_stdin
        mock_proc.stdout = fake_stdout
        mock_proc.stderr = io.StringIO()
        mock_proc.poll.return_value = None

        def fake_popen(cmd, **kwargs):
            return mock_proc

        with _patch2("agent.copilot_acp_client.subprocess.Popen", side_effect=fake_popen):
            text, reasoning = self.client._run_prompt("test prompt", timeout_seconds=5)

        # With the fix: text should be the reasoning content (fallback)
        self.assertIn("I cannot access that file.", text)
        self.assertNotEqual(text, "", "text should not be empty when reasoning content exists")


if __name__ == "__main__":
    unittest.main()
