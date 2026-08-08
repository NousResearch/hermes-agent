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



    def test_stream_true_preserves_tool_call_deltas(self) -> None:
        tool_response = (
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )

        with patch.object(self.client, "_run_prompt", return_value=(tool_response, "")):
            stream = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "read README.md"}],
                stream=True,
            )

        chunks = list(stream)
        delta = chunks[0].choices[0].delta
        self.assertIsNone(delta.content)
        self.assertEqual(chunks[0].choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(delta.tool_calls), 1)
        tool_delta = delta.tool_calls[0]
        self.assertEqual(tool_delta.index, 0)
        self.assertEqual(tool_delta.id, "call_read")
        self.assertEqual(tool_delta.function.name, "read_file")
        self.assertEqual(
            json.loads(tool_delta.function.arguments),
            {"path": "README.md"},
        )
        self.assertEqual(chunks[1].choices, [])


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

    def test_fs_read_text_file_decodes_as_utf8_under_non_utf8_locale(self) -> None:
        """Regression for #18637 (bug 2): fs/read_text_file used
        ``path.read_text()`` with no explicit encoding, so on Windows
        GBK/CP932/CP949 locales the Copilot read_file tool crashed on any
        source file with non-ASCII content (e.g. a CJK comment, an em dash,
        or UTF-8 BOM)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "note.md"
            target.write_text("# 中文标题\nem dash — here\n", encoding="utf-8")

            original_read_text = Path.read_text

            def strict_read_text(self, encoding=None, errors=None, **kwargs):
                if self == target and encoding != "utf-8":
                    raise UnicodeDecodeError(
                        "gbk", b"\x94", 0, 1, "illegal multibyte sequence"
                    )
                return original_read_text(
                    self, encoding=encoding, errors=errors, **kwargs
                )

            with patch.object(Path, "read_text", strict_read_text):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 10,
                        "method": "fs/read_text_file",
                        "params": {"path": str(target)},
                    },
                    cwd=str(root),
                )

        self.assertNotIn("error", response)
        content = ((response.get("result") or {}).get("content") or "")
        self.assertIn("中文标题", content)
        self.assertIn("em dash —", content)



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
        self.assertIn("HERMES_WRITE_SAFE_ROOT", str(response["error"]))
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


def test_run_prompt_preserves_real_home_when_profile_home_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()

    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Hermeticity: an ambient HERMES_REAL_HOME (exported by Hermes' own
    # terminal contract on dev boxes) outranks HOME in the candidate ladder,
    # and an ambient TERMINAL_HOME_MODE would change the policy under test.
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
    # Hermeticity: get_subprocess_home()'s auto mode prefers the profile home
    # when is_container() is True — on a containerized CI runner that real
    # probe flips the resolution this test asserts. The host/VM branch is the
    # contract under test; pin containment off.
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(real_home)
    assert captured["kwargs"]["env"]["HERMES_REAL_HOME"] == str(real_home)


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


# ── ACP mixed text + tool_use regressions (#33636) ─────────────────────

def test_handle_server_message_preserves_text_before_tool_use():
    """Structured tool_use updates are appended without dropping prior text."""
    from unittest.mock import MagicMock

    from agent.copilot_acp_client import CopilotACPClient, _extract_tool_calls_from_text

    client = CopilotACPClient()
    process = MagicMock()
    process.stdin = MagicMock()
    text_parts = ["I will inspect the file first.\n"]
    handled = client._handle_server_message(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "tool_use",
                    "content": {
                        "name": "read_file",
                        "id": "call_33636",
                        "input": {"path": "README.md"},
                    },
                }
            },
        },
        process=process,
        cwd="/tmp",
        text_parts=text_parts,
        reasoning_parts=[],
    )

    assert handled is True
    combined = "".join(text_parts)
    tool_calls, cleaned = _extract_tool_calls_from_text(combined)
    assert "I will inspect the file first." in cleaned
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_33636"
    assert tool_calls[0].function.name == "read_file"
    assert tool_calls[0].function.arguments == '{"path": "README.md"}'


def test_handle_server_message_tool_use_accepts_string_arguments():
    """tool_use arguments already encoded as JSON text are not double encoded."""
    from unittest.mock import MagicMock

    from agent.copilot_acp_client import CopilotACPClient, _extract_tool_calls_from_text

    client = CopilotACPClient()
    process = MagicMock()
    process.stdin = MagicMock()
    text_parts = []
    client._handle_server_message(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "tool_use",
                    "content": {
                        "name": "terminal",
                        "id": "call_args",
                        "input": '{"command": "pwd"}',
                    },
                }
            },
        },
        process=process,
        cwd="/tmp",
        text_parts=text_parts,
        reasoning_parts=[],
    )

    tool_calls, cleaned = _extract_tool_calls_from_text("".join(text_parts))
    assert cleaned == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.arguments == '{"command": "pwd"}'


def test_handle_server_message_tool_result_is_not_reinvoked():
    """tool_result updates preserve narration without becoming a second tool call."""
    from unittest.mock import MagicMock

    from agent.copilot_acp_client import CopilotACPClient, _extract_tool_calls_from_text

    client = CopilotACPClient()
    process = MagicMock()
    process.stdin = MagicMock()
    text_parts = ["I will inspect the file first.\n"]

    for update in (
        {
            "sessionUpdate": "tool_use",
            "content": {
                "name": "read_file",
                "id": "call_once",
                "input": {"path": "README.md"},
            },
        },
        {
            "sessionUpdate": "tool_result",
            "content": {
                "name": "read_file",
                "id": "call_once",
                "output": "README contents preserved",
            },
        },
    ):
        handled = client._handle_server_message(
            {
                "method": "session/update",
                "params": {"update": update},
            },
            process=process,
            cwd="/tmp",
            text_parts=text_parts,
            reasoning_parts=[],
        )
        assert handled is True

    tool_calls, cleaned = _extract_tool_calls_from_text("".join(text_parts))
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_once"
    assert tool_calls[0].function.name == "read_file"
    assert "I will inspect the file first." in cleaned
    assert "README contents preserved" in cleaned


class _FakeContentBlock:
    """Typed block object that Copilot ACP may emit instead of a plain dict."""

    def __init__(self, text: str) -> None:
        self.text = text


def test_session_update_agent_message_chunk_preserves_block_text():
    """Typed content blocks (non-dict) preserve their text for agent_message_chunk.

    Regression from jdell64 (#33668): Copilot ACP can surface content as
    objects with .text attributes rather than plain dicts.
    """
    from unittest.mock import MagicMock

    from agent.copilot_acp_client import CopilotACPClient

    client = CopilotACPClient()
    process = MagicMock()
    process.stdin = MagicMock()
    text_parts: list[str] = []
    handled = client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": _FakeContentBlock("Removing both keys now"),
                }
            },
        },
        process=process,
        cwd="/tmp",
        text_parts=text_parts,
        reasoning_parts=[],
    )

    assert handled is True
    assert text_parts == ["Removing both keys now"]
