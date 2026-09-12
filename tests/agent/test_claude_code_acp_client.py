"""Focused regressions for the Claude Code ACP shim safety layer.

Mirrors tests/agent/test_copilot_acp_client.py -- same coverage shape
(tool-call stream-chunk parsing, fs/read_text_file redaction, the UTF-8
decode-under-non-UTF-8-locale regression, fs/write_text_file safe-root
enforcement, HOME env propagation), adapted for ClaudeCodeACPClient /
cc-acp instead of CopilotACPClient / copilot.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.claude_code_acp_client import ClaudeCodeACPClient
from agent.claude_code_acp_client import _build_subprocess_env


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class _FakeACPProcess:
    """Fake subprocess.Popen result for exercising the full _run_prompt
    JSON-RPC loop against a scripted stdout stream, without spawning a
    real cc-acp process."""

    def __init__(self, stdout_lines: list) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(stdout_lines) + "\n")
        self.stderr = io.StringIO("")

    def poll(self):
        return None

    def terminate(self) -> None:
        pass

    def wait(self, timeout=None) -> None:
        pass

    def kill(self) -> None:
        pass


class ClaudeCodeACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = ClaudeCodeACPClient(acp_cwd="/tmp")



    def test_run_prompt_ignores_non_dict_json_lines_on_stdout(self) -> None:
        """Regression, found via a live spawn of the real cc-acp binary
        (claude-code-acp@0.1.1, 2026-08-08): cc-acp writes pretty-printed
        multi-line JSON.stringify debug/log output directly to stdout,
        intermixed with real single-line JSON-RPC messages. Individual
        lines from that pretty-print can themselves be standalone-valid
        JSON that is NOT an object -- e.g. the last element of an array
        with no trailing comma (`"SlashCommand"`) parses to a bare Python
        str via json.loads(), not a dict. Before the fix, that bare str
        reached _handle_server_message()/_request() and crashed on the
        first .get() call (AttributeError: 'str' object has no attribute
        'get'). This test drives the real _run_prompt() JSON-RPC loop
        against a scripted stdout stream reproducing exactly that shape."""
        stdout_lines = [
            '"SlashCommand"',  # noise: valid JSON, but not an object
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-123"}}),
            json.dumps({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess-123",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "PONG"},
                    },
                },
            }),
            json.dumps({"jsonrpc": "2.0", "id": 3, "result": None}),
        ]
        fake_proc = _FakeACPProcess(stdout_lines)

        with patch("agent.claude_code_acp_client.subprocess.Popen", return_value=fake_proc):
            text, reasoning = self.client._run_prompt("hi", timeout_seconds=5)

        self.assertEqual(text, "PONG")
        self.assertEqual(reasoning, "")

    def test_stream_true_preserves_tool_call_deltas(self) -> None:
        tool_response = (
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )

        with patch.object(self.client, "_run_prompt", return_value=(tool_response, "")):
            stream = self.client._create_chat_completion(
                model="claude-code-acp",
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
        or UTF-8 BOM). Same fs/read_text_file handler is reused verbatim by
        ClaudeCodeACPClient, so the same regression coverage applies here."""
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



    def test_permission_request_is_always_denied(self) -> None:
        """session/request_permission must be unconditionally denied (no
        interactive TTY on this path) -- same conservative default as
        CopilotACPClient. Mirrors the qa/ai-eng-warden request for explicit
        coverage of this dispatch branch, not just the fs/* branches."""
        response = self._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "session/request_permission",
                "params": {"sessionId": "sess-1", "toolCall": {"name": "Bash"}},
            },
            cwd="/tmp",
        )
        self.assertEqual(
            (response.get("result") or {}).get("outcome", {}).get("outcome"),
            "cancelled",
        )
        self.assertNotIn("error", response)

    def test_build_subprocess_env_defaults_to_disallowing_native_tools(self) -> None:
        """Regression, found via live-testing the real cc-acp binary
        (claude-code-acp@0.1.1, 2026-08-08): with CLAUDE_ALLOWED_TOOLS and
        CLAUDE_DISALLOWED_TOOLS both unset, cc-acp's bundled Claude Agent SDK
        silently executed its native Read tool against the real filesystem
        -- with the real HOME, no cwd confinement, no redaction -- and
        WITHOUT ever sending an ACP session/request_permission call to this
        client (a prompt asking it to read a canary file returned the real
        file content verbatim). This contradicts the package's own README
        ("we do not enable any tools unless you specify them"). The fix:
        _build_subprocess_env() must default CLAUDE_DISALLOWED_TOOLS to the
        full native-tool list unless the operator already made an explicit
        choice. Live-reverified after the fix: the same canary-read prompt,
        run through the real ClaudeCodeACPClient with no manual env
        override, no longer returned the file content."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLAUDE_ALLOWED_TOOLS", None)
            os.environ.pop("CLAUDE_DISALLOWED_TOOLS", None)
            env = _build_subprocess_env()

        disallowed = env.get("CLAUDE_DISALLOWED_TOOLS", "")
        for tool in ("Read", "Write", "Edit", "Bash", "Glob", "Grep", "WebFetch"):
            self.assertIn(tool, disallowed)

    def test_build_subprocess_env_respects_explicit_allowed_tools_override(self) -> None:
        """An operator-set CLAUDE_ALLOWED_TOOLS must win -- the safe default
        must not clobber an explicit, intentional choice."""
        with patch.dict(os.environ, {"CLAUDE_ALLOWED_TOOLS": "Read"}, clear=False):
            os.environ.pop("CLAUDE_DISALLOWED_TOOLS", None)
            env = _build_subprocess_env()

        self.assertEqual(env.get("CLAUDE_ALLOWED_TOOLS"), "Read")
        self.assertNotIn("CLAUDE_DISALLOWED_TOOLS", env)

    def test_build_subprocess_env_respects_explicit_disallowed_tools_override(self) -> None:
        """An operator-set CLAUDE_DISALLOWED_TOOLS must win too -- the safe
        default only fills in when the operator made no choice at all."""
        with patch.dict(os.environ, {"CLAUDE_DISALLOWED_TOOLS": "Bash"}, clear=False):
            os.environ.pop("CLAUDE_ALLOWED_TOOLS", None)
            env = _build_subprocess_env()

        self.assertEqual(env.get("CLAUDE_DISALLOWED_TOOLS"), "Bash")

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


# ── HOME env propagation tests ───────────────────────────────────────


def _make_home_client(tmp_path):
    return ClaudeCodeACPClient(
        api_key="claude-code-acp",
        base_url="acp://claude-code",
        acp_command="cc-acp",
        acp_args=[],
        acp_cwd=str(tmp_path),
    )


def _fake_popen_capture(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise FileNotFoundError("cc-acp not found")
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
    # when is_container() is True -- on a containerized CI runner that real
    # probe flips the resolution this test asserts. The host/VM branch is the
    # contract under test; pin containment off.
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)

    captured = {}
    client = _make_home_client(tmp_path)

    with patch("agent.claude_code_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Claude Code ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(real_home)
    assert captured["kwargs"]["env"]["HERMES_REAL_HOME"] == str(real_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with patch("agent.claude_code_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Claude Code ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]


if __name__ == "__main__":
    unittest.main()
