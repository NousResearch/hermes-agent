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

    def test_extracted_tool_calls_match_openai_sdk_shape(self) -> None:
        tool_response = (
            "I'll inspect that.\n"
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )

        with patch.object(self.client, "_run_prompt", return_value=(tool_response, "")):
            response = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "read README.md"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ],
            )

        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tool_call = choice.message.tool_calls[0]
        self.assertEqual(tool_call.id, "call_read")
        self.assertEqual(tool_call.function.name, "read_file")
        self.assertEqual(
            json.loads(tool_call.function.arguments),
            {"path": "README.md"},
        )
        self.assertEqual(dict(tool_call)["id"], "call_read")
        self.assertEqual(dict(tool_call.function)["name"], "read_file")
        self.assertEqual(choice.message.content, "I'll inspect that.")

    def test_stream_true_returns_iterable_text_chunks(self) -> None:
        with patch.object(self.client, "_run_prompt", return_value=("Hello from ACP", "")):
            stream = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )

        chunks = list(stream)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].choices[0].delta.content, "Hello from ACP")
        self.assertIsNone(chunks[0].choices[0].delta.tool_calls)
        self.assertEqual(chunks[0].choices[0].finish_reason, "stop")
        self.assertEqual(chunks[1].choices, [])
        self.assertEqual(chunks[1].usage.total_tokens, 0)

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

    def test_timeout_object_is_coerced_for_streaming_requests(self) -> None:
        captured: dict[str, float] = {}

        def fake_run_prompt(prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
            captured["timeout"] = timeout_seconds
            return "ok", ""

        timeout = type(
            "TimeoutLike",
            (),
            {"read": 12.0, "write": 5.0, "connect": 3.0, "pool": 1.0},
        )()

        with patch.object(self.client, "_run_prompt", side_effect=fake_run_prompt):
            list(
                self.client._create_chat_completion(
                    model="copilot-acp",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=timeout,
                    stream=True,
                )
            )

        self.assertEqual(captured["timeout"], 12.0)

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

            with patch(
                "agent.copilot_acp_client.get_write_denied_error",
                return_value="Write denied: protected",
                create=True,
            ):
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


class _FakeACPProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(
            "\n".join(
                [
                    json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "s1"}}),
                    json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}),
                ]
            )
            + "\n"
        )
        self.stderr = io.StringIO()
        self._terminated = False

    def poll(self):
        return None

    def terminate(self) -> None:
        self._terminated = True

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self._terminated = True


def test_run_prompt_uses_local_process_cwd_but_sends_remote_acp_cwd(tmp_path):
    process_cwd = tmp_path / "local-spawn"
    process_cwd.mkdir()
    remote_cwd = "/srv/remote/repo"
    captured = {}
    proc = _FakeACPProcess()

    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        captured["proc"] = proc
        return proc

    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="ssh",
        acp_args=["remote", "copilot", "--acp", "--stdio"],
        acp_cwd=remote_cwd,
        process_cwd=str(process_cwd),
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake):
        response_text, reasoning_text = client._run_prompt("hello", timeout_seconds=1)

    assert response_text == ""
    assert reasoning_text == ""
    assert captured["cmd"] == ["ssh", "remote", "copilot", "--acp", "--stdio"]
    assert captured["kwargs"]["cwd"] == str(process_cwd.resolve())

    sent = [
        json.loads(line)
        for line in proc.stdin.getvalue().splitlines()
        if line.strip()
    ]
    initialize = next(payload for payload in sent if payload["method"] == "initialize")
    assert "fs" not in initialize["params"]["clientCapabilities"]
    session_new = next(payload for payload in sent if payload["method"] == "session/new")
    assert session_new["params"]["cwd"] == remote_cwd


def test_ssh_acp_cwd_preserves_existing_local_path_as_remote(tmp_path):
    process_cwd = tmp_path / "local-spawn"
    process_cwd.mkdir()
    captured = {}
    proc = _FakeACPProcess()

    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return proc

    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="ssh",
        acp_args=["remote", "copilot", "--acp", "--stdio"],
        acp_cwd="/tmp",
        process_cwd=str(process_cwd),
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake):
        client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["cwd"] == str(process_cwd.resolve())
    sent = [
        json.loads(line)
        for line in proc.stdin.getvalue().splitlines()
        if line.strip()
    ]
    initialize = next(payload for payload in sent if payload["method"] == "initialize")
    assert "fs" not in initialize["params"]["clientCapabilities"]
    session_new = next(payload for payload in sent if payload["method"] == "session/new")
    assert session_new["params"]["cwd"] == "/tmp"


def test_remote_acp_cwd_rejects_local_fs_callbacks(tmp_path):
    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="ssh",
        acp_args=["remote", "copilot", "--acp", "--stdio"],
        acp_cwd="/srv/remote/repo",
        process_cwd=str(tmp_path),
    )
    process = _FakeProcess()

    handled = client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "fs/read_text_file",
            "params": {"path": "/srv/remote/repo/README.md"},
        },
        process=process,
        cwd="/srv/remote/repo",
        text_parts=[],
        reasoning_parts=[],
        fs_enabled=False,
    )

    assert handled is True
    response = json.loads(process.stdin.getvalue().strip())
    assert response["id"] == 99
    assert response["error"]["code"] == -32601
    assert "non-local session cwd" in response["error"]["message"]


def test_run_prompt_uses_acp_cwd_as_process_cwd_when_local(tmp_path):
    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())


def test_run_prompt_sends_resolved_session_cwd_for_relative_local_path(monkeypatch, tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    monkeypatch.chdir(tmp_path)
    captured = {}
    proc = _FakeACPProcess()

    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return proc

    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd="repo",
    )

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake):
        client._run_prompt("hello", timeout_seconds=1)

    expected = str(project.resolve())
    assert captured["kwargs"]["cwd"] == expected
    sent = [
        json.loads(line)
        for line in proc.stdin.getvalue().splitlines()
        if line.strip()
    ]
    initialize = next(payload for payload in sent if payload["method"] == "initialize")
    assert initialize["params"]["clientCapabilities"]["fs"] == {
        "readTextFile": True,
        "writeTextFile": True,
    }
    session_new = next(payload for payload in sent if payload["method"] == "session/new")
    assert session_new["params"]["cwd"] == expected


def test_run_prompt_preserves_real_home_when_profile_home_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()

    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

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
