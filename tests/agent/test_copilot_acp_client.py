"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.copilot_acp_client import (
    ACPInvocationError,
    CLAUDE_RESPONSE_FORMAT_MISMATCH,
    CopilotACPClient,
    _DEFAULT_TIMEOUT_SECONDS,
    _truncate_diagnostic,
    classify_acp_failure,
)


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


def test_create_chat_completion_uses_default_timeout(monkeypatch, tmp_path):
    client = _make_home_client(tmp_path)
    seen = {}

    def _fake_run(prompt_text, *, timeout_seconds):
        seen["timeout_seconds"] = timeout_seconds
        return ("ok", "")

    monkeypatch.setattr(client, "_run_prompt", _fake_run)
    response = client._create_chat_completion(messages=[{"role": "user", "content": "hello"}])

    assert response.choices[0].message.content == "ok"
    assert seen["timeout_seconds"] == _DEFAULT_TIMEOUT_SECONDS


def test_create_chat_completion_respects_explicit_timeout(monkeypatch, tmp_path):
    client = _make_home_client(tmp_path)
    seen = {}

    def _fake_run(prompt_text, *, timeout_seconds):
        seen["timeout_seconds"] = timeout_seconds
        return ("ok", "")

    monkeypatch.setattr(client, "_run_prompt", _fake_run)
    client._create_chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        timeout=45.0,
    )

    assert seen["timeout_seconds"] == 45.0


def test_classify_timeout_failure_retryable():
    category, retryable, message = classify_acp_failure(
        exception_text="Timed out waiting for ACP response to session/prompt after 45.0s.",
        timed_out=True,
    )

    assert category == "timeout"
    assert retryable is True
    assert "timed out" in message.lower()


def test_classify_http_500_failure_retryable():
    category, retryable, message = classify_acp_failure(
        stderr_text="API Error: 500 upstream exploded",
    )

    assert category == "http_500"
    assert retryable is True
    assert "500" in message


def test_classify_claude_response_format_mismatch_retryable():
    category, retryable, message = classify_acp_failure(
        stdout_text='event: message\ndata: {"type":"chunk"}',
        stderr_text='API Error: 500 Unexpected token \'e\', "event: mes"... is not valid JSON',
    )

    assert category == CLAUDE_RESPONSE_FORMAT_MISMATCH
    assert retryable is True
    assert "Expected JSON but received SSE/event-stream style response" in message
    assert "streaming output to a non-streaming caller" in message
    assert "not a Bash/Read tool permission problem" in message


def test_classify_unsupported_claude_acp_args_non_retryable():
    category, retryable, message = classify_acp_failure(
        stderr_text="error: unknown option '--acp'",
        exception_text="ACP process exited early while waiting for initialize.",
        command_text="claude",
        command_args=("--acp", "--stdio"),
    )

    assert category == "unsupported_cli_args"
    assert retryable is False
    assert "ACP subprocess failed before protocol startup." in message
    assert "does not support the requested ACP args" in message
    assert "Current claude CLI does not expose --acp." in message


def test_classify_non_retryable_permission_failure():
    category, retryable, message = classify_acp_failure(
        stderr_text="Permission denied while starting tool",
    )

    assert category == "non_transient"
    assert retryable is False
    assert "non-retryable" in message


def test_run_prompt_retries_once_on_transient_failure(monkeypatch, tmp_path):
    client = _make_home_client(tmp_path)
    calls = {"count": 0}

    def _fake_once(prompt_text, *, timeout_seconds):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ACPInvocationError(
                "ACP subprocess timed out waiting for a response.",
                category="timeout",
                retryable=True,
            )
        return ("ok", "")

    monkeypatch.setattr(client, "_run_prompt_once", _fake_once)
    result = client._run_prompt("hello", timeout_seconds=30.0)

    assert result == ("ok", "")
    assert calls["count"] == 2


def test_run_prompt_does_not_retry_non_transient_failure(monkeypatch, tmp_path):
    client = _make_home_client(tmp_path)
    calls = {"count": 0}

    def _fake_once(prompt_text, *, timeout_seconds):
        calls["count"] += 1
        raise ACPInvocationError(
            "Permission denied",
            category="non_transient",
            retryable=False,
        )

    monkeypatch.setattr(client, "_run_prompt_once", _fake_once)

    with pytest.raises(ACPInvocationError, match="Permission denied"):
        client._run_prompt("hello", timeout_seconds=30.0)

    assert calls["count"] == 1


def test_run_prompt_does_not_retry_unsupported_cli_args(monkeypatch, tmp_path):
    client = _make_home_client(tmp_path)
    calls = {"count": 0}

    def _fake_once(prompt_text, *, timeout_seconds):
        calls["count"] += 1
        raise ACPInvocationError(
            "ACP subprocess failed before protocol startup.\n"
            "The configured command does not support the requested ACP args.\n"
            "Current claude CLI does not expose --acp.",
            category="unsupported_cli_args",
            retryable=False,
        )

    monkeypatch.setattr(client, "_run_prompt_once", _fake_once)
    with pytest.raises(ACPInvocationError, match="Current claude CLI does not expose --acp."):
        client._run_prompt("hello", timeout_seconds=1)

    assert calls["count"] == 1


def test_run_prompt_reports_retry_exhaustion(monkeypatch, tmp_path):
    client = _make_home_client(tmp_path)

    def _fake_once(prompt_text, *, timeout_seconds):
        raise ACPInvocationError(
            "ACP subprocess returned malformed JSON.",
            category="malformed_json",
            retryable=True,
        )

    monkeypatch.setattr(client, "_run_prompt_once", _fake_once)

    with pytest.raises(ACPInvocationError, match="Retried once and still failed"):
        client._run_prompt("hello", timeout_seconds=30.0)


def test_truncate_diagnostic_marks_output_truncated():
    text = "x" * 5000
    truncated = _truncate_diagnostic(text, limit=120)

    assert truncated.endswith("[output truncated]")
    assert len(truncated) <= 120
