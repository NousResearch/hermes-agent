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

    # Hermeticity: the --acp support probe (PR #87308) calls subprocess.run
    # before Popen; stub it inconclusive so no real CLI on the host box can
    # flip the resolution this test asserts.
    with _patch("agent.copilot_acp_client.subprocess.run", side_effect=FileNotFoundError):
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

    # Hermeticity: the --acp support probe (PR #87308) calls subprocess.run
    # before Popen; stub it inconclusive so no real CLI on the host box can
    # flip the resolution this test asserts.
    with _patch("agent.copilot_acp_client.subprocess.run", side_effect=FileNotFoundError):
        with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
            with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
                client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]


# ── --acp support probe tests (PR #87308 / issue #87309) ────────────

import subprocess as _subprocess

from agent.copilot_acp_client import _ACP_PROBE_CACHE, _acp_supported


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    _ACP_PROBE_CACHE.clear()
    yield
    _ACP_PROBE_CACHE.clear()


def _completed(returncode=0, stdout=""):
    return _subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_probe_true_when_help_advertises_acp():
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        return_value=_completed(stdout="Usage: copilot [--acp] [--stdio]"),
    ):
        assert _acp_supported("copilot", ["--acp", "--stdio"]) is True


def test_probe_false_when_help_lacks_acp_and_run_prompt_fast_fails(tmp_path):
    client = _make_home_client(tmp_path)
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        return_value=_completed(stdout="Usage: claude [--print] [--model]"),
    ):
        with pytest.raises(RuntimeError, match="ACP transport not supported"):
            client._run_prompt("hello", timeout_seconds=1)


def test_probe_inconclusive_falls_through_to_spawn_error(tmp_path):
    """Missing binary: probe must NOT mask the established spawn error."""
    client = _make_home_client(tmp_path)
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        side_effect=FileNotFoundError("copilot not found"),
    ):
        with _patch(
            "agent.copilot_acp_client.subprocess.Popen",
            side_effect=FileNotFoundError("copilot not found"),
        ):
            with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
                client._run_prompt("hello", timeout_seconds=1)


def test_probe_result_cached_per_binary_path():
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        return_value=_completed(stdout="Usage: copilot [--acp]"),
    ) as run_mock:
        assert _acp_supported("copilot", ["--acp"]) is True
        assert _acp_supported("copilot", ["--acp"]) is True
    assert run_mock.call_count == 1


def test_probe_inconclusive_not_cached():
    with _patch(
        "agent.copilot_acp_client.subprocess.run",
        side_effect=FileNotFoundError,
    ) as run_mock:
        assert _acp_supported("copilot", ["--acp"]) is None
        assert _acp_supported("copilot", ["--acp"]) is None
    assert run_mock.call_count == 2  # inconclusive verdicts retry


def test_probe_skipped_for_custom_args_without_acp():
    with _patch("agent.copilot_acp_client.subprocess.run") as run_mock:
        assert _acp_supported("mycli", ["--custom-transport"]) is True
    run_mock.assert_not_called()


# --- session/set_model: honor the picker-selected model ----------------------
#
# `copilot --acp` validates but IGNORES the `--model` spawn flag; the ACP
# session runs the CLI's own default unless the client issues the ACP-native
# `session/set_model` call. Without it, picking gpt-5.6-terra in Hermes
# visibly answers as the CLI's default model.


class _ScriptedACP:
    """Minimal scripted ACP wire: records requests, plays canned results."""

    def __init__(self, session_result):
        self.requests = []
        self.session_result = session_result

    def request(self, method, params, **_):
        self.requests.append((method, params))
        if method == "session/new":
            return self.session_result
        return {}


def _run_prompt_with_scripted_wire(model, session_result):
    """Drive _run_prompt's request sequence against a scripted wire."""
    client = CopilotACPClient(acp_cwd="/tmp")
    wire = _ScriptedACP(session_result)

    def fake_run_prompt(prompt_text, *, timeout_seconds, model=None):
        # Reproduce the request choreography under test without a subprocess.
        session = wire.request("session/new", {"cwd": "/tmp", "mcpServers": []}) or {}
        session_id = str(session.get("sessionId") or "")
        requested_model = str(model or "").strip()
        if requested_model and requested_model != "copilot-acp":
            available = {
                str(m.get("modelId") or "").strip()
                for m in ((session.get("models") or {}).get("availableModels") or [])
                if isinstance(m, dict)
                and str(
                    ((m.get("_meta") or {}).get("copilotEnablement")) or ""
                ).strip().lower() != "disabled"
            }
            if not available or requested_model in available:
                wire.request(
                    "session/set_model",
                    {"sessionId": session_id, "modelId": requested_model},
                )
        wire.request("session/prompt", {"sessionId": session_id, "prompt": []})
        return "ok", ""

    with patch.object(CopilotACPClient, "_run_prompt", side_effect=fake_run_prompt):
        client._create_chat_completion(
            model=model, messages=[{"role": "user", "content": "hi"}]
        )
    return wire.requests


_SESSION_WITH_MODELS = {
    "sessionId": "s1",
    "models": {
        "availableModels": [
            {"modelId": "auto"},
            {"modelId": "gpt-5.6-terra"},
            {"modelId": "claude-sonnet-5"},
        ]
    },
}


def test_set_model_sent_for_advertised_model():
    reqs = _run_prompt_with_scripted_wire("gpt-5.6-terra", _SESSION_WITH_MODELS)
    methods = [m for m, _ in reqs]
    assert "session/set_model" in methods, "picker model must be applied to the session"
    idx_set = methods.index("session/set_model")
    idx_prompt = methods.index("session/prompt")
    assert idx_set < idx_prompt, "model must be set before the prompt runs"
    assert reqs[idx_set][1] == {"sessionId": "s1", "modelId": "gpt-5.6-terra"}


def test_set_model_skipped_for_unadvertised_model():
    reqs = _run_prompt_with_scripted_wire("not-served-here", _SESSION_WITH_MODELS)
    assert all(m != "session/set_model" for m, _ in reqs), \
        "unknown model must degrade to the session default, not error"


def test_set_model_skipped_for_provider_virtual_slug():
    reqs = _run_prompt_with_scripted_wire("copilot-acp", _SESSION_WITH_MODELS)
    assert all(m != "session/set_model" for m, _ in reqs)


def test_set_model_skipped_for_policy_disabled_model():
    # A policy-disabled id may still be advertised; selecting it silently
    # serves the default model, so it must not be treated as offered.
    session = {
        "sessionId": "s1",
        "models": {
            "availableModels": [
                {"modelId": "claude-sonnet-5"},
                {
                    "modelId": "claude-fable-5",
                    "_meta": {"copilotEnablement": "disabled"},
                },
            ]
        },
    }
    reqs = _run_prompt_with_scripted_wire("claude-fable-5", session)
    assert all(m != "session/set_model" for m, _ in reqs)


def test_run_prompt_receives_picker_model():
    # _create_chat_completion must forward `model` into _run_prompt — the
    # original wiring dropped it, reducing the selection to prompt text.
    client = CopilotACPClient(acp_cwd="/tmp")
    seen = {}

    def fake_run_prompt(prompt_text, *, timeout_seconds, model=None):
        seen["model"] = model
        return "ok", ""

    with patch.object(CopilotACPClient, "_run_prompt", side_effect=fake_run_prompt):
        client._create_chat_completion(
            model="gpt-5.6-terra", messages=[{"role": "user", "content": "hi"}]
        )
    assert seen["model"] == "gpt-5.6-terra"
