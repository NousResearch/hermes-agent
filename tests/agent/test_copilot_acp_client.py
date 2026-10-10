"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import io
import json
import os
import sys
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
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012", encoding="utf-8")

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
                # The repo encoding policy makes reads BOM-tolerant, so both
                # UTF-8 family codecs satisfy this regression guard.
                if self == target and encoding not in ("utf-8", "utf-8-sig"):
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


# --- session model selection -------------------------------------------------


def _session_with_config_options():
    return {
        "sessionId": "s1",
        "configOptions": [
            {
                "id": "model",
                "category": "model",
                "type": "select",
                "currentValue": "auto",
                "options": [
                    {"value": "auto", "name": "Auto"},
                    {"value": "gpt-5.6-terra", "name": "GPT-5.6 Terra"},
                    {
                        "value": "claude-fable-5",
                        "name": "Claude Fable 5",
                        "_meta": {"copilotEnablement": "disabled"},
                    },
                ],
            }
        ],
    }


def test_model_selection_prefers_stable_config_option():
    from agent.copilot_acp_client import _model_selection_request

    assert _model_selection_request(
        _session_with_config_options(), "gpt-5.6-terra"
    ) == (
        "session/set_config_option",
        {"sessionId": "s1", "configId": "model", "value": "gpt-5.6-terra"},
    )


def test_model_selection_rejects_disabled_config_option():
    from agent.copilot_acp_client import _model_selection_request

    assert _model_selection_request(
        _session_with_config_options(), "claude-fable-5"
    ) is None


def test_model_selection_rejects_unknown_config_option():
    from agent.copilot_acp_client import _model_selection_request

    assert _model_selection_request(
        _session_with_config_options(), "not-served-here"
    ) is None


def test_model_selection_falls_back_to_legacy_extension():
    from agent.copilot_acp_client import _model_selection_request

    legacy_session = {
        "sessionId": "s1",
        "models": {
            "availableModels": [
                {"modelId": "auto"},
                {"modelId": "gpt-5.6-terra"},
            ]
        },
    }
    assert _model_selection_request(legacy_session, "gpt-5.6-terra") == (
        "session/set_model",
        {"sessionId": "s1", "modelId": "gpt-5.6-terra"},
    )


def test_model_selection_skips_provider_virtual_slug():
    from agent.copilot_acp_client import _model_selection_request

    assert _model_selection_request(
        _session_with_config_options(), "copilot-acp"
    ) is None


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


def test_list_models_reads_enabled_session_config_options(tmp_path):
    server = tmp_path / "fake_copilot_acp.py"
    server.write_text(
        """import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": 1}
    elif method == "session/new":
        result = {
            "sessionId": "catalog-session",
            "configOptions": [{
                "id": "model",
                "category": "model",
                "options": [
                    {"value": "auto"},
                    {"value": "gpt-5.6-terra"},
                    {"value": "gpt-5.6-terra"},
                    {"value": "claude-fable-5", "_meta": {"copilotEnablement": "disabled"}},
                ],
            }],
            "models": {"availableModels": [{"modelId": "stale-legacy-model"}]},
        }
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    client = CopilotACPClient(
        command=sys.executable,
        args=[str(server)],
        acp_cwd=str(tmp_path),
    )

    assert client.list_models(timeout_seconds=30) == ["auto", "gpt-5.6-terra"]
    assert client.is_closed is True


def test_model_discovery_does_not_allow_file_requests(tmp_path):
    target = tmp_path / "should-not-be-read.txt"
    target.write_text("private", encoding="utf-8")
    server = tmp_path / "fake_copilot_acp_fs_request.py"
    server.write_text(
        f"""import json
import sys

initialize = json.loads(sys.stdin.readline())
print(json.dumps({{"jsonrpc": "2.0", "id": initialize["id"], "result": {{"protocolVersion": 1}}}}), flush=True)
session = json.loads(sys.stdin.readline())
print(json.dumps({{"jsonrpc": "2.0", "id": 99, "method": "fs/read_text_file", "params": {{"path": {str(target)!r}}}}}), flush=True)
file_response = json.loads(sys.stdin.readline())
assert file_response["error"]["code"] == -32601
print(json.dumps({{"jsonrpc": "2.0", "id": session["id"], "result": {{"sessionId": "catalog-session", "configOptions": [{{"id": "model", "options": [{{"value": "gpt-5.6-sol"}}]}}]}}}}), flush=True)
""",
        encoding="utf-8",
    )
    client = CopilotACPClient(
        command=sys.executable,
        args=[str(server)],
        acp_cwd=str(tmp_path),
    )

    assert client.list_models(timeout_seconds=30) == ["gpt-5.6-sol"]


# --- concurrent sessions on a shared client ---------------------------------
#
# Aux clients are cached per provider config and served to every concurrent
# caller, so one CopilotACPClient can run several ACP sessions at once. Each
# session must reap ITS OWN child on exit: reaping whatever most recently
# claimed shared state kills a sibling's live process and leaks the session's
# own.


_FAKE_ACP_SERVER = """import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": 1}
    elif method == "session/new":
        result = {"sessionId": "s1"}
    elif method == "session/prompt":
        time.sleep(0.4)
        print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "done"}},
        }}), flush=True)
        result = {"stopReason": "end_turn"}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _recording_client(tmp_path, spawned):
    server = tmp_path / "fake_copilot_acp.py"
    server.write_text(_FAKE_ACP_SERVER, encoding="utf-8")
    client = CopilotACPClient(command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path))
    real_spawn = client._spawn

    def record_spawn():
        proc = real_spawn()
        spawned.append(proc)
        return proc

    client._spawn = record_spawn
    return client


def test_overlapping_sessions_reap_their_own_process(tmp_path):
    spawned = []
    client = _recording_client(tmp_path, spawned)

    session_a = client._session(30)
    session_b = client._session(30)
    session_a.__enter__()
    session_b.__enter__()

    proc_a, proc_b = spawned
    session_a.__exit__(None, None, None)

    leaked = proc_a.poll() is None
    killed = proc_b.poll() is not None
    assert not leaked and not killed, (
        f"session A teardown: own child leaked={leaked}, sibling process killed={killed}"
    )

    assert client.is_closed is False, "a shared client is not closed while a sibling session is live"

    session_b.__exit__(None, None, None)
    assert proc_b.poll() is not None
    assert client.is_closed is True, "the last session to drain still flips is_closed for single-session callers"


def test_close_terminates_every_live_session_process(tmp_path):
    spawned = []
    client = _recording_client(tmp_path, spawned)

    session_a = client._session(30)
    session_b = client._session(30)
    session_a.__enter__()
    session_b.__enter__()

    client.close()

    assert all(proc.poll() is not None for proc in spawned)


_CRASHING_ACP_CLI = """
import subprocess, sys
# Hand stderr to a straggler that writes the crash text after this process has exited, so
# poll() reports the exit before the client has read a single stderr line.
subprocess.Popen([sys.executable, "-c",
                  "import sys, time; time.sleep(0.3); print('fatal: agent segfaulted', file=sys.stderr)"])
sys.exit(3)
"""


def test_cli_death_is_reported_as_a_crash_not_a_timeout(tmp_path):
    """A CLI that dies must surface its crash text even when stderr lags the exit: the old path
    raised TimeoutError there, which the agent loop retries on a different (larger) budget."""
    server = tmp_path / "crashing_acp.py"
    server.write_text(_CRASHING_ACP_CLI, encoding="utf-8")
    client = CopilotACPClient(command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path))
    try:
        with client._session(30):
            pass
    except RuntimeError as exc:
        assert "exited early: fatal: agent segfaulted" in str(exc)
    else:
        raise AssertionError("session on a dead CLI must raise")


_LAUNCHER_WITH_NATIVE_CHILD = (
    "import subprocess, sys, time\n"
    "# Shape of the npm @github/copilot launcher (#124835): a thin wrapper that spawns a\n"
    "# worker as a child, stays alive until terminated, and orphans that worker when only\n"
    "# the wrapper itself is signalled.\n"
    "w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "print(w.pid, flush=True)\n"
    "while True:\n"
    "    time.sleep(0.1)\n"
)

# The crash-path twin: the launcher itself dies right after spawning the worker,
# so the worker is orphaned while the teardown still owes it a signal.
_LAUNCHER_THAT_DIES_AND_LEAVES_A_WORKER = (
    "import subprocess, sys\n"
    "w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
    "print(w.pid, flush=True)\n"
)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_takes_the_native_descendant_with_it():
    """Terminating only the tracked launcher used to orphan the launcher's native
    child, which a container's PID 1 then never reaped — 16 leaked processes
    ≈ 2.6 GiB RSS in the reporter's deployment (#124835). The session spawn puts
    the tree in its own process group, so teardown must reach the descendant."""
    import contextlib
    import subprocess as sp
    import time as _time

    proc = sp.Popen(  # mirrors _spawn's own-process-group contract
        [sys.executable, "-c", _LAUNCHER_WITH_NATIVE_CHILD],
        stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, process_group=0,
    )
    try:
        worker_pid = int(proc.stdout.readline().decode().strip())
        assert worker_pid > 0

        CopilotACPClient._terminate_process(proc)

        deadline = _time.monotonic() + 5
        while _time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except OSError:
                break
            _time.sleep(0.05)
        else:
            with contextlib.suppress(OSError):
                os.kill(worker_pid, 9)
            raise AssertionError("the native worker survived the launcher's termination")
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_never_signals_a_shared_process_group(monkeypatch):
    """A spawn that did not get its own process group must fall back to
    launcher-only termination: signalling the shared group would take unrelated
    processes down with the session (#124835)."""
    import agent.copilot_acp_client as mod

    signalled, terminated = [], []

    class _Proc:
        pid = 111

        def terminate(self):
            terminated.append(111)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: 4242, raising=False)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False)
    CopilotACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    assert terminated == [111]
    assert signalled == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_sweeps_a_group_that_ignores_the_term(monkeypatch):
    """A descendant that ignores the graceful TERM must not outlive the session:
    after the launcher is reaped, a live group (pgid still claimed by a member)
    gets the KILL sweep (#124835)."""
    import signal as _signal

    import agent.copilot_acp_client as mod

    signalled, terminated = [], []

    class _Proc:
        pid = 111

        def terminate(self):
            terminated.append(111)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: 111, raising=False)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False)
    CopilotACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    # TERM to the group, probe (signal 0), then the KILL sweep for the survivor.
    assert signalled == [(111, _signal.SIGTERM), (111, 0), (111, _signal.SIGKILL)]
    assert terminated == []  # the group path never degrades to launcher-only


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_signals_a_reaped_launchers_group(monkeypatch):
    """The crash path: the CLI died early, _session's poll() reaped it, and
    getpgid can no longer see the pid — but the group it led outlives it while
    any member lives. Bailing there would signal nothing at all (the exact
    #124835 leak); the sweep must fall back to the launcher's pid, which
    _spawn guarantees is the group id (#124835 review)."""
    import signal as _signal

    import agent.copilot_acp_client as mod

    signalled, terminated = [], []

    class _Proc:
        pid = 111

        def terminate(self):
            terminated.append(111)

        def wait(self, timeout=None):
            return 0  # already reaped: Popen returns immediately

        def kill(self):
            pass

    def _getpgid(pid):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod.os, "getpgid", _getpgid, raising=False)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False)
    CopilotACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    assert signalled == [(111, _signal.SIGTERM), (111, 0), (111, _signal.SIGKILL)]
    assert terminated == []


def test_terminate_process_sweeps_the_tree_on_windows(monkeypatch):
    """No process groups on Windows: a launcher TERM stops at the npm wrapper,
    so teardown must also run the taskkill /T /F tree sweep instead of leaving
    the native descendant running (#124835's Windows twin)."""
    from types import SimpleNamespace

    import hermes_cli._subprocess_compat as compat
    import agent.copilot_acp_client as mod

    terminated, swept = [], []

    class _Proc:
        pid = 222

        def terminate(self):
            terminated.append(222)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    # A module-local os stand-in, NOT a global os.name patch: a global 'nt'
    # makes pathlib instantiate WindowsPath on POSIX hosts the moment a failed
    # assertion's repr touches Path, which turns a red test into an INTERNALERROR.
    monkeypatch.setattr(mod, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(compat, "kill_process_tree", lambda proc: swept.append(proc.pid))
    CopilotACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    assert terminated == [222]  # graceful window first
    assert swept == [222]  # then the tree sweep


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_reaches_survivors_after_the_launcher_is_reaped():
    """End-to-end crash path: the launcher exits on its own, poll() reaps it,
    and the native worker it spawned keeps running — getpgid can no longer see
    the pid. The teardown must still take the worker down; bailing on the
    ProcessLookupError is exactly the #124835 leak the tree-kill exists for."""
    import contextlib
    import subprocess as sp
    import time as _time

    argv = [sys.executable, "-c", _LAUNCHER_THAT_DIES_AND_LEAVES_A_WORKER]
    proc = sp.Popen(argv, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, process_group=0)
    try:
        worker_pid = int(proc.stdout.readline().decode().strip())
        proc.wait(timeout=10)  # the launcher exits immediately...
        assert proc.poll() is not None  # ...and is reaped, like _session's poll()
        os.kill(worker_pid, 0)  # the worker outlives its dead launcher

        CopilotACPClient._terminate_process(proc)

        deadline = _time.monotonic() + 5
        while _time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except OSError:
                break
            _time.sleep(0.05)
        else:
            with contextlib.suppress(OSError):
                os.kill(worker_pid, 9)
            raise AssertionError("the reaped launcher's worker survived the teardown")
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=5)
