"""Safety and lifecycle invariants of the public ACP stdio client (``agent.acp_client.ACPClient``).

Every ``external_process`` model provider (a plugin profile whose ``create_client`` returns an
``ACPClient`` subclass) inherits these, so they are pinned on the generic base with a vendor-neutral
subclass: the agent's own permission requests are never granted (Hermes has no human channel on that
path), the ``fs/*`` bridge is confined to the session cwd, honours the write deny-list and redacts
secrets it hands back, and every spawned agent process is reaped by its own session or by ``close()``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.acp_client import _ACP_PROBE_CACHE, ACPClient, _acp_supported, _model_selection_request
from tests.waiting import wait_until


class _AcmeClient(ACPClient):
    """A third-party subclass that customises only the launch defaults and labels."""

    default_command = "acme-agent"
    default_args = ("--acp", "--stdio")
    command_env_vars = ("ACME_ACP_COMMAND",)
    args_env_var = "ACME_ACP_ARGS"
    marker_base_url = "acp://acme"
    placeholder_model = "acme-acp"
    label = "Acme ACP"


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


def _dispatch(client: ACPClient, message: dict, *, cwd: str) -> dict:
    """Feed one agent->client request through the client and return the JSON-RPC reply it wrote."""
    process = _FakeProcess()
    handled = client._handle_server_message(message, process=process, cwd=cwd, text_parts=[], reasoning_parts=[])
    assert handled
    payload = process.stdin.getvalue().strip()
    assert payload, "the client must answer every agent request"
    return json.loads(payload)


def _request(method: str, params: dict, msg_id: int = 7) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}


@pytest.fixture
def client(tmp_path):
    return _AcmeClient(acp_cwd=str(tmp_path))


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    _ACP_PROBE_CACHE.clear()
    yield
    _ACP_PROBE_CACHE.clear()


# ── permission + fs bridge ──────────────────────────────────────────────────────────────────


def test_agent_permission_request_is_answered_cancelled_never_granted(client, tmp_path):
    response = _dispatch(client, _request("session/request_permission", {
        "sessionId": "s1", "toolCall": {"toolCallId": "tc1", "title": "rm -rf /", "kind": "execute"},
        "options": [{"optionId": "allow-always", "name": "Always", "kind": "allow_always"},
                    {"optionId": "allow-once", "name": "Allow", "kind": "allow_once"}]}), cwd=str(tmp_path))
    assert response["id"] == 7
    assert response["result"]["outcome"] == {"outcome": "cancelled"}, response


@pytest.mark.parametrize("redact_toggle", [True, False], ids=["redaction-on", "redaction-off"])
def test_read_text_file_redacts_sensitive_content(client, tmp_path, redact_toggle):
    secret_file = tmp_path / "config.env"
    secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012", encoding="utf-8")

    # agent.redact snapshots HERMES_REDACT_SECRETS at import time into _REDACT_ENABLED, so patching
    # os.environ is a no-op; flip the constant. The fs bridge hands file text to a third-party agent
    # process, so it redacts whatever the user's display toggle says.
    with patch("agent.redact._REDACT_ENABLED", redact_toggle):
        response = _dispatch(client, _request("fs/read_text_file", {"path": str(secret_file)}), cwd=str(tmp_path))

    content = (response.get("result") or {}).get("content") or ""
    assert "abc123def456" not in content
    assert "OPENAI_API_KEY=" in content


def test_read_text_file_outside_the_session_cwd_is_refused(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE-SECRET-5512", encoding="utf-8")
    escape = workspace / ".." / "outside.txt"

    for path in (outside, escape):
        response = _dispatch(client, _request("fs/read_text_file", {"path": str(path)}), cwd=str(workspace))
        assert "error" in response and "OUTSIDE-SECRET" not in json.dumps(response), response


def test_read_text_file_through_a_symlink_out_of_the_cwd_is_refused(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("OUTSIDE-SECRET-5512", encoding="utf-8")
    link = workspace / "inside-looking.txt"
    try:
        link.symlink_to(tmp_path / "outside.txt")
    except OSError:
        pytest.skip("symlinks unavailable on this host")

    response = _dispatch(client, _request("fs/read_text_file", {"path": str(link)}), cwd=str(workspace))
    assert "error" in response and "OUTSIDE-SECRET" not in json.dumps(response), response


def test_relative_fs_paths_are_refused(client, tmp_path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    response = _dispatch(client, _request("fs/read_text_file", {"path": "note.txt"}), cwd=str(tmp_path))
    assert "error" in response, response


def test_fs_read_text_file_decodes_as_utf8_under_non_utf8_locale(client, tmp_path):
    """Regression for #18637: the fs bridge read with the locale codec, so on GBK/CP932/CP949 hosts
    any source file with non-ASCII content crashed the agent's read_file."""
    target = tmp_path / "note.md"
    target.write_text("# 中文标题\nem dash — here\n", encoding="utf-8")
    original_read_text = Path.read_text

    def strict_read_text(self, encoding=None, errors=None, **kwargs):
        if self == target and encoding not in ("utf-8", "utf-8-sig"):
            raise UnicodeDecodeError("gbk", b"\x94", 0, 1, "illegal multibyte sequence")
        return original_read_text(self, encoding=encoding, errors=errors, **kwargs)

    with patch.object(Path, "read_text", strict_read_text):
        response = _dispatch(client, _request("fs/read_text_file", {"path": str(target)}), cwd=str(tmp_path))

    assert "error" not in response, response
    content = (response.get("result") or {}).get("content") or ""
    assert "中文标题" in content and "em dash —" in content


def test_write_text_file_outside_the_session_cwd_is_refused(client, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    response = _dispatch(client, _request("fs/write_text_file", {"path": str(outside), "content": "pwned"}),
                         cwd=str(workspace))
    assert "error" in response, response
    assert not outside.exists()


def test_write_text_file_respects_safe_root(client, tmp_path):
    safe_root = tmp_path / "workspace"
    safe_root.mkdir()
    outside = tmp_path / "outside.txt"

    with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
        response = _dispatch(client, _request("fs/write_text_file", {"path": str(outside), "content": "should-not-write"}),
                             cwd=str(tmp_path))

    assert "error" in response
    assert "HERMES_WRITE_SAFE_ROOT" in str(response["error"])
    assert not outside.exists()


def test_write_text_file_inside_the_cwd_lands(client, tmp_path):
    target = tmp_path / "sub" / "out.txt"
    response = _dispatch(client, _request("fs/write_text_file", {"path": str(target), "content": "written"}),
                         cwd=str(tmp_path))
    assert "error" not in response, response
    assert target.read_text(encoding="utf-8") == "written"


def test_unknown_agent_request_gets_method_not_found(client, tmp_path):
    response = _dispatch(client, _request("terminal/create", {"command": "id"}), cwd=str(tmp_path))
    assert response["error"]["code"] == -32601, response


# ── process lifecycle on a shared client ────────────────────────────────────────────────────
#
# Clients are cached per provider config and served to every concurrent caller, so one ACPClient can
# run several ACP sessions at once. Each session must reap ITS OWN child on exit: reaping whatever
# most recently claimed shared state kills a sibling's live process and leaks the session's own.


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
    server = tmp_path / "fake_acp_server.py"
    server.write_text(_FAKE_ACP_SERVER, encoding="utf-8")
    client = _AcmeClient(command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path))
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
    try:
        session_a.__exit__(None, None, None)

        leaked = proc_a.poll() is None
        killed = proc_b.poll() is not None
        assert not leaked and not killed, f"session A teardown: own child leaked={leaked}, sibling process killed={killed}"
        assert client.is_closed is False, "a shared client is not closed while a sibling session is live"

        session_b.__exit__(None, None, None)
        assert proc_b.poll() is not None
        assert client.is_closed is True, "the last session to drain still flips is_closed for single-session callers"
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()


def test_close_terminates_every_live_session_process(tmp_path):
    spawned = []
    client = _recording_client(tmp_path, spawned)

    session_a = client._session(30)
    session_b = client._session(30)
    session_a.__enter__()
    session_b.__enter__()
    try:
        client.close()
        assert all(proc.poll() is not None for proc in spawned), [proc.poll() for proc in spawned]
        assert client.is_closed is True
    finally:
        for proc in spawned:
            if proc.poll() is None:
                proc.kill()


def test_completed_prompt_leaves_no_agent_process(tmp_path):
    spawned = []
    client = _recording_client(tmp_path, spawned)

    completion = client.chat.completions.create(model="acme-acp", messages=[{"role": "user", "content": "hi"}])

    assert completion.choices[0].message.content == "done"
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert client.is_closed is True


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
    client = _AcmeClient(command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match="exited early: fatal: agent segfaulted"):
        with client._session(30):
            pass


_STDIN_CLOSING_ACP_CLI = """import os, sys, time
os.close(0)  # stops reading: the client's first write hits a closed pipe
open(sys.argv[1], "w").close()
time.sleep(0.3)
print('fatal: agent cannot start', file=sys.stderr)
sys.exit(2)
"""


def test_cli_that_stops_reading_is_reported_with_its_exit_text(tmp_path):
    """A CLI dying before it reads the first request: the write raised a bare BrokenPipeError, so the
    user never saw the CLI's own crash text (and the error read as a client bug)."""
    server = tmp_path / "stdin_closing_acp.py"
    server.write_text(_STDIN_CLOSING_ACP_CLI, encoding="utf-8")
    closed = tmp_path / "stdin-closed"
    client = _AcmeClient(command=sys.executable, args=[str(server), str(closed)], acp_cwd=str(tmp_path))
    real_spawn = client._spawn

    def spawn_then_wait_for_the_close():
        proc = real_spawn()
        wait_until(closed.exists, timeout=30, what="the fixture CLI closed its stdin")
        return proc

    client._spawn = spawn_then_wait_for_the_close
    with pytest.raises(RuntimeError, match="exited early: fatal: agent cannot start"):
        with client._session(30):
            pass

_DEAF_ACP_CLI = """import os, sys, time
os.close(0)  # stops reading but keeps running
open(sys.argv[1], "w").close()
time.sleep(120)
"""


def test_cli_that_stops_reading_but_keeps_running_fails_fast(tmp_path):
    """Alive with its stdin closed: the request can never be read, so the write's broken pipe is the
    answer; waiting for a reply would hang for the whole request timeout."""
    server = tmp_path / "deaf_acp.py"
    server.write_text(_DEAF_ACP_CLI, encoding="utf-8")
    closed = tmp_path / "stdin-closed"
    client = _AcmeClient(command=sys.executable, args=[str(server), str(closed)], acp_cwd=str(tmp_path))
    real_spawn = client._spawn

    def spawn_then_wait_for_the_close():
        proc = real_spawn()
        wait_until(closed.exists, timeout=30, what="the fixture CLI closed its stdin")
        return proc

    client._spawn = spawn_then_wait_for_the_close
    started = time.monotonic()
    with pytest.raises(OSError):
        with client._session(900):
            pass
    assert time.monotonic() - started < 60


def test_model_discovery_does_not_allow_file_requests(tmp_path):
    """``list_models`` opens a session only to read the catalog: an fs request on that path is refused."""
    target = tmp_path / "should-not-be-read.txt"
    target.write_text("private", encoding="utf-8")
    server = tmp_path / "fake_acp_fs_request.py"
    server.write_text(
        f"""import json
import sys

initialize = json.loads(sys.stdin.readline())
print(json.dumps({{"jsonrpc": "2.0", "id": initialize["id"], "result": {{"protocolVersion": 1}}}}), flush=True)
session = json.loads(sys.stdin.readline())
print(json.dumps({{"jsonrpc": "2.0", "id": 99, "method": "fs/read_text_file", "params": {{"path": {str(target)!r}}}}}), flush=True)
file_response = json.loads(sys.stdin.readline())
assert file_response["error"]["code"] == -32601
print(json.dumps({{"jsonrpc": "2.0", "id": session["id"], "result": {{"sessionId": "catalog-session", "configOptions": [{{"id": "model", "options": [{{"value": "model-sol"}}]}}]}}}}), flush=True)
""",
        encoding="utf-8",
    )
    client = _AcmeClient(command=sys.executable, args=[str(server)], acp_cwd=str(tmp_path))

    assert client.list_models(timeout_seconds=30) == ["model-sol"]


# ── OpenAI facade ───────────────────────────────────────────────────────────────────────────


def test_stream_true_preserves_tool_call_deltas(client):
    tool_response = (
        "<tool_call>"
        '{"id":"call_read","type":"function",'
        '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
        "</tool_call>"
    )

    with patch.object(client, "_run_prompt", return_value=(tool_response, "")):
        stream = client._create_chat_completion(
            model="acme-acp", messages=[{"role": "user", "content": "read README.md"}], stream=True)

    chunks = list(stream)
    delta = chunks[0].choices[0].delta
    assert delta.content is None
    assert chunks[0].choices[0].finish_reason == "tool_calls"
    assert len(delta.tool_calls) == 1
    tool_delta = delta.tool_calls[0]
    assert (tool_delta.index, tool_delta.id, tool_delta.function.name) == (0, "call_read", "read_file")
    assert json.loads(tool_delta.function.arguments) == {"path": "README.md"}
    assert chunks[1].choices == []


def test_run_prompt_receives_picker_model(tmp_path):
    """``_create_chat_completion`` forwards ``model`` into ``_run_prompt``: dropping it reduced the
    selection to prompt text."""
    client = _AcmeClient(acp_cwd=str(tmp_path))
    seen = {}

    def fake_run_prompt(prompt_text, *, timeout_seconds, model=None):
        seen["model"] = model
        return "ok", ""

    with patch.object(_AcmeClient, "_run_prompt", side_effect=fake_run_prompt):
        client._create_chat_completion(model="model-terra", messages=[{"role": "user", "content": "hi"}])
    assert seen["model"] == "model-terra"


# ── child HOME ──────────────────────────────────────────────────────────────────────────────


def _launch_client(tmp_path):
    return _AcmeClient(api_key="acme-acp", base_url="acp://acme", acp_command="acme-agent",
                       acp_args=["--acp", "--stdio"], acp_cwd=str(tmp_path))


def _capture_popen(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"], captured["kwargs"] = cmd, kwargs
        raise FileNotFoundError("acme-agent not found")
    return _fake


def test_run_prompt_preserves_real_home_when_profile_home_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Hermeticity: an ambient HERMES_REAL_HOME / TERMINAL_HOME_MODE outranks HOME in the candidate
    # ladder, and a containerized runner flips get_subprocess_home()'s auto mode; the host branch is
    # the contract under test.
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)
    captured = {}

    # The --acp support probe runs subprocess.run before Popen; inconclusive so no host CLI interferes.
    with patch("agent.acp_client.subprocess.run", side_effect=FileNotFoundError), \
            patch("agent.acp_client.subprocess.Popen", side_effect=_capture_popen(captured)), \
            pytest.raises(RuntimeError, match="Could not start Acme ACP command"):
        _launch_client(tmp_path)._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(real_home)
    assert captured["kwargs"]["env"]["HERMES_REAL_HOME"] == str(real_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    captured = {}

    with patch("agent.acp_client.subprocess.run", side_effect=FileNotFoundError), \
            patch("agent.acp_client.subprocess.Popen", side_effect=_capture_popen(captured)), \
            pytest.raises(RuntimeError, match="Could not start Acme ACP command"):
        _launch_client(tmp_path)._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"]


# ── --acp support probe (#87309) ────────────────────────────────────────────────────────────


def _completed(returncode=0, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_probe_true_when_help_advertises_acp():
    with patch("agent.acp_client.subprocess.run", return_value=_completed(stdout="Usage: acme-agent [--acp] [--stdio]")):
        assert _acp_supported("acme-agent", ["--acp", "--stdio"]) is True


def test_probe_false_when_help_lacks_acp_and_run_prompt_fast_fails(tmp_path):
    with patch("agent.acp_client.subprocess.run", return_value=_completed(stdout="Usage: other [--print] [--model]")), \
            pytest.raises(RuntimeError, match="ACP transport not supported"):
        _launch_client(tmp_path)._run_prompt("hello", timeout_seconds=1)


def test_probe_inconclusive_falls_through_to_spawn_error(tmp_path):
    """Missing binary: the probe must NOT mask the established spawn error."""
    with patch("agent.acp_client.subprocess.run", side_effect=FileNotFoundError("acme-agent not found")), \
            patch("agent.acp_client.subprocess.Popen", side_effect=FileNotFoundError("acme-agent not found")), \
            pytest.raises(RuntimeError, match="Could not start Acme ACP command"):
        _launch_client(tmp_path)._run_prompt("hello", timeout_seconds=1)


def test_probe_skipped_for_custom_args_without_acp():
    with patch("agent.acp_client.subprocess.run") as run_mock:
        assert _acp_supported("mycli", ["--custom-transport"]) is True
    run_mock.assert_not_called()


# ── session model selection ─────────────────────────────────────────────────────────────────
#
# Agent CLIs may validate but IGNORE a ``--model`` spawn flag; the session runs the CLI's default
# unless the client selects the model over ACP after ``session/new``.


def _session_with_config_options():
    return {"sessionId": "s1", "configOptions": [{
        "id": "model", "category": "model", "type": "select", "currentValue": "auto",
        "options": [{"value": "auto", "name": "Auto"}, {"value": "model-terra", "name": "Terra"},
                    {"value": "model-fable", "name": "Fable", "_meta": {"acmePolicy": "off"}}]}]}


def _acme_disabled(entry):
    return (entry.get("_meta") or {}).get("acmePolicy") == "off"


def test_model_selection_prefers_stable_config_option():
    assert _model_selection_request(_session_with_config_options(), "model-terra") == (
        "session/set_config_option", {"sessionId": "s1", "configId": "model", "value": "model-terra"})


def test_model_selection_rejects_disabled_config_option():
    assert _model_selection_request(_session_with_config_options(), "model-fable", disabled=_acme_disabled) is None


def test_model_selection_rejects_unknown_config_option():
    assert _model_selection_request(_session_with_config_options(), "not-served-here") is None


def test_model_selection_falls_back_to_legacy_extension():
    legacy_session = {"sessionId": "s1", "models": {"availableModels": [{"modelId": "auto"}, {"modelId": "model-terra"}]}}
    assert _model_selection_request(legacy_session, "model-terra") == (
        "session/set_model", {"sessionId": "s1", "modelId": "model-terra"})


def test_model_selection_skips_provider_placeholder_slug():
    assert _model_selection_request(_session_with_config_options(), "acme-acp", placeholder="acme-acp") is None
