"""Tests for :mod:`agent._acp_client_base`.

Exercise the shared ACP transport with an in-process fake subprocess so we
don't depend on a real copilot/claude-code binary.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent._acp_client_base import (
    _AcpClientBase,
    _ensure_path_within_cwd,
    _format_messages_as_prompt,
    _render_message_content,
    resolve_effective_timeout,
)
import agent._acp_client_base as _acp_base


# ---------------------------------------------------------------------------
# Fake subprocess — lives in tests/agent/conftest.py so the base + client
# test suites share one source of truth as the real transport evolves.
# ---------------------------------------------------------------------------
from tests.agent.conftest import _FakeProc, _FakeStdin  # noqa: E402


class _HarnessClient(_AcpClientBase):
    """Base subclass used only by tests — pins defaults so we can inspect them."""

    _provider_label = "harness-acp"
    _default_command = "harness-default"
    _default_args = ("--stdio",)
    _env_command_vars = ("HARNESS_ACP_CMD", "HARNESS_PATH")
    _env_args_var = "HARNESS_ACP_ARGS"
    _marker_base_url = "acp://harness"
    _client_name = "hermes-harness"

    def _create_chat_completion(self, **kwargs):  # pragma: no cover — not under test
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Env / resolver tests
# ---------------------------------------------------------------------------


def test_resolve_command_defaults(monkeypatch):
    for var in ("HARNESS_ACP_CMD", "HARNESS_PATH", "HARNESS_ACP_ARGS"):
        monkeypatch.delenv(var, raising=False)
    c = _HarnessClient()
    assert c._acp_command == "harness-default"
    assert c._acp_args == ["--stdio"]


def test_resolve_command_honors_first_env(monkeypatch):
    monkeypatch.setenv("HARNESS_ACP_CMD", "/opt/custom/harness")
    c = _HarnessClient()
    assert c._acp_command == "/opt/custom/harness"


def test_resolve_command_falls_back_to_second_env(monkeypatch):
    monkeypatch.delenv("HARNESS_ACP_CMD", raising=False)
    monkeypatch.setenv("HARNESS_PATH", "/custom/alt/harness")
    c = _HarnessClient()
    assert c._acp_command == "/custom/alt/harness"


def test_resolve_args_from_env(monkeypatch):
    monkeypatch.setenv("HARNESS_ACP_ARGS", "--foo bar --baz")
    c = _HarnessClient()
    assert c._acp_args == ["--foo", "bar", "--baz"]


def test_explicit_command_args_override_env(monkeypatch):
    monkeypatch.setenv("HARNESS_ACP_CMD", "/env/cmd")
    c = _HarnessClient(command="/ctor/cmd", args=["--x"])
    assert c._acp_command == "/ctor/cmd"
    assert c._acp_args == ["--x"]


# ---------------------------------------------------------------------------
# JSON-RPC correlation + server-initiated dispatch
# ---------------------------------------------------------------------------


def _make_client_and_fake_proc(tmp_path):
    c = _HarnessClient(acp_cwd=str(tmp_path))
    proc = _FakeProc()
    inbox: "queue.Queue[dict]" = queue.Queue()
    stderr_tail: deque[str] = deque(maxlen=40)

    # Bootstrap reader thread manually so we don't need Popen.
    def _reader():
        for line in proc:
            try:
                inbox.put(json.loads(line))
            except Exception:
                inbox.put({"raw": line.rstrip("\n")})

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return c, proc, inbox, stderr_tail, t


def test_request_id_correlation(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def responder():
        time.sleep(0.02)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": {"ok": 1}})

    threading.Thread(target=responder, daemon=True).start()
    result = c._request(proc, inbox, stderr_tail, "ping", {}, timeout_seconds=2.0)
    assert result == {"ok": 1}
    # Request was written to stdin
    assert len(proc.stdin.lines) == 1
    sent = json.loads(proc.stdin.lines[0])
    assert sent["method"] == "ping" and sent["id"] == 1


def test_request_ignores_id_mismatch_then_matches(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def responder():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 99, "result": "ignored"})
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "matched"})

    threading.Thread(target=responder, daemon=True).start()
    assert c._request(proc, inbox, stderr_tail, "m", {}, timeout_seconds=2.0) == "matched"


def test_session_update_dispatched_to_handler(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "method": "session/update", "params": {
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "hi"}}
        }})
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "method": "session/update", "params": {
            "update": {"sessionUpdate": "agent_thought_chunk", "content": {"text": "thinking"}}
        }})
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "done"})

    threading.Thread(target=pusher, daemon=True).start()
    result = c._request(
        proc, inbox, stderr_tail, "session/prompt", {},
        timeout_seconds=2.0,
        dispatch_ctx={"text_parts": text_parts, "reasoning_parts": reasoning_parts},
    )
    assert result == "done"
    assert text_parts == ["hi"]
    assert reasoning_parts == ["thinking"]


def test_fs_read_text_file_within_cwd(tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text("cyan\n")
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 7, "method": "fs/read_text_file",
                   "params": {"path": str(target)}})
        time.sleep(0.05)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "ack"})

    threading.Thread(target=pusher, daemon=True).start()
    c._request(proc, inbox, stderr_tail, "prompt", {}, timeout_seconds=2.0)
    # Second line on stdin is the fs/read response
    sent_responses = [json.loads(l) for l in proc.stdin.lines]
    read_resp = [r for r in sent_responses if r.get("id") == 7]
    assert read_resp and read_resp[0]["result"]["content"] == "cyan\n"


def test_fs_read_text_file_rejects_escape(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 9, "method": "fs/read_text_file",
                   "params": {"path": "/etc/passwd"}})
        time.sleep(0.05)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "ack"})

    threading.Thread(target=pusher, daemon=True).start()
    c._request(proc, inbox, stderr_tail, "prompt", {}, timeout_seconds=2.0)
    sent = [json.loads(l) for l in proc.stdin.lines]
    err = [m for m in sent if m.get("id") == 9][0]
    assert "error" in err
    assert "outside the session cwd" in err["error"]["message"]


def test_fs_write_text_file_creates_parents(tmp_path):
    target = tmp_path / "sub" / "dir" / "out.txt"
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 11, "method": "fs/write_text_file",
                   "params": {"path": str(target), "content": "ok"}})
        time.sleep(0.05)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "ack"})

    threading.Thread(target=pusher, daemon=True).start()
    c._request(proc, inbox, stderr_tail, "prompt", {}, timeout_seconds=2.0)
    assert target.exists()
    assert target.read_text() == "ok"


def test_session_request_permission_auto_allow(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 13, "method": "session/request_permission",
                   "params": {"toolCall": "x"}})
        time.sleep(0.05)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "ack"})

    threading.Thread(target=pusher, daemon=True).start()
    c._request(proc, inbox, stderr_tail, "prompt", {}, timeout_seconds=2.0)
    sent = [json.loads(l) for l in proc.stdin.lines]
    perm = [m for m in sent if m.get("id") == 13][0]
    assert perm["result"]["outcome"]["outcome"] == "allow_once"


def test_request_error_raises_runtime_error(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "bad"}})

    threading.Thread(target=pusher, daemon=True).start()
    with pytest.raises(RuntimeError, match="harness-acp m failed: bad"):
        c._request(proc, inbox, stderr_tail, "m", {}, timeout_seconds=2.0)


def test_timeout_raises_timeouterror(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)
    # No one responds — request must hit the deadline
    with pytest.raises(TimeoutError, match="harness-acp"):
        c._request(proc, inbox, stderr_tail, "m", {}, timeout_seconds=0.3)


def test_early_exit_reports_stderr_tail(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)
    stderr_tail.append("something broke")
    proc._return_code = 1
    with pytest.raises(RuntimeError, match="exited early: something broke"):
        c._request(proc, inbox, stderr_tail, "m", {}, timeout_seconds=0.3)


def test_unknown_server_method_returns_jsonrpc_error(tmp_path):
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)

    def pusher():
        time.sleep(0.01)
        proc.push({"jsonrpc": "2.0", "id": 5, "method": "fs/unlink", "params": {"path": "/x"}})
        time.sleep(0.05)
        proc.push({"jsonrpc": "2.0", "id": 1, "result": "ack"})

    threading.Thread(target=pusher, daemon=True).start()
    c._request(proc, inbox, stderr_tail, "prompt", {}, timeout_seconds=2.0)
    sent = [json.loads(l) for l in proc.stdin.lines]
    err = [m for m in sent if m.get("id") == 5][0]
    assert err["error"]["code"] == -32601
    assert "fs/unlink" in err["error"]["message"]


# ---------------------------------------------------------------------------
# close() escalation
# ---------------------------------------------------------------------------


def test_close_terminates_then_kills(tmp_path):
    c = _HarnessClient(acp_cwd=str(tmp_path))

    class FlakyProc:
        def __init__(self):
            self.terminated = False
            self.killed = False
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("cmd", timeout)
        def kill(self):
            self.killed = True

    proc = FlakyProc()
    c._active_process = proc  # type: ignore[assignment]
    c.close()
    assert proc.terminated is True
    assert proc.killed is True
    assert c.is_closed is True


# ---------------------------------------------------------------------------
# Missing subprocess binary
# ---------------------------------------------------------------------------


def test_missing_binary_friendly_error(monkeypatch, tmp_path):
    c = _HarnessClient(command="/does/not/exist", acp_cwd=str(tmp_path))
    with pytest.raises(RuntimeError, match="Could not start harness-acp command"):
        c._start_subprocess()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_format_messages_as_prompt_basic():
    out = _format_messages_as_prompt(
        [{"role": "user", "content": "hello"}],
        preamble=["You are a test."],
    )
    assert "You are a test." in out
    assert "User:\nhello" in out
    assert out.endswith("Continue the conversation from the latest user request.")


def test_format_messages_renders_list_content():
    out = _format_messages_as_prompt(
        [
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]}
        ],
        preamble=[],
    )
    assert "first" in out and "second" in out


def test_format_messages_with_tools_injects_schema():
    tools = [{
        "type": "function",
        "function": {
            "name": "do_it",
            "description": "does things",
            "parameters": {"type": "object"},
        },
    }]
    out = _format_messages_as_prompt(
        [{"role": "user", "content": "x"}],
        preamble=[],
        tools=tools,
        tool_call_instructions="CUSTOM INSTRUCTIONS",
    )
    assert "CUSTOM INSTRUCTIONS" in out
    assert '"name": "do_it"' in out


def test_resolve_effective_timeout_plain_number():
    assert resolve_effective_timeout(30, default=900) == 30.0


def test_resolve_effective_timeout_httpx_like():
    obj = SimpleNamespace(read=60, write=5, connect=2, pool=10)
    assert resolve_effective_timeout(obj, default=900) == 60.0


def test_resolve_effective_timeout_fallback():
    assert resolve_effective_timeout(None, default=777) == 777.0


def test_ensure_path_within_cwd_rejects_relative(tmp_path):
    with pytest.raises(PermissionError):
        _ensure_path_within_cwd("not/absolute", str(tmp_path))


def test_ensure_path_within_cwd_accepts_nested(tmp_path):
    nested = tmp_path / "a" / "b.txt"
    resolved = _ensure_path_within_cwd(str(nested), str(tmp_path))
    assert resolved == nested.resolve()


def test_multimodal_blocks_log_warning_not_silent_drop(caplog):
    """Image/audio blocks the flat-string path can't carry should warn once."""
    # Reset the once-flag to guarantee the warning fires for this test.
    _acp_base._MULTIMODAL_WARNED = False
    content = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://example/x.png"}},
    ]
    with caplog.at_level("WARNING", logger="agent._acp_client_base"):
        rendered = _render_message_content(content)
    assert "describe this" in rendered
    assert any("dropped content block" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Stderr log capture on abnormal subprocess exit
# ---------------------------------------------------------------------------


def test_open_stderr_log_file_writes_to_hermes_home(tmp_path, monkeypatch):
    """The stderr log file lands under ``<hermes_home>/logs/acp/`` and is
    writable. Writing to the returned handle roundtrips to the file on disk.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = _HarnessClient(acp_cwd=str(tmp_path))
    fh = c._open_stderr_log_file(pid=4242)
    assert fh is not None
    fh.write("line-1\nline-2\n")
    fh.close()
    assert c._stderr_log_path is not None
    assert c._stderr_log_path.parent == tmp_path / "logs" / "acp"
    assert c._stderr_log_path.read_text() == "line-1\nline-2\n"


def test_exited_early_error_includes_log_path(tmp_path, monkeypatch):
    """When the subprocess dies, the RuntimeError points at the full log."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c, proc, inbox, stderr_tail, _ = _make_client_and_fake_proc(tmp_path)
    # Simulate the reader having seen 200 lines on stderr; only the last 40
    # survive in the tail, but the full log on disk should record all of
    # them — we test the error-surface glue, not the tee.
    log_path = tmp_path / "logs" / "acp" / "harness-acp-123.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(f"full-line-{i}" for i in range(200)))
    c._stderr_log_path = log_path
    stderr_tail.append("tail-line-only")
    proc._return_code = 1
    with pytest.raises(RuntimeError) as excinfo:
        c._request(proc, inbox, stderr_tail, "m", {}, timeout_seconds=0.3)
    msg = str(excinfo.value)
    assert "tail-line-only" in msg
    assert f"Full stderr: {log_path}" in msg
    # The full log on disk is untruncated.
    assert "full-line-0" in log_path.read_text()
    assert "full-line-199" in log_path.read_text()


def test_open_stderr_log_file_disabled_when_home_unwritable(tmp_path, monkeypatch):
    """If the hermes_home dir can't be created, logging silently disables.

    Subprocess lifecycle must not depend on logging working — the fallback
    is to return None and set ``_stderr_log_path = None``. The exit error
    should then NOT include a ``Full stderr:`` suffix.
    """
    # Point HERMES_HOME at a path whose parent is a file (mkdir will fail).
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(blocker / "child"))
    c = _HarnessClient(acp_cwd=str(tmp_path))
    fh = c._open_stderr_log_file(pid=1)
    assert fh is None
    assert c._stderr_log_path is None


def test_cleanup_stale_acp_logs_deletes_old_files(tmp_path):
    """The 7-day sweep removes old logs, keeps fresh ones."""
    import os
    from agent.claude_code_sandbox import cleanup_stale_acp_logs

    log_dir = tmp_path / "logs" / "acp"
    log_dir.mkdir(parents=True)
    old = log_dir / "old.log"
    old.write_text("stale")
    recent = log_dir / "recent.log"
    recent.write_text("fresh")
    # Backdate `old` by 30 days.
    thirty_days_ago = time.time() - 30 * 86400
    os.utime(old, (thirty_days_ago, thirty_days_ago))

    removed = cleanup_stale_acp_logs(hermes_home=tmp_path)
    assert removed == 1
    assert not old.exists()
    assert recent.exists()


def test_cleanup_stale_acp_logs_handles_missing_dir(tmp_path):
    """No log dir → no-op, no crash."""
    from agent.claude_code_sandbox import cleanup_stale_acp_logs
    assert cleanup_stale_acp_logs(hermes_home=tmp_path) == 0
