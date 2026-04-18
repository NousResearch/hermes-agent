"""Tests for :mod:`agent.claude_code_acp_client`.

Exercise the Claude Code ACP subclass against an in-process fake subprocess
so we don't depend on ``npx`` or the real ``@zed-industries/claude-agent-acp``
package.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import pytest

from agent import claude_code_acp_client as cc
from agent.claude_code_acp_client import (
    ClaudeCodeACPClient,
    ToolCallRecord,
    trace_to_messages_snapshot,
)


# ---------------------------------------------------------------------------
# Fake subprocess lives in tests/agent/conftest.py — shared with
# test_acp_client_base.py so the two suites can't drift apart.
# ---------------------------------------------------------------------------
from tests.agent.conftest import _FakeProc, _FakeStdin  # noqa: E402, F401


def _make_client_with_fake_session(tmp_path, **kwargs):
    """Return (client, proc) with the fake session pre-wired in."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    # Stub out sandbox rebuild — we just point at a pre-made dir.
    (sandbox / ".mcp.json").write_text(json.dumps({"mcpServers": {}}))

    client = ClaudeCodeACPClient(
        command="/bin/true",
        acp_cwd=str(sandbox),
        hermes_home=tmp_path / "hermes",
        hermes_session_id="sess-123",
        **kwargs,
    )
    client._sandbox_path = sandbox
    client._acp_cwd = str(sandbox)

    proc = _FakeProc()
    inbox: "queue.Queue[dict]" = queue.Queue()
    stderr_tail: deque[str] = deque(maxlen=40)

    def _reader():
        for line in proc:
            try:
                inbox.put(json.loads(line))
            except Exception:
                inbox.put({"raw": line.rstrip("\n")})

    threading.Thread(target=_reader, daemon=True).start()

    client._session_proc = proc
    client._session_inbox = inbox
    client._session_stderr = stderr_tail
    client._session_id = "acp-session-xyz"
    return client, proc


# ---------------------------------------------------------------------------
# Class configuration
# ---------------------------------------------------------------------------


def test_class_defaults():
    assert ClaudeCodeACPClient._provider_label == "claude-code-acp"
    assert ClaudeCodeACPClient._default_command == "npx"
    assert ClaudeCodeACPClient._default_args == (
        "-y",
        "@zed-industries/claude-agent-acp",
    )
    assert ClaudeCodeACPClient._marker_base_url == "acp://claude-code"
    assert "HERMES_CLAUDE_CODE_ACP_COMMAND" in ClaudeCodeACPClient._env_command_vars
    assert "CLAUDE_ACP_PATH" in ClaudeCodeACPClient._env_command_vars


def test_env_overrides_command(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_ACP_COMMAND", "/opt/custom/claude-acp")
    monkeypatch.setenv("HERMES_CLAUDE_CODE_ACP_ARGS", "--stdio --debug")
    c = ClaudeCodeACPClient()
    assert c._acp_command == "/opt/custom/claude-acp"
    assert c._acp_args == ["--stdio", "--debug"]


def test_env_fallback_path(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ACP_COMMAND", raising=False)
    monkeypatch.setenv("CLAUDE_ACP_PATH", "/alt/claude-acp")
    c = ClaudeCodeACPClient()
    assert c._acp_command == "/alt/claude-acp"


# ---------------------------------------------------------------------------
# Native session/update handling
# ---------------------------------------------------------------------------


def test_text_chunk_triggers_stream_callback(tmp_path):
    chunks: list[str] = []
    client, _ = _make_client_with_fake_session(
        tmp_path, stream_delta_callback=chunks.append
    )
    text_parts: list[str] = []
    client._handle_session_update(
        {"sessionUpdate": "agent_message_chunk", "content": {"text": "hello"}},
        dispatch_ctx={"text_parts": text_parts, "reasoning_parts": []},
    )
    assert text_parts == ["hello"]
    assert chunks == ["hello"]


def test_thought_chunk_triggers_thinking_callback(tmp_path):
    thoughts: list[str] = []
    client, _ = _make_client_with_fake_session(
        tmp_path, thinking_callback=thoughts.append
    )
    reasoning_parts: list[str] = []
    client._handle_session_update(
        {"sessionUpdate": "agent_thought_chunk", "content": {"text": "pondering"}},
        dispatch_ctx={"text_parts": [], "reasoning_parts": reasoning_parts},
    )
    assert reasoning_parts == ["pondering"]
    assert thoughts == ["pondering"]


def test_tool_call_start_captures_record_and_fires_callback(tmp_path):
    events: list[tuple] = []

    def cb(*args, **kwargs):
        events.append((args, kwargs))

    client, _ = _make_client_with_fake_session(tmp_path, tool_progress_callback=cb)
    client._handle_session_update(
        {
            "sessionUpdate": "tool_call_start",
            "toolCallId": "tc-1",
            "title": "read_file",
            "rawInput": {"path": "/tmp/x"},
            "kind": "read",
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    trace = client.tool_trace
    assert len(trace) == 1
    rec = trace[0]
    assert rec.tool_call_id == "tc-1"
    assert rec.name == "read_file"
    assert rec.raw_input == {"path": "/tmp/x"}
    assert rec.status == "in_progress"
    assert rec.kind == "read"
    assert events and events[0][0][0] == "tool.started"
    assert events[0][0][1] == "read_file"


def test_tool_call_update_completes_matching_record(tmp_path):
    completed: list[tuple] = []

    def cb(phase, *args, **kwargs):
        if phase == "tool.completed":
            completed.append((args, kwargs))

    client, _ = _make_client_with_fake_session(tmp_path, tool_progress_callback=cb)
    # start
    client._handle_session_update(
        {
            "sessionUpdate": "tool_call_start",
            "toolCallId": "tc-9",
            "title": "terminal",
            "rawInput": {"cmd": "echo hi"},
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    # update with terminal status
    client._handle_session_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-9",
            "rawOutput": "hi\n",
            "status": "completed",
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    trace = client.tool_trace
    assert len(trace) == 1
    rec = trace[0]
    assert rec.status == "completed"
    assert rec.raw_output == "hi\n"
    assert rec.completed_at is not None
    assert rec.is_error is False
    # completion callback fired exactly once
    assert len(completed) == 1
    # duration kwarg carried through
    assert completed[0][1].get("duration", -1) >= 0.0
    assert completed[0][1].get("is_error") is False


def test_tool_call_update_marks_error(tmp_path):
    client, _ = _make_client_with_fake_session(tmp_path)
    client._handle_session_update(
        {
            "sessionUpdate": "tool_call_start",
            "toolCallId": "tc-e",
            "title": "bad_tool",
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    client._handle_session_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-e",
            "isError": True,
            "status": "failed",
            "content": [{"text": "kaboom"}],
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    rec = client.tool_trace[0]
    assert rec.is_error is True
    assert rec.status == "failed"
    # content fallback captured
    assert rec.raw_output == "kaboom"


def test_tool_call_update_without_start_creates_placeholder(tmp_path):
    """ACP adapters occasionally emit an update before the matching start
    event under load (race in the adapter-side dispatcher, not in Hermes).
    We tolerate this by synthesizing a placeholder ``ToolCallRecord`` — this
    test guards that behavior so future refactors don't re-introduce the
    "dropped trace entry" regression we fixed.
    """
    client, _ = _make_client_with_fake_session(tmp_path)
    client._handle_session_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "orphan-1",
            "rawOutput": "something",
            "status": "completed",
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    assert len(client.tool_trace) == 1
    assert client.tool_trace[0].tool_call_id == "orphan-1"
    assert client.tool_trace[0].status == "completed"


def test_plan_update_summarizes_to_thinking(tmp_path):
    msgs: list[str] = []
    client, _ = _make_client_with_fake_session(
        tmp_path, thinking_callback=msgs.append
    )
    client._handle_session_update(
        {
            "sessionUpdate": "plan",
            "entries": [
                {"content": "step one"},
                {"content": "step two"},
            ],
        },
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    assert msgs and "step one" in msgs[0] and "step two" in msgs[0]


def test_unknown_update_kind_silently_ignored(tmp_path):
    client, _ = _make_client_with_fake_session(tmp_path)
    # Must not raise.
    client._handle_session_update(
        {"sessionUpdate": "some_future_kind", "payload": 1},
        dispatch_ctx={"text_parts": [], "reasoning_parts": []},
    )
    assert client.tool_trace == []


# ---------------------------------------------------------------------------
# Full prompt cycle against the fake subprocess
# ---------------------------------------------------------------------------


def test_create_chat_completion_attaches_tool_trace(tmp_path):
    """Contract test for ``hermes_tool_trace`` on the response object.

    The AIAgent loop reads ``response.hermes_tool_trace`` (not
    ``response.choices[0].message.tool_calls``, which is intentionally empty
    because Claude ran the tools natively) to learn what tools fired. The
    shape of each trace entry is owned by
    :meth:`ClaudeCodeACPClient._create_chat_completion` — refactors there
    must keep this test passing.
    """
    chunks: list[str] = []
    progress: list[tuple] = []

    client, proc = _make_client_with_fake_session(
        tmp_path,
        stream_delta_callback=chunks.append,
        tool_progress_callback=lambda *a, **k: progress.append((a, k)),
    )

    def responder():
        time.sleep(0.02)
        proc.push({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "agent_message_chunk",
                                   "content": {"text": "Hello "}}},
        })
        time.sleep(0.01)
        proc.push({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "tool_call_start",
                                   "toolCallId": "t-1",
                                   "title": "read_file",
                                   "rawInput": {"path": "/tmp/hello"}}},
        })
        time.sleep(0.01)
        proc.push({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "tool_call_update",
                                   "toolCallId": "t-1",
                                   "rawOutput": "file contents",
                                   "status": "completed"}},
        })
        time.sleep(0.01)
        proc.push({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "agent_message_chunk",
                                   "content": {"text": "world"}}},
        })
        time.sleep(0.01)
        proc.push({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"usage": {"inputTokens": 17, "outputTokens": 3}},
        })

    threading.Thread(target=responder, daemon=True).start()

    resp = client._create_chat_completion(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hi"}],
        timeout=5.0,
    )

    assert resp.choices[0].message.content == "Hello world"
    assert resp.choices[0].message.tool_calls == []
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 17
    assert resp.usage.completion_tokens == 3
    # Native trace landed on the response — one record, with raw I/O intact.
    assert len(resp.hermes_tool_trace) == 1
    trace_rec = resp.hermes_tool_trace[0]
    assert trace_rec["name"] == "read_file"
    assert trace_rec["raw_input"] == {"path": "/tmp/hello"}
    assert trace_rec["raw_output"] == "file contents"
    # Streaming callback received both chunks
    assert chunks == ["Hello ", "world"]
    # tool_progress_callback saw started + completed
    phases = [args[0] for args, _ in progress]
    assert phases == ["tool.started", "tool.completed"]


def test_create_chat_completion_reuses_session(tmp_path):
    client, proc = _make_client_with_fake_session(tmp_path)

    def responder_once(id_value: int):
        def _r():
            time.sleep(0.01)
            proc.push({"jsonrpc": "2.0", "id": id_value, "result": {}})

        return _r

    # First turn.
    client._next_request_id = 0  # align to our expected id 1
    threading.Thread(target=responder_once(1), daemon=True).start()
    r1 = client._create_chat_completion(
        messages=[{"role": "user", "content": "turn one"}],
        timeout=2.0,
    )
    # Second turn: we still hold the same sessionId and the same subprocess.
    threading.Thread(target=responder_once(2), daemon=True).start()
    r2 = client._create_chat_completion(
        messages=[{"role": "user", "content": "turn two"}],
        timeout=2.0,
    )
    assert client._session_id == "acp-session-xyz"
    # Only one process was ever held (no re-init).
    assert client._session_proc is proc
    # Both turns share sessionId in their request payloads.
    sent = [json.loads(l) for l in proc.stdin.lines]
    prompts = [s for s in sent if s.get("method") == "session/prompt"]
    assert len(prompts) == 2
    assert prompts[0]["params"]["sessionId"] == "acp-session-xyz"
    assert prompts[1]["params"]["sessionId"] == "acp-session-xyz"


def test_close_resets_cached_session(tmp_path):
    client, _ = _make_client_with_fake_session(tmp_path)
    assert client._session_id == "acp-session-xyz"
    client.close()
    assert client._session_id is None
    assert client.is_closed is True


# ---------------------------------------------------------------------------
# MCP server translation
# ---------------------------------------------------------------------------


def test_load_mcp_servers_translates_dict_to_array(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / ".mcp.json").write_text(json.dumps({
        "mcpServers": {
            "hermes_tools": {
                "command": "hermes",
                "args": ["mcp", "tools-serve"],
                "env": {"HERMES_SESSION_ID": "sess-42"},
            },
        }
    }))

    client = ClaudeCodeACPClient(hermes_home=tmp_path, acp_cwd=str(sandbox))
    servers = client._load_mcp_servers(sandbox)
    assert len(servers) == 1
    srv = servers[0]
    assert srv["name"] == "hermes_tools"
    assert srv["command"] == "hermes"
    assert srv["args"] == ["mcp", "tools-serve"]
    assert {"name": "HERMES_SESSION_ID", "value": "sess-42"} in srv["env"]


def test_load_mcp_servers_handles_missing_file(tmp_path):
    client = ClaudeCodeACPClient(hermes_home=tmp_path, acp_cwd=str(tmp_path))
    assert client._load_mcp_servers(tmp_path) == []


# ---------------------------------------------------------------------------
# Sandbox lazy-build
# ---------------------------------------------------------------------------


def test_ensure_sandbox_invokes_builder(monkeypatch, tmp_path):
    built: list[tuple] = []

    def fake_builder(session_id, agent, *, hermes_home, platform, available_tools, available_toolsets, model=None):
        built.append(
            (session_id, hermes_home, platform, available_tools, available_toolsets)
        )
        out = hermes_home / "runtime" / "claude-code" / session_id
        out.mkdir(parents=True, exist_ok=True)
        (out / ".mcp.json").write_text(json.dumps({"mcpServers": {}}))
        return out

    monkeypatch.setattr(
        "agent.claude_code_sandbox.build_session_sandbox", fake_builder
    )

    client = ClaudeCodeACPClient(
        hermes_home=tmp_path,
        hermes_session_id="sess-A",
        platform="telegram",
        available_tools={"read_file"},
        available_toolsets={"fs"},
    )
    path = client._ensure_sandbox()
    assert path.exists()
    assert len(built) == 1
    (sess_id, hermes_home, platform, tools, tsets) = built[0]
    assert sess_id == "sess-A"
    assert hermes_home == tmp_path
    assert platform == "telegram"
    assert tools == {"read_file"}
    assert tsets == {"fs"}
    # Subsequent call is cached (builder not invoked again).
    client._ensure_sandbox()
    assert len(built) == 1


# ---------------------------------------------------------------------------
# trace_to_messages_snapshot
# ---------------------------------------------------------------------------


def test_trace_to_messages_snapshot_from_records():
    records = [
        ToolCallRecord(
            tool_call_id="t1",
            name="read_file",
            raw_input={"path": "/tmp/x"},
            raw_output="contents",
            status="completed",
        ),
        ToolCallRecord(
            tool_call_id="t2",
            name="terminal",
            raw_input={"cmd": "echo hi"},
            raw_output="hi",
            status="completed",
        ),
    ]
    msgs = trace_to_messages_snapshot(records)
    assert len(msgs) == 4
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"][0]["type"] == "tool_use"
    assert msgs[0]["content"][0]["id"] == "t1"
    assert msgs[0]["content"][0]["name"] == "read_file"
    assert msgs[0]["content"][0]["input"] == {"path": "/tmp/x"}
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_use_id"] == "t1"
    assert msgs[1]["content"] == "contents"
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["tool_use_id"] == "t2"


def test_trace_to_messages_snapshot_accepts_dicts():
    data = [
        {"id": "dx", "name": "web_search", "raw_input": {"q": "hi"}, "raw_output": "ok"},
    ]
    msgs = trace_to_messages_snapshot(data)
    assert len(msgs) == 2
    assert msgs[0]["content"][0]["id"] == "dx"
    assert msgs[1]["content"] == "ok"


def test_trace_to_messages_snapshot_handles_missing_id():
    data = [{"name": "x", "raw_input": {}, "raw_output": None}]
    msgs = trace_to_messages_snapshot(data)
    assert msgs[0]["content"][0]["id"]  # auto-generated
    assert msgs[1]["content"] == ""


# ---------------------------------------------------------------------------
# Text extraction helper
# ---------------------------------------------------------------------------


def test_extract_text_from_list_blocks():
    content = [{"text": "a"}, {"text": "b"}, "c"]
    assert cc._extract_text_from_content_blocks(content) == "a\nb\nc"


def test_extract_text_from_nested_dict():
    content = {"content": {"text": "inner"}}
    assert cc._extract_text_from_content_blocks(content) == "inner"


def test_short_preview_truncates():
    long = "x" * 200
    out = cc._short_preview(long, limit=40)
    assert len(out) <= 40
    assert out.endswith("…")


# ---------------------------------------------------------------------------
# Concurrency regression — trace mutations happen on the ACP reader thread
# while the main thread reads/serializes the trace.
# ---------------------------------------------------------------------------


def test_cancel_check_aborts_request_mid_poll(tmp_path):
    """cancel_check=True mid-turn should raise AcpCancelled instead of hanging."""
    from agent._acp_client_base import AcpCancelled

    client, proc = _make_client_with_fake_session(tmp_path)

    # Prime the fake agent with an interrupt flag that flips True after a
    # small delay, simulating a user Ctrl-C between ACP events.
    class _Fake:
        _interrupt_requested = False

    client._agent = _Fake()

    def _flip_after_delay():
        time.sleep(0.15)
        _Fake._interrupt_requested = True

    threading.Thread(target=_flip_after_delay, daemon=True).start()

    start = time.time()
    with pytest.raises(AcpCancelled):
        # No prompt response will arrive — the fake subprocess pushes nothing.
        # The cancel flag should fire inside the 0.1s poll loop.
        client._request(
            proc,
            client._session_inbox,
            client._session_stderr,
            "session/prompt",
            {"sessionId": "x", "prompt": []},
            timeout_seconds=30.0,
            cancel_check=lambda: _Fake._interrupt_requested,
        )
    elapsed = time.time() - start
    # Should abort within the poll tick (0.1s) plus the trigger delay (0.15s).
    assert elapsed < 1.0, f"cancel took too long ({elapsed:.2f}s)"


def test_tool_trace_is_thread_safe_under_concurrent_updates(tmp_path):
    """Stress test: a background thread hammers start/update events while the
    main thread repeatedly reads `client.tool_trace`. No exceptions should
    surface (unprotected list append/iteration under the GIL can still let
    a stale record leak to `to_dict`)."""
    client, _ = _make_client_with_fake_session(tmp_path)

    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        try:
            for i in range(500):
                tid = f"tc-{i}"
                client._handle_session_update(
                    {
                        "sessionUpdate": "tool_call_start",
                        "toolCallId": tid,
                        "title": "stress_tool",
                        "rawInput": {"i": i},
                    },
                    dispatch_ctx={"text_parts": [], "reasoning_parts": []},
                )
                client._handle_session_update(
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": tid,
                        "rawOutput": f"r-{i}",
                        "status": "completed",
                    },
                    dispatch_ctx={"text_parts": [], "reasoning_parts": []},
                )
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                snap = client.tool_trace
                # Serialize each record — mirrors what _create_chat_completion
                # does when building hermes_tool_trace.
                _ = [r.to_dict() for r in snap]
        except BaseException as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_w.start()
    t_r.start()
    t_w.join(timeout=10.0)
    t_r.join(timeout=10.0)

    assert not errors, f"race-condition errors surfaced: {errors}"
    assert len(client.tool_trace) == 500
    # All records should be in terminal state.
    assert all(r.status == "completed" for r in client.tool_trace)
    assert all(r.completed_at is not None for r in client.tool_trace)
