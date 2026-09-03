"""A stand-in for the ``claude`` CLI that speaks stream-json from canned fixtures.

Used by ``test_claude_code_session.py`` so the session adapter can be driven
end-to-end (real subprocess, real pipes, real reader threads) without the
Claude Code binary or a subscription.

Protocol facts it reproduces on purpose:

* Emits NOTHING until the first user message arrives on stdin (the real CLI
  does the same — see ClaudeCodeSession.ensure_started).
* ``system/init`` is sent once, right before the first turn's events.
* Text turns stream ``stream_event`` text deltas, then the ``assistant``
  message, then ``result``.
* A message containing ``TOOL`` performs an MCP tool call:
  ``assistant`` (tool_use) -> ``user`` (tool_result) -> ``assistant`` (text)
  -> ``result``.
* ``TRAILER`` appends a ``rate_limit_event`` immediately after ``result`` in
  the same flush (drain test).
* ``CRASH`` exits with status 3 without a ``result``.
* ``HANG`` never answers until a ``control_request`` interrupt arrives, then
  emits a ``result``.
* ``PERM`` (only when ``--permission-prompt-tool stdio`` is on argv) asks
  permission for a Bash call via ``control_request/can_use_tool`` and waits
  for the ``control_response``: allow -> ``tool_result`` "PERM-RAN";
  deny -> ``tool_result`` ``is_error`` carrying the message.
* With ``FAKE_CLAUDE_REJECT_RESUME=1`` and ``--resume`` on argv the fake
  answers the first user message with an error ``result`` ("No conversation
  found"), like the real CLI does for an unusable transcript. With
  ``--session-id`` naming an id listed in ``FAKE_CLAUDE_USED_IDS``
  (comma-separated) it exits 1 with "already in use", also like the CLI.

Argv and environment are recorded to ``$FAKE_CLAUDE_RECORD`` (JSON) so the
tests can assert on the spawn contract.
"""

from __future__ import annotations

import json
import os
import sys
import time

SESSION_ID = "sess-fake-0001"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _text_turn(text: str, *, trailer: bool = False) -> None:
    for i in range(0, len(text), 4):
        _emit(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text[i:i + 4]},
                },
            }
        )
    _emit(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            "session_id": SESSION_ID,
        }
    )
    _result(text)
    if trailer:
        _emit(
            {
                "type": "rate_limit_event",
                "rate_limit_info": {"unifiedWindows": {"five_hour": {"utilization": 0.1}}},
            }
        )


def _result(text: str, *, is_error: bool = False) -> None:
    _emit(
        {
            "type": "result",
            "subtype": "success" if not is_error else "error_during_execution",
            "is_error": is_error,
            "result": text,
            "session_id": SESSION_ID,
            "num_turns": 1,
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 7,
            },
            "total_cost_usd": 0.0,
            "modelUsage": {"claude-fake": {"contextWindow": 200000}},
        }
    )


def _tool_turn() -> None:
    _emit(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "toolu_fake_1",
                        "name": "mcp__hermes-tools__web_search",
                        "input": {"query": "swift version"},
                    },
                ],
            },
        }
    )
    _emit(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fake_1",
                        "content": [{"type": "text", "text": "Swift 6.2"}],
                    }
                ],
            },
        }
    )
    _text_turn("The version is 6.2")


def main() -> int:
    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        with open(record, "w", encoding="utf-8") as fh:
            json.dump(
                {"argv": sys.argv[1:], "env": dict(os.environ), "pid": os.getpid(), "cwd": os.getcwd()},
                fh,
            )

    argv = sys.argv[1:]
    prompt_tool = "--permission-prompt-tool" in argv
    resume_id = argv[argv.index("--resume") + 1] if "--resume" in argv else None
    reject_resume = resume_id is not None and (
        os.environ.get("FAKE_CLAUDE_REJECT_RESUME") == "1"
        or os.environ.get("FAKE_CLAUDE_REJECT_RESUME_ID") == resume_id
    )
    if "--session-id" in argv:
        sid = argv[argv.index("--session-id") + 1]
        used = [x for x in os.environ.get("FAKE_CLAUDE_USED_IDS", "").split(",") if x]
        if sid in used:
            sys.stderr.write(f"Error: Session ID {sid} is already in use.\n")
            return 1

    inited = False
    hanging = False
    pending_perm = None
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("type") == "control_request":
            if hanging and (msg.get("request") or {}).get("subtype") == "interrupt":
                hanging = False
                _emit({"type": "control_response", "response": {"subtype": "success", "request_id": msg.get("request_id")}})
                _result("interrupted")
            continue
        if msg.get("type") == "control_response":
            resp = (msg.get("response") or {})
            if pending_perm and resp.get("request_id") == pending_perm:
                pending_perm = None
                inner = resp.get("response") or {}
                if inner.get("behavior") == "allow":
                    _emit({"type": "user", "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_perm_1", "content": "PERM-RAN", "is_error": False}]}})
                    _text_turn("ran it")
                else:
                    _emit({"type": "user", "message": {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_perm_1",
                         "content": inner.get("message") or "denied", "is_error": True}]}})
                    _text_turn("it was denied")
            continue
        if msg.get("type") != "user":
            continue
        if not inited:
            inited = True
            _emit(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": SESSION_ID,
                    "tools": ["Bash", "Read", "ToolSearch"],
                    "mcp_servers": [{"name": "hermes-tools", "status": "connected"}],
                    "model": "claude-fake",
                    "permissionMode": "acceptEdits",
                }
            )
        content = msg["message"]["content"]
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        if reject_resume:
            sys.stderr.write("No conversation found with session ID\n"); sys.stderr.flush()
            _result("No conversation found with session ID", is_error=True)
            continue
        if os.environ.get("FAKE_CLAUDE_FAIL_WARMUP") == "auth":
            _result("Failed to authenticate: OAuth session expired and could not be refreshed", is_error=True)
            continue
        if "PERM" in text and prompt_tool:
            _emit({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_perm_1", "name": "Bash",
                 "input": {"command": "touch marker", "description": "make a marker"}}]}})
            pending_perm = "req-perm-1"
            _emit({"type": "control_request", "request_id": pending_perm, "request": {
                "subtype": "can_use_tool", "tool_name": "Bash", "display_name": "Bash",
                "input": {"command": "touch marker", "description": "make a marker"},
                "description": "make a marker", "permission_suggestions": [], "tool_use_id": "toolu_perm_1"}})
            continue
        if "CRASH" in text:
            sys.stdout.flush()
            os._exit(3)
        if "HANG" in text:
            hanging = True
            continue
        if "TOOL" in text:
            _tool_turn()
        elif "ready" in text:
            _text_turn("ready")
        else:
            _text_turn(f"echo: {text}", trailer="TRAILER" in text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
