#!/usr/bin/env python3
"""ACP JSON-RPC bridge for Claude Code CLI.

Translates Hermes CopilotACPClient's ACP JSON-RPC protocol into
`claude -p` CLI calls.  This lets Hermes delegate tasks to Claude Code
without modifying any upstream Hermes source code.

Usage (via HERMES_COPILOT_ACP_COMMAND env var):
    HERMES_COPILOT_ACP_COMMAND=python3 scripts/claude_acp_bridge.py

Protocol flow:
    1. Client sends "initialize" → respond with server info
    2. Client sends "session/new" → respond with sessionId
    3. Client sends "session/prompt" → run `claude -p`, stream back result
"""

import json
import os
import subprocess
import sys
import uuid


def _read_request():
    """Read one JSON-RPC request from stdin (line-delimited)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _write_msg(msg):
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _send_notification(method, params):
    """Send a JSON-RPC notification (no id)."""
    _write_msg({"jsonrpc": "2.0", "method": method, "params": params})


def _send_response(request_id, result):
    """Send a JSON-RPC success response."""
    _write_msg({"jsonrpc": "2.0", "id": request_id, "result": result})


def _send_error(request_id, code, message):
    """Send a JSON-RPC error response."""
    _write_msg({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    })


def _handle_initialize(request_id, _params):
    _send_response(request_id, {
        "protocolVersion": 1,
        "serverCapabilities": {},
        "serverInfo": {
            "name": "claude-code-bridge",
            "title": "Claude Code Bridge",
            "version": "0.1.0",
        },
    })


def _handle_session_new(request_id, params):
    session_id = str(uuid.uuid4())
    cwd = params.get("cwd", os.getcwd())
    _send_response(request_id, {"sessionId": session_id, "cwd": cwd})


def _handle_session_prompt(request_id, params):
    # Extract prompt text from ACP format
    prompt_parts = params.get("prompt", [])
    prompt_text = ""
    for part in prompt_parts:
        if isinstance(part, dict) and part.get("type") == "text":
            prompt_text += part.get("text", "")

    if not prompt_text:
        _send_error(request_id, -32602, "Empty prompt")
        return

    # Build claude CLI command
    cmd = ["claude", "-p", prompt_text, "--output-format", "json", "--dangerously-skip-permissions"]

    # Add model if specified via env var
    model = os.getenv("CLAUDE_CODE_BRIDGE_MODEL", "").strip()
    if model:
        cmd.extend(["--model", model])

    # Use cwd from session if available
    cwd = params.get("cwd") or os.getcwd()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=cwd if os.path.isdir(cwd) else None,
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"claude exited with code {result.returncode}"
            _send_notification("session/update", {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": f"Error from Claude Code: {error_msg}"},
                }
            })
            _send_response(request_id, {"status": "error"})
            return

        response_data = json.loads(result.stdout)
        response_text = response_data.get("result", "")

    except subprocess.TimeoutExpired:
        response_text = "Error: Claude Code timed out (900s limit)"
    except json.JSONDecodeError:
        response_text = result.stdout if result.stdout else "Error: Could not parse Claude Code output"
    except FileNotFoundError:
        response_text = "Error: 'claude' command not found. Install Claude Code CLI first."
    except Exception as exc:
        response_text = f"Error: {exc}"

    # Send response text as a single agent_message_chunk notification
    _send_notification("session/update", {
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"text": response_text},
        }
    })

    # Send completion response
    _send_response(request_id, {"status": "completed"})


_HANDLERS = {
    "initialize": _handle_initialize,
    "session/new": _handle_session_new,
    "session/prompt": _handle_session_prompt,
}


def main():
    # Log stderr so stdout stays clean for JSON-RPC
    import logging
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] claude-acp-bridge: %(message)s",
    )

    while True:
        request = _read_request()
        if request is None:
            break

        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        handler = _HANDLERS.get(method)
        if handler:
            handler(request_id, params)
        else:
            _send_error(request_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
