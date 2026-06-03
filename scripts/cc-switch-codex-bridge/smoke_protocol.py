#!/usr/bin/env python3
"""Live protocol smoke checks for the Claude Code Codex bridge.

The script talks to an already-running local bridge. It reads the auth token
only from ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or BRIDGE_AUTH_TOKEN and
never prints it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


BASH_TOOL = {
    "name": "Bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}


def _token() -> str:
    return (
        os.environ.get("BRIDGE_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    )


def _request(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json"}
    token = _token()
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _stream(base_url: str, payload: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    data = json.dumps({**payload, "stream": True}, ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json"}
    token = _token()
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages",
        data=data,
        headers=headers,
        method="POST",
    )
    events: list[dict[str, Any]] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        current_event = None
        current_data: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event: "):
                current_event = line[len("event: ") :]
            elif line.startswith("data: "):
                current_data.append(line[len("data: ") :])
            elif line == "" and current_event:
                try:
                    payload_obj = json.loads("".join(current_data))
                except Exception:
                    payload_obj = {"raw": "".join(current_data)}
                events.append({"event": current_event, "payload": payload_obj})
                if current_event == "message_stop":
                    break
                current_event = None
                current_data = []
    return events


def _block_types(resp: dict[str, Any]) -> list[str]:
    return [str(block.get("type")) for block in resp.get("content", []) if isinstance(block, dict)]


def _tool_use(resp: dict[str, Any]) -> dict[str, Any] | None:
    for block in resp.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return block
    return None


def _tool_result_message(tool: dict[str, Any], output: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool.get("id"),
                "content": [{"type": "text", "text": output}],
            }
        ],
    }


def _valid_event_order(events: list[dict[str, Any]]) -> bool:
    names = [event["event"] for event in events]
    if not names or names[0] != "message_start" or names[-1] != "message_stop":
        return False
    if "message_delta" not in names:
        return False
    return names.index("message_delta") < names.index("message_stop")


def run(base_url: str, timeout: float) -> dict[str, Any]:
    results: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    two_tool_payload = {
        "model": "gpt-5.5-high",
        "max_tokens": 1024,
        "tools": [BASH_TOOL],
        "messages": [
            {
                "role": "user",
                "content": "Use Bash to run pwd, then after that result use Bash to run git status --short, then summarize.",
            }
        ],
    }
    first = _request(base_url, two_tool_payload, timeout)
    first_tool = _tool_use(first)
    second = None
    if first_tool:
        second_payload = dict(two_tool_payload)
        second_payload["messages"] = two_tool_payload["messages"] + [
            {"role": "assistant", "content": [first_tool]},
            _tool_result_message(first_tool, "/repo"),
        ]
        second = _request(base_url, second_payload, timeout)
    results["two_tool_loop"] = {
        "first_stop_reason": first.get("stop_reason"),
        "first_block_types": _block_types(first),
        "continued_block_types": _block_types(second) if second else [],
        "continued_stop_reason": second.get("stop_reason") if second else None,
        "pass": first.get("stop_reason") == "tool_use" and bool(second),
    }

    continuation_payload = {
        "model": "gpt-5.5-high",
        "max_tokens": 1024,
        "tools": [BASH_TOOL],
        "messages": [
            {"role": "user", "content": "After first tool result, immediately use a second tool. Do not stop."}
        ],
    }
    cont_first = _request(base_url, continuation_payload, timeout)
    cont_tool = _tool_use(cont_first)
    cont_second = None
    if cont_tool:
        cont_payload = dict(continuation_payload)
        cont_payload["messages"] = continuation_payload["messages"] + [
            {"role": "assistant", "content": [cont_tool]},
            _tool_result_message(cont_tool, "first result"),
        ]
        cont_second = _request(base_url, cont_payload, timeout)
    results["explicit_continuation"] = {
        "first_stop_reason": cont_first.get("stop_reason"),
        "second_stop_reason": cont_second.get("stop_reason") if cont_second else None,
        "second_block_types": _block_types(cont_second) if cont_second else [],
        "pass": bool(cont_second) and cont_second.get("stop_reason") != "end_turn",
    }

    parity_payload = {
        "model": "gpt-5.5-high",
        "max_tokens": 1024,
        "tools": [BASH_TOOL],
        "messages": [{"role": "user", "content": "Use Bash once to print pwd."}],
    }
    nonstream = _request(base_url, parity_payload, timeout)
    stream_events = _stream(base_url, parity_payload, timeout)
    stream_block_types = [
        event["payload"].get("content_block", {}).get("type")
        for event in stream_events
        if event["event"] == "content_block_start"
    ]
    message_delta = next((event["payload"] for event in stream_events if event["event"] == "message_delta"), {})
    results["streaming_vs_nonstream"] = {
        "nonstream_block_types": _block_types(nonstream),
        "stream_block_types": stream_block_types,
        "nonstream_stop_reason": nonstream.get("stop_reason"),
        "stream_stop_reason": message_delta.get("delta", {}).get("stop_reason"),
        "event_order_valid": _valid_event_order(stream_events),
        "pass": _valid_event_order(stream_events)
        and nonstream.get("stop_reason") == message_delta.get("delta", {}).get("stop_reason"),
    }

    thinking_payload = {
        "model": "claude-sonnet-4-5-latest",
        "thinking": {"type": "enabled", "budget_tokens": 16000},
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Think through why a two-step tool loop might stop early, then answer briefly."}],
    }
    thinking_events = _stream(base_url, thinking_payload, timeout)
    deltas = [
        event["payload"].get("delta", {}).get("type")
        for event in thinking_events
        if event["event"] == "content_block_delta"
    ]
    results["thinking_alias"] = {
        "model": "claude-sonnet-4-5-latest",
        "thinking_delta": "thinking_delta" in deltas,
        "signature_delta": "signature_delta" in deltas,
        "event_order_valid": _valid_event_order(thinking_events),
        "pass": "thinking_delta" in deltas and _valid_event_order(thinking_events),
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:15722"))
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        result = run(args.base_url, args.timeout)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(json.dumps({"pass": False, "error": exc.__class__.__name__, "message": str(exc)}, indent=2))
        return 2
    ok = all(v.get("pass") for v in result.values() if isinstance(v, dict) and "pass" in v)
    result["pass"] = ok
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
