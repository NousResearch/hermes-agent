#!/usr/bin/env python3
"""Small, fail-closed ACP stdio client for a secondary Hermes profile."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

MAX_TIMEOUT = 600.0
DEFAULT_TIMEOUT = 300.0


def _rpc(process: subprocess.Popen[str], method: str, params: dict[str, Any], request_id: int, deadline: float) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
    process.stdin.flush()
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            break
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "method" in message and "id" in message:
            _answer_request(process, message)
            continue
        if message.get("id") != request_id:
            continue
        if "error" in message:
            error = message["error"]
            raise RuntimeError(f"ACP {method} failed: {error.get('message', error)}")
        return message.get("result") or {}
    raise TimeoutError(f"Timed out waiting for ACP {method}.")


def _answer_request(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
    """Reject permissions and capabilities not intentionally implemented here."""
    assert process.stdin is not None
    method = message.get("method")
    if method == "session/request_permission":
        result: dict[str, Any] = {"outcome": {"outcome": "cancelled"}}
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"], "result": result}
    else:
        response = {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"Unsupported ACP callback: {method}"}}
    process.stdin.write(json.dumps(response) + "\n")
    process.stdin.flush()


def _load_config(path: Path) -> tuple[list[str], Path, float]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config: {exc}") from exc
    profile = str(config.get("profile") or "").strip()
    if not profile or profile.startswith("REPLACE_"):
        raise ValueError("config.profile must name the secondary profile")
    cwd = Path(str(config.get("cwd") or "")).expanduser()
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ValueError("config.cwd must be an existing absolute directory")
    timeout = float(config.get("timeout_seconds", DEFAULT_TIMEOUT))
    if not 1.0 <= timeout <= MAX_TIMEOUT:
        raise ValueError(f"timeout_seconds must be between 1 and {int(MAX_TIMEOUT)}")
    command = str(config.get("command") or "hermes").strip()
    args = config.get("args")
    if args is None:
        argv = [command, "-p", profile, "acp"]
    elif isinstance(args, list) and all(isinstance(item, str) and item for item in args):
        argv = [command, *args]
    else:
        raise ValueError("config.args must be a list of non-empty strings")
    if not command or os.path.basename(command) in {"sh", "bash", "cmd", "powershell"}:
        raise ValueError("config.command must be a direct ACP executable, not a shell")
    return argv, cwd, timeout


def call(config_path: Path, prompt: str) -> str:
    argv, cwd, timeout = _load_config(config_path)
    deadline = time.monotonic() + timeout
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            argv, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=dict(os.environ),
        )
        _rpc(process, "initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "hermes-acp-delegate", "title": "Hermes ACP Delegate", "version": "0.1.0"},
        }, 1, deadline)
        session = _rpc(process, "session/new", {"cwd": str(cwd), "mcpServers": []}, 2, deadline)
        session_id = str(session.get("sessionId") or "").strip()
        if not session_id:
            raise RuntimeError("ACP session/new returned no sessionId")
        # The response arrives as session/update notifications while this request is outstanding.
        # _rpc currently consumes them; use a dedicated prompt exchange to retain message chunks.
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {
            "sessionId": session_id, "prompt": [{"type": "text", "text": prompt}],
        }}) + "\n")
        process.stdin.flush()
        chunks: list[str] = []
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("method") == "session/update":
                update = (message.get("params") or {}).get("update") or {}
                if update.get("sessionUpdate") == "agent_message_chunk":
                    content = update.get("content") or {}
                    if isinstance(content, dict) and content.get("text"):
                        chunks.append(str(content["text"]))
                continue
            if "method" in message and "id" in message:
                _answer_request(process, message)
                continue
            if message.get("id") == 3:
                if "error" in message:
                    raise RuntimeError(f"ACP session/prompt failed: {message['error']}")
                return "".join(chunks)
        raise TimeoutError("Timed out waiting for ACP session/prompt.")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not start ACP command: {exc}") from exc
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    try:
        print(call(args.config, args.prompt))
        return 0
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"acp-delegate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
