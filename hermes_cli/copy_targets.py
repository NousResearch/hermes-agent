"""What ``/copy code`` and ``/copy cmd`` copy: the fenced code blocks of the latest assistant
response that has any, or the shell commands the latest command-running turn executed.

Scope grammar follows Muse Code's ``/copy`` scope picker (full response / one code block / one
command); here it is typed as an argument so it works the same in the classic CLI and the TUI.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.code_fences import parse_code_fences

SHELL_TOOL_NAMES = frozenset({"terminal"})


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def latest_code_blocks(history: list[dict]) -> list[dict]:
    """Closed fences of the newest assistant message that has at least one (unclosed = still
    streaming or malformed, never copied)."""
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        fences = [f for f in parse_code_fences(_text(msg.get("content"))) if f["closed"]]
        if fences:
            return fences
    return []


def _shell_command(tool_call: dict) -> str:
    fn = tool_call.get("function") or {}
    if fn.get("name") not in SHELL_TOOL_NAMES:
        return ""
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return ""
    command = args.get("command") if isinstance(args, dict) else None
    return command.strip() if isinstance(command, str) else ""


def latest_commands(history: list[dict]) -> list[str]:
    """Shell commands of the newest turn (user message → next user message) that ran any, in
    execution order."""
    turn: list[str] = []
    for msg in reversed(history):
        role = msg.get("role")
        if role == "user":
            if turn:
                return turn[::-1]
            continue
        if role == "assistant":
            for call in reversed(msg.get("tool_calls") or []):
                if isinstance(call, dict) and (cmd := _shell_command(call)):
                    turn.append(cmd)
    return turn[::-1]
