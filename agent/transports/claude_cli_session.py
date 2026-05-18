"""Whole-turn adapter for Claude Code CLI.

This runtime intentionally bypasses Hermes' provider auth and API transports.
It shells out to ``claude -p`` so paid Claude plans authenticated in Claude
Code stay the source of truth.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class ClaudeCliTurnResult:
    final_text: str = ""
    error: Optional[str] = None
    interrupted: bool = False
    returncode: int = 0


class ClaudeCliSession:
    """Run one Hermes turn through the local ``claude`` CLI."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        claude_bin: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        extra_args: Optional[list[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._claude_bin = (
            claude_bin or os.getenv("HERMES_CLAUDE_CLI_BIN") or "claude"
        )
        self._model = (model or os.getenv("HERMES_CLAUDE_CLI_MODEL") or "").strip()
        self._timeout_seconds = timeout_seconds or float(
            os.getenv("HERMES_CLAUDE_CLI_TIMEOUT_SECONDS", "900")
        )
        env_args = shlex.split(os.getenv("HERMES_CLAUDE_CLI_EXTRA_ARGS", ""))
        self._extra_args = list(extra_args or []) + env_args
        self._env = dict(env) if env is not None else None

    def run_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        user_input: str,
    ) -> ClaudeCliTurnResult:
        prompt = render_claude_cli_prompt(messages=messages, user_input=user_input)
        cmd = [
            self._claude_bin,
            "-p",
            prompt,
            "--no-session-persistence",
            "--output-format",
            "text",
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        cmd.extend(_kanban_worker_hermes_tools_args())
        cmd.extend(self._extra_args)

        try:
            completed = subprocess.run(
                cmd,
                cwd=self._cwd,
                env=self._env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError:
            return ClaudeCliTurnResult(
                error=(
                    f"Claude CLI not found: {self._claude_bin}. "
                    "Install Claude Code or set HERMES_CLAUDE_CLI_BIN."
                ),
                returncode=127,
            )
        except subprocess.TimeoutExpired as exc:
            return ClaudeCliTurnResult(
                error=f"Claude CLI timed out after {self._timeout_seconds:g}s.",
                final_text=(
                    (exc.stdout or "").strip()
                    if isinstance(exc.stdout, str)
                    else ""
                ),
                interrupted=True,
                returncode=124,
            )

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            return ClaudeCliTurnResult(
                final_text=stdout,
                error=stderr or stdout or f"Claude CLI exited with {completed.returncode}.",
                returncode=completed.returncode,
            )
        return ClaudeCliTurnResult(final_text=stdout, returncode=completed.returncode)


def render_claude_cli_prompt(*, messages: list[dict[str, Any]], user_input: str) -> str:
    if not messages:
        return str(user_input or "")

    lines = [
        "You are being invoked by Hermes through Claude Code CLI.",
        "Continue the transcript below and answer the final user message.",
        "",
        "<transcript>",
    ]
    for msg in messages:
        role = str(msg.get("role") or "message").upper()
        content = _content_to_text(msg.get("content"))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            content = (content + "\n" if content else "") + _json_line(
                "tool_calls", tool_calls
            )
        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            content = f"[tool_call_id={tool_call_id}]\n{content}".strip()
        lines.append(f"{role}:\n{content}".rstrip())
        lines.append("")
    lines.append("</transcript>")
    return "\n".join(lines).strip()


_KANBAN_HERMES_TOOL_NAMES: tuple[str, ...] = (
    "kanban_show",
    "kanban_comment",
    "kanban_block",
    "kanban_complete",
    "kanban_heartbeat",
    "kanban_create",
    "kanban_list",
    "kanban_unblock",
    "kanban_link",
    "skill_view",
    "skills_list",
)

_CLAUDE_CODE_WORKER_TOOLS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "LS",
    "Bash",
    "TodoWrite",
)

_KANBAN_CLI_SYSTEM_PROMPT = (
    "You are running inside Hermes' Claude CLI bridge as a Kanban worker. "
    "Hermes tools are available as MCP tools from the hermes-tools server "
    "(for example mcp__hermes-tools__kanban_show and "
    "mcp__hermes-tools__kanban_block). Use those MCP tools for Kanban "
    "lifecycle operations and end every assigned task with kanban_complete "
    "or kanban_block; do not finish with prose only."
)


def _kanban_worker_hermes_tools_args() -> list[str]:
    """Attach Hermes' MCP tool bridge for dispatcher-spawned Claude workers.

    The plain Claude CLI runtime otherwise has only Claude Code's tools. A
    Kanban worker needs Hermes' own kanban_* tools so the dispatcher can observe
    block/complete transitions instead of seeing a clean prose-only exit.
    """
    if not os.getenv("HERMES_KANBAN_TASK"):
        return []

    mcp_config = {
        "mcpServers": {
            "hermes-tools": {
                "command": sys.executable,
                "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                "env": _hermes_tools_mcp_env(),
            }
        }
    }
    allowed_tools = list(_CLAUDE_CODE_WORKER_TOOLS)
    allowed_tools.extend(
        f"mcp__hermes-tools__{name}" for name in _KANBAN_HERMES_TOOL_NAMES
    )
    return [
        "--mcp-config",
        json.dumps(mcp_config, separators=(",", ":")),
        "--strict-mcp-config",
        "--allowedTools",
        ",".join(allowed_tools),
        "--append-system-prompt",
        _KANBAN_CLI_SYSTEM_PROMPT,
    ]


def _hermes_tools_mcp_env() -> dict[str, str]:
    repo_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = os.getenv("PYTHONPATH", "")
    pythonpath = (
        repo_root
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )

    env: dict[str, str] = {
        "PYTHONPATH": pythonpath,
        "HERMES_QUIET": "1",
        "HERMES_REDACT_SECRETS": "true",
    }

    for key in (
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "HERMES_HOME",
        "HERMES_PROFILE",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_BOARD",
        "HERMES_TENANT",
    ):
        value = os.getenv(key)
        if value:
            env[key] = value
    return env


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item.get("type") == "image_url":
                    parts.append("[image omitted]")
                else:
                    parts.append(_json_dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return _json_dumps(content)


def _json_line(label: str, value: Any) -> str:
    return f"[{label}: {_json_dumps(value)}]"


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)
