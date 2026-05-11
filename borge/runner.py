"""
BorgeRunner — Standalone Agent Loop

A minimal, self-contained agent loop that pairs the Borge cognitive layer
with the Anthropic SDK.  No Hermes required.

Usage:
    from borge import BorgeRunner
    runner = BorgeRunner(model="claude-opus-4-7")
    runner.run("help me refactor this function")

    # Or stream:
    for chunk in runner.stream("explain the bug"):
        print(chunk, end="", flush=True)
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .agent import BorgeAgent

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
MAX_TURNS = 20


# ---------------------------------------------------------------------------
# Built-in minimal tools (bash + read_file + write_file)
# ---------------------------------------------------------------------------

def _tool_bash(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        out = result.stdout
        err = result.stderr
        if err and not out:
            return err[:4000]
        return (out + ("\n" + err if err else ""))[:4000]
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def _tool_read_file(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {path}"
        return p.read_text(errors="replace")[:8000]
    except Exception as e:
        return f"Error: {e}"


def _tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


BUILTIN_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command and return stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from disk.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path to read"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]

_TOOL_HANDLERS: dict[str, Callable] = {
    "bash": lambda args: _tool_bash(args["command"]),
    "read_file": lambda args: _tool_read_file(args["path"]),
    "write_file": lambda args: _tool_write_file(args["path"], args["content"]),
}


# ---------------------------------------------------------------------------
# BorgeRunner
# ---------------------------------------------------------------------------

class BorgeRunner:
    """
    Standalone Borge agent — no Hermes dependency.

    Pairs BorgeAgent cognitive layer with a direct Anthropic API loop.
    Users can supply extra tools or replace the tool handler entirely.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        extra_tools: Optional[list[dict]] = None,
        tool_handler: Optional[Callable[[str, dict], str]] = None,
        db_path: Optional[str] = None,
        soul_path: Optional[str] = None,
        config: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._tools = BUILTIN_TOOLS + (extra_tools or [])
        self._tool_handler = tool_handler
        self._system = system_prompt or (
            "You are Borge, a thoughtful AI assistant with a cognitive architecture. "
            "You reason carefully before acting, acknowledge uncertainty, and adapt "
            "your communication style to how the user is feeling."
        )

        self._cognitive = BorgeAgent(
            agent_backend=None,
            db_path=db_path,
            soul_path=soul_path,
            config=config or {},
        )
        self._session_id: str = ""
        self._history: list[dict] = []

    # ── Public interface ──────────────────────────────────────────────────

    def run(self, user_message: str, session_id: str = "") -> str:
        """Run one user turn and return the final assistant response."""
        return "".join(self._turn(user_message, session_id, stream=False))

    def stream(self, user_message: str, session_id: str = "") -> Iterator[str]:
        """Stream one user turn, yielding text chunks."""
        yield from self._turn(user_message, session_id, stream=True)

    def reset(self) -> None:
        """End current session and clear history."""
        if self._session_id:
            self._cognitive.on_session_end(
                session_id=self._session_id, messages=self._history
            )
        self._history = []
        self._session_id = ""

    # ── Internal loop ─────────────────────────────────────────────────────

    def _turn(self, user_message: str, session_id: str, stream: bool) -> Iterator[str]:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for standalone use: pip install anthropic"
            )

        if not self._session_id:
            import uuid
            self._session_id = session_id or str(uuid.uuid4())[:8]
            self._cognitive.on_session_start()

        # Borge pre-turn: cognitive context injected as prefix on user message
        ctx = self._cognitive.pre_turn(user_message, self._history)
        effective_message = f"{ctx}\n\n{user_message}" if ctx else user_message
        self._history.append({"role": "user", "content": effective_message})

        client = anthropic.Anthropic(api_key=self._api_key)
        final_response = ""

        for _ in range(MAX_TURNS):
            response = client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=self._system,
                tools=self._tools,
                messages=self._history,
            )

            text_parts: list[str] = []
            tool_calls: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    if stream:
                        yield block.text
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

            full_text = "".join(text_parts)
            final_response = full_text or final_response
            self._history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn" or not tool_calls:
                break

            tool_results = []
            for tc in tool_calls:
                result = self._execute_tool(tc["name"], tc["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })
                self._cognitive.post_tool(tc["name"], result)

            self._history.append({"role": "user", "content": tool_results})

        self._cognitive.post_tool("assistant_turn", final_response)

        if not stream:
            yield final_response

    def _execute_tool(self, name: str, args: dict) -> str:
        if self._tool_handler:
            try:
                return str(self._tool_handler(name, args))
            except Exception as e:
                return f"Tool error: {e}"
        handler = _TOOL_HANDLERS.get(name)
        if handler:
            try:
                return str(handler(args))
            except Exception as e:
                return f"Tool error: {e}"
        return f"Unknown tool: {name}"
