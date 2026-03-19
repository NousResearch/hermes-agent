"""
Claude Code CLI Bridge — Wraps `claude` CLI into the swarm.

"Unlimited compute through Max subscription. The Professor approves."

Capabilities:
- Code editing, file read/write via tool use
- Session resume for multi-turn conversations
- Structured JSON output
- Streaming via stream-json format
- Budget cap per query

Spawns: claude -p --output-format json --model sonnet <prompt>
"""

import json
import os
import shutil
from typing import List, Optional, Set

from loguru import logger

from .base import CLIBridge, CLICapability, CLIResponse
from .rate_tracker import get_rate_tracker


class ClaudeCodeBridge(CLIBridge):
    """
    Claude Code CLI bridge.

    Uses the authenticated Claude Code CLI (Max subscription)
    for code editing, analysis, and multi-turn conversations.
    """

    def __init__(
        self,
        model: str = "sonnet",
        claude_path: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_concurrent: int = 2,
        default_timeout: int = 120,
        max_budget_usd: float = 0.50,
        allowed_tools: Optional[List[str]] = None,
    ):
        # Resolve claude executable
        if claude_path:
            executable = claude_path
        else:
            # Try common locations
            executable = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")

        super().__init__(
            cli_name="claude_code",
            executable=executable,
            max_concurrent=max_concurrent,
            default_timeout=default_timeout,
            working_dir=working_dir or "/workspace/Cosmos",
        )

        self.model = model
        self.max_budget_usd = max_budget_usd
        self.allowed_tools = allowed_tools or ["Read", "Grep", "Glob"]

        # Register with rate tracker (no hard daily limit for Max subscription)
        tracker = get_rate_tracker()
        tracker.register_cli("claude_code", daily_limit=None, minute_limit=10)

    def get_capabilities(self) -> Set[CLICapability]:
        return {
            CLICapability.CODE_EDIT,
            CLICapability.FILE_READ,
            CLICapability.FILE_WRITE,
            CLICapability.STRUCTURED_OUTPUT,
            CLICapability.STREAMING,
            CLICapability.SESSION_RESUME,
            CLICapability.TOOL_USE,
        }

    def build_command(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        streaming: bool = False,
    ) -> List[str]:
        model = model or self.model
        cmd = [self.executable]

        # Print mode (non-interactive)
        cmd.append("-p")

        # Output format
        if streaming:
            cmd.extend(["--output-format", "stream-json"])
        else:
            cmd.extend(["--output-format", "json"])

        # Model selection
        cmd.extend(["--model", model])

        # System prompt
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        # Session resume for multi-turn
        if session_id:
            cmd.extend(["--resume", session_id])

        # Tool restrictions (safe read-only by default for swarm queries)
        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        # Budget cap
        if self.max_budget_usd:
            cmd.extend(["--max-turns", "5"])

        # The prompt itself
        cmd.append(prompt)

        return cmd

    def parse_output(self, raw: str) -> CLIResponse:
        """Parse Claude Code JSON output."""
        try:
            data = json.loads(raw)

            # Claude Code JSON format has: result, session_id, cost_usd, num_turns, is_error
            content = data.get("result", "")
            is_error = data.get("is_error", False)

            if is_error:
                return CLIResponse(
                    content="",
                    error=content or "Claude Code returned an error",
                    model=f"claude-{self.model}",
                    session_id=data.get("session_id"),
                    cost_usd=data.get("cost_usd", 0),
                )

            return CLIResponse(
                content=content,
                model=f"claude-{self.model}",
                success=True,
                session_id=data.get("session_id"),
                cost_usd=data.get("cost_usd", 0),
                tokens_used=data.get("num_turns", 0),  # turns as proxy
            )

        except json.JSONDecodeError:
            # Fallback: treat raw output as plain text (--print mode compat)
            if raw:
                return CLIResponse(
                    content=raw,
                    model=f"claude-{self.model}",
                    success=True,
                )
            return CLIResponse(error="Failed to parse Claude Code output")

    def parse_stream_line(self, line: str) -> Optional[str]:
        """Parse a stream-json line from Claude Code."""
        try:
            data = json.loads(line)
            msg_type = data.get("type", "")

            # Content block delta
            if msg_type == "assistant" and "message" in data:
                return data["message"]

            # Content delta in stream-json format
            if msg_type == "content_block_delta":
                delta = data.get("delta", {})
                return delta.get("text")

            # Result message
            if msg_type == "result":
                return data.get("result")

        except json.JSONDecodeError:
            # Could be partial line, skip
            pass

        return None

    def detect_rate_limit(self, raw: str, returncode: int) -> bool:
        """Check if Claude Code hit a rate limit."""
        rate_signals = [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
            "concurrent",
        ]
        lower = raw.lower()
        return any(signal in lower for signal in rate_signals)


# Factory function
_claude_bridge: Optional[ClaudeCodeBridge] = None


def get_claude_code_bridge(**kwargs) -> ClaudeCodeBridge:
    """Get or create the global Claude Code bridge."""
    global _claude_bridge
    if _claude_bridge is None:
        _claude_bridge = ClaudeCodeBridge(**kwargs)
    return _claude_bridge
