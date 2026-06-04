"""Claude Code subprocess transport — runs turns through the ``claude`` CLI.

This transport spawns the official Anthropic ``claude`` CLI as a subprocess
and communicates via ``--output-format stream-json`` (for streaming) or
``--output-format json`` (for single-shot).  The CLI handles its own
authentication (OAuth subscription tokens from macOS Keychain), so all
usage bills against the user's Claude Pro/Max subscription — no API key
needed.

Architecture mirror: ``codex_app_server_session.py`` does the same for
OpenAI's ``codex`` CLI.

Lifecycle::

    session = ClaudeSubprocessSession(cwd="/home/x/proj")
    result = session.run_turn(user_input="explain this function")
    # result.final_text   → assistant response text
    # result.tool_calls   → any tool use the CLI performed
    # result.error        → set if the turn failed
    session.close()

Threading model: synchronous from the caller's perspective.  The ``claude``
CLI is invoked per-turn with ``-p`` (print mode), not as a long-running
server.  Each turn is a fresh subprocess invocation.  Session continuity is
maintained by passing ``--resume <session_id>`` on subsequent turns.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# How many stderr lines to surface in error messages.
_STDERR_TAIL_LINES = 12

# Auth failure markers in stderr — redirect to ``claude login``.
_AUTH_FAILURE_HINTS = (
    "not authenticated",
    "unauthenticated",
    "unauthorized",
    "401",
    "expired",
    "please log in",
    "please login",
    "oauth",
    "sign in",
    "session expired",
    "ANTHROPIC_API_KEY",
    "no valid credentials",
)


@dataclass
class ClaudeTurnResult:
    """Result of one user→assistant turn through the ``claude`` CLI."""

    final_text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    model: str = ""
    session_id: str = ""
    duration_ms: float = 0.0
    num_turns: int = 0
    is_error: bool = False
    error: Optional[str] = None
    should_retire: bool = False
    # Raw JSON messages from stream-json output
    raw_messages: list[dict] = field(default_factory=list)


class ClaudeSubprocessSession:
    """Drives turns through the ``claude`` CLI subprocess.

    Each ``run_turn()`` invokes ``claude -p --output-format stream-json``
    as a subprocess.  The CLI handles authentication, model selection,
    and tool execution internally.

    Session continuity: the first turn creates a Claude Code session.
    Subsequent turns use ``--resume <session_id>`` to continue in the
    same context window.
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        model: str = "sonnet",
        max_turns: int = 25,
        system_prompt: str | None = None,
        allowed_tools: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ):
        self.cwd = cwd or os.getcwd()
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.extra_flags = extra_flags or []
        self._session_id: Optional[str] = None
        self._closed = False
        self._total_cost_usd = 0.0

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def run_turn(
        self,
        user_input: str,
        *,
        timeout_seconds: int = 600,
    ) -> ClaudeTurnResult:
        """Execute one turn through the ``claude`` CLI.

        Args:
            user_input: The user message to send.
            timeout_seconds: Maximum time to wait for the CLI (default 10 min).

        Returns:
            ClaudeTurnResult with the assistant's response.
        """
        if self._closed:
            return ClaudeTurnResult(
                is_error=True,
                error="Session is closed",
                should_retire=True,
            )

        cmd = self._build_command(user_input)
        logger.info("Claude subprocess: %s", " ".join(cmd[:6]) + " ...")

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=timeout_seconds,
                env=self._build_env(),
            )
        except subprocess.TimeoutExpired:
            logger.error("Claude subprocess timed out after %ds", timeout_seconds)
            return ClaudeTurnResult(
                is_error=True,
                error=f"Claude CLI timed out after {timeout_seconds}s",
                should_retire=True,
            )
        except FileNotFoundError:
            return ClaudeTurnResult(
                is_error=True,
                error="claude CLI not found in PATH — install with: npm i -g @anthropic-ai/claude-code",
                should_retire=True,
            )
        except Exception as exc:
            logger.exception("Claude subprocess failed")
            return ClaudeTurnResult(
                is_error=True,
                error=f"Claude CLI failed: {exc}",
                should_retire=True,
            )

        duration_ms = (time.monotonic() - t0) * 1000

        stderr = proc.stderr.strip() if proc.stderr else ""
        stdout = proc.stdout.strip() if proc.stdout else ""

        # The CLI returns exit 0 even for auth failures when using
        # --output-format json/stream-json — it writes a structured
        # result JSON with is_error=true.  Check stdout first.
        if stdout:
            result = self._parse_stream_output(stdout, duration_ms=duration_ms)
            # Check for structured error in the result
            if result.is_error and not result.error:
                result.error = "Claude CLI returned an error (check `claude login`)"
            if result.is_error and self._is_auth_failure(result.error or stderr):
                result.should_retire = True
            return result

        # No stdout — check for process failure
        if proc.returncode != 0:
            error_msg = self._classify_error(stderr, proc.returncode)
            logger.error("Claude CLI exit %d: %s", proc.returncode, error_msg)
            return ClaudeTurnResult(
                is_error=True,
                error=error_msg,
                duration_ms=duration_ms,
                should_retire=self._is_auth_failure(stderr),
            )

        return ClaudeTurnResult(
            is_error=True,
            error="Claude CLI returned no output",
            duration_ms=duration_ms,
        )

    def _build_command(self, user_input: str) -> list[str]:
        """Build the ``claude`` CLI command."""
        cmd = [
            "claude",
            "-p",  # print mode (non-interactive)
            "--output-format", "json",
            "--model", self.model,
        ]

        # Session continuity — resume the previous session so context
        # persists across turns (mirrors Codex's thread_id).
        if self._session_id:
            cmd.extend(["--resume", self._session_id])

        # System prompt (only on the first turn to avoid duplication)
        if self.system_prompt and not self._session_id:
            cmd.extend(["--system-prompt", self.system_prompt])

        # Tool restrictions
        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        # Extra flags from caller
        cmd.extend(self.extra_flags)

        # The prompt itself (must be last)
        cmd.append(user_input)

        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build the environment for the subprocess.

        Inherits the parent environment.  The ``claude`` CLI reads its own
        auth from macOS Keychain / ~/.claude/.credentials.json, so we
        don't need to pass any API keys explicitly.

        NOTE: We do NOT set CI=1 because that skips the OAuth token
        refresh flow.  The CLI needs to be able to refresh expired tokens.
        """
        env = dict(os.environ)
        # Remove CI flag if set — it blocks OAuth refresh
        env.pop("CI", None)
        return env

    def _parse_stream_output(
        self,
        stdout: str,
        *,
        duration_ms: float = 0.0,
    ) -> ClaudeTurnResult:
        """Parse the stream-json output from the ``claude`` CLI.

        The CLI can emit:
        - Newline-delimited JSON objects (stream-json format)
        - A single JSON object (json format / result)

        We handle both formats and extract the final assistant text.
        """
        messages: list[dict] = []
        final_text_parts: list[str] = []
        session_id = self._session_id
        model = self.model
        cost_usd = 0.0
        num_turns = 0
        tool_calls: list[dict] = []
        is_error = False
        error_msg: str | None = None

        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Some lines may be non-JSON (progress indicators, etc.)
                continue

            messages.append(msg)
            msg_type = msg.get("type", "")

            # Extract session ID from system messages
            if msg_type == "system" and msg.get("session_id"):
                session_id = msg["session_id"]

            # Collect assistant text from streaming messages
            if msg_type == "assistant":
                content = msg.get("message", {}).get("content", "")
                if isinstance(content, str) and content:
                    final_text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                final_text_parts.append(text)
                        elif isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": block.get("input", {}),
                            })

            # Extract cost and final text from result message
            # This handles both stream-json (last message) and single json
            if msg_type == "result":
                cost_usd = msg.get("total_cost_usd", msg.get("cost_usd", 0.0)) or 0.0
                num_turns = msg.get("num_turns", 0) or 0
                model = msg.get("model", model) or model
                if msg.get("session_id"):
                    session_id = msg["session_id"]

                # Check for structured errors
                if msg.get("is_error"):
                    is_error = True
                    result_text = msg.get("result", "")
                    if result_text:
                        error_msg = str(result_text)
                    api_error = msg.get("api_error_status")
                    if api_error == 401:
                        error_msg = (
                            "Claude authentication failed (401). "
                            "Run `claude login` to re-authenticate with your "
                            "Claude Pro/Max subscription."
                        )
                else:
                    # Success — the result field has the final text
                    result_text = msg.get("result", "")
                    if isinstance(result_text, str) and result_text:
                        final_text_parts = [result_text]

        # Persist session ID for resume on next turn
        if session_id:
            self._session_id = session_id

        self._total_cost_usd += cost_usd

        return ClaudeTurnResult(
            final_text="\n".join(final_text_parts) if not is_error else "",
            tool_calls=tool_calls,
            cost_usd=cost_usd,
            model=model,
            session_id=session_id or "",
            duration_ms=duration_ms,
            num_turns=num_turns,
            is_error=is_error,
            error=error_msg,
            raw_messages=messages,
        )

    def _classify_error(self, stderr: str, returncode: int) -> str:
        """Classify a subprocess error from stderr."""
        if self._is_auth_failure(stderr):
            return (
                "Claude CLI authentication failed — run `claude login` to "
                "re-authenticate with your Anthropic subscription."
            )
        # Return last N stderr lines
        lines = stderr.strip().splitlines()
        tail = lines[-_STDERR_TAIL_LINES:] if len(lines) > _STDERR_TAIL_LINES else lines
        tail_text = "\n".join(tail).strip()
        if tail_text:
            return f"Claude CLI exit {returncode}:\n{tail_text}"
        return f"Claude CLI exited with code {returncode}"

    def _is_auth_failure(self, stderr: str) -> bool:
        """Check if stderr indicates an auth failure."""
        lower = stderr.lower()
        return any(hint in lower for hint in _AUTH_FAILURE_HINTS)

    def close(self):
        """Mark the session as closed."""
        self._closed = True
        logger.info(
            "Claude subprocess session closed (session=%s, total_cost=$%.4f)",
            self._session_id or "none",
            self._total_cost_usd,
        )

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd
