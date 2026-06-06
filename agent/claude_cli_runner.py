"""Claude Code CLI subprocess runner for Hermes agent integration.

Spawns `claude -p --output-format stream-json` and collects StreamEvent
objects. This is a simplified runner -- unlike CCDB's version, it collects
all events and returns them (no streaming to Discord needed).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from claude_code_core.parser import parse_line
from claude_code_core.types import MessageType, StreamEvent

logger = logging.getLogger(__name__)

_STRIPPED_ENV_KEYS = frozenset({
    "CLAUDECODE",
    "DISCORD_BOT_TOKEN",
    "DISCORD_TOKEN",
    "API_SECRET_KEY",
})


def _resolve_windows_cmd(cmd_path: Path) -> list[str] | None:
    """Resolve a Windows npm .cmd/.bat wrapper to [node, cli_js].

    npm installs a thin .cmd wrapper that references the real .js entry-point
    via the %~dp0 batch variable. create_subprocess_exec cannot execute .cmd
    files directly, so we parse the wrapper to find the .js path and prepend
    the node executable.

    Returns [node_exe, js_path] on success, None if unresolvable.
    """
    try:
        content = cmd_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'"%~dp0\\([^"]+\.js)"', content)
        if match:
            cli_js = cmd_path.parent / match.group(1)
            if cli_js.exists():
                node = shutil.which("node") or "node"
                return [node, str(cli_js)]
    except OSError:
        pass

    # Fallback: check the standard npm layout.
    cli_js = cmd_path.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
    if cli_js.exists():
        node = shutil.which("node") or "node"
        return [node, str(cli_js)]

    logger.warning(
        "Windows .cmd wrapper %s could not be resolved to a Node.js script; "
        "Claude CLI will likely fail to start",
        cmd_path,
    )
    return None


class ClaudeCliRunner:
    """Manages the lifecycle of a Claude Code CLI subprocess.

    Spawns ``claude -p --output-format stream-json`` processes, sends prompts
    via stdin in stream-json format, and collects parsed StreamEvent objects.
    """

    def __init__(
        self,
        command: str = "claude",
        working_dir: str | None = None,
    ) -> None:
        self.command = command
        self.working_dir = working_dir
        self._process: asyncio.subprocess.Process | None = None
        self._system_prompt_tempfile: str | None = None

    async def run(
        self,
        prompt: str,
        model: str,
        append_system_prompt: str | None = None,
        session_id: str | None = None,
        timeout: int = 600,
    ) -> list[StreamEvent]:
        """Run a prompt through Claude Code CLI and return all stream events.

        Args:
            prompt: The user message to send.
            model: Model identifier (e.g. "sonnet", "opus").
            append_system_prompt: Optional text appended to the system prompt.
            session_id: Optional session ID to resume (hex + hyphens only).
            timeout: Maximum seconds to wait before killing the process.

        Returns:
            List of StreamEvent objects collected from the CLI output.
        """
        args = self._build_args(model, append_system_prompt, session_id)
        env = self._build_env()
        cwd = self.working_dir or os.getcwd()

        logger.info(
            "Starting Claude CLI: %s (cwd=%s)",
            " ".join(args[:6]) + " ...",
            cwd,
        )

        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                limit=10 * 1024 * 1024,
            )
        except Exception:
            logger.error("Failed to spawn Claude CLI process", exc_info=True)
            self._cleanup_tempfile()
            return [StreamEvent(
                raw={},
                message_type=MessageType.RESULT,
                is_complete=True,
                error="Failed to spawn Claude CLI process",
            )]

        logger.info("Claude CLI started: pid=%s", self._process.pid)

        # Send the prompt via stdin as stream-json.
        await self._send_prompt(prompt)

        events: list[StreamEvent] = []
        try:
            events = await asyncio.wait_for(
                self._read_all_events(),
                timeout=timeout,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("Claude CLI timed out after %ds", timeout)
            events.append(StreamEvent(
                raw={},
                message_type=MessageType.RESULT,
                is_complete=True,
                error=f"Timed out after {timeout} seconds",
            ))
        finally:
            await self._cleanup()

        return events

    async def interrupt(self) -> None:
        """Send SIGINT (or terminate on Windows) for a graceful stop."""
        if self._process is None or self._process.returncode is not None:
            return

        if sys.platform == "win32":
            self._process.terminate()
        else:
            self._process.send_signal(signal.SIGINT)

        try:
            await asyncio.wait_for(self._process.wait(), timeout=10)
        except (TimeoutError, asyncio.TimeoutError):
            await self.kill()

    async def kill(self) -> None:
        """Terminate the subprocess, force-killing if it doesn't stop."""
        if self._process is None or self._process.returncode is not None:
            return

        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5)
        except (TimeoutError, asyncio.TimeoutError):
            self._process.kill()
            await self._process.wait()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_args(
        self,
        model: str,
        append_system_prompt: str | None,
        session_id: str | None,
    ) -> list[str]:
        """Build the CLI argument list."""
        args = [
            self.command,
            "-p",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--model", model,
            "--verbose",
            "--dangerously-skip-permissions",
        ]

        # System prompt: use a temp file for large prompts to avoid
        # exceeding OS argument-length limits.
        if append_system_prompt:
            if len(append_system_prompt) > 7000:
                tmpf = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    delete=False,
                    encoding="utf-8",
                )
                tmpf.write(append_system_prompt)
                tmpf.close()
                self._system_prompt_tempfile = tmpf.name
                args.extend(["--append-system-prompt-file", tmpf.name])
            else:
                args.extend(["--append-system-prompt", append_system_prompt])

        if session_id:
            if not re.match(r"^[a-f0-9\-]+$", session_id):
                raise ValueError(f"Invalid session_id format: {session_id!r}")
            args.extend(["--resume", session_id])

        # Windows .cmd resolution: create_subprocess_exec cannot run .cmd
        # files or bare command names directly.  Resolve to [node, cli.js].
        if sys.platform == "win32":
            cmd = args[0]
            cmd_path = Path(cmd)

            # If bare name (e.g. "claude"), resolve via shutil.which and
            # check for the .cmd variant that npm creates.
            if not cmd_path.suffix:
                resolved_path = shutil.which(cmd)
                if resolved_path:
                    cmd_path = Path(resolved_path)
                # Also check for the .cmd sibling explicitly
                cmd_variant = shutil.which(cmd + ".cmd")
                if cmd_variant:
                    cmd_path = Path(cmd_variant)

            if cmd_path.suffix.lower() in (".cmd", ".bat"):
                resolved = _resolve_windows_cmd(cmd_path)
                if resolved:
                    args = resolved + args[1:]
            elif not cmd_path.suffix and cmd_path.exists():
                # POSIX shell script on Windows — can't exec directly.
                # Look for the .cmd sibling in the same directory.
                cmd_sibling = cmd_path.with_suffix(".cmd")
                if cmd_sibling.exists():
                    resolved = _resolve_windows_cmd(cmd_sibling)
                    if resolved:
                        args = resolved + args[1:]

        return args

    def _build_env(self) -> dict[str, str]:
        """Build a clean environment for the subprocess.

        Strips nesting-detection keys and known secrets so the child
        process cannot leak them via tool calls.
        """
        return {
            k: v
            for k, v in os.environ.items()
            if k not in _STRIPPED_ENV_KEYS
        }

    async def _send_prompt(self, prompt: str) -> None:
        """Write the user prompt to stdin in stream-json format."""
        assert self._process is not None and self._process.stdin is not None

        message = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        }
        line = json.dumps(message) + "\n"
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
            logger.debug("Sent stream-json user message")
        except Exception:
            logger.warning("Failed to write prompt to stdin", exc_info=True)

    async def _read_all_events(self) -> list[StreamEvent]:
        """Read stdout line by line, parse events, and collect them."""
        assert self._process is not None and self._process.stdout is not None

        events: list[StreamEvent] = []
        line_count = 0

        while True:
            line = await self._process.stdout.readline()
            if not line:
                logger.info("Claude CLI stdout EOF after %d lines", line_count)
                break

            line_count += 1
            decoded = line.decode("utf-8", errors="replace")

            if line_count <= 3:
                logger.info(
                    "Claude CLI stdout line %d: %.100s",
                    line_count,
                    decoded.strip(),
                )

            event = parse_line(decoded)
            if event:
                events.append(event)
                if event.is_complete:
                    return events

        # Process exited without a RESULT event -- check for errors.
        if self._process.returncode is None:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except (TimeoutError, asyncio.TimeoutError):
                pass

        if self._process.returncode is not None and self._process.returncode > 0:
            stderr_data = b""
            if self._process.stderr:
                stderr_data = await self._process.stderr.read()
            stderr_text = stderr_data.decode("utf-8", errors="replace").strip()
            logger.error(
                "Claude CLI exited with code %d: %s",
                self._process.returncode,
                stderr_text[:500],
            )
            events.append(StreamEvent(
                raw={},
                message_type=MessageType.RESULT,
                is_complete=True,
                error=f"CLI exited with code {self._process.returncode}",
            ))

        return events

    async def _cleanup(self) -> None:
        """Ensure the subprocess is dead and temp files are removed."""
        await self.kill()
        self._cleanup_tempfile()

    def _cleanup_tempfile(self) -> None:
        """Remove the system-prompt temp file if one was created."""
        if self._system_prompt_tempfile:
            try:
                os.unlink(self._system_prompt_tempfile)
            except OSError:
                pass
            self._system_prompt_tempfile = None
