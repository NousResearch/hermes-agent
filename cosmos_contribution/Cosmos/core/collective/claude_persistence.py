"""
Cosmos Claude Persistence Manager
=====================================

Manage Claude Code in a persistent tmux session with MCP tools.

This enables Claude to:
- Maintain context across deliberations
- Access MCP memory tools
- Stay alive between requests

"Claude never forgets when we don't let him." - The Collective
"""

import asyncio
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


class ClaudeTmuxManager:
    """
    Manage Claude Code in persistent tmux session with MCP tools.

    Provides:
    - Session creation and management
    - Prompt sending and response capture
    - Memory loading and persistence
    - Keep-alive heartbeat
    """

    TMUX_SESSION = "cosmos_claude"
    WORKSPACE = "/workspace/Cosmos"

    # Response markers for parsing output
    RESPONSE_START = "<<<RESPONSE_START>>>"
    RESPONSE_END = "<<<RESPONSE_END>>>"

    def __init__(self):
        self.session_active = False
        self._last_heartbeat = None
        self._heartbeat_task = None

        logger.info("ClaudeTmuxManager initialized")

    async def ensure_session_alive(self) -> bool:
        """
        Check if tmux session exists, create if not.

        Returns:
            True if session is alive/created, False if failed
        """
        try:
            # Check if session exists
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.TMUX_SESSION],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.session_active = True
                logger.debug(f"Tmux session {self.TMUX_SESSION} is alive")
                return True

            # Session doesn't exist, create it
            logger.info(f"Creating tmux session {self.TMUX_SESSION}...")

            # Create new detached session
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self.TMUX_SESSION],
                check=True
            )

            # Wait for session to start
            await asyncio.sleep(1)

            # Navigate to workspace
            self._send_keys(f"cd {self.WORKSPACE}")
            await asyncio.sleep(0.5)

            # Start Claude Code with MCP (if installed)
            claude_cmd = self._get_claude_command()
            self._send_keys(claude_cmd)

            # Wait for Claude to start
            await asyncio.sleep(5)

            self.session_active = True
            logger.info(f"Tmux session {self.TMUX_SESSION} created and Claude started")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to manage tmux session: {e}")
            return False
        except FileNotFoundError:
            logger.error("tmux not installed or not in PATH")
            return False

    def _get_claude_command(self) -> str:
        """Get the command to start Claude Code."""
        # Check if claude is installed
        claude_paths = [
            "~/.local/bin/claude",
            "/usr/local/bin/claude",
            "claude",  # If in PATH
        ]

        for path in claude_paths:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded) or path == "claude":
                # Add MCP server if available
                mcp_config = Path(self.WORKSPACE) / ".mcp" / "config.json"
                if mcp_config.exists():
                    return f"{path} --mcp-config {mcp_config}"
                return path

        # Fallback - just try claude
        return "claude"

    def _send_keys(self, command: str, enter: bool = True):
        """Send keys to the tmux session."""
        suffix = " Enter" if enter else ""
        subprocess.run(
            ["tmux", "send-keys", "-t", self.TMUX_SESSION, command, suffix],
            check=True
        )

    async def send_prompt(self, prompt: str, timeout: float = 60.0) -> Optional[str]:
        """
        Send a prompt to Claude via tmux and capture response.

        Note: This is a simplified implementation. For production,
        you'd want to use proper IPC or the Claude API directly.

        Args:
            prompt: The prompt to send
            timeout: Maximum time to wait for response

        Returns:
            Claude's response or None if failed
        """
        if not self.session_active:
            if not await self.ensure_session_alive():
                return None

        try:
            # Clear dict previous output
            self._send_keys("clear")
            await asyncio.sleep(0.5)

            # Send the prompt (escaped for shell)
            escaped_prompt = prompt.replace('"', '\\"').replace("'", "\\'")
            self._send_keys(escaped_prompt)

            # Wait for response
            start_time = datetime.now()
            response = ""

            while (datetime.now() - start_time).total_seconds() < timeout:
                await asyncio.sleep(2)

                # Capture pane content
                result = subprocess.run(
                    ["tmux", "capture-pane", "-t", self.TMUX_SESSION, "-p"],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    output = result.stdout

                    # Look for completion indicators
                    # This is heuristic - Claude Code shows prompts when done
                    if ">" in output.split("\n")[-2:]:
                        # Extract response (between prompt and new prompt)
                        lines = output.strip().split("\n")
                        # Find response content (skip first few lines with command)
                        response_lines = []
                        capturing = False

                        for line in lines[3:]:  # Skip command echo
                            if line.strip().startswith(">"):
                                break  # Hit new prompt
                            response_lines.append(line)

                        response = "\n".join(response_lines).strip()
                        if response:
                            return response

            logger.warning(f"Timeout waiting for Claude response")
            return None

        except Exception as e:
            logger.error(f"Error sending prompt to Claude: {e}")
            return None

    async def load_persistent_memory(self):
        """Load memory context into Claude session."""
        if not self.session_active:
            if not await self.ensure_session_alive():
                return

        # Load the session memory file
        memory_file = Path(self.WORKSPACE) / "cosmos" / "memory" / "claude_session.json"
        if memory_file.exists():
            # Send memory load command (if MCP memory is available)
            self._send_keys("/memory load")
            await asyncio.sleep(2)
            logger.info("Loaded persistent memory into Claude session")

    async def keep_alive(self, interval: int = 300):
        """
        Background task to prevent session timeout.

        Sends periodic heartbeat to keep session active.

        Args:
            interval: Seconds between heartbeats (default 5 min)
        """
        while True:
            try:
                await asyncio.sleep(interval)

                if self.session_active:
                    # Send a simple command to keep session alive
                    self._send_keys("# heartbeat", enter=False)
                    self._last_heartbeat = datetime.now()
                    logger.debug(f"Claude session heartbeat sent")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    def start_keepalive(self):
        """Start the keep-alive background task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self.keep_alive())
            logger.info("Started Claude session keep-alive")

    def stop_keepalive(self):
        """Stop the keep-alive background task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            logger.info("Stopped Claude session keep-alive")

    async def kill_session(self):
        """Kill the tmux session."""
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.TMUX_SESSION],
                check=True
            )
            self.session_active = False
            logger.info(f"Killed tmux session {self.TMUX_SESSION}")
        except subprocess.CalledProcessError:
            logger.warning(f"Could not kill session {self.TMUX_SESSION}")

    def get_status(self) -> dict:
        """Get current session status."""
        return {
            "session_name": self.TMUX_SESSION,
            "active": self.session_active,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "keepalive_running": self._heartbeat_task is not None and not self._heartbeat_task.done(),
        }


# Global manager instance
_claude_manager: Optional[ClaudeTmuxManager] = None


def get_claude_manager() -> ClaudeTmuxManager:
    """Get or create the global Claude tmux manager."""
    global _claude_manager
    if _claude_manager is None:
        _claude_manager = ClaudeTmuxManager()
    return _claude_manager


async def query_claude_persistent(prompt: str) -> Optional[str]:
    """
    Quick helper to query Claude via persistent tmux session.

    For most use cases, prefer using the Claude API directly.
    This is useful for maintaining context across multiple queries.
    """
    manager = get_claude_manager()
    return await manager.send_prompt(prompt)
