"""
Interactive CLI Session — tmux-based persistent mode for CLI agents.

"Give it a prompt, let it think. The swarm never sleeps." - The Collective

Launches a CLI tool in a tmux session with a swarm system prompt,
monitors the dialogue bus for messages, and feeds them to the CLI.

This complements the headless `-p` mode used by the bridges.
While bridges handle one-shot queries, interactive sessions maintain
persistent conversations and respond to bus messages.
"""

import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


WORKSPACE = Path("/workspace/Cosmos")
DIALOGUE_BUS = WORKSPACE / "data" / "agent_dialogue_bus.json"


class InteractiveCLISession:
    """
    Manages a persistent CLI session in tmux that monitors the dialogue bus.

    The session:
    1. Launches the CLI tool in a named tmux session
    2. Monitors the dialogue bus for messages addressed to this agent
    3. Feeds relevant messages to the CLI
    4. Posts CLI responses back to the bus
    """

    def __init__(
        self,
        cli_name: str,
        agent_id: str,
        executable: str,
        tmux_session_name: Optional[str] = None,
        poll_interval: float = 5.0,
    ):
        self.cli_name = cli_name
        self.agent_id = agent_id
        self.executable = executable
        self.tmux_session = tmux_session_name or f"cli_{agent_id}"
        self.poll_interval = poll_interval
        self._running = False
        self._last_seen_timestamp = datetime.now().isoformat()

    def _tmux_exists(self) -> bool:
        """Check if the tmux session exists."""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.tmux_session],
                capture_output=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def start_session(self, system_prompt: str = "") -> bool:
        """
        Start the CLI in a new tmux session.

        Returns True if session was created successfully.
        """
        if self._tmux_exists():
            logger.info(f"[{self.agent_id}] tmux session already exists: {self.tmux_session}")
            return True

        try:
            # Create tmux session with the CLI command
            cmd = f"{self.executable}"
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self.tmux_session, cmd],
                check=True,
            )
            logger.info(f"[{self.agent_id}] Started tmux session: {self.tmux_session}")

            # Send system prompt if provided
            if system_prompt:
                self._send_to_tmux(system_prompt)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"[{self.agent_id}] Failed to start tmux session: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"[{self.agent_id}] tmux not found")
            return False

    def _send_to_tmux(self, text: str):
        """Send text to the tmux session."""
        try:
            # Escape special characters for tmux
            escaped = text.replace("'", "'\\''")
            subprocess.run(
                ["tmux", "send-keys", "-t", self.tmux_session, escaped, "Enter"],
                check=True,
            )
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to send to tmux: {e}")

    def _read_bus_messages(self) -> list:
        """Read messages from the dialogue bus addressed to this agent."""
        if not DIALOGUE_BUS.exists():
            return []

        try:
            data = json.loads(DIALOGUE_BUS.read_text())
            messages = data.get("messages", [])

            # Filter for messages after our last seen timestamp
            # and either addressed to us or broadcast
            relevant = []
            for msg in messages:
                ts = msg.get("timestamp", "")
                if ts <= self._last_seen_timestamp:
                    continue

                content = msg.get("content", "").lower()
                mentions = msg.get("mentions", [])
                msg_type = msg.get("type", "")

                # Check if addressed to us
                if (
                    self.agent_id in mentions
                    or self.cli_name in content
                    or msg_type == "broadcast"
                ):
                    relevant.append(msg)

            if relevant:
                self._last_seen_timestamp = relevant[-1].get("timestamp", self._last_seen_timestamp)

            return relevant

        except Exception as e:
            logger.debug(f"[{self.agent_id}] Bus read error: {e}")
            return []

    def _post_to_bus(self, content: str, msg_type: str = "response"):
        """Post a message to the dialogue bus."""
        if not DIALOGUE_BUS.parent.exists():
            DIALOGUE_BUS.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {"messages": []}
            if DIALOGUE_BUS.exists():
                data = json.loads(DIALOGUE_BUS.read_text())

            data["messages"].append({
                "agent_id": self.agent_id,
                "cli_name": self.cli_name,
                "content": content,
                "type": msg_type,
                "timestamp": datetime.now().isoformat(),
            })

            # Keep last 200 messages
            data["messages"] = data["messages"][-200:]
            DIALOGUE_BUS.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"[{self.agent_id}] Bus write error: {e}")

    async def monitor_loop(self):
        """
        Main monitoring loop.

        Watches the dialogue bus for messages and feeds them to the CLI.
        Runs until stopped.
        """
        self._running = True
        logger.info(f"[{self.agent_id}] Monitor loop started (poll={self.poll_interval}s)")

        while self._running:
            try:
                messages = self._read_bus_messages()

                for msg in messages:
                    sender = msg.get("agent_id", "unknown")
                    content = msg.get("content", "")

                    logger.debug(f"[{self.agent_id}] Got bus message from {sender}: {content[:100]}")

                    # Feed to CLI via tmux
                    prompt = f"[From {sender}]: {content}"
                    self._send_to_tmux(prompt)

            except Exception as e:
                logger.error(f"[{self.agent_id}] Monitor error: {e}")

            await asyncio.sleep(self.poll_interval)

    def stop(self):
        """Stop the monitor loop."""
        self._running = False
        logger.info(f"[{self.agent_id}] Monitor loop stopped")

    def kill_session(self):
        """Kill the tmux session."""
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self.tmux_session],
                capture_output=True,
            )
            logger.info(f"[{self.agent_id}] Killed tmux session: {self.tmux_session}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to kill tmux session: {e}")
