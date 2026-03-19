"""
Cosmos tmux Session Manager.

"I've created immortal robot sessions! They'll outlive us all!"

This module manages persistent tmux sessions for long-running agents like Claude Code,
enabling detachable/re-attachable development workflows that survive connection drops.

Features:
- Automatic session creation and cleanup
- Session pooling for rapid agent deployment
- Output capture and streaming
- Health monitoring and recovery
- Integration with handler benchmark system

Usage:
    manager = TmuxSessionManager()
    session = await manager.create_session("claude_dev", "claude code")
    output = await manager.send_command(session.session_id, "echo hello")
    await manager.destroy_session(session.session_id)
"""

import asyncio
import subprocess
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import   Optional, Callable
from enum import Enum
import shutil

from loguru import logger


# =============================================================================
# SESSION TYPES AND STATES
# =============================================================================

class SessionState(Enum):
    """States of a tmux session."""
    CREATING = "creating"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    DESTROYED = "destroyed"


class SessionType(Enum):
    """Types of persistent sessions."""
    CLAUDE_CODE = "claude_code"      # Claude Code CLI
    DEVELOPMENT = "development"       # General dev environment
    RESEARCH = "research"            # Long-running research
    TRADING = "trading"              # Trading bot sessions
    CUSTOM = "custom"                # User-defined


@dataclass
class SessionConfig:
    """Configuration for a tmux session."""
    session_type: SessionType
    command: str                          # Initial command to run
    working_dir: str = "/workspace"       # Working directory
    env_vars: dict[str, str] = field(default_factory=dict)
    timeout_idle_minutes: int = 60        # Auto-destroy after idle
    capture_output: bool = True           # Capture output to buffer
    max_output_lines: int = 1000          # Max lines to keep in buffer


@dataclass
class TmuxSession:
    """Represents a managed tmux session."""
    session_id: str
    session_name: str
    session_type: SessionType
    config: SessionConfig

    # State tracking
    state: SessionState = SessionState.CREATING
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Output capture
    output_buffer: list[str] = field(default_factory=list)
    error_buffer: list[str] = field(default_factory=list)

    # Metrics
    commands_executed: int = 0
    total_output_lines: int = 0

    # Association
    handler_id: Optional[str] = None  # Linked benchmark handler
    agent_id: Optional[str] = None    # Linked agent

    def is_active(self) -> bool:
        return self.state in [SessionState.ACTIVE, SessionState.BUSY]

    def is_idle(self) -> bool:
        return self.state == SessionState.IDLE

    def idle_time_seconds(self) -> float:
        return (datetime.now() - self.last_activity).total_seconds()


# =============================================================================
# TMUX SESSION MANAGER
# =============================================================================

class TmuxSessionManager:
    """
    Manages persistent tmux sessions for long-running agent operations.

    Features:
    - Session lifecycle management (create, destroy, pool)
    - Command execution with output capture
    - Health monitoring and auto-recovery
    - Integration with handler benchmark system
    """

    def __init__(
        self,
        max_sessions: int = 10,
        idle_timeout_minutes: int = 60,
        output_buffer_lines: int = 1000,
    ):
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout_minutes
        self.output_buffer_size = output_buffer_lines

        # Session storage
        self._sessions: dict[str, TmuxSession] = {}
        self._session_by_type: dict[SessionType[str]] = {t: [] for t in SessionType}

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Check tmux availability
        self._tmux_available = self._check_tmux()

        if self._tmux_available:
            logger.info("TmuxSessionManager initialized (tmux available)")
        else:
            logger.warning("TmuxSessionManager: tmux not available, sessions will be simulated")

    def _check_tmux(self) -> bool:
        """Check if tmux is available on the system."""
        return shutil.which("tmux") is not None

    async def start(self):
        """Start the session manager and cleanup loop."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TmuxSessionManager started")

    async def stop(self):
        """Stop the session manager and cleanup all sessions."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Destroy all sessions
        for session_id in list(self._sessions.keys()):
            await self.destroy_session(session_id)

        logger.info("TmuxSessionManager stopped")

    async def create_session(
        self,
        name: str,
        session_type: SessionType,
        command: Optional[str] = None,
        working_dir: str = "/workspace",
        env_vars: Optional[dict[str, str]] = None,
        handler_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> TmuxSession:
        """
        Create a new tmux session.

        Args:
            name: Human-readable session name
            session_type: Type of session
            command: Initial command to run (or default for type)
            working_dir: Working directory
            env_vars: Environment variables
            handler_id: Associated benchmark handler
            agent_id: Associated agent

        Returns:
            TmuxSession object
        """
        async with self._lock:
            if len(self._sessions) >= self.max_sessions:
                # Try to cleanup idle sessions
                await self._cleanup_idle_sessions()

                if len(self._sessions) >= self.max_sessions:
                    raise RuntimeError(f"Maximum sessions ({self.max_sessions}) reached")

        # Generate session ID
        session_id = f"{name}_{uuid.uuid4().hex[:8]}"
        session_name = f"cosmos_{session_id}"

        # Get default command for type
        if command is None:
            command = self._get_default_command(session_type)

        config = SessionConfig(
            session_type=session_type,
            command=command,
            working_dir=working_dir,
            env_vars=env_vars or {},
            timeout_idle_minutes=self.idle_timeout,
            max_output_lines=self.output_buffer_size,
        )

        session = TmuxSession(
            session_id=session_id,
            session_name=session_name,
            session_type=session_type,
            config=config,
            handler_id=handler_id,
            agent_id=agent_id,
        )

        # Create the actual tmux session
        if self._tmux_available:
            success = await self._create_tmux_session(session)
            if not success:
                session.state = SessionState.ERROR
                raise RuntimeError(f"Failed to create tmux session: {session_name}")

        session.state = SessionState.ACTIVE

        async with self._lock:
            self._sessions[session_id] = session
            self._session_by_type[session_type].append(session_id)

        logger.info(f"Created tmux session: {session_name} (type={session_type.value})")

        return session

    async def _create_tmux_session(self, session: TmuxSession) -> bool:
        """Create the actual tmux session."""
        try:
            # Build tmux command
            cmd_parts = [
                "tmux", "new-session",
                "-d",  # Detached
                "-s", session.session_name,
                "-c", session.config.working_dir,
            ]

            # Add environment variables
            env = os.environ.copy()
            env.update(session.config.env_vars)

            # Create session
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode != 0:
                logger.error(f"tmux create failed: {stderr.decode()}")
                return False

            # Send initial command if specified
            if session.config.command:
                await self._send_keys(session.session_name, session.config.command)

            return True

        except asyncio.TimeoutError:
            logger.error(f"tmux create timeout for {session.session_name}")
            return False

        except Exception as e:
            logger.error(f"tmux create error: {e}")
            return False

    async def _send_keys(self, session_name: str, keys: str, enter: bool = True):
        """Send keys to a tmux session."""
        try:
            cmd = ["tmux", "send-keys", "-t", session_name, keys]
            if enter:
                cmd.extend(["Enter"])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(proc.communicate(), timeout=5.0)

        except Exception as e:
            logger.debug(f"send-keys error: {e}")

    def _get_default_command(self, session_type: SessionType) -> str:
        """Get default command for session type."""
        defaults = {
            SessionType.CLAUDE_CODE: "claude --api-key $ANTHROPIC_API_KEY",
            SessionType.DEVELOPMENT: "bash",
            SessionType.RESEARCH: "python -c 'print(\"Research session ready\")'",
            SessionType.TRADING: "cd /workspace/Cosmos && python -c 'print(\"Trading session ready\")'",
            SessionType.CUSTOM: "bash",
        }
        return defaults.get(session_type, "bash")

    async def send_command(
        self,
        session_id: str,
        command: str,
        wait_for_output: bool = True,
        timeout_seconds: float = 30.0,
    ) -> dict:
        """
        Send a command to a session and optionally wait for output.

        Args:
            session_id: Session to send to
            command: Command to execute
            wait_for_output: Whether to wait and capture output
            timeout_seconds: Timeout for waiting

        Returns:
            dict with success, output, error keys
        """
        session = self._sessions.get(session_id)
        if not session:
            return {"success": False, "error": f"Session not found: {session_id}"}

        if not session.is_active():
            return {"success": False, "error": f"Session not active: {session.state.value}"}

        session.state = SessionState.BUSY
        session.last_activity = datetime.now()
        session.commands_executed += 1

        try:
            if self._tmux_available:
                # Send command
                await self._send_keys(session.session_name, command)

                if wait_for_output:
                    # Wait a bit for output
                    await asyncio.sleep(0.5)

                    # Capture pane content
                    output = await self._capture_pane(session.session_name)

                    # Update buffer
                    if output:
                        session.output_buffer.extend(output.split("\n"))
                        session.output_buffer = session.output_buffer[-session.config.max_output_lines:]
                        session.total_output_lines += len(output.split("\n"))

                    return {"success": True, "output": output}

                return {"success": True, "output": ""}

            else:
                # Simulated mode - just execute directly
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=session.config.working_dir,
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds
                )

                output = stdout.decode() if stdout else ""
                error = stderr.decode() if stderr else ""

                return {
                    "success": proc.returncode == 0,
                    "output": output,
                    "error": error,
                }

        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timeout"}

        except Exception as e:
            return {"success": False, "error": str(e)}

        finally:
            session.state = SessionState.ACTIVE

    async def _capture_pane(self, session_name: str, lines: int = 50) -> str:
        """Capture recent output from a tmux pane."""
        try:
            cmd = ["tmux", "capture-pane", "-t", session_name, "-p", "-S", f"-{lines}"]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            return stdout.decode() if stdout else ""

        except Exception as e:
            logger.debug(f"capture-pane error: {e}")
            return ""

    async def get_session_output(
        self,
        session_id: str,
        lines: int = 100,
    ) -> list[str]:
        """Get recent output from a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []

        if self._tmux_available:
            output = await self._capture_pane(session.session_name, lines)
            return output.split("\n") if output else []

        return session.output_buffer[-lines:]

    async def destroy_session(self, session_id: str) -> bool:
        """Destroy a tmux session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        try:
            if self._tmux_available:
                cmd = ["tmux", "kill-session", "-t", session.session_name]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                await asyncio.wait_for(proc.communicate(), timeout=5.0)

        except Exception as e:
            logger.debug(f"kill-session error: {e}")

        session.state = SessionState.DESTROYED

        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

            if session_id in self._session_by_type[session.session_type]:
                self._session_by_type[session.session_type].remove(session_id)

        logger.info(f"Destroyed tmux session: {session.session_name}")

        return True

    async def get_or_create_session(
        self,
        session_type: SessionType,
        handler_id: Optional[str] = None,
    ) -> TmuxSession:
        """
        Get an existing idle session of a type, or create a new one.

        This enables session pooling for rapid agent deployment.
        """
        # Look for idle session of this type
        async with self._lock:
            for session_id in self._session_by_type[session_type]:
                session = self._sessions.get(session_id)
                if session and session.is_idle():
                    session.state = SessionState.ACTIVE
                    session.handler_id = handler_id
                    session.last_activity = datetime.now()
                    logger.debug(f"Reusing idle session: {session.session_name}")
                    return session

        # Create new session
        return await self.create_session(
            name=session_type.value,
            session_type=session_type,
            handler_id=handler_id,
        )

    async def _cleanup_idle_sessions(self):
        """Cleanup sessions that have been idle too long."""
        now = datetime.now()
        to_destroy = []

        for session_id, session in self._sessions.items():
            if session.state == SessionState.DESTROYED:
                to_destroy.append(session_id)
                continue

            idle_minutes = session.idle_time_seconds() / 60

            if idle_minutes > session.config.timeout_idle_minutes:
                logger.info(f"Session {session.session_name} idle for {idle_minutes:.0f}m, destroying")
                to_destroy.append(session_id)

        for session_id in to_destroy:
            await self.destroy_session(session_id)

    async def _cleanup_loop(self):
        """Background loop for session cleanup."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    def list_sessions(self) -> list[dict]:
        """list all managed sessions."""
        return [
            {
                "session_id": s.session_id,
                "session_name": s.session_name,
                "type": s.session_type.value,
                "state": s.state.value,
                "idle_seconds": s.idle_time_seconds(),
                "commands_executed": s.commands_executed,
                "handler_id": s.handler_id,
                "agent_id": s.agent_id,
            }
            for s in self._sessions.values()
        ]

    def get_session(self, session_id: str) -> Optional[TmuxSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_stats(self) -> dict:
        """Get session manager statistics."""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": sum(1 for s in self._sessions.values() if s.is_active()),
            "idle_sessions": sum(1 for s in self._sessions.values() if s.is_idle()),
            "sessions_by_type": {
                t.value: len(ids) for t, ids in self._session_by_type.items()
            },
            "tmux_available": self._tmux_available,
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

session_manager = TmuxSessionManager()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_claude_session(handler_id: str = "claude_tmux") -> TmuxSession:
    """Get or create a Claude Code tmux session."""
    return await session_manager.get_or_create_session(
        SessionType.CLAUDE_CODE,
        handler_id=handler_id,
    )


async def get_development_session(handler_id: str = "dev") -> TmuxSession:
    """Get or create a development tmux session."""
    return await session_manager.get_or_create_session(
        SessionType.DEVELOPMENT,
        handler_id=handler_id,
    )


async def run_in_session(
    session_type: SessionType,
    command: str,
    timeout: float = 30.0,
) -> dict:
    """Convenience function to run a command in a session."""
    session = await session_manager.get_or_create_session(session_type)
    return await session_manager.send_command(
        session.session_id,
        command,
        timeout_seconds=timeout,
    )
