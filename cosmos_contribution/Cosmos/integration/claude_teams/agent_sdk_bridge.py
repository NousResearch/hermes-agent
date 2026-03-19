"""
AGENT SDK BRIDGE - Programmatic Control of Claude Agents
=========================================================

Interfaces with Claude Agent SDK to spawn and control Claude agents.
Supports both Python SDK and CLI-based approaches.

Key Capabilities:
- Spawn Claude agents with custom system prompts
- Send tasks and receive responses
- Manage agent lifecycle (start, stop, resume)
- Model selection (Sonnet, Opus, Haiku)
- Headless operation for server environments
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


class ClaudeModel(Enum):
    """Available Claude models for Agent SDK."""
    SONNET = "sonnet"
    OPUS = "opus"
    HAIKU = "haiku"
    # Specific versions
    SONNET_4 = "claude-sonnet-4-20250514"
    OPUS_4_5 = "claude-opus-4-5-20251101"
    OPUS_4_6 = "claude-opus-4-6-20260205"  # NEW - 1M context, 128k output


class AgentStatus(Enum):
    """Agent lifecycle status."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentSession:
    """Represents an active Claude agent session."""
    session_id: str
    model: ClaudeModel
    system_prompt: Optional[str] = None
    working_dir: str = "."
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, str]] = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    output_file: Optional[str] = None


@dataclass
class AgentResponse:
    """Response from a Claude agent."""
    session_id: str
    content: str
    model: str
    tool_uses: List[Dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0


class AgentSDKBridge:
    """
    Bridge to Claude Agent SDK for programmatic agent control.

    Supports multiple operation modes:
    1. Python SDK (claude-agent-sdk package)
    2. CLI subprocess (claude command)
    3. HTTP API (anthropic API direct)
    """

    def __init__(self, default_model: ClaudeModel = ClaudeModel.SONNET):
        self.default_model = default_model
        self.sessions: Dict[str, AgentSession] = {}
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

        # Check SDK availability
        self._sdk_available = self._check_sdk()
        self._cli_available = self._check_cli()

        logger.info(f"AgentSDKBridge initialized - SDK: {self._sdk_available}, CLI: {self._cli_available}")

    def _check_sdk(self) -> bool:
        """Check if Python SDK is available."""
        try:
            import claude_agent_sdk
            return True
        except ImportError:
            return False

    def _check_cli(self) -> bool:
        """Check if Claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    async def create_session(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[ClaudeModel] = None,
        working_dir: str = ".",
        allowed_tools: Optional[List[str]] = None,
    ) -> AgentSession:
        """Create a new Claude agent session."""
        session_id = f"farn_agent_{uuid.uuid4().hex[:8]}"

        session = AgentSession(
            session_id=session_id,
            model=model or self.default_model,
            system_prompt=system_prompt,
            working_dir=working_dir,
        )

        self.sessions[session_id] = session
        logger.info(f"Created agent session: {session_id} with model {session.model.value}")

        return session

    def is_available(self) -> bool:
        """Check if any communication method is available."""
        return self._sdk_available or self._cli_available or bool(self.api_key)

    async def send_message(
        self,
        session_id: str,
        message: str,
        timeout: float = 120.0,
    ) -> AgentResponse:
        """Send a message to an agent session with full fallback chain."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if not self.is_available():
            raise RuntimeError(
                "No Claude communication method available. "
                "Install claude-agent-sdk, or ensure 'claude' CLI is on PATH, "
                "or set ANTHROPIC_API_KEY environment variable."
            )

        session.status = AgentStatus.RUNNING
        session.messages.append({"role": "user", "content": message})

        last_error = None
        # Try each method in order, falling through on failure
        for method_name, method_available, method_fn in [
            ("SDK", self._sdk_available, self._send_via_sdk),
            ("CLI", self._cli_available, self._send_via_cli),
            ("API", bool(self.api_key), self._send_via_api),
        ]:
            if not method_available:
                continue
            try:
                response = await method_fn(session, message, timeout)
                session.messages.append({"role": "assistant", "content": response.content})
                session.status = AgentStatus.WAITING
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"Agent {method_name} method failed for {session_id}: {e}")
                continue

        session.status = AgentStatus.ERROR
        logger.error(f"All communication methods failed for {session_id}: {last_error}")
        raise RuntimeError(f"All Claude communication methods failed: {last_error}")

    async def close_session(self, session_id: str) -> None:
        """Close an agent session."""
        session = self.sessions.get(session_id)
        if session:
            if session.process:
                session.process.terminate()
            session.status = AgentStatus.COMPLETED
            del self.sessions[session_id]
            logger.info(f"Closed agent session: {session_id}")

    # =========================================================================
    # COMMUNICATION METHODS
    # =========================================================================

    async def _send_via_sdk(
        self,
        session: AgentSession,
        message: str,
        timeout: float,
    ) -> AgentResponse:
        """Send message via Python SDK."""
        try:
            from claude_agent_sdk import Agent, AgentConfig

            config = AgentConfig(
                model=session.model.value,
                system_prompt=session.system_prompt,
                working_directory=session.working_dir,
            )

            agent = Agent(config)
            result = await asyncio.wait_for(
                asyncio.to_thread(agent.send, message),
                timeout=timeout
            )

            return AgentResponse(
                session_id=session.session_id,
                content=result.content,
                model=session.model.value,
                tool_uses=result.tool_uses if hasattr(result, 'tool_uses') else [],
                cost_usd=result.cost if hasattr(result, 'cost') else 0.0,
            )

        except ImportError:
            logger.warning("SDK import failed, falling back to CLI")
            return await self._send_via_cli(session, message, timeout)

    async def _send_via_cli(
        self,
        session: AgentSession,
        message: str,
        timeout: float,
    ) -> AgentResponse:
        """Send message via Claude CLI (headless mode)."""
        try:
            # Build command - Claude CLI uses positional arg for prompt
            cmd = [
                "claude",
                "--print",  # Non-interactive output
                "--output-format", "json",
                "--model", session.model.value,
            ]

            if session.system_prompt:
                cmd.extend(["--append-system-prompt", session.system_prompt])

            # Add message as positional argument (last)
            cmd.append(message)

            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=session.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            # Parse JSON output
            output = stdout.decode('utf-8').strip()
            if output:
                try:
                    result = json.loads(output)
                    return AgentResponse(
                        session_id=session.session_id,
                        content=result.get("result", output),
                        model=session.model.value,
                        cost_usd=result.get("cost_usd", 0.0),
                        tokens_in=result.get("tokens_in", 0),
                        tokens_out=result.get("tokens_out", 0),
                    )
                except json.JSONDecodeError:
                    return AgentResponse(
                        session_id=session.session_id,
                        content=output,
                        model=session.model.value,
                    )

            # If no stdout, check stderr
            if stderr:
                error = stderr.decode('utf-8')
                raise RuntimeError(f"CLI error: {error}")

            return AgentResponse(
                session_id=session.session_id,
                content="No response",
                model=session.model.value,
            )

        except asyncio.TimeoutError:
            raise TimeoutError(f"CLI timeout after {timeout}s")

    async def _send_via_api(
        self,
        session: AgentSession,
        message: str,
        timeout: float,
    ) -> AgentResponse:
        """Send message via direct Anthropic API."""
        try:
            import httpx

            # Build messages
            messages = [{"role": "user", "content": message}]

            # Add conversation history
            for msg in session.messages[-10:]:  # Last 10 messages
                if msg["role"] in ["user", "assistant"]:
                    messages.insert(0, msg)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": session.model.value if session.model in [ClaudeModel.SONNET_4, ClaudeModel.OPUS_4_5] else f"claude-3-5-{session.model.value}-latest",
                        "max_tokens": 4096,
                        "system": session.system_prompt or "You are a helpful AI assistant.",
                        "messages": messages,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", [{}])[0].get("text", "")
                    usage = data.get("usage", {})

                    return AgentResponse(
                        session_id=session.session_id,
                        content=content,
                        model=data.get("model", session.model.value),
                        tokens_in=usage.get("input_tokens", 0),
                        tokens_out=usage.get("output_tokens", 0),
                    )
                else:
                    raise RuntimeError(f"API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Direct API error: {e}")
            raise

    # =========================================================================
    # SUBAGENT SPAWNING
    # =========================================================================

    async def spawn_subagent(
        self,
        task: str,
        model: ClaudeModel = ClaudeModel.HAIKU,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ) -> AgentResponse:
        """
        Spawn a quick subagent for a specific task.
        Uses Haiku by default for speed/cost efficiency.

        Falls back to Cosmos's shadow agents if Claude is unavailable.
        """
        if not self.is_available():
            # Fallback: use Cosmos's shadow agents instead
            logger.info("Claude SDK/CLI/API unavailable, falling back to shadow agents")
            try:
                from Cosmos.core.collective.persistent_agent import call_shadow_agent
                result = await call_shadow_agent("grok", task, timeout=timeout)
                if result:
                    agent_name, response_text = result
                    return AgentResponse(
                        session_id=f"fallback_{agent_name}",
                        content=response_text,
                        model=f"fallback_{agent_name}",
                    )
            except Exception as e:
                logger.warning(f"Shadow agent fallback also failed: {e}")

            return AgentResponse(
                session_id="unavailable",
                content="Claude Teams unavailable and fallback agents failed. Please configure ANTHROPIC_API_KEY or install Claude CLI.",
                model="none",
            )

        session = await self.create_session(
            system_prompt=system_prompt,
            model=model,
        )

        try:
            response = await self.send_message(session.session_id, task, timeout)
            return response
        finally:
            await self.close_session(session.session_id)

    async def spawn_team(
        self,
        team_name: str,
        team_size: int = 3,
        task: str = "",
        model: ClaudeModel = ClaudeModel.SONNET,
    ) -> Dict[str, AgentSession]:
        """
        Spawn a team of Claude agents for collaborative work.

        Note: Full Agent Teams requires CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
        This is a simulation using multiple sessions.
        """
        team = {}

        roles = [
            ("lead", "You are the team lead. Coordinate the team and make final decisions."),
            ("researcher", "You are the researcher. Gather information and analyze data."),
            ("developer", "You are the developer. Write and review code."),
            ("reviewer", "You are the reviewer. Critique and improve the team's work."),
        ]

        for i in range(min(team_size, len(roles))):
            role_name, role_prompt = roles[i]
            session = await self.create_session(
                system_prompt=f"{role_prompt}\n\nTeam: {team_name}\nTask: {task}",
                model=model,
            )
            team[f"{team_name}_{role_name}"] = session

        logger.info(f"Spawned team '{team_name}' with {len(team)} agents")
        return team

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "model": s.model.value,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
                "message_count": len(s.messages),
            }
            for s in self.sessions.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "active_sessions": len(self.sessions),
            "sdk_available": self._sdk_available,
            "cli_available": self._cli_available,
            "api_available": bool(self.api_key),
            "default_model": self.default_model.value,
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_sdk_bridge: Optional[AgentSDKBridge] = None


def get_sdk_bridge() -> AgentSDKBridge:
    """Get global Agent SDK bridge instance."""
    global _sdk_bridge
    if _sdk_bridge is None:
        _sdk_bridge = AgentSDKBridge()
    return _sdk_bridge
