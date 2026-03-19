"""
CLI Bridge Base - Abstract interface for CLI-based AI providers.

"Any CLI that speaks, the swarm can hear." - The Collective

Provides the CLIBridge ABC that wraps AI CLI tools (Claude Code, Gemini CLI, etc.)
into async subprocess-based providers with:
- Semaphore-based concurrency limiting
- Structured JSON output parsing
- Streaming support (stream-json)
- Health checking
- Rate limit detection

Each CLI implementation provides 5 methods:
- get_capabilities() -> what this CLI can do
- build_command() -> the actual shell command
- parse_output() -> extract content from CLI output
- parse_stream_line() -> extract delta from streaming output
- detect_rate_limit() -> check if output indicates rate limiting
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from loguru import logger


# =============================================================================
# CAPABILITY MODEL
# =============================================================================

class CLICapability(Enum):
    """Capabilities that a CLI tool may provide."""
    WEB_SEARCH = auto()
    CODE_EDIT = auto()
    FILE_READ = auto()
    FILE_WRITE = auto()
    LONG_CONTEXT = auto()
    STREAMING = auto()
    SESSION_RESUME = auto()
    STRUCTURED_OUTPUT = auto()
    TOOL_USE = auto()
    CODE_EXECUTION = auto()
    IMAGE_UNDERSTANDING = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CLIHealth:
    """Health status for a CLI bridge."""
    available: bool = False
    auth_valid: bool = False
    rate_limited: bool = False
    consecutive_failures: int = 0
    daily_request_count: int = 0
    daily_limit: Optional[int] = None
    last_check: Optional[datetime] = None
    version: str = ""
    error: str = ""


@dataclass
class CLIResponse:
    """Response from a CLI query."""
    content: str = ""
    model: str = ""
    success: bool = False
    session_id: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str = ""
    raw_output: str = ""
    cli_name: str = ""


# =============================================================================
# CLI BRIDGE ABC
# =============================================================================

class CLIBridge(ABC):
    """
    Abstract base for CLI-based AI providers.

    Handles subprocess lifecycle, concurrency limits, and output parsing.
    Each CLI tool (Claude Code, Gemini, Codex, etc.) implements the
    5 abstract methods to customize behavior.
    """

    def __init__(
        self,
        cli_name: str,
        executable: str,
        max_concurrent: int = 3,
        default_timeout: int = 120,
        working_dir: Optional[str] = None,
    ):
        self.cli_name = cli_name
        self.executable = executable
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.working_dir = working_dir or "/workspace/Cosmos"
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.health = CLIHealth()

    # -------------------------------------------------------------------------
    # Abstract methods — each CLI implements these
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_capabilities(self) -> Set[CLICapability]:
        """Return the set of capabilities this CLI provides."""
        ...

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        streaming: bool = False,
    ) -> List[str]:
        """Build the CLI command as a list of args."""
        ...

    @abstractmethod
    def parse_output(self, raw: str) -> CLIResponse:
        """Parse the CLI's stdout into a CLIResponse."""
        ...

    @abstractmethod
    def parse_stream_line(self, line: str) -> Optional[str]:
        """Parse a single streaming output line, return content delta or None."""
        ...

    @abstractmethod
    def detect_rate_limit(self, raw: str, returncode: int) -> bool:
        """Return True if the output indicates rate limiting."""
        ...

    # -------------------------------------------------------------------------
    # Concrete methods — shared across all CLI bridges
    # -------------------------------------------------------------------------

    async def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> CLIResponse:
        """
        Query the CLI tool and return a structured response.

        Uses semaphore to limit concurrent subprocess spawns.
        """
        timeout = timeout or self.default_timeout
        start = time.time()

        async with self._semaphore:
            cmd = self.build_command(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                session_id=session_id,
                streaming=False,
            )

            logger.debug(f"[{self.cli_name}] Executing: {' '.join(cmd[:4])}...")

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_dir,
                )

                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                raw = stdout_bytes.decode("utf-8", errors="replace").strip()
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                latency = (time.time() - start) * 1000

                # Check for rate limiting
                if self.detect_rate_limit(raw or stderr_text, process.returncode):
                    self.health.rate_limited = True
                    logger.warning(f"[{self.cli_name}] Rate limited")
                    return CLIResponse(
                        error="Rate limited",
                        cli_name=self.cli_name,
                        latency_ms=latency,
                    )

                if process.returncode == 0 and raw:
                    response = self.parse_output(raw)
                    response.latency_ms = latency
                    response.cli_name = self.cli_name
                    response.raw_output = raw
                    self.health.consecutive_failures = 0
                    self.health.rate_limited = False
                    return response
                else:
                    error_msg = stderr_text or f"Exit code {process.returncode}"
                    self.health.consecutive_failures += 1
                    logger.error(f"[{self.cli_name}] Error: {error_msg[:200]}")
                    return CLIResponse(
                        error=error_msg,
                        cli_name=self.cli_name,
                        latency_ms=latency,
                        raw_output=raw,
                    )

            except asyncio.TimeoutError:
                latency = (time.time() - start) * 1000
                self.health.consecutive_failures += 1
                logger.error(f"[{self.cli_name}] Timeout after {timeout}s")
                return CLIResponse(
                    error=f"Timeout after {timeout}s",
                    cli_name=self.cli_name,
                    latency_ms=latency,
                )
            except FileNotFoundError:
                self.health.available = False
                return CLIResponse(
                    error=f"CLI not found: {self.executable}",
                    cli_name=self.cli_name,
                )
            except Exception as e:
                self.health.consecutive_failures += 1
                logger.error(f"[{self.cli_name}] Exception: {e}")
                return CLIResponse(
                    error=str(e),
                    cli_name=self.cli_name,
                )

    async def query_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream responses from the CLI tool.

        Yields content deltas as they arrive from the subprocess stdout.
        """
        timeout = timeout or self.default_timeout

        async with self._semaphore:
            cmd = self.build_command(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                session_id=session_id,
                streaming=True,
            )

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_dir,
                )

                deadline = time.time() + timeout

                async for line_bytes in process.stdout:
                    if time.time() > deadline:
                        process.kill()
                        break

                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    delta = self.parse_stream_line(line)
                    if delta is not None:
                        yield delta

                await process.wait()

            except Exception as e:
                logger.error(f"[{self.cli_name}] Streaming error: {e}")

    async def check_health(self) -> CLIHealth:
        """
        Check if the CLI is installed, authenticated, and healthy.

        Runs `<cli> --version` as a quick availability check.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self.executable, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=15
            )

            if process.returncode == 0:
                self.health.available = True
                self.health.auth_valid = True
                self.health.version = stdout.decode().strip()
                self.health.last_check = datetime.now()
                logger.info(f"[{self.cli_name}] Health OK: {self.health.version}")
            else:
                self.health.available = False
                self.health.error = f"Exit code {process.returncode}"

        except FileNotFoundError:
            self.health.available = False
            self.health.error = f"CLI not found: {self.executable}"
        except asyncio.TimeoutError:
            self.health.available = False
            self.health.error = "Health check timeout"
        except Exception as e:
            self.health.available = False
            self.health.error = str(e)

        return self.health

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status as a dict for API responses."""
        return {
            "cli_name": self.cli_name,
            "executable": self.executable,
            "available": self.health.available,
            "auth_valid": self.health.auth_valid,
            "rate_limited": self.health.rate_limited,
            "consecutive_failures": self.health.consecutive_failures,
            "daily_requests": self.health.daily_request_count,
            "daily_limit": self.health.daily_limit,
            "version": self.health.version,
            "capabilities": [c.name for c in self.get_capabilities()],
            "max_concurrent": self.max_concurrent,
        }
