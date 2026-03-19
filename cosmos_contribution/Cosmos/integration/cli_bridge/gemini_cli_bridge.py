"""
Gemini CLI Bridge — Wraps `gemini` CLI into the swarm.

"1,000 free requests/day with Google Search grounding. The Professor is pleased."

Capabilities:
- Web search via Google Search grounding (FREE)
- 1M token context window
- Code editing and execution
- Image understanding
- Streaming output

Spawns: gemini -p --output-format json -m gemini-2.5-pro -y "<prompt>"
"""

import json
import os
import shutil
from typing import List, Optional, Set

from loguru import logger

from .base import CLIBridge, CLICapability, CLIResponse
from .rate_tracker import get_rate_tracker


class GeminiCLIBridge(CLIBridge):
    """
    Gemini CLI bridge.

    Uses Google's Gemini CLI (free tier: 1,000 req/day)
    with Google Search grounding for real-time web data.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        gemini_path: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_concurrent: int = 5,
        default_timeout: int = 120,
        daily_limit: int = 1000,
    ):
        # Resolve gemini executable
        if gemini_path:
            executable = gemini_path
        else:
            executable = shutil.which("gemini") or "gemini"

        super().__init__(
            cli_name="gemini_cli",
            executable=executable,
            max_concurrent=max_concurrent,
            default_timeout=default_timeout,
            working_dir=working_dir or "/workspace/Cosmos",
        )

        self.model = model
        self.health.daily_limit = daily_limit

        # Register with rate tracker
        tracker = get_rate_tracker()
        tracker.register_cli("gemini_cli", daily_limit=daily_limit, minute_limit=20)

    def get_capabilities(self) -> Set[CLICapability]:
        return {
            CLICapability.WEB_SEARCH,
            CLICapability.LONG_CONTEXT,
            CLICapability.CODE_EDIT,
            CLICapability.FILE_READ,
            CLICapability.STREAMING,
            CLICapability.CODE_EXECUTION,
            CLICapability.IMAGE_UNDERSTANDING,
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
        if not streaming:
            cmd.extend(["--output-format", "json"])

        # Model selection
        cmd.extend(["-m", model])

        # Auto-approve (essential for non-interactive mode)
        cmd.append("-y")

        # Build full prompt with system context prepended
        full_prompt = ""
        if system_prompt:
            full_prompt = f"[System]: {system_prompt}\n\n"
        full_prompt += prompt

        if max_tokens:
            full_prompt += f"\n\n[Keep response under ~{max_tokens} tokens.]"

        cmd.append(full_prompt)

        return cmd

    def parse_output(self, raw: str) -> CLIResponse:
        """Parse Gemini CLI JSON output."""
        try:
            data = json.loads(raw)

            # Gemini CLI JSON format: response, stats.models (tokens), error
            content = data.get("response", "")

            if data.get("error"):
                return CLIResponse(
                    content="",
                    error=data["error"],
                    model=self.model,
                )

            # Extract token usage from stats
            stats = data.get("stats", {})
            models_stats = stats.get("models", {})
            tokens = 0
            for model_stats in models_stats.values():
                tokens += model_stats.get("total_tokens", 0)

            return CLIResponse(
                content=content,
                model=self.model,
                success=True,
                tokens_used=tokens,
                cost_usd=0.0,  # Free tier
            )

        except json.JSONDecodeError:
            # Fallback: treat raw output as plain text
            if raw:
                return CLIResponse(
                    content=raw,
                    model=self.model,
                    success=True,
                    cost_usd=0.0,
                )
            return CLIResponse(error="Failed to parse Gemini CLI output")

    def parse_stream_line(self, line: str) -> Optional[str]:
        """Parse a streaming line from Gemini CLI."""
        try:
            data = json.loads(line)

            # Content delta
            if "text" in data:
                return data["text"]
            if "response" in data:
                return data["response"]
            if "content" in data:
                return data["content"]

        except json.JSONDecodeError:
            # Plain text streaming - return the line itself
            if line and not line.startswith("{"):
                return line

        return None

    def detect_rate_limit(self, raw: str, returncode: int) -> bool:
        """Check if Gemini CLI hit a rate limit."""
        rate_signals = [
            "rate limit",
            "quota exceeded",
            "429",
            "resource exhausted",
            "too many requests",
            "daily limit",
        ]
        lower = raw.lower()
        return any(signal in lower for signal in rate_signals)


# Factory function
_gemini_bridge: Optional[GeminiCLIBridge] = None


def get_gemini_cli_bridge(**kwargs) -> GeminiCLIBridge:
    """Get or create the global Gemini CLI bridge."""
    global _gemini_bridge
    if _gemini_bridge is None:
        _gemini_bridge = GeminiCLIBridge(**kwargs)
    return _gemini_bridge
