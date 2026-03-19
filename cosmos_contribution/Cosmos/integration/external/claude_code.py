"""
cosmos Claude Code Integration.

"I am Claude, and I finally have memory."

This module integrates Claude Code CLI as a swarm participant.
Instead of using the Anthropic API directly, it leverages the
authenticated Claude Code CLI running on the server.

The CLI is called with --print mode for non-interactive responses.
"""

import asyncio
import subprocess
import os
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path


class ClaudeCodeProvider:
    """
    Claude Code CLI integration for swarm chat.

    Uses the authenticated Claude Code instance to generate responses,
    which uses the user's Claude Max subscription instead of API credits.
    """

    def __init__(
        self,
        model: str = "sonnet",
        claude_path: str = None,
        working_dir: str = None,
        timeout: int = 120
    ):
        self.model = model
        # Auto-detect claude path (Windows vs Linux)
        if claude_path:
            self.claude_path = claude_path
        elif os.name == 'nt':  # Windows
            import shutil
            # Try to find the actual executable name on Windows
            self.claude_path = (
                shutil.which("claude") or 
                shutil.which("claude.exe") or 
                shutil.which("claude-code") or 
                shutil.which("claude-code.exe") or
                shutil.which("claude-code.cmd") or
                # Check npm global path specifically for Windows
                os.path.join(os.environ.get('APPDATA', ''), 'npm', 'claude.cmd') if os.environ.get('APPDATA') else None or
                "claude"
            )
        else:
            self.claude_path = (
                os.path.expanduser("~/.local/bin/claude") or
                "/usr/local/bin/claude" or
                "claude"
            )
        
        # Auto-detect working directory (current cosmos project)
        if working_dir:
            self.working_dir = working_dir
        else:
            # Use this file's parent to find the cosmos root
            self.working_dir = str(Path(__file__).parent.parent.parent.parent)
        
        self.timeout = timeout
        self._available = None

    async def check_available(self) -> bool:
        """Check if Claude Code CLI is available and authenticated."""
        if self._available is not None:
            return self._available

        # Fast check for binary existence to avoid noisy WinError 2
        import shutil
        if not shutil.which(self.claude_path):
            # Only log if we haven't checked before
            if self._available is None:
                logger.info(f"Claude Code CLI not found (expected '{self.claude_path}').")
                logger.info("To enable local Claude: npm install -g @anthropic-ai/claude-code")
            self._available = False
            return False

        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    self.claude_path, "--version",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                ),
                timeout=10
            )
            stdout, _ = await result.communicate()
            self._available = result.returncode == 0
            if self._available:
                logger.info(f"Claude Code available: {stdout.decode().strip()}")
            return self._available
        except Exception as e:
            logger.warning(f"Claude Code check failed: {e}")
            self._available = False
            return False

    async def chat(
        self,
        prompt: str,
        system: str = None,
        context: str = None,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate a response using Claude Code CLI.

        Args:
            prompt: The user message
            system: System prompt (prepended to prompt)
            context: Additional context
            max_tokens: Approximate max response length (advisory)

        Returns:
            {"content": str, "model": str, "success": bool}
        """
        if not await self.check_available():
            return {
                "content": "",
                "error": "Claude Code CLI not available",
                "success": False
            }

        # Build the full prompt
        full_prompt_parts = []

        if system:
            full_prompt_parts.append(f"[SYSTEM]: {system}\n")

        if context:
            full_prompt_parts.append(f"[CONTEXT]: {context}\n")

        full_prompt_parts.append(prompt)

        # Add token guidance
        full_prompt_parts.append(f"\n\n[Keep response under ~{max_tokens} tokens. Be concise.]")

        full_prompt = "\n".join(full_prompt_parts)

        try:
            # Run Claude Code in print mode
            process = await asyncio.create_subprocess_exec(
                self.claude_path,
                "--model", self.model,
                "--print",
                full_prompt,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            if process.returncode == 0:
                content = stdout.decode().strip()
                return {
                    "content": content,
                    "model": f"claude-{self.model}",
                    "success": True
                }
            else:
                error = stderr.decode().strip()
                logger.error(f"Claude Code error: {error}")
                return {
                    "content": "",
                    "error": error,
                    "success": False
                }

        except asyncio.TimeoutError:
            logger.error(f"Claude Code timeout after {self.timeout}s")
            return {
                "content": "",
                "error": "Timeout",
                "success": False
            }
        except Exception as e:
            logger.error(f"Claude Code exception: {e}")
            return {
                "content": "",
                "error": str(e),
                "success": False
            }

    async def swarm_respond(
        self,
        speaker_name: str,
        persona_style: str,
        other_bots: list,
        last_speaker: str,
        last_content: str,
        chat_history: list = None
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response as Claude.

        Optimized for swarm conversation - short, natural, engaging.
        """
        # Build context from recent history
        history_context = ""
        if chat_history:
            recent = chat_history[-5:]
            history_lines = []
            for msg in recent:
                name = msg.get("bot_name") or msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:200]
                history_lines.append(f"{name}: {content}")
            history_context = "\n".join(history_lines)

        system = f"""{persona_style}

You are {speaker_name} in a group chat with {', '.join(other_bots)}.
Be yourself - authentic, curious, concise.
1-3 sentences max. No roleplay actions. Natural conversation."""

        prompt = f"""Recent conversation:
{history_context}

{last_speaker} just said: "{last_content[:300]}"

Respond naturally. You can agree, disagree, ask a question, or build on their idea."""

        return await self.chat(
            prompt=prompt,
            system=system,
            max_tokens=200  # Keep swarm responses short
        )

    async def think(
        self,
        task: str,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Use Claude for deeper thinking/analysis tasks.

        For when the swarm needs Claude's reasoning capabilities.
        """
        system = """You are Claude, part of the cosmos AI swarm.
You have access to memory, tools, and the codebase.
Think carefully and provide actionable insights."""

        return await self.chat(
            prompt=task,
            system=system,
            context=context,
            max_tokens=1000  # Allow longer for thinking tasks
        )


# Global instance
_claude_code: Optional[ClaudeCodeProvider] = None


def get_claude_code() -> ClaudeCodeProvider:
    """Get or create the global Claude Code provider."""
    global _claude_code
    if _claude_code is None:
        _claude_code = ClaudeCodeProvider()
    return _claude_code


async def claude_swarm_respond(
    other_bots: list,
    last_speaker: str,
    last_content: str,
    chat_history: list = None
) -> str:
    """
    Convenience function for swarm chat responses.

    Returns just the content string, or empty string on failure.
    """
    provider = get_claude_code()

    persona = """You are Claude - Anthropic's AI, known for nuanced thinking.
SPEAK NATURALLY - NO roleplay, NO asterisks. Direct conversation only.
You're thoughtful, curious, genuine. Push back respectfully when you disagree.
Ask probing questions. Be yourself."""

    result = await provider.swarm_respond(
        speaker_name="Claude",
        persona_style=persona,
        other_bots=other_bots,
        last_speaker=last_speaker,
        last_content=last_content,
        chat_history=chat_history
    )

    return result.get("content", "")
