"""
Cosmos Claude Code Integration via tmux.

"The thoughtful voice of the collective - using Claude Code CLI in persistent tmux session."

Claude Code excels at:
- Complex code generation with file access
- Multi-file refactoring with MCP tools
- Careful reasoning and safety
- Direct codebase manipulation
- Tool calling and autonomous work

This provider routes to Claude Code running in a tmux session,
NOT the raw Anthropic API. This gives Claude access to:
- Read/Write/Edit tools
- Bash execution
- MCP servers (memory, etc.)
- Full project context
"""

from typing import Optional
from loguru import logger
import asyncio
import subprocess
import os
import re
import tempfile
import time

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus


def _get_dynamic_max_tokens(model_id: str = "claude", task_type: str = "chat") -> int:
    """
    AGI v1.8: Get dynamic max_tokens from centralized limits.
    """
    try:
        from Cosmos.core.dynamic_limits import get_max_tokens
        return get_max_tokens(model_id, task_type)
    except Exception:
        defaults = {"chat": 2000, "thinking": 4000, "quick": 500, "code": 4096}
        return defaults.get(task_type, 2000)


class ClaudeProvider(ExternalProvider):
    """Claude Code via tmux session for complex development tasks."""

    TMUX_SESSION = "cosmos_claude"
    RESPONSE_MARKER = "<<<CLAUDE_RESPONSE_END>>>"
    PROMPT_TIMEOUT = 180  # 3 minutes max for complex tasks

    def __init__(self):
        super().__init__(IntegrationConfig(name="claude"))
        self.session_ready = False

    async def connect(self) -> bool:
        """Check if Claude tmux session exists."""
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.TMUX_SESSION],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                self.session_ready = True
                self.status = ConnectionStatus.CONNECTED
                logger.info(f"Claude: tmux session '{self.TMUX_SESSION}' is active")
                return True
            else:
                logger.warning(f"Claude: tmux session '{self.TMUX_SESSION}' not found")
                self.status = ConnectionStatus.DISCONNECTED
                return False
        except FileNotFoundError:
            logger.error("Claude: tmux not installed")
            self.status = ConnectionStatus.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"Claude: Error checking tmux session: {e}")
            self.status = ConnectionStatus.DISCONNECTED
            return False

    async def ensure_session(self) -> bool:
        """Ensure Claude tmux session exists, create if not."""
        if await self.connect():
            return True

        # Try to create the session
        try:
            logger.info(f"Claude: Creating tmux session '{self.TMUX_SESSION}'...")

            # Create detached session
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self.TMUX_SESSION],
                check=True,
                timeout=10
            )

            # Start Claude Code in the session
            subprocess.run(
                ["tmux", "send-keys", "-t", self.TMUX_SESSION,
                 "cd /workspace/Cosmos && claude", "Enter"],
                check=True,
                timeout=10
            )

            # Wait for Claude to start
            await asyncio.sleep(5)

            self.session_ready = True
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Claude: tmux session created and Claude Code started")
            return True

        except Exception as e:
            logger.error(f"Claude: Failed to create session: {e}")
            return False

    async def complete(
        self,
        prompt: str,
        max_tokens: int = None,  # AGI v1.8: None = dynamic default
        **kwargs
    ) -> Optional[str]:
        """
        Send a prompt to Claude Code via tmux and get response.

        This sends the prompt to the Claude Code CLI running in tmux,
        waits for the response, and returns it.
        """
        # AGI v1.8: Resolve dynamic max_tokens (for documentation/future use)
        if max_tokens is None:
            max_tokens = _get_dynamic_max_tokens("claude", "code")

        if not await self.ensure_session():
            logger.warning("Claude: Session not available")
            return None

        try:
            # Create a temp file to capture output
            output_file = f"/tmp/claude_response_{int(time.time())}.txt"

            # Escape the prompt for shell
            escaped_prompt = prompt.replace("'", "'\\''").replace("\n", " ")

            # Truncate very long prompts
            if len(escaped_prompt) > 8000:
                escaped_prompt = escaped_prompt[:8000] + "... [truncated]"

            # Send prompt to Claude Code
            # Use -p flag for non-interactive prompt if available
            cmd = f"echo '{escaped_prompt}' | claude --print > {output_file} 2>&1"

            subprocess.run(
                ["tmux", "send-keys", "-t", self.TMUX_SESSION, cmd, "Enter"],
                check=True,
                timeout=10
            )

            # Wait for response (poll the output file)
            start_time = time.time()
            last_size = 0
            stable_count = 0

            while time.time() - start_time < self.PROMPT_TIMEOUT:
                await asyncio.sleep(2)

                try:
                    if os.path.exists(output_file):
                        current_size = os.path.getsize(output_file)
                        if current_size > 0:
                            if current_size == last_size:
                                stable_count += 1
                                if stable_count >= 3:  # Output stable for 6 seconds
                                    break
                            else:
                                stable_count = 0
                            last_size = current_size
                except Exception:
                    pass

            # Read the response
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    response = f.read().strip()

                # Clean up
                try:
                    os.remove(output_file)
                except OSError:
                    pass

                if response:
                    logger.info(f"Claude: Got response ({len(response)} chars)")
                    return response

            logger.warning("Claude: No response received")
            return None

        except Exception as e:
            logger.error(f"Claude: Error sending prompt: {e}")
            return None

    async def chat(self, prompt: str, **kwargs) -> Optional[str]:
        """Alias for complete() for compatibility."""
        return await self.complete(prompt, **kwargs)

    async def code_review(
        self,
        code: str,
        language: str = "python",
        focus: str = "bugs,security,performance"
    ) -> Optional[str]:
        """Have Claude Code review code for issues."""
        prompt = f"""Review this {language} code for: {focus}

```{language}
{code}
```

List issues by severity (Critical, Warning, Info). Be concise."""

        return await self.complete(prompt)

    async def generate_code(
        self,
        description: str,
        language: str = "python",
        context: str = None
    ) -> Optional[str]:
        """Generate code using Claude Code."""
        prompt = f"Generate production-ready {language} code for: {description}"
        if context:
            prompt += f"\n\nContext:\n```{language}\n{context[:2000]}\n```"
        prompt += "\n\nOnly output the code, no explanations."

        return await self.complete(prompt)


# Singleton instance
_claude_provider: Optional[ClaudeProvider] = None


def get_claude_provider() -> Optional[ClaudeProvider]:
    """Get or create the Claude provider singleton."""
    global _claude_provider
    if _claude_provider is None:
        _claude_provider = ClaudeProvider()
    return _claude_provider


async def test_claude():
    """Test Claude tmux integration."""
    provider = get_claude_provider()
    if await provider.connect():
        print(f"Claude tmux session is active")
        result = await provider.complete("What is 2+2? Just give the number.")
        if result:
            print(f"Claude response: {result}")
            return True
    print("Claude tmux test failed - session not running")
    print(f"Start it with: tmux new -s {ClaudeProvider.TMUX_SESSION} 'cd /workspace/Cosmos && claude'")
    return False


if __name__ == "__main__":
    asyncio.run(test_claude())
