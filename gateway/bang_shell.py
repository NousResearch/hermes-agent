"""``!<command>`` shell mode for the messaging gateway (Telegram / other chats).

``!git status`` typed in any chat runs directly in the gateway process. The agent is
never invoked — no user/assistant message or tool result enters history — so a bang
command costs zero tokens and cannot perturb role alternation or the prompt cache.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import suppress
from typing import Optional

USAGE_HINT = "Usage: !<command> — run a shell command without spending a model turn (e.g. !git status)"

# The gateway process holds every API key; keep a ceiling well under the terminal tool's
# foreground cap so a stray `!sleep 999` cannot wedge the gateway.
DEFAULT_TIMEOUT = 30


def is_bang_command(text: Optional[str]) -> bool:
    """True when *text* is a ``!`` shell-mode submission.

    Only a leading ``!`` (after surrounding whitespace) counts; ``fix the bug!`` is an
    ordinary prompt and must reach the agent untouched.
    """
    return isinstance(text, str) and text.strip().startswith("!")


def parse_bang_command(text: str) -> str:
    """The shell command inside a bang submission (``""`` when bare).

    ``!  ls -la`` -> ``ls -la``; ``!!`` -> ``!`` — a literal second bang belongs to the
    user's shell (history expansion), not to Hermes.
    """
    return text.strip()[1:].strip() if is_bang_command(text) else ""


async def run_bang_command(command: str, *, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute *command* and return the merged stdout/stderr as a string.

    Output is returned to the caller (Telegram reply, etc.); the gateway process's env is
    sanitized so provider API keys never reach the child. Redacts sensitive text in output.
    """
    try:
        from tools.environments.local import build_subprocess_env
        from agent.redact import redact_sensitive_text
    except Exception:
        return "!: environment not available — cannot run shell command"

    if not command.strip():
        return USAGE_HINT

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=build_subprocess_env(),
        )
    except Exception as exc:
        return f"!: failed to run command: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"!: command timed out after {timeout}s"
    except Exception as exc:
        proc.kill()
        with suppress(Exception):
            await proc.wait()
        return f"!: command interrupted: {exc}"

    output = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    combined = output + err
    if combined:
        combined = redact_sensitive_text(combined)
    if proc.returncode != 0 and not combined.strip():
        return f"!: command exited {proc.returncode}"
    if proc.returncode != 0 and combined.strip():
        return combined.rstrip()
    return combined.rstrip() or "Command returned no output."
