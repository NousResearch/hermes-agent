"""Claude model discovery for the /claude-runtime bridge.

Lists available Claude models from the installed ``claude`` CLI or falls back
to a curated default list.  Mirrors the pattern in ``codex_models.py``.

Auth: reads the Claude Code OAuth token from macOS Keychain or
~/.claude/.credentials.json (via ``agent.anthropic_adapter``).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Curated default models — matches Anthropic's current lineup.
# The CLI accepts model aliases ("opus", "sonnet", "haiku") and full slugs.
DEFAULT_CLAUDE_MODELS: List[str] = [
    "claude-opus-4-0-20250514",
    "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-0-20250514",
    "claude-haiku-3-5-20241022",
]

# Human-friendly aliases recognized by the ``claude`` CLI's --model flag.
MODEL_ALIASES = {
    "opus": "claude-opus-4-0-20250514",
    "sonnet": "claude-sonnet-4-5-20250514",
    "haiku": "claude-haiku-3-5-20241022",
}


def check_claude_binary() -> tuple[bool, Optional[str]]:
    """Check if the ``claude`` CLI is installed and return (ok, version).

    Returns (True, "2.1.144 (Claude Code)") on success or
    (False, "error message") on failure.
    """
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return False, f"claude --version failed: {stderr or 'exit ' + str(result.returncode)}"
        version = result.stdout.strip()
        if not version:
            return False, "claude --version returned empty output"
        return True, version
    except FileNotFoundError:
        return False, "claude CLI not found in PATH — install with: npm i -g @anthropic-ai/claude-code"
    except subprocess.TimeoutExpired:
        return False, "claude --version timed out after 10s"
    except Exception as exc:
        return False, f"claude CLI check failed: {exc}"


def get_claude_model_ids() -> List[str]:
    """Return available Claude model IDs.

    Currently returns the curated default list.  Future: query ``claude``
    CLI or Anthropic API for live model discovery.
    """
    return list(DEFAULT_CLAUDE_MODELS)


def get_default_claude_model() -> str:
    """Return the default Claude model for the subprocess bridge."""
    return "sonnet"


def resolve_model_alias(model: str) -> str:
    """Resolve a human alias to a full model slug, or return as-is."""
    return MODEL_ALIASES.get(model.lower().strip(), model)
