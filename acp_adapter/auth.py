"""Authentication helpers for ACP adapter.

Checks environment / dotenv for configured API keys, reusing Hermes CLI's
existing provider detection logic.
"""

import os
import sys
from pathlib import Path


def check_auth() -> dict:
    """Check if at least one inference provider is configured.

    Returns dict with:
        success (bool): True if a provider is available.
        error (str | None): Human-readable setup instructions on failure.
    """
    # Ensure project root is importable (entry.py normally handles this,
    # but be defensive).
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from hermes_cli.main import _has_any_provider_configured

        if _has_any_provider_configured():
            return {"success": True, "error": None}
    except Exception:
        # If we can't import the helper, fall back to a direct env-var check.
        env_vars = [
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_BASE_URL",
        ]
        if any(os.getenv(v) for v in env_vars):
            return {"success": True, "error": None}

    return {
        "success": False,
        "error": (
            "No inference provider configured. "
            "Run 'hermes setup' to configure interactively, "
            "or set an API key (e.g. OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY) "
            "in your environment or in ~/.hermes/.env. "
            "See https://github.com/NousResearch/hermes-agent#setup for all supported providers."
        ),
    }
