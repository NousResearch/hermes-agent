"""Hermes plugin bridge for FreeLLMAPI free-tier routing."""

from __future__ import annotations

from .cli import freellmapi_command, register_cli


def register(ctx) -> None:
    """Register ``hermes freellmapi {setup,doctor,status}``."""
    ctx.register_cli_command(
        name="freellmapi",
        help="Setup and health-check the FreeLLMAPI free-tier router",
        setup_fn=register_cli,
        handler_fn=freellmapi_command,
        description=(
            "Enable the FreeLLMAPI integration plugin, probe the local OpenAI-compatible "
            "proxy at http://127.0.0.1:3001/v1, and wire freellmapi/auto into "
            "fallback_providers. Requires FREELLMAPI_API_KEY in ~/.hermes/.env."
        ),
    )
