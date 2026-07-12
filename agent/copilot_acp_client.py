"""GitHub Copilot ACP client — thin subclass of the generic ACP client.

The protocol implementation lives in :mod:`agent.acp_client` (issue #5257
generalized it to all ACP-compatible agents). This module keeps the
Copilot-specific pieces — the deprecated ``gh copilot`` extension detection
and the legacy ``HERMES_COPILOT_ACP_COMMAND`` / ``COPILOT_CLI_PATH`` env
contract — plus backwards-compatible re-exports for existing imports.
"""

from __future__ import annotations

import os
import shlex

# Backwards-compatible re-exports: these helpers moved to agent.acp_client.
from agent.acp_client import (  # noqa: F401
    ACPClient,
    _build_openai_tool_call,
    _build_subprocess_env,
    _completion_to_stream_chunks,
    _ensure_path_within_cwd,
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
    _jsonrpc_error,
    _permission_denied,
    _render_message_content,
    _resolve_home_dir,
)

ACP_MARKER_BASE_URL = "acp://copilot"

# Stderr fingerprint of the deprecated `gh copilot` CLI extension
# (https://github.blog/changelog/2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension).
# We require BOTH the literal product name ("gh-copilot") AND a deprecation
# marker, so generic stderr from the NEW `@github/copilot` CLI — whose repo
# is github.com/github/copilot-cli and which legitimately mentions "copilot-cli"
# in its own banners and error messages — doesn't get misclassified as the
# deprecated extension.
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""

    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


class CopilotACPClient(ACPClient):
    """Minimal OpenAI-client-compatible facade for Copilot ACP."""

    def __init__(self, **kwargs):
        kwargs.setdefault("api_key", "copilot-acp")
        kwargs.setdefault("base_url", ACP_MARKER_BASE_URL)
        super().__init__(agent_name="copilot", **kwargs)

    def _early_exit_error(self, stderr_text: str) -> str | None:
        if _is_gh_copilot_deprecation_message(stderr_text):
            return (
                "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                "(github.com/github/copilot-cli), but the binary it just "
                "spawned is the deprecated `gh copilot` extension.\n\n"
                "Install the new CLI:\n"
                "  npm install -g @github/copilot\n"
                "  # then verify with: copilot --help\n\n"
                "If `copilot` already resolves to the new CLI but you still see this,\n"
                "point Hermes at it explicitly:\n"
                "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                f"Original error:\n{stderr_text}"
            )
        return None
