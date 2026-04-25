"""Anthropic Messages API adapter for Hermes Agent — façade.

All logic moved to agent/adapters/. This file re-exports the public API
for backwards compatibility.
"""

from pathlib import Path

from agent.adapters.base import (
    ADAPTIVE_EFFORT_MAP,
    THINKING_BUDGET,
    _ANTHROPIC_DEFAULT_OUTPUT_LIMIT,
    _ANTHROPIC_OUTPUT_LIMITS,
    _get_anthropic_max_output,
    _supports_adaptive_thinking,
)

from agent.adapters.anthropic_auth import (
    _CLAUDE_CODE_SYSTEM_PREFIX,
    _CLAUDE_CODE_VERSION_FALLBACK,
    _MCP_TOOL_PREFIX,
    _OAUTH_ONLY_BETAS,
    _claude_code_version_cache,
    _is_oauth_token,
    _refresh_oauth_token,
    _write_claude_code_credentials,
    resolve_anthropic_token,
    run_oauth_setup_token,
    read_claude_code_credentials,
    read_claude_managed_key,
    is_claude_code_token_valid,
    refresh_anthropic_oauth_pure,
    read_hermes_oauth_credentials,
    run_hermes_oauth_login_pure,
)

from agent.adapters.anthropic_client import (
    _anthropic_sdk,
    build_anthropic_client,
    normalize_model_name,
)

from agent.adapters.anthropic_messages import (
    _to_plain_data,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    build_anthropic_kwargs,
    normalize_anthropic_response,
)

# Public constants also available directly from this module for convenience
from agent.adapters.base import (
    THINKING_BUDGET as THOUGHT_BUDGET,
)

__all__ = [
    # base
    "THINKING_BUDGET",
    "THOUGHT_BUDGET",
    "ADAPTIVE_EFFORT_MAP",
    "_ANTHROPIC_DEFAULT_OUTPUT_LIMIT",
    "_ANTHROPIC_OUTPUT_LIMITS",
    "_get_anthropic_max_output",
    "_supports_adaptive_thinking",
    "_refresh_oauth_token",
    "_write_claude_code_credentials",
    "_is_oauth_token",
    # auth
    "_CLAUDE_CODE_SYSTEM_PREFIX",
    "_CLAUDE_CODE_VERSION_FALLBACK",
    "_MCP_TOOL_PREFIX",
    "_OAUTH_ONLY_BETAS",
    "_claude_code_version_cache",
    "resolve_anthropic_token",
    "run_oauth_setup_token",
    "read_claude_code_credentials",
    "read_claude_managed_key",
    "is_claude_code_token_valid",
    "refresh_anthropic_oauth_pure",
    "read_hermes_oauth_credentials",
    "run_hermes_oauth_login_pure",
    # client
    "_anthropic_sdk",
    "build_anthropic_client",
    "normalize_model_name",
    # messages
    "_to_plain_data",
    "convert_messages_to_anthropic",
    "convert_tools_to_anthropic",
    "build_anthropic_kwargs",
    "normalize_anthropic_response",
]
