"""agent.adapters package — Anthropic Messages API adapter submodules.

This package contains the split-out logic from agent/anthropic_adapter.py.
The parent module (agent/anthropic_adapter.py) is a thin façade that re-exports
all public symbols for backwards compatibility.
"""

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
    build_anthropic_client,
    normalize_model_name,
)

from agent.adapters.anthropic_messages import (
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
    build_anthropic_kwargs,
    normalize_anthropic_response,
)
