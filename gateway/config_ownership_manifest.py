"""Static ownership manifest for config surfaces visible to the gateway.

This module is deliberately data-only: it must not read profile config, state,
credentials, sessions, or runtime files.  The intent is to keep config.yaml
surfaces honest about which subsystem owns them before gateway code starts
bridging more CLI config into runtime behavior.
"""

from __future__ import annotations

from typing import Final, Literal

Ownership = Literal[
    "gateway_config",
    "gateway_run_runtime",
    "terminal_bridge",
    "plugin_owned",
    "agent_core_only",
    "cli_only",
    "deprecated",
    "needs_owner",
]

KNOWN_OWNERS: Final[frozenset[Ownership]] = frozenset(
    {
        "gateway_config",
        "gateway_run_runtime",
        "terminal_bridge",
        "plugin_owned",
        "agent_core_only",
        "cli_only",
        "deprecated",
        "needs_owner",
    }
)

# Top-level keys from hermes_cli.config.DEFAULT_CONFIG.  Classifications are
# conservative: keys are gateway-owned only when gateway/config.py currently
# consumes them directly or they describe gateway-specific runtime/session
# behavior.  Platform adapter blocks are plugin-owned even though the gateway
# dispatches their config bridge hooks.
DEFAULT_CONFIG_ROOT_OWNERS: Final[dict[str, Ownership]] = {
    "model": "agent_core_only",
    "providers": "agent_core_only",
    "fallback_providers": "agent_core_only",
    "credential_pool_strategies": "agent_core_only",
    "toolsets": "cli_only",
    "max_concurrent_sessions": "gateway_config",
    "max_live_sessions": "agent_core_only",
    "agent": "agent_core_only",
    "terminal": "terminal_bridge",
    "web": "agent_core_only",
    "browser": "agent_core_only",
    "computer_use": "agent_core_only",
    "checkpoints": "agent_core_only",
    "context_file_max_chars": "agent_core_only",
    "file_read_max_chars": "agent_core_only",
    "mcp_discovery_timeout": "agent_core_only",
    "tool_output": "agent_core_only",
    "tool_loop_guardrails": "agent_core_only",
    "compression": "agent_core_only",
    "kanban": "agent_core_only",
    "moa": "agent_core_only",
    "prompt_caching": "agent_core_only",
    "openrouter": "agent_core_only",
    "bedrock": "agent_core_only",
    "vertex": "agent_core_only",
    "auxiliary": "agent_core_only",
    "display": "agent_core_only",
    "dashboard": "cli_only",
    "desktop": "cli_only",
    "privacy": "agent_core_only",
    "tts": "agent_core_only",
    "stt": "gateway_config",
    "voice": "agent_core_only",
    "human_delay": "agent_core_only",
    "context": "agent_core_only",
    "memory": "agent_core_only",
    "delegation": "agent_core_only",
    "prefill_messages_file": "agent_core_only",
    "goals": "agent_core_only",
    "skills": "agent_core_only",
    "curator": "agent_core_only",
    "honcho": "agent_core_only",
    "timezone": "agent_core_only",
    "slack": "plugin_owned",
    "discord": "plugin_owned",
    "whatsapp": "plugin_owned",
    "telegram": "plugin_owned",
    "mattermost": "plugin_owned",
    "matrix": "plugin_owned",
    "approvals": "agent_core_only",
    "command_allowlist": "agent_core_only",
    "quick_commands": "gateway_config",
    "platform_hints": "gateway_config",
    "hooks": "agent_core_only",
    "hooks_auto_accept": "agent_core_only",
    "personalities": "agent_core_only",
    "security": "agent_core_only",
    "cron": "agent_core_only",
    "code_execution": "agent_core_only",
    "tools": "agent_core_only",
    "logging": "agent_core_only",
    "model_catalog": "agent_core_only",
    "network": "agent_core_only",
    "gateway": "gateway_config",
    "streaming": "gateway_config",
    "sessions": "gateway_config",
    "onboarding": "cli_only",
    "updates": "cli_only",
    "lsp": "cli_only",
    "x_search": "agent_core_only",
    "secrets": "agent_core_only",
    "paste_collapse_threshold": "deprecated",
    "paste_collapse_threshold_fallback": "deprecated",
    "paste_collapse_char_threshold": "deprecated",
    "_config_version": "cli_only",
}

# GatewayConfig dataclass fields and GatewayConfig.to_dict() keys.  Keeping
# these explicit avoids runtime introspection in the manifest itself while tests
# can still compare this list with the actual dataclass/to_dict contract.
GATEWAY_CONFIG_FIELD_OWNERS: Final[dict[str, Ownership]] = {
    "platforms": "gateway_config",
    "default_reset_policy": "gateway_config",
    "reset_by_type": "gateway_config",
    "reset_by_platform": "gateway_config",
    "reset_triggers": "gateway_config",
    "quick_commands": "gateway_config",
    "sessions_dir": "gateway_run_runtime",
    "write_sessions_json": "gateway_config",
    "always_log_local": "gateway_config",
    "filter_silence_narration": "gateway_config",
    "stt_enabled": "gateway_config",
    "stt_echo_transcripts": "gateway_config",
    "group_sessions_per_user": "gateway_config",
    "thread_sessions_per_user": "gateway_config",
    "max_concurrent_sessions": "gateway_config",
    "multiplex_profiles": "gateway_config",
    "unauthorized_dm_behavior": "gateway_config",
    "streaming": "gateway_config",
    "session_store_max_age_days": "gateway_config",
}

# Nested config.yaml surfaces that gateway/config.py currently bridges into the
# gateway schema.  This list is intentionally small and named, not a wildcard
# over config.yaml.
GATEWAY_BRIDGED_CONFIG_PATH_OWNERS: Final[dict[str, Ownership]] = {
    "session_reset": "gateway_config",
    "quick_commands": "gateway_config",
    "stt.enabled": "gateway_config",
    "stt_echo_transcripts": "gateway_config",
    "stt.echo_transcripts": "gateway_config",
    "group_sessions_per_user": "gateway_config",
    "thread_sessions_per_user": "gateway_config",
    "max_concurrent_sessions": "gateway_config",
    "multiplex_profiles": "gateway_config",
    "gateway.max_concurrent_sessions": "gateway_config",
    "gateway.multiplex_profiles": "gateway_config",
    "streaming": "gateway_config",
    "gateway.streaming": "gateway_config",
    "reset_triggers": "gateway_config",
    "always_log_local": "gateway_config",
    "write_sessions_json": "gateway_config",
    "gateway.write_sessions_json": "gateway_config",
    "filter_silence_narration": "gateway_config",
    "unauthorized_dm_behavior": "gateway_config",
    "platforms": "gateway_config",
    "gateway.platforms": "gateway_config",
    "require_mention": "gateway_config",
    "signal.require_mention": "gateway_config",
}

# A needs_owner entry is allowed only with an explicit reason here.  The current
# manifest intentionally has no unresolved owners.
NEEDS_OWNER_ALLOWLIST: Final[dict[str, str]] = {}


def owner_for_default_root(root: str) -> Ownership:
    """Return the owner classification for a DEFAULT_CONFIG top-level key."""

    return DEFAULT_CONFIG_ROOT_OWNERS.get(root, "needs_owner")


def owner_for_gateway_config_field(field_name: str) -> Ownership:
    """Return the owner classification for a GatewayConfig dataclass field."""

    return GATEWAY_CONFIG_FIELD_OWNERS.get(field_name, "needs_owner")


def unresolved_owner_entries() -> dict[str, Ownership]:
    """Return manifest entries still classified as needs_owner."""

    unresolved: dict[str, Ownership] = {}
    for namespace, entries in {
        "default": DEFAULT_CONFIG_ROOT_OWNERS,
        "gateway_field": GATEWAY_CONFIG_FIELD_OWNERS,
        "gateway_bridge": GATEWAY_BRIDGED_CONFIG_PATH_OWNERS,
    }.items():
        for key, owner in entries.items():
            if owner == "needs_owner":
                unresolved[f"{namespace}:{key}"] = owner
    return unresolved
