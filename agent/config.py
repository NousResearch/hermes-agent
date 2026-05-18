"""Configuration objects for agent construction."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, MutableMapping, Sequence, TypeVar, cast


AgentCallback = Callable[..., Any]
T = TypeVar("T")


@dataclass
class AgentConfig:
    base_url: str | None = None
    api_key: str | None = None
    provider: str | None = None
    api_mode: str | None = None
    acp_command: str | None = None
    acp_args: list[str] | None = None
    command: str | None = None
    args: list[str] | None = None
    model: str = ""
    max_iterations: int = 90
    tool_delay: float = 1.0
    enabled_toolsets: list[str] | None = None
    disabled_toolsets: list[str] | None = None
    save_trajectories: bool = False
    verbose_logging: bool = False
    quiet_mode: bool = False
    ephemeral_system_prompt: str | None = None
    log_prefix_chars: int = 100
    log_prefix: str = ""
    providers_allowed: list[str] | None = None
    providers_ignored: list[str] | None = None
    providers_order: list[str] | None = None
    provider_sort: str | None = None
    provider_require_parameters: bool = False
    provider_data_collection: str | None = None
    openrouter_min_coding_score: float | None = None
    session_id: str | None = None
    max_tokens: int | None = None
    reasoning_config: dict[str, Any] | None = None
    service_tier: str | None = None
    request_overrides: dict[str, Any] | None = None
    prefill_messages: list[dict[str, Any]] | None = None
    skip_context_files: bool = False
    load_soul_identity: bool = False
    skip_memory: bool = False
    session_db: Any | None = None
    parent_session_id: str | None = None
    iteration_budget: Any | None = None
    fallback_model: dict[str, Any] | None = None
    credential_pool: Any | None = None
    pass_session_id: bool = False


@dataclass
class CallbackConfig:
    tool_progress_callback: AgentCallback | None = None
    tool_start_callback: AgentCallback | None = None
    tool_complete_callback: AgentCallback | None = None
    thinking_callback: AgentCallback | None = None
    reasoning_callback: AgentCallback | None = None
    clarify_callback: AgentCallback | None = None
    step_callback: AgentCallback | None = None
    stream_delta_callback: AgentCallback | None = None
    interim_assistant_callback: AgentCallback | None = None
    tool_gen_callback: AgentCallback | None = None
    status_callback: AgentCallback | None = None


@dataclass
class PlatformContext:
    platform: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    chat_id: str | None = None
    chat_name: str | None = None
    chat_type: str | None = None
    thread_id: str | None = None
    gateway_session_key: str | None = None


@dataclass
class CheckpointConfig:
    enabled: bool = False
    max_snapshots: int = 20
    max_total_size_mb: int = 500
    max_file_size_mb: int = 10


LEGACY_INIT_AGENT_PARAMS: tuple[str, ...] = (
    "base_url",
    "api_key",
    "provider",
    "api_mode",
    "acp_command",
    "acp_args",
    "command",
    "args",
    "model",
    "max_iterations",
    "tool_delay",
    "enabled_toolsets",
    "disabled_toolsets",
    "save_trajectories",
    "verbose_logging",
    "quiet_mode",
    "ephemeral_system_prompt",
    "log_prefix_chars",
    "log_prefix",
    "providers_allowed",
    "providers_ignored",
    "providers_order",
    "provider_sort",
    "provider_require_parameters",
    "provider_data_collection",
    "openrouter_min_coding_score",
    "session_id",
    "tool_progress_callback",
    "tool_start_callback",
    "tool_complete_callback",
    "thinking_callback",
    "reasoning_callback",
    "clarify_callback",
    "step_callback",
    "stream_delta_callback",
    "interim_assistant_callback",
    "tool_gen_callback",
    "status_callback",
    "max_tokens",
    "reasoning_config",
    "service_tier",
    "request_overrides",
    "prefill_messages",
    "platform",
    "user_id",
    "user_name",
    "chat_id",
    "chat_name",
    "chat_type",
    "thread_id",
    "gateway_session_key",
    "skip_context_files",
    "load_soul_identity",
    "skip_memory",
    "session_db",
    "parent_session_id",
    "iteration_budget",
    "fallback_model",
    "credential_pool",
    "checkpoints_enabled",
    "checkpoint_max_snapshots",
    "checkpoint_max_total_size_mb",
    "checkpoint_max_file_size_mb",
    "pass_session_id",
)


_CHECKPOINT_LEGACY_NAMES: dict[str, str] = {
    "checkpoints_enabled": "enabled",
    "checkpoint_max_snapshots": "max_snapshots",
    "checkpoint_max_total_size_mb": "max_total_size_mb",
    "checkpoint_max_file_size_mb": "max_file_size_mb",
}


def _merge_dataclass(instance: T, kwargs: MutableMapping[str, Any]) -> T:
    updates: dict[str, Any] = {}
    dataclass_instance = cast(Any, instance)
    for field in fields(dataclass_instance):
        if field.name in kwargs:
            updates[field.name] = kwargs.pop(field.name)
    if updates:
        return cast(T, replace(dataclass_instance, **updates))
    return instance


def _merge_checkpoint_config(
    instance: CheckpointConfig, kwargs: MutableMapping[str, Any]
) -> CheckpointConfig:
    updates: dict[str, Any] = {}
    for legacy_name, field_name in _CHECKPOINT_LEGACY_NAMES.items():
        if legacy_name in kwargs:
            updates[field_name] = kwargs.pop(legacy_name)
    if updates:
        return replace(instance, **updates)
    return instance


def coerce_init_agent_configs(
    legacy_args: Sequence[Any],
    legacy_kwargs: MutableMapping[str, Any],
    *,
    config: AgentConfig | None = None,
    agent_config: AgentConfig | None = None,
    callback_config: CallbackConfig | None = None,
    platform_context: PlatformContext | None = None,
    checkpoint_config: CheckpointConfig | None = None,
) -> tuple[AgentConfig, CallbackConfig, PlatformContext, CheckpointConfig]:
    """Build init-agent config objects from either dataclasses or legacy kwargs."""

    if config is not None and agent_config is not None:
        raise TypeError("init_agent() received both config and agent_config")
    if len(legacy_args) > len(LEGACY_INIT_AGENT_PARAMS):
        raise TypeError(
            "init_agent() received too many positional arguments "
            f"({len(legacy_args)} given after agent)"
        )

    for name, value in zip(LEGACY_INIT_AGENT_PARAMS, legacy_args):
        if name in legacy_kwargs:
            raise TypeError(f"init_agent() got multiple values for argument '{name}'")
        legacy_kwargs[name] = value

    resolved_agent_config = agent_config or config or AgentConfig()
    resolved_callback_config = callback_config or CallbackConfig()
    resolved_platform_context = platform_context or PlatformContext()
    resolved_checkpoint_config = checkpoint_config or CheckpointConfig()

    resolved_agent_config = _merge_dataclass(resolved_agent_config, legacy_kwargs)
    resolved_callback_config = _merge_dataclass(resolved_callback_config, legacy_kwargs)
    resolved_platform_context = _merge_dataclass(resolved_platform_context, legacy_kwargs)
    resolved_checkpoint_config = _merge_checkpoint_config(
        resolved_checkpoint_config,
        legacy_kwargs,
    )

    if legacy_kwargs:
        unexpected = next(iter(legacy_kwargs))
        raise TypeError(f"init_agent() got an unexpected keyword argument '{unexpected}'")

    return (
        resolved_agent_config,
        resolved_callback_config,
        resolved_platform_context,
        resolved_checkpoint_config,
    )
