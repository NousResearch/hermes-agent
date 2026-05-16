"""Configuration helpers for the DCP context engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DCP_DEFAULT_PROTECTED_TOOLS = {
    "delegate_task",
    "todo",
    "memory",
    "skill_view",
    "skill_manage",
    "write_file",
    "patch",
    "clarify",
    "cronjob",
    "compress",
}


@dataclass(slots=True)
class DCPStrategyConfig:
    enabled: bool = True
    protected_tools: set[str] = field(default_factory=set)


@dataclass(slots=True)
class DCPPurgeErrorsConfig(DCPStrategyConfig):
    turns: int = 4


@dataclass(slots=True)
class DCPCompressConfig:
    mode: Literal["range", "message"] = "range"
    permission: Literal["allow", "ask", "deny"] = "allow"
    show_compression: bool = False
    summary_buffer: bool = True
    max_context_limit: int | str = 100000
    min_context_limit: int | str = 50000
    model_max_limits: dict[str, int | str] = field(default_factory=dict)
    model_min_limits: dict[str, int | str] = field(default_factory=dict)
    nudge_frequency: int = 5
    iteration_nudge_threshold: int = 15
    nudge_force: Literal["soft", "strong"] = "soft"
    protected_tools: set[str] = field(default_factory=set)
    protect_user_messages: bool = False


@dataclass(slots=True)
class DCPTurnProtectionConfig:
    enabled: bool = False
    turns: int = 4


@dataclass(slots=True)
class DCPManualModeConfig:
    enabled: bool = False
    automatic_strategies: bool = True


@dataclass(slots=True)
class DCPCommandsConfig:
    enabled: bool = True
    protected_tools: set[str] = field(default_factory=set)


@dataclass(slots=True)
class DCPExperimentalConfig:
    allow_subagents: bool = False
    custom_prompts: bool = False


@dataclass(slots=True)
class DCPConfig:
    enabled: bool = True
    debug: bool = False
    prune_notification: Literal["off", "minimal", "detailed"] = "detailed"
    prune_notification_type: Literal["chat", "toast"] = "chat"
    commands: DCPCommandsConfig = field(default_factory=DCPCommandsConfig)
    manual_mode: DCPManualModeConfig = field(default_factory=DCPManualModeConfig)
    turn_protection: DCPTurnProtectionConfig = field(default_factory=DCPTurnProtectionConfig)
    experimental: DCPExperimentalConfig = field(default_factory=DCPExperimentalConfig)
    protected_file_patterns: list[str] = field(default_factory=list)
    compress: DCPCompressConfig = field(default_factory=DCPCompressConfig)
    deduplication: DCPStrategyConfig = field(default_factory=DCPStrategyConfig)
    purge_errors: DCPPurgeErrorsConfig = field(default_factory=DCPPurgeErrorsConfig)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _as_choice(value: Any, choices: set[str], default: str) -> str:
    if isinstance(value, str) and value in choices:
        return value
    return default


def _as_limit(value: Any, default: int | str) -> int | str:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            try:
                pct = float(raw[:-1])
            except ValueError:
                return default
            if pct > 0:
                return raw
        else:
            try:
                parsed = int(raw)
            except ValueError:
                return default
            if parsed > 0:
                return parsed
    return default


def _as_limit_map(value: Any) -> dict[str, int | str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int | str] = {}
    for key, limit in value.items():
        if not isinstance(key, str):
            continue
        parsed = _as_limit(limit, 0)
        if parsed:
            out[key] = parsed
    return out


def _tool_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item.strip()}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def parse_dcp_config(config: dict[str, Any] | None) -> DCPConfig:
    """Parse ``context.dcp`` config into typed defaults."""
    raw = config if isinstance(config, dict) else {}

    commands_raw = raw.get("commands", {}) if isinstance(raw.get("commands", {}), dict) else {}
    manual_raw = raw.get("manualMode", {}) if isinstance(raw.get("manualMode", {}), dict) else {}
    turn_raw = raw.get("turnProtection", {}) if isinstance(raw.get("turnProtection", {}), dict) else {}
    exp_raw = raw.get("experimental", {}) if isinstance(raw.get("experimental", {}), dict) else {}
    compress_raw = raw.get("compress", {}) if isinstance(raw.get("compress", {}), dict) else {}
    strategies_raw = raw.get("strategies", {}) if isinstance(raw.get("strategies", {}), dict) else {}
    dedup_raw = strategies_raw.get("deduplication", {}) if isinstance(strategies_raw.get("deduplication", {}), dict) else {}
    purge_raw = strategies_raw.get("purgeErrors", {}) if isinstance(strategies_raw.get("purgeErrors", {}), dict) else {}

    return DCPConfig(
        enabled=_as_bool(raw.get("enabled"), True),
        debug=_as_bool(raw.get("debug"), False),
        prune_notification=_as_choice(raw.get("pruneNotification"), {"off", "minimal", "detailed"}, "detailed"),  # type: ignore[arg-type]
        prune_notification_type=_as_choice(raw.get("pruneNotificationType"), {"chat", "toast"}, "chat"),  # type: ignore[arg-type]
        commands=DCPCommandsConfig(
            enabled=_as_bool(commands_raw.get("enabled"), True),
            protected_tools=_tool_set(commands_raw.get("protectedTools")),
        ),
        manual_mode=DCPManualModeConfig(
            enabled=_as_bool(manual_raw.get("enabled"), False),
            automatic_strategies=_as_bool(manual_raw.get("automaticStrategies"), True),
        ),
        turn_protection=DCPTurnProtectionConfig(
            enabled=_as_bool(turn_raw.get("enabled"), False),
            turns=_as_int(turn_raw.get("turns"), 4, minimum=0),
        ),
        experimental=DCPExperimentalConfig(
            allow_subagents=_as_bool(exp_raw.get("allowSubAgents"), False),
            custom_prompts=_as_bool(exp_raw.get("customPrompts"), False),
        ),
        protected_file_patterns=_str_list(raw.get("protectedFilePatterns")),
        compress=DCPCompressConfig(
            mode=_as_choice(compress_raw.get("mode"), {"range", "message"}, "range"),  # type: ignore[arg-type]
            permission=_as_choice(compress_raw.get("permission"), {"allow", "ask", "deny"}, "allow"),  # type: ignore[arg-type]
            show_compression=_as_bool(compress_raw.get("showCompression"), False),
            summary_buffer=_as_bool(compress_raw.get("summaryBuffer"), True),
            max_context_limit=_as_limit(compress_raw.get("maxContextLimit"), 100000),
            min_context_limit=_as_limit(compress_raw.get("minContextLimit"), 50000),
            model_max_limits=_as_limit_map(compress_raw.get("modelMaxLimits")),
            model_min_limits=_as_limit_map(compress_raw.get("modelMinLimits")),
            nudge_frequency=_as_int(compress_raw.get("nudgeFrequency"), 5, minimum=1),
            iteration_nudge_threshold=_as_int(compress_raw.get("iterationNudgeThreshold"), 15, minimum=1),
            nudge_force=_as_choice(compress_raw.get("nudgeForce"), {"soft", "strong"}, "soft"),  # type: ignore[arg-type]
            protected_tools=_tool_set(compress_raw.get("protectedTools")),
            protect_user_messages=_as_bool(compress_raw.get("protectUserMessages"), False),
        ),
        deduplication=DCPStrategyConfig(
            enabled=_as_bool(dedup_raw.get("enabled"), True),
            protected_tools=_tool_set(dedup_raw.get("protectedTools")),
        ),
        purge_errors=DCPPurgeErrorsConfig(
            enabled=_as_bool(purge_raw.get("enabled"), True),
            protected_tools=_tool_set(purge_raw.get("protectedTools")),
            turns=_as_int(purge_raw.get("turns"), 4, minimum=0),
        ),
    )


def resolve_limit(limit: int | str, context_length: int) -> int:
    """Resolve an absolute token limit or percentage string."""
    if isinstance(limit, int):
        return limit
    raw = limit.strip()
    if raw.endswith("%"):
        try:
            pct = float(raw[:-1]) / 100.0
        except ValueError:
            return 0
        return int(context_length * pct)
    try:
        return int(raw)
    except ValueError:
        return 0


def resolve_model_limit(
    limits: dict[str, int | str],
    *,
    provider: str | None,
    model: str | None,
    context_length: int,
    fallback: int | str,
) -> int:
    """Resolve DCP per-model limits with a simple provider/model key."""
    keys = []
    if provider and model:
        keys.append(f"{provider}/{model}")
    if model:
        keys.append(model)
    for key in keys:
        if key in limits:
            return resolve_limit(limits[key], context_length)
    return resolve_limit(fallback, context_length)
