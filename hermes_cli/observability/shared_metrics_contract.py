"""Bounded product contract for the first Hermes shared-metrics slice."""

from __future__ import annotations

from math import isfinite
from typing import Any

from agent.error_classifier import FailoverReason
from agent.relay_runtime import (
    LOGICAL_LLM_SCOPE,
    RUNTIME_INSTANCE_KEY,
    RUNTIME_SCHEMA_KEY,
    RUNTIME_SCHEMA_VERSION,
)
from hermes_cli.platforms import PLATFORMS
from toolsets import BUILTIN_TOOL_NAMES, BUILTIN_TOOLSET_NAMES

SCHEMA_KEY = "hermes.metrics.schema_version"
# A random per-emit token (no payload) on rows whose producer waits to learn they were SAVED: facts
# recovered from a file that is deleted only then. Events persist on the Relay thread.
COMMIT_TICKET_KEY = "hermes.shared_metrics.commit_ticket"
SCHEMA_VERSION = "hermes.metrics.event.v4"
MODEL_CALL_SCOPE = "hermes.model_call"
MODEL_CALL_PROFILE_MODEL = "unknown"
TASK_SCOPE = "hermes.task_run"
TOOL_CALL_SCOPE = "hermes.tool_call"
CLIENT_ACTIVE_MARK = "hermes.client.active"
TOOL_APPROVAL_MARK = "hermes.tool_approval"
SKILL_LIFECYCLE_MARK = "hermes.skill.lifecycle"
SKILL_LOAD_MARK = "hermes.skill.load"
INSTALL_SNAPSHOT_MARK = "hermes.install.snapshot"
SESSION_MARK = "hermes.session"
SETUP_COMPLETED_MARK = "hermes.setup.completed"
MODEL_TOKENS_MARK = "hermes.model_tokens"
COMPRESSION_MARK = "hermes.compression"
MODEL_SWITCH_MARK = "hermes.model_switch"
FALLBACK_MARK = "hermes.fallback"
SLASH_COMMAND_MARK = "hermes.slash_command"
EXTENSION_INSTALL_MARK = "hermes.extension.install"
# ---- v4 install ----
STARTUP_LATENCY_MARK = "hermes.startup.latency"
SUBSCRIBER_NAME = "hermes.nemo_relay.shared_metrics"
CLIENT_ACTIVE_METRIC = "hermes.client.active"
LEGACY_MODEL_CALL_METRIC = "hermes.model_call.count"
MODEL_ROUTE_METRIC = "hermes.model_route.count"
TASK_STARTED_METRIC = "hermes.task_run.started"
TASK_FINISHED_METRIC = "hermes.task_run.finished"
TASK_DURATION_METRIC = "hermes.task_run.duration"
TOOL_CALL_METRIC = "hermes.tool_call.count"
TOOL_LATENCY_METRIC = "hermes.tool_call.latency"
TOOL_APPROVAL_METRIC = "hermes.tool_approval.count"
SKILL_LIFECYCLE_METRIC = "hermes.skill.lifecycle.count"
SKILL_LOAD_METRIC = "hermes.skill.load.count"
TOOL_USAGE_METRIC = "hermes.tool.usage.count"
INSTALL_SNAPSHOT_METRIC = "hermes.install.snapshot"
SESSION_METRIC = "hermes.session.count"
MILESTONE_METRIC = "hermes.install.milestone"
SETUP_COMPLETED_METRIC = "hermes.setup.completed"
MODEL_TOKENS_METRIC = "hermes.model_tokens.sum"
COMPRESSION_METRIC = "hermes.compression.count"
MODEL_SWITCH_METRIC = "hermes.model_switch.count"
FALLBACK_METRIC = "hermes.fallback.count"
SLASH_COMMAND_METRIC = "hermes.slash_command.count"
EXTENSION_INSTALL_METRIC = "hermes.extension.install.count"
# ---- v4 loop ----
MEMORY_OP_MARK = "hermes.memory.op"
CURATOR_RUN_MARK = "hermes.curator.run"
DELEGATION_RUN_MARK = "hermes.delegation.run"
EXECUTION_BACKEND_MARK = "hermes.execution_backend"
MEMORY_OP_METRIC = "hermes.memory.op.count"
CURATOR_RUN_METRIC = "hermes.curator.run.count"
DELEGATION_RUN_METRIC = "hermes.delegation.run.count"
EXECUTION_BACKEND_METRIC = "hermes.execution_backend.count"
# ---- end v4 loop ----
# ---- v4 model ----
MODEL_TOOL_QUALITY_MARK = "hermes.model_tool_quality"
MODEL_FRICTION_MARK = "hermes.model_friction"
CONTEXT_PEAK_MARK = "hermes.context_peak"
MODEL_TOOL_QUALITY_METRIC = "hermes.model_tool_quality.count"
MODEL_FRICTION_METRIC = "hermes.model_friction.count"
CONTEXT_PEAK_METRIC = "hermes.context_peak.count"
# ---- end v4 model ----
# ---- v4 install ----
STARTUP_LATENCY_METRIC = "hermes.startup.latency"
# ---- v5 engagement ----
# Not a counter: one attended turn's serving model, folded into the local daily engagement rollup.
ENGAGEMENT_TURN_MARK = "hermes.engagement.turn"
ENGAGEMENT_DAY_METRIC = "hermes.engagement.day.count"
ENGAGEMENT_SURFACE_METRIC = "hermes.engagement.surface_day.count"
MODEL_SWITCH_AFTER_MARK = "hermes.model_switch_after"
MODEL_SWITCH_AFTER_METRIC = "hermes.model_switch_after.count"
# ---- end v5 engagement ----
MODEL_IDENTIFIER_MAX_LENGTH = 256
PROVIDER_IDENTIFIER_MAX_LENGTH = 64
_METRIC_IDENTIFIER_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._:/@+-")
_METRIC_IDENTIFIER_START_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")

EXECUTION_SURFACES = frozenset({
    "acp", "api", "batch", "cli", "desktop", "gateway", "python", "scheduled_task", "tui",
    "other", "unknown",
})
TASK_OUTCOMES = frozenset({"cancelled", "failed", "success", "timed_out", "unknown"})
TASK_END_REASONS = frozenset({
    "approval_denied", "completed", "failed", "guardrail_blocked", "iteration_limit",
    "system_aborted", "timed_out", "unknown", "user_cancelled",
})
TASK_TERMINATIONS = frozenset({"none", "system_aborted", "timed_out", "unknown", "user_cancelled"})
# ``one_shot``: a finite CLI run (``hermes -z``, ``hermes chat -q`` off a TTY, ``-Q``, ``--oneshot``) that
# answers one prompt and exits — a person's shell line or their script, never the REPL.
TASK_ENTRYPOINTS = frozenset({
    "api", "background", "batch", "delegated", "gateway_message", "interactive", "one_shot", "other",
    "python", "scheduled_task", "unknown",
})
DURATION_BUCKETS = frozenset({
    "1s_to_5s", "2m_to_10m", "30s_to_2m", "5s_to_30s", "gte_10m", "lt_1s",
})
COUNT_BUCKETS = frozenset({"0", "1", "2", "3_to_5", "6_to_10", "gte_11"})
TOOL_CATEGORIES = frozenset({
    "browser", "code_execution", "communication", "computer_use", "delegation", "file",
    "home_automation", "mcp", "media", "memory", "other", "planning", "project", "scheduler",
    "skill", "terminal", "unknown", "web",
})
TOOL_OUTCOMES = frozenset({"blocked", "cancelled", "failed", "success", "timed_out", "unknown"})
TOOL_APPROVAL_OUTCOMES = frozenset({"approved", "cancelled", "denied", "not_required", "timed_out", "unknown"})
TOOL_APPROVAL_ATTRIBUTIONS = frozenset({"tool_call", "unattributed"})
TOOL_LATENCY_BUCKETS = frozenset({
    "100ms_to_250ms", "10s_to_30s", "1s_to_2s", "250ms_to_500ms", "2s_to_5s", "500ms_to_1s",
    "5s_to_10s", "gte_30s", "lt_100ms", "unknown",
})
TOOL_RETRY_BUCKETS = COUNT_BUCKETS | frozenset({"unknown"})
SKILL_LIFECYCLE_ACTIONS = frozenset({
    "archived", "created", "edited", "installed", "patched", "restored", "stale",
})
SKILL_PROVENANCES = frozenset({"agent_created", "external", "installed", "local", "unknown"})
SKILL_REUSE_STATES = frozenset({"first_use", "reused"})
SKILL_POST_PATCH_STATES = frozenset({"no_new_patch", "not_applicable", "reused_after_patch"})
CLIENT_OS_FAMILIES = frozenset({"linux", "macos", "unknown", "windows"})
CLIENT_ARCHITECTURES = frozenset({"arm", "arm64", "unknown", "x86", "x86_64"})
CLIENT_INSTALL_METHODS = frozenset({
    "apt", "docker", "git", "home-manager", "homebrew", "nixos", "pip", "unknown",
})
CLIENT_RESOURCE_KEYS = frozenset({"architecture", "hermes_version", "install_method", "os_family"})

# ---- v3 taxonomies -------------------------------------------------------------------------
MODEL_CALL_ROLES = frozenset({"auxiliary", "primary"})
MODEL_OUTCOMES = frozenset({"cancelled", "failed", "success"})
# The classifier's own closed enum, so the exported vocabulary is exactly what recovery acts on.
MODEL_ERROR_CLASSES = frozenset(reason.value for reason in FailoverReason) | {"none"}
TASK_FAILURE_CLASSES = MODEL_ERROR_CLASSES | frozenset({
    "context_compression", "empty_response", "exception", "local_error", "other", "persistence",
    "repeated_errors", "restart_limit", "session_busy", "shutdown",
})
# Surfaces that are not messaging platforms keep their own execution_surface value.
# ``relay``: a connector inbound whose platform neither the inbound nor a single-platform connector names.
_CORE_GATEWAY_PLATFORMS = (frozenset(PLATFORMS) - {"api_server", "cli", "cron"}) | {"none", "plugin", "relay"}
TOOL_NAMES = BUILTIN_TOOL_NAMES | {"mcp", "plugin", "unknown"}
TOOL_ERROR_CLASSES = frozenset({
    "blocked", "contract_violation", "exception", "interrupted", "invalid_arguments", "none",
    "timeout", "tool_error", "unknown",
})
SIZE_BUCKETS = frozenset({
    "0", "1", "2", "3_to_5", "6_to_10", "11_to_25", "26_to_100", "101_to_250", "gte_251",
})


def _bundled_memory_providers() -> frozenset[str]:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "plugins" / "memory"
    try:
        return frozenset(p.name for p in root.iterdir() if (p / "__init__.py").is_file())
    except OSError:
        return frozenset()


class _CatalogValues:
    """A closed enum backed by a public catalog (see shared_metrics_catalog), loaded on first use."""

    def __init__(self, *loaders: str, extra: frozenset[str]) -> None:
        self.loaders, self.extra = loaders, extra

    def values(self) -> frozenset[str]:
        from . import shared_metrics_catalog as catalog

        return self.extra.union(*(catalog._safe(getattr(catalog, name)) for name in self.loaders))

    def __contains__(self, value: object) -> bool:
        return value in self.extra or value in self.values()


# A provider that moved from plugins/memory/ to the plugin catalog keeps its public name; any other
# third-party provider reports as "plugin".
MEMORY_PROVIDERS = _CatalogValues("plugin_catalog_names", extra=_bundled_memory_providers() | {"builtin", "plugin"})

# ---- decision-data taxonomies ----------------------------------------------------------------
SESSION_DURATION_BUCKETS = frozenset({"lt_1m", "1m_to_5m", "5m_to_30m", "30m_to_2h", "2h_to_8h", "gte_8h"})
INSTALL_AGE_BUCKETS = frozenset({
    "lt_1h", "1h_to_1d", "1d_to_7d", "7d_to_30d", "30d_to_90d", "gte_90d", "unknown",
})
MILESTONES = frozenset({
    "setup_completed", "first_task_started", "first_task_success", "first_tool_success",
    "first_gateway_message", "first_scheduled_task", "first_delegation", "first_skill_created",
    "first_skill_reused", "first_mcp_tool_success", "first_long_session",
})
SETUP_SURFACES = frozenset({"cli", "desktop", "other"})
TOKEN_TYPES = frozenset({"input", "output", "cache_read", "cache_write", "reasoning"})
TTFT_BUCKETS = frozenset({
    "lt_500ms", "500ms_to_1s", "1s_to_2s", "2s_to_5s", "5s_to_15s", "gte_15s", "not_streamed", "unknown",
})
COMPRESSION_TRIGGERS = frozenset({"auto", "manual", "other", "overflow"})
COMPRESSION_OUTCOMES = frozenset({"failed", "skipped", "success"})
# Why an attempt was skipped or failed: the attempt log's own ``failure_class`` names
# (agent/conversation_compression*.py, agent/context_compressor.py). Skips are attempts that ended
# before anything could fail (another path held the lock, nothing summarizable so no model call, the
# user stopped it, a newer attempt replaced it). ``exception:<Type>`` / ``rollback:<Type>`` keep only
# their prefix; any other value reads ``other``, a failure with no class ``unknown``.
COMPRESSION_SKIP_CLASSES = frozenset({
    "attempt_superseded", "empty_post_handoff_window", "explicit_interrupt", "insufficient_messages",
    "lock_contended", "no_compressible_window", "snapshot_stale",
})
COMPRESSION_FAILURE_CLASSES = COMPRESSION_SKIP_CLASSES | frozenset({
    "aux_model_fallback", "commit_fence_cancelled", "empty_transcript", "exception", "feasibility_skip",
    "no_progress", "none",
    "other", "pool_saturated", "rollback", "session_split_failed", "stall_deterministic_fallback",
    "stall_interrupted", "summary_auth_failure", "summary_empty_content_failure", "summary_generation_aborted",
    "summary_generation_failed", "summary_network_failure", "summary_overload_degraded",
    "summary_overload_failure", "summary_truncated_failure", "unknown", "would_grow",
})
CONTEXT_FILL_BUCKETS = frozenset({"lt_50", "50_to_75", "75_to_90", "90_to_100", "gte_100", "unknown"})
EXTENSION_KINDS = frozenset({"mcp_server", "plugin", "skill"})
EXTENSION_SOURCES = frozenset({"bundled", "catalog", "hub", "local", "other", "url"})
EXTENSION_OUTCOMES = frozenset({"failed", "success"})
# Every built-in backend in tools/terminal_tool_config.py::_BUILTIN_BACKENDS.
TERMINAL_BACKENDS = frozenset({
    "daytona", "docker", "local", "managed_modal", "modal", "other", "singularity", "ssh", "vercel_sandbox",
})
AUX_TASKS = _CatalogValues("aux_task_names", extra=frozenset({"none", "other"}))
SLASH_COMMANDS = _CatalogValues("slash_command_names", extra=frozenset({"plugin", "skill", "unknown"}))
SKILL_NAMES = _CatalogValues("bundled_skill_names", extra=frozenset({"custom"}))
EXTENSION_NAMES = _CatalogValues(
    "bundled_skill_names", "mcp_catalog_names", "plugin_catalog_names", extra=frozenset({"custom"})
)
DISPLAY_LANGUAGES = _CatalogValues("display_languages", extra=frozenset({"other"}))
# ---- v4 model ----
TOOL_CALL_ISSUES = frozenset({
    "empty_arguments", "invalid_json", "none", "repaired", "schema_mismatch", "unknown_tool",
})
FRICTION_SIGNALS = frozenset({"interrupt", "quick_abandon", "retry", "switch_away", "undo"})
CONTEXT_WINDOW_BUCKETS = frozenset({
    "lt_32k", "32k_to_128k", "128k_to_256k", "256k_to_1m", "gte_1m", "unknown",
})
LIMIT_HIT_VALUES = frozenset({"no", "yes"})
# ---- end v4 model ----

# ---- v4 loop ----
MEMORY_OPS = frozenset({"add", "other", "patch", "read", "remove", "replace", "search"})
MEMORY_OP_OUTCOMES = frozenset({"failed", "rejected", "success"})
# Who asked: the user's turn, or the unattended self-improvement review fork.
MEMORY_OP_ORIGINS = frozenset({"background_review", "foreground"})
# Why a memory op was refused or failed. Built-in classes name each refusal/failure return of
# tools/memory_tool.py and tools/memory_tool_store.py; plugin providers report ``provider_error`` (the
# tool returned an error) or ``exception`` (it raised). Successes are ``none``.
MEMORY_OP_FAILURE_CLASSES = frozenset({
    "ambiguous", "disabled", "drift", "exception", "gate_refused", "invalid_args", "missing_content",
    "missing_old_text", "no_match", "none", "other", "over_budget", "provider_error", "read_failed",
    "retry_cap", "scan_blocked", "staged", "stale_entry", "unknown", "would_empty",
})
CURATOR_OUTCOMES = frozenset({"failed", "skipped", "success"})
CURATOR_TRIGGERS = frozenset({"manual", "scheduled"})
DELEGATION_OUTCOMES = frozenset({"cancelled", "failed", "partial", "success"})
DELEGATION_DEPTHS = frozenset({"1", "2", "3", "gte_4"})
DELEGATION_MODES = frozenset({"background", "foreground"})
EXECUTION_KINDS = frozenset({"browser", "code", "terminal"})
# Shipped browser backends: tools/browser_{camofox,lightpanda,tool_cdp,tool_cloud,extension_router}.py
# plus the bundled cloud providers in plugins/browser/.
BROWSER_BACKENDS = frozenset({
    "browser-use", "browserbase", "camofox", "cdp", "extension", "firecrawl", "lightpanda", "local", "other",
})
CODE_BACKENDS = frozenset({"local", "other", "remote"})
EXECUTION_BACKENDS_BY_KIND = {
    "browser": BROWSER_BACKENDS, "code": CODE_BACKENDS, "terminal": TERMINAL_BACKENDS,
}
EXECUTION_BACKENDS = BROWSER_BACKENDS | CODE_BACKENDS | TERMINAL_BACKENDS
EXECUTION_OUTCOMES = frozenset({"failed", "success"})
# ---- end v4 loop ----
# ---- v4 gateway ----
PLATFORM_HEALTH_MARK = PLATFORM_HEALTH_METRIC = "hermes.platform.health"
PLATFORM_DELIVERY_MARK = PLATFORM_DELIVERY_METRIC = "hermes.platform.delivery"
REPLY_LATENCY_MARK = REPLY_LATENCY_METRIC = "hermes.gateway.reply_latency"
CRON_RUN_MARK = CRON_RUN_METRIC = "hermes.cron.run"
# Plugin platforms are named only when Nous ships them (plugins/platforms/) or the installer proved a
# plugin-catalog install (shared_metrics_catalog.platform_metric_name); every other one is ``plugin``.
_PLATFORM_CATALOGS = ("bundled_platform_names", "catalog_platform_names")
GATEWAY_PLATFORMS = _CatalogValues(*_PLATFORM_CATALOGS, extra=_CORE_GATEWAY_PLATFORMS)


def _gateway_adapter_platforms() -> frozenset[str]:
    """The gateway's own adapter vocabulary: ``Platform``'s declared members (relay, msgraph_webhook
    and api_server are adapters but not setup-wizard platforms; plugin pseudo-members are not
    declared). Loaded on first use: gateway.config is too heavy for this module's import."""
    from gateway.config import Platform

    return frozenset(member.value for member in Platform)


class _AdapterPlatforms(_CatalogValues):
    def values(self) -> frozenset[str]:
        return super().values() | _gateway_adapter_platforms()


ADAPTER_PLATFORMS = _AdapterPlatforms(*_PLATFORM_CATALOGS, extra=_CORE_GATEWAY_PLATFORMS | {"api_server"})
PLATFORM_HEALTH_EVENTS = frozenset({"connect_ok", "connect_failed", "reconnect", "disconnect"})
PLATFORM_ERROR_CLASSES = frozenset({"none", "auth", "network", "rate_limited", "config", "other"})
DELIVERY_OUTCOMES = frozenset({"sent", "failed"})
DELIVERY_FAILURE_CLASSES = frozenset({"none", "rate_limited", "too_long", "auth", "network", "forbidden", "other"})
REPLY_LATENCY_BUCKETS = frozenset({"lt_2s", "2s_to_5s", "5s_to_15s", "15s_to_60s", "gte_60s"})
CRON_RUN_OUTCOMES = frozenset({"success", "failed", "missed", "skipped"})
CRON_DELIVERY_KINDS = frozenset({"local", "platform", "webhook", "none", "other"})

# ---- v4 install ----
STARTUP_SURFACES = frozenset({"cli", "desktop_attach", "gateway_boot", "serve_boot", "tui"})
STARTUP_LATENCY_BUCKETS = frozenset({
    "lt_500ms", "500ms_to_1s", "1s_to_2s", "2s_to_5s", "5s_to_10s", "gte_10s",
})
RELEASE_CHANNELS = frozenset({"dev", "main", "stable", "unknown"})
VERSION_AGE_BUCKETS = frozenset({"lt_7d", "7d_to_30d", "30d_to_90d", "gte_90d", "unknown"})
BEHIND_BUCKETS = COUNT_BUCKETS | {"unknown"}
RAM_BUCKETS = frozenset({
    "lt_8g", "8g_to_16g", "16g_to_32g", "32g_to_64g", "64g_to_128g", "gte_128g", "unknown",
})
GPU_CLASSES = frozenset({"amd", "apple_silicon", "intel", "none", "nvidia", "unknown"})
YES_NO = frozenset({"no", "yes"})
_INSTALL_V4_SNAPSHOT_DIMENSIONS = {
    "behind_bucket": BEHIND_BUCKETS, "gpu_class": GPU_CLASSES, "local_model_provider_used": YES_NO,
    "ram_bucket": RAM_BUCKETS, "release_channel": RELEASE_CHANNELS, "version_age_bucket": VERSION_AGE_BUCKETS,
}

# ---- v5 engagement ----
ENGAGEMENT_SURFACES = frozenset({"acp", "cli", "cron", "desktop", "gateway", "tui"})
ACTIVE_MINUTES_BUCKETS = frozenset({"0", "lt_5m", "5m_to_30m", "30m_to_2h", "2h_to_6h", "gte_6h"})
SURFACES_USED_BUCKETS = frozenset({"0", "1", "2", "3", "gte_4"})
TURNS_BEFORE_SWITCH_BUCKETS = frozenset({"1", "2_to_3", "4_to_10", "11_to_30", "gte_31"})
# SIZE_BUCKETS plus a longer tail: agentic conversations run to thousands of messages.
LONG_SIZE_BUCKETS = SIZE_BUCKETS | frozenset({"251_to_1000", "gte_1001"})
_LONG_SIZE_THRESHOLDS = (
    (1, "0"), (2, "1"), (3, "2"), (6, "3_to_5"), (11, "6_to_10"), (26, "11_to_25"),
    (101, "26_to_100"), (251, "101_to_250"), (1001, "251_to_1000"),
)


def long_size_bucket(count: int) -> str:
    return _bucket(max(0, int(count)), _LONG_SIZE_THRESHOLDS, "gte_1001")
# ---- end v5 engagement ----

# ---- v4 reliability ----
UPDATE_RUN_MARK = UPDATE_RUN_METRIC = "hermes.update.run"
UPDATE_STAGE_MARK = UPDATE_STAGE_METRIC = "hermes.update.stage"
PROCESS_EXIT_MARK = PROCESS_EXIT_METRIC = "hermes.process.exit"
UPDATE_KINDS = frozenset({"cli", "desktop"})
UPDATE_OUTCOMES = frozenset({"failed", "noop", "refused", "success"})
# ``partial``: committed, but interrupted after the commit point or the user's changes stay parked
# (only `hermes update` receipts; the Desktop RPC keeps UPDATE_OUTCOMES).
UPDATE_RUN_OUTCOMES = UPDATE_OUTCOMES | {"partial"}
# `hermes update` pipeline stages, in pipeline order (the receipt's stage marks use these names).
UPDATE_STAGE_ORDER = ("plan", "snapshot", "apply", "deps", "build", "restart", "verify")
UPDATE_STAGES = frozenset(UPDATE_STAGE_ORDER)
# Desktop's packaged updaters (electron-updater, App Installer, Store) fail at their own steps.
DESKTOP_UPDATE_STAGES = frozenset({"apply", "download", "restart", "verify"})
UPDATE_FAILED_STAGES = UPDATE_STAGES | DESKTOP_UPDATE_STAGES | {"none", "other"}
UPDATE_STAGE_OUTCOMES = frozenset({"failed", "skipped", "success"})
UPDATE_DURATION_BUCKETS = frozenset({"lt_30s", "30s_to_2m", "2m_to_5m", "5m_to_15m", "gte_15m"})
# package = an OS/app-store style installer applied the update (Desktop packaged builds).
UPDATE_APPLY_MODES = frozenset({"external", "git", "package", "unknown", "zip"})
PROCESS_KINDS = frozenset({"cli", "cron", "gateway", "other", "serve", "tui"})
PROCESS_EXIT_KINDS = frozenset({"clean", "crash", "killed", "watchdog"})
CRASH_CLASSES = frozenset({"import_error", "memory_error", "none", "os_error", "other", "runtime_error"})
_UPDATE_DURATION_THRESHOLDS = (
    (30_000, "lt_30s"), (120_000, "30s_to_2m"), (300_000, "2m_to_5m"), (900_000, "5m_to_15m"),
)
_DAY_MS = 86_400_000
_VERSION_AGE_THRESHOLDS = ((7 * _DAY_MS, "lt_7d"), (30 * _DAY_MS, "7d_to_30d"), (90 * _DAY_MS, "30d_to_90d"))


# ---- v5 signals ----
TOOL_UNAVAILABLE_MARK = TOOL_UNAVAILABLE_METRIC = "hermes.tool_unavailable.count"
PROVIDER_SETUP_MARK = PROVIDER_SETUP_METRIC = "hermes.provider_setup.count"
FEATURE_ADOPTION_METRIC = "hermes.feature_adoption.count"
# Not a counter: a first-use fact the subscriber turns into at most one adoption row per feature.
FEATURE_USED_MARK = "hermes.feature.used"
PROVIDER_SETUP_SURFACES = frozenset({"cli_model", "cli_setup", "dashboard", "desktop", "tui"})
PROVIDER_SETUP_EVENTS = frozenset({"abandoned", "completed", "failed", "started"})
PROVIDER_SETUP_FAILURE_CLASSES = frozenset({"auth", "cancelled", "network", "no_models", "none", "other"})
FEATURES = frozenset({
    "browser", "bot_mode", "cron", "curator", "delegation", "desktop", "gateway_platform", "kanban", "mcp",
    "memory", "plugins", "projects", "skills_created", "tui", "voice",
})
DAYS_SINCE_INSTALL_BUCKETS = frozenset({"same_day", "1d_to_7d", "7d_to_30d", "30d_to_90d", "gte_90d", "unknown"})
FEATURE_DISABLED_MARK = FEATURE_DISABLED_METRIC = "hermes.feature_disabled.count"
FEATURE_DISABLED_KINDS = frozenset({
    "compression", "curator", "memory", "platform", "plugin", "setting", "skill", "toolset",
})
FEATURE_DISABLED_SURFACES = frozenset({"cli_config", "cli_slash", "cli_tools", "dashboard", "desktop", "tui"})
FEATURE_DISABLED_EVENTS = frozenset({"disabled", "re_enabled"})
FEATURE_DISABLED_NAME_MAX_LENGTH = 64
# ---- end v5 signals ----

# ---- iuf c1 ----
# Why an extension install failed, named at each failure exit of the three emitters
# (hermes_cli/skills_hub.py::_install_skill, hermes_cli/plugins_cmd_install.py::recorded_install,
# hermes_cli/mcp_catalog.py::recorded_catalog_install / record_mcp_install); ``none`` on success.
# A raised exception with no class of its own is classified by its TYPE: network / permission /
# filesystem_error, anything else ``exception:<Type>``, which keeps only the ``exception`` prefix
# (as on compression rows). Never the error message.
EXTENSION_COMMON_FAILURE_CLASSES = frozenset({
    "exception", "filesystem_error", "network", "none", "other", "permission",
})
EXTENSION_SKILL_FAILURE_CLASSES = EXTENSION_COMMON_FAILURE_CLASSES | frozenset({
    "ambiguous",          # a short name matched several skills in different registries
    "auth_rejected",      # GitHub refused every configured credential (401)
    "fetch_failed",       # no registry served the files (unreachable, 404, removed)
    "invalid_bundle",     # unsafe path, symlink or invalid skill name in the bundle
    "invalid_name",       # a URL skill with no usable name (non-interactive, bad --name, cancelled prompt)
    "not_found",          # a short name no registry knows, or a pinned registry that does not exist
    "rate_limited",       # the GitHub API rate limit was exhausted
    "scan_blocked",       # the security scan refused the bundle
    "stale_index",        # listed in a registry index whose files no longer exist upstream
})
EXTENSION_PLUGIN_FAILURE_CLASSES = EXTENSION_COMMON_FAILURE_CLASSES | frozenset({
    "already_installed",  # the plugin already exists / is pinned and needs --force or --ref
    "clone_failed",       # git clone / fetch / checkout of the plugin repository failed or timed out
    "deps_declined",      # the user declined the Python dependency prompt
    "deps_failed",        # PM could not prepare the plugin's dependencies
    "git_missing",        # git is not installed
    "incompatible",       # newer manifest_version, unsupported platform / GPU / runtime version
    "invalid_source",     # the identifier or subdirectory does not name a usable git source
    "manifest_invalid",   # unreadable or malformed plugin.yaml / plugin.json / dependency declaration
    "non_interactive",    # Python dependencies need consent and nobody could answer (no --yes-deps)
    "removed_from_catalog",  # on the catalog kill list
    "scan_blocked",       # the security scan refused the plugin
})
EXTENSION_MCP_FAILURE_CLASSES = EXTENSION_COMMON_FAILURE_CLASSES | frozenset({
    "auth_required",      # the server refused the sign-in (OAuth error, HTTP 401/403)
    "bootstrap_failed",   # a catalog entry's bootstrap command exited non-zero
    "clone_failed",       # git clone / checkout of a catalog entry's repository failed
    "config_invalid",     # no such catalog entry, undeclared env var, or a manifest that cannot be built
    "config_rejected",    # Hermes refused a suspicious command/args configuration
    "connect_failed",     # the probe could not reach or initialize the server
    "git_missing",        # git is not installed
    "missing_credentials",  # a required credential was not provided
    "server_start_failed",  # the stdio server command is missing, or its native module needs another Node
})
EXTENSION_KIND_FAILURE_CLASSES = {
    "mcp_server": EXTENSION_MCP_FAILURE_CLASSES, "plugin": EXTENSION_PLUGIN_FAILURE_CLASSES,
    "skill": EXTENSION_SKILL_FAILURE_CLASSES,
}
EXTENSION_FAILURE_CLASSES = (
    EXTENSION_SKILL_FAILURE_CLASSES | EXTENSION_PLUGIN_FAILURE_CLASSES | EXTENSION_MCP_FAILURE_CLASSES
)
# The skills-hub adapter that resolved or served a skill: every ``SOURCE_ID`` the router in
# tools/skills_hub_search.py::create_source_router builds. ``none`` = no adapter was involved
# (bundled restores, plugin and MCP rows), ``unresolved`` = a hub lookup no adapter answered,
# ``other`` = an adapter id this list does not name.
EXTENSION_REGISTRY_IDS = frozenset({
    "browse-sh", "clawhub", "github", "hermes-index", "lobehub", "official", "skills-sh", "url", "well-known",
})
EXTENSION_REGISTRIES = EXTENSION_REGISTRY_IDS | frozenset({"none", "other", "unresolved"})
# Why a `hermes update` run failed or was refused, derived from the FINAL receipt only
# (shared_metrics_update.update_failure_class); ``none`` unless outcome is failed/refused/partial.
UPDATE_FAILURE_CLASSES = frozenset({
    "aborted_before_apply",  # fallback: exited before the checkout moved with no reason recorded
    "build_failed",          # the build stage reported failure
    "deps_failed",           # PM dependency preparation failed (no deps mark, or a PM error type)
    "exception",             # the run ended on an uncaught exception (type name dropped)
    "fleet_stale",           # post-update verification found a gateway still stale or down
    "fleet_unverified",      # verification could not prove the fleet current (no rows, unaccounted runtime)
    "git_failed",            # a git/installer subprocess failed before the checkout moved (stash, pull, merge)
    "interrupted",           # KeyboardInterrupt / exit 130
    "lock_held",             # another updater held the update lock (refused, exit 2)
    "managed_install",       # admission refused: image/package-managed or commit-build install (refused)
    "none", "other",
    "os_error",              # the run ended on an OSError (disk full, permission, file in use)
    "restart_failed",        # the gateway restart failed, or a skipped restart left the fleet owing one
    "subprocess_failed",     # a subprocess error (CalledProcessError, TimeoutExpired) reached the boundary
    "unknown",               # the reporter sent no reason (Desktop packaged updaters)
    "disk_full",             # out of disk space (ENOSPC, or no room to stage the ZIP)
    "local_changes_parked",  # committed, but the user's stashed changes could not be re-applied (partial)
    "permission_denied",     # the run ended on a PermissionError
})
# The exit that stopped a run before its apply stage mark, recorded at the exit itself as one of
# these tokens (update_receipt.record_stop_reason -> the receipt's ``stop_class``).
UPDATE_STOP_CLASSES = frozenset({
    "branch_missing",         # the target branch exists neither locally nor on origin
    "branch_unsupported",     # --branch on the Windows ZIP fallback
    "channel_unresolved",     # the update channel could not be resolved
    "checkout_move_failed",   # git could not move the checkout (ff refused, reset or switch failed)
    "commit_point_refused",   # the commit point could not be armed durably; nothing moved
    "detached_head",          # a detached checkout stayed put, or its commits could not be backed up
    "disk_full",              # (see above)
    "download_failed",        # the ZIP fallback download failed
    "fetch_failed",           # git fetch failed
    "gateway_pause_failed",   # Windows: the gateways could not be paused safely before the venv moves
    "git_in_progress",        # a merge/rebase/cherry-pick/... was already in progress
    "git_index_locked",       # git refused: another git process's .git/index.lock exists
    "git_timeout",            # a network git call hit the updater's time limit
    "head_moved",             # a ref moved during the update; HEAD is not the selected commit
    "local_changes_blocked",  # local changes git could not stash, or a dirty tree the ZIP will not overwrite
    "lock_held",              # (see above)
    "managed_install",        # (see above): HERMES_MANAGED installs refuse before any receipt
    "merge_conflict",         # local commits on a custom branch conflict with upstream
    "not_git_checkout",       # the install is not a git checkout (non-Windows)
    "old_version_handoff",    # an older updater's hand-off to this version could not finish
    "parked_branch_blocked",  # the checkout is parked on another branch that is unsafe to switch
    "permission_denied",      # (see above): git could not write a file it needs (e.g. index.lock's directory)
    "stash_restore_rejected",  # re-applied local changes broke Hermes; the update stopped
    "syntax_rollback",        # the pulled code failed the syntax check and was rolled back
    "target_syntax_error",    # the target failed the syntax check before anything moved
    "target_unresolved",      # the fetched target ref did not resolve to a commit
    "unexpected_branch",      # the checkout ended on a branch that is not the target
    "venv_foreign_owner",     # the venv belongs to another OS user
    "zip_failed",             # the ZIP fallback swap failed and was rolled back
})
UPDATE_FAILURE_CLASSES |= UPDATE_STOP_CLASSES
# ---- end iuf c1 ----
# ---- iuf c2 ----
# One fresh-install run of scripts/install.sh / install.ps1, from the local receipt the installer
# leaves (the installer itself never sends); counted by a later Hermes start while collection is on.
INSTALL_RUN_MARK = INSTALL_RUN_METRIC = "hermes.install.run"
INSTALL_RUN_INSTALLERS = frozenset({"install_ps1", "install_sh", "other"})
INSTALL_RUN_OUTCOMES = frozenset({"failed", "success"})
# The installers' stage_names / $Stages ladder (hyphens as underscores).
INSTALL_RUN_STAGES = frozenset({
    "complete", "config", "gateway", "prerequisites", "products", "python_deps", "repository", "setup", "venv",
})
INSTALL_RUN_FAILED_STAGES = INSTALL_RUN_STAGES | {"none", "other"}
# The code each installer fail()/Fail call site passes; `none` on success.
INSTALL_RUN_FAILURE_CLASSES = frozenset({
    "commit_not_on_branch", "curl_missing", "deps_install_failed", "dir_not_checkout", "download_digest_mismatch",
    "download_failed", "filesystem_error", "gateway_failed", "git_checkout_failed", "git_clone_failed",
    "git_extract_failed", "git_fetch_failed", "git_missing", "git_reset_failed", "interrupted",
    "libstdcxx_missing", "local_changes_blocked", "none", "other", "products_build_failed",
    "python_install_failed", "setup_failed", "unsupported_platform", "uv_unusable",
})
# ---- end iuf c2 ----


def update_duration_bucket(duration_ms: Any) -> str:
    """Bucket an update (or update stage) wall time; non-numbers count as instant."""
    value = _non_negative_number(duration_ms) or 0
    return _bucket(value, _UPDATE_DURATION_THRESHOLDS, "gte_15m")


def version_age_bucket(age_ms: Any) -> str:
    """Bucket the age of the version an update started from; unknown when not measurable."""
    value = _non_negative_number(age_ms)
    return "unknown" if value is None else _bucket(value, _VERSION_AGE_THRESHOLDS, "gte_90d")
# ---- end v4 reliability ----

# ---- v5 harness ----
FILE_EDIT_MARK = "hermes.file_edit"
LOOP_GUARD_MARK = "hermes.loop_guard"
TOOL_RECOVERY_MARK = "hermes.tool_recovery"
TERMINAL_OUTCOME_MARK = "hermes.terminal.outcome"
MODEL_REPLY_ISSUE_MARK = "hermes.model_reply_issue"
FILE_EDIT_METRIC = "hermes.file_edit.count"
LOOP_GUARD_METRIC = "hermes.loop_guard.count"
TOOL_RECOVERY_METRIC = "hermes.tool_recovery.count"
TERMINAL_OUTCOME_METRIC = "hermes.terminal.outcome.count"
MODEL_REPLY_ISSUE_METRIC = "hermes.model_reply_issue.count"
FILE_EDIT_TOOLS = frozenset({"patch", "write_file"})
FILE_EDIT_MODES = frozenset({"replace", "v4a", "whole_file"})
FILE_EDIT_OUTCOMES = frozenset({"already_applied", "ambiguous", "applied", "failed", "no_match"})


def _fuzzy_match_strategies() -> frozenset[str]:
    # The patch tool's own strategy chain, so a new strategy is a new bucket, never "none".
    from tools.fuzzy_match import STRATEGIES

    return frozenset(name for name, _fn in STRATEGIES)


FILE_EDIT_STRATEGIES = _fuzzy_match_strategies() | {"none"}
LOOP_GUARD_SIGNALS = frozenset({"iteration_cap", "loop_detected", "repeated_tool_call"})
LOOP_GUARD_DETECTORS = frozenset({
    "exact_failure", "idempotent_no_progress", "identical_call_streak", "identical_cycle", "iteration_budget",
    "same_tool_failure", "subagent_cap", "web_search_cap",
})
TOOL_RECOVERY_OUTCOMES = frozenset({"error", "gave_up", "no_tool_call", "success"})
TOOL_RECOVERY_NEXT_TOOLS = frozenset({"different", "none", "same"})
TERMINAL_COMMAND_KINDS = frozenset({
    "build", "container", "file_ops", "git", "network", "node", "other", "package_manager", "python",
    "shell", "shell_builtin", "test_runner",
})
TERMINAL_OUTCOMES = frozenset({"killed", "nonzero", "ok", "timeout"})
MODEL_REPLY_ISSUES = frozenset({"empty", "none", "reasoning_only", "refusal", "truncated_length"})
# ---- end v5 harness ----
# ---- v5 efficiency ----
TASK_COST_MARK = TASK_COST_METRIC = "hermes.task_cost.count"
WASTED_TOKENS_MARK = WASTED_TOKENS_METRIC = "hermes.wasted_tokens.count"
TOOL_OUTPUT_TRUNCATION_MARK = TOOL_OUTPUT_TRUNCATION_METRIC = "hermes.tool_output_truncation.count"
TOOL_OVERHEAD_MARK = TOOL_OVERHEAD_METRIC = "hermes.tool_overhead.count"
TOOL_ENABLED_UNUSED_MARK = TOOL_ENABLED_UNUSED_METRIC = "hermes.tool_enabled_unused.count"
CACHE_BREAK_MARK = CACHE_BREAK_METRIC = "hermes.cache_break.count"
# Prompt + completion tokens of one user turn, summed over its primary calls.
TURN_TOKEN_BUCKETS = frozenset({
    "lt_2k", "2k_to_10k", "10k_to_50k", "50k_to_200k", "200k_to_1m", "gte_1m", "unknown",
})
TASK_COST_OUTCOMES = frozenset({"completed", "failed", "interrupted"})
# Tool and API calls of one agentic turn: COUNT_BUCKETS stops at gte_11, far below real loops.
TURN_ACTIVITY_BUCKETS = frozenset({
    "0", "1", "2", "3_to_5", "6_to_10", "11_to_25", "26_to_50", "51_to_100", "gte_101",
})
WASTE_REASONS = frozenset({"interrupt", "retry", "undo"})
# Characters of one raw tool result, before Hermes spills or truncates it.
TOOL_OUTPUT_SIZE_BUCKETS = frozenset({
    "lt_1k", "1k_to_10k", "10k_to_50k", "50k_to_100k", "100k_to_500k", "gte_500k", "unknown",
})
TOOL_SCHEMA_TOKEN_BUCKETS = frozenset({
    "0", "lt_2k", "2k_to_5k", "5k_to_10k", "10k_to_20k", "20k_to_40k", "gte_40k",
})
# Toolsets Hermes ships; MCP servers and plugin toolsets collapse to custom.
TOOLSET_NAMES = BUILTIN_TOOLSET_NAMES | {"custom"}
# compression is the expected cause; the rest are Hermes invalidating a warm prefix (bugs or
# user-driven), or the provider reporting a cold read Hermes did not cause.
CACHE_BREAK_CAUSES = frozenset({
    "cache_expired", "compression", "model_switch", "provider_reported_miss", "system_prompt_rebuild",
    "toolset_change",
})
# ---- end v5 efficiency ----
# ---- v5 desktop ----
# Desktop love/hate: which areas get used, what gets in the way, where first run stops. Every value
# is a code-defined Desktop surface id (apps/desktop/src/store/desktop-metrics.ts mirrors these
# sets); anything the client names outside them collapses to `other` before it is recorded.
DESKTOP_FEATURE_USE_MARK = DESKTOP_FEATURE_USE_METRIC = "hermes.desktop.feature_use"
DESKTOP_FRICTION_MARK = DESKTOP_FRICTION_METRIC = "hermes.desktop.friction"
DESKTOP_ONBOARDING_MARK = DESKTOP_ONBOARDING_METRIC = "hermes.desktop.onboarding"
# Settings views are `SETTINGS_VIEWS` in apps/desktop/src/app/settings/index.tsx (the legacy
# `connections` tab redirects to `gateway`).
DESKTOP_SETTINGS_AREAS = frozenset({
    "settings_about", "settings_billing", "settings_config_advanced", "settings_config_appearance",
    "settings_config_browser", "settings_config_chat", "settings_config_memory", "settings_config_model",
    "settings_config_safety", "settings_config_voice", "settings_config_workspace", "settings_gateway",
    "settings_keybinds", "settings_keys", "settings_notifications", "settings_other", "settings_providers",
    "settings_sessions", "settings_vault",
})
DESKTOP_FEATURE_AREAS = DESKTOP_SETTINGS_AREAS | {
    # panes and panels
    "browser_pane", "file_pane", "review_pane", "terminal_pane",
    # overlays and pickers
    "command_palette", "find_in_page", "model_picker", "session_picker", "session_switcher",
    # first-class features
    "bot_mode", "projects", "session_search", "skins", "voice_conversation", "voice_dictation",
    # full pages (APP_ROUTES in apps/desktop/src/app/routes.ts, plus contributed plugin pages)
    "agents", "artifacts", "capabilities", "command_center", "cron", "extension_page", "kanban", "messaging",
    "profiles", "session_import", "starmap", "webhooks",
    "other",
}
# notice_dismissed: stable toast ids and dismissible strips the Desktop code defines.
DESKTOP_NOTICE_IDS = frozenset({
    "artifacts_partial_load", "background_queue_stuck", "backend_skew", "billing_banner", "billing_block",
    "build_discontinued", "client_behind", "composer_queue_stuck", "credits",
    "free_tier_notice", "gateway_error", "gui_skew", "install_method", "mcp_health", "model_warning",
    "onboarding_handoff", "restored_draft", "runtime_not_ready", "session_compress", "terminal_backend", "tip",
    "update_available", "voice_live_unavailable", "voice_stop_hint", "other",
})
# error_toast: the notifyError summary rule that matched (apps/desktop/src/store/notifications.ts).
DESKTOP_ERROR_TOAST_CATEGORIES = frozenset({
    "api_key_missing", "api_key_rejected", "disk_full", "gateway_auth_failed", "method_not_allowed",
    "microphone_permission", "pool_slot_timeout", "restart_required", "rpc_out_of_sync", "storage_failure",
    "timeout", "unclassified", "other",
})
DESKTOP_SLOW_FRAME_BUCKETS = frozenset({"100ms_to_250ms", "250ms_to_1s", "1s_to_5s", "gte_5s"})
DESKTOP_FRICTION_DETAILS: dict[str, frozenset[str]] = {
    "backend_disconnect": frozenset({"backend_exit", "network", "timeout", "other"}),
    "error_toast": DESKTOP_ERROR_TOAST_CATEGORIES,
    "notice_dismissed": DESKTOP_NOTICE_IDS,
    # Electron render-process-gone reasons: crashed -> crash; oom; killed; anything else -> other.
    "renderer_crash": frozenset({"crash", "killed", "oom", "other"}),
    "slow_frame": DESKTOP_SLOW_FRAME_BUCKETS,
}
DESKTOP_FRICTION_KINDS = frozenset(DESKTOP_FRICTION_DETAILS)
DESKTOP_FRICTION_DETAIL_VALUES = frozenset().union(*DESKTOP_FRICTION_DETAILS.values())
# The Desktop first-run flows: the classic provider overlay (store/onboarding.ts), the guided flow
# (store/onboarding-gate.ts phases), free-tier sign-in, then the consent answer and the first message.
DESKTOP_ONBOARDING_STEPS = frozenset({
    "choose_later", "consent", "first_message", "free_tier_ready", "guide", "guide_skip", "model_pick",
    "provider_api_key", "provider_local", "provider_oauth", "provider_setup", "sign_in",
})
DESKTOP_ONBOARDING_EVENTS = frozenset({"abandoned", "completed", "reached"})
# Bot Mode (a bot's canonical chat or a bot side-chat in front) vs regular Sessions mode, per day.
DESKTOP_MODE_USE_MARK = DESKTOP_MODE_USE_METRIC = "hermes.desktop.mode_use"
DESKTOP_MODES = frozenset({"bots", "sessions"})
DESKTOP_ACTIVE_MINUTES_BUCKETS = frozenset({"0", "lt_5m", "5m_to_30m", "30m_to_2h", "2h_to_6h", "gte_6h"})
_DESKTOP_ACTIVE_THRESHOLDS = ((300_000, "lt_5m"), (1_800_000, "5m_to_30m"), (7_200_000, "30m_to_2h"),
                              (21_600_000, "2h_to_6h"))


def desktop_active_minutes_bucket(active_ms: Any) -> str:
    """Bucket a day's active time in one Desktop mode; nothing measurable reads 0."""
    value = _non_negative_number(active_ms) or 0
    return "0" if value <= 0 else _bucket(value, _DESKTOP_ACTIVE_THRESHOLDS, "gte_6h")
# ---- end v5 desktop ----

# ---- v5 desktop actions ----
# Button presses per day. Action ids are the Desktop keybinding registry (KEYBIND_ACTIONS in
# apps/desktop/src/lib/keybinds/actions.ts) plus the typed button table DESKTOP_BUTTON_ACTIONS in
# apps/desktop/src/store/desktop-metrics.ts; contributed/plugin actions collapse to `other`.
DESKTOP_ACTION_USE_MARK = DESKTOP_ACTION_USE_METRIC = "hermes.desktop.action_use"
DESKTOP_KEYBIND_ACTION_IDS = frozenset({
    "appearance.toggleMode", "composer.cancel", "composer.dictate", "composer.focus", "composer.help",
    "composer.history", "composer.mention", "composer.modelPicker", "composer.newline", "composer.queue",
    "composer.reasoningDown", "composer.reasoningUp", "composer.send", "composer.sendQueued", "composer.slash",
    "composer.steer", "composer.voice", "conversation.scrollPageDown", "conversation.scrollPageUp",
    "hud.snapToPointer", "keybinds.openPanel", "nav.agents", "nav.artifacts", "nav.capabilities",
    "nav.commandCenter", "nav.commandPalette", "nav.cron", "nav.messaging", "nav.profiles", "nav.settings",
    "profile.create", "profile.default", "profile.next", "profile.prev", "profile.toggleAll",
    "session.archive", "session.focusSearch", "session.new", "session.newTab", "session.newWindow",
    "session.next", "session.prev", "session.togglePin", "view.closeTab", "view.closeTerminal",
    "view.cycleSidebarGrouping", "view.findInPage", "view.findNext", "view.findPrevious", "view.flipPanes",
    "view.newTerminal", "view.nextTerminal", "view.prevTerminal", "view.reopenTab", "view.selectionToComposer",
    "view.showBrowser", "view.showFiles", "view.showTerminal", "view.terminalCopy", "view.terminalPaste",
    "view.toggleHud", "view.toggleProfileRail", "view.toggleReview", "view.toggleRightSidebar",
    "view.toggleSidebar", "view.toggleSimpleMode", "view.toggleStatusbar", "view.toggleTabStrip",
    "workspace.newWorktree", "workspace.openFolder",
})
DESKTOP_BUTTON_ACTION_IDS = frozenset({"composer.attach", "message.copy", "message.retry"})
DESKTOP_ACTION_IDS = DESKTOP_KEYBIND_ACTION_IDS | DESKTOP_BUTTON_ACTION_IDS | {"other"}
DESKTOP_ACTION_VIAS = frozenset({"click", "menu", "palette", "shortcut"})
# Dislike signals, each naming its target from a closed set (setting keys are DEFAULT_CONFIG leaf
# paths, checked against the live schema by shared_metrics_desktop and shape-checked here).
DESKTOP_DISLIKE_MARK = DESKTOP_DISLIKE_METRIC = "hermes.desktop.dislike"
DESKTOP_FLOW_IDS = frozenset({
    "command_palette", "free_tier_sign_in", "keybind_capture", "model_picker", "project_create", "provider_oauth",
    "session_picker", "session_switcher", "other",
})
DESKTOP_FEATURE_TOGGLES = frozenset({
    "backdrop", "bot_activity_toasts", "composer_popout_gestures", "intro_splash", "native_notifications",
    "notification_kind", "reactions", "thread_timeline", "tips", "tours", "vibe_hearts", "other",
})
DESKTOP_UNDO_TARGETS = frozenset({"closed_tab", "restored_draft", "other"})
DESKTOP_DISLIKE_TARGETS: dict[str, frozenset[str]] = {
    "cancelled": DESKTOP_FLOW_IDS,
    "feature_disabled": DESKTOP_FEATURE_TOGGLES,
    "quick_close": DESKTOP_FEATURE_AREAS,
    "rage_click": DESKTOP_ACTION_IDS,
    "setting_off_default": frozenset({"setting"}),
    "undo": DESKTOP_UNDO_TARGETS,
}
DESKTOP_DISLIKE_SIGNALS = frozenset(DESKTOP_DISLIKE_TARGETS)
DESKTOP_DISLIKE_TARGET_VALUES = frozenset().union(*DESKTOP_DISLIKE_TARGETS.values())
DESKTOP_DISLIKE_DIRECTIONS = frozenset({"away_from_default", "none", "to_default"})
DESKTOP_SETTING_KEY_MAX_LENGTH = 96
# ---- end v5 desktop actions ----

_ARCHITECTURE_ALIASES = {
    "amd64": "x86_64", "x64": "x86_64", "x86_64": "x86_64",
    "aarch64": "arm64", "arm64": "arm64",
    "i386": "x86", "i486": "x86", "i586": "x86", "i686": "x86", "x86": "x86",
}
_OS_FAMILIES = {"darwin": "macos", "linux": "linux", "macos": "macos", "windows": "windows"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _allowlisted(normalized: str, allowed: frozenset[str]) -> str:
    return normalized if normalized in allowed else "unknown"


def client_os_family(value: Any) -> str:
    """Map a platform system name to the shared-metrics OS taxonomy."""
    return _OS_FAMILIES.get(_norm(value), "unknown")


def client_architecture(value: Any) -> str:
    """Map a machine architecture to the shared-metrics taxonomy."""
    normalized = _norm(value).replace("-", "_")
    if normalized in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[normalized]
    return "arm" if normalized.startswith("armv") else "unknown"


def client_install_method(value: Any) -> str:
    """Return an allowlisted Hermes installation method."""
    normalized = _norm(value)
    return _allowlisted("nixos" if normalized == "nix" else normalized, CLIENT_INSTALL_METHODS)


def client_resource(
    hermes_version: Any, *, os_name: Any, architecture: Any, install_method: Any
) -> dict[str, str]:
    """Build the bounded client resource attached to aggregate packages."""
    version = str(hermes_version or "").strip()
    return {
        "architecture": client_architecture(architecture),
        "hermes_version": version if 0 < len(version) <= 64 else "unknown",
        "install_method": client_install_method(install_method),
        "os_family": client_os_family(os_name),
    }


def client_resource_is_valid(resource: Any) -> bool:
    """Return whether a package resource exactly matches the bounded contract."""
    if not isinstance(resource, dict) or set(resource) != CLIENT_RESOURCE_KEYS:
        return False
    version = resource.get("hermes_version")
    return (
        isinstance(version, str)
        and 0 < len(version) <= 64
        and resource.get("os_family") in CLIENT_OS_FAMILIES
        and resource.get("architecture") in CLIENT_ARCHITECTURES
        and resource.get("install_method") in CLIENT_INSTALL_METHODS
    )


_LEGACY_PROVIDER_FAMILIES = frozenset({"aggregator", "custom", "direct", "local", "unknown"})
_LEGACY_MODEL_LOCALITIES = frozenset({"local", "remote", "unknown"})
_LEGACY_MODEL_OUTCOMES = frozenset({"cancelled", "failed", "success"})
_LEGACY_MODEL_FAMILIES = frozenset({
    "claude", "deepseek", "gemini", "gemma", "glm", "gpt", "grok", "kimi", "llama", "minimax",
    "mimo", "mistral", "nemotron", "nova", "o1", "o3", "o4", "qwen", "step", "trinity",
    "unknown",
})

_COUNTER_DIMENSION_VALUES: dict[str, dict[str, frozenset[str]]] = {
    CLIENT_ACTIVE_METRIC: {},
    # Retained only so pre-v2 pending rows remain packageable.
    LEGACY_MODEL_CALL_METRIC: {
        "call_role": frozenset({"primary"}), "locality": _LEGACY_MODEL_LOCALITIES,
        "model_family": _LEGACY_MODEL_FAMILIES, "outcome": _LEGACY_MODEL_OUTCOMES,
        "provider_family": _LEGACY_PROVIDER_FAMILIES,
    },
    TASK_STARTED_METRIC: {
        "entrypoint": TASK_ENTRYPOINTS, "execution_surface": EXECUTION_SURFACES,
        "platform": GATEWAY_PLATFORMS,
    },
    TASK_FINISHED_METRIC: {
        "duration_bucket": DURATION_BUCKETS, "end_reason": TASK_END_REASONS,
        "entrypoint": TASK_ENTRYPOINTS, "execution_surface": EXECUTION_SURFACES,
        "failure_class": TASK_FAILURE_CLASSES,
        "model_call_count_bucket": COUNT_BUCKETS, "outcome": TASK_OUTCOMES,
        "platform": GATEWAY_PLATFORMS,
        "retry_count_bucket": COUNT_BUCKETS, "termination": TASK_TERMINATIONS,
        "tool_call_count_bucket": COUNT_BUCKETS,
    },
    MODEL_ROUTE_METRIC: {
        "call_role": MODEL_CALL_ROLES, "error_class": MODEL_ERROR_CLASSES, "outcome": MODEL_OUTCOMES,
        "ttft_bucket": TTFT_BUCKETS,
    },
    TOOL_CALL_METRIC: {
        "approval_outcome": TOOL_APPROVAL_OUTCOMES, "latency_bucket": TOOL_LATENCY_BUCKETS,
        "outcome": TOOL_OUTCOMES, "retry_count_bucket": TOOL_RETRY_BUCKETS,
        "tool_category": TOOL_CATEGORIES,
    },
    TOOL_LATENCY_METRIC: {
        "latency_bucket": TOOL_LATENCY_BUCKETS, "retry_count_bucket": TOOL_RETRY_BUCKETS,
        "tool_category": TOOL_CATEGORIES,
    },
    TASK_DURATION_METRIC: {
        "duration_bucket": DURATION_BUCKETS, "execution_surface": EXECUTION_SURFACES,
        "outcome": TASK_OUTCOMES, "retry_count_bucket": COUNT_BUCKETS,
    },
    TOOL_USAGE_METRIC: {
        "error_class": TOOL_ERROR_CLASSES, "outcome": TOOL_OUTCOMES, "tool_name": TOOL_NAMES,
    },
    TOOL_APPROVAL_METRIC: {
        "attribution": TOOL_APPROVAL_ATTRIBUTIONS,
        "outcome": TOOL_APPROVAL_OUTCOMES - {"not_required"},
    },
    SKILL_LIFECYCLE_METRIC: {"action": SKILL_LIFECYCLE_ACTIONS, "provenance": SKILL_PROVENANCES},
    SKILL_LOAD_METRIC: {
        "post_patch_state": SKILL_POST_PATCH_STATES, "provenance": SKILL_PROVENANCES,
        "reuse_state": SKILL_REUSE_STATES, "skill_name": SKILL_NAMES, "use_count_bucket": COUNT_BUCKETS,
    },
    INSTALL_SNAPSHOT_METRIC: {
        "cron_job_count_bucket": SIZE_BUCKETS, "display_language": DISPLAY_LANGUAGES,
        "install_age_bucket": INSTALL_AGE_BUCKETS, "mcp_server_count_bucket": SIZE_BUCKETS,
        "memory_provider": MEMORY_PROVIDERS, "messaging_platform_count_bucket": SIZE_BUCKETS,
        "plugin_count_bucket": SIZE_BUCKETS, "profile_count_bucket": SIZE_BUCKETS,
        "skill_count_bucket": SIZE_BUCKETS, "terminal_backend": TERMINAL_BACKENDS,
        **_INSTALL_V4_SNAPSHOT_DIMENSIONS,
    },
    SESSION_METRIC: {
        "active_duration_bucket": SESSION_DURATION_BUCKETS, "entrypoint": TASK_ENTRYPOINTS,
        "execution_surface": EXECUTION_SURFACES, "failed_turn_count_bucket": COUNT_BUCKETS,
        "last_outcome": TASK_OUTCOMES, "platform": GATEWAY_PLATFORMS, "turn_count_bucket": SIZE_BUCKETS,
    },
    MILESTONE_METRIC: {"install_age_bucket": INSTALL_AGE_BUCKETS, "milestone": MILESTONES},
    SETUP_COMPLETED_METRIC: {"surface": SETUP_SURFACES},
    MODEL_TOKENS_METRIC: {"aux_task": AUX_TASKS, "call_role": MODEL_CALL_ROLES, "token_type": TOKEN_TYPES},
    COMPRESSION_METRIC: {
        "context_fill_bucket": CONTEXT_FILL_BUCKETS, "failure_class": COMPRESSION_FAILURE_CLASSES,
        "outcome": COMPRESSION_OUTCOMES, "trigger": COMPRESSION_TRIGGERS,
    },
    MODEL_SWITCH_METRIC: {"execution_surface": EXECUTION_SURFACES},
    FALLBACK_METRIC: {"error_class": MODEL_ERROR_CLASSES},
    SLASH_COMMAND_METRIC: {"command": SLASH_COMMANDS, "execution_surface": EXECUTION_SURFACES},
    EXTENSION_INSTALL_METRIC: {
        "kind": EXTENSION_KINDS, "name": EXTENSION_NAMES, "outcome": EXTENSION_OUTCOMES,
        "source": EXTENSION_SOURCES,
        # ---- iuf c1 ----
        "failure_class": EXTENSION_FAILURE_CLASSES, "registry": EXTENSION_REGISTRIES,
        # ---- end iuf c1 ----
    },
    # ---- v4 loop ----
    MEMORY_OP_METRIC: {
        "failure_class": MEMORY_OP_FAILURE_CLASSES, "op": MEMORY_OPS, "origin": MEMORY_OP_ORIGINS,
        "outcome": MEMORY_OP_OUTCOMES, "provider": MEMORY_PROVIDERS,
    },
    CURATOR_RUN_METRIC: {
        "archived_bucket": COUNT_BUCKETS, "created_bucket": COUNT_BUCKETS, "merged_bucket": COUNT_BUCKETS,
        "outcome": CURATOR_OUTCOMES, "patched_bucket": COUNT_BUCKETS, "trigger": CURATOR_TRIGGERS,
    },
    DELEGATION_RUN_METRIC: {
        "depth": DELEGATION_DEPTHS, "mode": DELEGATION_MODES, "outcome": DELEGATION_OUTCOMES,
        "subagent_count_bucket": COUNT_BUCKETS,
    },
    EXECUTION_BACKEND_METRIC: {
        "backend": EXECUTION_BACKENDS, "error_class": TOOL_ERROR_CLASSES, "kind": EXECUTION_KINDS,
        "outcome": EXECUTION_OUTCOMES,
    },
    # ---- end v4 loop ----
    # ---- v4 gateway ----
    PLATFORM_HEALTH_METRIC: {
        "error_class": PLATFORM_ERROR_CLASSES, "event": PLATFORM_HEALTH_EVENTS, "platform": ADAPTER_PLATFORMS,
    },
    PLATFORM_DELIVERY_METRIC: {
        "failure_class": DELIVERY_FAILURE_CLASSES, "outcome": DELIVERY_OUTCOMES, "platform": ADAPTER_PLATFORMS,
    },
    REPLY_LATENCY_METRIC: {"first_response_bucket": REPLY_LATENCY_BUCKETS, "platform": ADAPTER_PLATFORMS},
    CRON_RUN_METRIC: {
        "delivery_kind": CRON_DELIVERY_KINDS, "duration_bucket": DURATION_BUCKETS, "outcome": CRON_RUN_OUTCOMES,
    },
    # ---- v4 model ----
    MODEL_TOOL_QUALITY_METRIC: {"call_role": MODEL_CALL_ROLES, "issue": TOOL_CALL_ISSUES},
    MODEL_FRICTION_METRIC: {"signal": FRICTION_SIGNALS},
    CONTEXT_PEAK_METRIC: {
        "limit_hit": LIMIT_HIT_VALUES, "peak_fill_bucket": CONTEXT_FILL_BUCKETS,
        "window_bucket": CONTEXT_WINDOW_BUCKETS,
    },
    # ---- end v4 model ----
    # ---- v4 install ----
    STARTUP_LATENCY_METRIC: {"latency_bucket": STARTUP_LATENCY_BUCKETS, "surface": STARTUP_SURFACES},
    # ---- v4 reliability ----
    UPDATE_RUN_METRIC: {
        "apply_mode": UPDATE_APPLY_MODES, "duration_bucket": UPDATE_DURATION_BUCKETS,
        "failed_stage": UPDATE_FAILED_STAGES, "from_version_age_bucket": VERSION_AGE_BUCKETS,
        "kind": UPDATE_KINDS, "outcome": UPDATE_RUN_OUTCOMES,
        # ---- iuf c1 ----
        "failure_class": UPDATE_FAILURE_CLASSES,
        # ---- end iuf c1 ----
    },
    UPDATE_STAGE_METRIC: {
        "duration_bucket": UPDATE_DURATION_BUCKETS, "outcome": UPDATE_STAGE_OUTCOMES, "stage": UPDATE_STAGES,
    },
    PROCESS_EXIT_METRIC: {
        "crash_class": CRASH_CLASSES, "exit_kind": PROCESS_EXIT_KINDS, "process_kind": PROCESS_KINDS,
    },
    # ---- end v4 reliability ----
    # ---- v5 harness ----
    FILE_EDIT_METRIC: {
        "match_strategy": FILE_EDIT_STRATEGIES, "mode": FILE_EDIT_MODES, "outcome": FILE_EDIT_OUTCOMES,
        "tool": FILE_EDIT_TOOLS,
    },
    LOOP_GUARD_METRIC: {"detector": LOOP_GUARD_DETECTORS, "signal": LOOP_GUARD_SIGNALS},
    TOOL_RECOVERY_METRIC: {
        "next_outcome": TOOL_RECOVERY_OUTCOMES, "next_tool": TOOL_RECOVERY_NEXT_TOOLS, "tool": TOOL_NAMES,
    },
    TERMINAL_OUTCOME_METRIC: {
        "backend": TERMINAL_BACKENDS, "command_kind": TERMINAL_COMMAND_KINDS, "outcome": TERMINAL_OUTCOMES,
    },
    MODEL_REPLY_ISSUE_METRIC: {"issue": MODEL_REPLY_ISSUES},
    # ---- end v5 harness ----
    # ---- v5 efficiency ----
    TASK_COST_METRIC: {
        "api_calls_bucket": TURN_ACTIVITY_BUCKETS, "outcome": TASK_COST_OUTCOMES,
        "tokens_bucket": TURN_TOKEN_BUCKETS, "tool_calls_bucket": TURN_ACTIVITY_BUCKETS,
    },
    WASTED_TOKENS_METRIC: {"reason": WASTE_REASONS, "tokens_bucket": TURN_TOKEN_BUCKETS},
    TOOL_OUTPUT_TRUNCATION_METRIC: {
        "original_size_bucket": TOOL_OUTPUT_SIZE_BUCKETS, "tool": TOOL_NAMES, "truncated": YES_NO,
    },
    TOOL_OVERHEAD_METRIC: {
        "enabled_tool_count_bucket": SIZE_BUCKETS, "execution_surface": EXECUTION_SURFACES,
        "tool_schema_tokens_bucket": TOOL_SCHEMA_TOKEN_BUCKETS,
    },
    TOOL_ENABLED_UNUSED_METRIC: {"toolset": TOOLSET_NAMES, "used": YES_NO},
    CACHE_BREAK_METRIC: {"cause": CACHE_BREAK_CAUSES},
    # ---- end v5 efficiency ----
    # ---- v5 engagement ----
    ENGAGEMENT_DAY_METRIC: {
        "active_minutes_bucket": ACTIVE_MINUTES_BUCKETS, "active_profile_count_bucket": SIZE_BUCKETS,
        "surfaces_used_count": SURFACES_USED_BUCKETS,
    },
    ENGAGEMENT_SURFACE_METRIC: {"active_minutes_bucket": ACTIVE_MINUTES_BUCKETS, "surface": ENGAGEMENT_SURFACES},
    MODEL_SWITCH_AFTER_METRIC: {"turns_before_switch_bucket": TURNS_BEFORE_SWITCH_BUCKETS},
    # ---- end v5 engagement ----
    # ---- v5 desktop ----
    DESKTOP_FEATURE_USE_METRIC: {"area": DESKTOP_FEATURE_AREAS},
    DESKTOP_FRICTION_METRIC: {"detail": DESKTOP_FRICTION_DETAIL_VALUES, "kind": DESKTOP_FRICTION_KINDS},
    DESKTOP_ONBOARDING_METRIC: {"event": DESKTOP_ONBOARDING_EVENTS, "step": DESKTOP_ONBOARDING_STEPS},
    DESKTOP_MODE_USE_METRIC: {
        "active_minutes_bucket": DESKTOP_ACTIVE_MINUTES_BUCKETS, "bot_count_bucket": SIZE_BUCKETS,
        "messages_sent_bucket": SIZE_BUCKETS, "mode": DESKTOP_MODES,
    },
    DESKTOP_ACTION_USE_METRIC: {"action": DESKTOP_ACTION_IDS, "count_bucket": SIZE_BUCKETS, "via": DESKTOP_ACTION_VIAS},
    DESKTOP_DISLIKE_METRIC: {
        "direction": DESKTOP_DISLIKE_DIRECTIONS, "signal": DESKTOP_DISLIKE_SIGNALS, "target": DESKTOP_DISLIKE_TARGET_VALUES,
    },
    # ---- end v5 desktop ----
    # ---- v5 signals ----
    TOOL_UNAVAILABLE_METRIC: {"tool_name": BUILTIN_TOOL_NAMES},
    PROVIDER_SETUP_METRIC: {
        "event": PROVIDER_SETUP_EVENTS, "failure_class": PROVIDER_SETUP_FAILURE_CLASSES,
        "surface": PROVIDER_SETUP_SURFACES,
    },
    FEATURE_ADOPTION_METRIC: {"days_since_install_bucket": DAYS_SINCE_INSTALL_BUCKETS, "feature": FEATURES},
    FEATURE_DISABLED_METRIC: {
        "event": FEATURE_DISABLED_EVENTS, "kind": FEATURE_DISABLED_KINDS, "surface": FEATURE_DISABLED_SURFACES,
    },
    # ---- end v5 signals ----
    # ---- iuf c2 ----
    INSTALL_RUN_METRIC: {
        "duration_bucket": UPDATE_DURATION_BUCKETS, "failed_stage": INSTALL_RUN_FAILED_STAGES,
        "failure_class": INSTALL_RUN_FAILURE_CLASSES, "installer": INSTALL_RUN_INSTALLERS,
        "outcome": INSTALL_RUN_OUTCOMES,
    },
    # ---- end iuf c2 ----
}
_MODEL_ROUTE_MAX_LENGTHS = {
    "model": MODEL_IDENTIFIER_MAX_LENGTH, "provider": PROVIDER_IDENTIFIER_MAX_LENGTH,
}
# metric -> {field: max length} for provider/model identifiers, validated by shape (no catalog).
_IDENTIFIER_FIELDS: dict[str, dict[str, int]] = {
    MODEL_ROUTE_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    MODEL_TOKENS_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    SETUP_COMPLETED_METRIC: {"provider": PROVIDER_IDENTIFIER_MAX_LENGTH},
    MODEL_SWITCH_METRIC: dict.fromkeys(("from_provider", "to_provider"), PROVIDER_IDENTIFIER_MAX_LENGTH),
    FALLBACK_METRIC: dict.fromkeys(("from_provider", "to_provider"), PROVIDER_IDENTIFIER_MAX_LENGTH),
    INSTALL_SNAPSHOT_METRIC: {"main_provider": PROVIDER_IDENTIFIER_MAX_LENGTH},
    # ---- v4 model ----
    MODEL_TOOL_QUALITY_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    MODEL_FRICTION_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    CONTEXT_PEAK_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    # ---- end v4 model ----
    # ---- v5 harness ----
    LOOP_GUARD_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    TOOL_RECOVERY_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    MODEL_REPLY_ISSUE_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    # ---- end v5 harness ----
    # ---- v5 efficiency ----
    TASK_COST_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    WASTED_TOKENS_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    CACHE_BREAK_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    # ---- end v5 efficiency ----
    # ---- v5 engagement ----
    ENGAGEMENT_DAY_METRIC: {
        "primary_model": MODEL_IDENTIFIER_MAX_LENGTH, "primary_provider": PROVIDER_IDENTIFIER_MAX_LENGTH,
    },
    MODEL_SWITCH_AFTER_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    # ---- end v5 engagement ----
    # ---- v5 desktop ----
    DESKTOP_DISLIKE_METRIC: {"setting": DESKTOP_SETTING_KEY_MAX_LENGTH},
    # ---- end v5 desktop ----
    # ---- v5 signals ----
    TOOL_UNAVAILABLE_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    PROVIDER_SETUP_METRIC: {"provider": PROVIDER_IDENTIFIER_MAX_LENGTH},
    FEATURE_DISABLED_METRIC: {"name": FEATURE_DISABLED_NAME_MAX_LENGTH},
    # ---- end v5 signals ----
}
# ---- v5 engagement ----
# Conversation volume on the session row: new fields, so rows recorded before them still package.
SESSION_VOLUME_DIMENSIONS = dict.fromkeys(
    ("message_count_bucket", "model_call_count_bucket", "tool_call_count_bucket"), LONG_SIZE_BUCKETS,
)
_COUNTER_DIMENSION_VALUES[SESSION_METRIC] = {**_COUNTER_DIMENSION_VALUES[SESSION_METRIC], **SESSION_VOLUME_DIMENSIONS}
# ---- end v5 engagement ----
# metric -> closed dimension field set
_METRIC_FIELDS: dict[str, frozenset[str]] = {
    name: frozenset(contract) | frozenset(_IDENTIFIER_FIELDS.get(name, ()))
    for name, contract in _COUNTER_DIMENSION_VALUES.items()
}
# Fields v2 carried on the per-task and per-tool-call rows. Together they made nearly every task and
# tool call its own row; v3 moves duration/retries/latency to the small split counters above (call
# counts per task are already on hermes.task_cost.count). Their values stay in the contract so rows
# recorded before an upgrade still validate and package.
_TASK_FINISHED_V2_ONLY = frozenset(
    {"duration_bucket", "model_call_count_bucket", "retry_count_bucket", "tool_call_count_bucket"}
)
_TOOL_CALL_V2_ONLY = frozenset({"latency_bucket", "retry_count_bucket"})
_METRIC_FIELDS[TASK_FINISHED_METRIC] -= _TASK_FINISHED_V2_ONLY
_METRIC_FIELDS[TOOL_CALL_METRIC] -= _TOOL_CALL_V2_ONLY
# Older field sets still accepted at packaging so counters recorded before an upgrade drain.
_LEGACY_METRIC_FIELDS: dict[str, tuple[frozenset[str], ...]] = {
    MODEL_ROUTE_METRIC: (
        frozenset(_MODEL_ROUTE_MAX_LENGTHS), _METRIC_FIELDS[MODEL_ROUTE_METRIC] - {"ttft_bucket"},
    ),
    TASK_STARTED_METRIC: (_METRIC_FIELDS[TASK_STARTED_METRIC] - {"platform"},),
    TASK_FINISHED_METRIC: (frozenset(_COUNTER_DIMENSION_VALUES[TASK_FINISHED_METRIC]) - {"failure_class", "platform"},),
    TOOL_CALL_METRIC: (frozenset(_COUNTER_DIMENSION_VALUES[TOOL_CALL_METRIC]),),
    # v3 rows, recorded before the failure_class split, drain as they were counted.
    COMPRESSION_METRIC: (_METRIC_FIELDS[COMPRESSION_METRIC] - {"failure_class"},),
    MEMORY_OP_METRIC: (_METRIC_FIELDS[MEMORY_OP_METRIC] - {"failure_class"},),
    SKILL_LOAD_METRIC: (_METRIC_FIELDS[SKILL_LOAD_METRIC] - {"skill_name"},),
    INSTALL_SNAPSHOT_METRIC: (_METRIC_FIELDS[INSTALL_SNAPSHOT_METRIC] - {
        "display_language", "install_age_bucket", "main_provider", "messaging_platform_count_bucket",
        "terminal_backend", *_INSTALL_V4_SNAPSHOT_DIMENSIONS,
    },
        # ---- v4 install ----
        _METRIC_FIELDS[INSTALL_SNAPSHOT_METRIC] - set(_INSTALL_V4_SNAPSHOT_DIMENSIONS),
    ),
}
# ---- v5 engagement ----
_LEGACY_METRIC_FIELDS[SESSION_METRIC] = (_METRIC_FIELDS[SESSION_METRIC] - set(SESSION_VOLUME_DIMENSIONS),)
# ---- end v5 engagement ----
# ---- iuf c1 ----
# Rows recorded before the failure_class / registry split drain as they were counted.
_LEGACY_METRIC_FIELDS[EXTENSION_INSTALL_METRIC] = (
    _METRIC_FIELDS[EXTENSION_INSTALL_METRIC] - {"failure_class", "registry"},
)
_LEGACY_METRIC_FIELDS[UPDATE_RUN_METRIC] = (_METRIC_FIELDS[UPDATE_RUN_METRIC] - {"failure_class"},)
# ---- end iuf c1 ----
COUNTER_METRICS = frozenset(_METRIC_FIELDS) - {LEGACY_MODEL_CALL_METRIC}
# Counters whose value is a summed quantity rather than an event count.
SUM_METRICS = frozenset({MODEL_TOKENS_METRIC})
_SKILL_MARK_METRICS = {
    SKILL_LIFECYCLE_MARK: SKILL_LIFECYCLE_METRIC, SKILL_LOAD_MARK: SKILL_LOAD_METRIC,
}
# Marks projected one-to-one onto a counter of the same contract.
_DECISION_MARK_METRICS = {
    SESSION_MARK: SESSION_METRIC, SETUP_COMPLETED_MARK: SETUP_COMPLETED_METRIC,
    COMPRESSION_MARK: COMPRESSION_METRIC, MODEL_SWITCH_MARK: MODEL_SWITCH_METRIC,
    FALLBACK_MARK: FALLBACK_METRIC, SLASH_COMMAND_MARK: SLASH_COMMAND_METRIC,
    EXTENSION_INSTALL_MARK: EXTENSION_INSTALL_METRIC,
    # ---- v4 loop ----
    MEMORY_OP_MARK: MEMORY_OP_METRIC, CURATOR_RUN_MARK: CURATOR_RUN_METRIC,
    DELEGATION_RUN_MARK: DELEGATION_RUN_METRIC, EXECUTION_BACKEND_MARK: EXECUTION_BACKEND_METRIC,
    # ---- end v4 loop ----
    # ---- v4 gateway ----
    PLATFORM_HEALTH_MARK: PLATFORM_HEALTH_METRIC, PLATFORM_DELIVERY_MARK: PLATFORM_DELIVERY_METRIC,
    REPLY_LATENCY_MARK: REPLY_LATENCY_METRIC, CRON_RUN_MARK: CRON_RUN_METRIC,
    # ---- v4 model ----
    MODEL_TOOL_QUALITY_MARK: MODEL_TOOL_QUALITY_METRIC, MODEL_FRICTION_MARK: MODEL_FRICTION_METRIC,
    CONTEXT_PEAK_MARK: CONTEXT_PEAK_METRIC,
    # ---- end v4 model ----
    # ---- v4 install ----
    STARTUP_LATENCY_MARK: STARTUP_LATENCY_METRIC,
    # ---- v4 reliability ----
    UPDATE_RUN_MARK: UPDATE_RUN_METRIC, UPDATE_STAGE_MARK: UPDATE_STAGE_METRIC,
    PROCESS_EXIT_MARK: PROCESS_EXIT_METRIC,
    # ---- end v4 reliability ----
    # ---- v5 harness ----
    FILE_EDIT_MARK: FILE_EDIT_METRIC, LOOP_GUARD_MARK: LOOP_GUARD_METRIC,
    TOOL_RECOVERY_MARK: TOOL_RECOVERY_METRIC, TERMINAL_OUTCOME_MARK: TERMINAL_OUTCOME_METRIC,
    MODEL_REPLY_ISSUE_MARK: MODEL_REPLY_ISSUE_METRIC,
    # ---- end v5 harness ----
    # ---- v5 efficiency ----
    TASK_COST_MARK: TASK_COST_METRIC, WASTED_TOKENS_MARK: WASTED_TOKENS_METRIC,
    TOOL_OUTPUT_TRUNCATION_MARK: TOOL_OUTPUT_TRUNCATION_METRIC, TOOL_OVERHEAD_MARK: TOOL_OVERHEAD_METRIC,
    TOOL_ENABLED_UNUSED_MARK: TOOL_ENABLED_UNUSED_METRIC, CACHE_BREAK_MARK: CACHE_BREAK_METRIC,
    # ---- end v5 efficiency ----
    # ---- v5 engagement ----
    MODEL_SWITCH_AFTER_MARK: MODEL_SWITCH_AFTER_METRIC,
    # ---- end v5 engagement ----
    # ---- v5 desktop ----
    DESKTOP_FEATURE_USE_MARK: DESKTOP_FEATURE_USE_METRIC, DESKTOP_FRICTION_MARK: DESKTOP_FRICTION_METRIC,
    DESKTOP_ONBOARDING_MARK: DESKTOP_ONBOARDING_METRIC, DESKTOP_MODE_USE_MARK: DESKTOP_MODE_USE_METRIC,
    DESKTOP_ACTION_USE_MARK: DESKTOP_ACTION_USE_METRIC, DESKTOP_DISLIKE_MARK: DESKTOP_DISLIKE_METRIC,
    # ---- end v5 desktop ----
    # ---- v5 signals ----
    TOOL_UNAVAILABLE_MARK: TOOL_UNAVAILABLE_METRIC, PROVIDER_SETUP_MARK: PROVIDER_SETUP_METRIC,
    FEATURE_DISABLED_MARK: FEATURE_DISABLED_METRIC,
    # ---- end v5 signals ----
    # ---- iuf c2 ----
    INSTALL_RUN_MARK: INSTALL_RUN_METRIC,
    # ---- end iuf c2 ----
}


def counter_dimensions_are_valid(metric_name: str, dimensions: dict[str, Any]) -> bool:
    """Return whether dimensions match one closed shared-metric contract (current or legacy)."""
    fields = frozenset(dimensions)
    if fields != _METRIC_FIELDS.get(metric_name) and fields not in _LEGACY_METRIC_FIELDS.get(metric_name, ()):
        return False
    identifiers = _IDENTIFIER_FIELDS.get(metric_name, {})
    contract = _COUNTER_DIMENSION_VALUES[metric_name]
    return all(
        isinstance(value := dimensions[field], str) and (
            value == _metric_identifier(value, max_length=identifiers[field]) if field in identifiers
            else value in contract[field]
        )
        for field in fields
    ) and _extension_install_fields_agree(metric_name, dimensions)


# ---- iuf c1 ----
def _extension_install_fields_agree(metric_name: str, dimensions: dict[str, Any]) -> bool:
    """Cross-field rule the per-field enums cannot state: an install row's ``failure_class`` is in
    its kind's set and is ``none`` exactly on success; only skill rows name a registry."""
    if metric_name != EXTENSION_INSTALL_METRIC or "failure_class" not in dimensions:
        return True  # other metrics, and the pre-split shape without failure_class/registry
    kind, failure_class = dimensions["kind"], dimensions["failure_class"]
    return (
        failure_class in EXTENSION_KIND_FAILURE_CLASSES[kind]
        and (failure_class == "none") == (dimensions["outcome"] == "success")
        and (kind == "skill" or dimensions["registry"] == "none")
    )
# ---- end iuf c1 ----


def _relay_metadata(
    event: Any, schema_key: str, schema_version: str, *extra_keys: str
) -> dict | None:
    """Return the event metadata when it carries only the allowlisted Relay keys."""
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get(schema_key) != schema_version:
        return None
    allowed = {schema_key, RUNTIME_INSTANCE_KEY, COMMIT_TICKET_KEY, "otel.status_code", *extra_keys}
    if set(metadata) - allowed or metadata.get("otel.status_code", "OK") not in {"OK", "ERROR"}:
        return None
    return metadata


def _event_text(event: Any, attr: str) -> str:
    return str(getattr(event, attr, "") or "")


def _event_shape_matches(event: Any, **expected: Any) -> bool:
    """Match the coarse Relay event shape (``kind`` plus any of name/category/scope_category/
    category_profile).

    A ``str`` expectation compares against the stringified attribute; ``None`` requires the
    attribute itself to be ``None``; anything else (the ``category_profile`` dict) compares
    with plain equality. Unmentioned attributes are not checked.
    """
    for attr, value in expected.items():
        actual = _event_text(event, attr) if isinstance(value, str) else getattr(event, attr, None)
        if actual != value:
            return False
    return True


def _bounded_dimensions(metric_name: str, data: Any) -> dict[str, str] | None:
    """Project ``data`` onto the metric's closed field set, or None when it does not fit."""
    expected_fields = _METRIC_FIELDS[metric_name]
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    valid = counter_dimensions_are_valid(metric_name, dimensions) and _catalog_holds(metric_name, dimensions)
    return dimensions if valid else None


def _catalog_holds(metric_name: str, dimensions: dict[str, str]) -> bool:
    """Re-run the producer's catalog pass on a mark's provider/model fields: an identifier it would
    rewrite (a user-named provider, a local model id) never reaches a counter. Record-time only, so
    catalog drift can never block packaging rows already stored. ``none`` is the unset-provider value."""
    from .shared_metrics_catalog import model_metric_name, provider_metric_name

    for field, max_length in _IDENTIFIER_FIELDS.get(metric_name, {}).items():
        value = dimensions[field]
        if field.endswith("provider"):
            expected = value if value == "none" else provider_metric_name(value)
        elif field.endswith("model"):
            expected = model_metric_name(value, dimensions[field[:-5] + "provider"], max_length=max_length)
        else:
            continue
        if value != expected:
            return False
    return True


def _valid_shape(event: Any, **shape: Any) -> bool:
    """Metadata allowlist check plus :func:`_event_shape_matches` in one step."""
    return (
        _relay_metadata(event, SCHEMA_KEY, SCHEMA_VERSION) is not None
        and _event_shape_matches(event, **shape)
    )


def _bounded_counter(metric_name: str | None, event: Any) -> tuple[str, dict[str, str]] | None:
    if metric_name is None:
        return None
    dimensions = _bounded_dimensions(metric_name, getattr(event, "data", None))
    return None if dimensions is None else (metric_name, dimensions)


def _scoped_dimensions(event: Any, metric_name: str, **shape: Any) -> dict[str, str] | None:
    """Bounded ``event.data`` for a scope *end* event of the given shape, else None."""
    if not _valid_shape(event, kind="scope", scope_category="end", **shape):
        return None
    return _bounded_dimensions(metric_name, getattr(event, "data", None))


_MARK_SHAPE = dict(kind="mark", category=None, scope_category=None, category_profile=None)


def _mark_counter(event: Any, metrics_by_mark: dict[str, str]) -> tuple[str, dict[str, str]] | None:
    """Return the bounded counter for a safe Relay mark whose name is in *metrics_by_mark*."""
    if not _valid_shape(event, **_MARK_SHAPE):
        return None
    return _bounded_counter(metrics_by_mark.get(_event_text(event, "name")), event)


def client_active_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the active-install counter for one empty allowlisted mark."""
    return _mark_counter(event, {CLIENT_ACTIVE_MARK: CLIENT_ACTIVE_METRIC})


def model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one valid logical model-call end event."""
    auxiliary = _auxiliary_model_call_dimensions(event)
    if auxiliary is not None:
        return auxiliary
    # The synthetic scope can span provider fallback. The accepted terminal
    # route is carried in the validated payload rather than this start profile.
    return _scoped_dimensions(
        event, MODEL_ROUTE_METRIC, category="llm", name=MODEL_CALL_SCOPE,
        category_profile={"model_name": MODEL_CALL_PROFILE_MODEL},
    )


def _auxiliary_model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Project a terminal auxiliary route from its Hermes logical scope."""
    metadata = _relay_metadata(
        event, RUNTIME_SCHEMA_KEY, RUNTIME_SCHEMA_VERSION, "hermes.call_role"
    )
    call_role = (metadata or {}).get("hermes.call_role")
    data = getattr(event, "data", None)
    if (
        not isinstance(call_role, str)
        or not call_role.startswith("auxiliary:")
        or not _event_shape_matches(
            event, kind="scope", category="function", name=LOGICAL_LLM_SCOPE,
            scope_category="end", category_profile=None,
        )
        or not isinstance(data, dict)
        or set(data) - {"response_model", "error_class"} != {"model", "outcome", "provider"}
        or data.get("outcome") not in _LEGACY_MODEL_OUTCOMES
    ):
        return None
    outcome = data["outcome"]
    # Same reading as a primary call: a cancelled call reports ``none``, a failure without a
    # classified reason ``unknown``, a success the last error it recovered from.
    error_class = "none" if outcome == "cancelled" else data.get("error_class") or "none"
    if outcome == "failed" and error_class == "none":
        error_class = "unknown"
    dimensions = model_route_fields(data, call_role="auxiliary", outcome=outcome, error_class=error_class)
    return dimensions if counter_dimensions_are_valid(MODEL_ROUTE_METRIC, dimensions) else None


def task_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated task counter from a task scope event."""
    if not _valid_shape(
        event, kind="scope", category="function", name=TASK_SCOPE, category_profile=None
    ):
        return None
    phase = _event_text(event, "scope_category")
    if phase == "start":
        return _bounded_counter(TASK_STARTED_METRIC, event)
    return _task_end_counter(event, TASK_FINISHED_METRIC) if phase == "end" else None


def task_duration_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the duration/retry counter carried by the same task end event."""
    if not _valid_shape(
        event, kind="scope", scope_category="end", category="function", name=TASK_SCOPE,
        category_profile=None,
    ):
        return None
    return _task_end_counter(event, TASK_DURATION_METRIC)


# The task end payload (task_terminal_fields) also carries per-task call counts for other Relay
# subscribers; hermes.task_cost.count already buckets them, so no counter projects them here.
_TASK_END_FIELDS = (
    _METRIC_FIELDS[TASK_FINISHED_METRIC] | _METRIC_FIELDS[TASK_DURATION_METRIC]
    | {"model_call_count_bucket", "tool_call_count_bucket"}
)


def _task_end_counter(event: Any, metric_name: str) -> tuple[str, dict[str, str]] | None:
    """Project the task end event onto one counter; both projections must validate first."""
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != _TASK_END_FIELDS:
        return None
    projections = {
        name: _bounded_dimensions(name, {f: data[f] for f in _METRIC_FIELDS[name]})
        for name in (TASK_FINISHED_METRIC, TASK_DURATION_METRIC)
    }
    return (metric_name, projections[metric_name]) if all(projections.values()) else None


def tool_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one allowlisted tool lifecycle end event."""
    return _tool_end_projection(event, TOOL_CALL_METRIC)


def tool_usage_dimensions(event: Any) -> dict[str, str] | None:
    """Return the per-tool usage dimensions carried by the same tool lifecycle end event."""
    return _tool_end_projection(event, TOOL_USAGE_METRIC)


def tool_latency_dimensions(event: Any) -> dict[str, str] | None:
    """Return the per-category latency/retry dimensions carried by the same tool end event."""
    return _tool_end_projection(event, TOOL_LATENCY_METRIC)


_TOOL_END_METRICS = (TOOL_CALL_METRIC, TOOL_USAGE_METRIC, TOOL_LATENCY_METRIC)
_TOOL_END_FIELDS = frozenset().union(*(_METRIC_FIELDS[name] for name in _TOOL_END_METRICS))


def _tool_end_projection(event: Any, metric_name: str) -> dict[str, str] | None:
    """Project the tool end event onto one counter; both projections must validate first."""
    if not _valid_shape(
        event, kind="scope", scope_category="end", category="tool", name=TOOL_CALL_SCOPE,
        category_profile={},
    ):
        return None
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != _TOOL_END_FIELDS:
        return None
    projections = {
        name: _bounded_dimensions(name, {f: data[f] for f in _METRIC_FIELDS[name]})
        for name in _TOOL_END_METRICS
    }
    return projections[metric_name] if all(projections.values()) else None


def install_snapshot_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the daily install-configuration snapshot from a safe mark."""
    return _mark_counter(event, {INSTALL_SNAPSHOT_MARK: INSTALL_SNAPSHOT_METRIC})


def tool_approval_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated approval counter from a safe Relay mark event."""
    return _mark_counter(event, {TOOL_APPROVAL_MARK: TOOL_APPROVAL_METRIC})


def skill_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated skill lifecycle or load counter from a safe mark."""
    return _mark_counter(event, _SKILL_MARK_METRICS)


def decision_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the counter for one session/setup/compression/switch/fallback/command/install mark."""
    return _mark_counter(event, _DECISION_MARK_METRICS)


_TOKEN_MARK_DIMENSIONS = frozenset({"aux_task", "call_role", "model", "provider"})


def model_token_counters(event: Any) -> list[tuple[str, dict[str, str], int]]:
    """Expand one token-usage mark into ``(metric, dimensions, amount)`` sums, one per token type."""
    if not _valid_shape(event, **_MARK_SHAPE) or _event_text(event, "name") != MODEL_TOKENS_MARK:
        return []
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != _TOKEN_MARK_DIMENSIONS | TOKEN_TYPES:
        return []
    base = {field: data[field] for field in _TOKEN_MARK_DIMENSIONS}
    counters = []
    for token_type in sorted(TOKEN_TYPES):
        amount = data[token_type]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            return []
        dimensions = {**base, "token_type": token_type}
        if not counter_dimensions_are_valid(MODEL_TOKENS_METRIC, dimensions) or not _catalog_holds(
            MODEL_TOKENS_METRIC, dimensions,
        ):
            return []
        if amount:
            counters.append((MODEL_TOKENS_METRIC, dimensions, amount))
    return counters


def skill_lifecycle_fields(kwargs: dict[str, Any]) -> dict[str, str] | None:
    """Build bounded fields for one successful non-load skill transition."""
    action = _norm(kwargs.get("action"))
    if action not in SKILL_LIFECYCLE_ACTIONS:
        return None
    return {"action": action, "provenance": skill_provenance(kwargs.get("provenance"))}


def skill_load_fields(kwargs: dict[str, Any]) -> dict[str, str] | None:
    """Build bounded skill-use fields without exporting local skill identity."""
    use_count, reused = kwargs.get("use_count"), kwargs.get("reused")
    reuse_after_patch = kwargs.get("reuse_after_patch")
    if (
        isinstance(use_count, bool) or not isinstance(use_count, int) or use_count < 1
        or not isinstance(reused, bool) or not isinstance(reuse_after_patch, bool)
        or (reuse_after_patch and not reused)
    ):
        return None
    return {
        "post_patch_state": (
            "not_applicable" if not reused
            else "reused_after_patch" if reuse_after_patch
            else "no_new_patch"
        ),
        "provenance": skill_provenance(kwargs.get("provenance")),
        "reuse_state": "reused" if reused else "first_use",
        "skill_name": _skill_metric_name(kwargs.get("skill_name")),
        "use_count_bucket": count_bucket(use_count),
    }


def _skill_metric_name(value: Any) -> str:
    from .shared_metrics_catalog import skill_metric_name

    return skill_metric_name(value)


def skill_provenance(value: Any) -> str:
    """Normalize producer provenance to the closed shared-metrics taxonomy."""
    return _allowlisted(_norm(value), SKILL_PROVENANCES)


_SURFACE_ALIASES = {
    "api_server": "api",
    # The relay connector's own platform: an inbound it could not stamp is still a gateway message.
    "relay": "gateway",
    **dict.fromkeys(("cron", "scheduler", "scheduled"), "scheduled_task"),
}
_KNOWN_GATEWAY_PLATFORMS = frozenset({"discord", "email", "slack", "telegram", "teams", "whatsapp"})


def execution_surface(kwargs: dict[str, Any]) -> str:
    """Normalize the safe session surface carried by the parent Relay scope."""
    value = _norm(kwargs.get("execution_surface") or kwargs.get("platform") or "unknown")
    if value in EXECUTION_SURFACES:
        return value
    if value in _SURFACE_ALIASES:
        return _SURFACE_ALIASES[value]
    try:
        from hermes_cli.platforms import get_all_platforms

        if value in get_all_platforms():
            return "gateway"
    except Exception:
        pass
    return "gateway" if value in _KNOWN_GATEWAY_PLATFORMS else "other"


def task_start_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    """Build the bounded fields recorded on a task scope start event."""
    surface = execution_surface(kwargs)
    return {
        "entrypoint": task_entrypoint(kwargs, surface), "execution_surface": surface,
        "platform": gateway_platform(kwargs, surface),
    }


def gateway_platform(kwargs: dict[str, Any], surface: str | None = None) -> str:
    """The messaging platform for a gateway task: core, bundled or proven catalog plugin platforms
    by name, every other plugin platform anonymous."""
    if (surface or execution_surface(kwargs)) != "gateway":
        return "none"
    name = adapter_platform(kwargs.get("platform"))
    return name if name in GATEWAY_PLATFORMS else "plugin"


def adapter_platform(value: Any) -> str:
    """Public name of a gateway adapter's platform (Platform enum or its value)."""
    from .shared_metrics_catalog import platform_metric_name

    return platform_metric_name(value, _CORE_GATEWAY_PLATFORMS | _gateway_adapter_platforms())


_SURFACE_ENTRYPOINTS = {
    # An ACP session is a human in an editor (VS Code / Zed / JetBrains), same
    # dispatch shape as the other interactive surfaces.
    **dict.fromkeys(("acp", "cli", "desktop", "tui"), "interactive"),
    **{s: s for s in ("api", "batch", "python", "scheduled_task", "unknown")},
    "gateway": "gateway_message",
}


def task_entrypoint(kwargs: dict[str, Any], surface: str | None = None) -> str:
    """Normalize the task dispatch owner without exporting source strings."""
    declared = _norm(kwargs.get("entrypoint"))
    if declared in TASK_ENTRYPOINTS:
        return declared
    if kwargs.get("parent_task_id") or kwargs.get("parent_session_id"):
        return "delegated"
    return _SURFACE_ENTRYPOINTS.get(surface or execution_surface(kwargs), "other")


def task_terminal_fields(
    kwargs: dict[str, Any], *, duration_ms: int, model_call_count: int, tool_call_count: int,
    retry_count: int,
) -> dict[str, str]:
    """Build the bounded terminal payload for one task scope."""
    outcome, end_reason, termination = task_terminal_state(kwargs)
    return {
        **task_start_fields(kwargs),
        "duration_bucket": duration_bucket(duration_ms),
        "end_reason": end_reason,
        "failure_class": task_failure_class(kwargs, outcome),
        "model_call_count_bucket": count_bucket(model_call_count),
        "outcome": outcome,
        "retry_count_bucket": count_bucket(retry_count),
        "termination": termination,
        "tool_call_count_bucket": count_bucket(tool_call_count),
    }


_LOCAL_FAILURE_CLASSES = {"interpreter_shutdown": "shutdown", "session_busy": "session_busy"}
# turn_exit_reason prefix -> class, for failures that never reached the provider classifier.
_EXIT_REASON_FAILURE_CLASSES = (
    ("empty_response", "empty_response"), ("all_retries_exhausted", "empty_response"),
    ("context_compression", "context_compression"), ("compaction_", "context_compression"),
    ("ollama_runtime_context", "context_compression"),
    ("local_processing_error", "local_error"), ("repeated_outer_errors", "repeated_errors"),
    ("error_near_max_iterations", "repeated_errors"), ("session_persistence", "persistence"),
    ("redirect_restart_limit", "restart_limit"), ("rebuilt_restart_limit", "restart_limit"),
)


def task_failure_class(kwargs: dict[str, Any], outcome: str | None = None) -> str:
    """Why a failed task failed, as a closed class; ``none`` for any non-failed outcome."""
    if (outcome or task_terminal_state(kwargs)[0]) != "failed":
        return "none"
    declared = _norm(kwargs.get("failure_class"))
    if declared in TASK_FAILURE_CLASSES - {"none"}:
        return declared
    failure_reason = _norm(kwargs.get("failure_reason"))
    if failure_reason in MODEL_ERROR_CLASSES - {"none"}:
        return failure_reason
    if failure_reason in _LOCAL_FAILURE_CLASSES:
        return _LOCAL_FAILURE_CLASSES[failure_reason]
    reason = _norm(kwargs.get("turn_exit_reason"))
    return next(
        (label for prefix, label in _EXIT_REASON_FAILURE_CLASSES if reason.startswith(prefix)),
        "other",
    )


def task_terminal_state(kwargs: dict[str, Any]) -> tuple[str, str, str]:
    """Map Hermes terminal state to bounded (outcome, end_reason, termination)."""
    reason = _norm(kwargs.get("turn_exit_reason"))
    if kwargs.get("interrupted") or "interrupt" in reason or "cancel" in reason:
        return "cancelled", "user_cancelled", "user_cancelled"
    if "timeout" in reason or "timed_out" in reason:
        return "timed_out", "timed_out", "timed_out"
    if "max_iterations" in reason or "budget_exhausted" in reason:
        return "failed", "iteration_limit", "system_aborted"
    if "approval" in reason and ("denied" in reason or "rejected" in reason):
        return "failed", "approval_denied", "none"
    if "guardrail" in reason:
        return "failed", "guardrail_blocked", "system_aborted"
    if reason == "system_aborted":
        return "failed", "system_aborted", "system_aborted"
    if kwargs.get("completed") is True:
        return "success", "completed", "none"
    if kwargs.get("failed") is True or (reason and reason != "unknown"):
        return "failed", "failed", "none"
    return "unknown", "unknown", "unknown"


# (exclusive upper bound, label) — ascending; the trailing label catches the rest.
_DURATION_THRESHOLDS = (
    (1_000, "lt_1s"), (5_000, "1s_to_5s"), (30_000, "5s_to_30s"),
    (120_000, "30s_to_2m"), (600_000, "2m_to_10m"),
)
_COUNT_THRESHOLDS = ((1, "0"), (2, "1"), (3, "2"), (6, "3_to_5"), (11, "6_to_10"))
_LATENCY_THRESHOLDS = (
    (100, "lt_100ms"), (250, "100ms_to_250ms"), (500, "250ms_to_500ms"), (1_000, "500ms_to_1s"),
    (2_000, "1s_to_2s"), (5_000, "2s_to_5s"), (10_000, "5s_to_10s"), (30_000, "10s_to_30s"),
)


def _bucket(value: float, thresholds: tuple[tuple[float, str], ...], last: str) -> str:
    return next((label for upper, label in thresholds if value < upper), last)


def duration_bucket(duration_ms: int) -> str:
    """Bucket a non-negative task duration into a fixed low-cardinality range."""
    return _bucket(max(0, int(duration_ms)), _DURATION_THRESHOLDS, "gte_10m")


def count_bucket(count: int) -> str:
    """Bucket a non-negative per-task count into a fixed range."""
    return _bucket(max(0, int(count)), _COUNT_THRESHOLDS, "gte_11")


_TOOL_CATEGORY_EXACT = {
    **{category: category for category in TOOL_CATEGORIES},
    "clarify": "planning", "kanban": "planning", "todo": "planning", "session_search": "memory",
    "cronjob": "scheduler", "skills": "skill", "x_search": "web",
}
_TOOL_CATEGORY_PREFIXES = (
    ("mcp", "mcp"),
    ("browser", "browser"),
    (("image", "tts", "video", "vision"), "media"),
    ("homeassistant", "home_automation"),  # the homeassistant catalog plugin's toolset
    (("discord", "email", "feishu", "hermes-yuanbao", "slack", "sms"), "communication"),
)


def tool_category(kwargs: dict[str, Any]) -> str:
    """Map Hermes registry toolset metadata to a low-cardinality category."""
    toolset = _norm(kwargs.get("toolset"))
    if not toolset:
        return "unknown"
    if toolset in _TOOL_CATEGORY_EXACT:
        return _TOOL_CATEGORY_EXACT[toolset]
    for prefixes, category in _TOOL_CATEGORY_PREFIXES:
        if toolset.startswith(prefixes):
            return category
    return "other"


_TOOL_STATUS_OUTCOMES = {
    **{s: s for s in ("blocked", "cancelled", "failed", "success", "timed_out")},
    "error": "failed", "ok": "success", "timeout": "timed_out",
}


def tool_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize the terminal Hermes tool status without inspecting its result."""
    return _TOOL_STATUS_OUTCOMES.get(_norm(kwargs.get("status")), "unknown")


_APPROVAL_CHOICES = {
    **dict.fromkeys(
        ("always", "approve", "approved", "once", "session", "smart_approve"), "approved"
    ),
    **dict.fromkeys(("deny", "denied", "smart_deny"), "denied"),
    **dict.fromkeys(("timed_out", "timeout"), "timed_out"),
    "cancelled": "cancelled",  # prompt withdrawn / undeliverable / unanswered — not a user decision
}


def tool_approval_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize a terminal approval choice to a bounded outcome."""
    return _APPROVAL_CHOICES.get(_norm(kwargs.get("choice")), "unknown")


def tool_terminal_fields(
    kwargs: dict[str, Any], *, category: str | None = None, approval_outcome: str = "not_required",
    fallback_duration_ms: int | None = None, tool_name: str | None = None,
) -> dict[str, str]:
    """Build one bounded tool-call terminal payload (feeds the category and per-tool counters)."""
    outcome = tool_outcome(kwargs)
    return {
        "approval_outcome": _allowlisted(approval_outcome, TOOL_APPROVAL_OUTCOMES),
        "error_class": tool_error_class(kwargs, outcome),
        "latency_bucket": tool_latency_bucket(
            kwargs.get("duration_ms"), fallback_duration_ms=fallback_duration_ms
        ),
        "outcome": outcome,
        "retry_count_bucket": tool_retry_bucket(kwargs.get("retry_count")),
        "tool_category": category if category in TOOL_CATEGORIES else tool_category(kwargs),
        "tool_name": tool_name if tool_name is not None and tool_name in TOOL_NAMES
        else tool_metric_name(kwargs),
    }


def tool_metric_name(kwargs: dict[str, Any]) -> str:
    """A built-in tool's own name; MCP and plugin tools collapse to their source kind."""
    name = _norm(kwargs.get("tool_name"))
    if name in BUILTIN_TOOL_NAMES:
        return name
    if not name:
        return "unknown"
    return "mcp" if name.startswith("mcp_") or tool_category(kwargs) == "mcp" else "plugin"


_TOOL_STATUS_ERROR_CLASSES = {
    "blocked": "blocked", "cancelled": "interrupted", "success": "none", "timed_out": "timeout",
}
_TOOL_ERROR_TYPES = {
    **dict.fromkeys(("keyboard_interrupt", "tool_interrupted", "user_interrupt"), "interrupted"),
    "invalid_tool_arguments": "invalid_arguments", "thread_missing_result": "exception",
    "tool_error": "tool_error", "tool_result_contract": "contract_violation",
    "tool_timeout": "timeout",
}


def tool_error_class(kwargs: dict[str, Any], outcome: str | None = None) -> str:
    """Closed failure class from Hermes's own error_type; exception class names become
    ``exception`` so plugin-defined identifiers never leave the machine."""
    outcome = outcome or tool_outcome(kwargs)
    if outcome in _TOOL_STATUS_ERROR_CLASSES:
        return _TOOL_STATUS_ERROR_CLASSES[outcome]
    if outcome != "failed":
        return "unknown"
    error_type = _norm(kwargs.get("error_type"))
    return _TOOL_ERROR_TYPES.get(error_type, "exception" if error_type else "unknown")


def tool_latency_bucket(value: Any, *, fallback_duration_ms: int | None = None) -> str:
    """Bucket a tool duration reported in milliseconds."""
    duration_ms = _non_negative_number(value)
    if duration_ms is None:
        duration_ms = _non_negative_number(fallback_duration_ms)
    if duration_ms is None:
        return "unknown"
    return _bucket(duration_ms, _LATENCY_THRESHOLDS, "gte_30s")


def tool_retry_bucket(value: Any) -> str:
    """Bucket only explicit tool retries; missing relationships stay unknown."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "unknown"
    return count_bucket(value)


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) and number >= 0 else None


def model_call_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    """Return the terminal model identity and provider route known to Hermes."""
    from .shared_metrics_catalog import model_metric_name, provider_metric_name

    provider = provider_metric_name(kwargs.get("provider"))
    model = model_metric_name(kwargs.get("response_model"), provider, max_length=MODEL_IDENTIFIER_MAX_LENGTH)
    if model == "unknown":
        model = model_metric_name(kwargs.get("model"), provider, max_length=MODEL_IDENTIFIER_MAX_LENGTH)
    return {"model": model, "provider": provider}


def model_route_fields(
    kwargs: dict[str, Any], *, call_role: str, outcome: str, error_class: str,
    ttft_bucket: str = "unknown",
) -> dict[str, str]:
    """The terminal route plus how the logical call ended. ``error_class`` is the last
    classified attempt error, so ``success`` + ``rate_limit`` reads as "recovered from a 429"."""
    return {
        **model_call_fields(kwargs),
        "call_role": _allowlisted(call_role, MODEL_CALL_ROLES),
        "error_class": error_class if error_class in MODEL_ERROR_CLASSES else "unknown",
        "outcome": outcome if outcome in MODEL_OUTCOMES else "failed",
        "ttft_bucket": ttft_bucket if ttft_bucket in TTFT_BUCKETS else "unknown",
    }


def model_error_class(kwargs: dict[str, Any]) -> str:
    """The classifier's FailoverReason for one failed provider attempt."""
    reason = _norm(kwargs.get("reason"))
    return reason if reason in MODEL_ERROR_CLASSES - {"none"} else "unknown"


# (exclusive upper bound, label) for install-scale counts (skills routinely exceed 100).
_SIZE_THRESHOLDS = (
    (1, "0"), (2, "1"), (3, "2"), (6, "3_to_5"), (11, "6_to_10"), (26, "11_to_25"),
    (101, "26_to_100"), (251, "101_to_250"),
)


def size_bucket(count: int) -> str:
    """Bucket a non-negative install-scale count."""
    return _bucket(max(0, int(count)), _SIZE_THRESHOLDS, "gte_251")


def install_snapshot_fields(
    *, memory_provider: Any, mcp_servers: int, plugins: int, skills: int, cron_jobs: int,
    profiles: int, messaging_platforms: int, install_age_bucket: str, main_provider: Any,
    terminal_backend: Any, display_language: Any,
) -> dict[str, str]:
    """Bounded daily configuration snapshot: counts, closed enums and public names only."""
    from .shared_metrics_catalog import display_language_metric_name, provider_metric_name

    provider = _norm(memory_provider)
    backend = _norm(terminal_backend) or "local"
    return {
        "cron_job_count_bucket": size_bucket(cron_jobs),
        "display_language": display_language_metric_name(display_language),
        "install_age_bucket": install_age_bucket if install_age_bucket in INSTALL_AGE_BUCKETS else "unknown",
        "main_provider": provider_metric_name(main_provider) if main_provider else "none",
        "mcp_server_count_bucket": size_bucket(mcp_servers),
        "memory_provider": "builtin" if not provider
        else provider if provider in MEMORY_PROVIDERS else "plugin",
        "messaging_platform_count_bucket": size_bucket(messaging_platforms),
        "plugin_count_bucket": size_bucket(plugins),
        "profile_count_bucket": size_bucket(profiles),
        "skill_count_bucket": size_bucket(skills),
        "terminal_backend": backend if backend in TERMINAL_BACKENDS else "other",
    }


def _metric_identifier(value: Any, *, max_length: int) -> str:
    """Normalize one structurally safe identifier without a product catalog."""
    if not isinstance(value, str):
        return "unknown"
    identifier = value.strip().lower()
    if (
        not identifier
        or len(identifier) > max_length
        or identifier[0] not in _METRIC_IDENTIFIER_START_CHARACTERS
        or not _METRIC_IDENTIFIER_CHARACTERS.issuperset(identifier)
    ):
        return "unknown"
    return identifier
