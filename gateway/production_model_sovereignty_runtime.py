"""Reviewed production startup contract for Cloud Muncho.

This module contains only mechanical policy and identity checks.  It does not
classify a request, choose a tool, decide whether work is complete, or route a
message by authored text.  The normal :mod:`gateway.run` / :class:`AIAgent`
loop remains the sole semantic authority.

The producer deliberately accepts the *observed* production config bytes and
an exact digest.  It deep-copies that mapping, changes only the reviewed
runtime boundaries, and returns digest-bound config and systemd unit bytes.
Unknown business/operator settings survive unchanged.  The public manifest
contains digests and identities only; it never contains config or credential
material.
"""

from __future__ import annotations

import copy
import grp
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import posixpath
import pwd
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from gateway.mac_ops_edge_client import MacOpsEdgeClientConfig
from gateway.production_access_policy import (
    PRODUCTION_OWNER_DISCORD_USER_ID,
    production_access_config,
)
from gateway.support_ops_team_registry import (
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    SKYVISION_GUILD_ID,
)
from gateway.production_capability_prerequisites import (
    API_APPROVAL_CREDENTIAL_NAME,
    API_APPROVAL_CREDENTIAL_PATH,
    API_SERVER_CREDENTIAL_NAME,
    API_SERVER_CREDENTIAL_PATH,
    API_SERVER_HOST,
    API_SERVER_PORT,
    BROWSER_ARTIFACT_PATH,
    BROWSER_UNIT,
    CODEX_AUTH_PATH,
    FIRST_WAVE_TOOLSETS,
    MAC_OPS_CONFIG_PATH,
    MAC_OPS_SOCKET_PATH,
    MAC_OPS_UNIT,
    PHASE_B_RECEIPT_PATH,
    PHASE_B_UNIT,
    PREREQUISITE_LIFECYCLE_COMMITTED,
    PREREQUISITE_PATH,
    PUBLIC_CONNECTOR_UNIT,
    ROUTEBACK_EDGE_CONFIG_PATH,
    ROUTEBACK_EDGE_SOCKET_PATH,
    ROUTEBACK_EDGE_UNIT,
    attest_live_production_gateway_service_identity,
    load_production_capability_prerequisite_receipt,
    production_capability_topology_identity_sha256,
    validate_production_capability_prerequisite_receipt,
    validate_production_capability_topology,
)
from gateway.isolated_worker_units import (
    ISOLATED_WORKER_CLIENT_GROUP,
    ISOLATED_WORKER_SERVICE_UNIT,
    ISOLATED_WORKER_SOCKET_UNIT,
)
from tools.browser_controller_client import (
    CLIENT_CONFIG_SCHEMA as BROWSER_CLIENT_CONFIG_SCHEMA,
    BrowserControllerClientConfig,
)


CONTRACT_SCHEMA = "muncho-production-model-sovereignty-gateway-contract.v1"
PRODUCTION_CONFIG_PATH = Path(
    "/opt/adventico-ai-platform/hermes-home/config.yaml"
)
PRODUCTION_HOME = PRODUCTION_CONFIG_PATH.parent
PRODUCTION_RELEASES = Path(
    "/opt/adventico-ai-platform/hermes-agent-releases"
)
LEGACY_API_BEARER_SOURCE_PATH = Path(
    "/etc/muncho/keys/api-server-control.key"
)
OWNER_APPROVAL_PASSKEY_SOURCE_PATH = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/api-approval-passkey"
)
GATEWAY_UNIT = "hermes-cloud-gateway.service"
WRITER_UNIT = "muncho-canonical-writer.service"
CONNECTOR_UNIT = PUBLIC_CONNECTOR_UNIT
WRITER_RUNTIME = Path("/run/muncho-canonical-writer")
CONNECTOR_RUNTIME = Path("/run/muncho-discord-connector")
CONNECTOR_SOCKET = CONNECTOR_RUNTIME / "connector.sock"
RELAY_URL = f"unix://{CONNECTOR_SOCKET}"
STARTUP_FLAG = "--require-production-model-sovereignty"
BROWSER_SOCKET_GROUP = "muncho-capability-browser"

# The production ``web`` toolset is backed by one exact bundled mechanical
# executor.  It contributes no model tool, hook, middleware, prompt, or routing
# policy: GPT remains the sole authority deciding when and how to search.
PRODUCTION_WEB_PLUGIN_KEY = "web/ddgs"
PRODUCTION_WEB_PROVIDER_NAME = "ddgs"
PRODUCTION_WEB_DISTRIBUTION = "ddgs"
PRODUCTION_WEB_DISTRIBUTION_VERSION = "9.14.4"
PRODUCTION_PLUGIN_ALLOWLIST = frozenset({PRODUCTION_WEB_PLUGIN_KEY})
_PRODUCTION_WEB_CONFIG = {
    "backend": "",
    "search_backend": PRODUCTION_WEB_PROVIDER_NAME,
    "extract_backend": "",
}
_PRODUCTION_DELEGATION_CONFIG = {
    # Empty route fields mean exact inheritance from the already pinned
    # primary gpt-5.6-sol/openai-codex runtime.  In particular, there is no
    # cheaper model, alternate provider, endpoint, wire protocol, credential,
    # or reasoning-effort override for delegated work.
    "provider": "",
    "model": "",
    "base_url": "",
    "api_key": "",
    "api_mode": "",
    "reasoning_effort": "",
    # Complex work gets the same full turn budget without an arbitrary wall
    # clock kill, while fan-out and nesting remain mechanically bounded.
    "max_iterations": 90,
    "child_timeout_seconds": 0,
    "max_concurrent_children": 4,
    "orchestrator_enabled": True,
    "max_spawn_depth": 2,
    # Dangerous mutations consume exact owner-approved plan capabilities;
    # delegated threads never receive blanket approval.
    "subagent_auto_approve": False,
}

_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SYSTEMD_IDENTITY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_DISCORD_SNOWFLAKE = re.compile(r"[0-9]{17,20}")
_PRODUCTION_CRON_ORIGIN_FIELDS = frozenset(
    {"platform", "chat_id", "chat_name", "thread_id", "user_id"}
)
_STALE_MODEL_SENTENCE = (
    "gpt-5.6-sol; do not route GPT-5.5 through OPENAI_API_KEY."
)
_CURRENT_MODEL_SENTENCE = "gpt-5.6-sol."

_MODEL = {
    "default": "gpt-5.6-sol",
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
}
_AUXILIARY_ROUTE = {
    "provider": "openai-codex",
    "model": "gpt-5.6-sol",
    "base_url": "",
    "api_key": "",
    "fallback_chain": [],
    "extra_body": {},
}
_AUXILIARY_TASKS = (
    "vision",
    "web_extract",
    "compression",
    "skills_hub",
    "mcp",
    "title_generation",
    "tts_audio_tags",
    "profile_describer",
    "curator",
    "monitor",
    "background_review",
    "moa_reference",
    "moa_aggregator",
)
_AUXILIARY_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "vision": {"timeout": 120, "download_timeout": 30},
    "web_extract": {"timeout": 360},
    "compression": {"timeout": 120},
    "skills_hub": {"timeout": 30},
    "mcp": {"timeout": 30},
    "title_generation": {"timeout": 30, "language": ""},
    "tts_audio_tags": {"timeout": 30},
    "profile_describer": {"timeout": 60},
    "curator": {"timeout": 600},
    "monitor": {"timeout": 60},
    "background_review": {"timeout": 120},
    "moa_reference": {"timeout": 900},
    "moa_aggregator": {"timeout": 900},
}
_RELAY_PLATFORM = {
    "enabled": True,
    "extra": {"relay_url": RELAY_URL},
}
_API_PLATFORM = {
    "enabled": True,
    "extra": {
        "host": API_SERVER_HOST,
        "port": API_SERVER_PORT,
        "key_verifier_credential": API_SERVER_CREDENTIAL_NAME,
        "approval_verifier_credential": API_APPROVAL_CREDENTIAL_NAME,
    },
}
_PLATFORMS = {
    "api_server": _API_PLATFORM,
    "relay": _RELAY_PLATFORM,
}
_FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "DISCORD_BOT_TOKEN",
        "DISCORD_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "HERMES_API_KEY",
        "HERMES_BASE_URL",
        "HERMES_MODEL",
        "HERMES_PROVIDER",
        "HERMES_YOLO_MODE",
        "HERMES_IGNORE_USER_CONFIG",
        "HERMES_ENABLE_PROJECT_PLUGINS",
        "HERMES_SAFE_MODE",
        "GATEWAY_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "DISCORD_ALLOWED_USERS",
    }
)
_SYSTEMD_OPTIONAL_ENVIRONMENT_NAMES = frozenset(
    {
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "LOGNAME",
        "NOTIFY_SOCKET",
        "PWD",
        "RUNTIME_DIRECTORY",
        "SHELL",
        "SYSTEMD_EXEC_PID",
        "SYSTEMD_NSS_BYPASS_SYNTHETIC",
        "USER",
    }
)

# Origins checked before READY.  ``run_agent`` and ``model_tools`` remain lazy
# by design; the SHA-first PYTHONPATH and sys.path attestation protects their
# later import without eagerly expanding the gateway's startup surface.
REQUIRED_IMPORT_ORIGINS = {
    "gateway.run": "gateway/run.py",
    "gateway.production_model_sovereignty_runtime": (
        "gateway/production_model_sovereignty_runtime.py"
    ),
    "gateway.production_capability_prerequisites": (
        "gateway/production_capability_prerequisites.py"
    ),
    "gateway.canonical_writer_boundary": (
        "gateway/canonical_writer_boundary.py"
    ),
    "agent.conversation_loop": "agent/conversation_loop.py",
    "hermes_cli.config": "hermes_cli/config.py",
    "providers": "providers/__init__.py",
    "providers.base": "providers/base.py",
    "agent.web_search_registry": "agent/web_search_registry.py",
    "hermes_cli.plugins": "hermes_cli/plugins.py",
    "plugins.model_providers.openai_codex": (
        "plugins/model-providers/openai-codex/__init__.py"
    ),
    "plugins.web.ddgs": "plugins/web/ddgs/__init__.py",
    "plugins.web.ddgs.provider": "plugins/web/ddgs/provider.py",
}


class ProductionContractError(RuntimeError):
    """Stable, non-secret production contract failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictLoader(yaml.SafeLoader):
    """Reject aliases plus duplicate, empty, and non-string mapping keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ProductionContractError("production_config_alias_forbidden")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.nodes.MappingNode):
            raise ProductionContractError("production_config_mapping_invalid")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or not key or key in result:
                raise ProductionContractError("production_config_key_invalid")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _StrictLoader.construct_mapping,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def production_release_root(revision: str) -> Path:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ProductionContractError("production_revision_invalid")
    return PRODUCTION_RELEASES / f"hermes-agent-{revision[:12]}"


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    candidate = value.get(key)
    if not isinstance(candidate, dict):
        raise ProductionContractError(f"production_config_{key}_invalid")
    return candidate


def _pinned_auxiliary_task(
    value: Any,
    *,
    task: str,
) -> dict[str, Any]:
    if value is None:
        result = dict(_AUXILIARY_DEFAULTS[task])
    elif isinstance(value, Mapping):
        result = copy.deepcopy(dict(value))
    else:
        raise ProductionContractError(f"production_aux_{task}_invalid")
    result.update(copy.deepcopy(_AUXILIARY_ROUTE))
    for key, default in _AUXILIARY_DEFAULTS[task].items():
        result.setdefault(key, default)
    return result


def _terminal_config(topology: Mapping[str, Any]) -> dict[str, Any]:
    worker = topology["isolated_worker"]
    return {
        "backend": "isolated_worker",
        "cwd": "/workspace",
        "timeout": 180,
        "home_mode": "profile",
        "lifetime_seconds": 900,
        "isolated_worker_socket": worker["socket_path"],
        "isolated_worker_server_uid": worker["server_uid"],
        "isolated_worker_server_gid": worker["server_gid"],
        "isolated_worker_socket_uid": worker["socket_uid"],
        "isolated_worker_socket_gid": worker["socket_gid"],
    }


def _browser_controller_mapping(topology: Mapping[str, Any]) -> dict[str, Any]:
    browser = topology["browser"]
    return {
        "schema": BROWSER_CLIENT_CONFIG_SCHEMA,
        "socket_path": browser["socket_path"],
        "server_uid": browser["service_uid"],
        "artifact_root": str(BROWSER_ARTIFACT_PATH),
        "connect_timeout_seconds": 5,
        "request_timeout_seconds": 120,
    }


def _browser_config(topology: Mapping[str, Any]) -> dict[str, Any]:
    return {"controller": _browser_controller_mapping(topology)}


def production_browser_controller_client_config(
    topology: Mapping[str, Any],
) -> BrowserControllerClientConfig:
    """Return the exact validated gateway-side browser controller binding."""

    try:
        topology = validate_production_capability_topology(topology)
        return BrowserControllerClientConfig.from_mapping(
            _browser_controller_mapping(topology)
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionContractError(
            "production_browser_controller_client_config_invalid"
        ) from exc


def _production_platform_toolsets() -> dict[str, list[str]]:
    """Return a fresh copy of the exact reviewed platform tool surfaces."""

    return {
        "api_server": list(FIRST_WAVE_TOOLSETS),
        "relay": list(FIRST_WAVE_TOOLSETS),
        # The privileged local Discord connector emits authentic Discord
        # SessionSource records even though RelayAdapter carries the transport.
        # Pin that concrete source platform so it cannot fall back to the broad
        # upstream ``hermes-discord`` composite.
        "discord": list(FIRST_WAVE_TOOLSETS),
        "cron": list(FIRST_WAVE_TOOLSETS),
    }


def load_strict_production_config(raw: bytes) -> dict[str, Any]:
    """Parse bounded YAML bytes without normalization or env expansion."""

    if not isinstance(raw, bytes) or not 0 < len(raw) <= 2 * 1024 * 1024:
        raise ProductionContractError("production_config_size_invalid")
    try:
        value = yaml.load(raw.decode("utf-8", errors="strict"), Loader=_StrictLoader)
    except ProductionContractError:
        raise
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ProductionContractError("production_config_yaml_invalid") from exc
    if not isinstance(value, dict):
        raise ProductionContractError("production_config_root_invalid")
    return value


def overlay_production_gateway_config(
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bounded model-sovereignty overlay.

    The operation is semantic-preserving outside the explicitly reviewed
    config paths.  Arbitrary company settings, sessions, skills, and display
    preferences are retained.  The reviewed capability topology, toolsets,
    terminal/browser projection, and API/relay transports are exact.
    """

    if not isinstance(observed, Mapping):
        raise ProductionContractError("production_config_root_invalid")
    target = copy.deepcopy(dict(observed))

    if _mapping(target, "model") != _MODEL:
        raise ProductionContractError("production_model_route_drifted")

    agent = _mapping(target, "agent")
    if agent.get("reasoning_effort") != "high" or agent.get("max_turns") != 90:
        raise ProductionContractError("production_agent_baseline_drifted")
    if (
        agent.get("task_completion_guidance") is not True
        or agent.get("parallel_tool_call_guidance") is not True
    ):
        raise ProductionContractError("production_agent_guidance_drifted")
    hint = agent.get("environment_hint")
    if not isinstance(hint, str) or not hint:
        raise ProductionContractError("production_environment_hint_invalid")
    if _STALE_MODEL_SENTENCE in hint:
        if hint.count(_STALE_MODEL_SENTENCE) != 1:
            raise ProductionContractError("production_environment_hint_ambiguous")
        hint = hint.replace(_STALE_MODEL_SENTENCE, _CURRENT_MODEL_SENTENCE)
    elif "gpt-5.5" in hint.casefold():
        raise ProductionContractError("production_environment_hint_drifted")
    agent["environment_hint"] = hint
    agent["reasoning_effort"] = "high"
    agent["max_turns"] = 90
    agent["adaptive_reasoning"] = {"enabled": True, "max_effort": "max"}
    agent["tool_use_enforcement"] = True
    agent["verify_on_stop"] = False
    agent["verification_ledger_enabled"] = False
    agent["background_review_enabled"] = False

    compression = _mapping(target, "compression")
    compression["enabled"] = True
    compression["abort_on_summary_failure"] = True

    auxiliary = _mapping(target, "auxiliary")
    # These legacy semantic side-model slots are retired in Cloud Muncho.
    # Strip them deterministically rather than pinning dormant decision
    # surfaces into the production source of truth.
    for retired_task in ("approval", "goal_judge", "triage_specifier", "kanban_decomposer"):
        auxiliary.pop(retired_task, None)
    for name, item in auxiliary.items():
        if name == "transient_retries":
            if type(item) is not int or not 0 <= item <= 6:
                raise ProductionContractError(
                    "production_aux_transient_retries_invalid"
                )
            continue
        if name not in _AUXILIARY_TASKS and isinstance(item, Mapping) and any(
            key in item
            for key in (
                "provider",
                "model",
                "base_url",
                "api_key",
                "fallback_chain",
            )
        ):
            raise ProductionContractError("production_aux_route_unreviewed")
    for task in _AUXILIARY_TASKS:
        auxiliary[task] = _pinned_auxiliary_task(
            auxiliary.get(task),
            task=task,
        )

    context = target.setdefault("context", {})
    if not isinstance(context, dict):
        raise ProductionContractError("production_config_context_invalid")
    context["engine"] = "compressor"

    memory = _mapping(target, "memory")
    memory["provider"] = ""
    memory["memory_enabled"] = True
    memory["user_profile_enabled"] = True

    curator = _mapping(target, "curator")
    curator["enabled"] = False
    curator["consolidate"] = False
    curator["prune_builtins"] = False

    kanban = _mapping(target, "kanban")
    kanban["auxiliary_planning_enabled"] = False
    kanban["auto_decompose"] = False
    kanban["dispatch_in_gateway"] = False

    guardrails = _mapping(target, "tool_loop_guardrails")
    guardrails["warnings_enabled"] = True
    guardrails["hard_stop_enabled"] = False

    # Progressive Tool Search selects which model tools are visible by
    # tokenising their names/descriptions and ranking them outside the model.
    # Production Muncho keeps the full reviewed tool surface model-visible:
    # GPT-5.6-sol, not a BM25/keyword bridge, is the semantic authority.
    tools = target.setdefault("tools", {})
    if not isinstance(tools, dict):
        raise ProductionContractError("production_config_tools_invalid")
    tools["tool_search"] = {"enabled": "off"}

    delegation = target.setdefault("delegation", {})
    if not isinstance(delegation, dict):
        raise ProductionContractError("production_config_delegation_invalid")
    delegation.update(copy.deepcopy(_PRODUCTION_DELEGATION_CONFIG))

    canonical = target.setdefault("canonical_brain", {})
    if not isinstance(canonical, dict):
        raise ProductionContractError("production_config_canonical_brain_invalid")
    canonical["writer_boundary"] = {"enabled": True}
    canonical["discord_edge"] = {"enabled": True}
    canonical["tools_enabled"] = True

    target["plugins"] = {"enabled": [], "disabled": []}
    target["hooks"] = {}
    target["hooks_auto_accept"] = False
    target["mcp_servers"] = {}
    target["fallback_model"] = []
    target["fallback_providers"] = []
    target["provider_routing"] = {}
    # Public Discord threads are operationally shared, but conversation
    # transcripts and cached agent state carry participant-specific authority.
    # Keep those lanes separate by exact gateway identity; cross-user
    # continuity belongs in Canonical Brain rather than another participant's
    # raw prompt/tool history.
    target["thread_sessions_per_user"] = True
    # Exact author-ID authorization is mechanical and is bound into the same
    # immutable config digest as the model/tool/runtime projection.  The model
    # never decides who is the owner.
    target["production_access"] = production_access_config()
    # ``type: exec`` quick commands launch a local shell from the gateway
    # process.  Production keeps this surface empty until it is backed by an
    # isolated, identity-bound worker.
    target["quick_commands"] = {}
    # Persistent command text/glob allowlists bypass per-plan capability and
    # owner approval checks. Production admits no blanket command authority.
    target["command_allowlist"] = []
    skills = target.setdefault("skills", {})
    if not isinstance(skills, dict):
        raise ProductionContractError("production_skills_config_invalid")
    skills["inline_shell"] = False
    security = target.setdefault("security", {})
    if not isinstance(security, dict):
        raise ProductionContractError("production_security_config_invalid")
    security["allow_lazy_installs"] = False
    target["web"] = copy.deepcopy(_PRODUCTION_WEB_CONFIG)

    cron = _mapping(target, "cron")
    cron["enabled"] = True
    cron["provider"] = "builtin"

    approvals = target.setdefault("approvals", {})
    if not isinstance(approvals, dict):
        raise ProductionContractError("production_approvals_invalid")
    approvals["mode"] = "manual"
    approvals["cron_mode"] = "deny"
    approvals["plan_owner_user_ids"] = [PRODUCTION_OWNER_DISCORD_USER_ID]
    approvals["gateway_authorized_user_ids"] = [
        PRODUCTION_OWNER_DISCORD_USER_ID
    ]
    approvals["gateway_authorized_user_names"] = []
    approvals["gateway_authorized_labels"] = ["Емо"]
    approvals["gateway_owner_escalation"] = {
        "enabled": True,
        "owner_user_id": PRODUCTION_OWNER_DISCORD_USER_ID,
        "owner_guild_id": SKYVISION_GUILD_ID,
        "owner_channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "owner_target_type": "guild_channel",
    }

    goals = target.setdefault("goals", {})
    if not isinstance(goals, dict):
        raise ProductionContractError("production_goals_invalid")
    # The owner approved plan-to-completion behavior. Zero disables only the
    # arbitrary cross-turn pause; every per-turn model/tool budget, permission,
    # capability, worker quota, and explicit pause/clear boundary remains.
    goals["max_turns"] = 0

    topology = validate_production_capability_topology(
        _mapping(target, "production_capabilities")
    )
    target["terminal"] = _terminal_config(topology)
    target["browser"] = _browser_config(topology)
    mac_ops = _mapping(target, "mac_ops_edge")
    try:
        MacOpsEdgeClientConfig.from_mapping(mac_ops)
    except (TypeError, ValueError) as exc:
        raise ProductionContractError("production_mac_ops_edge_invalid") from exc

    platforms = copy.deepcopy(_PLATFORMS)
    target["platforms"] = platforms
    gateway = target.setdefault("gateway", {})
    if not isinstance(gateway, dict):
        raise ProductionContractError("production_config_gateway_invalid")
    gateway["isolated_runtime"] = False
    gateway["relay_url"] = RELAY_URL
    gateway["api_server"] = {"max_concurrent_runs": 1}
    gateway["platforms"] = copy.deepcopy(platforms)
    target["platform_toolsets"] = _production_platform_toolsets()

    validate_production_gateway_config(target)
    return target


def validate_production_gateway_config(raw: Mapping[str, Any]) -> None:
    """Fail closed unless the effective config is the reviewed normal loop."""

    if not isinstance(raw, Mapping):
        raise ProductionContractError("production_config_root_invalid")
    if raw.get("model") != _MODEL:
        raise ProductionContractError("production_model_route_not_exact")
    if raw.get("thread_sessions_per_user") is not True:
        raise ProductionContractError(
            "production_thread_session_isolation_not_exact"
        )

    agent = raw.get("agent")
    if not isinstance(agent, Mapping) or any(
        (
            agent.get("reasoning_effort") != "high",
            agent.get("max_turns") != 90,
            agent.get("adaptive_reasoning")
            != {"enabled": True, "max_effort": "max"},
            agent.get("tool_use_enforcement") is not True,
            agent.get("verify_on_stop") is not False,
            agent.get("verification_ledger_enabled") is not False,
            agent.get("background_review_enabled") is not False,
            agent.get("task_completion_guidance") is not True,
            agent.get("parallel_tool_call_guidance") is not True,
        )
    ):
        raise ProductionContractError("production_agent_policy_not_exact")
    hint = agent.get("environment_hint")
    if not isinstance(hint, str) or not hint or "gpt-5.5" in hint.casefold():
        raise ProductionContractError("production_environment_hint_not_exact")

    compression = raw.get("compression")
    if (
        not isinstance(compression, Mapping)
        or compression.get("enabled") is not True
        or compression.get("abort_on_summary_failure") is not True
    ):
        raise ProductionContractError("production_compression_policy_not_exact")
    auxiliary = raw.get("auxiliary")
    if not isinstance(auxiliary, Mapping):
        raise ProductionContractError("production_auxiliary_not_exact")
    if any(
        retired_task in auxiliary
        for retired_task in ("approval", "goal_judge", "triage_specifier", "kanban_decomposer")
    ):
        raise ProductionContractError("production_retired_auxiliary_present")
    for task in _AUXILIARY_TASKS:
        item = auxiliary.get(task)
        if not isinstance(item, Mapping) or any(
            item.get(key) != expected
            for key, expected in _AUXILIARY_ROUTE.items()
        ):
            raise ProductionContractError("production_auxiliary_route_not_exact")
    for name, item in auxiliary.items():
        if name in _AUXILIARY_TASKS or name == "transient_retries":
            continue
        if isinstance(item, Mapping) and any(
            key in item
            for key in (
                "provider",
                "model",
                "base_url",
                "api_key",
                "fallback_chain",
            )
        ):
            raise ProductionContractError("production_auxiliary_route_unreviewed")
    context = raw.get("context")
    if not isinstance(context, Mapping) or context.get("engine") != "compressor":
        raise ProductionContractError("production_context_engine_not_exact")
    memory = raw.get("memory")
    if (
        not isinstance(memory, Mapping)
        or memory.get("provider") != ""
        or memory.get("memory_enabled") is not True
        or memory.get("user_profile_enabled") is not True
    ):
        raise ProductionContractError("production_memory_provider_not_exact")

    curator = raw.get("curator")
    if not isinstance(curator, Mapping) or any(
        curator.get(key) is not False
        for key in ("enabled", "consolidate", "prune_builtins")
    ):
        raise ProductionContractError("production_curator_boundary_not_exact")
    kanban = raw.get("kanban")
    if not isinstance(kanban, Mapping) or any(
        kanban.get(key) is not False
        for key in (
            "auxiliary_planning_enabled",
            "auto_decompose",
            "dispatch_in_gateway",
        )
    ):
        raise ProductionContractError("production_kanban_boundary_not_exact")
    guardrails = raw.get("tool_loop_guardrails")
    if (
        not isinstance(guardrails, Mapping)
        or guardrails.get("warnings_enabled") is not True
        or guardrails.get("hard_stop_enabled") is not False
    ):
        raise ProductionContractError("production_tool_loop_policy_not_exact")
    tools = raw.get("tools")
    if (
        not isinstance(tools, Mapping)
        or tools.get("tool_search") != {"enabled": "off"}
    ):
        raise ProductionContractError("production_tool_search_not_disabled")
    delegation = raw.get("delegation")
    if not isinstance(delegation, Mapping) or any(
        delegation.get(key) != expected
        for key, expected in _PRODUCTION_DELEGATION_CONFIG.items()
    ):
        raise ProductionContractError("production_delegation_route_not_exact")

    canonical = raw.get("canonical_brain")
    if (
        not isinstance(canonical, Mapping)
        or canonical.get("writer_boundary") != {"enabled": True}
        or canonical.get("discord_edge") != {"enabled": True}
        or canonical.get("tools_enabled") is not True
    ):
        raise ProductionContractError("production_writer_boundary_not_exact")
    if raw.get("plugins") != {"enabled": [], "disabled": []}:
        raise ProductionContractError("production_plugin_config_not_empty")
    if raw.get("hooks") != {} or raw.get("hooks_auto_accept") is not False:
        raise ProductionContractError("production_hook_config_not_empty")
    if raw.get("mcp_servers") != {}:
        raise ProductionContractError("production_mcp_config_not_empty")
    if raw.get("fallback_model") != [] or raw.get("fallback_providers") != []:
        raise ProductionContractError("production_fallback_route_not_empty")
    if raw.get("provider_routing") != {}:
        raise ProductionContractError("production_provider_routing_not_empty")
    if raw.get("production_access") != production_access_config():
        raise ProductionContractError("production_access_policy_not_exact")
    if raw.get("quick_commands") != {}:
        raise ProductionContractError("production_quick_commands_not_empty")
    if raw.get("command_allowlist") != []:
        raise ProductionContractError("production_command_allowlist_not_empty")
    skills = raw.get("skills")
    if not isinstance(skills, Mapping) or skills.get("inline_shell") is not False:
        raise ProductionContractError("production_skills_inline_shell_not_disabled")
    security = raw.get("security")
    if (
        not isinstance(security, Mapping)
        or security.get("allow_lazy_installs") is not False
    ):
        raise ProductionContractError("production_lazy_installs_not_disabled")
    if raw.get("web") != _PRODUCTION_WEB_CONFIG:
        raise ProductionContractError("production_web_boundary_not_exact")

    cron = raw.get("cron")
    if (
        not isinstance(cron, Mapping)
        or cron.get("enabled") is not True
        or cron.get("provider") != "builtin"
    ):
        raise ProductionContractError("production_cron_boundary_not_exact")

    approvals = raw.get("approvals")
    if (
        not isinstance(approvals, Mapping)
        or approvals.get("mode") != "manual"
        or approvals.get("cron_mode") != "deny"
        or approvals.get("plan_owner_user_ids")
        != [PRODUCTION_OWNER_DISCORD_USER_ID]
        or approvals.get("gateway_authorized_user_ids")
        != [PRODUCTION_OWNER_DISCORD_USER_ID]
        or approvals.get("gateway_authorized_user_names") != []
        or approvals.get("gateway_authorized_labels") != ["Емо"]
        or approvals.get("gateway_owner_escalation")
        != {
            "enabled": True,
            "owner_user_id": PRODUCTION_OWNER_DISCORD_USER_ID,
            "owner_guild_id": SKYVISION_GUILD_ID,
            "owner_channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
            "owner_target_type": "guild_channel",
        }
    ):
        raise ProductionContractError("production_approvals_not_exact")

    goals = raw.get("goals")
    if (
        not isinstance(goals, Mapping)
        or type(goals.get("max_turns")) is not int
        or goals.get("max_turns") != 0
    ):
        raise ProductionContractError("production_goal_continuation_not_exact")

    try:
        topology = validate_production_capability_topology(
            raw.get("production_capabilities")
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionContractError("production_capability_topology_not_exact") from exc
    if raw.get("terminal") != _terminal_config(topology):
        raise ProductionContractError("production_terminal_boundary_not_exact")
    if raw.get("browser") != _browser_config(topology):
        raise ProductionContractError("production_browser_boundary_not_exact")
    mac_ops = raw.get("mac_ops_edge")
    if not isinstance(mac_ops, Mapping):
        raise ProductionContractError("production_mac_ops_edge_not_exact")
    try:
        MacOpsEdgeClientConfig.from_mapping(mac_ops)
    except (TypeError, ValueError) as exc:
        raise ProductionContractError("production_mac_ops_edge_not_exact") from exc

    if raw.get("platforms") != _PLATFORMS:
        raise ProductionContractError("production_platform_boundary_not_exact")
    gateway = raw.get("gateway")
    if (
        not isinstance(gateway, Mapping)
        or gateway.get("isolated_runtime") is not False
        or gateway.get("relay_url") != RELAY_URL
        or gateway.get("api_server") != {"max_concurrent_runs": 1}
        or gateway.get("platforms") != _PLATFORMS
    ):
        raise ProductionContractError("production_gateway_loop_not_exact")
    if "discord" in raw.get("platforms", {}) or "discord" in gateway.get(
        "platforms", {}
    ):
        raise ProductionContractError("production_direct_discord_forbidden")
    platform_toolsets = raw.get("platform_toolsets")
    if platform_toolsets != _production_platform_toolsets():
        raise ProductionContractError("production_toolsets_not_exact")


def render_production_gateway_config(
    source_bytes: bytes,
    *,
    expected_source_sha256: str,
) -> bytes:
    if _SHA256.fullmatch(expected_source_sha256 or "") is None:
        raise ProductionContractError("production_source_config_sha256_invalid")
    if _sha256(source_bytes) != expected_source_sha256:
        raise ProductionContractError("production_source_config_sha256_mismatch")
    target = overlay_production_gateway_config(
        load_strict_production_config(source_bytes)
    )
    try:
        rendered = yaml.safe_dump(
            target,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=4096,
        ).encode("utf-8")
    except (TypeError, UnicodeError, yaml.YAMLError) as exc:
        raise ProductionContractError("production_config_render_failed") from exc
    if len(rendered) > 2 * 1024 * 1024:
        raise ProductionContractError("production_config_target_oversized")
    reparsed = load_strict_production_config(rendered)
    if reparsed != target:
        raise ProductionContractError("production_config_render_drifted")
    validate_production_gateway_config(reparsed)
    return rendered


def validate_production_cron_jobs(jobs: Sequence[Mapping[str, Any]]) -> None:
    """Admit only exact primary-model scheduled turns.

    Disabled records are inert history. Enabled agent jobs must resolve
    directly to the same pinned GPT-5.6 Codex route as interactive turns; no
    fallback, alternate endpoint, or unpinned provider is accepted.  Local
    no-agent and pre-run scripts are denied because they would execute as the
    credential-bearing gateway UID; an isolated cron worker may reintroduce
    them under a separate reviewed contract.
    """

    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes)):
        raise ProductionContractError("production_cron_jobs_invalid")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ProductionContractError("production_cron_job_invalid")
        if job.get("enabled") is False:
            continue
        no_agent = job.get("no_agent")
        if type(no_agent) is not bool:
            raise ProductionContractError("production_cron_execution_mode_invalid")
        # Production cron runs only the primary model loop.  Require the
        # normalized persisted representation (no script field at all) rather
        # than accepting malformed non-string values that happen not to match a
        # string-only check.
        if no_agent or job.get("script") is not None:
            raise ProductionContractError("production_cron_local_script_forbidden")
        # ``workdir`` is a host path in the generic scheduler: it affects
        # context-file reads before the isolated execution environment is
        # entered.  Keep it absent until production has a server-issued,
        # session-bound workspace lease contract; never accept a caller path.
        if job.get("workdir") is not None:
            raise ProductionContractError("production_cron_workdir_forbidden")
        if not isinstance(job.get("prompt"), str) or not job["prompt"].strip():
            raise ProductionContractError("production_cron_agent_prompt_missing")
        deliver = job.get("deliver")
        if type(deliver) is not str or deliver not in {"local", "origin"}:
            raise ProductionContractError("production_cron_delivery_not_exact")
        if deliver == "origin":
            origin = job.get("origin")
            if (
                not isinstance(origin, Mapping)
                or set(origin) - _PRODUCTION_CRON_ORIGIN_FIELDS
                or origin.get("platform") != "discord"
                or not isinstance(origin.get("chat_id"), str)
                or _DISCORD_SNOWFLAKE.fullmatch(origin["chat_id"]) is None
            ):
                raise ProductionContractError("production_cron_origin_not_exact")
            for optional in ("chat_name", "thread_id", "user_id"):
                value = origin.get(optional)
                if value is not None and not isinstance(value, str):
                    raise ProductionContractError(
                        "production_cron_origin_not_exact"
                    )
            thread_id = origin.get("thread_id")
            if thread_id not in (None, "") and (
                _DISCORD_SNOWFLAKE.fullmatch(thread_id) is None
                or thread_id != origin["chat_id"]
            ):
                raise ProductionContractError(
                    "production_cron_origin_thread_not_exact"
                )
            user_id = origin.get("user_id")
            if user_id not in (None, "") and (
                _DISCORD_SNOWFLAKE.fullmatch(user_id) is None
            ):
                raise ProductionContractError("production_cron_origin_not_exact")
        if (
            job.get("provider") != "openai-codex"
            or job.get("model") != "gpt-5.6-sol"
            or job.get("base_url") not in (None, "")
        ):
            raise ProductionContractError("production_cron_primary_route_not_exact")
        if any(
            job.get(key) not in (None, "", [], {})
            for key in (
                "fallback_model",
                "fallback_models",
                "fallback_provider",
                "fallback_providers",
            )
        ):
            raise ProductionContractError("production_cron_fallback_route_forbidden")

        enabled_toolsets = job.get("enabled_toolsets")
        if enabled_toolsets is None:
            continue
        if (
            type(enabled_toolsets) is not list
            or not enabled_toolsets
            or any(type(name) is not str or not name for name in enabled_toolsets)
            or len(set(enabled_toolsets)) != len(enabled_toolsets)
        ):
            raise ProductionContractError(
                "production_cron_enabled_toolsets_invalid"
            )
        if not set(enabled_toolsets).issubset(FIRST_WAVE_TOOLSETS):
            raise ProductionContractError(
                "production_cron_enabled_toolset_not_allowed"
            )


def resolve_production_cron_enabled_toolsets(
    job: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    """Re-attest and resolve one production cron agent's exact tool surface.

    This is a mechanical schema boundary.  It never infers intent from a job
    prompt or chooses tools for the model.  A job may narrow the reviewed
    first-wave surface with an exact unique list; omission inherits that exact
    surface.  Missing or drifted runtime config always fails closed.
    """

    validate_production_cron_jobs([job])
    if not isinstance(config, Mapping):
        raise ProductionContractError("production_cron_config_invalid")
    platform_toolsets = config.get("platform_toolsets")
    if (
        not isinstance(platform_toolsets, Mapping)
        or platform_toolsets.get("cron") != list(FIRST_WAVE_TOOLSETS)
    ):
        raise ProductionContractError("production_cron_toolsets_not_exact")
    enabled_toolsets = job.get("enabled_toolsets")
    if enabled_toolsets is None:
        return list(FIRST_WAVE_TOOLSETS)
    return list(enabled_toolsets)


def validate_production_provider_registry(provider_registry: Any) -> None:
    """Attest the one-way, single-provider registry after discovery."""

    expected_allowlist = frozenset({"openai-codex"})
    registry = getattr(provider_registry, "_REGISTRY", None)
    aliases = getattr(provider_registry, "_ALIASES", None)
    if (
        getattr(provider_registry, "_discovered", None) is not True
        or getattr(provider_registry, "_discovery_error", None) is not None
        or getattr(provider_registry, "_isolated_provider_allowlist", None)
        != expected_allowlist
        or getattr(provider_registry, "_isolated_discovery_validated", None)
        is not True
        or not isinstance(registry, Mapping)
        or set(registry) != {"openai-codex"}
        or aliases
        != {"codex": "openai-codex", "openai_codex": "openai-codex"}
    ):
        raise ProductionContractError("production_provider_registry_not_exact")
    profile = registry["openai-codex"]
    if (
        getattr(profile, "name", None) != "openai-codex"
        or tuple(getattr(profile, "aliases", ())) != ("codex", "openai_codex")
        or getattr(profile, "api_mode", None) != "codex_responses"
        or getattr(profile, "base_url", None) != _MODEL["base_url"]
        or getattr(profile, "auth_type", None) != "oauth_external"
        or tuple(getattr(profile, "env_vars", ())) != ()
    ):
        raise ProductionContractError("production_provider_profile_not_exact")


def validate_production_extension_surface(
    plugin_manager: Any,
    gateway_hooks: Any,
    provider_registry: Any,
) -> None:
    """Attest one exact web executor and zero behavior-changing extensions."""

    if (
        getattr(plugin_manager, "_discovered", None) is not True
        or getattr(plugin_manager, "_isolated_allowlist", None)
        != PRODUCTION_PLUGIN_ALLOWLIST
        or getattr(plugin_manager, "_isolated_discovery_failure", None) is not None
    ):
        raise ProductionContractError("production_plugin_discovery_not_isolated")
    plugins = getattr(plugin_manager, "_plugins", None)
    if not isinstance(plugins, Mapping) or set(plugins) != PRODUCTION_PLUGIN_ALLOWLIST:
        raise ProductionContractError("production_plugin_surface_not_exact")
    loaded = plugins[PRODUCTION_WEB_PLUGIN_KEY]
    manifest = getattr(loaded, "manifest", None)
    manifest_path = Path(str(getattr(manifest, "path", "")))
    if any(
        (
            getattr(loaded, "enabled", None) is not True,
            getattr(loaded, "error", None) is not None,
            getattr(loaded, "deferred", None) is not False,
            getattr(loaded, "module", None) is None,
            getattr(loaded, "tools_registered", None) != [],
            getattr(loaded, "hooks_registered", None) != [],
            getattr(loaded, "middleware_registered", None) != [],
            getattr(loaded, "commands_registered", None) != [],
            getattr(manifest, "key", None) != PRODUCTION_WEB_PLUGIN_KEY,
            getattr(manifest, "name", None) != "web-ddgs",
            getattr(manifest, "source", None) != "bundled",
            getattr(manifest, "kind", None) != "backend",
            manifest_path.name != "ddgs",
            manifest_path.parent.name != "web",
        )
    ):
        raise ProductionContractError("production_plugin_surface_not_exact")
    empty = {
        "_hooks": {},
        "_middleware": {},
        "_plugin_tool_names": set(),
        "_plugin_platform_names": set(),
        "_cli_commands": {},
        "_plugin_commands": {},
        "_plugin_skills": {},
        "_aux_tasks": {},
        "_slack_action_handlers": [],
    }
    if any(
        getattr(plugin_manager, name, None) != expected
        for name, expected in empty.items()
    ):
        raise ProductionContractError("production_plugin_surface_not_empty")
    if (
        getattr(plugin_manager, "_context_engine", None) is not None
        or getattr(plugin_manager, "_cli_ref", None) is not None
    ):
        raise ProductionContractError("production_plugin_runtime_not_empty")
    if (
        getattr(gateway_hooks, "_handlers", None) != {}
        or getattr(gateway_hooks, "_loaded_hooks", None) != []
    ):
        raise ProductionContractError("production_gateway_hooks_not_empty")

    try:
        from agent.web_search_registry import list_providers
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        web_providers = list_providers()
        distribution_version = importlib_metadata.version(
            PRODUCTION_WEB_DISTRIBUTION
        )
    except Exception as exc:
        raise ProductionContractError("production_web_executor_unavailable") from exc
    if (
        len(web_providers) != 1
        or type(web_providers[0]) is not DDGSWebSearchProvider
        or web_providers[0].name != PRODUCTION_WEB_PROVIDER_NAME
        or web_providers[0].supports_search() is not True
        or web_providers[0].supports_extract() is not False
        or distribution_version != PRODUCTION_WEB_DISTRIBUTION_VERSION
    ):
        raise ProductionContractError("production_web_executor_not_exact")
    try:
        web_available = web_providers[0].is_available()
    except Exception as exc:
        raise ProductionContractError("production_web_executor_unavailable") from exc
    if web_available is not True:
        raise ProductionContractError("production_web_executor_unavailable")
    validate_production_provider_registry(provider_registry)


def validate_production_gateway_adapters(adapters: Any) -> None:
    """Require the exact live loopback API and public Discord relay adapters."""

    from gateway.config import Platform
    from gateway.api_verifier_credentials import (
        APIApprovalScryptVerifier,
        APIBearerVerifier,
    )
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.discord_connector_transport import (
        DiscordConnectorRelayTransport,
    )

    if not isinstance(adapters, Mapping) or set(adapters) != {
        Platform.API_SERVER,
        Platform.RELAY,
    }:
        raise ProductionContractError("production_adapter_set_not_exact")
    api = adapters[Platform.API_SERVER]
    site = getattr(api, "_site", None)
    server = getattr(site, "_server", None)
    if (
        not isinstance(api, APIServerAdapter)
        or getattr(api, "is_connected", None) is not True
        or getattr(api, "_host", None) != API_SERVER_HOST
        or getattr(api, "_port", None) != API_SERVER_PORT
        or getattr(api, "_api_key", None) != ""
        or not isinstance(
            getattr(api, "_api_bearer_verifier", None),
            APIBearerVerifier,
        )
        or getattr(api, "_approval_passkey", None) != ""
        or not isinstance(
            getattr(api, "_approval_passkey_verifier", None),
            APIApprovalScryptVerifier,
        )
        or getattr(api, "_model_routes", None) != {}
        or getattr(api, "_max_concurrent_runs", None) != 1
        or getattr(api, "_app", None) is None
        or getattr(api, "_runner", None) is None
        or site is None
        or server is None
        or not callable(getattr(server, "is_serving", None))
        or not server.is_serving()
    ):
        raise ProductionContractError("production_api_adapter_not_ready")
    relay = adapters[Platform.RELAY]
    if not isinstance(relay, RelayAdapter):
        raise ProductionContractError("production_relay_adapter_not_exact")
    transport = getattr(relay, "_transport", None)
    descriptor = getattr(relay, "descriptor", None)
    poller = getattr(transport, "_poller", None)
    authorizer = getattr(transport, "server_authorizer", None)
    if (
        not isinstance(transport, DiscordConnectorRelayTransport)
        or getattr(transport, "_connected", None) is not True
        or poller is None
        or not callable(getattr(poller, "done", None))
        or poller.done()
        or transport.socket_path != str(CONNECTOR_SOCKET)
        or getattr(authorizer, "server_unit", None) != CONNECTOR_UNIT
        or getattr(descriptor, "platform", None) != "discord"
        or getattr(descriptor, "contract_version", None) != 1
        or getattr(descriptor, "max_message_length", None) != 2_000
        or getattr(descriptor, "supports_draft_streaming", None) is not False
        or getattr(descriptor, "supports_edit", None) is not False
        or getattr(descriptor, "supports_threads", None) is not True
        or getattr(descriptor, "len_unit", None) != "chars"
    ):
        raise ProductionContractError("production_relay_adapter_not_ready")


def _canonical_posix_path(value: str, *, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\x00" in value
        or posixpath.normpath(value) != value
    ):
        raise ProductionContractError(code)
    return value


def validate_production_release_import_identity(
    *,
    revision: str,
    source_commit_bytes: bytes,
    executable: str,
    sys_path: Sequence[str],
    import_origins: Mapping[str, str],
) -> None:
    """Pure validation of the release, interpreter, and import projection."""

    release = str(production_release_root(revision))
    interpreter = f"{release}/.venv/bin/python"
    if source_commit_bytes != f"{revision}\n".encode("ascii"):
        raise ProductionContractError("production_release_revision_mismatch")
    if _canonical_posix_path(executable, code="production_executable_invalid") != interpreter:
        raise ProductionContractError("production_executable_not_revision_bound")
    if (
        not isinstance(sys_path, Sequence)
        or isinstance(sys_path, (str, bytes))
        or not sys_path
    ):
        raise ProductionContractError("production_sys_path_invalid")
    canonical_paths = [
        _canonical_posix_path(item, code="production_sys_path_invalid")
        for item in sys_path
    ]
    if canonical_paths[0] != release:
        raise ProductionContractError("production_release_not_first_on_sys_path")
    for item in canonical_paths[1:]:
        under_release = item == release or item.startswith(release + "/")
        if not under_release and (
            "/site-packages" in item
            or "/dist-packages" in item
            or "/hermes-agent" in item
            or "/hermes-agent-releases/" in item
        ):
            raise ProductionContractError("production_editable_import_path_present")

    if not isinstance(import_origins, Mapping) or set(import_origins) != set(
        REQUIRED_IMPORT_ORIGINS
    ):
        raise ProductionContractError("production_import_origin_set_not_exact")
    for module_name, relative in REQUIRED_IMPORT_ORIGINS.items():
        observed = _canonical_posix_path(
            import_origins[module_name], code="production_import_origin_invalid"
        )
        if observed != f"{release}/{relative}":
            raise ProductionContractError("production_import_origin_not_revision_bound")


def _read_stable_source_marker(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionContractError("production_release_marker_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= 128
        ):
            raise ProductionContractError("production_release_marker_invalid")
        raw = os.read(descriptor, 129)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ProductionContractError("production_release_marker_changed")
    return raw


def attest_current_production_release_import_identity(
    *, revision: str,
) -> dict[str, Any]:
    """Collect and validate the current process import identity before READY."""

    release = production_release_root(revision)
    try:
        release_lstat = os.lstat(release)
        resolved_release = release.resolve(strict=True)
    except OSError as exc:
        raise ProductionContractError("production_release_unavailable") from exc
    if (
        not stat.S_ISDIR(release_lstat.st_mode)
        or stat.S_ISLNK(release_lstat.st_mode)
        or resolved_release != release
    ):
        raise ProductionContractError("production_release_identity_invalid")
    source_commit = _read_stable_source_marker(release / ".codex-source-commit")
    origins: dict[str, str] = {}
    for module_name in REQUIRED_IMPORT_ORIGINS:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ProductionContractError("production_required_import_missing")
        try:
            origin_lstat = os.lstat(module_file)
            resolved_origin = Path(module_file).resolve(strict=True)
        except OSError as exc:
            raise ProductionContractError("production_import_origin_unavailable") from exc
        if not stat.S_ISREG(origin_lstat.st_mode) or stat.S_ISLNK(origin_lstat.st_mode):
            raise ProductionContractError("production_import_origin_invalid")
        origins[module_name] = str(resolved_origin)
    validate_production_release_import_identity(
        revision=revision,
        source_commit_bytes=source_commit,
        executable=sys.executable,
        sys_path=list(sys.path),
        import_origins=origins,
    )
    unsigned = {
        "schema": "muncho-production-release-import-attestation.v1",
        "release_revision": revision,
        "release_root": str(release),
        "source_commit_sha256": _sha256(source_commit),
        "executable": sys.executable,
        "import_origins": origins,
        "sys_path_sha256": _sha256(
            _canonical_bytes({"entries": list(sys.path)})
        ),
    }
    return {
        **unsigned,
        "attestation_sha256": _sha256(_canonical_bytes(unsigned)),
    }


def validate_production_gateway_environment(
    environment: Mapping[str, str],
    *,
    revision: str,
    config_sha256: str,
    topology: Mapping[str, Any],
) -> None:
    """Validate static startup environment without constraining systemd fields."""

    release = str(production_release_root(revision))
    if _SHA256.fullmatch(config_sha256 or "") is None:
        raise ProductionContractError("production_config_sha256_invalid")
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ProductionContractError("production_environment_invalid")
    try:
        topology = validate_production_capability_topology(topology)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionContractError(
            "production_environment_topology_invalid"
        ) from exc
    worker = topology["isolated_worker"]
    required = {
        "HOME": str(PRODUCTION_HOME),
        "HERMES_CONFIG": str(PRODUCTION_CONFIG_PATH),
        "HERMES_HOME": str(PRODUCTION_HOME),
        "HERMES_MAX_ITERATIONS": "90",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": release,
        "GATEWAY_RELAY_URL": RELAY_URL,
        "GATEWAY_RELAY_PLATFORMS": "discord",
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_CWD": "/workspace",
        "TERMINAL_TIMEOUT": "180",
        "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": worker["socket_path"],
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(worker["server_uid"]),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(worker["server_gid"]),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": str(worker["socket_uid"]),
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(worker["socket_gid"]),
        "CREDENTIALS_DIRECTORY": f"/run/credentials/{GATEWAY_UNIT}",
        "_HERMES_GATEWAY": "1",
    }
    if any(environment.get(name) != value for name, value in required.items()):
        raise ProductionContractError("production_environment_values_drifted")
    if _FORBIDDEN_ENVIRONMENT_NAMES & set(environment) or any(
        name.upper().endswith(("_API_KEY", "_TOKEN", "_PASSWORD", "_SECRET"))
        for name in environment
    ):
        raise ProductionContractError(
            "production_environment_secret_or_override_present"
        )
    allowed = set(required) | set(_SYSTEMD_OPTIONAL_ENVIRONMENT_NAMES)
    unexpected = set(environment) - allowed
    if unexpected:
        raise ProductionContractError("production_environment_name_not_allowed")
    optional = {
        name: environment[name]
        for name in _SYSTEMD_OPTIONAL_ENVIRONMENT_NAMES
        if name in environment
    }
    if (
        (
            "INVOCATION_ID" in optional
            and re.fullmatch(r"[0-9a-f]{32}", optional["INVOCATION_ID"])
            is None
        )
        or (
            "JOURNAL_STREAM" in optional
            and re.fullmatch(r"[0-9]+:[0-9]+", optional["JOURNAL_STREAM"]) is None
        )
        or (
            "NOTIFY_SOCKET" in optional
            and optional["NOTIFY_SOCKET"] != "/run/systemd/notify"
            and re.fullmatch(r"@[A-Za-z0-9_./:-]{1,255}", optional["NOTIFY_SOCKET"])
            is None
        )
        or ("PWD" in optional and optional["PWD"] != release)
        or (
            "RUNTIME_DIRECTORY" in optional
            and optional["RUNTIME_DIRECTORY"] != "/run/hermes-cloud-gateway"
        )
        or ("SHELL" in optional and optional["SHELL"] != "/usr/sbin/nologin")
        or (
            "SYSTEMD_EXEC_PID" in optional
            and (
                not optional["SYSTEMD_EXEC_PID"].isdigit()
                or int(optional["SYSTEMD_EXEC_PID"]) <= 0
            )
        )
        or (
            "SYSTEMD_NSS_BYPASS_SYNTHETIC" in optional
            and optional["SYSTEMD_NSS_BYPASS_SYNTHETIC"] not in {"0", "1"}
        )
        or (
            "USER" in optional
            and _SYSTEMD_IDENTITY.fullmatch(optional["USER"]) is None
        )
        or (
            "LOGNAME" in optional
            and _SYSTEMD_IDENTITY.fullmatch(optional["LOGNAME"]) is None
        )
        or (
            "USER" in optional
            and "LOGNAME" in optional
            and optional["USER"] != optional["LOGNAME"]
        )
    ):
        raise ProductionContractError("production_environment_value_not_allowed")


def _read_stable_production_config() -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(PRODUCTION_CONFIG_PATH, flags)
    except OSError as exc:
        raise ProductionContractError("production_config_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= 2 * 1024 * 1024
        ):
            raise ProductionContractError("production_config_file_identity_invalid")
        raw = os.read(descriptor, 2 * 1024 * 1024 + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ProductionContractError("production_config_file_changed")
    return raw


def attest_current_production_capability_prerequisites(
    *,
    revision: str,
    config_sha256: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Validate the config-bound prerequisite receipt immediately before READY."""

    if _SHA256.fullmatch(config_sha256 or "") is None:
        raise ProductionContractError("production_config_sha256_invalid")
    raw = _read_stable_production_config()
    if _sha256(raw) != config_sha256:
        raise ProductionContractError("production_config_sha256_mismatch")
    config = load_strict_production_config(raw)
    validate_production_gateway_config(config)
    topology = config["production_capabilities"]
    try:
        receipt = load_production_capability_prerequisite_receipt(
            revision=revision,
            topology=topology,
            lifecycle_phase=PREREQUISITE_LIFECYCLE_COMMITTED,
            now_unix=now_unix,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        code = getattr(exc, "code", "production_prerequisite_invalid")
        raise ProductionContractError(str(code)) from exc
    return {
        "schema": receipt["schema"],
        "release_revision": receipt["release_revision"],
        "lifecycle_phase": receipt["lifecycle_phase"],
        "topology_identity_sha256": (
            production_capability_topology_identity_sha256(topology)
        ),
        "boot_id_sha256": receipt["boot_id_sha256"],
        "observed_at_unix": receipt["observed_at_unix"],
        "receipt_sha256": receipt["receipt_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def attest_current_production_gateway_service_identity(
    *,
    revision: str,
    config_sha256: str,
) -> Mapping[str, Any]:
    """Prove this process is the exact generated systemd MainPID pre-READY."""

    if _REVISION.fullmatch(revision or "") is None:
        raise ProductionContractError("production_revision_invalid")
    if _SHA256.fullmatch(config_sha256 or "") is None:
        raise ProductionContractError("production_config_sha256_invalid")
    raw = _read_stable_production_config()
    if _sha256(raw) != config_sha256:
        raise ProductionContractError("production_config_sha256_mismatch")
    config = load_strict_production_config(raw)
    validate_production_gateway_config(config)
    topology = validate_production_capability_topology(
        config["production_capabilities"]
    )
    identity = topology["gateway_identity"]
    try:
        gateway_user = pwd.getpwuid(identity["uid"]).pw_name
        gateway_group = grp.getgrgid(identity["gid"]).gr_name
    except KeyError as exc:
        raise ProductionContractError(
            "production_gateway_identity_unavailable"
        ) from exc
    expected_unit = render_production_gateway_unit(
        revision=revision,
        config_sha256=config_sha256,
        gateway_user=gateway_user,
        gateway_group=gateway_group,
        topology=topology,
    )
    try:
        observed = attest_live_production_gateway_service_identity(
            expected_unit=expected_unit,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        code = getattr(
            exc,
            "code",
            "production_live_gateway_service_identity_invalid",
        )
        raise ProductionContractError(str(code)) from exc
    if (
        observed["effective_uid"] != identity["uid"]
        or observed["effective_gid"] != identity["gid"]
    ):
        raise ProductionContractError("production_gateway_identity_drifted")
    return {
        "unit": observed["unit"],
        "fragment_sha256": observed["fragment_sha256"],
        "unit_service_contract_sha256": observed[
            "unit_service_contract_sha256"
        ],
        "main_pid": observed["main_pid"],
        "main_pid_executable": observed["main_pid_executable"],
        "main_pid_uid": observed["main_pid_uid"],
        "main_pid_gid": observed["main_pid_gid"],
        "main_pid_groups": observed["main_pid_groups"],
        "main_pid_cmdline_sha256": observed["main_pid_cmdline_sha256"],
        "main_pid_cgroup": observed["main_pid_cgroup"],
        "main_pid_mount_namespace_inode": observed[
            "main_pid_mount_namespace_inode"
        ],
        "main_pid_network_namespace_inode": observed[
            "main_pid_network_namespace_inode"
        ],
        "process_identity_matches_unit": True,
        "ready_not_yet_published": True,
    }


def render_production_gateway_unit(
    *,
    revision: str,
    config_sha256: str,
    gateway_user: str,
    gateway_group: str,
    topology: Mapping[str, Any],
) -> bytes:
    release = production_release_root(revision)
    if _SHA256.fullmatch(config_sha256 or "") is None:
        raise ProductionContractError("production_config_sha256_invalid")
    if _SYSTEMD_IDENTITY.fullmatch(gateway_user or "") is None:
        raise ProductionContractError("production_gateway_user_invalid")
    if _SYSTEMD_IDENTITY.fullmatch(gateway_group or "") is None:
        raise ProductionContractError("production_gateway_group_invalid")
    try:
        topology = validate_production_capability_topology(topology)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ProductionContractError("production_capability_topology_not_exact") from exc
    interpreter = release / ".venv/bin/python"
    source_marker = release / ".codex-source-commit"
    unset = " ".join(sorted(_FORBIDDEN_ENVIRONMENT_NAMES))
    terminal = _terminal_config(topology)
    worker = topology["isolated_worker"]
    browser = topology["browser"]
    service_dependencies = " ".join(
        (
            PHASE_B_UNIT,
            ROUTEBACK_EDGE_UNIT,
            CONNECTOR_UNIT,
            MAC_OPS_UNIT,
            worker["socket_unit"],
            worker["service_unit"],
            browser["unit"],
            WRITER_UNIT,
        )
    )
    bound_dependencies = " ".join(
        (
            ROUTEBACK_EDGE_UNIT,
            CONNECTOR_UNIT,
            MAC_OPS_UNIT,
            worker["socket_unit"],
            worker["service_unit"],
            browser["unit"],
            WRITER_UNIT,
        )
    )
    lines = [
        "# Exact SHA-bound Cloud Muncho production gateway; do not edit.",
        f"# ReleaseRevision={revision}",
        f"# ConfigSHA256={config_sha256}",
        "# DiscordCredentialInGateway=false",
        "[Unit]",
        "Description=Cloud Muncho GPT-5.6 model-sovereignty gateway",
        "Wants=network-online.target",
        f"Requires={service_dependencies}",
        f"BindsTo={bound_dependencies}",
        f"After=network-online.target {service_dependencies}",
        f"AssertPathExists={PRODUCTION_CONFIG_PATH}",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={source_marker}",
        f"AssertPathIsDirectory={WRITER_RUNTIME}",
        f"AssertPathIsDirectory={CONNECTOR_RUNTIME}",
        f"AssertPathExists={PHASE_B_RECEIPT_PATH}",
        f"AssertPathExists={ROUTEBACK_EDGE_CONFIG_PATH}",
        f"AssertPathExists={ROUTEBACK_EDGE_SOCKET_PATH}",
        f"AssertPathExists={MAC_OPS_CONFIG_PATH}",
        f"AssertPathExists={MAC_OPS_SOCKET_PATH}",
        f"AssertPathExists={worker['config_path']}",
        f"AssertPathExists={worker['socket_path']}",
        f"AssertPathExists={browser['config_path']}",
        f"AssertPathExists={browser['socket_path']}",
        f"AssertPathExists={CODEX_AUTH_PATH}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={gateway_user}",
        f"Group={gateway_group}",
        (
            "SupplementaryGroups=muncho-writer-client muncho-discord-egress "
            "muncho-discord-connector muncho-mac-ops-edge "
            f"{ISOLATED_WORKER_CLIENT_GROUP} {BROWSER_SOCKET_GROUP}"
        ),
        f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:{API_SERVER_CREDENTIAL_PATH}",
        (
            f"LoadCredential={API_APPROVAL_CREDENTIAL_NAME}:"
            f"{API_APPROVAL_CREDENTIAL_PATH}"
        ),
        "RuntimeDirectory=hermes-cloud-gateway",
        "RuntimeDirectoryMode=0700",
        "RuntimeDirectoryPreserve=no",
        f"WorkingDirectory={release}",
        f"Environment=HOME={PRODUCTION_HOME}",
        f"Environment=HERMES_CONFIG={PRODUCTION_CONFIG_PATH}",
        f"Environment=HERMES_HOME={PRODUCTION_HOME}",
        "Environment=HERMES_MAX_ITERATIONS=90",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=PYTHONNOUSERSITE=1",
        f"Environment=PYTHONPATH={release}",
        f"Environment=GATEWAY_RELAY_URL={RELAY_URL}",
        "Environment=GATEWAY_RELAY_PLATFORMS=discord",
        "Environment=TERMINAL_ENV=isolated_worker",
        "Environment=TERMINAL_CWD=/workspace",
        "Environment=TERMINAL_TIMEOUT=180",
        "Environment=TERMINAL_HOME_MODE=profile",
        "Environment=TERMINAL_LIFETIME_SECONDS=900",
        (
            "Environment=TERMINAL_ISOLATED_WORKER_SOCKET="
            f"{terminal['isolated_worker_socket']}"
        ),
        (
            "Environment=TERMINAL_ISOLATED_WORKER_SERVER_UID="
            f"{terminal['isolated_worker_server_uid']}"
        ),
        (
            "Environment=TERMINAL_ISOLATED_WORKER_SERVER_GID="
            f"{terminal['isolated_worker_server_gid']}"
        ),
        (
            "Environment=TERMINAL_ISOLATED_WORKER_SOCKET_UID="
            f"{terminal['isolated_worker_socket_uid']}"
        ),
        (
            "Environment=TERMINAL_ISOLATED_WORKER_SOCKET_GID="
            f"{terminal['isolated_worker_socket_gid']}"
        ),
        "Environment=_HERMES_GATEWAY=1",
        f"UnsetEnvironment={unset}",
        (
            f"ExecStartPre=+{interpreter} -B -P -s -m "
            "gateway.production_capability_prerequisites collect "
            f"--revision {revision} --config-sha256 {config_sha256} "
            f"--lifecycle-phase {PREREQUISITE_LIFECYCLE_COMMITTED}"
        ),
        (
            f"ExecStart={interpreter} -B -P -s -m gateway.run "
            f"--config {PRODUCTION_CONFIG_PATH} {STARTUP_FLAG} "
            f"--production-release-revision {revision} "
            f"--production-config-sha256 {config_sha256}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=180s",
        "TimeoutStopSec=90s",
        "KillMode=mixed",
        "LimitCORE=0",
        "UMask=0077",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "PrivateDevices=yes",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectHostname=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "IPAddressDeny=169.254.169.254/32",
        f"ReadOnlyPaths={release}",
        f"ReadOnlyPaths={PRODUCTION_CONFIG_PATH}",
        f"ReadOnlyPaths={WRITER_RUNTIME}",
        f"ReadOnlyPaths={CONNECTOR_RUNTIME}",
        f"ReadOnlyPaths={PHASE_B_RECEIPT_PATH}",
        f"ReadOnlyPaths={ROUTEBACK_EDGE_CONFIG_PATH}",
        f"ReadOnlyPaths={ROUTEBACK_EDGE_SOCKET_PATH.parent}",
        f"ReadOnlyPaths={MAC_OPS_CONFIG_PATH}",
        f"ReadOnlyPaths={MAC_OPS_SOCKET_PATH.parent}",
        f"ReadOnlyPaths={worker['config_path']}",
        f"ReadOnlyPaths={Path(worker['socket_path']).parent}",
        f"ReadOnlyPaths={browser['config_path']}",
        f"ReadOnlyPaths={Path(browser['socket_path']).parent}",
        *(f"InaccessiblePaths={browser[field]}" for field in (
            "node_path",
            "wrapper_path",
            "native_path",
            "executable",
            "agent_browser_config_path",
        )),
        f"InaccessiblePaths=-{PRODUCTION_HOME}/.discord-token",
        f"InaccessiblePaths=-{LEGACY_API_BEARER_SOURCE_PATH}",
        f"InaccessiblePaths=-{OWNER_APPROVAL_PASSKEY_SOURCE_PATH}",
        "InaccessiblePaths=-/etc/muncho/discord-connector-credentials",
        "InaccessiblePaths=-/etc/muncho/discord-edge-credentials",
        "InaccessiblePaths=-/etc/muncho/mac-ops-edge-credentials",
        "ReadWritePaths=/run/hermes-cloud-gateway",
        f"ReadWritePaths={CODEX_AUTH_PATH}",
        f"ReadWritePaths={PREREQUISITE_PATH.parent}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    unit = "\n".join(lines).encode("utf-8")
    text = unit.decode("utf-8")
    if any(
        marker in text
        for marker in (
            "--require-capability-canary",
            "--require-canonical-writer",
            "EnvironmentFile=",
            "PassEnvironment=",
            "DISCORD_BOT_TOKEN=",
            "Restart=always",
            "TERMINAL_DOCKER_",
            "docker.sock",
            "remote-debugging-port",
            "127.0.0.1:9222",
        )
    ):
        raise ProductionContractError("production_gateway_unit_unsafe")
    return unit


@dataclass(frozen=True)
class ProductionGatewayContract:
    revision: str
    release_root: Path
    source_config_sha256: str
    config_bytes: bytes
    config_sha256: str
    unit_bytes: bytes
    unit_sha256: str
    gateway_user: str
    gateway_group: str

    def public_manifest(self) -> dict[str, Any]:
        unsigned = {
            "schema": CONTRACT_SCHEMA,
            "release_revision": self.revision,
            "release_root": str(self.release_root),
            "interpreter": str(self.release_root / ".venv/bin/python"),
            "source_config_sha256": self.source_config_sha256,
            "target_config_path": str(PRODUCTION_CONFIG_PATH),
            "target_config_sha256": self.config_sha256,
            "target_unit": GATEWAY_UNIT,
            "target_unit_sha256": self.unit_sha256,
            "gateway_user": self.gateway_user,
            "gateway_group": self.gateway_group,
            "startup_contract": STARTUP_FLAG,
            "required_services": [
                PHASE_B_UNIT,
                ROUTEBACK_EDGE_UNIT,
                CONNECTOR_UNIT,
                MAC_OPS_UNIT,
                ISOLATED_WORKER_SOCKET_UNIT,
                ISOLATED_WORKER_SERVICE_UNIT,
                BROWSER_UNIT,
                WRITER_UNIT,
            ],
            "writer_required": True,
            "connector_required": True,
            "routeback_edge_required": True,
            "mac_ops_edge_required": True,
            "browser_controller_required": True,
            "isolated_worker_required": True,
            "codex_credential_lease_required": True,
            "api_loopback": f"http://{API_SERVER_HOST}:{API_SERVER_PORT}",
            "api_control_credential": API_SERVER_CREDENTIAL_NAME,
            "api_approval_owner_credential": API_APPROVAL_CREDENTIAL_NAME,
            "api_positive_approval_requires_owner_authority": True,
            "toolsets": list(FIRST_WAVE_TOOLSETS),
            "host_workspace_projection_enabled": False,
            "direct_discord_enabled": False,
            "discord_dm_allowed": False,
            "relay_url": RELAY_URL,
            "normal_agent_loop": True,
            "primary_model_cron_jobs_allowed": True,
            "mcp_auto_discovery": False,
            "plugin_allowlist": sorted(PRODUCTION_PLUGIN_ALLOWLIST),
            "provider_allowlist": ["openai-codex"],
            "secret_material_recorded": False,
        }
        return {
            **unsigned,
            "contract_sha256": _sha256(_canonical_bytes(unsigned)),
        }


def produce_production_gateway_contract(
    source_config_bytes: bytes,
    *,
    expected_source_sha256: str,
    revision: str,
    gateway_user: str,
    gateway_group: str,
) -> ProductionGatewayContract:
    config = render_production_gateway_config(
        source_config_bytes,
        expected_source_sha256=expected_source_sha256,
    )
    config_sha256 = _sha256(config)
    unit = render_production_gateway_unit(
        revision=revision,
        config_sha256=config_sha256,
        gateway_user=gateway_user,
        gateway_group=gateway_group,
        topology=load_strict_production_config(config)["production_capabilities"],
    )
    return ProductionGatewayContract(
        revision=revision,
        release_root=production_release_root(revision),
        source_config_sha256=expected_source_sha256,
        config_bytes=config,
        config_sha256=config_sha256,
        unit_bytes=unit,
        unit_sha256=_sha256(unit),
        gateway_user=gateway_user,
        gateway_group=gateway_group,
    )


__all__ = [
    "CONNECTOR_UNIT",
    "CONTRACT_SCHEMA",
    "GATEWAY_UNIT",
    "PRODUCTION_CONFIG_PATH",
    "PRODUCTION_HOME",
    "ProductionContractError",
    "ProductionGatewayContract",
    "RELAY_URL",
    "REQUIRED_IMPORT_ORIGINS",
    "STARTUP_FLAG",
    "WRITER_UNIT",
    "attest_current_production_capability_prerequisites",
    "attest_current_production_gateway_service_identity",
    "attest_current_production_release_import_identity",
    "load_strict_production_config",
    "overlay_production_gateway_config",
    "produce_production_gateway_contract",
    "production_browser_controller_client_config",
    "production_release_root",
    "render_production_gateway_config",
    "render_production_gateway_unit",
    "resolve_production_cron_enabled_toolsets",
    "validate_production_cron_jobs",
    "validate_production_capability_prerequisite_receipt",
    "validate_production_capability_topology",
    "validate_production_extension_surface",
    "validate_production_gateway_adapters",
    "validate_production_gateway_config",
    "validate_production_gateway_environment",
    "validate_production_provider_registry",
    "validate_production_release_import_identity",
]
