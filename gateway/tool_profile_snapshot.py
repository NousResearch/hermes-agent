"""Tool capability ontology and profile snapshots for Hermes gateway.

This module is intentionally pure/observe-only. It describes tool surfaces so
later phases can choose right-sized context without changing live defaults here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "hermes.tool_profile_snapshot.v1"

PROFILE_VERSIONS: dict[str, str] = {
    "conversation_direct": "conversation_direct.v1",
    "conversation_tools": "conversation_tools.v1",
    "heavy_work": "heavy_work.v1",
    "heavy_read": "heavy_read.v1",
    "heavy_action": "heavy_action.v1",
    "heavy_full_debug": "heavy_full_debug.v1",
}

TOOL_LOADER_DENIED_REASON_CODES = [
    "LOADER_UNAVAILABLE_FOR_PROFILE",
    "TOOL_NOT_IN_ALLOWED_ENUM",
    "TOOL_CONFIG_DISABLED",
    "TOOL_GATED_UNAVAILABLE",
    "TOOL_SIDE_EFFECT_APPROVAL_REQUIRED",
    "INVALID_TOOL_LIFETIME",
]

SECRET_SHAPED_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{12,}|api[_-]?key|token|secret|password|bearer\s+[a-z0-9._-]{12,})"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_description(text: str, *, max_chars: int = 160) -> str:
    clean = " ".join((text or "").split())
    clean = SECRET_SHAPED_RE.sub("[REDACTED_SECRET_SHAPED]", clean)
    if len(clean) > max_chars:
        return clean[: max_chars - 1].rstrip() + "..."
    return clean


def estimate_json_tokens(value: Any) -> int:
    """Cheap deterministic estimate; exact provider tokenizer is phase-152 work."""

    return max(1, len(_canonical_json(value)) // 4)


def tool_name_from_schema(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return function["name"]
    if isinstance(schema.get("name"), str):
        return schema["name"]
    return "unknown_tool"


def tool_description_from_schema(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("description"), str):
        return function["description"]
    if isinstance(schema.get("description"), str):
        return schema["description"]
    return ""


def normalize_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI-style and provider-native tool schemas for hashing."""

    name = tool_name_from_schema(schema)
    if "function" in schema:
        return dict(schema)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool_description_from_schema(schema),
            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
        },
    }


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    capability_class: str
    io_class: str
    risk_class: str
    latency_class: str
    data_exposure_class: str
    side_effect_class: str
    requires_human_approval: bool
    approval_scope: str
    secrets_possible: bool
    model_facing_response_cap_tokens: int
    overlap_group: str
    schema_tokens_est: int
    toolsets: tuple[str, ...] = ()
    gated: bool = False
    config_disabled: bool = False
    unknown_metadata: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["toolsets"] = list(self.toolsets)
        return data


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    capability_class: str
    risk_class: str
    io_class: str
    side_effect_class: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolLoaderContract:
    schema: str
    profile_name: str
    loader_available: bool
    allowed_tool_enum: tuple[str, ...]
    allowed_tool_enum_hash: str
    compact_catalog: tuple[ToolCatalogEntry, ...]
    allowed_lifetimes: tuple[str, ...]
    default_lifetime: str | None
    turn_end_cleanup_required: bool
    session_end_cleanup_required: bool
    profile_change_cleanup_required: bool
    side_effect_approval_preserved: bool
    denied_reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_tool_enum"] = list(self.allowed_tool_enum)
        data["compact_catalog"] = [entry.to_dict() for entry in self.compact_catalog]
        data["allowed_lifetimes"] = list(self.allowed_lifetimes)
        data["denied_reason_codes"] = list(self.denied_reason_codes)
        return data


@dataclass(frozen=True)
class ToolProfileSnapshot:
    schema: str
    profile_name: str
    profile_version: str
    tool_names: tuple[str, ...]
    tool_count: int
    tool_schema_hash: str
    system_prompt_hash: str
    static_prefix_hash: str
    tool_loader_allowed_enum_hash: str | None
    cache_candidate: bool
    provider_cache_observable: bool
    provider_reported_cached_tokens: int | None
    cache_break_reason: str | None
    profile_immutability_window: str
    tool_schema_tokens_est: int
    tool_choice_risk: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_names"] = list(self.tool_names)
        return data


@dataclass(frozen=True)
class ToolRiskSummary:
    schema: str
    profile_name: str
    tool_count: int
    tool_schema_tokens_est: int
    overlap_groups: dict[str, list[str]]
    read_action_mix: dict[str, int]
    data_exposure_mix: dict[str, int]
    side_effect_tools: list[str]
    approval_required_tools: list[str]
    unknown_metadata_tools: list[str]
    wrong_tool_risk: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_tool_metadata(
    name: str,
    schema: Mapping[str, Any] | None = None,
    *,
    toolsets: Sequence[str] = (),
    gated: bool = False,
    config_disabled: bool = False,
) -> ToolMetadata:
    """Classify a tool by registry metadata/name.

    This is a static capability ontology, not a natural-language intent router.
    """

    schema = schema or {}
    lname = name.lower()
    description = _safe_description(tool_description_from_schema(schema))
    schema_tokens_est = estimate_json_tokens(normalize_tool_schema(schema)) if schema else 1

    capability = "unknown"
    io_class = "read_only"
    risk = "low"
    latency = "local"
    data_exposure = "unknown"
    side_effect = "none"
    requires_approval = False
    approval_scope = "none"
    secrets_possible = False
    response_cap = 900
    overlap_group = "unknown"

    if lname in {"brainstack_recall", "brainstack_inspect", "brainstack_stats", "session_search"}:
        capability = "memory.recall"
        data_exposure = "user_private"
        risk = "medium"
        overlap_group = "memory_retrieval"
        response_cap = 900 if lname != "session_search" else 700
    elif lname in {"memory", "brainstack_remember", "brainstack_supersede", "brainstack_workstream_recap"}:
        capability = "memory.write"
        io_class = "read_write"
        data_exposure = "user_private"
        side_effect = "memory_write_possible"
        risk = "medium"
        overlap_group = "memory_mutation"
        approval_scope = "per_profile_policy"
    elif lname in {"brainstack_consolidate"}:
        capability = "memory.maintenance"
        io_class = "read_write"
        data_exposure = "user_private"
        side_effect = "maintenance_write_possible"
        risk = "medium"
        overlap_group = "memory_maintenance"
        approval_scope = "per_profile_policy"
    elif lname in {"read_file", "search_files"}:
        capability = "file.read"
        data_exposure = "local_private"
        secrets_possible = True
        risk = "medium"
        overlap_group = "file_access"
        response_cap = 1000
    elif lname in {"write_file", "patch"}:
        capability = "file.write"
        io_class = "side_effect"
        data_exposure = "local_private"
        side_effect = "local_write"
        secrets_possible = True
        requires_approval = True
        approval_scope = "per_call"
        risk = "high"
        overlap_group = "file_access"
        response_cap = 900
    elif lname in {"terminal", "process", "execute_code"}:
        capability = "code.execute"
        io_class = "side_effect"
        data_exposure = "local_private"
        side_effect = "local_execute"
        secrets_possible = True
        requires_approval = True
        approval_scope = "per_call"
        risk = "high"
        latency = "heavy"
        overlap_group = "code_execution"
        response_cap = 1200
    elif lname in {"web_search", "web_extract"} or lname.startswith("browser_"):
        capability = "web.browse"
        data_exposure = "public_web"
        risk = "medium"
        latency = "external_network"
        overlap_group = "web_browser"
        response_cap = 1200
    elif lname.startswith("discord") or lname == "send_message":
        capability = "messaging"
        io_class = "side_effect"
        data_exposure = "user_private_account"
        side_effect = "external_action"
        requires_approval = True
        approval_scope = "per_call"
        risk = "high"
        latency = "external_network"
        overlap_group = "messaging"
        response_cap = 900
    elif lname.startswith("ha_"):
        capability = "home_automation"
        data_exposure = "user_private_account"
        risk = "medium"
        latency = "external_network"
        overlap_group = "home_automation"
        response_cap = 900
        if lname == "ha_call_service":
            io_class = "side_effect"
            side_effect = "external_action"
            requires_approval = True
            approval_scope = "per_call"
            risk = "high"
    elif lname.startswith("feishu_"):
        capability = "workspace_document"
        data_exposure = "workspace_private"
        latency = "external_network"
        risk = "medium"
        overlap_group = "workspace_document"
        if any(part in lname for part in ("add", "reply", "write", "create", "delete")):
            io_class = "side_effect"
            side_effect = "external_action"
            requires_approval = True
            approval_scope = "per_call"
            risk = "high"
    elif lname.startswith("rl_"):
        capability = "reinforcement_learning"
        data_exposure = "workspace_private"
        latency = "heavy"
        risk = "medium"
        overlap_group = "rl"
        if any(part in lname for part in ("start", "stop", "edit", "select")):
            io_class = "side_effect"
            side_effect = "external_action"
            requires_approval = True
            approval_scope = "per_call"
            risk = "high"
    elif lname in {"skills_list", "skill_view"}:
        capability = "skills.read"
        data_exposure = "local_private"
        overlap_group = "skills"
    elif lname == "skill_manage":
        capability = "skills.write"
        io_class = "side_effect"
        data_exposure = "local_private"
        side_effect = "local_write"
        requires_approval = True
        approval_scope = "per_call"
        risk = "high"
        overlap_group = "skills"
    elif lname == "delegate_task":
        capability = "delegation"
        io_class = "side_effect"
        data_exposure = "workspace_private"
        side_effect = "agent_spawn"
        requires_approval = True
        approval_scope = "per_turn"
        risk = "high"
        latency = "heavy"
        overlap_group = "delegation"
    elif lname == "mixture_of_agents":
        capability = "multi_agent_synthesis"
        io_class = "side_effect"
        data_exposure = "workspace_private"
        side_effect = "agent_spawn"
        requires_approval = True
        approval_scope = "per_turn"
        risk = "high"
        latency = "heavy"
        overlap_group = "delegation"
    elif lname == "clarify":
        capability = "user_clarification"
        data_exposure = "user_private"
        overlap_group = "clarification"
        response_cap = 400
    elif lname == "todo":
        capability = "planning.todo"
        io_class = "read_write"
        data_exposure = "local_private"
        side_effect = "local_write"
        risk = "medium"
        overlap_group = "planning"
    elif lname == "cronjob":
        capability = "scheduler"
        io_class = "side_effect"
        data_exposure = "workspace_private"
        side_effect = "scheduler_action"
        requires_approval = True
        approval_scope = "per_workflow"
        risk = "high"
        latency = "external_network"
        overlap_group = "scheduler"
    elif lname in {"vision_analyze", "image_generate", "text_to_speech"}:
        capability = "media"
        data_exposure = "user_private"
        risk = "medium"
        latency = "external_network"
        overlap_group = "media"
        response_cap = 1000
    elif lname.startswith("spotify_"):
        capability = "media.spotify"
        data_exposure = "user_private_account"
        risk = "medium"
        latency = "external_network"
        overlap_group = "spotify"
        response_cap = 900
        if lname in {"spotify_playback", "spotify_queue"}:
            io_class = "side_effect"
            side_effect = "external_action"
            requires_approval = True
            approval_scope = "per_call"
            risk = "high"
    else:
        capability = "unknown"
        risk = "unknown"
        unknown = True
        return ToolMetadata(
            name=name,
            capability_class=capability,
            io_class=io_class,
            risk_class=risk,
            latency_class=latency,
            data_exposure_class=data_exposure,
            side_effect_class=side_effect,
            requires_human_approval=requires_approval,
            approval_scope=approval_scope,
            secrets_possible=secrets_possible,
            model_facing_response_cap_tokens=response_cap,
            overlap_group=overlap_group,
            schema_tokens_est=schema_tokens_est,
            toolsets=tuple(sorted(set(toolsets))),
            gated=gated,
            config_disabled=config_disabled,
            unknown_metadata=unknown,
            description=description,
        )

    return ToolMetadata(
        name=name,
        capability_class=capability,
        io_class=io_class,
        risk_class=risk,
        latency_class=latency,
        data_exposure_class=data_exposure,
        side_effect_class=side_effect,
        requires_human_approval=requires_approval,
        approval_scope=approval_scope,
        secrets_possible=secrets_possible,
        model_facing_response_cap_tokens=response_cap,
        overlap_group=overlap_group,
        schema_tokens_est=schema_tokens_est,
        toolsets=tuple(sorted(set(toolsets))),
        gated=gated,
        config_disabled=config_disabled,
        unknown_metadata=False,
        description=description,
    )


def build_tool_registry(
    schemas: Sequence[Mapping[str, Any]],
    *,
    toolset_memberships: Mapping[str, Sequence[str]] | None = None,
    gated_tools: Sequence[str] = (),
    config_disabled_tools: Sequence[str] = (),
) -> dict[str, ToolMetadata]:
    memberships = toolset_memberships or {}
    gated = set(gated_tools)
    disabled = set(config_disabled_tools)
    registry: dict[str, ToolMetadata] = {}
    for schema in schemas:
        normalized = normalize_tool_schema(schema)
        name = tool_name_from_schema(normalized)
        registry[name] = classify_tool_metadata(
            name,
            normalized,
            toolsets=memberships.get(name, ()),
            gated=name in gated,
            config_disabled=name in disabled,
        )
    return dict(sorted(registry.items()))


def _compact_catalog(registry: Mapping[str, ToolMetadata], names: Sequence[str]) -> tuple[ToolCatalogEntry, ...]:
    entries: list[ToolCatalogEntry] = []
    for name in sorted(set(names)):
        meta = registry.get(name)
        if not meta:
            continue
        entries.append(
            ToolCatalogEntry(
                name=name,
                capability_class=meta.capability_class,
                risk_class=meta.risk_class,
                io_class=meta.io_class,
                side_effect_class=meta.side_effect_class,
                description=meta.description,
            )
        )
    return tuple(entries)


def build_tool_loader_contract(
    profile_name: str,
    profile_tool_names: Sequence[str],
    registry: Mapping[str, ToolMetadata],
    *,
    loader_available: bool | None = None,
) -> ToolLoaderContract:
    if loader_available is None:
        loader_available = profile_name not in {"conversation_direct"}

    enum = tuple(sorted(set(profile_tool_names))) if loader_available else ()
    return ToolLoaderContract(
        schema=SCHEMA_VERSION,
        profile_name=profile_name,
        loader_available=loader_available,
        allowed_tool_enum=enum,
        allowed_tool_enum_hash=_hash_json(enum) if enum else "",
        compact_catalog=_compact_catalog(registry, enum),
        allowed_lifetimes=("ephemeral", "pinned") if loader_available else (),
        default_lifetime="ephemeral" if loader_available else None,
        turn_end_cleanup_required=loader_available,
        session_end_cleanup_required=loader_available,
        profile_change_cleanup_required=loader_available,
        side_effect_approval_preserved=True,
        denied_reason_codes=tuple(TOOL_LOADER_DENIED_REASON_CODES),
    )


def evaluate_tool_load_request(
    contract: ToolLoaderContract,
    registry: Mapping[str, ToolMetadata],
    tool_name: str,
    *,
    lifetime: str = "ephemeral",
    approval_granted: bool = False,
) -> tuple[bool, str]:
    if not contract.loader_available:
        return False, "LOADER_UNAVAILABLE_FOR_PROFILE"
    if tool_name not in contract.allowed_tool_enum:
        return False, "TOOL_NOT_IN_ALLOWED_ENUM"
    if lifetime not in contract.allowed_lifetimes:
        return False, "INVALID_TOOL_LIFETIME"
    meta = registry.get(tool_name)
    if meta is None:
        return False, "TOOL_NOT_IN_ALLOWED_ENUM"
    if meta.config_disabled:
        return False, "TOOL_CONFIG_DISABLED"
    if meta.gated:
        return False, "TOOL_GATED_UNAVAILABLE"
    if meta.requires_human_approval and not approval_granted:
        return False, "TOOL_SIDE_EFFECT_APPROVAL_REQUIRED"
    return True, "ALLOWED"


def build_tool_risk_summary(
    profile_name: str,
    registry: Mapping[str, ToolMetadata],
    tool_names: Sequence[str],
) -> ToolRiskSummary:
    metas = [registry[name] for name in sorted(set(tool_names)) if name in registry]
    groups: dict[str, list[str]] = {}
    read_action: dict[str, int] = {}
    exposures: dict[str, int] = {}
    side_effect_tools: list[str] = []
    approval_tools: list[str] = []
    unknown_tools: list[str] = []
    token_total = 0
    for meta in metas:
        groups.setdefault(meta.overlap_group, []).append(meta.name)
        read_action[meta.io_class] = read_action.get(meta.io_class, 0) + 1
        exposures[meta.data_exposure_class] = exposures.get(meta.data_exposure_class, 0) + 1
        token_total += meta.schema_tokens_est
        if meta.side_effect_class != "none":
            side_effect_tools.append(meta.name)
        if meta.requires_human_approval:
            approval_tools.append(meta.name)
        if meta.unknown_metadata:
            unknown_tools.append(meta.name)

    max_overlap = max((len(v) for v in groups.values()), default=0)
    if unknown_tools or any(meta.risk_class == "high" for meta in metas) or max_overlap >= 6:
        wrong_tool_risk = "high"
    elif len(metas) >= 10 or max_overlap >= 3:
        wrong_tool_risk = "medium"
    else:
        wrong_tool_risk = "low"

    return ToolRiskSummary(
        schema=SCHEMA_VERSION,
        profile_name=profile_name,
        tool_count=len(metas),
        tool_schema_tokens_est=token_total,
        overlap_groups={k: sorted(v) for k, v in sorted(groups.items())},
        read_action_mix=dict(sorted(read_action.items())),
        data_exposure_mix=dict(sorted(exposures.items())),
        side_effect_tools=sorted(side_effect_tools),
        approval_required_tools=sorted(approval_tools),
        unknown_metadata_tools=sorted(unknown_tools),
        wrong_tool_risk=wrong_tool_risk,
    )


def build_tool_profile_snapshot(
    profile_name: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    registry: Mapping[str, ToolMetadata],
    *,
    profile_version: str | None = None,
    system_prompt: str = "",
    static_prefix: str = "",
    provider_cache_observable: bool = False,
    provider_reported_cached_tokens: int | None = None,
    previous_static_prefix_hash: str | None = None,
) -> ToolProfileSnapshot:
    normalized = [normalize_tool_schema(schema) for schema in tool_schemas]
    normalized.sort(key=tool_name_from_schema)
    tool_names = tuple(tool_name_from_schema(schema) for schema in normalized)
    loader = build_tool_loader_contract(profile_name, tool_names, registry)
    system_prompt_hash = sha256(system_prompt.encode("utf-8")).hexdigest()
    static_prefix_hash = sha256(static_prefix.encode("utf-8")).hexdigest()
    tool_schema_hash = _hash_json(normalized)
    risk = build_tool_risk_summary(profile_name, registry, tool_names)
    cache_break_reason = None
    if previous_static_prefix_hash and previous_static_prefix_hash != static_prefix_hash:
        cache_break_reason = "STATIC_PREFIX_HASH_CHANGED_REQUIRES_PROFILE_VERSION_BUMP"

    return ToolProfileSnapshot(
        schema=SCHEMA_VERSION,
        profile_name=profile_name,
        profile_version=profile_version or PROFILE_VERSIONS.get(profile_name, f"{profile_name}.v1"),
        tool_names=tool_names,
        tool_count=len(tool_names),
        tool_schema_hash=tool_schema_hash,
        system_prompt_hash=system_prompt_hash,
        static_prefix_hash=static_prefix_hash,
        tool_loader_allowed_enum_hash=loader.allowed_tool_enum_hash if loader.loader_available else None,
        cache_candidate=True,
        provider_cache_observable=provider_cache_observable,
        provider_reported_cached_tokens=provider_reported_cached_tokens,
        cache_break_reason=cache_break_reason,
        profile_immutability_window="profile_version_bump_required_for_static_prefix_change",
        tool_schema_tokens_est=estimate_json_tokens(normalized),
        tool_choice_risk=risk.wrong_tool_risk,
    )


def render_tool_risk_markdown(summaries: Sequence[ToolRiskSummary]) -> str:
    lines = [
        "# Phase 151 Tool Risk Report",
        "",
        "Observe-only report. No runtime profile switch.",
        "",
        "| profile | tools | schema_tokens_est | wrong_tool_risk | side_effect_tools | approval_required | unknown_metadata |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {profile} | {tools} | {tokens} | {risk} | {side} | {approval} | {unknown} |".format(
                profile=summary.profile_name,
                tools=summary.tool_count,
                tokens=summary.tool_schema_tokens_est,
                risk=summary.wrong_tool_risk,
                side=len(summary.side_effect_tools),
                approval=len(summary.approval_required_tools),
                unknown=len(summary.unknown_metadata_tools),
            )
        )
    lines.append("")
    for summary in summaries:
        lines.append(f"## {summary.profile_name}")
        lines.append("")
        lines.append(f"- overlap_groups: {_canonical_json(summary.overlap_groups)}")
        lines.append(f"- read_action_mix: {_canonical_json(summary.read_action_mix)}")
        lines.append(f"- data_exposure_mix: {_canonical_json(summary.data_exposure_mix)}")
        lines.append(f"- side_effect_tools: {', '.join(summary.side_effect_tools) or 'none'}")
        lines.append(f"- approval_required_tools: {', '.join(summary.approval_required_tools) or 'none'}")
        lines.append(f"- unknown_metadata_tools: {', '.join(summary.unknown_metadata_tools) or 'none'}")
        lines.append("")
    return "\n".join(lines)
