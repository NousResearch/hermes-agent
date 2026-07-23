"""REST payload contracts for the desktop/dashboard API.

These PEP 257-documented TypedDicts are the source of truth for generated
TypeScript gateway contracts. They intentionally describe plain JSON values only.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

from hermes_cli.contract_types import OpaqueValue
from tui_gateway.contracts import GatewaySessionRuntimeInfo

class ConfigFieldSchema(TypedDict):
    category: NotRequired[str]
    description: NotRequired[str]
    options: NotRequired[list[OpaqueValue]]
    type: NotRequired[Literal['boolean'] | Literal['list'] | Literal['number'] | Literal['select'] | Literal['string'] | Literal['text']]

class ConfigSchemaResponse(TypedDict):
    category_order: NotRequired[list[str]]
    fields: dict[str, ConfigFieldSchema]

class AudioTranscriptionResponse(TypedDict):
    ok: bool
    provider: NotRequired[str]
    transcript: str

class AudioSpeakResponse(TypedDict):
    ok: bool
    data_url: str
    mime_type: str
    provider: NotRequired[str]

class ElevenLabsVoice(TypedDict):
    label: str
    name: str
    voice_id: str

class ElevenLabsVoicesResponse(TypedDict):
    available: bool
    voices: list[ElevenLabsVoice]

class OAuthProviderStatus(TypedDict):
    error: NotRequired[str]
    expires_at: NotRequired[None | str]
    has_refresh_token: NotRequired[bool]
    last_refresh: NotRequired[None | str]
    logged_in: bool
    source: NotRequired[None | str]
    source_label: NotRequired[None | str]
    token_preview: NotRequired[None | str]

class OAuthProvider(TypedDict):
    cli_command: str
    disconnect_command: NotRequired[None | str]
    """Shell command that clears an external provider's credentials, run in the
 embedded terminal. Null when Hermes doesn't know how to remove it."""
    disconnect_hint: NotRequired[None | str]
    disconnectable: NotRequired[bool]
    docs_url: str
    flow: Literal['device_code'] | Literal['external'] | Literal['pkce']
    id: str
    name: str
    status: OAuthProviderStatus

class OAuthProvidersResponse(TypedDict):
    providers: list[OAuthProvider]

class OAuthSubmitResponse(TypedDict):
    message: NotRequired[str]
    ok: bool
    status: Literal['approved'] | Literal['error']

class OAuthPollResponse(TypedDict):
    error_message: NotRequired[None | str]
    expires_at: NotRequired[None | float]
    session_id: str
    status: Literal['approved'] | Literal['denied'] | Literal['error'] | Literal['expired'] | Literal['pending']

class MemoryProviderOAuthStatus(TypedDict):
    auth: Literal['apikey'] | Literal['oauth'] | None
    connected: bool
    detail: str
    state: Literal['connected'] | Literal['error'] | Literal['idle'] | Literal['pending']

class EnvVarInfo(TypedDict):
    advanced: bool
    category: str
    channel_managed: NotRequired[bool]
    description: str
    is_password: bool
    is_set: bool
    provider: NotRequired[str]
    provider_label: NotRequired[str]
    redacted_value: None | str
    tools: list[str]
    url: None | str

MemoryProviderFieldKind: TypeAlias = Literal['bool'] | Literal['json'] | Literal['number'] | Literal['secret'] | Literal['select'] | Literal['text']

class MemoryProviderFieldOption(TypedDict):
    description: str
    label: str
    value: str

class MemoryProviderField(TypedDict):
    description: str
    group: str
    info: NotRequired[str]
    inline: bool
    is_set: bool
    key: str
    kind: MemoryProviderFieldKind
    label: str
    options: list[MemoryProviderFieldOption]
    placeholder: str
    value: str

class MemoryProviderConfig(TypedDict):
    docs_url: str
    fields: list[MemoryProviderField]
    label: str
    name: str

class CustomEndpoint(TypedDict):
    api_key_preview: NotRequired[None | str]
    base_url: str
    context_length: NotRequired[None | float]
    discover_models: bool
    has_api_key: bool
    id: str
    is_current: NotRequired[bool]
    model: str
    models: list[str]
    name: str
    source: NotRequired[str]

class CustomEndpointUpdate(TypedDict):
    api_key: NotRequired[str]
    base_url: str
    context_length: NotRequired[float]
    discover_models: NotRequired[bool]
    id: NotRequired[str]
    make_default: NotRequired[bool]
    model: str
    name: str

class CustomEndpointValidationResponse(TypedDict):
    message: str
    models: list[str]
    ok: bool
    reachable: bool

class MessagingEnvVarInfo(TypedDict):
    advanced: bool
    description: str
    is_password: bool
    is_set: bool
    key: str
    prompt: str
    redacted_value: None | str
    required: bool
    url: None | str

class MessagingHomeChannel(TypedDict):
    chat_id: str
    name: str
    platform: str
    thread_id: NotRequired[str]

class MessagingPlatformInfo(TypedDict):
    configured: bool
    description: str
    docs_url: str
    enabled: bool
    env_vars: list[MessagingEnvVarInfo]
    error_code: NotRequired[None | str]
    error_message: NotRequired[None | str]
    gateway_running: bool
    home_channel: NotRequired[MessagingHomeChannel | None]
    id: str
    name: str
    state: NotRequired[None | str]
    updated_at: NotRequired[None | str]

class MessagingPlatformsResponse(TypedDict):
    platforms: list[MessagingPlatformInfo]

class MessagingPlatformUpdate(TypedDict):
    clear_env: NotRequired[list[str]]
    enabled: NotRequired[bool]
    env: NotRequired[dict[str, str]]

class MessagingPlatformTestResponse(TypedDict):
    message: str
    ok: bool
    state: NotRequired[None | str]

class GatewayReadyPayload(TypedDict):
    skin: NotRequired[OpaqueValue]

class ModelInfoResponse(TypedDict):
    auto_context_length: NotRequired[float]
    capabilities: NotRequired[dict[str, OpaqueValue]]
    config_context_length: NotRequired[float]
    effective_context_length: NotRequired[float]
    model: str
    provider: str

class ModelPricing(TypedDict):
    input: str
    """Formatted $/Mtok input price, e.g. "$3.00", or "free", or "" if unknown."""
    output: str
    """Formatted $/Mtok output price."""
    cache: str | None
    """Formatted $/Mtok cached-input price, or null when the model has none."""
    free: bool
    """True when the model costs nothing (free tier eligible)."""
    discount_percent: NotRequired[float]
    """Sale: rounded percent off list when gateway sends pricing.original."""
    was_input: NotRequired[str]
    """Sale: formatted pre-discount input $/Mtok ("was")."""
    was_output: NotRequired[str]
    """Sale: formatted pre-discount output $/Mtok ("was")."""

class ModelOptionProvider(TypedDict):
    is_current: NotRequired[bool]
    models: NotRequired[list[str]]
    name: str
    slug: str
    total_models: NotRequired[float]
    warning: NotRequired[str]
    authenticated: NotRequired[bool]
    """True when the provider has usable credentials. False for canonical
 providers surfaced by `include_unconfigured` that the user hasn't set up
 yet — render these with a setup affordance instead of hiding them."""
    auth_type: NotRequired[str]
    """Auth flow for an unconfigured provider: "api_key" can be activated inline
 by pasting `key_env`; anything else (oauth_*, external, aws_sdk, …) needs
 the `hermes model` CLI / onboarding OAuth flow."""
    key_env: NotRequired[str]
    """Env var to paste an API key into, for unconfigured `api_key` providers."""
    is_user_defined: NotRequired[bool]
    """True for providers defined via the user's `providers:` config block."""
    pricing: NotRequired[dict[str, ModelPricing]]
    """Per-model pricing keyed by model id (present when the picker requested
 pricing and the provider supports live pricing)."""
    free_tier: NotRequired[bool]
    """Nous only: whether the current account is on the free tier."""
    unavailable_models: NotRequired[list[str]]
    """Nous only: paid models a free-tier user cannot select (shown disabled)."""
    capabilities: NotRequired[dict[str, ModelCapabilities]]
    """Per-model option support, keyed by model id (present when the picker
 requested capabilities). Lets the UI gate fast/reasoning controls."""

class ModelCapabilities(TypedDict):
    fast: bool
    reasoning: bool

class ModelOptionsResponse(TypedDict):
    model: NotRequired[str]
    provider: NotRequired[str]
    providers: NotRequired[list[ModelOptionProvider]]

class SessionCreateResponse(TypedDict):
    info: NotRequired[GatewaySessionRuntimeInfo]
    message_count: NotRequired[float]
    messages: NotRequired[list[SessionMessage]]
    session_id: str
    stored_session_id: NotRequired[str]

class SessionInfo(TypedDict):
    archived: NotRequired[bool]
    cwd: NotRequired[None | str]
    git_branch: NotRequired[None | str]
    """Git branch checked out in {@link cwd} when the session started/resumed.
 The sidebar groups main-checkout sessions by this so feature-branch work
 doesn't collapse under a single directory-named "main" row. Null for
 non-git workspaces and sessions created before branch capture landed."""
    git_repo_root: NotRequired[None | str]
    """Git repo root that owns {@link cwd} — the authoritative project key,
 resolved server-side at cwd-set (and backfilled for history). The sidebar
 groups by this instead of probing git in the GUI. Null for non-git
 workspaces and not-yet-backfilled rows."""
    ended_at: None | float
    id: str
    _lineage_root_id: NotRequired[None | str]
    """Original root id of a compression chain, when this entry is a projected
 continuation tip. Stable across compressions — used as the durable id for
 pins so a pinned conversation survives auto-compression."""
    input_tokens: float
    is_active: bool
    last_active: float
    message_count: float
    model: None | str
    output_tokens: float
    parent_session_id: NotRequired[None | str]
    """Parent conversation when this row is a /branch fork."""
    preview: None | str
    source: None | str
    started_at: float
    title: None | str
    tool_call_count: float
    handoff_platform: NotRequired[None | str]
    """Origin platform when this session was handed off from a messaging
 platform (e.g. a Telegram thread continued in the desktop app). The live
 {@link source} becomes local (tui/desktop) after a handoff, so the origin
 is preserved here to surface the platform badge on the row."""
    handoff_state: NotRequired[None | str]
    """Handoff lifecycle: 'pending' | 'in_progress' | 'completed' | 'failed'."""
    handoff_error: NotRequired[None | str]
    profile: NotRequired[str]
    """Owning profile name, set by the cross-profile aggregator
 (`/api/profiles/sessions`). Absent on legacy single-profile responses,
 which the UI treats as the default profile."""
    is_default_profile: NotRequired[bool]
    """True when {@link profile} is the default profile."""

class TimelineModelDisplayMetadata(TypedDict):
    """Model-switch metadata attached to a timeline display event."""

    model: str
    provider: NotRequired[str]


class TimelineDelegationDisplayMetadata(TypedDict):
    """Delegated-task completion metadata attached to a timeline event."""

    delegation_id: str
    task_count: float
    completed_count: NotRequired[float]
    failed_count: NotRequired[float]
    duration_seconds: NotRequired[float]


TimelineDisplayMetadata: TypeAlias = (
    TimelineModelDisplayMetadata | TimelineDelegationDisplayMetadata
)


class SessionMessage(TypedDict):
    codex_reasoning_items: NotRequired[OpaqueValue]
    content: OpaqueValue
    context: NotRequired[OpaqueValue]
    name: NotRequired[str]
    reasoning: NotRequired[None | str]
    reasoning_content: NotRequired[None | str]
    reasoning_details: NotRequired[OpaqueValue]
    display_kind: NotRequired[Literal['async_delegation_complete'] | Literal['hidden'] | Literal['model_switch'] | str]
    display_metadata: NotRequired[TimelineDisplayMetadata]
    role: Literal['assistant'] | Literal['system'] | Literal['tool'] | Literal['user']
    text: NotRequired[OpaqueValue]
    timestamp: NotRequired[float]
    tool_call_id: NotRequired[None | str]
    tool_calls: NotRequired[OpaqueValue]
    tool_name: NotRequired[str]

class SessionMessagesResponse(TypedDict):
    messages: list[SessionMessage]
    session_id: str

class UsageStats(TypedDict):
    calls: float
    context_max: NotRequired[float]
    context_percent: NotRequired[float]
    context_used: NotRequired[float]
    cost_usd: NotRequired[float]
    input: float
    output: float
    total: float

class StarmapNode(TypedDict):
    """One graph node in the star map (learned skill or memory chunk)."""

    id: str
    label: str
    kind: Literal['memory'] | Literal['skill']
    memorySource: NotRequired[Literal['memory'] | Literal['profile']]
    timestamp: NotRequired[None | float]
    category: str
    useCount: float
    state: str
    createdBy: None | str
    pinned: bool

class StarmapEdge(TypedDict):
    """A declared `related_skills` link; both endpoints are guaranteed to be nodes."""

    source: str
    target: str

class StarmapCluster(TypedDict):
    category: str
    count: float

class StarmapMemoryCard(TypedDict):
    """Freeform memory rendered as a card — never a graph node."""

    source: Literal['memory'] | Literal['profile']
    timestamp: NotRequired[None | float]
    title: str
    body: str

class StarmapGraph(TypedDict):
    nodes: list[StarmapNode]
    edges: list[StarmapEdge]
    clusters: list[StarmapCluster]
    memory: list[StarmapMemoryCard]
    stats: dict[str, OpaqueValue]

class ContextUsageCategory(TypedDict):
    color: str
    id: str
    label: str
    tokens: float

class ContextBreakdown(TypedDict):
    categories: list[ContextUsageCategory]
    context_max: float
    context_percent: float
    context_used: float
    estimated_total: float
    model: NotRequired[str]

class AnalyticsDailyEntry(TypedDict):
    actual_cost: float
    api_calls: float
    cache_read_tokens: float
    day: str
    estimated_cost: float
    input_tokens: float
    output_tokens: float
    reasoning_tokens: float
    sessions: float

class AnalyticsModelEntry(TypedDict):
    api_calls: float
    estimated_cost: float
    input_tokens: float
    model: str
    output_tokens: float
    sessions: float

class AnalyticsToolEntry(TypedDict):
    count: float
    percentage: float
    tool: str

class AnalyticsSkillEntry(TypedDict):
    last_used_at: None | float
    manage_count: float
    percentage: float
    skill: str
    total_count: float
    view_count: float

class AnalyticsSkillsSummary(TypedDict):
    distinct_skills_used: float
    total_skill_actions: float
    total_skill_edits: float
    total_skill_loads: float

class AnalyticsTotals(TypedDict):
    total_actual_cost: float
    total_api_calls: None | float
    total_cache_read: None | float
    total_estimated_cost: float
    total_input: None | float
    total_output: None | float
    total_reasoning: None | float
    total_sessions: float

class CronJob(TypedDict):
    deliver: NotRequired[None | str]
    enabled: bool
    id: str
    last_error: NotRequired[None | str]
    last_run_at: NotRequired[None | str]
    model: NotRequired[None | str]
    name: NotRequired[None | str]
    next_run_at: NotRequired[None | str]
    no_agent: NotRequired[bool]
    prompt: NotRequired[None | str]
    provider: NotRequired[None | str]
    schedule: NotRequired[CronJobSchedule]
    schedule_display: NotRequired[None | str]
    script: NotRequired[None | str]
    state: NotRequired[None | str]

class CronJobCreatePayload(TypedDict):
    deliver: NotRequired[str]
    model: NotRequired[str]
    name: NotRequired[str]
    prompt: str
    provider: NotRequired[str]
    schedule: str

class CronJobSchedule(TypedDict):
    display: NotRequired[str]
    expr: NotRequired[str]
    kind: NotRequired[str]

class CronJobUpdates(TypedDict):
    deliver: NotRequired[str]
    enabled: NotRequired[bool]
    model: NotRequired[None | str]
    name: NotRequired[str]
    prompt: NotRequired[str]
    provider: NotRequired[None | str]
    schedule: NotRequired[str]

class ProfileCreatePayload(TypedDict):
    clone_all: NotRequired[bool]
    clone_from: NotRequired[None | str]
    clone_from_default: NotRequired[bool]
    name: str
    no_skills: NotRequired[bool]

class ProfileInfo(TypedDict):
    has_env: bool
    is_default: bool
    model: None | str
    name: str
    path: str
    provider: None | str
    skill_count: float

class ProfileSetupCommand(TypedDict):
    command: str

class ProjectFolder(TypedDict):
    path: str
    label: None | str
    is_primary: bool
    added_at: float

class ProjectInfo(TypedDict):
    id: str
    slug: str
    name: str
    description: None | str
    icon: None | str
    color: None | str
    board_slug: None | str
    primary_path: None | str
    archived: bool
    created_at: float
    folders: list[ProjectFolder]

class ProjectsPayload(TypedDict):
    projects: list[ProjectInfo]
    active_id: None | str

class ProfileSoul(TypedDict):
    content: str
    exists: bool

class ProfilesResponse(TypedDict):
    profiles: list[ProfileInfo]

class SkillInfo(TypedDict):
    category: str
    description: str
    enabled: bool
    name: str
    usage: NotRequired[float]
    """Total observed activity (use + view + patch). Absent on older backends."""
    provenance: NotRequired[Literal['agent'] | Literal['bundled'] | Literal['hub']]
    """'agent' = learned/local (editable), 'bundled' = ships with Hermes, 'hub' = installed."""

class ToolsetInfo(TypedDict):
    configured: bool
    description: str
    enabled: bool
    label: str
    name: str
    tools: list[str]

class ToolEnvVar(TypedDict):
    key: str
    prompt: str
    url: str | None
    default: str | None
    is_set: bool

# Server-computed readiness for a provider picker row. Absent on older
# backends that predate the truthful-readiness endpoint.
ToolProviderStatus: TypeAlias = Literal['ready'] | Literal['needs_setup'] | Literal['needs_auth'] | Literal['needs_keys']

class ToolProvider(TypedDict):
    name: str
    badge: str
    tag: str
    env_vars: list[ToolEnvVar]
    post_setup: str | None
    requires_nous_auth: bool
    is_active: bool
    """True when this is the provider currently written to config (mirrors the
 CLI `hermes tools` active-provider detection)."""
    status: NotRequired[ToolProviderStatus]
    """Honest readiness computed server-side (keys ∧ Nous entitlement ∧
 post-setup install state). Optional for older backends."""
    web_backend: NotRequired[str]
    """Web toolset only: the backend key written to web.*backend config
 (e.g. 'searxng'). Absent on other toolsets and older backends."""
    tts_provider: NotRequired[str]
    """TTS toolset only: the provider key written to tts.provider when this row
 is selected (e.g. 'openai'). Doubles as the config section that holds the
 provider's voice/model settings (tts.<key>.*). Absent on other toolsets
 and older backends."""
    capabilities: NotRequired[list[WebCapability]]
    """Web toolset only: capabilities this backend can serve. Search-only
 providers (ddgs, brave-free) report ['search']."""

# A web toolset capability — the runtime dispatches web_search and
# web_extract to independently configurable backends.
WebCapability: TypeAlias = Literal['search'] | Literal['extract']

class ToolsetConfig(TypedDict):
    name: str
    has_category: bool
    providers: list[ToolProvider]
    active_provider: str | None
    """Name of the currently active provider, or null if none is configured."""
    active_search_backend: NotRequired[str | None]
    """Web toolset only: backend the web_search tool resolves to right now
 (web.search_backend → web.backend → credential auto-detect)."""
    active_extract_backend: NotRequired[str | None]
    """Web toolset only: backend the web_extract tool resolves to right now."""

# Health status of a terminal execution backend row.
#
# `ready` — usable now; `needs_setup` — selectable but missing a dependency
# or credential (detail says which); `unavailable` — the probe itself failed.
TerminalBackendStatus: TypeAlias = Literal['ready'] | Literal['needs_setup'] | Literal['unavailable']

class TerminalBackendInfo(TypedDict):
    """One row from `GET /api/tools/terminal/backends`."""

    name: str
    label: str
    description: str
    active: bool
    """True when this backend is the current `terminal.backend` config value."""
    status: TerminalBackendStatus
    detail: str
    """Setup guidance / probe detail for non-ready rows (empty when ready)."""

class TerminalBackendsResponse(TypedDict):
    """Shape of `GET /api/tools/terminal/backends`."""

    active: str
    backends: list[TerminalBackendInfo]

class ToolsetModel(TypedDict):
    """One model row from a toolset backend's catalog (image/video gen)."""

    id: str
    display: str
    speed: str
    strengths: str
    price: str

class ToolsetModelsResponse(TypedDict):
    """Shape of `GET /api/tools/toolsets/{name}/models`."""

    name: str
    has_models: bool
    provider: NotRequired[str | None]
    plugin: NotRequired[str | None]
    models: list[ToolsetModel]
    current: str | None
    default: str | None

class ComputerUsePermissionSource(TypedDict):
    """Shape of `GET /api/tools/computer-use/status`.

 cua-driver runs on macOS, Windows, and Linux. `ready` is the single OS-aware
 readiness signal: on macOS both TCC grants (Accessibility + Screen
 Recording, which attach to cua-driver's own `com.trycua.driver` identity,
 not Hermes); elsewhere, driver health from `cua-driver doctor`. `null`
 means unknown (binary missing / probe failed)."""

    attribution: NotRequired[str]
    executable: NotRequired[str]
    note: NotRequired[str]
    pid: NotRequired[float]
    responsible_ppid: NotRequired[float]

class ComputerUseCheck(TypedDict):
    label: str
    status: str
    message: str

class ComputerUseStatus(TypedDict):
    platform: str
    """`sys.platform`: "darwin" | "win32" | "linux" | ..."""
    platform_supported: bool
    """cua-driver has a runtime backend for this platform."""
    installed: bool
    """cua-driver binary resolved on PATH."""
    version: str | None
    """e.g. "cua-driver 0.5.1", or null when unknown."""
    ready: bool | None
    """Unified readiness — both TCC grants (macOS) or driver health (else)."""
    can_grant: bool
    """Whether a permission grant flow exists (macOS-only TCC)."""
    checks: list[ComputerUseCheck]
    """Cross-platform `cua-driver doctor` probes."""
    accessibility: bool | None
    """macOS TCC detail — `null` off macOS or when unknown."""
    screen_recording: bool | None
    screen_recording_capturable: bool | None
    source: ComputerUsePermissionSource | None
    error: str | None
    """Populated when the status probe itself failed."""

class SessionSearchResult(TypedDict):
    lineage_root: NotRequired[str | None]
    """Lineage root of the matched conversation. Stable across compression and
 used as the durable pin id; falls back to session_id when absent."""
    model: str | None
    role: str | None
    session_id: str
    """Live compression tip of the matched conversation — resume by this id."""
    session_started: float | None
    snippet: str
    source: str | None

class SessionSearchResponse(TypedDict):
    results: list[SessionSearchResult]

class LogsResponse(TypedDict):
    file: str
    lines: list[str]

class PlatformStatus(TypedDict):
    error_code: NotRequired[str]
    error_message: NotRequired[str]
    state: str
    updated_at: str

class StatusResponse(TypedDict):
    active_sessions: float
    config_path: str
    config_version: float
    env_path: str
    gateway_exit_reason: str | None
    gateway_health_url: str | None
    gateway_pid: float | None
    gateway_platforms: dict[str, PlatformStatus]
    gateway_running: bool
    gateway_state: str | None
    gateway_updated_at: str | None
    hermes_home: str
    latest_config_version: float
    release_date: str
    version: str

class ActionResponse(TypedDict):
    name: str
    ok: bool
    pid: float

class ActionStatusResponse(TypedDict):
    exit_code: float | None
    lines: list[str]
    name: str
    pid: float | None
    running: bool

class BackendUpdateCommit(TypedDict):
    sha: str
    summary: str
    author: str
    at: float

class BackendUpdateCheckResponse(TypedDict):
    """Shape of `GET /api/hermes/update/check` — the backend's own update state.
 Used by the desktop's remote update overlay so the backend version (not the
 Electron client clone) drives "what's changed + Install" in remote mode."""

    install_method: str
    current_version: str
    behind: float | None
    update_available: bool
    can_apply: bool
    update_command: str | None
    message: str | None
    commits: NotRequired[list[BackendUpdateCommit]]

class AuxiliaryTaskAssignment(TypedDict):
    base_url: str
    model: str
    provider: str
    task: str

class MoaModelSlot(TypedDict):
    provider: str
    model: str
    reasoning_effort: NotRequired[str]
    """Optional per-slot reasoning effort — round-tripped, not edited here."""

class ModelAssignmentRequest(TypedDict):
    api_key: NotRequired[str]
    """Optional API key for a custom/local endpoint. Persisted to model.api_key
 (where the runtime reads it) for self-hosted endpoints that require auth.
 Only honored for custom/local providers on the main slot."""
    base_url: NotRequired[str]
    """OpenAI-compatible endpoint URL. Only honored for custom/local providers
 on the main slot — wires a self-hosted endpoint into runtime resolution."""
    model: str
    provider: str
    scope: Literal['main'] | Literal['auxiliary']
    task: NotRequired[str]

class StaleAuxAssignment(TypedDict):
    """An auxiliary task still pinned to a provider that differs from the
 newly-selected main provider after a main-model switch."""

    task: str
    provider: str
    model: str

class SkillHubSource(TypedDict):
    """One skill-hub source (official index, GitHub, skills.sh, …) as reported by
 `GET /api/skills/hub/sources`."""

    id: str
    label: str
    available: NotRequired[bool]
    rate_limited: NotRequired[bool]
    searchable: NotRequired[bool]

class SkillHubResult(TypedDict):
    """A searchable/installable hub skill from `GET /api/skills/hub/search`."""

    name: str
    description: str
    source: str
    identifier: str
    trust_level: str
    repo: str | None
    tags: list[str]

class SkillHubInstalledEntry(TypedDict):
    name: str | None
    trust_level: str | None
    scan_verdict: str | None

class SkillHubSourcesResponse(TypedDict):
    sources: list[SkillHubSource]
    index_available: bool
    featured: list[SkillHubResult]
    installed: dict[str, SkillHubInstalledEntry]

class SkillHubSearchResponse(TypedDict):
    results: list[SkillHubResult]
    source_counts: dict[str, float]
    timed_out: list[str]
    installed: dict[str, SkillHubInstalledEntry]

class SkillHubPreview(TypedDict):
    """`GET /api/skills/hub/preview` — SKILL.md + manifest without installing."""

    name: str
    description: str
    source: str
    identifier: str
    trust_level: str
    repo: str | None
    tags: list[str]
    skill_md: str
    files: list[str]

class SkillHubScanFinding(TypedDict):
    severity: str
    category: str
    file: str
    line: float | None
    description: str

class SkillHubScanResult(TypedDict):
    """`GET /api/skills/hub/scan` — install-time security scan verdict."""

    name: str
    identifier: str
    source: str
    trust_level: str
    verdict: str
    summary: str
    policy: Literal['allow'] | Literal['ask'] | Literal['block']
    policy_reason: str | None
    findings: list[SkillHubScanFinding]
    severity_counts: dict[str, float]

class McpServerSummary(TypedDict):
    """One configured MCP server row from `GET /api/mcp/servers`."""

    name: str
    transport: str
    command: str | None
    args: list[str]
    url: str | None
    enabled: bool
    tools: list[str] | None

class CuratorStatusResponse(TypedDict):
    """`GET /api/curator` — background skill-curator status."""

    enabled: bool
    paused: bool
    interval_hours: float | None
    last_run_at: str | None
    min_idle_hours: float | None
    stale_after_days: float | None
    archive_after_days: float | None

class DebugShareResponse(TypedDict):
    """`POST /api/ops/debug-share` — shareable diagnostics upload result."""

    ok: bool
    urls: dict[str, str]
    failures: dict[str, str]
    redacted: bool
    auto_delete_seconds: float | None

class ModelAssignmentResponse(TypedDict):
    base_url: NotRequired[str]
    """Persisted endpoint URL for custom/local providers (echoed back)."""
    gateway_tools: NotRequired[list[str]]
    """Toolset keys auto-routed through the Nous Tool Gateway as a result of
 switching the main provider to Nous. Empty unless provider === 'nous'
 and the user is a paid subscriber with unconfigured tools."""
    model: NotRequired[str]
    ok: bool
    provider: NotRequired[str]
    reset: NotRequired[bool]
    scope: NotRequired[str]
    stale_aux: NotRequired[list[StaleAuxAssignment]]
    """Auxiliary slots still pinned to a different provider than the new main.
 Switching main never clears aux pins; this lets the UI warn the user
 their helper tasks aren't following the switch. Only set on scope:'main'."""
    tasks: NotRequired[list[str]]

EXPORTED_CONTRACT_NAMES = (
    ConfigFieldSchema,
    ConfigSchemaResponse,
    AudioTranscriptionResponse,
    AudioSpeakResponse,
    ElevenLabsVoice,
    ElevenLabsVoicesResponse,
    OAuthProviderStatus,
    OAuthProvider,
    OAuthProvidersResponse,
    OAuthSubmitResponse,
    OAuthPollResponse,
    MemoryProviderOAuthStatus,
    EnvVarInfo,
    MemoryProviderFieldKind,
    MemoryProviderFieldOption,
    MemoryProviderField,
    MemoryProviderConfig,
    CustomEndpoint,
    CustomEndpointUpdate,
    CustomEndpointValidationResponse,
    MessagingEnvVarInfo,
    MessagingHomeChannel,
    MessagingPlatformInfo,
    MessagingPlatformsResponse,
    MessagingPlatformUpdate,
    MessagingPlatformTestResponse,
    GatewayReadyPayload,
    ModelInfoResponse,
    ModelPricing,
    ModelOptionProvider,
    ModelCapabilities,
    ModelOptionsResponse,
    SessionCreateResponse,
    SessionInfo,
    SessionMessage,
    SessionMessagesResponse,
    UsageStats,
    StarmapNode,
    StarmapEdge,
    StarmapCluster,
    StarmapMemoryCard,
    StarmapGraph,
    ContextUsageCategory,
    ContextBreakdown,
    AnalyticsDailyEntry,
    AnalyticsModelEntry,
    AnalyticsToolEntry,
    AnalyticsSkillEntry,
    AnalyticsSkillsSummary,
    AnalyticsTotals,
    CronJob,
    CronJobCreatePayload,
    CronJobSchedule,
    CronJobUpdates,
    ProfileCreatePayload,
    ProfileInfo,
    ProfileSetupCommand,
    ProjectFolder,
    ProjectInfo,
    ProjectsPayload,
    ProfileSoul,
    ProfilesResponse,
    SkillInfo,
    ToolsetInfo,
    ToolEnvVar,
    ToolProviderStatus,
    ToolProvider,
    WebCapability,
    ToolsetConfig,
    TerminalBackendStatus,
    TerminalBackendInfo,
    TerminalBackendsResponse,
    ToolsetModel,
    ToolsetModelsResponse,
    ComputerUsePermissionSource,
    ComputerUseCheck,
    ComputerUseStatus,
    SessionSearchResult,
    SessionSearchResponse,
    LogsResponse,
    PlatformStatus,
    StatusResponse,
    ActionResponse,
    ActionStatusResponse,
    BackendUpdateCommit,
    BackendUpdateCheckResponse,
    AuxiliaryTaskAssignment,
    MoaModelSlot,
    ModelAssignmentRequest,
    StaleAuxAssignment,
    SkillHubSource,
    SkillHubResult,
    SkillHubInstalledEntry,
    SkillHubSourcesResponse,
    SkillHubSearchResponse,
    SkillHubPreview,
    SkillHubScanFinding,
    SkillHubScanResult,
    McpServerSummary,
    CuratorStatusResponse,
    DebugShareResponse,
    ModelAssignmentResponse,
)

EXPORTED_CONTRACTS = {
    contract.__name__: contract
    for contract in EXPORTED_CONTRACT_NAMES
    if isinstance(contract, type)
} | {
    "MemoryProviderFieldKind": MemoryProviderFieldKind,
    "ToolProviderStatus": ToolProviderStatus,
    "WebCapability": WebCapability,
    "TerminalBackendStatus": TerminalBackendStatus,
}
