"""Pydantic request/response models for the Hermes dashboard web server."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, SecretStr, field_validator


class ConfigUpdate(BaseModel):
    config: dict
    profile: Optional[str] = None

class EnvVarUpdate(BaseModel):
    key: str
    value: str
    profile: Optional[str] = None
    # Bearer for the OPENAI_BASE_URL connectivity probe (auth-gated /v1/models otherwise looks
    # "reachable but empty"); ignored by plain PUT /api/env.
    api_key: str = ""

class EnvVarDelete(BaseModel):
    key: str
    profile: Optional[str] = None

class EnvVarReveal(EnvVarDelete):
    pass

class MemoryProviderConfigUpdate(BaseModel):
    values: Dict[str, Any] = {}

class MemoryProviderSetupRequest(BaseModel):
    values: Dict[str, Any] = {}

class CustomEndpointUpdate(BaseModel):
    id: str = ""
    name: str
    base_url: str
    model: str
    api_key: Optional[str] = None
    context_length: Optional[int] = None
    discover_models: bool = True
    make_default: bool = False
    models: Optional[List[str]] = None

class MessagingPlatformUpdate(BaseModel):
    enabled: Optional[bool] = None
    env: Dict[str, str] = {}
    clear_env: List[str] = []
    # Explicit body profile beats the switcher's query param (same as other scoped writes).
    profile: Optional[str] = None

class TelegramOnboardingStart(BaseModel):
    bot_name: Optional[str] = None

class TelegramOnboardingApply(BaseModel):
    allowed_user_ids: List[str]
    profile: Optional[str] = None

class WhatsAppOnboardingStart(BaseModel):
    mode: Optional[str] = "bot"
    allowed_users: Optional[str] = ""
    profile: Optional[str] = None

class WhatsAppOnboardingApply(BaseModel):
    mode: Optional[str] = None
    allowed_users: Optional[str] = None
    profile: Optional[str] = None

class AudioTranscriptionRequest(BaseModel):
    data_url: str
    mime_type: Optional[str] = None

class ManagedFileUpload(BaseModel):
    path: str
    data_url: str
    overwrite: bool = True

class ChatImageUpload(BaseModel):
    data_url: str
    filename: Optional[str] = None

class ManagedDirectoryCreate(BaseModel):
    path: str

class ManagedFileDelete(BaseModel):
    path: str
    recursive: bool = False

class ModelAssignment(BaseModel):
    """POST /api/model/set — assign a provider/model to a slot.

    scope="main" → model.provider + model.default; scope="auxiliary" → auxiliary.<task>.*
    (task="" = every auxiliary slot, task="__reset__" = reset every slot to provider="auto").
    """
    scope: str
    provider: str
    model: str
    task: str = ""
    # Custom/local endpoint URL + key, honored on main AND auxiliary slots: the runtime resolvers
    # read model.base_url / auxiliary.<task>.base_url (+ .api_key) and ignore OPENAI_BASE_URL.
    base_url: str = ""
    api_key: str = ""
    confirm_expensive_model: bool = False
    profile: Optional[str] = None

class MoaModelSlot(BaseModel):
    provider: str = ""
    model: str = ""
    # Declared so a GET round-trip doesn't strip and wipe it.
    reasoning_effort: Optional[str] = None
    enabled: bool = True

class _MoaReferenceControls(BaseModel):
    # None = no per-preset override; inherits auxiliary.moa_reference.timeout (900s default).
    reference_timeout: Optional[float] = None
    degraded_reference_policy: Literal["loud", "silent"] = "loud"

    @field_validator("reference_timeout", mode="before")
    @classmethod
    def _validate_reference_timeout(cls, value: Any) -> Optional[float]:
        """Reject JSON booleans/non-finite values before float coercion."""
        if value is None or value == "":
            return None
        try:
            timeout = float(value) if not isinstance(value, bool) else math.nan
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_timeout must be a finite positive number") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("reference_timeout must be a finite positive number")
        return timeout

class MoaPresetPayload(_MoaReferenceControls):
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    # None = temperature omitted from API calls (provider default), as for single-model agents.
    reference_temperature: Optional[float] = None
    aggregator_temperature: Optional[float] = None
    max_tokens: int = 4096
    # Newer per-preset knobs (moa_config._normalize_preset): optional for older clients,
    # declared so GET round-trips don't erase them.
    reference_max_tokens: Optional[int] = None
    fanout: Optional[str] = None
    enabled: bool = True

class MoaConfigPayload(_MoaReferenceControls):
    default_preset: str = "default"
    active_preset: str = ""
    presets: dict[str, MoaPresetPayload] = {}
    # Backward-compatible flat payload fields for older dashboard/desktop clients.
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    reference_temperature: Optional[float] = None
    aggregator_temperature: Optional[float] = None
    max_tokens: int = 4096
    reference_max_tokens: Optional[int] = None
    fanout: Optional[str] = None
    enabled: bool = True
    profile: Optional[str] = None

class FsWriteText(BaseModel):
    path: str
    content: str

class GitPathBody(BaseModel):
    path: str

class GitFileBody(BaseModel):
    path: str
    file: Optional[str] = None

class GitPrListBody(BaseModel):
    path: str
    branches: List[str] = []
    # PRs a session recovered from its transcript — known by number, not branch.
    numbers: List[int] = []

class SessionPrScanBody(BaseModel):
    ids: List[str] = []

class GitCommitBody(BaseModel):
    path: str
    message: str
    push: bool = False

class GitWorktreeAddBody(BaseModel):
    path: str
    name: Optional[str] = None
    branch: Optional[str] = None
    base: Optional[str] = None
    existingBranch: Optional[str] = None

class GitWorktreeRemoveBody(BaseModel):
    path: str
    worktreePath: str
    force: bool = False

class GitBranchSwitchBody(BaseModel):
    path: str
    branch: str

class CuratorPause(BaseModel):
    paused: bool

class LearningNodeRef(BaseModel):
    id: str
    profile: Optional[str] = None

class LearningNodeEdit(BaseModel):
    id: str
    content: str
    profile: Optional[str] = None

class DebugShareRequest(BaseModel):
    # Redaction scrubs credential-shaped tokens before logs leave the machine; opt-out only.
    redact: bool = True
    lines: int = 200  # recent log lines in the summary tail (full logs are separate)

class TTSSpeakRequest(BaseModel):
    text: str

class TTSLeaseRequest(BaseModel):
    """POST /api/audio/tts-lease: ``lease`` names the toggle/surface holding the lease
    (``desktop:read-aloud``, ``desktop:conversation``); ``active`` True acquires + warms, False releases."""
    lease: str
    active: bool = True

class OAuthSubmitBody(BaseModel):
    session_id: str
    code: str

class BulkDeleteSessions(BaseModel):
    ids: List[str]
    profile: Optional[str] = None

class SessionImport(BaseModel):
    sessions: List[Dict[str, Any]]
    profile: Optional[str] = None

class SessionRename(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None
    hidden: Optional[bool] = None  # also used by cross-profile reconciliation
    pinned: Optional[bool] = None  # durable "keep" (Desktop pins); exempt from auto_archive
    # Read-state watermark (sessions.last_read_at): True = unread, False = read now, None = leave.
    unread: Optional[bool] = None
    profile: Optional[str] = None  # session owned by another profile (opens its state.db)

class SessionOwnerBackfill(BaseModel):
    """POST /api/sessions/owner-backfill (legacy migration). ``profile`` scopes WHICH state.db is
    stamped; the stamped value is always that store's own serving-profile identity — the caller
    cannot inject an arbitrary owner."""
    profile: Optional[str] = None

class SessionPrune(BaseModel):
    older_than_days: Optional[float] = 90
    source: Optional[str] = None
    profile: Optional[str] = None
    # Extended filters (all optional, ANDed — mirrors the CLI flags); *_before/after = epoch s
    started_before: Optional[float] = None
    started_after: Optional[float] = None
    title_like: Optional[str] = None
    end_reason: Optional[str] = None
    cwd_prefix: Optional[str] = None
    min_messages: Optional[int] = None
    max_messages: Optional[int] = None
    model_like: Optional[str] = None
    provider: Optional[str] = None
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    branch_like: Optional[str] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    min_tool_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    include_archived: bool = False
    dry_run: bool = False

class CronJobCreate(BaseModel):
    prompt: str = ""
    schedule: str
    name: str = ""
    deliver: str = "local"
    skills: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    script: Optional[str] = None
    context_from: Optional[Any] = None
    enabled_toolsets: Optional[List[str]] = None
    workdir: Optional[str] = None
    no_agent: bool = False

class CronJobUpdate(BaseModel):
    updates: dict

class AutomationBlueprintInstantiate(BaseModel):
    blueprint: str  # blueprint key, e.g. "morning-brief"
    values: Dict[str, Any] = {}  # filled slot values from the form

class MCPServerCreate(BaseModel):
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    env: Dict[str, str] = {}  # KEY=VALUE for stdio servers (API keys, etc.)
    auth: Optional[str] = None  # "none" | "oauth" | "header" | None
    # One-time provisioning input; persisted only to the profile's .env.
    bearer_token: Optional[SecretStr] = None
    profile: Optional[str] = None

class MCPServersReplace(BaseModel):
    # Whole-map replace (name → raw config) for the GUI mcp.json editor.
    servers: Dict[str, Dict[str, Any]] = {}
    profile: Optional[str] = None

class MCPEnabledToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None

class MCPCatalogInstall(BaseModel):
    name: str
    env: Dict[str, str] = {}  # KEY=VALUE for entries declaring required env vars
    enable: bool = True
    profile: Optional[str] = None

class PairingApprove(BaseModel):
    platform: str
    code: str = ""
    request_id: str = ""
    profile: Optional[str] = None

class PairingRevoke(BaseModel):
    platform: str
    user_id: str
    profile: Optional[str] = None

class WebhookCreate(BaseModel):
    name: str
    description: Optional[str] = None
    events: List[str] = []
    prompt: Optional[str] = None
    script: Optional[str] = None
    skills: List[str] = []
    deliver: str = "log"
    deliver_only: bool = False
    deliver_chat_id: Optional[str] = None
    secret: Optional[str] = None  # omit to auto-generate

class WebhookEnabledToggle(BaseModel):
    enabled: bool

class CredentialPoolAdd(BaseModel):
    provider: str
    api_key: str  # OAuth pooling stays CLI-only (needs an interactive browser flow)
    label: Optional[str] = None

class MemoryProviderSelect(BaseModel):
    provider: str  # "" or "built-in" disables the external provider

class MemoryReset(BaseModel):
    target: str = "all"  # "all" | "memory" | "user"

class BackupRequest(BaseModel):
    output: Optional[str] = None  # defaults to a timestamped zip in the home dir

class ImportRequest(BaseModel):
    archive: str
    # --force: the spawned `hermes import` has stdin=DEVNULL, so its "Continue? [y/N]" prompt would
    # hit EOF and abort; the dashboard confirms in its own modal.
    force: bool = False

class HookCreate(BaseModel):
    event: str
    command: str
    matcher: Optional[str] = None
    timeout: Optional[int] = None
    # Also write the consent allowlist entry; without it the hook won't fire until approved.
    approve: bool = True

class HookDelete(BaseModel):
    event: str
    command: str

class SkillInstallRequest(BaseModel):
    identifier: str
    profile: Optional[str] = None

class SkillUninstallRequest(BaseModel):
    name: str
    profile: Optional[str] = None

class SkillsUpdateRequest(BaseModel):
    profile: Optional[str] = None

class ProfileCreate(BaseModel):
    name: str
    clone_from: Optional[str] = None
    clone_from_default: bool = False  # legacy clients; new ones send clone_from explicitly
    clone_all: bool = False
    no_skills: bool = False
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # Profile-builder additions, applied best-effort AFTER the profile dir exists (a hiccup never 500s).
    mcp_servers: List["MCPServerCreate"] = []
    keep_skills: List[str] = []  # skills to KEEP: non-empty = replace semantics (unlisted seeded ones disabled)
    # Installed async via `hermes -p <name> skills install` (skills_hub.SKILLS_DIR is import-time-bound,
    # so HERMES_HOME can't redirect it); PIDs go back for the UI to poll.
    hub_skills: List[str] = []

class ProfileRename(BaseModel):
    new_name: str

class ProfileExport(BaseModel):
    extra_files: Dict[str, str] = {}  # extra root-level files, filename → text
    output: str = ""  # archive path; empty → a staging path under HERMES_HOME

class ProfileImport(BaseModel):
    archive: str  # profile .tar.gz on the backend's filesystem
    name: Optional[str] = None  # overrides the name inferred from the archive root

class ProfileSoulUpdate(BaseModel):
    content: str

class ProfileActiveUpdate(BaseModel):
    name: str

class ProfileDescriptionUpdate(BaseModel):
    description: str = ""

class ProfileModelUpdate(BaseModel):
    provider: str
    model: str

class ProfileDescribeAuto(BaseModel):
    overwrite: bool = False

class SkillToggle(BaseModel):
    name: str
    enabled: bool
    profile: Optional[str] = None

class SkillCreate(BaseModel):
    name: str
    content: str
    category: Optional[str] = None
    profile: Optional[str] = None

class SkillContentUpdate(BaseModel):
    name: str
    content: str
    profile: Optional[str] = None

class ToolsetToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None

class ToolsetProviderSelect(BaseModel):
    provider: str
    # Web-only scope 'search' | 'extract'; omitted → whole-provider (legacy web.backend path).
    capability: Optional[str] = None
    profile: Optional[str] = None

class ToolsetModelSelect(BaseModel):
    model: str
    provider: Optional[str] = None
    profile: Optional[str] = None

class ToolsetEnvUpdate(BaseModel):
    env: Dict[str, str]
    profile: Optional[str] = None

class ToolsetPostSetup(BaseModel):
    key: str
    profile: Optional[str] = None

class TerminalBackendSelect(BaseModel):
    backend: str
    profile: Optional[str] = None

class RawConfigUpdate(BaseModel):
    yaml_text: str
    profile: Optional[str] = None

class ThemeSetBody(BaseModel):
    name: str

class FontSetBody(BaseModel):
    font: str

class _AgentPluginInstallBody(BaseModel):
    identifier: str
    force: bool = False
    enable: bool = True

class _PluginProvidersPutBody(BaseModel):
    memory_provider: Optional[str] = None
    context_engine: Optional[str] = None

class _PluginVisibilityBody(BaseModel):
    hidden: bool

