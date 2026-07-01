"""No-model MCP profile router primitives.

This module is intentionally small and side-effect free: it only parses
fully-qualified profile refs and exposes read-only local profile inventory for
the ChatGPT ↔ Hermes MCP router, plus policy-gated read-only workspace access.
It must not import or call the Hermes agent loop, provider clients, or AI coding
CLIs. Filesystem/terminal action tools are private/direct only, no-model, and
must remain gated by explicit profile/root/context policy rather than exposing
arbitrary Hermes tools by default.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import ipaddress
import json
import os
import posixpath
import re
import shlex
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from hermes_cli.profiles import (
    ProfileInfo,
    get_profile_dir,
    list_profiles as _list_local_profile_infos,
    normalize_profile_name,
    validate_profile_name,
)


COST_CLASS_NO_MODEL = "no_model"
COST_CLASS_EXTERNAL_API_NO_MODEL = "external_api_no_model"
COST_CLASS_MAY_CALL_AUX_MODEL = "may_call_aux_model"
COST_CLASS_SCHEDULES_FUTURE_MODEL = "schedules_future_model"
COST_CLASS_CALLS_HERMES_AGENT_MODEL = "calls_hermes_agent_model"
ROUTE_HINT_PROFILE_ROUTER_NO_MODEL_WRAPPER = "execute_via_profile_router_no_model_wrapper"
ROUTE_HINT_USE_CLIENT_NATIVE = "use_client_native"
ROUTE_HINT_REQUIRES_DETERMINISTIC_WRAPPER = "requires_deterministic_no_model_wrapper"

NO_MODEL_DEFAULT_COST_CLASSES = frozenset(
    {COST_CLASS_NO_MODEL, COST_CLASS_EXTERNAL_API_NO_MODEL}
)
DEFAULT_ALLOWED_HOSTS = frozenset({"local", "mac"})
LOCAL_HOST = "local"
PROFILE_ROUTER_CONFIG_KEY = "profile_router"
PROFILE_ROUTER_TOOL_GROUP = "profile_router"
PROFILE_ROUTER_CAPABILITY_GROUPS = (
    "filesystem",
    "terminal",
    "git",
    "cron",
    "messaging",
    "skills",
    "memory",
    "session",
    "web",
    "browser",
    "api",
)
PROFILE_ROUTER_METADATA_CAPABILITY_GROUPS = (
    "profile",
    "workspace",
    *PROFILE_ROUTER_CAPABILITY_GROUPS,
)
CONTEXT_SKILLS_READ_POLICY = "context.skills.read"
CONTEXT_SESSIONS_SEARCH_POLICY = "context.sessions.search"
CONTEXT_VIKING_READ_POLICY = "profile_router.context.viking.read"
DEFAULT_PROTECTED_BRANCHES = ("main", "master", "develop", "production")
DEFAULT_ALLOWED_COST_CLASSES = (COST_CLASS_NO_MODEL,)


def _default_profile_capability_groups() -> dict[str, bool]:
    """Return the explicit deny-by-default profile capability map."""

    return {group: False for group in PROFILE_ROUTER_CAPABILITY_GROUPS}


SECRET_PATH_NAMES = frozenset(
    {
        ".aws",
        ".azure",
        ".config/gcloud",
        ".env",
        ".git",
        ".git-credentials",
        ".gnupg",
        ".hermes",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".ssh",
        "auth.json",
        "credentials",
        "credentials.json",
        "funciones.txt",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "mcp_tokens",
        "mcp_tokens.json",
        "secrets.json",
    }
)
SECRET_PATH_PREFIXES = (".env.",)
SENSITIVE_PATH_RE = re.compile(
    r"(?i)(^|[._-])"
    r"(api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"credential|credentials|passwd|password|private[_-]?key|secret|secrets|token|tokens)"
    r"($|[._-])"
)
HOST_ROOT_PATH_RE = re.compile(
    r"(?i)(/Users/[^\s,;'\"<>]+|/home/[^\s,;'\"<>]+|"
    r"/private/var/[^\s,;'\"<>]+|/var/folders/[^\s,;'\"<>]+|"
    r"/etc/[^\s,;'\"<>]+|[A-Z]:[\\/][^\s,;'\"<>]+)"
)
MAX_FILE_READ_LINES = 200
MAX_FILE_READ_CHARS = 60_000
MAX_FILE_LIST_RESULTS = 200
MAX_FILE_SEARCH_RESULTS = 50
MAX_FILE_SEARCH_BYTES = 1_000_000
MAX_SEARCH_LINE_CHARS = 500
MAX_FILE_WRITE_CHARS = 200_000
MAX_PATCH_APPLY_OPERATIONS = 10
MAX_PATCH_APPLY_TOTAL_WRITE_CHARS = 400_000
MAX_WRITE_DIFF_CHARS = 20_000
MAX_WORKSPACE_DIFF_CHARS = 30_000
MAX_WORKSPACE_DIFF_FILES = 100
MAX_TERMINAL_COMMAND_CHARS = 4_000
MAX_TERMINAL_TIMEOUT_SECONDS = 30
MAX_TERMINAL_OUTPUT_CHARS = 60_000
MAX_STATUS_PROBE_OUTPUT_CHARS = 2_000
MAX_PROCESS_LIST_RESULTS = 50
MAX_PROCESS_LOG_CHARS = 20_000
PROCESS_TERMINATE_GRACE_SECONDS = 2
MAX_GIT_STATUS_ENTRIES = 100
MAX_GIT_LOG_COUNT = 50
MAX_GIT_BRANCH_COUNT = 100
MAX_CRON_LIST_RESULTS = 50
MAX_CRON_JOB_NAME_CHARS = 160
MAX_CRON_SCHEDULE_CHARS = 120
MAX_CRON_SCRIPT_PATH_CHARS = 240
SAFE_CRON_SCRIPT_SUFFIXES = (".py", ".sh", ".bash")
CRON_JOB_REF_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
MAX_MESSAGING_DESTINATION_CHARS = 240
MAX_MESSAGING_MESSAGE_CHARS = 4_000
MESSAGING_DESTINATION_RE = re.compile(r"^([a-z][a-z0-9_-]{0,31}):(.{1,200})$")
BROADCAST_MESSAGING_DESTINATIONS = frozenset({"*", "all", "broadcast", "everyone"})
TERMINAL_SANITIZED_ENV_ALLOWED_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
)
TERMINAL_SANITIZED_ENV_DEFAULTS = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "LC_CTYPE": "C.UTF-8",
    "TERM": "dumb",
    "TMPDIR": "/tmp",
}
TERMINAL_SANITIZED_ENV_BLOCKED_MARKERS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
ALLOWED_SEARCH_OUTPUT_MODES = frozenset({"content", "files_only", "count"})
MAX_CONTEXT_FILE_CHARS = 12_000
MAX_CONTEXT_FILE_BYTES = 128_000
MAX_CONTEXT_HASH_BYTES = 1_000_000
MAX_SKILLS_LIST_RESULTS = 100
MAX_SKILL_LINKED_FILES_PER_DIR = 50
MAX_SKILL_DESCRIPTION_CHARS = 500
MAX_SESSION_SEARCH_RESULTS = 10
MAX_SESSION_SEARCH_QUERY_CHARS = 200
MAX_SESSION_SNIPPET_CHARS = 500
MAX_VIKING_SEARCH_RESULTS = 10
MAX_VIKING_QUERY_CHARS = 500
MAX_VIKING_URI_CHARS = 512
MAX_VIKING_ABSTRACT_CHARS = 1_200
MAX_VIKING_OVERVIEW_CHARS = 4_000
MAX_VIKING_FULL_CHARS = 8_000
MAX_VIKING_HTTP_RESPONSE_BYTES = 256_000
OPENVIKING_REQUEST_TIMEOUT_SECONDS = 10.0
OPENVIKING_ALLOWED_SEARCH_MODES = frozenset({"auto", "fast", "deep"})
OPENVIKING_ALLOWED_READ_LEVELS = frozenset({"abstract", "overview", "full"})
OPENVIKING_PSEUDO_SUMMARY_FILES = frozenset(
    {".abstract.md", ".overview.md", ".read.md", ".full.md"}
)
SAFE_SKILL_SUPPORT_DIRS = ("references", "templates", "scripts", "assets")
PROFILE_CONTEXT_FILES = ("SOUL.md",)
WORKSPACE_INSTRUCTION_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    "DESIGN.md",
    "SOUL.md",
)
FUNCIONES_TXT_FILENAME = "funciones.txt"
PROTECTED_WORKSPACE_DIFF_DIRS = frozenset({".hermes"})
SAFE_WORKSPACE_PLAN_DIR = ".hermes/plans"
SAFE_WORKSPACE_PLAN_SUFFIXES = (".md",)
SAFE_GIT_DIFF_FLAGS = ("--no-ext-diff", "--no-textconv")
SHELL_CONTROL_TOKENS = frozenset({";", "&&", "||", "|", "|&", "&", "(", ")"})
CHATGPT_SCRATCH_SMOKE_DIR = "tmp"
CHATGPT_SCRATCH_SMOKE_BASENAME_PREFIX = "chatgpt-hermes-action-smoke-"
CHATGPT_SCRATCH_SMOKE_INITIAL = "alpha\n"
CHATGPT_SCRATCH_SMOKE_PATCHED = "beta\n"
MODEL_COMMAND_NAMES = frozenset(
    {
        "aider",
        "claude",
        "codex",
        "fable",
        "gemini",
        "openai",
        "opencode",
    }
)
MODEL_COMMAND_TEXT_MARKERS = ("run_conversation", "delegate_task")
HERMES_MODEL_SUBCOMMANDS = frozenset({"chat"})
FORBIDDEN_MODEL_LOOP_TOOL_NAMES = frozenset(
    {
        "aider",
        "claude",
        "codex",
        "create_message",
        "createmessage",
        "delegate_task",
        "fable",
        "gemini",
        "hermes_agent_run",
        "hermes_chat",
        "image_generate",
        "llm_summarize",
        "mcp_create_message",
        "mcp_sampling",
        "openai",
        "opencode",
        "run_conversation",
        "sampling_create_message",
        "vision_analyze",
    }
)
DESTRUCTIVE_COMMAND_PATTERNS = (
    (
        "rm_rf",
        re.compile(r"(?i)(?:^|[;&|]\s*)rm\s+-(?=\S*r)(?=\S*f)\S*(?:\s|$)"),
    ),
    ("git_reset_hard", re.compile(r"(?i)\bgit\s+reset\s+--hard\b")),
    ("git_clean_force", re.compile(r"(?i)\bgit\s+clean\s+-(?=\S*f)\S*\b")),
)
PROTECTED_GIT_COMMAND_PATTERNS = (
    ("git_push", re.compile(r"(?i)\bgit\s+push\b")),
    ("git_merge", re.compile(r"(?i)\bgit\s+merge\b")),
    ("git_rebase", re.compile(r"(?i)\bgit\s+rebase\b")),
)
DEPLOY_COMMAND_PATTERNS = (
    ("easypanel", re.compile(r"(?i)\beasypanel\b")),
    ("kubectl", re.compile(r"(?i)\bkubectl\b")),
    ("vercel_deploy", re.compile(r"(?i)\bvercel\s+(?:deploy|prod|--prod)\b")),
    ("fly_deploy", re.compile(r"(?i)\bfly\s+deploy\b")),
    ("railway", re.compile(r"(?i)\brailway\s+(?:up|deploy)\b")),
)


class ProfileRouterError(ValueError):
    """Raised when a profile-router input is invalid or unsupported."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ProfileRef:
    """A validated, fully qualified profile reference."""

    host: str
    profile: str

    @property
    def value(self) -> str:
        return f"{self.host}:{self.profile}"


@dataclass(frozen=True)
class HostRoutePolicy:
    """Host-scoped routing policy.

    Host policies are intentionally separate from profile policies so future
    filesystem tools can prove that profile roots are contained by a host-local
    allowlist before touching any path.
    """

    host: str
    enabled: bool = False
    allowed_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalExecutionPolicy:
    """Explicit allowlist gate for a future no-shell terminal executor.

    ``terminal.enabled`` only permits preflight. Actual command execution must
    stay disabled unless this nested policy is enabled and the command matches a
    configured allowlist. The public MCP surface still does not register
    ``terminal_run``; this object is server-side policy/audit scaffolding only.
    """

    enabled: bool = False
    allowed_commands: tuple[str, ...] = ()
    allowed_command_prefixes: tuple[str, ...] = ()
    require_no_shell: bool = True


@dataclass(frozen=True)
class CronPolicy:
    """Explicit no-agent cron policy for a selected profile/workspace.

    ``cron.enabled`` only permits inspecting and operating on script-only
    ``no_agent`` jobs. Creating/resuming/running jobs additionally requires the
    script path to match ``cron.allowed_scripts`` so ChatGPT cannot schedule an
    arbitrary script that might call a model or leak data.
    """

    enabled: bool = False
    allowed_scripts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalSubprocessPlan:
    """Internal no-shell execution scaffold for future terminal_run support.

    This object is deliberately separate from the executor: it contains the
    argv/cwd/env that the private direct runner receives only after fresh
    context, route policy, no-shell parsing, and terminal.execution allowlist
    checks. ``terminal_run`` remains absent from public MCP registration.
    """

    argv: tuple[str, ...]
    cwd: Path
    public_cwd: str
    env: Mapping[str, str]
    timeout_seconds: int
    max_output_chars: int
    uses_shell: bool = False
    executes: bool = False


@dataclass(frozen=True)
class TerminalOutputStreamShape:
    """Bounded terminal output stream prepared for future MCP responses."""

    text: str
    truncated: bool
    original_chars: int
    returned_chars: int


@dataclass(frozen=True)
class ProfileRoutePolicy:
    """Deny-by-default policy for one fully-qualified profile ref."""

    ref: ProfileRef
    enabled: bool = False
    display_name: str = ""
    description: str = ""
    allowed_roots: tuple[str, ...] = ()
    allowed_tool_groups: tuple[str, ...] = ()
    capability_groups: Mapping[str, bool] = field(
        default_factory=_default_profile_capability_groups
    )
    messaging_allowed_recipients: tuple[str, ...] = ()
    allow_filesystem_read: bool = False
    allow_filesystem_write: bool = False
    allow_terminal: bool = False
    allow_messaging: bool = False
    allow_cron: bool = False
    allow_git_push: bool = False
    allow_deploy: bool = False
    allow_model_tools: bool = False
    allow_context_skills_read: bool = False
    allow_context_sessions_search: bool = False
    protected_branches: tuple[str, ...] = DEFAULT_PROTECTED_BRANCHES
    allowed_cost_classes: tuple[str, ...] = DEFAULT_ALLOWED_COST_CLASSES
    terminal_execution_policy: TerminalExecutionPolicy = field(
        default_factory=TerminalExecutionPolicy
    )
    cron_policy: CronPolicy = field(default_factory=CronPolicy)

    def capability_enabled(self, group: str) -> bool:
        """Return whether an explicit no-model capability group is enabled."""

        normalized = str(group or "").strip().lower()
        if normalized not in PROFILE_ROUTER_CAPABILITY_GROUPS:
            raise ProfileRouterError(
                "unknown_capability_group",
                f"Unknown profile-router capability group: {group}",
            )
        return bool(self.capability_groups.get(normalized, False))


@dataclass(frozen=True)
class ProfileRouterPolicy:
    """Parsed profile-router policy from ``config.yaml``.

    An absent ``profile_router`` section exposes no profiles.  A profile is
    exposed only when the host is enabled, the profile is explicitly enabled,
    and the read-only ``profile_router`` tool group is allowed.
    """

    hosts: Mapping[str, HostRoutePolicy]
    profiles: Mapping[str, ProfileRoutePolicy]
    allow_global_viking_read: bool = False

    def is_profile_exposed(self, policy: ProfileRoutePolicy) -> bool:
        host_policy = self.hosts.get(policy.ref.host)
        return bool(
            host_policy
            and host_policy.enabled
            and policy.enabled
            and PROFILE_ROUTER_TOOL_GROUP in policy.allowed_tool_groups
        )

    def iter_profiles(
        self,
        *,
        host: str | None = None,
        active_only: bool = True,
    ) -> list[ProfileRoutePolicy]:
        selected: list[ProfileRoutePolicy] = []
        normalized_host = host.lower() if host else None
        for policy in self.profiles.values():
            if normalized_host and policy.ref.host != normalized_host:
                continue
            if active_only and not self.is_profile_exposed(policy):
                continue
            selected.append(policy)
        return selected

    def get_profile_policy(self, ref: ProfileRef) -> ProfileRoutePolicy:
        policy = self.profiles.get(ref.value)
        if policy is None:
            raise ProfileRouterError(
                "profile_not_enabled",
                f"Profile is not enabled in profile_router policy: {ref.value}",
            )
        if not self.is_profile_exposed(policy):
            raise ProfileRouterError(
                "profile_disabled",
                f"Profile is disabled by profile_router policy: {ref.value}",
            )
        return policy


def _policy_mapping(value: Any, field: str) -> MappingABC:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ProfileRouterError("invalid_policy", f"{field} must be a mapping")
    return value


def _policy_bool(value: Any, field: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ProfileRouterError("invalid_policy", f"{field} must be a boolean")


def _policy_string_tuple(
    value: Any,
    field: str,
    *,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, IterableABC) and not isinstance(value, (bytes, MappingABC)):
        values = list(value)
    else:
        raise ProfileRouterError("invalid_policy", f"{field} must be a string list")

    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ProfileRouterError("invalid_policy", f"{field} entries must be strings")
        text = item.strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _policy_allowed_roots(value: Any, field: str) -> tuple[str, ...]:
    roots = []
    for root in _policy_string_tuple(value, field):
        if not root.startswith("/"):
            raise ProfileRouterError(
                "invalid_policy",
                f"{field} entries must be absolute host-local paths",
            )
        roots.append(posixpath.normpath(root))
    return tuple(dict.fromkeys(roots))


def _policy_terminal_command_tuple(value: Any, field: str) -> tuple[str, ...]:
    commands = _policy_string_tuple(value, field)
    for command in commands:
        if "\n" in command or "\r" in command:
            raise ProfileRouterError(
                "invalid_policy",
                f"{field} entries must be single-line commands",
            )
        if len(command) > MAX_TERMINAL_COMMAND_CHARS:
            raise ProfileRouterError(
                "invalid_policy",
                f"{field} entries must be <= {MAX_TERMINAL_COMMAND_CHARS} characters",
            )
    return commands


def _terminal_command_has_shell_control(tokens: Iterable[str]) -> bool:
    return any(
        token in SHELL_CONTROL_TOKENS or all(char in ";&|()<>" for char in token)
        for token in tokens
    )


def _parse_terminal_execution_policy(value: Any, field: str) -> TerminalExecutionPolicy:
    policy = _policy_mapping(value, field)
    require_no_shell = _policy_bool(
        policy.get("require_no_shell"), f"{field}.require_no_shell", default=True
    )
    if not require_no_shell:
        raise ProfileRouterError(
            "terminal_shell_execution_not_allowed",
            f"{field}.require_no_shell cannot be false; terminal execution must be no-shell",
        )

    execution_policy = TerminalExecutionPolicy(
        enabled=_policy_bool(policy.get("enabled"), f"{field}.enabled"),
        allowed_commands=_policy_terminal_command_tuple(
            policy.get("allowed_commands"), f"{field}.allowed_commands"
        ),
        allowed_command_prefixes=_policy_terminal_command_tuple(
            policy.get("allowed_command_prefixes"),
            f"{field}.allowed_command_prefixes",
        ),
        require_no_shell=require_no_shell,
    )
    if execution_policy.enabled and not (
        execution_policy.allowed_commands or execution_policy.allowed_command_prefixes
    ):
        raise ProfileRouterError(
            "terminal_allowlist_required",
            f"{field}.enabled requires allowed_commands or allowed_command_prefixes",
        )
    for command in (
        *execution_policy.allowed_commands,
        *execution_policy.allowed_command_prefixes,
    ):
        tokens = _shell_command_tokens(command)
        if _terminal_command_has_shell_control(tokens):
            raise ProfileRouterError(
                "terminal_allowlist_shell_control_not_allowed",
                f"{field} entries cannot require shell control operators",
            )
    return execution_policy


def _normalize_cron_script_path(script: str, field: str = "script") -> str:
    """Validate a cron script identifier without treating it as a shell command."""

    if not isinstance(script, str):
        raise ProfileRouterError("invalid_cron_script", f"{field} must be a string")
    text = script.strip()
    if not text:
        raise ProfileRouterError("invalid_cron_script", f"{field} is required")
    if len(text) > MAX_CRON_SCRIPT_PATH_CHARS:
        raise ProfileRouterError(
            "invalid_cron_script",
            f"{field} must be <= {MAX_CRON_SCRIPT_PATH_CHARS} characters",
        )
    if "\x00" in text or "\n" in text or "\r" in text or "\\" in text:
        raise ProfileRouterError("invalid_cron_script", f"{field} must be a single POSIX path")
    if text.startswith("/") or "://" in text:
        raise ProfileRouterError(
            "invalid_cron_script",
            f"{field} must be a relative allowlisted Hermes script path",
        )
    normalized = posixpath.normpath(text)
    if normalized in {".", ""} or normalized.startswith("../"):
        raise ProfileRouterError("invalid_cron_script", f"{field} must not escape the scripts directory")
    parts = [part for part in normalized.split("/") if part]
    if any(part in {".", ".."} or part.startswith(".") for part in parts):
        raise ProfileRouterError("invalid_cron_script", f"{field} must not contain dot path segments")
    if _is_secret_path(normalized):
        raise ProfileRouterError("secret_path_denied", f"{field} is blocked by the secret denylist")
    if not normalized.endswith(SAFE_CRON_SCRIPT_SUFFIXES):
        raise ProfileRouterError(
            "invalid_cron_script",
            f"{field} must end with one of: {', '.join(SAFE_CRON_SCRIPT_SUFFIXES)}",
        )
    return normalized


def _policy_cron_script_tuple(value: Any, field: str) -> tuple[str, ...]:
    scripts = []
    for script in _policy_string_tuple(value, field):
        normalized = _normalize_cron_script_path(script, field)
        if normalized not in scripts:
            scripts.append(normalized)
    return tuple(scripts)


def _normalize_messaging_destination(destination: str, field: str = "destination") -> tuple[str, str, str]:
    """Validate an allowlisted messaging destination without exposing it later."""

    if not isinstance(destination, str):
        raise ProfileRouterError("invalid_messaging_destination", f"{field} must be a string")
    text = destination.strip()
    if not text:
        raise ProfileRouterError("invalid_messaging_destination", f"{field} is required")
    if len(text) > MAX_MESSAGING_DESTINATION_CHARS:
        raise ProfileRouterError(
            "invalid_messaging_destination",
            f"{field} must be <= {MAX_MESSAGING_DESTINATION_CHARS} characters",
        )
    if "\x00" in text or "\n" in text or "\r" in text:
        raise ProfileRouterError(
            "invalid_messaging_destination",
            f"{field} must be a single-line destination",
        )
    if text.lower() in BROADCAST_MESSAGING_DESTINATIONS:
        raise ProfileRouterError("messaging_broadcast_not_allowed", "broadcast messaging destinations are not allowed")
    match = MESSAGING_DESTINATION_RE.fullmatch(text)
    if not match:
        raise ProfileRouterError(
            "invalid_messaging_destination",
            f"{field} must use '<platform>:<recipient>' form",
        )
    platform = match.group(1).lower()
    recipient = match.group(2).strip()
    if not recipient or recipient.lower() in BROADCAST_MESSAGING_DESTINATIONS:
        raise ProfileRouterError("messaging_broadcast_not_allowed", "broadcast messaging destinations are not allowed")
    redacted = _redact_sensitive_text_fields(text)
    if redacted != text:
        raise ProfileRouterError("messaging_destination_secret_denied", "destination is blocked by the secret denylist")
    return platform, recipient, f"{platform}:{recipient}"


def _policy_messaging_destination_tuple(value: Any, field: str) -> tuple[str, ...]:
    destinations = []
    for destination in _policy_string_tuple(value, field):
        _platform, _recipient, normalized = _normalize_messaging_destination(destination, field)
        if normalized not in destinations:
            destinations.append(normalized)
    return tuple(destinations)


def _parse_cron_policy(value: Any, field: str) -> CronPolicy:
    policy = _policy_mapping(value, field)
    return CronPolicy(
        enabled=_policy_bool(policy.get("enabled"), f"{field}.enabled"),
        allowed_scripts=_policy_cron_script_tuple(
            policy.get("allowed_scripts"), f"{field}.allowed_scripts"
        ),
    )


def _path_within_root(path: str, root: str) -> bool:
    normalized_path = posixpath.normpath(path)
    normalized_root = posixpath.normpath(root)
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root.rstrip("/") + "/"
    )


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalize_absolute_host_path(path: str, field: str) -> str:
    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", f"{field} must be a string")
    text = path.strip()
    if not text:
        raise ProfileRouterError("invalid_path", f"{field} is required")
    if not text.startswith("/"):
        raise ProfileRouterError(
            "invalid_path", f"{field} must be an absolute host-local path"
        )
    return posixpath.normpath(text)


def _is_secret_path(path: str) -> bool:
    normalized = posixpath.normpath(path)
    normalized_lower = normalized.lower()
    if normalized_lower in SECRET_PATH_NAMES:
        return True
    parts = [part.lower() for part in normalized_lower.split("/") if part]
    for index, part in enumerate(parts):
        joined_tail = "/".join(parts[index:])
        if part in SECRET_PATH_NAMES or joined_tail in SECRET_PATH_NAMES:
            return True
        if any(part.startswith(prefix) for prefix in SECRET_PATH_PREFIXES):
            return True
        if part.endswith((".key", ".pem", ".p12", ".pfx")):
            return True
        if SENSITIVE_PATH_RE.search(part):
            return True
    return False


def _ensure_not_secret_path(path: str) -> None:
    if _is_secret_path(path):
        raise ProfileRouterError(
            "secret_path_denied",
            "Path is blocked by the profile-router secret denylist",
        )


def _is_safe_workspace_plan_path(
    relative_path: str,
    *,
    allow_directory: bool = False,
) -> bool:
    """Allow only explicit markdown development plans under ``.hermes/plans``.

    This is intentionally narrower than opening ``.hermes/**``: it supports
    ChatGPT reading repo-local development plans while keeping runtime state,
    tokens, configs, JSON state, dotfiles, nested metadata, and secret-looking
    paths blocked.  Contents are still bounded and redacted by the file-read
    layer.
    """

    if not isinstance(relative_path, str):
        return False
    normalized = posixpath.normpath(relative_path.strip() or ".")
    if normalized == SAFE_WORKSPACE_PLAN_DIR:
        return allow_directory
    parts = [part for part in normalized.split("/") if part]
    if len(parts) != 3 or "/".join(parts[:2]) != SAFE_WORKSPACE_PLAN_DIR:
        return False
    filename = parts[2]
    filename_lower = filename.lower()
    if filename.startswith(".") or filename_lower == FUNCIONES_TXT_FILENAME:
        return False
    if filename_lower in SECRET_PATH_NAMES:
        return False
    if any(filename_lower.startswith(prefix) for prefix in SECRET_PATH_PREFIXES):
        return False
    if filename_lower.endswith((".key", ".pem", ".p12", ".pfx")):
        return False
    return filename_lower.endswith(SAFE_WORKSPACE_PLAN_SUFFIXES)


def _ensure_workspace_read_path_not_secret(
    path: str,
    relative_path: str,
    *,
    allow_plan_directory: bool = False,
) -> None:
    if _is_secret_path(path) and not _is_safe_workspace_plan_path(
        relative_path,
        allow_directory=allow_plan_directory,
    ):
        raise ProfileRouterError(
            "secret_path_denied",
            "Path is blocked by the profile-router secret denylist",
        )


def _resolved_relative_workspace_path(root: Path, candidate: Path) -> str:
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root") from exc
    return relative or "."


def _resolve_existing_local_path(path: str, field: str) -> Path:
    try:
        return Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ProfileRouterError("path_not_found", f"{field} not found") from exc
    except OSError as exc:
        raise ProfileRouterError("invalid_path", f"{field} is not accessible") from exc


def _parse_host_policy(host_name: str, raw_policy: Any) -> HostRoutePolicy:
    host = str(host_name or "").strip().lower()
    if host not in DEFAULT_ALLOWED_HOSTS:
        raise ProfileRouterError("unsupported_host", f"Unsupported profile host: {host}")

    policy = _policy_mapping(raw_policy, f"hosts.{host}")
    return HostRoutePolicy(
        host=host,
        enabled=_policy_bool(policy.get("enabled"), f"hosts.{host}.enabled"),
        allowed_roots=_policy_allowed_roots(
            policy.get("allowed_roots"), f"hosts.{host}.allowed_roots"
        ),
    )


def _parse_profile_policy(
    profile_ref: str,
    raw_policy: Any,
    hosts: Mapping[str, HostRoutePolicy],
) -> ProfileRoutePolicy:
    ref = parse_profile_ref(profile_ref)
    policy = _policy_mapping(raw_policy, f"profiles.{ref.value}")
    enabled = _policy_bool(policy.get("enabled"), f"profiles.{ref.value}.enabled")

    raw_tool_groups = policy.get("allowed_tool_groups")
    allowed_tool_groups = _policy_string_tuple(
        raw_tool_groups,
        f"profiles.{ref.value}.allowed_tool_groups",
        default=(PROFILE_ROUTER_TOOL_GROUP,) if enabled and raw_tool_groups is None else (),
    )
    allowed_roots = _policy_allowed_roots(
        policy.get("allowed_roots"), f"profiles.{ref.value}.allowed_roots"
    )
    host_policy = hosts.get(ref.host)
    host_allowed_roots = host_policy.allowed_roots if host_policy is not None else ()
    if allowed_roots and not host_allowed_roots:
        raise ProfileRouterError(
            "root_outside_allowed_host_roots",
            f"Profile {ref.value} declares roots but host {ref.host} has no allowed_roots",
        )
    for root in allowed_roots:
        if not any(_path_within_root(root, host_root) for host_root in host_allowed_roots):
            raise ProfileRouterError(
                "root_outside_allowed_host_roots",
                f"Profile root {root} is outside host {ref.host} allowed_roots",
            )

    filesystem = _policy_mapping(policy.get("filesystem"), f"profiles.{ref.value}.filesystem")
    terminal = _policy_mapping(policy.get("terminal"), f"profiles.{ref.value}.terminal")
    messaging = _policy_mapping(policy.get("messaging"), f"profiles.{ref.value}.messaging")
    cron = _policy_mapping(policy.get("cron"), f"profiles.{ref.value}.cron")
    git = _policy_mapping(policy.get("git"), f"profiles.{ref.value}.git")
    deploy = _policy_mapping(policy.get("deploy"), f"profiles.{ref.value}.deploy")
    skills = _policy_mapping(policy.get("skills"), f"profiles.{ref.value}.skills")
    memory = _policy_mapping(policy.get("memory"), f"profiles.{ref.value}.memory")
    session = _policy_mapping(policy.get("session"), f"profiles.{ref.value}.session")
    web = _policy_mapping(policy.get("web"), f"profiles.{ref.value}.web")
    browser = _policy_mapping(policy.get("browser"), f"profiles.{ref.value}.browser")
    api = _policy_mapping(policy.get("api"), f"profiles.{ref.value}.api")

    allow_filesystem_read = _policy_bool(
        filesystem.get("read"), f"profiles.{ref.value}.filesystem.read"
    )
    allow_filesystem_write = _policy_bool(
        filesystem.get("write"), f"profiles.{ref.value}.filesystem.write"
    )
    allow_terminal = _policy_bool(
        terminal.get("enabled"), f"profiles.{ref.value}.terminal.enabled"
    )
    allow_messaging = _policy_bool(
        messaging.get("enabled"), f"profiles.{ref.value}.messaging.enabled"
    )
    allow_cron = _policy_bool(cron.get("enabled"), f"profiles.{ref.value}.cron.enabled")
    cron_policy = _parse_cron_policy(cron, f"profiles.{ref.value}.cron")
    allow_git = _policy_bool(git.get("enabled"), f"profiles.{ref.value}.git.enabled")
    allow_git_push = _policy_bool(
        git.get("allow_push"), f"profiles.{ref.value}.git.allow_push"
    )
    allow_deploy = _policy_bool(
        deploy.get("enabled"), f"profiles.{ref.value}.deploy.enabled"
    )
    allow_skills = _policy_bool(skills.get("enabled"), f"profiles.{ref.value}.skills.enabled")
    allow_memory = _policy_bool(memory.get("enabled"), f"profiles.{ref.value}.memory.enabled")
    allow_session = _policy_bool(session.get("enabled"), f"profiles.{ref.value}.session.enabled")
    allow_web = _policy_bool(web.get("enabled"), f"profiles.{ref.value}.web.enabled")
    allow_browser = _policy_bool(
        browser.get("enabled"), f"profiles.{ref.value}.browser.enabled"
    )
    allow_api = _policy_bool(api.get("enabled"), f"profiles.{ref.value}.api.enabled")

    terminal_execution_policy = _parse_terminal_execution_policy(
        terminal.get("execution"), f"profiles.{ref.value}.terminal.execution"
    )
    if terminal_execution_policy.enabled and not allow_terminal:
        raise ProfileRouterError(
            "terminal_execution_requires_terminal",
            f"profiles.{ref.value}.terminal.execution.enabled requires terminal.enabled=true",
        )
    context = _policy_mapping(policy.get("context"), f"profiles.{ref.value}.context")
    context_skills = _policy_mapping(
        context.get("skills"), f"profiles.{ref.value}.context.skills"
    )
    context_sessions = _policy_mapping(
        context.get("sessions"), f"profiles.{ref.value}.context.sessions"
    )
    model_tools = _policy_mapping(
        policy.get("model_tools", policy.get("model_policy")),
        f"profiles.{ref.value}.model_tools",
    )
    allow_model_tools = _policy_bool(
        model_tools.get("allow_model_tools"),
        f"profiles.{ref.value}.model_tools.allow_model_tools",
    )
    allowed_cost_classes = _policy_string_tuple(
        model_tools.get("allowed_cost_classes"),
        f"profiles.{ref.value}.model_tools.allowed_cost_classes",
        default=DEFAULT_ALLOWED_COST_CLASSES,
    )
    if not allow_model_tools:
        unsafe_cost_classes = sorted(set(allowed_cost_classes) - NO_MODEL_DEFAULT_COST_CLASSES)
        if unsafe_cost_classes:
            raise ProfileRouterError(
                "model_cost_class_not_allowed",
                "Model-consuming cost classes require allow_model_tools=true: "
                + ", ".join(unsafe_cost_classes),
            )

    capability_groups = {
        "filesystem": allow_filesystem_read or allow_filesystem_write,
        "terminal": allow_terminal,
        "git": allow_git or allow_git_push,
        "cron": allow_cron,
        "messaging": allow_messaging,
        "skills": allow_skills,
        "memory": allow_memory,
        "session": allow_session,
        "web": allow_web,
        "browser": allow_browser,
        "api": allow_api,
    }

    return ProfileRoutePolicy(
        ref=ref,
        enabled=enabled,
        display_name=str(policy.get("display_name") or "").strip(),
        description=str(policy.get("description") or "").strip(),
        allowed_roots=allowed_roots,
        allowed_tool_groups=allowed_tool_groups,
        capability_groups=capability_groups,
        messaging_allowed_recipients=_policy_messaging_destination_tuple(
            messaging.get("allowed_recipients"),
            f"profiles.{ref.value}.messaging.allowed_recipients",
        ),
        allow_filesystem_read=allow_filesystem_read,
        allow_filesystem_write=allow_filesystem_write,
        allow_terminal=allow_terminal,
        allow_messaging=allow_messaging,
        allow_cron=allow_cron,
        allow_git_push=allow_git_push,
        allow_deploy=allow_deploy,
        allow_model_tools=allow_model_tools,
        allow_context_skills_read=_policy_bool(
            context_skills.get("read"), f"profiles.{ref.value}.context.skills.read"
        ),
        allow_context_sessions_search=_policy_bool(
            context_sessions.get("search"),
            f"profiles.{ref.value}.context.sessions.search",
        ),
        protected_branches=_policy_string_tuple(
            git.get("protected_branches"),
            f"profiles.{ref.value}.git.protected_branches",
            default=DEFAULT_PROTECTED_BRANCHES,
        ),
        allowed_cost_classes=allowed_cost_classes,
        terminal_execution_policy=terminal_execution_policy,
        cron_policy=cron_policy,
    )


def load_profile_router_policy(config: Mapping[str, Any] | None = None) -> ProfileRouterPolicy:
    """Load explicit profile-router policy from config.

    Missing config is safe: it exposes no profiles and grants no filesystem,
    terminal, messaging, cron, deploy, or model-consuming capability.
    """

    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    if not isinstance(config, MappingABC):
        raise ProfileRouterError("invalid_policy", "config must be a mapping")

    raw_section = config.get(PROFILE_ROUTER_CONFIG_KEY) or {}
    section = _policy_mapping(raw_section, PROFILE_ROUTER_CONFIG_KEY)
    raw_hosts = _policy_mapping(section.get("hosts"), f"{PROFILE_ROUTER_CONFIG_KEY}.hosts")
    raw_profiles = _policy_mapping(
        section.get("profiles"), f"{PROFILE_ROUTER_CONFIG_KEY}.profiles"
    )
    global_context = _policy_mapping(
        section.get("context"), f"{PROFILE_ROUTER_CONFIG_KEY}.context"
    )
    global_context_viking = _policy_mapping(
        global_context.get("viking"), f"{PROFILE_ROUTER_CONFIG_KEY}.context.viking"
    )
    allow_global_viking_read = _policy_bool(
        global_context_viking.get("read"),
        f"{PROFILE_ROUTER_CONFIG_KEY}.context.viking.read",
    )

    hosts = {
        host: _parse_host_policy(host, host_policy)
        for host, host_policy in raw_hosts.items()
    }
    profiles = {
        parse_profile_ref(profile_ref).value: _parse_profile_policy(
            profile_ref, profile_policy, hosts
        )
        for profile_ref, profile_policy in raw_profiles.items()
    }
    return ProfileRouterPolicy(
        hosts=hosts,
        profiles=profiles,
        allow_global_viking_read=allow_global_viking_read,
    )


@dataclass(frozen=True)
class RouterToolMetadata:
    """Security/cost metadata required before exposing a router tool."""

    name: str
    description: str
    cost_class: str
    llm_calls: int
    enabled_by_default: bool = True
    mutates_state: bool = False
    requires_context: bool = False
    requires_profile_ref: bool = False
    requires_context_policy: str | None = None
    tool_group: str = "profile_router"
    execution_status: str = "executable_no_model"
    blocked_reason: str | None = None
    route_hint: str = ROUTE_HINT_PROFILE_ROUTER_NO_MODEL_WRAPPER


def _blocked_catalog_route_hint(tool_name: str) -> str:
    """Return a no-model route hint for a catalog-visible blocked tool."""

    if tool_name in HERMES_CATALOG_MODEL_BACKED_TOOL_NAMES:
        return ROUTE_HINT_USE_CLIENT_NATIVE
    return ROUTE_HINT_REQUIRES_DETERMINISTIC_WRAPPER


def parse_profile_ref(
    profile_ref: str,
    *,
    allowed_hosts: Iterable[str] = DEFAULT_ALLOWED_HOSTS,
) -> ProfileRef:
    """Parse and validate ``host:profile`` profile refs.

    The router boundary requires explicit host qualification so a public MCP
    client cannot accidentally target the wrong profile namespace.  Profile
    names are normalized with the same rules as ``hermes profile``.
    """

    if not isinstance(profile_ref, str):
        raise ProfileRouterError("invalid_profile_ref", "profile_ref must be a string")

    raw = profile_ref.strip()
    if raw.count(":") != 1:
        raise ProfileRouterError(
            "invalid_profile_ref",
            "profile_ref must use fully qualified form '<host>:<profile>'",
        )

    raw_host, raw_profile = raw.split(":", 1)
    host = raw_host.strip().lower()
    if not host:
        raise ProfileRouterError("invalid_profile_ref", "profile_ref host is required")

    allowed = {str(item).strip().lower() for item in allowed_hosts}
    if host not in allowed:
        raise ProfileRouterError("unsupported_host", f"Unsupported profile host: {host}")

    try:
        profile = normalize_profile_name(raw_profile)
        validate_profile_name(profile)
    except ValueError as exc:
        raise ProfileRouterError("invalid_profile_ref", str(exc)) from exc

    return ProfileRef(host=host, profile=profile)


ROUTER_TOOL_METADATA: Mapping[str, RouterToolMetadata] = {
    "profiles_list": RouterToolMetadata(
        name="profiles_list",
        description="List local Hermes profiles as fully-qualified refs.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "profile_get": RouterToolMetadata(
        name="profile_get",
        description="Get non-secret metadata for one local Hermes profile.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_profile_ref=True,
    ),
    "profile_health": RouterToolMetadata(
        name="profile_health",
        description="Report read-only local profile health without executing tools.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_profile_ref=True,
    ),
    "profile_context_get": RouterToolMetadata(
        name="profile_context_get",
        description="Load bounded no-model profile SOUL/policy context for a profile.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_profile_ref=True,
    ),
    "skills_list": RouterToolMetadata(
        name="skills_list",
        description=(
            "List bounded, sanitized profile skill metadata after context.skills.read policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=True,
        requires_profile_ref=True,
        requires_context_policy=CONTEXT_SKILLS_READ_POLICY,
    ),
    "skill_view": RouterToolMetadata(
        name="skill_view",
        description=(
            "Read bounded, sanitized profile SKILL.md or safe supporting files "
            "after context.skills.read policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=True,
        requires_profile_ref=True,
        requires_context_policy=CONTEXT_SKILLS_READ_POLICY,
    ),
    "session_search": RouterToolMetadata(
        name="session_search",
        description=(
            "Search bounded, sanitized user/assistant snippets in a selected "
            "profile state.db after context.sessions.search policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=True,
        requires_profile_ref=True,
        requires_context_policy=CONTEXT_SESSIONS_SEARCH_POLICY,
    ),
    "viking_search": RouterToolMetadata(
        name="viking_search",
        description=(
            "Search the server-configured local/private OpenViking memory surface "
            "after explicit context.viking.read policy; no model calls."
        ),
        cost_class=COST_CLASS_EXTERNAL_API_NO_MODEL,
        llm_calls=0,
        enabled_by_default=True,
        requires_context_policy=CONTEXT_VIKING_READ_POLICY,
    ),
    "viking_read": RouterToolMetadata(
        name="viking_read",
        description=(
            "Read bounded content from the server-configured local/private "
            "OpenViking memory surface after context.viking.read policy."
        ),
        cost_class=COST_CLASS_EXTERNAL_API_NO_MODEL,
        llm_calls=0,
        enabled_by_default=True,
        requires_context_policy=CONTEXT_VIKING_READ_POLICY,
    ),
    "workspace_instructions_get": RouterToolMetadata(
        name="workspace_instructions_get",
        description="Hydrate bounded workspace instructions and return a context token.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_context_status": RouterToolMetadata(
        name="workspace_context_status",
        description="Report whether an opened workspace context is loaded or stale.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_open": RouterToolMetadata(
        name="workspace_open",
        description="Open a policy-gated read-only local workspace with an opaque ID.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_get": RouterToolMetadata(
        name="workspace_get",
        description="Inspect an opened workspace by opaque ID without revealing its root.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_close": RouterToolMetadata(
        name="workspace_close",
        description="Close an opened workspace and remove its server-side registry entry.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
    ),
    "workspace_file_list": RouterToolMetadata(
        name="workspace_file_list",
        description="List bounded, sanitized files after workspace context hydration.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_context=True,
    ),
    "workspace_file_read": RouterToolMetadata(
        name="workspace_file_read",
        description="Read a bounded sanitized file slice after workspace context hydration.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_context=True,
    ),
    "workspace_file_stat": RouterToolMetadata(
        name="workspace_file_stat",
        description="Return bounded sanitized file/directory metadata after workspace context hydration.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_context=True,
    ),
    "workspace_file_search": RouterToolMetadata(
        name="workspace_file_search",
        description="Search bounded sanitized workspace files after workspace context hydration.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        requires_context=True,
    ),
    "workspace_status_probe": RouterToolMetadata(
        name="workspace_status_probe",
        description=(
            "Run a fixed, sanitized workspace status probe after context hydration; "
            "does not accept arbitrary commands and does not expose host roots."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        requires_context=True,
    ),
    "workspace_scratch_smoke": RouterToolMetadata(
        name="workspace_scratch_smoke",
        description=(
            "Run a fixed scratch write/read/patch/delete smoke after context and "
            "write-policy gates; does not accept arbitrary paths or content."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "file_read": RouterToolMetadata(
        name="file_read",
        description="Legacy direct file-read alias; public v1 uses workspace_file_read.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        requires_context=True,
    ),
    "file_search": RouterToolMetadata(
        name="file_search",
        description="Legacy direct file-search helper; not registered on the public v1 HTTP surface.",
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        requires_context=True,
    ),
    "file_patch": RouterToolMetadata(
        name="file_patch",
        description=(
            "Direct write tool; disabled from default public MCP exposure and always "
            "requires fresh workspace context plus filesystem.write policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "patch_apply": RouterToolMetadata(
        name="patch_apply",
        description=(
            "Direct bounded multi-file literal patch tool; disabled from default "
            "public MCP exposure and always requires fresh workspace context plus "
            "filesystem.write policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "file_write": RouterToolMetadata(
        name="file_write",
        description=(
            "Direct write tool; disabled from default public MCP exposure and always "
            "requires fresh workspace context plus filesystem.write policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "file_move": RouterToolMetadata(
        name="file_move",
        description=(
            "Direct file move/rename tool; disabled from default public MCP exposure "
            "and always requires fresh workspace context plus filesystem.write policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "file_delete": RouterToolMetadata(
        name="file_delete",
        description=(
            "Direct file delete tool; disabled from default public MCP exposure "
            "and always requires fresh workspace context plus filesystem.write policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "directory_create": RouterToolMetadata(
        name="directory_create",
        description=(
            "Direct directory creation tool; disabled from default public MCP exposure "
            "and always requires fresh workspace context plus filesystem.write policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "workspace_diff": RouterToolMetadata(
        name="workspace_diff",
        description=(
            "Public v1 read-only Git diff/audit tool; requires fresh workspace "
            "context and returns bounded sanitized output with llm_calls=0."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=True,
        mutates_state=False,
        requires_context=True,
    ),
    "terminal_run": RouterToolMetadata(
        name="terminal_run",
        description=(
            "Private terminal tool; disabled from default public MCP exposure. "
            "Direct calls require fresh workspace context and explicit "
            "terminal.execution allowlist, use sanitized no-shell subprocess "
            "execution, and block model/destructive/protected/deploy patterns."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "process_start": RouterToolMetadata(
        name="process_start",
        description=(
            "Private runtime-owned background process launch; disabled from default "
            "public MCP exposure, requires fresh workspace context plus terminal.execution "
            "allowlist, and tracks only opaque process ids owned by this runtime."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "process_list": RouterToolMetadata(
        name="process_list",
        description=(
            "Private tracked-process listing scaffold; disabled from default public MCP "
            "exposure and limited to processes launched/tracked by this runtime."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "process_poll": RouterToolMetadata(
        name="process_poll",
        description=(
            "Private tracked-process status scaffold; disabled from default public MCP "
            "exposure and never inspects arbitrary host processes."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "process_log": RouterToolMetadata(
        name="process_log",
        description=(
            "Private tracked-process log scaffold; disabled from default public MCP "
            "exposure and bounded/redacted before any future public wrapper."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "process_kill": RouterToolMetadata(
        name="process_kill",
        description=(
            "Private tracked-process kill scaffold; disabled from default public MCP "
            "exposure and scoped only to runtime-owned process ids."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "git_status": RouterToolMetadata(
        name="git_status",
        description=(
            "Private read-only Git status wrapper; disabled from default public MCP "
            "exposure and requires fresh workspace context plus git.enabled policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "git_diff": RouterToolMetadata(
        name="git_diff",
        description=(
            "Private read-only Git diff wrapper; disabled from default public MCP "
            "exposure and requires fresh workspace context plus git.enabled policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "git_log": RouterToolMetadata(
        name="git_log",
        description=(
            "Private read-only Git log wrapper; disabled from default public MCP "
            "exposure and requires fresh workspace context plus git.enabled policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "git_branch": RouterToolMetadata(
        name="git_branch",
        description=(
            "Private read-only Git branch wrapper; disabled from default public MCP "
            "exposure and requires fresh workspace context plus git.enabled policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "cron_list": RouterToolMetadata(
        name="cron_list",
        description=(
            "Private no-model cron listing wrapper; disabled from default public MCP "
            "exposure and requires fresh workspace context plus cron.enabled policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=False,
        requires_context=True,
    ),
    "cron_pause": RouterToolMetadata(
        name="cron_pause",
        description=(
            "Private no-model cron pause wrapper for script-only no_agent jobs; disabled "
            "from default public MCP exposure and requires fresh context plus cron policy."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "cron_resume": RouterToolMetadata(
        name="cron_resume",
        description=(
            "Private no-model cron resume wrapper for allowlisted script-only no_agent jobs; "
            "disabled from default public MCP exposure."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "cron_run": RouterToolMetadata(
        name="cron_run",
        description=(
            "Private no-model cron trigger wrapper for allowlisted script-only no_agent jobs; "
            "disabled from default public MCP exposure and never runs agent/model loops."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "cron_create_script_only": RouterToolMetadata(
        name="cron_create_script_only",
        description=(
            "Private no-model cron creation wrapper that only creates allowlisted script-only "
            "no_agent jobs; disabled from default public MCP exposure."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "message_send": RouterToolMetadata(
        name="message_send",
        description=(
            "Private no-model messaging dry-run scaffold; disabled from default public MCP "
            "exposure and requires fresh workspace context plus messaging.enabled and an "
            "explicit allowed_recipients policy entry."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
    "telegram_send": RouterToolMetadata(
        name="telegram_send",
        description=(
            "Private Telegram-specific no-model messaging dry-run scaffold; disabled from "
            "default public MCP exposure and requires fresh workspace context plus an "
            "explicit telegram:<recipient> allowlist entry."
        ),
        cost_class=COST_CLASS_NO_MODEL,
        llm_calls=0,
        enabled_by_default=False,
        mutates_state=True,
        requires_context=True,
    ),
}


# Static Phase 9 inventory of Hermes registry tools discovered from tools/*.py
# registry.register(name=...). Keep this list synchronized with the source
# registry so full-catalog parity has an explicit no-model representation for
# every Hermes tool without importing tool modules or invoking model paths.
HERMES_REGISTRY_TOOL_NAMES: tuple[str, ...] = (
    "browser_back",
    "browser_cdp",
    "browser_click",
    "browser_console",
    "browser_dialog",
    "browser_get_images",
    "browser_navigate",
    "browser_press",
    "browser_scroll",
    "browser_snapshot",
    "browser_type",
    "browser_vision",
    "clarify",
    "computer_use",
    "cronjob",
    "delegate_task",
    "discord",
    "discord_admin",
    "execute_code",
    "feishu_doc_read",
    "feishu_drive_add_comment",
    "feishu_drive_list_comment_replies",
    "feishu_drive_list_comments",
    "feishu_drive_reply_comment",
    "ha_call_service",
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "image_generate",
    "kanban_block",
    "kanban_comment",
    "kanban_complete",
    "kanban_create",
    "kanban_heartbeat",
    "kanban_link",
    "kanban_list",
    "kanban_show",
    "kanban_unblock",
    "memory",
    "mixture_of_agents",
    "patch",
    "process",
    "read_file",
    "read_terminal",
    "search_files",
    "send_message",
    "session_search",
    "skill_manage",
    "skill_view",
    "skills_list",
    "terminal",
    "text_to_speech",
    "todo",
    "video_analyze",
    "video_generate",
    "vision_analyze",
    "web_extract",
    "web_search",
    "write_file",
    "x_search",
    "yb_query_group_info",
    "yb_query_group_members",
    "yb_search_sticker",
    "yb_send_dm",
    "yb_send_sticker",
)

HERMES_CATALOG_MODEL_BACKED_TOOL_NAMES = frozenset(
    {
        "browser_vision",
        "computer_use",
        "delegate_task",
        "image_generate",
        "mixture_of_agents",
        "video_analyze",
        "video_generate",
        "vision_analyze",
        "web_extract",
        "x_search",
    }
)
HERMES_CATALOG_SIDE_EFFECT_TOOL_NAMES = frozenset(
    {
        "browser_back",
        "browser_cdp",
        "browser_click",
        "browser_console",
        "browser_dialog",
        "browser_navigate",
        "browser_press",
        "browser_scroll",
        "browser_type",
        "clarify",
        "computer_use",
        "cronjob",
        "delegate_task",
        "discord",
        "discord_admin",
        "execute_code",
        "feishu_drive_add_comment",
        "feishu_drive_reply_comment",
        "ha_call_service",
        "image_generate",
        "kanban_block",
        "kanban_comment",
        "kanban_complete",
        "kanban_create",
        "kanban_heartbeat",
        "kanban_link",
        "kanban_unblock",
        "memory",
        "patch",
        "process",
        "send_message",
        "skill_manage",
        "terminal",
        "text_to_speech",
        "todo",
        "video_generate",
        "write_file",
        "yb_send_dm",
        "yb_send_sticker",
    }
)
HERMES_CATALOG_TOOL_CAPABILITY_GROUPS: Mapping[str, str] = {
    "browser_back": "browser",
    "browser_cdp": "browser",
    "browser_click": "browser",
    "browser_console": "browser",
    "browser_dialog": "browser",
    "browser_get_images": "browser",
    "browser_navigate": "browser",
    "browser_press": "browser",
    "browser_scroll": "browser",
    "browser_snapshot": "browser",
    "browser_type": "browser",
    "browser_vision": "browser",
    "computer_use": "browser",
    "cronjob": "cron",
    "clarify": "api",
    "delegate_task": "api",
    "discord": "messaging",
    "discord_admin": "messaging",
    "execute_code": "terminal",
    "feishu_doc_read": "api",
    "feishu_drive_add_comment": "api",
    "feishu_drive_list_comment_replies": "api",
    "feishu_drive_list_comments": "api",
    "feishu_drive_reply_comment": "api",
    "ha_call_service": "api",
    "ha_get_state": "api",
    "ha_list_entities": "api",
    "ha_list_services": "api",
    "image_generate": "api",
    "kanban_block": "api",
    "kanban_comment": "api",
    "kanban_complete": "api",
    "kanban_create": "api",
    "kanban_heartbeat": "api",
    "kanban_link": "api",
    "kanban_list": "api",
    "kanban_show": "api",
    "kanban_unblock": "api",
    "memory": "memory",
    "mixture_of_agents": "api",
    "patch": "filesystem",
    "process": "terminal",
    "read_file": "filesystem",
    "read_terminal": "terminal",
    "search_files": "filesystem",
    "send_message": "messaging",
    "skill_manage": "skills",
    "terminal": "terminal",
    "text_to_speech": "api",
    "todo": "api",
    "video_analyze": "api",
    "video_generate": "api",
    "vision_analyze": "api",
    "web_extract": "web",
    "web_search": "web",
    "write_file": "filesystem",
    "x_search": "web",
    "yb_query_group_info": "messaging",
    "yb_query_group_members": "messaging",
    "yb_search_sticker": "messaging",
    "yb_send_dm": "messaging",
    "yb_send_sticker": "messaging",
}
HERMES_CATALOG_BLOCKED_TOOL_NAMES: tuple[str, ...] = tuple(
    name for name in HERMES_REGISTRY_TOOL_NAMES if name not in ROUTER_TOOL_METADATA
)

ROUTER_TOOL_METADATA = {
    **ROUTER_TOOL_METADATA,
    **{
        name: RouterToolMetadata(
            name=name,
            description=(
                f"Hermes registry tool {name!r} is catalog-visible for ChatGPT "
                "parity but blocked by this no-model connector until an explicit "
                "deterministic llm_calls=0 wrapper and policy are implemented."
            ),
            cost_class=COST_CLASS_NO_MODEL,
            llm_calls=0,
            enabled_by_default=False,
            mutates_state=False,
            requires_context=False,
            execution_status="blocked_no_model",
            blocked_reason=(
                "model_backed_tool_blocked"
                if name in HERMES_CATALOG_MODEL_BACKED_TOOL_NAMES
                else "requires_no_model_implementation"
            ),
            route_hint=_blocked_catalog_route_hint(name),
        )
        for name in HERMES_CATALOG_BLOCKED_TOOL_NAMES
    },
}


ROUTER_WORKSPACE_REQUIRED_TOOLS = frozenset(
    {
        "workspace_instructions_get",
        "workspace_context_status",
        "workspace_get",
        "workspace_close",
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_status_probe",
        "workspace_scratch_smoke",
        "workspace_diff",
        "file_read",
        "file_search",
        "file_patch",
        "patch_apply",
        "file_write",
        "file_move",
        "file_delete",
        "directory_create",
        "terminal_run",
        "process_start",
        "process_list",
        "process_poll",
        "process_log",
        "process_kill",
        "git_status",
        "git_diff",
        "git_log",
        "git_branch",
        "cron_list",
        "cron_pause",
        "cron_resume",
        "cron_run",
        "cron_create_script_only",
        "message_send",
        "telegram_send",
    }
)


def _router_capability_group(tool_name: str) -> str:
    if tool_name in HERMES_CATALOG_TOOL_CAPABILITY_GROUPS:
        return HERMES_CATALOG_TOOL_CAPABILITY_GROUPS[tool_name]
    if tool_name.startswith("profile") or tool_name == "profiles_list":
        return "profile"
    if tool_name in {"skills_list", "skill_view"}:
        return "skills"
    if tool_name == "session_search":
        return "session"
    if tool_name in {"viking_search", "viking_read"}:
        return "memory"
    if tool_name.startswith("git_"):
        return "git"
    if tool_name.startswith("cron_"):
        return "cron"
    if tool_name in {"message_send", "telegram_send"}:
        return "messaging"
    if tool_name == "terminal_run" or tool_name.startswith("process_"):
        return "terminal"
    if tool_name in {
        "workspace_file_list",
        "workspace_file_read",
        "workspace_file_stat",
        "workspace_file_search",
        "workspace_status_probe",
        "workspace_scratch_smoke",
        "workspace_diff",
        "file_read",
        "file_search",
        "file_patch",
        "patch_apply",
        "file_write",
        "file_move",
        "file_delete",
        "directory_create",
    }:
        return "filesystem"
    if tool_name.startswith("workspace_"):
        return "workspace"
    return PROFILE_ROUTER_TOOL_GROUP


def get_router_tool_metadata() -> dict[str, dict]:
    """Return serializable metadata for all current profile-router tools."""

    metadata: dict[str, dict] = {}
    for name, meta in ROUTER_TOOL_METADATA.items():
        data = asdict(meta)
        data.update(
            {
                "allowed_by_default": meta.enabled_by_default,
                "side_effects": meta.mutates_state,
                "requires_workspace": name in ROUTER_WORKSPACE_REQUIRED_TOOLS,
                "requires_approval": meta.mutates_state,
                "capability_group": _router_capability_group(name),
            }
        )
        metadata[name] = data
    return metadata


def assert_default_tools_are_no_model(
    metadata: Mapping[str, RouterToolMetadata] = ROUTER_TOOL_METADATA,
) -> None:
    """Fail closed if a default-exposed router tool may spend model tokens."""

    assert_no_model_loop_tools_absent(metadata)
    unsafe = [
        meta.name
        for meta in metadata.values()
        if meta.enabled_by_default
        and (meta.cost_class not in NO_MODEL_DEFAULT_COST_CLASSES or meta.llm_calls != 0)
    ]
    if unsafe:
        raise ProfileRouterError(
            "unsafe_default_tool_exposure",
            "Default profile-router tools must be no-model: " + ", ".join(sorted(unsafe)),
        )


def assert_no_model_loop_tools_absent(
    tool_names_or_metadata: Iterable[str] | Mapping[str, Any] = ROUTER_TOOL_METADATA,
) -> None:
    """Fail closed if model-loop names are executable instead of blocked."""

    if isinstance(tool_names_or_metadata, MappingABC):
        unsafe: list[str] = []
        for raw_name, meta in tool_names_or_metadata.items():
            name = str(raw_name).strip().lower()
            if name not in FORBIDDEN_MODEL_LOOP_TOOL_NAMES:
                continue
            if isinstance(meta, MappingABC):
                execution_status = meta.get("execution_status")
                llm_calls = meta.get("llm_calls")
                cost_class = meta.get("cost_class")
            else:
                execution_status = getattr(meta, "execution_status", None)
                llm_calls = getattr(meta, "llm_calls", None)
                cost_class = getattr(meta, "cost_class", None)
            if (
                execution_status == "blocked_no_model"
                and llm_calls == 0
                and cost_class == COST_CLASS_NO_MODEL
            ):
                continue
            unsafe.append(name)
    else:
        normalized_names = {str(name).strip().lower() for name in tool_names_or_metadata}
        unsafe = sorted(normalized_names & FORBIDDEN_MODEL_LOOP_TOOL_NAMES)
    if unsafe:
        raise ProfileRouterError(
            "model_loop_tool_exposure_forbidden",
            "Model-loop tools must be blocked no-model catalog entries, not executable: "
            + ", ".join(sorted(unsafe)),
        )


@dataclass(frozen=True)
class WorkspaceMetadata:
    """Server-side metadata for one opened profile-router workspace.

    Workspace IDs are opaque to MCP clients.  The root is kept in server-side
    metadata and every future file/search/write/terminal path must resolve
    through this object before touching the filesystem.
    """

    workspace_id: str
    profile_ref: str
    host: str
    profile: str
    root: str
    mode: str = "checkout"
    read_only: bool = True
    cost_class: str = COST_CLASS_NO_MODEL
    llm_calls: int = 0

    def to_public_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProfileSkillRecord:
    """Internal profile skill metadata with host paths kept server-side only."""

    skill_id: str
    name: str
    category: str
    description: str
    tags: tuple[str, ...]
    skill_dir: Path
    skill_file: Path
    linked_files: Mapping[str, tuple[str, ...]]
    content_truncated: bool = False


@dataclass(frozen=True)
class WorkspaceContextSnapshot:
    """Server-side record proving workspace instructions were hydrated."""

    workspace_id: str
    context_token: str
    hashes: Mapping[str, str]


def _local_profile_exists(profile: str) -> bool:
    return any(info.name == profile for info in _list_local_profile_infos())


def _resolve_allowed_local_roots(allowed_roots: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for allowed_root in allowed_roots:
        _ensure_not_secret_path(allowed_root)
        resolved.append(_resolve_existing_local_path(allowed_root, "allowed root"))
    return resolved


def create_workspace_metadata(
    profile_ref: str,
    root: str,
    *,
    mode: str = "checkout",
    workspace_id: str | None = None,
    policy: ProfileRouterPolicy | None = None,
) -> WorkspaceMetadata:
    """Validate a local read workspace and return opaque no-model metadata.

    This helper intentionally does not expose an MCP tool yet.  It is the common
    safety gate future ``workspace_open``, ``file_read``, and ``file_search``
    handlers must use before resolving paths.
    """

    assert_default_tools_are_no_model()
    if mode not in {"checkout", "worktree", "read_only"}:
        raise ProfileRouterError("invalid_workspace_mode", f"Unsupported workspace mode: {mode}")

    ref = parse_profile_ref(profile_ref)
    if ref.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local workspaces are supported by this skeleton, got: {ref.host}",
        )
    if not _local_profile_exists(ref.profile):
        raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}")

    router_policy = policy or load_profile_router_policy()
    route_policy = router_policy.get_profile_policy(ref)
    if not route_policy.allow_filesystem_read:
        raise ProfileRouterError(
            "filesystem_read_not_allowed",
            f"Filesystem read is disabled by profile_router policy: {ref.value}",
        )

    normalized_root = _normalize_absolute_host_path(root, "workspace root")
    _ensure_not_secret_path(normalized_root)
    if not any(_path_within_root(normalized_root, allowed_root) for allowed_root in route_policy.allowed_roots):
        raise ProfileRouterError(
            "workspace_root_not_allowed",
            f"Workspace root is outside allowed roots for {ref.value}",
        )

    resolved_root = _resolve_existing_local_path(normalized_root, "workspace root")
    allowed_roots = _resolve_allowed_local_roots(route_policy.allowed_roots)
    if not any(_path_is_relative_to(resolved_root, allowed_root) for allowed_root in allowed_roots):
        raise ProfileRouterError(
            "symlink_traversal_denied",
            f"Resolved workspace root escapes allowed roots for {ref.value}",
        )

    return WorkspaceMetadata(
        workspace_id=workspace_id or f"ws_{uuid4().hex}",
        profile_ref=ref.value,
        host=ref.host,
        profile=ref.profile,
        root=str(resolved_root),
        mode=mode,
        read_only=True,
    )


def resolve_workspace_path(
    workspace: WorkspaceMetadata,
    path: str,
    *,
    require_exists: bool = True,
) -> str:
    """Resolve a client path inside a workspace and reject escapes/secrets."""

    if not isinstance(workspace, WorkspaceMetadata):
        raise ProfileRouterError("invalid_workspace", "workspace metadata is required")
    if workspace.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local workspace paths are supported by this skeleton, got: {workspace.host}",
        )
    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", "path must be a string")

    raw_path = path.strip() or "."
    if raw_path.startswith("/"):
        raise ProfileRouterError("absolute_path_not_allowed", "path must be workspace-relative")

    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, raw_path))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    normalized_relative_path = posixpath.normpath(raw_path)
    allow_plan_directory = normalized_relative_path == SAFE_WORKSPACE_PLAN_DIR
    _ensure_workspace_read_path_not_secret(
        normalized_candidate,
        normalized_relative_path,
        allow_plan_directory=allow_plan_directory,
    )

    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    if require_exists:
        resolved_candidate = _resolve_existing_local_path(normalized_candidate, "workspace path")
    else:
        resolved_parent = _resolve_existing_local_path(
            posixpath.dirname(normalized_candidate), "workspace path parent"
        )
        resolved_candidate = resolved_parent / posixpath.basename(normalized_candidate)

    if not _path_is_relative_to(resolved_candidate, resolved_root):
        raise ProfileRouterError("symlink_traversal_denied", "path escapes workspace root")
    resolved_relative_path = _resolved_relative_workspace_path(resolved_root, resolved_candidate)
    if _is_safe_workspace_plan_path(
        normalized_relative_path,
        allow_directory=allow_plan_directory,
    ) and resolved_relative_path != normalized_relative_path:
        raise ProfileRouterError("symlink_traversal_denied", "workspace plan path may not be a symlink")
    _ensure_workspace_read_path_not_secret(
        str(resolved_candidate),
        resolved_relative_path,
        allow_plan_directory=allow_plan_directory,
    )
    return str(resolved_candidate)


def resolve_workspace_write_path(workspace: WorkspaceMetadata, path: str) -> Path:
    """Resolve a workspace-relative write path without following unsafe symlinks."""

    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", "path must be a string")
    raw_path = path.strip()
    if not raw_path or raw_path == ".":
        raise ProfileRouterError("invalid_path", "write path must name a file")
    if raw_path.startswith("/"):
        raise ProfileRouterError("absolute_path_not_allowed", "path must be workspace-relative")

    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, raw_path))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    _ensure_not_secret_path(normalized_candidate)

    candidate = Path(normalized_candidate)
    if candidate.is_symlink():
        raise ProfileRouterError("symlink_traversal_denied", "write path may not be a symlink")
    if candidate.exists():
        resolved = Path(resolve_workspace_path(workspace, raw_path, require_exists=True))
        if resolved.is_dir():
            raise ProfileRouterError("not_a_file", "write path must be a file")
        return resolved
    return Path(resolve_workspace_path(workspace, raw_path, require_exists=False))


def resolve_workspace_directory_create_path(
    workspace: WorkspaceMetadata,
    path: str,
    *,
    parents: bool = False,
) -> Path:
    """Resolve a workspace-relative directory target without following symlinks."""

    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", "path must be a string")
    raw_path = path.strip()
    if not raw_path or raw_path == ".":
        raise ProfileRouterError("invalid_path", "directory path must name a subdirectory")
    if raw_path.startswith("/"):
        raise ProfileRouterError("absolute_path_not_allowed", "path must be workspace-relative")

    normalized_relative = posixpath.normpath(raw_path)
    if normalized_relative in {"", ".", ".."} or normalized_relative.startswith("../"):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, normalized_relative))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    _ensure_not_secret_path(normalized_candidate)

    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    current = resolved_root
    parts = [part for part in normalized_relative.split("/") if part]
    for index, part in enumerate(parts):
        if part in {".", ".."}:
            raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
        child = current / part
        is_last = index == len(parts) - 1
        if child.is_symlink():
            raise ProfileRouterError("symlink_traversal_denied", "directory path may not traverse a symlink")
        if child.exists():
            try:
                resolved_child = child.resolve(strict=True)
            except OSError as exc:
                raise ProfileRouterError("invalid_path", "directory path is not accessible") from exc
            if not _path_is_relative_to(resolved_child, resolved_root):
                raise ProfileRouterError("symlink_traversal_denied", "directory path escapes workspace root")
            if not resolved_child.is_dir():
                raise ProfileRouterError("not_a_directory", "directory path collides with a file")
            current = resolved_child
            continue
        if not is_last and not parents:
            raise ProfileRouterError("parent_directory_not_found", "parent directory does not exist")
        current = child

    return Path(normalized_candidate)


def _normalize_workspace_mutation_relative_path(path: str, *, label: str) -> str:
    if not isinstance(path, str):
        raise ProfileRouterError("invalid_path", f"{label} must be a string")
    raw_path = path.strip()
    if not raw_path or raw_path == ".":
        raise ProfileRouterError("invalid_path", f"{label} must name a file")
    if raw_path.startswith("/"):
        raise ProfileRouterError("absolute_path_not_allowed", "path must be workspace-relative")
    normalized_relative = posixpath.normpath(raw_path)
    if normalized_relative in {"", ".", ".."} or normalized_relative.startswith("../"):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    return normalized_relative


def resolve_workspace_move_source_path(workspace: WorkspaceMetadata, path: str) -> tuple[Path, str]:
    """Resolve an existing workspace-relative regular file source without symlinks."""

    normalized_relative = _normalize_workspace_mutation_relative_path(path, label="source_path")
    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, normalized_relative))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    _ensure_not_secret_path(normalized_candidate)

    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    current = resolved_root
    parts = [part for part in normalized_relative.split("/") if part]
    for index, part in enumerate(parts):
        child = current / part
        if child.is_symlink():
            raise ProfileRouterError("symlink_traversal_denied", "move source may not traverse a symlink")
        if not child.exists():
            raise ProfileRouterError("file_not_found", "move source does not exist")
        try:
            resolved_child = child.resolve(strict=True)
        except OSError as exc:
            raise ProfileRouterError("invalid_path", "move source is not accessible") from exc
        if not _path_is_relative_to(resolved_child, resolved_root):
            raise ProfileRouterError("symlink_traversal_denied", "move source escapes workspace root")
        is_last = index == len(parts) - 1
        if not is_last:
            if not resolved_child.is_dir():
                raise ProfileRouterError("parent_not_directory", "move source parent is not a directory")
            current = resolved_child
            continue
        if not resolved_child.is_file():
            raise ProfileRouterError("not_a_file", "move source must be a file")
        return resolved_child, normalized_relative

    raise ProfileRouterError("invalid_path", "source_path must name a file")


def resolve_workspace_delete_path(workspace: WorkspaceMetadata, path: str) -> tuple[Path, str]:
    """Resolve an existing workspace-relative regular file delete target without symlinks."""

    normalized_relative = _normalize_workspace_mutation_relative_path(path, label="path")
    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, normalized_relative))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    _ensure_not_secret_path(normalized_candidate)

    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    current = resolved_root
    parts = [part for part in normalized_relative.split("/") if part]
    for index, part in enumerate(parts):
        child = current / part
        if child.is_symlink():
            raise ProfileRouterError("symlink_traversal_denied", "delete path may not traverse a symlink")
        if not child.exists():
            raise ProfileRouterError("file_not_found", "delete target does not exist")
        try:
            resolved_child = child.resolve(strict=True)
        except OSError as exc:
            raise ProfileRouterError("invalid_path", "delete target is not accessible") from exc
        if not _path_is_relative_to(resolved_child, resolved_root):
            raise ProfileRouterError("symlink_traversal_denied", "delete path escapes workspace root")
        is_last = index == len(parts) - 1
        if not is_last:
            if not resolved_child.is_dir():
                raise ProfileRouterError("parent_not_directory", "delete target parent is not a directory")
            current = resolved_child
            continue
        if not resolved_child.is_file():
            raise ProfileRouterError("not_a_file", "delete target must be a file")
        return resolved_child, normalized_relative

    raise ProfileRouterError("invalid_path", "path must name a file")


def resolve_workspace_move_destination_path(workspace: WorkspaceMetadata, path: str) -> tuple[Path, str]:
    """Resolve a non-existing workspace-relative file move destination without symlinks."""

    normalized_relative = _normalize_workspace_mutation_relative_path(path, label="destination_path")
    normalized_candidate = posixpath.normpath(posixpath.join(workspace.root, normalized_relative))
    if not _path_within_root(normalized_candidate, workspace.root):
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root")
    _ensure_not_secret_path(normalized_candidate)

    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    parts = [part for part in normalized_relative.split("/") if part]
    current = resolved_root
    for part in parts[:-1]:
        child = current / part
        if child.is_symlink():
            raise ProfileRouterError("symlink_traversal_denied", "move destination may not traverse a symlink")
        if not child.exists():
            raise ProfileRouterError("parent_directory_not_found", "destination parent directory does not exist")
        try:
            resolved_child = child.resolve(strict=True)
        except OSError as exc:
            raise ProfileRouterError("invalid_path", "move destination parent is not accessible") from exc
        if not _path_is_relative_to(resolved_child, resolved_root):
            raise ProfileRouterError("symlink_traversal_denied", "move destination escapes workspace root")
        if not resolved_child.is_dir():
            raise ProfileRouterError("parent_not_directory", "destination parent is not a directory")
        current = resolved_child

    destination = current / parts[-1]
    if destination.is_symlink():
        raise ProfileRouterError("symlink_traversal_denied", "move destination may not be a symlink")
    if destination.exists():
        raise ProfileRouterError("destination_already_exists", "move destination already exists")
    if not _path_is_relative_to(destination.parent.resolve(strict=True), resolved_root):
        raise ProfileRouterError("symlink_traversal_denied", "move destination escapes workspace root")
    return destination, normalized_relative


class WorkspaceRegistry:
    """In-memory server-side registry for opaque workspace IDs.

    MCP clients only receive the opaque ``workspace_id``. Host-local roots stay
    in this registry and every read/search path is resolved through
    ``resolve_workspace_path`` before file access.
    """

    def __init__(self) -> None:
        self._workspaces: dict[str, WorkspaceMetadata] = {}
        self._contexts: dict[str, WorkspaceContextSnapshot] = {}

    def open(
        self,
        profile_ref: str,
        root: str,
        *,
        mode: str = "checkout",
    ) -> WorkspaceMetadata:
        workspace = create_workspace_metadata(profile_ref, root, mode=mode)
        self._workspaces[workspace.workspace_id] = workspace
        return workspace

    def get(self, workspace_id: str) -> WorkspaceMetadata:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ProfileRouterError("invalid_workspace", "workspace_id is required")
        workspace = self._workspaces.get(workspace_id.strip())
        if workspace is None:
            raise ProfileRouterError(
                "workspace_not_found", f"Workspace is not open: {workspace_id}"
            )
        return workspace

    def close(self, workspace_id: str) -> WorkspaceMetadata:
        workspace = self.get(workspace_id)
        del self._workspaces[workspace.workspace_id]
        self._contexts.pop(workspace.workspace_id, None)
        return workspace

    def record_context(
        self,
        workspace_id: str,
        *,
        context_token: str,
        hashes: Mapping[str, str],
    ) -> WorkspaceContextSnapshot:
        workspace = self.get(workspace_id)
        snapshot = WorkspaceContextSnapshot(
            workspace_id=workspace.workspace_id,
            context_token=context_token,
            hashes=dict(hashes),
        )
        self._contexts[workspace.workspace_id] = snapshot
        return snapshot

    def get_context(self, workspace_id: str) -> WorkspaceContextSnapshot | None:
        workspace = self.get(workspace_id)
        return self._contexts.get(workspace.workspace_id)


DEFAULT_WORKSPACE_REGISTRY = WorkspaceRegistry()


def _public_workspace_dict(workspace: WorkspaceMetadata) -> dict:
    """Return workspace metadata safe for an external MCP client.

    The host-local root is intentionally omitted; clients must use the opaque
    ID and workspace-relative paths instead of learning or replaying server
    absolute paths.
    """

    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "host": workspace.host,
        "profile": workspace.profile,
        "mode": workspace.mode,
        "read_only": workspace.read_only,
        "cost_class": workspace.cost_class,
        "llm_calls": workspace.llm_calls,
    }


def _stable_json_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> tuple[str, bool]:
    """Hash a context file without returning its full contents."""

    digest = hashlib.sha256()
    total = 0
    truncated = False
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_CONTEXT_HASH_BYTES:
                    digest.update(chunk[: max(0, len(chunk) - (total - MAX_CONTEXT_HASH_BYTES))])
                    truncated = True
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ProfileRouterError("context_file_not_readable", f"Context file is not readable: {path.name}") from exc
    return digest.hexdigest(), truncated


def _redact_sensitive_text_fields(text: str) -> str:
    """Best-effort redaction for secret-looking fields in bounded outputs."""

    if not isinstance(text, str) or not text:
        return text
    redacted = re.sub(
        r"(?im)(\b[A-Z0-9_.-]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|"
        r"REFRESH[_-]?TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE[_-]?KEY|"
        r"CREDENTIALS?)[A-Z0-9_.-]*\b\s*[:=]\s*)"
        r"([^\s,;#]+|\"[^\"\n]*\"|'[^'\n]*')",
        r"\1[REDACTED]",
        text,
    )
    redacted = re.sub(
        r"(?im)(\"(?:api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|"
        r"token|secret|password|passwd|private[_-]?key|credentials?)\"\s*:\s*)"
        r"(\"[^\"\n]*\"|[^,}\n]+)",
        r"\1\"[REDACTED]\"",
        redacted,
    )
    redacted = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r"\b(?:sk|pk|xox[baprs])-[-A-Za-z0-9_]{12,}\b", "[REDACTED_TOKEN]", redacted)
    return redacted


def _redact_context_excerpt(text: str) -> str:
    """Best-effort secret redaction for bounded instruction excerpts."""

    return _redact_sensitive_text_fields(text)


def _is_explicit_profile_context_file(
    path: Path,
    public_path: str,
    profile_dir: Path | None,
) -> bool:
    """Return whether a file is a profile instruction allowlisted by name.

    Real Hermes profiles usually live under ``~/.hermes/profiles/<profile>``.
    The broad secret denylist intentionally blocks ``.hermes`` for workspace
    reads, but profile context hydration still needs to read a tiny, explicit
    set of non-secret profile instruction files such as ``SOUL.md``.  This
    helper keeps that exception narrow: only direct children of the selected
    profile directory whose relative path is listed in ``PROFILE_CONTEXT_FILES``
    qualify.  Secret-looking filenames still fail closed.
    """

    if profile_dir is None:
        return False
    try:
        relative_path = path.relative_to(profile_dir)
    except ValueError:
        return False
    relative_text = relative_path.as_posix()
    return (
        relative_text == public_path
        and relative_text in PROFILE_CONTEXT_FILES
        and not _is_secret_path(relative_text)
    )


def _read_context_file(
    path: Path,
    public_path: str,
    *,
    profile_context_root: Path | None = None,
) -> dict:
    """Read bounded, sanitized context-file metadata and excerpt."""

    if _is_secret_path(str(path)) and not _is_explicit_profile_context_file(
        path,
        public_path,
        profile_context_root,
    ):
        raise ProfileRouterError(
            "secret_path_denied",
            "Path is blocked by the profile-router secret denylist",
        )
    try:
        stat = path.stat()
    except OSError as exc:
        raise ProfileRouterError("context_file_not_readable", f"Context file is not accessible: {public_path}") from exc
    if not path.is_file():
        raise ProfileRouterError("context_file_not_readable", f"Context path is not a file: {public_path}")

    sha256, hash_truncated = _hash_file(path)
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_CONTEXT_FILE_BYTES + 1)
    except OSError as exc:
        raise ProfileRouterError("context_file_not_readable", f"Context file is not readable: {public_path}") from exc
    if b"\x00" in raw[:4096]:
        raise ProfileRouterError("binary_file_not_supported", f"Context file is binary: {public_path}")

    truncated = len(raw) > MAX_CONTEXT_FILE_BYTES or len(raw.decode("utf-8", errors="replace")) > MAX_CONTEXT_FILE_CHARS
    text = raw[:MAX_CONTEXT_FILE_BYTES].decode("utf-8", errors="replace")[:MAX_CONTEXT_FILE_CHARS]
    return {
        "path": public_path,
        "sha256": sha256,
        "hash_truncated": hash_truncated,
        "size_bytes": stat.st_size,
        "truncated": truncated,
        "excerpt": _redact_context_excerpt(text),
    }


def _require_local_profile_policy(profile_ref: str) -> tuple[ProfileRef, ProfileRoutePolicy, ProfileRouterPolicy]:
    ref = parse_profile_ref(profile_ref)
    if ref.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local profiles are supported by this skeleton, got: {ref.host}",
        )
    if not _local_profile_exists(ref.profile):
        raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}")
    router_policy = load_profile_router_policy()
    route_policy = router_policy.get_profile_policy(ref)
    return ref, route_policy, router_policy


def _resolve_local_profile_dir(ref: ProfileRef) -> Path:
    """Resolve a local profile directory without allowing symlink escapes."""

    profile_dir_path = get_profile_dir(ref.profile)
    if profile_dir_path.is_symlink():
        raise ProfileRouterError(
            "profile_symlink_denied",
            f"Profile directory may not be a symlink: {ref.value}",
        )
    if ref.profile != "default" and profile_dir_path.parent.is_symlink():
        raise ProfileRouterError(
            "profile_symlink_denied",
            f"Profile parent directory may not be a symlink: {ref.value}",
        )
    try:
        profile_dir = profile_dir_path.resolve(strict=True)
    except OSError as exc:
        raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}") from exc
    if not profile_dir.is_dir():
        raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}")
    if ref.profile != "default":
        profile_parent = profile_dir_path.parent.resolve(strict=True)
        if not _path_is_relative_to(profile_dir, profile_parent):
            raise ProfileRouterError(
                "profile_symlink_denied",
                f"Profile directory escapes profiles root: {ref.value}",
            )
    return profile_dir


def _public_policy_context(route_policy: ProfileRoutePolicy, router_policy: ProfileRouterPolicy) -> dict:
    return {
        "enabled": router_policy.is_profile_exposed(route_policy),
        "allowed_roots": list(route_policy.allowed_roots),
        "allowed_tool_groups": list(route_policy.allowed_tool_groups),
        "capabilities": {
            "filesystem_read": route_policy.allow_filesystem_read,
            "filesystem_write": route_policy.allow_filesystem_write,
            "terminal": route_policy.allow_terminal,
            "messaging": route_policy.allow_messaging,
            "cron": route_policy.allow_cron,
            "git": route_policy.capability_enabled("git"),
            "git_push": route_policy.allow_git_push,
            "deploy": route_policy.allow_deploy,
            "skills": route_policy.capability_enabled("skills"),
            "memory": route_policy.capability_enabled("memory"),
            "session": route_policy.capability_enabled("session"),
            "web": route_policy.capability_enabled("web"),
            "browser": route_policy.capability_enabled("browser"),
            "api": route_policy.capability_enabled("api"),
        },
        "capability_groups": dict(route_policy.capability_groups),
        "context": {
            "skills": {"read": route_policy.allow_context_skills_read},
            "sessions": {"search": route_policy.allow_context_sessions_search},
        },
        "messaging_recipients_configured": bool(route_policy.messaging_allowed_recipients),
        "messaging_policy": {
            "enabled": route_policy.allow_messaging,
            "allowed_recipients_count": len(route_policy.messaging_allowed_recipients),
            "allowlist_redacted": True,
            "broadcast_allowed": False,
            "external_delivery_enabled": False,
            "public_mcp_exposure": "disabled_pending_phase_10_messaging_policy_review",
        },
        "terminal_execution_policy": {
            "enabled": route_policy.terminal_execution_policy.enabled,
            "require_no_shell": route_policy.terminal_execution_policy.require_no_shell,
            "allowed_commands_count": len(
                route_policy.terminal_execution_policy.allowed_commands
            ),
            "allowed_command_prefixes_count": len(
                route_policy.terminal_execution_policy.allowed_command_prefixes
            ),
            "allowlist_redacted": True,
            "public_mcp_exposure": "disabled_pending_http_auth_config_review",
        },
        "cron_policy": {
            "enabled": route_policy.cron_policy.enabled,
            "allowed_scripts_count": len(route_policy.cron_policy.allowed_scripts),
            "allowlist_redacted": True,
            "no_agent_only": True,
            "model_backed_crons_allowed": False,
            "public_mcp_exposure": "disabled_pending_phase_10_cron_parity_policy_review",
        },
        "protected_branches": list(route_policy.protected_branches),
        "model_policy": {
            "allow_model_tools": route_policy.allow_model_tools,
            "allowed_cost_classes": list(route_policy.allowed_cost_classes),
            "default_public_cost_classes": sorted(NO_MODEL_DEFAULT_COST_CLASSES),
        },
    }


def build_profile_context(profile_ref: str) -> dict:
    """Build bounded profile policy/SOUL context without invoking a model."""

    assert_default_tools_are_no_model()
    ref, route_policy, router_policy = _require_local_profile_policy(profile_ref)
    profile_dir = _resolve_local_profile_dir(ref)
    policy_context = _public_policy_context(route_policy, router_policy)

    files: list[dict] = []
    for filename in PROFILE_CONTEXT_FILES:
        candidate = profile_dir / filename
        if not candidate.exists():
            continue
        resolved = candidate.resolve(strict=True)
        if not _path_is_relative_to(resolved, profile_dir):
            raise ProfileRouterError("symlink_traversal_denied", f"Profile context file escapes profile root: {filename}")
        files.append(
            _read_context_file(
                resolved,
                filename,
                profile_context_root=profile_dir,
            )
        )

    hashes = {f"profile:{item['path']}": item["sha256"] for item in files}
    hashes["profile:policy"] = _stable_json_hash(policy_context)
    context_token = _stable_json_hash(hashes)
    return {
        "profile_ref": ref.value,
        "host": ref.host,
        "profile": ref.profile,
        "context_token": context_token,
        "context_hashes": hashes,
        "policy": policy_context,
        "profile_instructions": files,
        "instruction_hierarchy": [
            "router_security_policy",
            "profile_SOUL_and_profile_router_policy",
            "workspace_AGENTS_and_project_instructions",
            "untrusted_file_or_web_content",
        ],
        "secret_handling": {
            "secret_files_excluded": [".env", "auth.json", ".ssh", "mcp_tokens"],
            "funciones_txt_content_excluded": True,
        },
    }


def _workspace_instruction_snapshots(workspace: WorkspaceMetadata) -> list[dict]:
    files: list[dict] = []
    for filename in WORKSPACE_INSTRUCTION_FILES:
        candidate = Path(workspace.root) / filename
        if not candidate.exists():
            continue
        resolved = Path(resolve_workspace_path(workspace, filename))
        files.append(_read_context_file(resolved, filename))
    return files


def _workspace_funciones_metadata(workspace: WorkspaceMetadata) -> dict:
    metadata = {
        "path": FUNCIONES_TXT_FILENAME,
        "exists": False,
        "content_included": False,
        "git_policy": "never_stage_commit_push_or_include_in_pr",
    }
    candidate = Path(workspace.root) / FUNCIONES_TXT_FILENAME
    if not candidate.exists():
        return metadata
    metadata["exists"] = True
    metadata["status"] = "excluded_from_context_bundle"
    return metadata


def build_workspace_context(workspace_id: str, *, registry: WorkspaceRegistry | None = None) -> dict:
    """Build bounded workspace instructions and profile context for ChatGPT."""

    assert_default_tools_are_no_model()
    workspace = (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)
    profile_context = build_profile_context(workspace.profile_ref)
    workspace_files = _workspace_instruction_snapshots(workspace)
    hashes = dict(profile_context["context_hashes"])
    hashes.update({f"workspace:{item['path']}": item["sha256"] for item in workspace_files})
    funciones = _workspace_funciones_metadata(workspace)
    hashes["workspace:funciones_txt_exists"] = str(bool(funciones["exists"])).lower()
    context_token = _stable_json_hash(hashes)
    return {
        "workspace_id": workspace.workspace_id,
        "workspace": _public_workspace_dict(workspace),
        "context_token": context_token,
        "context_hashes": hashes,
        "context_loaded": True,
        "profile_context": profile_context,
        "workspace_instructions": workspace_files,
        "funciones_txt": funciones,
        "enforcement": {
            "powerful_tools_require_fresh_context": True,
            "fail_closed_errors": ["context_not_loaded", "context_stale"],
            "custom_gpt_instructions_are_not_enforcement": True,
        },
    }


def hydrate_workspace_context(workspace_id: str, *, registry: WorkspaceRegistry | None = None) -> dict:
    """Build and record current workspace context for future powerful tools."""

    selected_registry = registry or DEFAULT_WORKSPACE_REGISTRY
    context = build_workspace_context(workspace_id, registry=selected_registry)
    selected_registry.record_context(
        workspace_id,
        context_token=context["context_token"],
        hashes=context["context_hashes"],
    )
    return context


def get_workspace_context_status(workspace_id: str, *, registry: WorkspaceRegistry | None = None) -> dict:
    """Return context freshness for an opened workspace."""

    selected_registry = registry or DEFAULT_WORKSPACE_REGISTRY
    workspace = selected_registry.get(workspace_id)
    snapshot = selected_registry.get_context(workspace_id)
    if snapshot is None:
        return {
            "workspace_id": workspace.workspace_id,
            "context_loaded": False,
            "state": "not_loaded",
            "context_required": True,
            "next_tool": "workspace_instructions_get",
        }

    current = build_workspace_context(workspace_id, registry=selected_registry)
    stale = current["context_token"] != snapshot.context_token
    return {
        "workspace_id": workspace.workspace_id,
        "context_loaded": True,
        "state": "stale" if stale else "loaded",
        "context_required": stale,
        "context_token": snapshot.context_token,
        "current_context_token": current["context_token"],
    }


def require_fresh_workspace_context(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Fail closed unless workspace context was hydrated and is still fresh."""

    selected_registry = registry or DEFAULT_WORKSPACE_REGISTRY
    status = get_workspace_context_status(workspace_id, registry=selected_registry)
    if not status["context_loaded"]:
        raise ProfileRouterError(
            "context_not_loaded",
            "Workspace context must be loaded with workspace_instructions_get before powerful tools",
        )
    if status["state"] != "loaded":
        raise ProfileRouterError(
            "context_stale",
            "Workspace context is stale; refresh with workspace_instructions_get before powerful tools",
        )
    if context_token is not None and context_token != status["context_token"]:
        raise ProfileRouterError(
            "context_stale",
            "Provided context_token does not match the loaded workspace context",
        )
    return selected_registry.get(workspace_id)


def _bounded_int(
    value: int | None,
    field: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileRouterError("invalid_pagination", f"{field} must be an integer")
    if value < minimum:
        raise ProfileRouterError(
            "invalid_pagination", f"{field} must be >= {minimum}"
        )
    return min(value, maximum)


def _ensure_text_file(path: Path) -> None:
    if not path.is_file():
        raise ProfileRouterError("not_a_file", f"Path is not a file: {path.name}")
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"File is not readable: {path.name}") from exc
    if b"\x00" in sample:
        raise ProfileRouterError("binary_file_not_supported", "Binary files are not readable")


def open_workspace(
    profile_ref: str,
    root: str,
    *,
    mode: str = "checkout",
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Open a read-only local workspace through the policy/secret gate."""

    assert_default_tools_are_no_model()
    return (registry or DEFAULT_WORKSPACE_REGISTRY).open(profile_ref, root, mode=mode)


def get_workspace(
    workspace_id: str,
    *,
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Return metadata for an opened workspace without invoking a model."""

    assert_default_tools_are_no_model()
    return (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)


def close_workspace(
    workspace_id: str,
    *,
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Remove an opened workspace from the server-side registry."""

    assert_default_tools_are_no_model()
    active_registry = registry or DEFAULT_WORKSPACE_REGISTRY
    workspace = active_registry.get(workspace_id)
    if active_registry is DEFAULT_WORKSPACE_REGISTRY:
        _WORKSPACE_PROCESS_REGISTRY.kill_all_for_workspace(workspace)
    return active_registry.close(workspace.workspace_id)


def read_workspace_file(
    workspace_id: str,
    path: str,
    *,
    offset: int | None = 1,
    limit: int | None = MAX_FILE_READ_LINES,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Read a bounded text slice from an opened workspace."""

    assert_default_tools_are_no_model()
    line_offset = _bounded_int(
        offset,
        "offset",
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    line_limit = _bounded_int(
        limit,
        "limit",
        default=MAX_FILE_READ_LINES,
        minimum=1,
        maximum=MAX_FILE_READ_LINES,
    )
    workspace = (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)
    resolved_path = Path(resolve_workspace_path(workspace, path))
    _ensure_text_file(resolved_path)

    selected_lines: list[str] = []
    chars = 0
    truncated = False
    try:
        with resolved_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < line_offset:
                    continue
                if len(selected_lines) >= line_limit:
                    truncated = True
                    break
                remaining_chars = MAX_FILE_READ_CHARS - chars
                if remaining_chars <= 0:
                    truncated = True
                    break
                if len(line) > remaining_chars:
                    selected_lines.append(line[:remaining_chars])
                    truncated = True
                    break
                selected_lines.append(line)
                chars += len(line)
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"File is not readable: {path}") from exc

    content = _redact_sensitive_text_fields("".join(selected_lines))
    return {
        "workspace_id": workspace.workspace_id,
        "path": posixpath.normpath(path.strip() or "."),
        "offset": line_offset,
        "limit": line_limit,
        "content": content,
        "lines_returned": len(selected_lines),
        "truncated": truncated,
    }


def _list_workspace_file_entry(workspace: WorkspaceMetadata, path: Path) -> dict:
    rel_path = _workspace_relative_path(workspace, path)
    try:
        stat = path.stat()
    except OSError:
        stat = None
    return {
        "path": rel_path,
        "type": "directory" if path.is_dir() else "file",
        "size_bytes": stat.st_size if stat is not None and path.is_file() else None,
    }


def stat_workspace_file(
    workspace_id: str,
    path: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Return bounded file/directory metadata after context hydration."""

    assert_default_tools_are_no_model()
    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    resolved_path = Path(resolve_workspace_path(workspace, path))
    try:
        stat = resolved_path.stat()
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"Path is not accessible: {path}") from exc
    file_type = "directory" if resolved_path.is_dir() else "file"
    if not (resolved_path.is_dir() or resolved_path.is_file()):
        file_type = "other"
    return {
        "workspace_id": workspace.workspace_id,
        "path": posixpath.normpath(path.strip() or "."),
        "type": file_type,
        "size_bytes": stat.st_size if resolved_path.is_file() else None,
        "within_file_read_size_cap": bool(resolved_path.is_file() and stat.st_size <= MAX_FILE_READ_CHARS),
        "audit": {
            "tool": "workspace_file_stat",
            "llm_calls": 0,
            "root_exposed": False,
        },
    }


def list_workspace_files(
    workspace_id: str,
    *,
    path: str | None = None,
    file_glob: str | None = None,
    limit: int | None = MAX_FILE_LIST_RESULTS,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """List bounded, sanitized workspace files after context hydration."""

    assert_default_tools_are_no_model()
    max_results = _bounded_int(
        limit,
        "limit",
        default=MAX_FILE_LIST_RESULTS,
        minimum=1,
        maximum=MAX_FILE_LIST_RESULTS,
    )
    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    list_path = path if path is not None else "."
    root = Path(resolve_workspace_path(workspace, list_path))
    normalized_glob = file_glob.strip() if isinstance(file_glob, str) and file_glob.strip() else None
    entries: list[dict] = []
    skipped: list[dict] = []
    truncated = False
    stop_walk = False

    def _add_skipped(path: str, reason: str) -> None:
        nonlocal truncated
        if len(skipped) < max_results:
            skipped.append({"path": path, "reason": reason})
        else:
            truncated = True

    if root.is_file():
        rel_path = _workspace_relative_path(workspace, root)
        if not normalized_glob or fnmatch.fnmatch(rel_path, normalized_glob) or fnmatch.fnmatch(root.name, normalized_glob):
            entries.append(_list_workspace_file_entry(workspace, root))
        return {
            "workspace_id": workspace.workspace_id,
            "path": posixpath.normpath((list_path or ".").strip() or "."),
            "file_glob": file_glob,
            "entries": entries,
            "skipped": skipped,
            "limit": max_results,
            "truncated": False,
        }
    if not root.is_dir():
        raise ProfileRouterError("not_a_directory", "workspace_file_list path must be a file or directory")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        rel_dir = _workspace_relative_path(workspace, current)
        safe_dirnames = []
        for dirname in dirnames:
            child_rel = posixpath.normpath(posixpath.join(rel_dir, dirname))
            try:
                resolve_workspace_path(workspace, child_rel)
            except ProfileRouterError as exc:
                _add_skipped(child_rel, exc.code)
                continue
            safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames

        for name in [*safe_dirnames, *filenames]:
            candidate = current / name
            rel_path = _workspace_relative_path(workspace, candidate)
            if normalized_glob and not (
                fnmatch.fnmatch(rel_path, normalized_glob)
                or fnmatch.fnmatch(name, normalized_glob)
            ):
                continue
            try:
                resolve_workspace_path(workspace, rel_path)
            except ProfileRouterError as exc:
                _add_skipped(rel_path, exc.code)
                continue
            if len(entries) >= max_results:
                truncated = True
                _add_skipped(rel_path, "file_limit_exceeded")
                dirnames[:] = []
                stop_walk = True
                break
            entries.append(_list_workspace_file_entry(workspace, candidate))
        if stop_walk:
            break

    return {
        "workspace_id": workspace.workspace_id,
        "path": posixpath.normpath((list_path or ".").strip() or "."),
        "file_glob": file_glob,
        "entries": entries,
        "skipped": skipped,
        "limit": max_results,
        "truncated": truncated,
    }


def _workspace_relative_path(workspace: WorkspaceMetadata, path: Path) -> str:
    try:
        return path.relative_to(Path(workspace.root)).as_posix()
    except ValueError as exc:
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root") from exc


def _iter_search_candidates(
    workspace: WorkspaceMetadata,
    search_root: Path,
    *,
    file_glob: str | None,
):
    if search_root.is_file():
        rel_path = _workspace_relative_path(workspace, search_root)
        if not file_glob or fnmatch.fnmatch(rel_path, file_glob) or fnmatch.fnmatch(search_root.name, file_glob):
            yield rel_path, search_root
        return

    if not search_root.is_dir():
        raise ProfileRouterError("not_a_directory", "Search path must be a file or directory")

    for dirpath, dirnames, filenames in os.walk(search_root, followlinks=False):
        rel_dir = _workspace_relative_path(workspace, Path(dirpath))
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _is_secret_path(posixpath.join(rel_dir, dirname))
        ]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            rel_path = _workspace_relative_path(workspace, candidate)
            if file_glob and not (
                fnmatch.fnmatch(rel_path, file_glob)
                or fnmatch.fnmatch(filename, file_glob)
            ):
                continue
            yield rel_path, candidate


def search_workspace_files(
    workspace_id: str,
    pattern: str,
    *,
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "content",
    limit: int | None = MAX_FILE_SEARCH_RESULTS,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Search bounded text files inside an opened workspace."""

    assert_default_tools_are_no_model()
    if not isinstance(pattern, str) or not pattern:
        raise ProfileRouterError("invalid_search_pattern", "pattern must be a non-empty string")
    if file_glob is not None and not isinstance(file_glob, str):
        raise ProfileRouterError("invalid_file_glob", "file_glob must be a string")
    if output_mode not in ALLOWED_SEARCH_OUTPUT_MODES:
        raise ProfileRouterError(
            "invalid_output_mode",
            "output_mode must be one of: " + ", ".join(sorted(ALLOWED_SEARCH_OUTPUT_MODES)),
        )
    max_results = _bounded_int(
        limit,
        "limit",
        default=MAX_FILE_SEARCH_RESULTS,
        minimum=1,
        maximum=MAX_FILE_SEARCH_RESULTS,
    )
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ProfileRouterError("invalid_search_pattern", str(exc)) from exc

    workspace = (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)
    search_path = path if path is not None else "."
    search_root = Path(resolve_workspace_path(workspace, search_path))

    matches: list[dict] = []
    files: list[str] = []
    counts: list[dict] = []
    skipped = {"secret": 0, "binary": 0, "large": 0, "unreadable": 0}
    truncated = False
    normalized_glob = file_glob.strip() if isinstance(file_glob, str) and file_glob.strip() else None

    for rel_path, candidate in _iter_search_candidates(
        workspace,
        search_root,
        file_glob=normalized_glob,
    ):
        try:
            resolved_candidate = Path(resolve_workspace_path(workspace, rel_path))
            _ensure_text_file(resolved_candidate)
        except ProfileRouterError as exc:
            if exc.code == "secret_path_denied":
                skipped["secret"] += 1
                continue
            if exc.code == "binary_file_not_supported":
                skipped["binary"] += 1
                continue
            skipped["unreadable"] += 1
            continue

        try:
            if resolved_candidate.stat().st_size > MAX_FILE_SEARCH_BYTES:
                skipped["large"] += 1
                continue
        except OSError:
            skipped["unreadable"] += 1
            continue

        file_match_count = 0
        try:
            with resolved_candidate.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not compiled.search(line):
                        continue
                    file_match_count += 1
                    if output_mode == "content":
                        matches.append(
                            {
                                "path": rel_path,
                                "line": line_number,
                                "content": _redact_sensitive_text_fields(
                                    line.rstrip("\n")[:MAX_SEARCH_LINE_CHARS]
                                ),
                            }
                        )
                        if len(matches) >= max_results:
                            truncated = True
                            break
                if output_mode == "content" and truncated:
                    break
        except OSError:
            skipped["unreadable"] += 1
            continue

        if file_match_count:
            if output_mode == "files_only" and rel_path not in files:
                files.append(rel_path)
                if len(files) >= max_results:
                    truncated = True
                    break
            elif output_mode == "count":
                counts.append({"path": rel_path, "count": file_match_count})
                if len(counts) >= max_results:
                    truncated = True
                    break

    payload = {
        "workspace_id": workspace.workspace_id,
        "path": posixpath.normpath(search_path.strip() or "."),
        "pattern": pattern,
        "file_glob": file_glob,
        "output_mode": output_mode,
        "limit": max_results,
        "truncated": truncated,
        "skipped": skipped,
    }
    if output_mode == "content":
        payload["matches"] = matches
    elif output_mode == "files_only":
        payload["files"] = files
    else:
        payload["counts"] = counts
    return payload


def _require_workspace_write_access(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> WorkspaceMetadata:
    """Require fresh context and explicit filesystem.write policy."""

    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _ref, route_policy, _router_policy = _require_local_profile_policy(workspace.profile_ref)
    if not route_policy.allow_filesystem_write:
        raise ProfileRouterError(
            "filesystem_write_not_allowed",
            f"Filesystem write is disabled by profile_router policy: {workspace.profile_ref}",
        )
    return workspace


def _require_workspace_terminal_access(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> tuple[WorkspaceMetadata, ProfileRoutePolicy]:
    """Require fresh context and explicit terminal policy before preflight."""

    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _ref, route_policy, _router_policy = _require_local_profile_policy(workspace.profile_ref)
    if not route_policy.allow_terminal:
        raise ProfileRouterError(
            "terminal_not_allowed",
            f"Terminal access is disabled by profile_router policy: {workspace.profile_ref}",
        )
    return workspace, route_policy


def _require_workspace_git_read_access(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> tuple[WorkspaceMetadata, ProfileRoutePolicy]:
    """Require fresh context plus explicit read-only Git policy."""

    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _ref, route_policy, _router_policy = _require_local_profile_policy(workspace.profile_ref)
    if not route_policy.capability_enabled("git"):
        raise ProfileRouterError(
            "git_read_not_allowed",
            f"Git read access is disabled by profile_router policy: {workspace.profile_ref}",
        )
    _require_git_workspace(workspace)
    return workspace, route_policy


def _require_workspace_cron_access(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> tuple[WorkspaceMetadata, ProfileRoutePolicy]:
    """Require fresh context plus explicit script-only cron policy."""

    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _ref, route_policy, _router_policy = _require_local_profile_policy(workspace.profile_ref)
    if not route_policy.capability_enabled("cron") or not route_policy.cron_policy.enabled:
        raise ProfileRouterError(
            "cron_not_allowed",
            f"Cron access is disabled by profile_router policy: {workspace.profile_ref}",
        )
    return workspace, route_policy


def _require_workspace_messaging_access(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> tuple[WorkspaceMetadata, ProfileRoutePolicy]:
    """Require fresh context plus explicit messaging policy and recipient allowlist."""

    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _ref, route_policy, _router_policy = _require_local_profile_policy(workspace.profile_ref)
    if not route_policy.capability_enabled("messaging") or not route_policy.allow_messaging:
        raise ProfileRouterError(
            "messaging_not_allowed",
            f"Messaging access is disabled by profile_router policy: {workspace.profile_ref}",
        )
    if not route_policy.messaging_allowed_recipients:
        raise ProfileRouterError(
            "messaging_no_allowlist",
            "Messaging requires at least one explicit allowed_recipients entry",
        )
    return workspace, route_policy


def _cron_jobs_backend():
    """Return Hermes cron storage helpers lazily so tests can monkeypatch safely."""

    from cron import jobs as cron_jobs

    return cron_jobs


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    text = _redact_sensitive_text_fields(text)
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _normalize_cron_job_ref(job_ref: str) -> str:
    if not isinstance(job_ref, str) or not job_ref.strip():
        raise ProfileRouterError("invalid_cron_job_ref", "cron job reference is required")
    ref = job_ref.strip()
    if not CRON_JOB_REF_RE.fullmatch(ref):
        raise ProfileRouterError("invalid_cron_job_ref", "cron job reference must be an opaque id or safe name")
    return ref


def _cron_job_script(job: Mapping[str, Any]) -> str | None:
    script = job.get("script")
    if not isinstance(script, str) or not script.strip():
        return None
    try:
        return _normalize_cron_script_path(script)
    except ProfileRouterError:
        return None


def _cron_job_is_script_only_no_agent(job: Mapping[str, Any]) -> bool:
    return bool(job.get("no_agent") is True and _cron_job_script(job))


def _cron_script_allowed(route_policy: ProfileRoutePolicy, script: str) -> bool:
    try:
        normalized = _normalize_cron_script_path(script)
    except ProfileRouterError:
        return False
    return normalized in route_policy.cron_policy.allowed_scripts


def _require_cron_script_allowed(route_policy: ProfileRoutePolicy, script: str) -> str:
    normalized = _normalize_cron_script_path(script)
    if not route_policy.cron_policy.allowed_scripts:
        raise ProfileRouterError(
            "cron_script_allowlist_required",
            "cron.allowed_scripts must explicitly allow script-only no_agent jobs before scheduling or running them",
        )
    if normalized not in route_policy.cron_policy.allowed_scripts:
        raise ProfileRouterError(
            "cron_script_not_allowed",
            "cron script is not allowed by profile_router cron.allowed_scripts policy",
        )
    return normalized


def _resolve_cron_job(job_ref: str, *, backend: Any | None = None) -> Mapping[str, Any]:
    selected_backend = backend or _cron_jobs_backend()
    ref = _normalize_cron_job_ref(job_ref)
    try:
        if hasattr(selected_backend, "resolve_job_ref"):
            job = selected_backend.resolve_job_ref(ref)
        else:
            job = selected_backend.get_job(ref)
    except LookupError as exc:
        raise ProfileRouterError("cron_job_ambiguous", "cron job reference is ambiguous; use the job id") from exc
    except Exception as exc:  # pragma: no cover - defensive against storage failures
        raise ProfileRouterError("cron_storage_unavailable", "cron storage could not be queried safely") from exc
    if not job:
        raise ProfileRouterError("cron_job_not_found", "cron job not found")
    if not isinstance(job, MappingABC):
        raise ProfileRouterError("invalid_cron_job", "cron storage returned an invalid job record")
    return job


def _require_script_only_cron_job(
    job: Mapping[str, Any],
    route_policy: ProfileRoutePolicy,
    *,
    require_script_allowlist: bool = True,
) -> str:
    if not _cron_job_is_script_only_no_agent(job):
        raise ProfileRouterError(
            "cron_model_backed_job_denied",
            "model-backed cron jobs are denied by the ChatGPT no-model connector policy",
        )
    script = _cron_job_script(job)
    if script is None:
        raise ProfileRouterError("invalid_cron_script", "cron job script is invalid")
    if require_script_allowlist:
        return _require_cron_script_allowed(route_policy, script)
    return script


def _sanitize_cron_job(job: Mapping[str, Any], route_policy: ProfileRoutePolicy) -> dict:
    script = _cron_job_script(job)
    script_only = _cron_job_is_script_only_no_agent(job)
    script_allowed = _cron_script_allowed(route_policy, script) if script else False
    return {
        "job_id": _bounded_text(job.get("id"), 80),
        "name": _bounded_text(job.get("name"), MAX_CRON_JOB_NAME_CHARS),
        "schedule_display": _bounded_text(job.get("schedule_display"), MAX_CRON_SCHEDULE_CHARS),
        "enabled": bool(job.get("enabled", True)),
        "state": _bounded_text(job.get("state"), 40),
        "next_run_at": _bounded_text(job.get("next_run_at"), 80),
        "last_run_at": _bounded_text(job.get("last_run_at"), 80),
        "last_status": _bounded_text(job.get("last_status"), 40),
        "no_agent": bool(job.get("no_agent") is True),
        "script_only": script_only,
        "model_backed": not script_only,
        "script": {
            "present": script is not None,
            "allowed_by_profile_policy": script_allowed,
            "path_exposed": False,
        },
    }


def _cron_audit(tool_name: str, *, mutates_state: bool, action: str) -> dict:
    return {
        "tool": tool_name,
        "action": action,
        "llm_calls": 0,
        "mutates_state": mutates_state,
        "root_exposed": False,
        "no_agent_only": True,
        "model_backed_crons_allowed": False,
    }


def list_workspace_cron_jobs(
    workspace_id: str,
    *,
    context_token: str | None = None,
    include_disabled: bool = False,
    limit: int | None = MAX_CRON_LIST_RESULTS,
    backend: Any | None = None,
) -> dict:
    """List sanitized cron jobs after context/cron policy without model calls."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_cron_access(
        workspace_id,
        context_token=context_token,
    )
    max_results = _bounded_int(
        limit,
        "limit",
        default=MAX_CRON_LIST_RESULTS,
        minimum=1,
        maximum=MAX_CRON_LIST_RESULTS,
    )
    selected_backend = backend or _cron_jobs_backend()
    try:
        jobs = selected_backend.list_jobs(include_disabled=bool(include_disabled))
    except Exception as exc:  # pragma: no cover - defensive against storage failures
        raise ProfileRouterError("cron_storage_unavailable", "cron storage could not be listed safely") from exc
    sanitized = [_sanitize_cron_job(job, route_policy) for job in list(jobs)[:max_results]]
    total_count = len(jobs) if isinstance(jobs, list) else len(sanitized)
    return {
        "workspace_id": workspace.workspace_id,
        "jobs": sanitized,
        "job_count": len(sanitized),
        "total_count": total_count,
        "limit": max_results,
        "truncated": total_count > len(sanitized),
        "policy": {
            "cron_enabled": route_policy.cron_policy.enabled,
            "script_allowlist_count": len(route_policy.cron_policy.allowed_scripts),
            "model_backed_crons_allowed": False,
        },
        "audit": _cron_audit("cron_list", mutates_state=False, action="list"),
    }


def pause_workspace_cron_job(
    workspace_id: str,
    job_ref: str,
    *,
    context_token: str | None = None,
    reason: str | None = None,
    backend: Any | None = None,
) -> dict:
    """Pause a script-only no_agent cron job after context/cron policy."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_cron_access(workspace_id, context_token=context_token)
    selected_backend = backend or _cron_jobs_backend()
    job = _resolve_cron_job(job_ref, backend=selected_backend)
    _require_script_only_cron_job(job, route_policy, require_script_allowlist=False)
    pause_reason = _bounded_text(reason, 120) if reason else "paused by no-model profile-router cron tool"
    try:
        updated = selected_backend.pause_job(str(job.get("id") or job_ref), reason=pause_reason)
    except Exception as exc:  # pragma: no cover - defensive against storage failures
        raise ProfileRouterError("cron_storage_unavailable", "cron job could not be paused safely") from exc
    if not updated:
        raise ProfileRouterError("cron_job_not_found", "cron job not found")
    return {
        "workspace_id": workspace.workspace_id,
        "job": _sanitize_cron_job(updated, route_policy),
        "audit": _cron_audit("cron_pause", mutates_state=True, action="pause"),
    }


def resume_workspace_cron_job(
    workspace_id: str,
    job_ref: str,
    *,
    context_token: str | None = None,
    backend: Any | None = None,
) -> dict:
    """Resume an allowlisted script-only no_agent cron job without model calls."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_cron_access(workspace_id, context_token=context_token)
    selected_backend = backend or _cron_jobs_backend()
    job = _resolve_cron_job(job_ref, backend=selected_backend)
    _require_script_only_cron_job(job, route_policy, require_script_allowlist=True)
    try:
        updated = selected_backend.resume_job(str(job.get("id") or job_ref))
    except Exception as exc:  # pragma: no cover - defensive against storage failures
        raise ProfileRouterError("cron_storage_unavailable", "cron job could not be resumed safely") from exc
    if not updated:
        raise ProfileRouterError("cron_job_not_found", "cron job not found")
    return {
        "workspace_id": workspace.workspace_id,
        "job": _sanitize_cron_job(updated, route_policy),
        "audit": _cron_audit("cron_resume", mutates_state=True, action="resume"),
    }


def trigger_workspace_cron_job(
    workspace_id: str,
    job_ref: str,
    *,
    context_token: str | None = None,
    backend: Any | None = None,
) -> dict:
    """Schedule an allowlisted script-only no_agent cron job for the next tick."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_cron_access(workspace_id, context_token=context_token)
    selected_backend = backend or _cron_jobs_backend()
    job = _resolve_cron_job(job_ref, backend=selected_backend)
    _require_script_only_cron_job(job, route_policy, require_script_allowlist=True)
    try:
        updated = selected_backend.trigger_job(str(job.get("id") or job_ref))
    except Exception as exc:  # pragma: no cover - defensive against storage failures
        raise ProfileRouterError("cron_storage_unavailable", "cron job could not be triggered safely") from exc
    if not updated:
        raise ProfileRouterError("cron_job_not_found", "cron job not found")
    return {
        "workspace_id": workspace.workspace_id,
        "job": _sanitize_cron_job(updated, route_policy),
        "audit": _cron_audit("cron_run", mutates_state=True, action="trigger_next_tick"),
    }


def create_workspace_cron_script_job(
    workspace_id: str,
    schedule: str,
    script: str,
    *,
    context_token: str | None = None,
    name: str | None = None,
    repeat: int | None = None,
    backend: Any | None = None,
) -> dict:
    """Create only allowlisted script-only no_agent cron jobs; never agent/model jobs."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_cron_access(workspace_id, context_token=context_token)
    normalized_script = _require_cron_script_allowed(route_policy, script)
    if not isinstance(schedule, str) or not schedule.strip():
        raise ProfileRouterError("invalid_cron_schedule", "schedule is required")
    schedule_text = schedule.strip()
    if len(schedule_text) > MAX_CRON_SCHEDULE_CHARS:
        raise ProfileRouterError(
            "invalid_cron_schedule",
            f"schedule must be <= {MAX_CRON_SCHEDULE_CHARS} characters",
        )
    if repeat is not None and (isinstance(repeat, bool) or not isinstance(repeat, int)):
        raise ProfileRouterError("invalid_cron_repeat", "repeat must be an integer")
    if repeat is not None and repeat > 1000:
        raise ProfileRouterError("invalid_cron_repeat", "repeat must be <= 1000")
    job_name = _bounded_text(name, MAX_CRON_JOB_NAME_CHARS) if name else None
    selected_backend = backend or _cron_jobs_backend()
    try:
        # Validate schedule before any storage write when the backend exposes the parser.
        if hasattr(selected_backend, "parse_schedule"):
            selected_backend.parse_schedule(schedule_text)
        job = selected_backend.create_job(
            "",
            schedule_text,
            name=job_name,
            repeat=repeat,
            deliver="local",
            origin=None,
            skill=None,
            skills=None,
            model=None,
            provider=None,
            base_url=None,
            script=normalized_script,
            context_from=None,
            enabled_toolsets=None,
            workdir=workspace.root,
            no_agent=True,
        )
    except ValueError as exc:
        raise ProfileRouterError("invalid_cron_schedule", "cron schedule or job shape is invalid") from exc
    except Exception as exc:  # pragma: no cover - defensive against storage failures
        raise ProfileRouterError("cron_storage_unavailable", "cron job could not be created safely") from exc
    return {
        "workspace_id": workspace.workspace_id,
        "job": _sanitize_cron_job(job, route_policy),
        "audit": _cron_audit("cron_create_script_only", mutates_state=True, action="create_script_only"),
    }


def _messaging_destination_fingerprint(destination: str) -> str:
    return hashlib.sha256(destination.encode("utf-8")).hexdigest()[:16]


def _normalize_message_text(message: str) -> str:
    if not isinstance(message, str):
        raise ProfileRouterError("invalid_message", "message must be a string")
    text = message.strip()
    if not text:
        raise ProfileRouterError("invalid_message", "message is required")
    if "\x00" in text:
        raise ProfileRouterError("invalid_message", "message must not contain NUL bytes")
    if len(text) > MAX_MESSAGING_MESSAGE_CHARS:
        raise ProfileRouterError(
            "message_too_large",
            f"message must be <= {MAX_MESSAGING_MESSAGE_CHARS} characters",
        )
    if _redact_sensitive_text_fields(text) != text:
        raise ProfileRouterError(
            "message_content_secret_denied",
            "message content appears to contain a secret/token and is blocked",
        )
    return text


def prepare_workspace_message_send(
    workspace_id: str,
    destination: str,
    message: str,
    *,
    context_token: str | None = None,
    dry_run: bool = True,
    required_platform: str | None = None,
    tool_name: str = "message_send",
) -> dict:
    """Validate an allowlisted messaging send without invoking gateway adapters."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_messaging_access(
        workspace_id,
        context_token=context_token,
    )
    if not isinstance(dry_run, bool):
        raise ProfileRouterError("invalid_messaging_option", "dry_run must be a boolean")
    platform, _recipient, normalized_destination = _normalize_messaging_destination(destination)
    if required_platform and platform != required_platform:
        raise ProfileRouterError(
            "messaging_platform_mismatch",
            f"{tool_name} only accepts {required_platform} destinations",
        )
    if normalized_destination not in route_policy.messaging_allowed_recipients:
        raise ProfileRouterError(
            "messaging_destination_not_allowed",
            "destination is not in profile_router.messaging.allowed_recipients",
        )
    message_text = _normalize_message_text(message)
    if not dry_run:
        raise ProfileRouterError(
            "message_delivery_not_enabled",
            "external message delivery remains disabled pending explicit runtime policy/review",
        )
    return {
        "workspace_id": workspace.workspace_id,
        "messaging": {
            "status": "dry_run_ready",
            "platform": platform,
            "destination_hash": _messaging_destination_fingerprint(normalized_destination),
            "destination_allowed": True,
            "destination_exposed": False,
            "message_chars": len(message_text),
            "message_content_logged": False,
            "dry_run": True,
            "delivery_attempted": False,
            "external_delivery_enabled": False,
            "policy": {
                "messaging_enabled": route_policy.allow_messaging,
                "allowed_recipients_count": len(route_policy.messaging_allowed_recipients),
                "allowlist_redacted": True,
                "broadcast_allowed": False,
            },
            "audit": {
                "tool": tool_name,
                "llm_calls": 0,
                "root_exposed": False,
                "uses_gateway_adapter": False,
                "destination_exposed": False,
                "message_content_logged": False,
                "public_mcp_exposure": "disabled_pending_phase_10_messaging_policy_review",
            },
        },
    }


def _bounded_terminal_int(
    value: int | None,
    field: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileRouterError("invalid_terminal_option", f"{field} must be an integer")
    if value < minimum:
        raise ProfileRouterError(
            "invalid_terminal_option", f"{field} must be >= {minimum}"
        )
    return min(value, maximum)


def _workspace_public_relative_path(workspace: WorkspaceMetadata, resolved_path: Path) -> str:
    try:
        relative = resolved_path.relative_to(Path(workspace.root))
    except ValueError as exc:
        raise ProfileRouterError("path_outside_workspace", "path escapes workspace root") from exc
    public_path = relative.as_posix()
    return public_path if public_path else "."


def _resolve_terminal_working_directory(
    workspace: WorkspaceMetadata,
    working_directory: str,
) -> tuple[Path, str]:
    resolved = Path(resolve_workspace_path(workspace, working_directory, require_exists=True))
    if not resolved.is_dir():
        raise ProfileRouterError(
            "working_directory_not_directory",
            "terminal working_directory must be an existing workspace-relative directory",
        )
    return resolved, _workspace_public_relative_path(workspace, resolved)


def _read_patchable_text(path: Path, public_path: str) -> str:
    _ensure_text_file(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ProfileRouterError(
            "text_encoding_not_supported",
            f"File must be valid UTF-8 for write/patch operations: {public_path}",
        ) from exc
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"File is not readable: {public_path}") from exc


def _validate_write_content(content: str) -> None:
    if not isinstance(content, str):
        raise ProfileRouterError("invalid_content", "content must be a string")
    if len(content) > MAX_FILE_WRITE_CHARS:
        raise ProfileRouterError(
            "content_too_large",
            f"content exceeds {MAX_FILE_WRITE_CHARS} characters",
        )


def _bounded_unified_diff(before: str, after: str, rel_path: str) -> dict:
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )
    diff = _redact_sensitive_text_fields(diff)
    truncated = len(diff) > MAX_WRITE_DIFF_CHARS
    if truncated:
        diff = diff[:MAX_WRITE_DIFF_CHARS] + "\n... [diff truncated]\n"
    return {"unified": diff, "truncated": truncated, "max_chars": MAX_WRITE_DIFF_CHARS}


def _write_operation_payload(
    workspace: WorkspaceMetadata,
    *,
    tool_name: str,
    rel_path: str,
    before: str,
    after: str,
    replacements: int | None = None,
) -> dict:
    payload = {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "path": posixpath.normpath(rel_path.strip()),
        "bytes_written": len(after.encode("utf-8")),
        "changed": before != after,
        "diff": _bounded_unified_diff(before, after, posixpath.normpath(rel_path.strip())),
        "audit": {
            "tool": tool_name,
            "llm_calls": 0,
            "root_exposed": False,
        },
    }
    if replacements is not None:
        payload["replacements"] = replacements
    return payload


def _normalize_patch_apply_operations(patches: Any) -> list[dict[str, Any]]:
    if not isinstance(patches, list):
        raise ProfileRouterError("invalid_patch", "patches must be a list of patch operations")
    if not patches:
        raise ProfileRouterError("invalid_patch", "patches must include at least one operation")
    if len(patches) > MAX_PATCH_APPLY_OPERATIONS:
        raise ProfileRouterError(
            "patch_batch_too_large",
            f"patches exceeds {MAX_PATCH_APPLY_OPERATIONS} operations",
        )

    normalized: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(patches, start=1):
        if not isinstance(raw_operation, MappingABC):
            raise ProfileRouterError(
                "invalid_patch",
                f"patch operation {index} must be an object",
            )
        path = raw_operation.get("path")
        old_string = raw_operation.get("old_string")
        new_string = raw_operation.get("new_string")
        replace_all = raw_operation.get("replace_all", False)
        if not isinstance(path, str) or not path.strip():
            raise ProfileRouterError(
                "invalid_patch",
                f"patch operation {index} path must be a non-empty string",
            )
        if not isinstance(old_string, str) or old_string == "":
            raise ProfileRouterError(
                "invalid_patch",
                f"patch operation {index} old_string must be a non-empty string",
            )
        if not isinstance(new_string, str):
            raise ProfileRouterError(
                "invalid_patch",
                f"patch operation {index} new_string must be a string",
            )
        if not isinstance(replace_all, bool):
            raise ProfileRouterError(
                "invalid_patch",
                f"patch operation {index} replace_all must be a boolean",
            )
        _validate_write_content(new_string)
        normalized.append(
            {
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
                "replace_all": replace_all,
            }
        )
    return normalized


def apply_workspace_patch(
    workspace_id: str,
    patches: list[Mapping[str, Any]],
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Apply a bounded multi-file literal patch batch after write-policy gates."""

    assert_default_tools_are_no_model()
    operations = _normalize_patch_apply_operations(patches)
    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )

    prepared: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    total_after_chars = 0
    for operation in operations:
        resolved_path = resolve_workspace_write_path(workspace, operation["path"])
        relative_path = _workspace_public_relative_path(workspace, resolved_path)
        if relative_path in seen_paths:
            raise ProfileRouterError(
                "patch_duplicate_path",
                "patch batch may target each file at most once",
            )
        seen_paths.add(relative_path)
        before = _read_patchable_text(resolved_path, relative_path)
        old_string = operation["old_string"]
        new_string = operation["new_string"]
        replace_all = operation["replace_all"]
        match_count = before.count(old_string)
        if match_count == 0:
            raise ProfileRouterError(
                "patch_match_not_found",
                f"old_string was not found: {relative_path}",
            )
        if match_count > 1 and not replace_all:
            raise ProfileRouterError(
                "patch_match_not_unique",
                f"old_string matched more than once in {relative_path}; set replace_all=true to replace all matches",
            )
        replacements = match_count if replace_all else 1
        after = before.replace(old_string, new_string, replacements)
        _validate_write_content(after)
        total_after_chars += len(after)
        if total_after_chars > MAX_PATCH_APPLY_TOTAL_WRITE_CHARS:
            raise ProfileRouterError(
                "patch_batch_too_large",
                f"patch batch exceeds {MAX_PATCH_APPLY_TOTAL_WRITE_CHARS} characters after replacement",
            )
        prepared.append(
            {
                "resolved_path": resolved_path,
                "relative_path": relative_path,
                "before": before,
                "after": after,
                "replacements": replacements,
            }
        )

    written: list[dict[str, Any]] = []
    try:
        for item in prepared:
            item["resolved_path"].write_text(item["after"], encoding="utf-8")
            written.append(item)
    except OSError as exc:
        for item in reversed(written):
            try:
                item["resolved_path"].write_text(item["before"], encoding="utf-8")
            except OSError:
                pass
        raise ProfileRouterError("file_not_writable", "Patch batch could not be written") from exc

    files = [
        {
            "path": item["relative_path"],
            "bytes_written": len(item["after"].encode("utf-8")),
            "changed": item["before"] != item["after"],
            "replacements": item["replacements"],
            "diff": _bounded_unified_diff(
                item["before"],
                item["after"],
                item["relative_path"],
            ),
        }
        for item in prepared
    ]
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "patch_count": len(prepared),
        "file_count": len(files),
        "total_replacements": sum(item["replacements"] for item in prepared),
        "bytes_written": sum(file["bytes_written"] for file in files),
        "changed": any(file["changed"] for file in files),
        "files": files,
        "audit": {
            "tool": "patch_apply",
            "llm_calls": 0,
            "root_exposed": False,
        },
    }


def patch_workspace_file(
    workspace_id: str,
    path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Apply a bounded literal text patch after context and write-policy gates."""

    assert_default_tools_are_no_model()
    if not isinstance(old_string, str) or old_string == "":
        raise ProfileRouterError("invalid_patch", "old_string must be a non-empty string")
    if not isinstance(new_string, str):
        raise ProfileRouterError("invalid_patch", "new_string must be a string")
    _validate_write_content(new_string)

    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    resolved_path = resolve_workspace_write_path(workspace, path)
    before = _read_patchable_text(resolved_path, path)
    match_count = before.count(old_string)
    if match_count == 0:
        raise ProfileRouterError("patch_match_not_found", "old_string was not found")
    if match_count > 1 and not replace_all:
        raise ProfileRouterError(
            "patch_match_not_unique",
            "old_string matched more than once; set replace_all=true to replace all matches",
        )
    replacements = match_count if replace_all else 1
    after = before.replace(old_string, new_string, replacements)
    _validate_write_content(after)
    try:
        resolved_path.write_text(after, encoding="utf-8")
    except OSError as exc:
        raise ProfileRouterError("file_not_writable", f"File is not writable: {path}") from exc
    return _write_operation_payload(
        workspace,
        tool_name="file_patch",
        rel_path=path,
        before=before,
        after=after,
        replacements=replacements,
    )


def write_workspace_file(
    workspace_id: str,
    path: str,
    content: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Write a UTF-8 text file after context and write-policy gates."""

    assert_default_tools_are_no_model()
    _validate_write_content(content)
    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    resolved_path = resolve_workspace_write_path(workspace, path)
    before = ""
    if resolved_path.exists():
        before = _read_patchable_text(resolved_path, path)
    try:
        resolved_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ProfileRouterError("file_not_writable", f"File is not writable: {path}") from exc
    return _write_operation_payload(
        workspace,
        tool_name="file_write",
        rel_path=path,
        before=before,
        after=content,
    )


def move_workspace_file(
    workspace_id: str,
    source_path: str,
    destination_path: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Move/rename one regular workspace file after context and write-policy gates."""

    assert_default_tools_are_no_model()
    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    source, source_relative = resolve_workspace_move_source_path(workspace, source_path)
    destination, destination_relative = resolve_workspace_move_destination_path(
        workspace,
        destination_path,
    )
    if source == destination:
        raise ProfileRouterError("source_destination_same", "move source and destination must differ")
    try:
        bytes_moved = source.stat().st_size
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", "move source is not readable") from exc
    try:
        source.rename(destination)
    except OSError as exc:
        raise ProfileRouterError("file_not_writable", "file could not be moved") from exc
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "source_path": source_relative,
        "destination_path": destination_relative,
        "type": "file",
        "bytes_moved": bytes_moved,
        "moved": True,
        "audit": {
            "tool": "file_move",
            "llm_calls": 0,
            "root_exposed": False,
        },
    }


def delete_workspace_file(
    workspace_id: str,
    path: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Delete one regular workspace file after context and write-policy gates."""

    assert_default_tools_are_no_model()
    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    target, relative_path = resolve_workspace_delete_path(workspace, path)
    try:
        bytes_deleted = target.stat().st_size
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", "delete target is not readable") from exc
    try:
        target.unlink()
    except OSError as exc:
        raise ProfileRouterError("file_not_writable", "file could not be deleted") from exc
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "path": relative_path,
        "type": "file",
        "bytes_deleted": bytes_deleted,
        "deleted": True,
        "audit": {
            "tool": "file_delete",
            "llm_calls": 0,
            "root_exposed": False,
        },
    }


def probe_workspace_status(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Run a fixed status probe without exposing arbitrary command execution."""

    assert_default_tools_are_no_model()
    preflight = preflight_terminal_command(
        workspace_id,
        "pwd",
        timeout=5,
        working_directory=".",
        max_output_chars=MAX_STATUS_PROBE_OUTPUT_CHARS,
        context_token=context_token,
        registry=registry,
    )
    if preflight["blocked"] or not preflight["execution_plan"]["available"]:
        raise ProfileRouterError(
            "workspace_status_probe_not_allowed",
            "workspace status probe is not allowlisted by profile-router terminal policy",
        )
    terminal_result = _run_preflighted_terminal_command(
        workspace_id,
        "pwd",
        preflight=preflight,
        context_token=context_token,
        registry=registry,
    )
    if terminal_result["status"] != "success":
        raise ProfileRouterError(
            "workspace_status_probe_failed",
            "workspace status probe completed without success",
        )
    workspace = (registry or DEFAULT_WORKSPACE_REGISTRY).get(workspace_id)
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "status": "ok",
        "probe": {
            "kind": "workspace_status",
            "fixed_server_side_action": True,
            "command_exposed": False,
            "cwd_workspace_relative": terminal_result["working_directory"],
            "stdout_workspace_marker_seen": "<workspace>" in terminal_result["stdout"]["text"],
            "returncode": terminal_result["returncode"],
            "output_truncated": terminal_result["output"]["truncated"],
            "root_exposed": False,
        },
        "audit": {
            "tool": "workspace_status_probe",
            "llm_calls": 0,
            "root_exposed": False,
            "uses_shell": False,
            "executes": True,
            "arbitrary_command_accepted": False,
        },
    }


def run_workspace_scratch_smoke(
    workspace_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Run a fixed scratch write/read/patch/read/delete smoke safely."""

    assert_default_tools_are_no_model()
    selected_registry = registry or DEFAULT_WORKSPACE_REGISTRY
    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=selected_registry,
    )
    scratch_dir = CHATGPT_SCRATCH_SMOKE_DIR
    create_workspace_directory(
        workspace_id,
        scratch_dir,
        parents=True,
        exist_ok=True,
        context_token=context_token,
        registry=selected_registry,
    )
    for _attempt in range(10):
        scratch_path = posixpath.join(
            scratch_dir,
            f"{CHATGPT_SCRATCH_SMOKE_BASENAME_PREFIX}{uuid4().hex}.txt",
        )
        if not resolve_workspace_write_path(workspace, scratch_path).exists():
            break
    else:
        raise ProfileRouterError(
            "scratch_smoke_path_unavailable",
            "could not allocate a unique scratch smoke path",
        )

    write_result: dict | None = None
    patched_result: dict | None = None
    read_initial: dict | None = None
    read_patched: dict | None = None
    cleanup_result: dict | None = None
    created = False
    try:
        write_result = write_workspace_file(
            workspace_id,
            scratch_path,
            CHATGPT_SCRATCH_SMOKE_INITIAL,
            context_token=context_token,
            registry=selected_registry,
        )
        created = True
        read_initial = read_workspace_file(
            workspace_id,
            scratch_path,
            offset=1,
            limit=5,
            registry=selected_registry,
        )
        patched_result = patch_workspace_file(
            workspace_id,
            scratch_path,
            CHATGPT_SCRATCH_SMOKE_INITIAL,
            CHATGPT_SCRATCH_SMOKE_PATCHED,
            context_token=context_token,
            registry=selected_registry,
        )
        read_patched = read_workspace_file(
            workspace_id,
            scratch_path,
            offset=1,
            limit=5,
            registry=selected_registry,
        )
        cleanup_result = delete_workspace_file(
            workspace_id,
            scratch_path,
            context_token=context_token,
            registry=selected_registry,
        )
        created = False
    except Exception:
        if created:
            try:
                delete_workspace_file(
                    workspace_id,
                    scratch_path,
                    context_token=context_token,
                    registry=selected_registry,
                )
            except Exception:
                pass
        raise

    initial_content = (read_initial or {}).get("content", "")
    patched_content = (read_patched or {}).get("content", "")
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "path": scratch_path,
        "server_chosen_path": True,
        "arbitrary_path_accepted": False,
        "arbitrary_content_accepted": False,
        "write": {
            "ok": bool(write_result and write_result.get("changed") is True),
            "bytes_written": (write_result or {}).get("bytes_written"),
            "root_exposed": False,
        },
        "read_initial": {
            "ok": initial_content == CHATGPT_SCRATCH_SMOKE_INITIAL,
            "lines_returned": (read_initial or {}).get("lines_returned"),
            "content_sha256": hashlib.sha256(initial_content.encode("utf-8")).hexdigest(),
            "root_exposed": False,
        },
        "patch": {
            "ok": bool(patched_result and patched_result.get("changed") is True),
            "replacements": (patched_result or {}).get("replacements"),
            "root_exposed": False,
        },
        "read_patched": {
            "ok": patched_content == CHATGPT_SCRATCH_SMOKE_PATCHED,
            "lines_returned": (read_patched or {}).get("lines_returned"),
            "content_sha256": hashlib.sha256(patched_content.encode("utf-8")).hexdigest(),
            "root_exposed": False,
        },
        "cleanup": {
            "ok": bool(cleanup_result and cleanup_result.get("deleted") is True),
            "deleted": bool(cleanup_result and cleanup_result.get("deleted") is True),
            "root_exposed": False,
        },
        "audit": {
            "tool": "workspace_scratch_smoke",
            "llm_calls": 0,
            "root_exposed": False,
            "scratch_content_server_generated": True,
        },
    }


def create_workspace_directory(
    workspace_id: str,
    path: str,
    *,
    parents: bool = False,
    exist_ok: bool = False,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Create a workspace-relative directory after context and write-policy gates."""

    assert_default_tools_are_no_model()
    if not isinstance(parents, bool):
        raise ProfileRouterError("invalid_directory_option", "parents must be a boolean")
    if not isinstance(exist_ok, bool):
        raise ProfileRouterError("invalid_directory_option", "exist_ok must be a boolean")
    workspace = _require_workspace_write_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    resolved_path = resolve_workspace_directory_create_path(
        workspace,
        path,
        parents=parents,
    )
    existed = resolved_path.exists()
    if existed:
        if not resolved_path.is_dir():
            raise ProfileRouterError("not_a_directory", "directory path collides with a file")
        if not exist_ok:
            raise ProfileRouterError("directory_already_exists", "directory already exists")
    try:
        resolved_path.mkdir(parents=parents, exist_ok=exist_ok)
    except FileExistsError as exc:
        raise ProfileRouterError("directory_already_exists", "directory already exists") from exc
    except FileNotFoundError as exc:
        raise ProfileRouterError("parent_directory_not_found", "parent directory does not exist") from exc
    except OSError as exc:
        raise ProfileRouterError("directory_not_writable", "directory could not be created") from exc

    try:
        resolved_after = resolved_path.resolve(strict=True)
    except OSError as exc:
        raise ProfileRouterError("directory_not_writable", "created directory is not accessible") from exc
    resolved_root = _resolve_existing_local_path(workspace.root, "workspace root")
    if not _path_is_relative_to(resolved_after, resolved_root):
        raise ProfileRouterError("symlink_traversal_denied", "created directory escapes workspace root")
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "path": posixpath.normpath(path.strip()),
        "type": "directory",
        "created": not existed,
        "existed": existed,
        "parents": parents,
        "exist_ok": exist_ok,
        "audit": {
            "tool": "directory_create",
            "llm_calls": 0,
            "root_exposed": False,
        },
    }


def _run_workspace_git(
    workspace: WorkspaceMetadata,
    args: list[str],
    *,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded, no-shell Git command inside a workspace root."""

    try:
        return subprocess.run(
            ["git", "-C", workspace.root, *args],
            cwd=workspace.root,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            },
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProfileRouterError("git_unavailable", "git executable is not available") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProfileRouterError("git_timeout", "git command timed out") from exc
    except OSError as exc:
        raise ProfileRouterError("git_unavailable", "git command could not be started") from exc


def _require_git_workspace(workspace: WorkspaceMetadata) -> None:
    probe = _run_workspace_git(workspace, ["rev-parse", "--is-inside-work-tree"], timeout=5)
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        raise ProfileRouterError("not_a_git_workspace", "workspace_diff requires a Git workspace")


def _git_output(workspace: WorkspaceMetadata, args: list[str], *, timeout: int = 10) -> str:
    result = _run_workspace_git(workspace, args, timeout=timeout)
    if result.returncode != 0:
        command = " ".join(args[:2]) if args else "command"
        raise ProfileRouterError("git_command_failed", f"git {command} failed")
    return result.stdout


def _split_nul_output(output: str) -> list[str]:
    return [item for item in output.split("\0") if item]


def _normalize_git_relative_path(path: str) -> str | None:
    if not isinstance(path, str):
        return None
    raw = path.strip()
    if not raw or raw.startswith("/"):
        return None
    normalized = posixpath.normpath(raw)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        return None
    return normalized


def _workspace_diff_path_status(workspace: WorkspaceMetadata, rel_path: str) -> tuple[str | None, str | None]:
    normalized = _normalize_git_relative_path(rel_path)
    if normalized is None:
        return None, "invalid_path"
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] in PROTECTED_WORKSPACE_DIFF_DIRS:
        if not _is_safe_workspace_plan_path(normalized):
            return normalized, "protected_local_metadata"
    if parts and parts[-1] == FUNCIONES_TXT_FILENAME:
        return normalized, "protected_local_metadata"
    if _is_secret_path(normalized) and not _is_safe_workspace_plan_path(normalized):
        return normalized, "secret_path_denied"

    candidate = posixpath.normpath(posixpath.join(workspace.root, normalized))
    if not _path_within_root(candidate, workspace.root):
        return normalized, "path_outside_workspace"
    candidate_path = Path(candidate)
    if candidate_path.exists():
        try:
            resolve_workspace_path(workspace, normalized)
        except ProfileRouterError as exc:
            return normalized, exc.code
    return normalized, None


def _filter_workspace_diff_paths(
    workspace: WorkspaceMetadata,
    paths: Iterable[str],
    *,
    max_files: int,
) -> tuple[list[str], list[dict], bool]:
    safe: list[str] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    truncated = False
    for path in paths:
        normalized, reason = _workspace_diff_path_status(workspace, path)
        public_path = normalized or str(path)
        if reason is not None:
            skipped.append({"path": public_path, "reason": reason})
            continue
        if normalized is None:
            skipped.append({"path": public_path, "reason": "invalid_path"})
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        if len(safe) >= max_files:
            truncated = True
            skipped.append({"path": normalized, "reason": "file_limit_exceeded"})
            continue
        safe.append(normalized)
    return safe, skipped, truncated


def _bounded_workspace_diff_text(diff_text: str) -> dict:
    diff_text = _redact_sensitive_text_fields(diff_text)
    truncated = len(diff_text) > MAX_WORKSPACE_DIFF_CHARS
    if truncated:
        diff_text = diff_text[:MAX_WORKSPACE_DIFF_CHARS] + "\n... [diff truncated]\n"
    return {
        "unified": diff_text,
        "truncated": truncated,
        "max_chars": MAX_WORKSPACE_DIFF_CHARS,
    }


def _literal_git_pathspecs(paths: Iterable[str]) -> list[str]:
    """Build Git literal pathspecs so odd filenames cannot expand matches."""

    return [f":(literal){path}" for path in paths]


def _safe_git_public_text(value: str, *, max_chars: int = 240) -> str:
    """Return a bounded, secret-redacted Git label/message."""

    text = _redact_sensitive_text_fields(str(value or "").strip())
    if _is_secret_path(text) or SENSITIVE_PATH_RE.search(text):
        return "[REDACTED]"
    return text[:max_chars]


def _git_audit(tool_name: str, workspace: WorkspaceMetadata) -> dict:
    return {
        "tool": tool_name,
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "git_read_only": True,
        "git_mutation_allowed": False,
        "public_mcp_exposure": "disabled_pending_git_parity_policy_review",
        "workspace_id": workspace.workspace_id,
    }


def _git_has_head(workspace: WorkspaceMetadata) -> bool:
    return _run_workspace_git(workspace, ["rev-parse", "--verify", "HEAD"], timeout=5).returncode == 0


def _git_short_head(workspace: WorkspaceMetadata) -> str | None:
    result = _run_workspace_git(workspace, ["rev-parse", "--short", "HEAD"], timeout=5)
    if result.returncode != 0:
        return None
    return _safe_git_public_text(result.stdout.strip(), max_chars=64) or None


def _git_current_branch(workspace: WorkspaceMetadata) -> str | None:
    result = _run_workspace_git(workspace, ["branch", "--show-current"], timeout=5)
    if result.returncode != 0:
        return None
    return _safe_git_public_text(result.stdout.strip(), max_chars=200) or None


def _parse_git_status_entries(
    workspace: WorkspaceMetadata,
    output: str,
    *,
    limit: int,
) -> tuple[list[dict], list[dict], bool]:
    raw_entries = _split_nul_output(output)
    changes: list[dict] = []
    skipped: list[dict] = []
    truncated = False
    index = 0
    while index < len(raw_entries):
        raw_entry = raw_entries[index]
        status_code = raw_entry[:2]
        raw_path = raw_entry[3:] if len(raw_entry) > 3 else ""
        normalized, reason = _workspace_diff_path_status(workspace, raw_path)
        public_path = normalized or _safe_git_public_text(raw_path)
        if reason is not None or normalized is None:
            skipped.append({"path": public_path, "reason": reason or "invalid_path"})
        elif len(changes) >= limit:
            truncated = True
            skipped.append({"path": normalized, "reason": "entry_limit_exceeded"})
        else:
            changes.append({"status": status_code.strip() or status_code, "path": normalized})
        if status_code[:1] in {"R", "C"} and index + 1 < len(raw_entries):
            index += 2
        else:
            index += 1
    return changes, skipped, truncated


def _git_branch_summary(workspace: WorkspaceMetadata, route_policy: ProfileRoutePolicy) -> dict:
    current_branch = _git_current_branch(workspace)
    protected = set(route_policy.protected_branches)
    return {
        "current": current_branch,
        "detached": current_branch is None and _git_has_head(workspace),
        "head": _git_short_head(workspace),
        "protected": bool(current_branch and current_branch in protected),
        "protected_branches": sorted(protected),
    }


def _shell_command_tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError as exc:
        raise ProfileRouterError(
            "invalid_terminal_command",
            "terminal command could not be parsed safely",
        ) from exc


def _terminal_token_basename(token: str) -> str:
    return posixpath.basename(token).lower()


def _terminal_non_control_tokens(tokens: Iterable[str]) -> list[str]:
    return [
        token
        for token in tokens
        if not _terminal_command_has_shell_control([token])
    ]


def _find_hermes_model_command(tokens: list[str]) -> str | None:
    non_control = _terminal_non_control_tokens(tokens)
    for index, token in enumerate(non_control):
        if _terminal_token_basename(token) != "hermes":
            continue
        following = [item for item in non_control[index + 1 :] if not item.startswith("-")]
        if any(item.lower() in HERMES_MODEL_SUBCOMMANDS for item in following):
            return "hermes chat"
        # Bare Hermes, option-only Hermes (for example ``hermes --profile maker``),
        # and other Hermes subcommands stay blocked until a safe-subcommand policy
        # exists. The CLI defaults to chat when no subcommand is provided.
        return "hermes"
    return None


def _append_terminal_pattern_reasons(
    reasons: list[dict],
    *,
    command: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
    code: str,
    detail_prefix: str,
) -> None:
    for name, pattern in patterns:
        if pattern.search(command):
            reasons.append(
                {
                    "code": code,
                    "name": name,
                    "detail": f"{detail_prefix}: {name}",
                }
            )


def classify_terminal_command(command: str) -> dict:
    """Classify a future terminal command without executing it.

    This is a no-model, fail-closed Phase 6 helper. It intentionally returns no
    raw command text because command strings may contain secrets. The actual
    ``terminal_run`` wrapper still requires fresh workspace context first and
    remains disabled until execution policy, output limits, and audit are built.
    """

    assert_default_tools_are_no_model()
    if not isinstance(command, str):
        raise ProfileRouterError("invalid_terminal_command", "command must be a string")
    stripped = command.strip()
    if not stripped:
        raise ProfileRouterError("invalid_terminal_command", "command is required")
    if len(stripped) > MAX_TERMINAL_COMMAND_CHARS:
        raise ProfileRouterError(
            "terminal_command_too_large",
            f"command exceeds {MAX_TERMINAL_COMMAND_CHARS} characters",
        )

    tokens = _shell_command_tokens(stripped)
    has_shell_control = (
        "\n" in stripped
        or "\r" in stripped
        or _terminal_command_has_shell_control(tokens)
    )
    non_control = _terminal_non_control_tokens(tokens)
    basenames = {_terminal_token_basename(token) for token in non_control}
    reasons: list[dict] = []

    for marker in MODEL_COMMAND_TEXT_MARKERS:
        if re.search(rf"(?<![\w.]){re.escape(marker)}(?![\w.])", stripped):
            reasons.append(
                {
                    "code": "model_command",
                    "name": marker,
                    "detail": f"Hermes agent/model marker is blocked: {marker}",
                }
            )
    for name in sorted(basenames & MODEL_COMMAND_NAMES):
        reasons.append(
            {
                "code": "model_command",
                "name": name,
                "detail": f"Model/agent CLI is blocked: {name}",
            }
        )
    hermes_model_command = _find_hermes_model_command(tokens)
    if hermes_model_command is not None:
        reasons.append(
            {
                "code": "model_command",
                "name": hermes_model_command,
                "detail": f"Hermes model command is blocked: {hermes_model_command}",
            }
        )

    _append_terminal_pattern_reasons(
        reasons,
        command=stripped,
        patterns=DESTRUCTIVE_COMMAND_PATTERNS,
        code="destructive_command",
        detail_prefix="Destructive filesystem/Git command is blocked",
    )
    _append_terminal_pattern_reasons(
        reasons,
        command=stripped,
        patterns=PROTECTED_GIT_COMMAND_PATTERNS,
        code="protected_git_command",
        detail_prefix="Protected Git operation is blocked pending policy",
    )
    _append_terminal_pattern_reasons(
        reasons,
        command=stripped,
        patterns=DEPLOY_COMMAND_PATTERNS,
        code="deploy_command",
        detail_prefix="Deploy/production command is blocked pending policy",
    )

    blocked = bool(reasons)
    return {
        "cost_class": COST_CLASS_NO_MODEL,
        "llm_calls": 0,
        "uses_shell": False,
        "executes": False,
        "no_shell_compatible": not has_shell_control,
        "command_length": len(stripped),
        "parsed_token_count": len(tokens),
        "blocked": blocked,
        "decision": "blocked" if blocked else "disabled_pending_execution_policy",
        "risk_level": "blocked" if blocked else "low_unexecuted",
        "reasons": reasons,
    }


def _terminal_allowlist_match(
    command: str,
    execution_policy: TerminalExecutionPolicy,
) -> tuple[str | None, int | None]:
    stripped = command.strip()
    for index, allowed in enumerate(execution_policy.allowed_commands):
        if stripped == allowed:
            return "exact", index
    for index, prefix in enumerate(execution_policy.allowed_command_prefixes):
        if stripped == prefix or stripped.startswith(prefix + " "):
            return "prefix", index
    return None, None


def _evaluate_terminal_execution_policy(
    command: str,
    classification: Mapping[str, Any],
    route_policy: ProfileRoutePolicy,
) -> dict:
    execution_policy = route_policy.terminal_execution_policy
    match_type, match_index = _terminal_allowlist_match(command, execution_policy)
    reasons: list[dict] = []

    if execution_policy.require_no_shell and not classification.get(
        "no_shell_compatible", False
    ):
        reasons.append(
            {
                "code": "terminal_shell_control_not_allowed",
                "name": "require_no_shell",
                "detail": "Terminal execution policy requires no-shell argv-compatible commands",
            }
        )
    if execution_policy.enabled and match_type is None:
        reasons.append(
            {
                "code": "terminal_command_not_allowlisted",
                "name": "execution_allowlist",
                "detail": "Terminal execution policy requires an explicit command allowlist match",
            }
        )

    blocked = execution_policy.enabled and bool(reasons)
    if not execution_policy.enabled:
        decision = "execution_disabled_by_policy"
    elif blocked:
        decision = "blocked_by_execution_policy"
    else:
        decision = "allowlisted_pending_execution_implementation"

    return {
        "enabled": execution_policy.enabled,
        "require_no_shell": execution_policy.require_no_shell,
        "allowed_commands_count": len(execution_policy.allowed_commands),
        "allowed_command_prefixes_count": len(
            execution_policy.allowed_command_prefixes
        ),
        "allowlist_redacted": True,
        "allowlist_match": match_type is not None,
        "allowlist_match_type": match_type,
        "allowlist_match_index": match_index,
        "blocked": blocked,
        "decision": decision,
        "reasons": reasons,
    }


def _terminal_argv_shape(tokens: Iterable[str]) -> dict:
    argv = list(tokens)
    return {
        "shell": False,
        "argv_redacted": True,
        "argc": len(argv),
        "argument_count": max(len(argv) - 1, 0),
        "option_count": sum(1 for token in argv[1:] if token.startswith("-")),
        "path_like_token_count": sum(1 for token in argv if "/" in token),
        "assignment_prefix_count": sum(
            1
            for token in argv
            if "=" in token and not token.startswith("-") and token.split("=", 1)[0]
        ),
    }


def _terminal_sanitized_env_policy() -> dict:
    return {
        "mode": "sanitized_minimal",
        "inherits_parent_env": False,
        "values_redacted": True,
        "allowed_keys": list(TERMINAL_SANITIZED_ENV_ALLOWED_KEYS),
        "blocked_name_markers": list(TERMINAL_SANITIZED_ENV_BLOCKED_MARKERS),
        "explicit_env_overrides_allowed": False,
    }


def _terminal_env_key_is_allowed(key: str) -> bool:
    upper_key = key.upper()
    return key in TERMINAL_SANITIZED_ENV_ALLOWED_KEYS and not any(
        marker in upper_key for marker in TERMINAL_SANITIZED_ENV_BLOCKED_MARKERS
    )


def _build_terminal_sanitized_env() -> dict[str, str]:
    """Build the future executor's minimal env without inheriting secrets.

    The returned mapping is for internal subprocess use only and is never exposed
    through MCP responses. Values are deterministic defaults rather than parent
    process values, so profile credentials and model-provider keys cannot leak
    into terminal command environments by inheritance.
    """

    env: dict[str, str] = {}
    for key in TERMINAL_SANITIZED_ENV_ALLOWED_KEYS:
        if not _terminal_env_key_is_allowed(key):
            continue
        value = TERMINAL_SANITIZED_ENV_DEFAULTS.get(key)
        if not isinstance(value, str) or not value or "\x00" in value:
            continue
        env[key] = value
    return env


def _prepare_terminal_subprocess_plan(
    command: str,
    *,
    resolved_cwd: Path,
    public_cwd: str,
    timeout_seconds: int,
    max_output_chars: int,
) -> TerminalSubprocessPlan:
    """Prepare a bounded no-shell subprocess plan without executing it."""

    stripped = command.strip()
    tokens = _shell_command_tokens(stripped)
    if "\n" in stripped or "\r" in stripped or _terminal_command_has_shell_control(tokens):
        raise ProfileRouterError(
            "terminal_shell_control_not_allowed",
            "terminal command cannot contain shell control operators",
        )
    if not tokens:
        raise ProfileRouterError("invalid_terminal_command", "command is required")
    return TerminalSubprocessPlan(
        argv=tuple(tokens),
        cwd=resolved_cwd,
        public_cwd=public_cwd,
        env=_build_terminal_sanitized_env(),
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )


def _coerce_terminal_output_stream(value: str | bytes, stream_name: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    raise ProfileRouterError(
        "invalid_terminal_result",
        f"{stream_name} must be text or bytes for terminal result shaping",
    )


def _shape_terminal_output_stream(
    value: str | bytes,
    *,
    stream_name: str,
    budget_chars: int,
) -> TerminalOutputStreamShape:
    text = _coerce_terminal_output_stream(value, stream_name)
    budget = max(budget_chars, 0)
    returned = text[:budget]
    return TerminalOutputStreamShape(
        text=returned,
        truncated=len(text) > len(returned),
        original_chars=len(text),
        returned_chars=len(returned),
    )


def _terminal_result_status(returncode: int | None, *, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if not isinstance(returncode, int):
        raise ProfileRouterError(
            "invalid_terminal_result",
            "returncode must be an integer unless the terminal command timed out",
        )
    if returncode == 0:
        return "success"
    return "failed"


def _shape_terminal_subprocess_result(
    plan: TerminalSubprocessPlan,
    *,
    returncode: int | None,
    stdout: str | bytes,
    stderr: str | bytes,
    timed_out: bool = False,
) -> dict:
    """Shape a future subprocess result without exposing roots, env, or argv.

    This is Phase 6 scaffolding only. It defines the bounded output/status/audit
    contract that a later no-shell executor must satisfy; it does not call
    ``subprocess.run`` and current ``TerminalSubprocessPlan`` instances still
    carry ``executes=False``.
    """

    max_output_chars = _bounded_terminal_int(
        plan.max_output_chars,
        "max_output_chars",
        default=MAX_TERMINAL_OUTPUT_CHARS,
        minimum=1,
        maximum=MAX_TERMINAL_OUTPUT_CHARS,
    )
    stdout_shape = _shape_terminal_output_stream(
        stdout,
        stream_name="stdout",
        budget_chars=max_output_chars,
    )
    stderr_shape = _shape_terminal_output_stream(
        stderr,
        stream_name="stderr",
        budget_chars=max_output_chars - stdout_shape.returned_chars,
    )
    normalized_returncode = None if returncode is None else int(returncode)

    return {
        "status": _terminal_result_status(normalized_returncode, timed_out=timed_out),
        "returncode": normalized_returncode,
        "timed_out": bool(timed_out),
        "stdout": asdict(stdout_shape),
        "stderr": asdict(stderr_shape),
        "output": {
            "max_output_chars": max_output_chars,
            "returned_chars": stdout_shape.returned_chars + stderr_shape.returned_chars,
            "truncated": stdout_shape.truncated or stderr_shape.truncated,
            "stdout_truncated": stdout_shape.truncated,
            "stderr_truncated": stderr_shape.truncated,
        },
        "working_directory": plan.public_cwd,
        "audit": {
            "tool": "terminal_run",
            "llm_calls": 0,
            "root_exposed": False,
            "uses_shell": plan.uses_shell,
            "executes": plan.executes,
            "execution_attempted": plan.executes,
            "subprocess_run_allowed": plan.executes,
            "subprocess_run_called": plan.executes,
            "argv_redacted": True,
            "env_values_exposed": False,
            "public_mcp_exposure": "disabled_pending_http_auth_config_review",
        },
    }


def _terminal_public_cwd_label(public_cwd: str) -> str:
    if public_cwd in {"", "."}:
        return "<workspace>"
    return f"<workspace>/{public_cwd}"


def _terminal_output_redactions(
    workspace: WorkspaceMetadata,
    plan: TerminalSubprocessPlan,
) -> list[tuple[str, str]]:
    """Return private output redactions for host roots and env values."""

    root = Path(workspace.root)
    candidates: list[tuple[str, str]] = []
    try:
        candidates.append((str(plan.cwd.resolve()), _terminal_public_cwd_label(plan.public_cwd)))
    except OSError:
        candidates.append((str(plan.cwd), _terminal_public_cwd_label(plan.public_cwd)))
    try:
        candidates.append((str(root.resolve()), "<workspace>"))
    except OSError:
        candidates.append((str(root), "<workspace>"))
    candidates.append((str(plan.cwd), _terminal_public_cwd_label(plan.public_cwd)))
    candidates.append((str(root), "<workspace>"))
    for value in plan.env.values():
        if value:
            candidates.append((value, "<redacted_env_value>"))

    redactions: dict[str, str] = {}
    for source, replacement in candidates:
        # Never redact filesystem separators or empty strings globally.
        if source and source != "/":
            redactions.setdefault(source, replacement)
    return sorted(redactions.items(), key=lambda item: len(item[0]), reverse=True)


def _redact_terminal_output_stream(
    value: str | bytes,
    *,
    workspace: WorkspaceMetadata,
    plan: TerminalSubprocessPlan,
    stream_name: str,
) -> str:
    text = _coerce_terminal_output_stream(value, stream_name)
    for source, replacement in _terminal_output_redactions(workspace, plan):
        text = text.replace(source, replacement)
    return text


def _run_terminal_subprocess_plan(
    plan: TerminalSubprocessPlan,
    *,
    workspace: WorkspaceMetadata,
) -> dict:
    """Run one allowlisted no-shell command with sanitized env and output caps."""

    if plan.uses_shell:
        raise ProfileRouterError(
            "terminal_shell_execution_not_allowed",
            "terminal_run only supports shell=false subprocess execution",
        )
    executing_plan = replace(plan, executes=True)
    try:
        completed = subprocess.run(
            list(plan.argv),
            cwd=str(plan.cwd),
            env=dict(plan.env),
            text=True,
            capture_output=True,
            timeout=plan.timeout_seconds,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ProfileRouterError(
            "terminal_executable_not_found",
            "allowlisted terminal executable is not available in the sanitized PATH",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout = _redact_terminal_output_stream(
            exc.stdout or "",
            workspace=workspace,
            plan=plan,
            stream_name="stdout",
        )
        stderr = _redact_terminal_output_stream(
            exc.stderr or "",
            workspace=workspace,
            plan=plan,
            stream_name="stderr",
        )
        return _shape_terminal_subprocess_result(
            executing_plan,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
        )
    except OSError as exc:
        raise ProfileRouterError(
            "terminal_execution_failed",
            "terminal command could not be started safely",
        ) from exc

    stdout = _redact_terminal_output_stream(
        completed.stdout,
        workspace=workspace,
        plan=plan,
        stream_name="stdout",
    )
    stderr = _redact_terminal_output_stream(
        completed.stderr,
        workspace=workspace,
        plan=plan,
        stream_name="stderr",
    )
    return _shape_terminal_subprocess_result(
        executing_plan,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminal_shaped_result_contract(plan: TerminalSubprocessPlan) -> dict:
    """Describe the private terminal result shape without producing output."""

    return {
        "shape": "terminal_subprocess_result",
        "status_values": ["success", "failed", "timeout"],
        "stdout_stderr_bounded": True,
        "returncode_included": True,
        "timed_out_included": True,
        "max_output_chars": plan.max_output_chars,
        "working_directory": plan.public_cwd,
        "root_exposed": False,
        "argv_values_exposed": False,
        "env_values_exposed": False,
        "uses_shell": plan.uses_shell,
        "llm_calls": 0,
    }


def _terminal_executor_boundary_audit(plan: TerminalSubprocessPlan) -> dict:
    """Private, non-executing adapter boundary for future terminal execution.

    The boundary intentionally consumes a ``TerminalSubprocessPlan`` but does not
    invoke ``subprocess.run``. It exists so later real execution has a narrow
    handoff point and must preserve the already-tested redacted result contract.
    """

    return {
        "adapter": "non_executing_terminal_executor_boundary",
        "implementation_status": "pending_execution_policy_audit_public_exposure_review",
        "accepts_plan_type": "TerminalSubprocessPlan",
        "execution_attempted": False,
        "subprocess_run_allowed": False,
        "subprocess_run_called": False,
        "executes": plan.executes,
        "shell": plan.uses_shell,
        "timeout_seconds": plan.timeout_seconds,
        "max_output_chars": plan.max_output_chars,
        "argv_redacted": True,
        "argc": len(plan.argv),
        "env_key_count": len(plan.env),
        "env_values_exposed": False,
        "cwd": {
            "workspace_relative": plan.public_cwd,
            "root_exposed": False,
            "resolved_host_path_exposed": False,
        },
        "result_contract": _terminal_shaped_result_contract(plan),
    }


def _terminal_env_is_sanitized(plan: TerminalSubprocessPlan) -> bool:
    """Return whether a future executor plan uses only deterministic safe env."""

    return dict(plan.env) == _build_terminal_sanitized_env() and all(
        _terminal_env_key_is_allowed(key) for key in plan.env
    )


def _terminal_execution_readiness_review_gate(
    plan: TerminalSubprocessPlan,
    *,
    classification: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    executor_boundary: Mapping[str, Any],
    fresh_context_validated: bool,
    metadata: Mapping[str, RouterToolMetadata] = ROUTER_TOOL_METADATA,
) -> dict:
    """Private pre-executor review gate for future terminal execution.

    This gate records the checks that must hold before the private direct runner
    may use ``subprocess.run``. Public MCP exposure remains blocked, and raw
    command, argv, env-value, and host-root serialization stay prohibited.
    """

    terminal_meta = metadata.get("terminal_run")
    result_contract = _terminal_shaped_result_contract(plan)
    checks = {
        "fresh_context_validated_upstream": bool(fresh_context_validated),
        "classification_not_blocked": classification.get("blocked") is False,
        "decision_pending_execution_implementation": classification.get("decision")
        == "disabled_pending_execution_implementation",
        "terminal_execution_policy_enabled": execution_policy.get("enabled") is True,
        "allowlist_match": execution_policy.get("allowlist_match") is True,
        "require_no_shell": execution_policy.get("require_no_shell") is True,
        "plan_shell_false": plan.uses_shell is False,
        "plan_executes_false": plan.executes is False,
        "sanitized_env": _terminal_env_is_sanitized(plan),
        "bounded_result_contract": bool(
            result_contract["stdout_stderr_bounded"]
            and result_contract["root_exposed"] is False
            and result_contract["argv_values_exposed"] is False
            and result_contract["env_values_exposed"] is False
            and result_contract["uses_shell"] is False
            and result_contract["llm_calls"] == 0
            and plan.max_output_chars <= MAX_TERMINAL_OUTPUT_CHARS
        ),
        "executor_boundary_non_executing": bool(
            executor_boundary.get("execution_attempted") is False
            and executor_boundary.get("subprocess_run_allowed") is False
            and executor_boundary.get("subprocess_run_called") is False
            and executor_boundary.get("executes") is False
            and executor_boundary.get("shell") is False
        ),
        "tool_metadata_no_model": bool(
            terminal_meta
            and terminal_meta.cost_class == COST_CLASS_NO_MODEL
            and terminal_meta.llm_calls == 0
        ),
        "public_mcp_absent_by_default": bool(
            terminal_meta
            and terminal_meta.enabled_by_default is False
            and terminal_meta.requires_context is True
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]

    return {
        "gate": "terminal_execution_readiness_review",
        "scope": "private_non_executing_terminal_scaffold",
        "pre_executor_checks_passed": not failed_checks,
        "current_phase_allows_subprocess_run": not failed_checks,
        "subprocess_run_allowed": not failed_checks,
        "public_mcp_subprocess_run_allowed": False,
        "real_executor_status": "private_direct_runner_enabled_public_mcp_blocked",
        "checks": checks,
        "failed_checks": failed_checks,
        "fresh_context_enforced_before_gate": bool(fresh_context_validated),
        "raw_command_exposed": False,
        "argv_values_exposed": False,
        "env_values_exposed": False,
        "root_exposed": False,
        "llm_calls": 0,
        "sanitized_env": {
            "inherits_parent_env": False,
            "key_count": len(plan.env),
            "values_redacted": True,
        },
    }


def _terminal_execution_plan_audit(
    command: str,
    *,
    classification: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    resolved_cwd: Path,
    public_cwd: str,
    timeout_seconds: int,
    max_output_chars: int,
    fresh_context_validated: bool = False,
) -> dict:
    """Return a redacted, non-executing argv/env plan for allowlisted commands."""

    plan_available = bool(
        not classification.get("blocked", False)
        and execution_policy.get("enabled", False)
        and execution_policy.get("allowlist_match", False)
    )
    prepared_plan = None
    executor_boundary = None
    execution_readiness_review = None
    if plan_available:
        prepared_plan = _prepare_terminal_subprocess_plan(
            command,
            resolved_cwd=resolved_cwd,
            public_cwd=public_cwd,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        executor_boundary = _terminal_executor_boundary_audit(prepared_plan)
        execution_readiness_review = _terminal_execution_readiness_review_gate(
            prepared_plan,
            classification=classification,
            execution_policy=execution_policy,
            executor_boundary=executor_boundary,
            fresh_context_validated=fresh_context_validated,
        )

    return {
        "available": plan_available,
        "implementation_status": "private_no_shell_subprocess_runner_available",
        "executes": False,
        "shell": False,
        "argv": _terminal_argv_shape(prepared_plan.argv) if prepared_plan else None,
        "argv_redacted": True,
        "env_policy": _terminal_sanitized_env_policy(),
        "executor_boundary": executor_boundary,
        "execution_readiness_review": execution_readiness_review,
        "cwd": {
            "workspace_relative": public_cwd,
            "root_exposed": False,
            "resolved_host_path_exposed": False,
        },
        "limits": {
            "timeout_seconds": timeout_seconds,
            "max_output_chars": max_output_chars,
        },
    }


def preflight_terminal_command(
    workspace_id: str,
    command: str,
    *,
    timeout: int | None = 30,
    working_directory: str = ".",
    max_output_chars: int | None = MAX_TERMINAL_OUTPUT_CHARS,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Validate terminal policy/containment/caps without executing anything."""

    assert_default_tools_are_no_model()
    workspace, route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    timeout_seconds = _bounded_terminal_int(
        timeout,
        "timeout",
        default=30,
        minimum=1,
        maximum=MAX_TERMINAL_TIMEOUT_SECONDS,
    )
    capped_output_chars = _bounded_terminal_int(
        max_output_chars,
        "max_output_chars",
        default=MAX_TERMINAL_OUTPUT_CHARS,
        minimum=1,
        maximum=MAX_TERMINAL_OUTPUT_CHARS,
    )
    _resolved_cwd, public_cwd = _resolve_terminal_working_directory(
        workspace, working_directory
    )
    classification = classify_terminal_command(command)
    execution_policy = _evaluate_terminal_execution_policy(
        command, classification, route_policy
    )
    if execution_policy["blocked"]:
        classification = {
            **classification,
            "blocked": True,
            "decision": "blocked",
            "risk_level": "blocked",
            "reasons": [*classification["reasons"], *execution_policy["reasons"]],
        }
    elif (
        not classification["blocked"]
        and execution_policy["enabled"]
        and execution_policy["allowlist_match"]
    ):
        classification = {
            **classification,
            "decision": "disabled_pending_execution_implementation",
            "risk_level": "low_allowlisted_unexecuted",
        }
    execution_plan = _terminal_execution_plan_audit(
        command,
        classification=classification,
        execution_policy=execution_policy,
        resolved_cwd=_resolved_cwd,
        public_cwd=public_cwd,
        timeout_seconds=timeout_seconds,
        max_output_chars=capped_output_chars,
        fresh_context_validated=True,
    )
    return {
        **classification,
        "working_directory": public_cwd,
        "timeout_seconds": timeout_seconds,
        "max_output_chars": capped_output_chars,
        "policy": {
            "terminal_allowed": route_policy.allow_terminal,
            "git_push_allowed": route_policy.allow_git_push,
            "deploy_allowed": route_policy.allow_deploy,
            "protected_branches": list(route_policy.protected_branches),
        },
        "execution_policy": execution_policy,
        "execution_plan": execution_plan,
        "audit": {
            "tool": "terminal_run",
            "llm_calls": 0,
            "root_exposed": False,
            "uses_shell": False,
            "executes": False,
            "execution_policy_enabled": execution_policy["enabled"],
            "allowlist_match": execution_policy["allowlist_match"],
            "execution_plan_available": execution_plan["available"],
            "no_shell_compatible": classification["no_shell_compatible"],
            "public_mcp_exposure": "disabled_pending_http_auth_config_review",
        },
    }


def _run_preflighted_terminal_command(
    workspace_id: str,
    command: str,
    *,
    preflight: Mapping[str, Any],
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Execute a command only after the sanitized preflight allows it."""

    if preflight.get("blocked") or not preflight.get("execution_plan", {}).get("available"):
        raise ProfileRouterError(
            "terminal_execution_not_allowlisted",
            "terminal_run requires a fresh-context allowlisted execution plan",
        )
    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    resolved_cwd, public_cwd = _resolve_terminal_working_directory(
        workspace, str(preflight.get("working_directory") or ".")
    )
    timeout_seconds = _bounded_terminal_int(
        preflight.get("timeout_seconds"),
        "timeout",
        default=30,
        minimum=1,
        maximum=MAX_TERMINAL_TIMEOUT_SECONDS,
    )
    max_output_chars = _bounded_terminal_int(
        preflight.get("max_output_chars"),
        "max_output_chars",
        default=MAX_TERMINAL_OUTPUT_CHARS,
        minimum=1,
        maximum=MAX_TERMINAL_OUTPUT_CHARS,
    )
    plan = _prepare_terminal_subprocess_plan(
        command,
        resolved_cwd=resolved_cwd,
        public_cwd=public_cwd,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )
    return _run_terminal_subprocess_plan(plan, workspace=workspace)


@dataclass
class WorkspaceProcessRecord:
    """Server-side runtime-owned background process record.

    ``process_id`` is the only identifier exposed to MCP clients. The host PID,
    raw argv/command, host root, and env values remain server-side only.
    """

    process_id: str
    workspace_id: str
    profile_ref: str
    plan: TerminalSubprocessPlan
    process: Any
    stdout_path: Path
    stderr_path: Path
    started_at_monotonic: float
    timeout_seconds: int
    public_cwd: str
    killed: bool = False
    timed_out: bool = False
    stopped_at_monotonic: float | None = None
    timer: threading.Timer | None = None


class WorkspaceProcessRegistry:
    """In-memory registry for processes launched by this no-model runtime only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, WorkspaceProcessRecord] = {}
        self._log_dir = Path(tempfile.gettempdir()) / "hermes-profile-router-processes"

    def _ensure_log_dir(self) -> Path:
        self._log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        return self._log_dir

    def start(
        self,
        workspace: WorkspaceMetadata,
        plan: TerminalSubprocessPlan,
    ) -> WorkspaceProcessRecord:
        if plan.uses_shell:
            raise ProfileRouterError(
                "terminal_shell_execution_not_allowed",
                "process_start only supports shell=false subprocess execution",
            )
        process_id = f"proc_{uuid4().hex}"
        log_dir = self._ensure_log_dir()
        stdout_path = log_dir / f"{process_id}.stdout.log"
        stderr_path = log_dir / f"{process_id}.stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                list(plan.argv),
                cwd=str(plan.cwd),
                env=dict(plan.env),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                shell=False,
            )
        except FileNotFoundError as exc:
            raise ProfileRouterError(
                "terminal_executable_not_found",
                "allowlisted process executable is not available in the sanitized PATH",
            ) from exc
        except OSError as exc:
            raise ProfileRouterError(
                "process_start_failed",
                "background process could not be started safely",
            ) from exc
        finally:
            stdout_handle.close()
            stderr_handle.close()

        record = WorkspaceProcessRecord(
            process_id=process_id,
            workspace_id=workspace.workspace_id,
            profile_ref=workspace.profile_ref,
            plan=replace(plan, executes=True),
            process=process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            started_at_monotonic=time.monotonic(),
            timeout_seconds=plan.timeout_seconds,
            public_cwd=plan.public_cwd,
        )
        timer = threading.Timer(plan.timeout_seconds, self._timeout_kill, args=(process_id,))
        timer.daemon = True
        record.timer = timer
        with self._lock:
            self._records[process_id] = record
        timer.start()
        return record

    def _timeout_kill(self, process_id: str) -> None:
        with self._lock:
            record = self._records.get(process_id)
        if record is None:
            return
        if record.process.poll() is not None:
            self._mark_stopped(record)
            return
        record.timed_out = True
        try:
            record.process.terminate()
            record.process.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            record.process.kill()
            record.process.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
        except OSError:
            pass
        record.stopped_at_monotonic = record.stopped_at_monotonic or time.monotonic()

    def _mark_stopped(self, record: WorkspaceProcessRecord) -> None:
        if record.stopped_at_monotonic is None:
            record.stopped_at_monotonic = time.monotonic()
        if record.timer is not None:
            record.timer.cancel()

    def _get(self, workspace: WorkspaceMetadata, process_id: str) -> WorkspaceProcessRecord | None:
        normalized = _normalize_process_id(process_id)
        with self._lock:
            record = self._records.get(normalized)
        if record is None or record.workspace_id != workspace.workspace_id:
            return None
        if record.process.poll() is not None:
            self._mark_stopped(record)
        return record

    def list(self, workspace: WorkspaceMetadata, *, limit: int) -> list[WorkspaceProcessRecord]:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.workspace_id == workspace.workspace_id
            ]
        records.sort(key=lambda record: record.started_at_monotonic)
        selected = records[:limit]
        for record in selected:
            if record.process.poll() is not None:
                self._mark_stopped(record)
        return selected

    def poll(self, workspace: WorkspaceMetadata, process_id: str) -> WorkspaceProcessRecord | None:
        return self._get(workspace, process_id)

    def read_log(
        self,
        workspace: WorkspaceMetadata,
        process_id: str,
        *,
        max_chars: int,
    ) -> tuple[WorkspaceProcessRecord | None, dict | None]:
        record = self._get(workspace, process_id)
        if record is None:
            return None, None
        stdout = _read_process_log_file(record.stdout_path)
        stderr = _read_process_log_file(record.stderr_path)
        stdout = _redact_terminal_output_stream(
            stdout,
            workspace=workspace,
            plan=record.plan,
            stream_name="stdout",
        )
        stderr = _redact_terminal_output_stream(
            stderr,
            workspace=workspace,
            plan=record.plan,
            stream_name="stderr",
        )
        stdout_shape = _shape_terminal_output_stream(
            stdout,
            stream_name="stdout",
            budget_chars=max_chars,
        )
        stderr_shape = _shape_terminal_output_stream(
            stderr,
            stream_name="stderr",
            budget_chars=max_chars - stdout_shape.returned_chars,
        )
        return record, {
            "available": True,
            "max_chars": max_chars,
            "stdout": asdict(stdout_shape),
            "stderr": asdict(stderr_shape),
            "returned_chars": stdout_shape.returned_chars + stderr_shape.returned_chars,
            "truncated": stdout_shape.truncated or stderr_shape.truncated,
            "root_exposed": False,
            "host_pid_exposed": False,
            "raw_command_exposed": False,
        }

    def kill(self, workspace: WorkspaceMetadata, process_id: str) -> WorkspaceProcessRecord | None:
        record = self._get(workspace, process_id)
        if record is None:
            return None
        if record.process.poll() is None:
            record.killed = True
            try:
                record.process.terminate()
                record.process.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                record.process.kill()
                record.process.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
            except OSError:
                pass
        self._mark_stopped(record)
        return record

    def kill_all_for_workspace(self, workspace: WorkspaceMetadata) -> int:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.workspace_id == workspace.workspace_id
            ]
        killed = 0
        for record in records:
            if self.kill(workspace, record.process_id) is not None:
                killed += 1
        return killed


_WORKSPACE_PROCESS_REGISTRY = WorkspaceProcessRegistry()


def _read_process_log_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _process_status(record: WorkspaceProcessRecord) -> str:
    returncode = record.process.poll()
    if record.timed_out:
        return "timeout"
    if returncode is None:
        return "running"
    if record.killed:
        return "killed"
    if returncode == 0:
        return "success"
    return "failed"


def _process_returncode(record: WorkspaceProcessRecord) -> int | None:
    returncode = record.process.poll()
    return returncode if isinstance(returncode, int) else None


def _process_public_record(record: WorkspaceProcessRecord) -> dict:
    status = _process_status(record)
    return {
        "process_id": record.process_id,
        "workspace_id": record.workspace_id,
        "profile_ref": record.profile_ref,
        "status": status,
        "running": status == "running",
        "returncode": _process_returncode(record),
        "timed_out": record.timed_out,
        "tracked_by_runtime": True,
        "host_pid_exposed": False,
        "raw_command_exposed": False,
        "root_exposed": False,
        "working_directory": record.public_cwd,
        "timeout_seconds": record.timeout_seconds,
    }


def _process_registry_audit(tool_name: str, workspace: WorkspaceMetadata) -> dict:
    return {
        "tool": tool_name,
        "llm_calls": 0,
        "root_exposed": False,
        "uses_shell": False,
        "tracked_processes_only": True,
        "host_process_listing": False,
        "host_pid_exposed": False,
        "raw_command_exposed": False,
        "public_mcp_exposure": "disabled_pending_process_registry_policy_review",
        "workspace_id": workspace.workspace_id,
    }


def _normalize_process_id(process_id: str) -> str:
    if not isinstance(process_id, str):
        raise ProfileRouterError("invalid_process_id", "process_id must be a string")
    normalized = process_id.strip()
    if not normalized:
        raise ProfileRouterError("invalid_process_id", "process_id is required")
    if len(normalized) > 128 or "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        raise ProfileRouterError("invalid_process_id", "process_id is not a valid opaque process id")
    return normalized


def start_workspace_process(
    workspace_id: str,
    command: str,
    *,
    timeout: int | None = 30,
    working_directory: str = ".",
    context_token: str | None = None,
    max_output_chars: int | None = MAX_PROCESS_LOG_CHARS,
    registry: WorkspaceRegistry | None = None,
    process_registry: WorkspaceProcessRegistry | None = None,
) -> dict:
    """Start one allowlisted no-shell background process owned by this runtime."""

    assert_default_tools_are_no_model()
    preflight = preflight_terminal_command(
        workspace_id,
        command,
        timeout=timeout,
        working_directory=working_directory,
        max_output_chars=max_output_chars,
        context_token=context_token,
        registry=registry,
    )
    if preflight.get("blocked") or not preflight.get("execution_plan", {}).get("available"):
        return {
            "ok": False,
            "error": {
                "code": "terminal_command_blocked",
                "message": "process_start requires a fresh-context allowlisted no-shell execution plan",
            },
            "terminal_command": preflight,
            "audit": {
                **_process_registry_audit(
                    "process_start",
                    require_fresh_workspace_context(
                        workspace_id,
                        context_token=context_token,
                        registry=registry,
                    ),
                ),
                "process_started": False,
            },
        }

    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    resolved_cwd, public_cwd = _resolve_terminal_working_directory(workspace, working_directory)
    timeout_seconds = _bounded_terminal_int(
        preflight.get("timeout_seconds"),
        "timeout",
        default=30,
        minimum=1,
        maximum=MAX_TERMINAL_TIMEOUT_SECONDS,
    )
    capped_output_chars = _bounded_terminal_int(
        preflight.get("max_output_chars"),
        "max_output_chars",
        default=MAX_PROCESS_LOG_CHARS,
        minimum=1,
        maximum=MAX_TERMINAL_OUTPUT_CHARS,
    )
    plan = _prepare_terminal_subprocess_plan(
        command,
        resolved_cwd=resolved_cwd,
        public_cwd=public_cwd,
        timeout_seconds=timeout_seconds,
        max_output_chars=capped_output_chars,
    )
    record = (process_registry or _WORKSPACE_PROCESS_REGISTRY).start(workspace, plan)
    return {
        "ok": True,
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "process": _process_public_record(record),
        "terminal_command": preflight,
        "audit": {
            **_process_registry_audit("process_start", workspace),
            "process_started": True,
            "timeout_seconds": timeout_seconds,
            "max_output_chars": capped_output_chars,
        },
    }


def list_workspace_processes(
    workspace_id: str,
    *,
    context_token: str | None = None,
    limit: int | None = MAX_PROCESS_LIST_RESULTS,
    registry: WorkspaceRegistry | None = None,
    process_registry: WorkspaceProcessRegistry | None = None,
) -> dict:
    """Return runtime-tracked process metadata without inspecting host processes."""

    assert_default_tools_are_no_model()
    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    selected_limit = _bounded_int(
        limit,
        "limit",
        default=MAX_PROCESS_LIST_RESULTS,
        minimum=1,
        maximum=MAX_PROCESS_LIST_RESULTS,
    )
    records = (process_registry or _WORKSPACE_PROCESS_REGISTRY).list(
        workspace,
        limit=selected_limit,
    )
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "processes": [_process_public_record(record) for record in records],
        "process_count": len(records),
        "limit": selected_limit,
        "truncated": len(records) >= selected_limit,
        "registry": {
            "enabled": True,
            "tracked_processes_only": True,
            "host_process_listing": False,
            "launch_supported": True,
            "status": "runtime_owned_background_process_registry",
        },
        "audit": _process_registry_audit("process_list", workspace),
    }


def _untracked_process_payload(
    tool_name: str,
    workspace_id: str,
    process_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _normalize_process_id(process_id)
    return {
        "ok": False,
        "error": {
            "code": "process_not_found",
            "message": "No runtime-tracked process exists for this opaque process id",
        },
        "process": {
            "tracked_by_runtime": False,
            "id_exposed": False,
            "host_pid_exposed": False,
            "workspace_id": workspace.workspace_id,
        },
        "audit": _process_registry_audit(tool_name, workspace),
    }


def poll_workspace_process(
    workspace_id: str,
    process_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
    process_registry: WorkspaceProcessRegistry | None = None,
) -> dict:
    """Poll only runtime-tracked processes; arbitrary host process polling is denied."""

    assert_default_tools_are_no_model()
    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    record = (process_registry or _WORKSPACE_PROCESS_REGISTRY).poll(workspace, process_id)
    if record is None:
        return _untracked_process_payload(
            "process_poll",
            workspace_id,
            process_id,
            context_token=context_token,
            registry=registry,
        )
    return {
        "ok": True,
        "process": _process_public_record(record),
        "audit": _process_registry_audit("process_poll", workspace),
    }


def read_workspace_process_log(
    workspace_id: str,
    process_id: str,
    *,
    context_token: str | None = None,
    max_chars: int | None = MAX_PROCESS_LOG_CHARS,
    registry: WorkspaceRegistry | None = None,
    process_registry: WorkspaceProcessRegistry | None = None,
) -> dict:
    """Read bounded logs only for runtime-tracked processes."""

    assert_default_tools_are_no_model()
    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    selected_max_chars = _bounded_int(
        max_chars,
        "max_chars",
        default=MAX_PROCESS_LOG_CHARS,
        minimum=1,
        maximum=MAX_PROCESS_LOG_CHARS,
    )
    record, log_payload = (process_registry or _WORKSPACE_PROCESS_REGISTRY).read_log(
        workspace,
        process_id,
        max_chars=selected_max_chars,
    )
    if record is None or log_payload is None:
        payload = _untracked_process_payload(
            "process_log",
            workspace_id,
            process_id,
            context_token=context_token,
            registry=registry,
        )
        payload["log"] = {
            "available": False,
            "max_chars": selected_max_chars,
            "truncated": False,
            "root_exposed": False,
        }
        return payload
    return {
        "ok": True,
        "process": _process_public_record(record),
        "log": log_payload,
        "audit": _process_registry_audit("process_log", workspace),
    }


def kill_workspace_process(
    workspace_id: str,
    process_id: str,
    *,
    context_token: str | None = None,
    registry: WorkspaceRegistry | None = None,
    process_registry: WorkspaceProcessRegistry | None = None,
) -> dict:
    """Kill only runtime-tracked processes; arbitrary host process control is denied."""

    assert_default_tools_are_no_model()
    workspace, _route_policy = _require_workspace_terminal_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    record = (process_registry or _WORKSPACE_PROCESS_REGISTRY).kill(workspace, process_id)
    if record is None:
        return _untracked_process_payload(
            "process_kill",
            workspace_id,
            process_id,
            context_token=context_token,
            registry=registry,
        )
    return {
        "ok": True,
        "process": _process_public_record(record),
        "audit": _process_registry_audit("process_kill", workspace),
    }


def diff_workspace(
    workspace_id: str,
    *,
    context_token: str | None = None,
    max_files: int | None = MAX_WORKSPACE_DIFF_FILES,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Return a bounded, root-redacted Git diff/audit for an opened workspace."""

    assert_default_tools_are_no_model()
    selected_max_files = _bounded_int(
        max_files,
        "max_files",
        default=MAX_WORKSPACE_DIFF_FILES,
        minimum=1,
        maximum=MAX_WORKSPACE_DIFF_FILES,
    )
    workspace = require_fresh_workspace_context(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    _require_git_workspace(workspace)

    head_probe = _run_workspace_git(workspace, ["rev-parse", "--verify", "HEAD"], timeout=5)
    has_head = head_probe.returncode == 0
    tracked_candidates: list[str] = []
    if has_head:
        tracked_candidates = _split_nul_output(
            _git_output(
                workspace,
                [
                    "diff",
                    *SAFE_GIT_DIFF_FLAGS,
                    "--name-only",
                    "-z",
                    "--relative",
                    "HEAD",
                    "--",
                    ".",
                ],
            )
        )
    untracked_candidates = _split_nul_output(
        _git_output(
            workspace,
            ["ls-files", "--others", "--exclude-standard", "-z", "--", "."],
        )
    )

    tracked_files, skipped_tracked, tracked_truncated = _filter_workspace_diff_paths(
        workspace,
        tracked_candidates,
        max_files=selected_max_files,
    )
    remaining_file_slots = max(0, selected_max_files - len(tracked_files))
    untracked_files, skipped_untracked, untracked_truncated = _filter_workspace_diff_paths(
        workspace,
        untracked_candidates,
        max_files=remaining_file_slots,
    )

    diff_text = ""
    if has_head and tracked_files:
        diff_text = _git_output(
            workspace,
            [
                "diff",
                *SAFE_GIT_DIFF_FLAGS,
                "--relative",
                "HEAD",
                "--",
                *_literal_git_pathspecs(tracked_files),
            ],
        )

    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "tracked_files": tracked_files,
        "untracked_files": untracked_files,
        "skipped": skipped_tracked + skipped_untracked,
        "file_limit": selected_max_files,
        "truncated_files": tracked_truncated or untracked_truncated,
        "has_head": has_head,
        "diff": _bounded_workspace_diff_text(diff_text),
        "audit": {
            "tool": "workspace_diff",
            "llm_calls": 0,
            "root_exposed": False,
            "uses_shell": False,
            "git_read_only": True,
            "public_mcp_exposure": "enabled_read_only_v1",
        },
    }


def read_workspace_git_status(
    workspace_id: str,
    *,
    context_token: str | None = None,
    limit: int | None = MAX_GIT_STATUS_ENTRIES,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Return bounded read-only Git status after fresh context and git policy."""

    assert_default_tools_are_no_model()
    selected_limit = _bounded_int(
        limit,
        "limit",
        default=MAX_GIT_STATUS_ENTRIES,
        minimum=1,
        maximum=MAX_GIT_STATUS_ENTRIES,
    )
    workspace, route_policy = _require_workspace_git_read_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    output = _git_output(
        workspace,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    changes, skipped, truncated = _parse_git_status_entries(
        workspace,
        output,
        limit=selected_limit,
    )
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "branch": _git_branch_summary(workspace, route_policy),
        "changes": changes,
        "change_count": len(changes),
        "skipped": skipped,
        "limit": selected_limit,
        "truncated": truncated,
        "clean": not changes and not skipped,
        "audit": _git_audit("git_status", workspace),
    }


def read_workspace_git_diff(
    workspace_id: str,
    *,
    context_token: str | None = None,
    max_files: int | None = MAX_WORKSPACE_DIFF_FILES,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Return bounded read-only Git diff after fresh context and git policy."""

    assert_default_tools_are_no_model()
    selected_max_files = _bounded_int(
        max_files,
        "max_files",
        default=MAX_WORKSPACE_DIFF_FILES,
        minimum=1,
        maximum=MAX_WORKSPACE_DIFF_FILES,
    )
    workspace, route_policy = _require_workspace_git_read_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    del route_policy
    head_probe = _run_workspace_git(workspace, ["rev-parse", "--verify", "HEAD"], timeout=5)
    has_head = head_probe.returncode == 0
    tracked_candidates: list[str] = []
    if has_head:
        tracked_candidates = _split_nul_output(
            _git_output(
                workspace,
                [
                    "diff",
                    *SAFE_GIT_DIFF_FLAGS,
                    "--name-only",
                    "-z",
                    "--relative",
                    "HEAD",
                    "--",
                    ".",
                ],
            )
        )
    tracked_files, skipped_tracked, tracked_truncated = _filter_workspace_diff_paths(
        workspace,
        tracked_candidates,
        max_files=selected_max_files,
    )
    diff_text = ""
    if has_head and tracked_files:
        diff_text = _git_output(
            workspace,
            [
                "diff",
                *SAFE_GIT_DIFF_FLAGS,
                "--relative",
                "HEAD",
                "--",
                *_literal_git_pathspecs(tracked_files),
            ],
        )
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "tracked_files": tracked_files,
        "skipped": skipped_tracked,
        "file_limit": selected_max_files,
        "truncated_files": tracked_truncated,
        "has_head": has_head,
        "diff": _bounded_workspace_diff_text(diff_text),
        "audit": _git_audit("git_diff", workspace),
    }


def read_workspace_git_log(
    workspace_id: str,
    *,
    context_token: str | None = None,
    limit: int | None = MAX_GIT_LOG_COUNT,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Return bounded read-only commit log metadata after git policy."""

    assert_default_tools_are_no_model()
    selected_limit = _bounded_int(
        limit,
        "limit",
        default=MAX_GIT_LOG_COUNT,
        minimum=1,
        maximum=MAX_GIT_LOG_COUNT,
    )
    workspace, route_policy = _require_workspace_git_read_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    commits: list[dict] = []
    has_head = _git_has_head(workspace)
    if has_head:
        raw_log = _git_output(
            workspace,
            [
                "log",
                f"--max-count={selected_limit}",
                "--pretty=format:%H%x00%h%x00%ct%x00%s%x1e",
                "--",
            ],
        )
        for record in [item for item in raw_log.split("\x1e") if item]:
            fields = record.strip("\n").split("\0")
            if len(fields) < 4:
                continue
            full_sha, short_sha, timestamp, subject = fields[:4]
            commits.append(
                {
                    "sha": _safe_git_public_text(full_sha, max_chars=64),
                    "short_sha": _safe_git_public_text(short_sha, max_chars=24),
                    "timestamp": _safe_git_public_text(timestamp, max_chars=32),
                    "subject": _safe_git_public_text(subject, max_chars=240),
                }
            )
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "branch": _git_branch_summary(workspace, route_policy),
        "commits": commits,
        "commit_count": len(commits),
        "limit": selected_limit,
        "has_head": has_head,
        "audit": _git_audit("git_log", workspace),
    }


def read_workspace_git_branch(
    workspace_id: str,
    *,
    context_token: str | None = None,
    limit: int | None = MAX_GIT_BRANCH_COUNT,
    registry: WorkspaceRegistry | None = None,
) -> dict:
    """Return bounded local branch metadata after fresh context and git policy."""

    assert_default_tools_are_no_model()
    selected_limit = _bounded_int(
        limit,
        "limit",
        default=MAX_GIT_BRANCH_COUNT,
        minimum=1,
        maximum=MAX_GIT_BRANCH_COUNT,
    )
    workspace, route_policy = _require_workspace_git_read_access(
        workspace_id,
        context_token=context_token,
        registry=registry,
    )
    current_branch = _git_current_branch(workspace)
    protected = set(route_policy.protected_branches)
    raw = _git_output(
        workspace,
        ["for-each-ref", "--format=%(refname:short)%00%(objectname:short)", "refs/heads"],
    )
    branches: list[dict] = []
    skipped: list[dict] = []
    for line in [item for item in raw.splitlines() if item.strip()]:
        fields = line.split("\0")
        name = _safe_git_public_text(fields[0], max_chars=200) if fields else ""
        short_sha = _safe_git_public_text(fields[1], max_chars=24) if len(fields) > 1 else None
        if not name:
            continue
        if name == "[REDACTED]":
            skipped.append({"reason": "secret_branch_name_denied"})
            continue
        if len(branches) >= selected_limit:
            skipped.append({"name": name, "reason": "branch_limit_exceeded"})
            continue
        branches.append(
            {
                "name": name,
                "current": name == current_branch,
                "protected": name in protected,
                "head": short_sha,
            }
        )
    return {
        "workspace_id": workspace.workspace_id,
        "profile_ref": workspace.profile_ref,
        "current": current_branch,
        "branches": branches,
        "branch_count": len(branches),
        "limit": selected_limit,
        "truncated": len(skipped) > 0,
        "skipped": skipped,
        "protected_branches": sorted(protected),
        "audit": _git_audit("git_branch", workspace),
    }


def _safe_profile_summary(
    info: ProfileInfo,
    *,
    route_policy: ProfileRoutePolicy | None = None,
    router_policy: ProfileRouterPolicy | None = None,
) -> dict:
    """Convert ``ProfileInfo`` to public, non-secret router metadata."""

    summary = {
        "profile_ref": f"{LOCAL_HOST}:{info.name}",
        "host": LOCAL_HOST,
        "profile": info.name,
        "is_default": bool(info.is_default),
        "gateway_running": bool(info.gateway_running),
        "skill_count": int(info.skill_count or 0),
        "description": info.description or "",
    }
    if route_policy is not None and router_policy is not None:
        summary["display_name"] = route_policy.display_name or info.name
        if route_policy.description:
            summary["description"] = route_policy.description
        summary["policy"] = {
            "enabled": router_policy.is_profile_exposed(route_policy),
            "allowed_tool_groups": list(route_policy.allowed_tool_groups),
            "capabilities": {
                "filesystem_read": route_policy.allow_filesystem_read,
                "filesystem_write": route_policy.allow_filesystem_write,
                "terminal": route_policy.allow_terminal,
                "messaging": route_policy.allow_messaging,
                "cron": route_policy.allow_cron,
                "git": route_policy.capability_enabled("git"),
                "git_push": route_policy.allow_git_push,
                "deploy": route_policy.allow_deploy,
                "skills": route_policy.capability_enabled("skills"),
                "memory": route_policy.capability_enabled("memory"),
                "session": route_policy.capability_enabled("session"),
                "web": route_policy.capability_enabled("web"),
                "browser": route_policy.capability_enabled("browser"),
                "api": route_policy.capability_enabled("api"),
            },
            "capability_groups": dict(route_policy.capability_groups),
            "context": {
                "skills": {"read": route_policy.allow_context_skills_read},
                "sessions": {"search": route_policy.allow_context_sessions_search},
            },
            "messaging_recipients_configured": bool(
                route_policy.messaging_allowed_recipients
            ),
            "cron_policy": {
                "enabled": route_policy.cron_policy.enabled,
                "allowed_scripts_count": len(route_policy.cron_policy.allowed_scripts),
                "allowlist_redacted": True,
                "no_agent_only": True,
                "model_backed_crons_allowed": False,
            },
            "model_policy": {
                "allow_model_tools": route_policy.allow_model_tools,
                "allowed_cost_classes": list(route_policy.allowed_cost_classes),
            },
        }
    return summary


def list_local_profiles(
    *,
    active_only: bool = True,
    policy: ProfileRouterPolicy | None = None,
) -> list[dict]:
    """Return read-only summaries for policy-enabled local profiles."""

    assert_default_tools_are_no_model()
    router_policy = policy or load_profile_router_policy()
    infos = {info.name: info for info in _list_local_profile_infos()}
    summaries: list[dict] = []
    for route_policy in router_policy.iter_profiles(host=LOCAL_HOST, active_only=active_only):
        info = infos.get(route_policy.ref.profile)
        if info is None:
            continue
        summaries.append(
            _safe_profile_summary(
                info,
                route_policy=route_policy,
                router_policy=router_policy,
            )
        )
    return summaries


def get_local_profile(profile_ref: str) -> dict:
    """Return one local profile summary, or raise ``ProfileRouterError``."""

    assert_default_tools_are_no_model()
    ref = parse_profile_ref(profile_ref)
    if ref.host != LOCAL_HOST:
        raise ProfileRouterError(
            "unsupported_host",
            f"Only local profiles are supported by this skeleton, got: {ref.host}",
        )

    router_policy = load_profile_router_policy()
    router_policy.get_profile_policy(ref)
    for profile in list_local_profiles(active_only=False, policy=router_policy):
        if profile["profile"] == ref.profile:
            return profile

    raise ProfileRouterError("profile_not_found", f"Profile not found: {ref.value}")


def _tool_envelope(tool_name: str, payload: dict) -> str:
    meta = ROUTER_TOOL_METADATA[tool_name]
    data = {
        **payload,
        "cost_class": meta.cost_class,
        "llm_calls": meta.llm_calls,
    }
    return json.dumps(data, indent=2, sort_keys=True)


def _tool_error(tool_name: str, exc: ProfileRouterError) -> str:
    return _tool_envelope(
        tool_name,
        {
            "ok": False,
            "error": {"code": exc.code, "message": exc.message},
        },
    )


def hermes_catalog_blocked_tool(tool_name: str) -> str:
    """Return a no-model blocked response for a full-catalog Hermes tool name."""

    name = str(tool_name or "").strip()
    try:
        if name not in HERMES_CATALOG_BLOCKED_TOOL_NAMES:
            raise ProfileRouterError(
                "unknown_catalog_tool",
                "Requested tool is not a blocked Hermes catalog entry",
            )
        meta = ROUTER_TOOL_METADATA[name]
        return _tool_envelope(
            name,
            {
                "ok": False,
                "error": {
                    "code": meta.blocked_reason or "requires_no_model_implementation",
                    "message": (
                        "This Hermes tool is visible for catalog parity but is not "
                        "executable through the ChatGPT no-model connector."
                    ),
                },
                "catalog_tool": {
                    "name": name,
                    "execution_status": meta.execution_status,
                    "native_side_effects": name in HERMES_CATALOG_SIDE_EFFECT_TOOL_NAMES,
                    "capability_group": _router_capability_group(name),
                    "route_hint": meta.route_hint,
                    "root_exposed": False,
                },
                "route_hint": meta.route_hint,
            },
        )
    except ProfileRouterError as exc:
        return json.dumps(
            {
                "ok": False,
                "error": {"code": exc.code, "message": exc.message},
                "cost_class": COST_CLASS_NO_MODEL,
                "llm_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )


def _require_profile_context_policy(
    profile_ref: str,
    policy_name: str,
) -> tuple[ProfileRef, ProfileRoutePolicy, ProfileRouterPolicy]:
    """Require a dedicated deny-by-default context permission for a profile."""

    ref, route_policy, router_policy = _require_local_profile_policy(profile_ref)
    if policy_name == CONTEXT_SKILLS_READ_POLICY:
        allowed = route_policy.allow_context_skills_read
        error_code = "context_skills_read_not_allowed"
    elif policy_name == CONTEXT_SESSIONS_SEARCH_POLICY:
        allowed = route_policy.allow_context_sessions_search
        error_code = "context_sessions_search_not_allowed"
    else:
        raise ProfileRouterError("invalid_context_policy", "Unknown context policy")
    if not allowed:
        raise ProfileRouterError(
            error_code,
            f"{policy_name} is disabled by profile_router policy: {ref.value}",
        )
    return ref, route_policy, router_policy


def _is_unsafe_relative_context_path(path: str) -> bool:
    if not isinstance(path, str):
        return True
    text = path.strip()
    if not text or "\x00" in text or "\\" in text or text.startswith("/"):
        return True
    raw_parts = text.split("/")
    if any(part in {"", ".", ".."} or part.startswith(".") for part in raw_parts):
        return True
    normalized = posixpath.normpath(text)
    if normalized in {"", "."} or normalized.startswith("../"):
        return True
    parts = [part for part in normalized.split("/") if part]
    return any(part in {".", ".."} or part.startswith(".") for part in parts)


def _normalize_skill_lookup_name(name: str) -> str:
    if not isinstance(name, str):
        raise ProfileRouterError("invalid_skill_name", "skill name must be a string")
    text = name.strip()
    if len(text) > 160 or _is_unsafe_relative_context_path(text) or ":" in text:
        raise ProfileRouterError("invalid_skill_name", "skill name must be a safe relative skill name")
    normalized = posixpath.normpath(text)
    if _is_secret_path(normalized):
        raise ProfileRouterError("secret_path_denied", "Skill name is blocked by the secret denylist")
    return normalized


def _normalize_skill_file_path(file_path: str) -> str:
    if not isinstance(file_path, str):
        raise ProfileRouterError("invalid_skill_file_path", "file_path must be a string")
    text = file_path.strip()
    if len(text) > 240 or _is_unsafe_relative_context_path(text):
        raise ProfileRouterError("invalid_skill_file_path", "file_path must be a safe relative path")
    normalized = posixpath.normpath(text)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) < 2 or parts[0] not in SAFE_SKILL_SUPPORT_DIRS:
        raise ProfileRouterError(
            "invalid_skill_file_path",
            "skill supporting files must live under references/, templates/, scripts/, or assets/",
        )
    if _is_secret_path(normalized):
        raise ProfileRouterError(
            "secret_path_denied",
            "Skill supporting file path is blocked by the secret denylist",
        )
    return normalized


def _yaml_scalar_text(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'\"', "'"}:
        return text[1:-1]
    return text


def _yaml_inline_list(value: str) -> list[str]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    items: list[str] = []
    for raw_item in text[1:-1].split(","):
        item = _yaml_scalar_text(raw_item)
        if item:
            items.append(item)
    return items


def _parse_skill_frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter.append(line)
    parsed: dict[str, Any] = {}
    collecting_list: str | None = None
    for line in frontmatter:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if collecting_list and stripped.startswith("-"):
            item = _yaml_scalar_text(stripped[1:].strip())
            if item:
                parsed.setdefault(collecting_list, []).append(item)
            continue
        collecting_list = None
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if key in {"name", "description"}:
            parsed[key] = _yaml_scalar_text(value)
        elif key == "tags":
            if value:
                parsed[key] = _yaml_inline_list(value)
            else:
                parsed[key] = []
                collecting_list = key
    return parsed


def _safe_skill_display_name(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 120
        or ":" in text
        or "/" in text
        or "\\" in text
        or text.startswith(".")
        or "\x00" in text
        or _is_secret_path(text)
    ):
        return fallback
    return text


def _safe_skill_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_tags = [value]
    elif isinstance(value, IterableABC) and not isinstance(value, (bytes, MappingABC)):
        raw_tags = list(value)
    else:
        raw_tags = []
    tags: list[str] = []
    for raw_tag in raw_tags:
        tag = str(raw_tag or "").strip()
        if not tag or len(tag) > 80 or "/" in tag or "\\" in tag or tag.startswith("."):
            continue
        if _is_secret_path(tag):
            continue
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= 20:
            break
    return tuple(tags)


def _read_skill_text_file(
    path: Path,
    public_path: str,
    *,
    reject_oversized: bool = False,
) -> dict:
    if _is_secret_path(public_path):
        raise ProfileRouterError(
            "secret_path_denied",
            "Skill file path is blocked by the secret denylist",
        )
    try:
        stat = path.stat()
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"Skill file is not accessible: {public_path}") from exc
    if not path.is_file():
        raise ProfileRouterError("not_a_file", f"Skill path is not a file: {public_path}")
    if reject_oversized and stat.st_size > MAX_CONTEXT_FILE_BYTES:
        raise ProfileRouterError(
            "file_too_large",
            f"Skill supporting file exceeds {MAX_CONTEXT_FILE_BYTES} bytes: {public_path}",
        )
    sha256, hash_truncated = _hash_file(path)
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_CONTEXT_FILE_BYTES + 1)
    except OSError as exc:
        raise ProfileRouterError("file_not_readable", f"Skill file is not readable: {public_path}") from exc
    if b"\x00" in raw[:4096]:
        raise ProfileRouterError("binary_file_not_supported", f"Skill file is binary: {public_path}")
    if reject_oversized and len(raw) > MAX_CONTEXT_FILE_BYTES:
        raise ProfileRouterError(
            "file_too_large",
            f"Skill supporting file exceeds {MAX_CONTEXT_FILE_BYTES} bytes: {public_path}",
        )
    decoded = raw[:MAX_CONTEXT_FILE_BYTES].decode("utf-8", errors="replace")
    truncated = len(raw) > MAX_CONTEXT_FILE_BYTES or len(decoded) > MAX_CONTEXT_FILE_CHARS
    content = _redact_context_text(decoded[:MAX_CONTEXT_FILE_CHARS])
    return {
        "path": public_path,
        "sha256": sha256,
        "hash_truncated": hash_truncated,
        "size_bytes": stat.st_size,
        "truncated": truncated,
        "content": content,
    }


def _resolve_profile_skills_root(ref: ProfileRef) -> tuple[Path, Path]:
    profile_dir = _resolve_local_profile_dir(ref)
    skills_root_path = profile_dir / "skills"
    if not skills_root_path.exists():
        return profile_dir, skills_root_path
    if skills_root_path.is_symlink():
        raise ProfileRouterError(
            "profile_skill_symlink_denied",
            f"Profile skills directory may not be a symlink: {ref.value}",
        )
    try:
        skills_root = skills_root_path.resolve(strict=True)
    except OSError as exc:
        raise ProfileRouterError("profile_skills_not_readable", "Profile skills directory is not readable") from exc
    if not _path_is_relative_to(skills_root, profile_dir):
        raise ProfileRouterError(
            "profile_skill_symlink_denied",
            f"Profile skills directory escapes profile root: {ref.value}",
        )
    return profile_dir, skills_root


def _safe_resolve_skill_child(root: Path, relative_path: str) -> Path:
    current = root
    for part in [part for part in relative_path.split("/") if part]:
        current = current / part
        if current.is_symlink():
            raise ProfileRouterError(
                "symlink_traversal_denied",
                f"Skill path uses a symlink: {relative_path}",
            )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise ProfileRouterError("file_not_found", f"Skill file not found: {relative_path}") from exc
    if not _path_is_relative_to(resolved, root):
        raise ProfileRouterError("symlink_traversal_denied", "Skill path escapes its skill directory")
    return resolved


def _list_skill_linked_files(skill_dir: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    linked: dict[str, tuple[str, ...]] = {}
    skipped = {"secret": 0, "symlink": 0, "unreadable": 0, "limit": 0}
    for top_dir in SAFE_SKILL_SUPPORT_DIRS:
        base = skill_dir / top_dir
        if not base.exists():
            continue
        if base.is_symlink():
            skipped["symlink"] += 1
            continue
        try:
            resolved_base = base.resolve(strict=True)
        except OSError:
            skipped["unreadable"] += 1
            continue
        if not _path_is_relative_to(resolved_base, skill_dir) or not resolved_base.is_dir():
            skipped["unreadable"] += 1
            continue
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(resolved_base, followlinks=False):
            current = Path(dirpath)
            safe_dirnames: list[str] = []
            for dirname in sorted(dirnames):
                candidate = current / dirname
                rel = candidate.relative_to(resolved_base).as_posix()
                public_rel = posixpath.join(top_dir, rel)
                if dirname.startswith(".") or _is_secret_path(public_rel):
                    skipped["secret"] += 1
                    continue
                if candidate.is_symlink():
                    skipped["symlink"] += 1
                    continue
                safe_dirnames.append(dirname)
            dirnames[:] = safe_dirnames
            for filename in sorted(filenames):
                candidate = current / filename
                rel = candidate.relative_to(resolved_base).as_posix()
                public_rel = posixpath.join(top_dir, rel)
                if filename.startswith(".") or _is_secret_path(public_rel):
                    skipped["secret"] += 1
                    continue
                if candidate.is_symlink():
                    skipped["symlink"] += 1
                    continue
                if len(files) >= MAX_SKILL_LINKED_FILES_PER_DIR:
                    skipped["limit"] += 1
                    continue
                files.append(rel)
        if files:
            linked[top_dir] = tuple(files)
    return linked, skipped


def _skill_record_summary(record: ProfileSkillRecord) -> dict:
    return {
        "id": record.skill_id,
        "name": record.name,
        "category": record.category,
        "description": record.description,
        "tags": list(record.tags),
        "linked_files": {key: list(value) for key, value in record.linked_files.items()},
        "content_truncated": record.content_truncated,
    }


def _scan_profile_skills(ref: ProfileRef) -> tuple[list[ProfileSkillRecord], dict[str, int]]:
    profile_dir, skills_root = _resolve_profile_skills_root(ref)
    del profile_dir
    skipped = {"secret": 0, "symlink": 0, "binary": 0, "unreadable": 0, "limit": 0}
    if not skills_root.exists():
        return [], skipped
    if not skills_root.is_dir():
        raise ProfileRouterError("profile_skills_not_readable", "Profile skills path is not a directory")
    records: list[ProfileSkillRecord] = []
    for dirpath, dirnames, filenames in os.walk(skills_root, followlinks=False):
        current = Path(dirpath)
        safe_dirnames: list[str] = []
        for dirname in sorted(dirnames):
            candidate = current / dirname
            rel = candidate.relative_to(skills_root).as_posix()
            if dirname.startswith(".") or _is_secret_path(rel):
                skipped["secret"] += 1
                continue
            if candidate.is_symlink():
                skipped["symlink"] += 1
                continue
            safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames
        if "SKILL.md" not in filenames:
            continue
        public_id = current.relative_to(skills_root).as_posix()
        if public_id in {"", "."} or _is_secret_path(public_id):
            skipped["secret"] += 1
            dirnames[:] = []
            continue
        skill_file = current / "SKILL.md"
        if skill_file.is_symlink():
            skipped["symlink"] += 1
            dirnames[:] = []
            continue
        try:
            resolved_skill_file = skill_file.resolve(strict=True)
        except OSError:
            skipped["unreadable"] += 1
            dirnames[:] = []
            continue
        if not _path_is_relative_to(resolved_skill_file, skills_root):
            skipped["symlink"] += 1
            dirnames[:] = []
            continue
        try:
            file_info = _read_skill_text_file(resolved_skill_file, "SKILL.md")
        except ProfileRouterError as exc:
            if exc.code == "binary_file_not_supported":
                skipped["binary"] += 1
            else:
                skipped["unreadable"] += 1
            dirnames[:] = []
            continue
        frontmatter = _parse_skill_frontmatter(file_info["content"])
        fallback_name = current.name
        display_name = _safe_skill_display_name(frontmatter.get("name"), fallback_name)
        description = _redact_context_text(str(frontmatter.get("description") or "")).strip()
        if len(description) > MAX_SKILL_DESCRIPTION_CHARS:
            description = description[:MAX_SKILL_DESCRIPTION_CHARS]
        linked_files, linked_skipped = _list_skill_linked_files(current)
        for key, value in linked_skipped.items():
            skipped[key] = skipped.get(key, 0) + value
        category = posixpath.dirname(public_id)
        records.append(
            ProfileSkillRecord(
                skill_id=public_id,
                name=display_name,
                category=category,
                description=description,
                tags=_safe_skill_tags(frontmatter.get("tags")),
                skill_dir=current,
                skill_file=resolved_skill_file,
                linked_files=linked_files,
                content_truncated=bool(file_info["truncated"]),
            )
        )
        dirnames[:] = []
    return sorted(records, key=lambda record: record.skill_id), skipped


def _find_profile_skill(ref: ProfileRef, name: str) -> tuple[ProfileSkillRecord, dict[str, int]]:
    lookup = _normalize_skill_lookup_name(name)
    profile_dir, skills_root = _resolve_profile_skills_root(ref)
    del profile_dir
    if "/" in lookup:
        candidate_dir = skills_root / lookup
        if candidate_dir.is_symlink():
            raise ProfileRouterError(
                "profile_skill_symlink_denied",
                f"Profile skill directory may not be a symlink: {lookup}",
            )
        try:
            resolved_candidate_dir = candidate_dir.resolve(strict=True)
        except OSError as exc:
            raise ProfileRouterError("skill_not_found", f"Skill not found: {lookup}") from exc
        if not _path_is_relative_to(resolved_candidate_dir, skills_root):
            raise ProfileRouterError("profile_skill_symlink_denied", "Profile skill directory escapes skills root")
        if not (resolved_candidate_dir / "SKILL.md").exists():
            raise ProfileRouterError("skill_not_found", f"Skill not found: {lookup}")
    else:
        direct_candidate_dir = skills_root / lookup
        if direct_candidate_dir.is_symlink():
            raise ProfileRouterError(
                "profile_skill_symlink_denied",
                f"Profile skill directory may not be a symlink: {lookup}",
            )
    records, skipped = _scan_profile_skills(ref)
    matches = [
        record
        for record in records
        if record.skill_id == lookup or record.name == lookup or posixpath.basename(record.skill_id) == lookup
    ]
    if not matches:
        raise ProfileRouterError("skill_not_found", f"Skill not found: {lookup}")
    if len(matches) > 1:
        raise ProfileRouterError("skill_ambiguous", "Skill name is ambiguous; use the returned skill id")
    return matches[0], skipped


def list_profile_skills(profile_ref: str, *, limit: int | None = MAX_SKILLS_LIST_RESULTS) -> dict:
    assert_default_tools_are_no_model()
    ref, _route_policy, _router_policy = _require_profile_context_policy(
        profile_ref,
        CONTEXT_SKILLS_READ_POLICY,
    )
    max_results = _bounded_int(
        limit,
        "limit",
        default=MAX_SKILLS_LIST_RESULTS,
        minimum=1,
        maximum=MAX_SKILLS_LIST_RESULTS,
    )
    records, skipped = _scan_profile_skills(ref)
    selected = records[:max_results]
    return {
        "profile_ref": ref.value,
        "skills": [_skill_record_summary(record) for record in selected],
        "total_count": len(records),
        "count": len(selected),
        "limit": max_results,
        "truncated": len(records) > max_results,
        "skipped": skipped,
        "audit": {"tool": "skills_list", "llm_calls": 0, "root_exposed": False},
    }


def view_profile_skill(profile_ref: str, name: str, *, file_path: str | None = None) -> dict:
    assert_default_tools_are_no_model()
    ref, _route_policy, _router_policy = _require_profile_context_policy(
        profile_ref,
        CONTEXT_SKILLS_READ_POLICY,
    )
    record, skipped = _find_profile_skill(ref, name)
    if file_path is None:
        file_info = _read_skill_text_file(record.skill_file, "SKILL.md")
    else:
        normalized_file_path = _normalize_skill_file_path(file_path)
        resolved_file = _safe_resolve_skill_child(record.skill_dir, normalized_file_path)
        file_info = _read_skill_text_file(
            resolved_file,
            normalized_file_path,
            reject_oversized=True,
        )
    return {
        "profile_ref": ref.value,
        "skill": _skill_record_summary(record),
        "file": file_info,
        "skipped": skipped,
        "audit": {"tool": "skill_view", "llm_calls": 0, "root_exposed": False},
    }


def _normalize_session_search_query(query: str | None) -> str | None:
    if query is None:
        return None
    if not isinstance(query, str):
        raise ProfileRouterError("invalid_session_search_query", "query must be a string")
    text = query.strip()
    if not text:
        return None
    if len(text) > MAX_SESSION_SEARCH_QUERY_CHARS:
        raise ProfileRouterError(
            "invalid_session_search_query",
            f"query must be <= {MAX_SESSION_SEARCH_QUERY_CHARS} characters",
        )
    return text


def _normalize_session_search_sort(sort: str | None) -> str:
    if sort is None:
        return "newest"
    if not isinstance(sort, str):
        raise ProfileRouterError("invalid_session_search_sort", "sort must be newest or oldest")
    normalized = sort.strip().lower() or "newest"
    if normalized not in {"newest", "oldest"}:
        raise ProfileRouterError("invalid_session_search_sort", "sort must be newest or oldest")
    return normalized


def _escape_sql_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _redact_context_text(text: str, roots: Iterable[Path | str] = ()) -> str:
    redacted = _redact_sensitive_text_fields(str(text or ""))
    root_texts = sorted(
        {
            str(root)
            for root in roots
            if root is not None and len(str(root)) > 3
        },
        key=len,
        reverse=True,
    )
    for root_text in root_texts:
        redacted = redacted.replace(root_text, "[REDACTED_PATH]")
    redacted = HOST_ROOT_PATH_RE.sub("[REDACTED_PATH]", redacted)
    return re.sub(r"\s+", " ", redacted).strip()


def _redact_session_text(text: str, roots: Iterable[Path | str]) -> str:
    return _redact_context_text(text, roots)


def _session_snippet(content: Any, *, query: str | None, roots: Iterable[Path | str]) -> str:
    if isinstance(content, str):
        text = content
    else:
        try:
            text = json.dumps(content, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(content or "")
    redacted = _redact_session_text(text, roots)
    if len(redacted) <= MAX_SESSION_SNIPPET_CHARS:
        return redacted
    if query:
        index = redacted.lower().find(query.lower())
        if index >= 0:
            start = max(0, index - (MAX_SESSION_SNIPPET_CHARS // 3))
            end = min(len(redacted), start + MAX_SESSION_SNIPPET_CHARS)
            start = max(0, end - MAX_SESSION_SNIPPET_CHARS)
            return redacted[start:end]
    return redacted[:MAX_SESSION_SNIPPET_CHARS]


def _resolve_profile_session_db(ref: ProfileRef) -> tuple[Path, Path | None]:
    profile_dir = _resolve_local_profile_dir(ref)
    state_db_path = profile_dir / "state.db"
    if not state_db_path.exists():
        return profile_dir, None
    if state_db_path.is_symlink():
        raise ProfileRouterError(
            "profile_session_db_symlink_denied",
            f"Profile session database may not be a symlink: {ref.value}",
        )
    try:
        resolved_state_db = state_db_path.resolve(strict=True)
    except OSError as exc:
        raise ProfileRouterError("session_db_unavailable", "Profile session database is not readable") from exc
    if not _path_is_relative_to(resolved_state_db, profile_dir):
        raise ProfileRouterError(
            "profile_session_db_symlink_denied",
            "Profile session database escapes profile root",
        )
    if not resolved_state_db.is_file():
        raise ProfileRouterError("session_db_unavailable", "Profile session database is not a file")
    return profile_dir, resolved_state_db


def search_profile_sessions(
    profile_ref: str,
    *,
    query: str | None = None,
    limit: int | None = 3,
    sort: str | None = None,
) -> dict:
    assert_default_tools_are_no_model()
    ref, route_policy, _router_policy = _require_profile_context_policy(
        profile_ref,
        CONTEXT_SESSIONS_SEARCH_POLICY,
    )
    normalized_query = _normalize_session_search_query(query)
    normalized_sort = _normalize_session_search_sort(sort)
    max_results = _bounded_int(
        limit,
        "limit",
        default=3,
        minimum=1,
        maximum=MAX_SESSION_SEARCH_RESULTS,
    )
    profile_dir, state_db_path = _resolve_profile_session_db(ref)
    audit = {
        "tool": "session_search",
        "llm_calls": 0,
        "root_exposed": False,
        "state_db_read_only": True,
        "roles": ["user", "assistant"],
    }
    if state_db_path is None:
        return {
            "profile_ref": ref.value,
            "state_db_present": False,
            "query_supplied": normalized_query is not None,
            "sort": normalized_sort,
            "roles": ["user", "assistant"],
            "results": [],
            "count": 0,
            "limit": max_results,
            "truncated": False,
            "audit": audit,
        }

    where = ["m.role IN (?, ?)", "COALESCE(m.active, 1) = 1"]
    params: list[Any] = ["user", "assistant"]
    if normalized_query is not None:
        where.append("COALESCE(m.content, '') LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_sql_like(normalized_query)}%")
    order = "ASC" if normalized_sort == "oldest" else "DESC"
    params.append(max_results + 1)
    sql = f"""
        SELECT
            m.session_id,
            m.role,
            COALESCE(m.content, '') AS content,
            m.timestamp,
            s.source,
            s.started_at AS session_started,
            s.title
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE {' AND '.join(where)}
        ORDER BY m.timestamp {order}, m.id {order}
        LIMIT ?
    """
    fts_sql = f"""
        SELECT
            m.session_id,
            m.role,
            COALESCE(m.content, '') AS content,
            m.timestamp,
            s.source,
            s.started_at AS session_started,
            s.title
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN sessions s ON s.id = m.session_id
        WHERE messages_fts MATCH ?
          AND m.role IN (?, ?)
          AND COALESCE(m.active, 1) = 1
        ORDER BY m.timestamp {order}, m.id {order}
        LIMIT ?
    """
    roots: list[Path | str] = [profile_dir, *route_policy.allowed_roots]
    try:
        conn = sqlite3.connect(
            f"file:{state_db_path}?mode=ro",
            uri=True,
            timeout=1.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            rows = None
            if normalized_query is not None:
                fts_present = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'messages_fts' LIMIT 1"
                ).fetchone()
                if fts_present is not None:
                    fts_query = '"' + normalized_query.replace('"', '""') + '"'
                    try:
                        rows = conn.execute(
                            fts_sql,
                            [fts_query, "user", "assistant", max_results + 1],
                        ).fetchall()
                    except sqlite3.DatabaseError:
                        rows = None
            if rows is None:
                rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise ProfileRouterError("session_db_unavailable", "Profile session database is not readable") from exc

    selected_rows = rows[:max_results]
    results = []
    for row in selected_rows:
        title = _redact_session_text(row["title"] or "", roots)[:120]
        results.append(
            {
                "session_id_hash": hashlib.sha256(
                    str(row["session_id"] or "").encode("utf-8", errors="replace")
                ).hexdigest()[:16],
                "role": str(row["role"] or ""),
                "snippet": _session_snippet(
                    row["content"],
                    query=normalized_query,
                    roots=roots,
                ),
                "timestamp": row["timestamp"],
                "session": {
                    "source": str(row["source"] or ""),
                    "started_at": row["session_started"],
                    "title": title,
                },
            }
        )
    return {
        "profile_ref": ref.value,
        "state_db_present": True,
        "query_supplied": normalized_query is not None,
        "sort": normalized_sort,
        "roles": ["user", "assistant"],
        "results": results,
        "count": len(results),
        "limit": max_results,
        "truncated": len(rows) > max_results,
        "audit": audit,
    }


def _raise_context_tool_not_implemented(tool_name: str) -> NoReturn:
    raise ProfileRouterError(
        "tool_not_implemented",
        (
            f"{tool_name} has Phase 8 metadata and policy gates, but its bounded "
            "reader is not implemented yet"
        ),
    )


def _require_global_viking_context_policy() -> dict:
    """Require at least one exposed local route to opt into global OpenViking read.

    OpenViking is intentionally treated as a global memory surface for this
    phase, but it still fails closed unless the operator enables the dedicated
    deny-by-default ``context.viking.read`` flag on an exposed local profile.
    """

    router_policy = load_profile_router_policy()
    if not router_policy.allow_global_viking_read:
        raise ProfileRouterError(
            "context_viking_read_not_allowed",
            "global profile_router.context.viking.read is disabled",
        )
    authorized_routes = [
        policy
        for policy in router_policy.iter_profiles(active_only=True)
        if policy.ref.host == LOCAL_HOST and _local_profile_exists(policy.ref.profile)
    ]
    if not authorized_routes:
        raise ProfileRouterError(
            "context_viking_read_not_allowed",
            "global OpenViking read requires at least one exposed local profile_router profile",
        )
    return {
        "type": "global",
        "authorized_profiles_count": len(authorized_routes),
        "profile_refs_exposed": False,
    }


def _validate_openviking_endpoint(endpoint: str) -> str:
    text = str(endpoint or "").strip().rstrip("/")
    parsed = urlparse(text)
    if (
        not text
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ProfileRouterError(
            "openviking_endpoint_invalid",
            "OpenViking endpoint must be a server-configured local/private origin",
        )

    hostname = parsed.hostname
    if hostname is None:
        raise ProfileRouterError(
            "openviking_endpoint_invalid",
            "OpenViking endpoint must be a server-configured local/private origin",
        )
    host_lower = hostname.lower()
    is_private = host_lower == "localhost"
    if not is_private:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ProfileRouterError(
                "openviking_endpoint_not_private",
                "OpenViking endpoint host must be localhost or a private IP address",
            ) from exc
        is_private = bool(address.is_loopback or address.is_private or address.is_link_local)
    if not is_private:
        raise ProfileRouterError(
            "openviking_endpoint_not_private",
            "OpenViking endpoint host must be localhost or a private IP address",
        )
    return text


def _openviking_endpoint() -> str:
    endpoint = os.environ.get("OPENVIKING_ENDPOINT")
    if not endpoint or not endpoint.strip():
        raise ProfileRouterError(
            "openviking_endpoint_unconfigured",
            "OPENVIKING_ENDPOINT must be explicitly configured server-side",
        )
    return _validate_openviking_endpoint(endpoint)


def _openviking_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-OpenViking-Agent": os.environ.get("OPENVIKING_AGENT", "hermes") or "hermes",
        "X-OpenViking-Account": os.environ.get("OPENVIKING_ACCOUNT", "default") or "default",
        "X-OpenViking-User": os.environ.get("OPENVIKING_USER", "default") or "default",
    }
    api_key = os.environ.get("OPENVIKING_API_KEY", "")
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _openviking_request_json(
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict:
    if not path.startswith("/api/v1/") and path != "/health":
        raise ProfileRouterError("openviking_request_invalid", "Invalid OpenViking API path")
    endpoint = _openviking_endpoint()
    url = f"{endpoint}{path}"
    if params:
        safe_params = {key: value for key, value in params.items() if value is not None}
        if safe_params:
            url = f"{url}?{urlencode(safe_params)}"
    method_upper = method.upper()
    data = None
    if method_upper == "POST":
        data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers=_openviking_headers(), method=method_upper)
    try:
        with urlopen(request, timeout=OPENVIKING_REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_VIKING_HTTP_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ProfileRouterError(
            "openviking_request_failed",
            "OpenViking request failed without exposing endpoint, headers, or query text",
        ) from exc
    if len(raw) > MAX_VIKING_HTTP_RESPONSE_BYTES:
        raise ProfileRouterError(
            "openviking_response_too_large",
            "OpenViking response exceeded the profile-router bound",
        )
    try:
        decoded = raw.decode("utf-8")
        data_obj = json.loads(decoded) if decoded else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileRouterError(
            "openviking_response_invalid",
            "OpenViking response was not valid bounded JSON",
        ) from exc
    if not isinstance(data_obj, dict):
        raise ProfileRouterError(
            "openviking_response_invalid",
            "OpenViking response must be a JSON object",
        )
    return data_obj


def _unwrap_openviking_result(payload: Any) -> Any:
    if isinstance(payload, MappingABC) and "result" in payload:
        return payload.get("result")
    return payload


def _redact_viking_text(text: Any) -> str:
    return _redact_context_text(str(text or ""))


def _bounded_viking_text(text: Any, max_chars: int) -> tuple[str, bool]:
    redacted = _redact_viking_text(text)
    if len(redacted) <= max_chars:
        return redacted, False
    return redacted[:max_chars], True


def _normalize_viking_query(query: str) -> str:
    if not isinstance(query, str):
        raise ProfileRouterError("invalid_viking_query", "query must be a string")
    text = query.strip()
    if not text:
        raise ProfileRouterError("invalid_viking_query", "query is required")
    if len(text) > MAX_VIKING_QUERY_CHARS:
        raise ProfileRouterError(
            "invalid_viking_query",
            f"query must be <= {MAX_VIKING_QUERY_CHARS} characters",
        )
    return text


def _normalize_viking_search_mode(mode: str | None) -> str:
    if mode is None:
        return "auto"
    if not isinstance(mode, str):
        raise ProfileRouterError("invalid_viking_mode", "mode must be auto, fast, or deep")
    normalized = mode.strip().lower() or "auto"
    if normalized not in OPENVIKING_ALLOWED_SEARCH_MODES:
        raise ProfileRouterError("invalid_viking_mode", "mode must be auto, fast, or deep")
    return normalized


def _normalize_viking_read_level(level: str | None) -> str:
    if level is None:
        return "overview"
    if not isinstance(level, str):
        raise ProfileRouterError("invalid_viking_level", "level must be abstract, overview, or full")
    normalized = level.strip().lower() or "overview"
    if normalized not in OPENVIKING_ALLOWED_READ_LEVELS:
        raise ProfileRouterError(
            "invalid_viking_level", "level must be abstract, overview, or full"
        )
    return normalized


def _validate_viking_uri(uri: str, *, allow_pseudo_summary: bool = True) -> str:
    if not isinstance(uri, str):
        raise ProfileRouterError("invalid_viking_uri", "uri must be a viking:// URI")
    text = uri.strip()
    if (
        not text
        or len(text) > MAX_VIKING_URI_CHARS
        or "\x00" in text
        or "\n" in text
        or "\r" in text
        or "\\" in text
    ):
        raise ProfileRouterError("invalid_viking_uri", "uri must be a safe viking:// URI")
    parsed = urlparse(text)
    if (
        parsed.scheme != "viking"
        or not parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ProfileRouterError("invalid_viking_uri", "uri must be a viking:// URI without query or fragment")
    raw_parts = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    parts: list[str] = []
    for raw_part in raw_parts:
        try:
            part = unquote(raw_part, encoding="utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProfileRouterError("invalid_viking_uri", "OpenViking URI encoding is invalid") from exc
        if (
            not part
            or "%" in part
            or "/" in part
            or "\\" in part
            or "\x00" in part
            or part.strip() != part
        ):
            raise ProfileRouterError("invalid_viking_uri", "OpenViking URI path segment is invalid")
        parts.append(part)
    if len(parts) < 2:
        raise ProfileRouterError("invalid_viking_uri", "broad OpenViking roots are not exposed")
    for index, part in enumerate(parts):
        if part in {"", ".", ".."} or ".." in part:
            raise ProfileRouterError("invalid_viking_uri", "OpenViking URI traversal is not allowed")
        is_allowed_pseudo = allow_pseudo_summary and index == len(parts) - 1 and part in OPENVIKING_PSEUDO_SUMMARY_FILES
        if part.startswith(".") and not is_allowed_pseudo:
            raise ProfileRouterError("invalid_viking_uri", "hidden OpenViking paths are not exposed")
    normalized_path = "/".join(parts)
    if _is_secret_path(normalized_path):
        raise ProfileRouterError("secret_path_denied", "OpenViking URI is blocked by the secret denylist")
    if re.search(r"(?i)(^|/)(users|home|etc|var|tmp)(/|$)", normalized_path):
        raise ProfileRouterError("invalid_viking_uri", "host-root-looking OpenViking URIs are not exposed")
    return text


def _normalize_viking_summary_uri(uri: str) -> str:
    for suffix in OPENVIKING_PSEUDO_SUMMARY_FILES:
        if uri.endswith("/" + suffix):
            return uri[: -len(suffix) - 1] or "viking://"
    return uri


def _openviking_is_directory_uri(uri: str) -> bool | None:
    try:
        payload = _openviking_request_json("GET", "/api/v1/fs/stat", params={"uri": uri})
    except ProfileRouterError:
        return None
    result = _unwrap_openviking_result(payload)
    if isinstance(result, MappingABC):
        if "isDir" in result:
            return bool(result.get("isDir"))
        if "is_dir" in result:
            return bool(result.get("is_dir"))
        if result.get("type") == "dir":
            return True
        if result.get("type") == "file":
            return False
    return None


def _extract_openviking_search_items(payload: Any) -> tuple[list[tuple[float, dict]], int]:
    result = _unwrap_openviking_result(payload)
    scored_entries: list[tuple[float, dict]] = []
    skipped = 0
    if not isinstance(result, MappingABC):
        return [], 0

    bucket_specs: list[tuple[str, Any]] = []
    for bucket_name in ("memories", "resources", "skills"):
        bucket_specs.append((bucket_name.rstrip("s"), result.get(bucket_name, [])))
    if not any(bucket for _bucket_type, bucket in bucket_specs):
        bucket_specs = [("item", result.get("results") or result.get("items") or [])]

    for bucket_type, items in bucket_specs:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, MappingABC):
                skipped += 1
                continue
            raw_uri = str(item.get("uri") or "")
            try:
                safe_uri = _validate_viking_uri(raw_uri)
            except ProfileRouterError:
                skipped += 1
                continue
            raw_score = item.get("score")
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            abstract, abstract_truncated = _bounded_viking_text(
                item.get("abstract") or item.get("summary") or item.get("content") or "",
                MAX_VIKING_ABSTRACT_CHARS,
            )
            entry = {
                "uri": safe_uri,
                "type": str(item.get("type") or bucket_type),
                "score": round(score, 3),
                "abstract": abstract,
                "abstract_truncated": abstract_truncated,
            }
            related = []
            for relation in item.get("relations") or item.get("related") or []:
                if not isinstance(relation, MappingABC):
                    continue
                try:
                    related.append(_validate_viking_uri(str(relation.get("uri") or "")))
                except ProfileRouterError:
                    skipped += 1
                if len(related) >= 3:
                    break
            if related:
                entry["related"] = related
            scored_entries.append((score, entry))
    return scored_entries, skipped


def search_openviking_context(
    query: str,
    *,
    mode: str = "auto",
    scope: str | None = None,
    limit: int | None = 10,
) -> dict:
    assert_default_tools_are_no_model()
    policy_scope = _require_global_viking_context_policy()
    normalized_query = _normalize_viking_query(query)
    normalized_mode = _normalize_viking_search_mode(mode)
    max_results = _bounded_int(
        limit,
        "limit",
        default=10,
        minimum=1,
        maximum=MAX_VIKING_SEARCH_RESULTS,
    )
    normalized_scope = _validate_viking_uri(scope, allow_pseudo_summary=False) if scope else None
    payload: dict[str, Any] = {"query": normalized_query, "limit": max_results}
    if normalized_mode != "auto":
        payload["mode"] = normalized_mode
    if normalized_scope:
        payload["target_uri"] = normalized_scope

    request_shape = "limit"
    try:
        response_payload = _openviking_request_json(
            "POST", "/api/v1/search/find", payload=payload
        )
    except ProfileRouterError as first_error:
        fallback_payload = dict(payload)
        fallback_payload.pop("limit", None)
        fallback_payload["top_k"] = max_results
        request_shape = "top_k_fallback"
        try:
            response_payload = _openviking_request_json(
                "POST", "/api/v1/search/find", payload=fallback_payload
            )
        except ProfileRouterError:
            raise first_error

    scored_entries, skipped = _extract_openviking_search_items(response_payload)
    scored_entries.sort(key=lambda item: item[0], reverse=True)
    selected = [entry for _score, entry in scored_entries[:max_results]]
    result_obj = _unwrap_openviking_result(response_payload)
    total = len(scored_entries)
    if isinstance(result_obj, MappingABC):
        raw_total = result_obj.get("total")
        if isinstance(raw_total, int) and raw_total >= total:
            total = raw_total
    return {
        "policy_scope": policy_scope,
        "endpoint": {"server_configured": True, "local_private": True, "url_exposed": False},
        "query_supplied": True,
        "mode": normalized_mode,
        "scope": normalized_scope,
        "results": selected,
        "count": len(selected),
        "limit": max_results,
        "total": total,
        "truncated": total > len(selected),
        "skipped": skipped,
        "request_shape": request_shape,
        "audit": {
            "tool": "viking_search",
            "llm_calls": 0,
            "root_exposed": False,
            "endpoint_local_private": True,
            "raw_query_logged": False,
            "url_exposed": False,
        },
    }


def read_openviking_context(uri: str, *, level: str = "overview") -> dict:
    assert_default_tools_are_no_model()
    policy_scope = _require_global_viking_context_policy()
    normalized_uri = _validate_viking_uri(uri)
    normalized_level = _normalize_viking_read_level(level)
    summary_level = normalized_level in {"abstract", "overview"}
    resolved_uri = _normalize_viking_summary_uri(normalized_uri) if summary_level else normalized_uri
    used_fallback = False

    if summary_level and resolved_uri == normalized_uri:
        is_dir = _openviking_is_directory_uri(normalized_uri)
        if is_dir is False:
            used_fallback = True

    endpoint = "/api/v1/content/read"
    if not used_fallback:
        if normalized_level == "abstract":
            endpoint = "/api/v1/content/abstract"
        elif normalized_level == "overview":
            endpoint = "/api/v1/content/overview"
    try:
        response_payload = _openviking_request_json(
            "GET", endpoint, params={"uri": resolved_uri}
        )
    except ProfileRouterError:
        if not summary_level or resolved_uri != normalized_uri or used_fallback:
            raise
        endpoint = "/api/v1/content/read"
        resolved_uri = normalized_uri
        used_fallback = True
        response_payload = _openviking_request_json(
            "GET", endpoint, params={"uri": resolved_uri}
        )

    result = _unwrap_openviking_result(response_payload)
    if isinstance(result, str):
        content_value = result
    elif isinstance(result, MappingABC):
        content_value = result.get("content") or result.get("text") or result.get("abstract") or ""
    else:
        content_value = ""
    cap = MAX_VIKING_FULL_CHARS
    if normalized_level == "overview":
        cap = MAX_VIKING_OVERVIEW_CHARS
    elif normalized_level == "abstract":
        cap = MAX_VIKING_ABSTRACT_CHARS
    content, truncated = _bounded_viking_text(content_value, cap)
    return {
        "policy_scope": policy_scope,
        "endpoint": {"server_configured": True, "local_private": True, "url_exposed": False},
        "uri": normalized_uri,
        "resolved_uri": resolved_uri,
        "level": normalized_level,
        "content": content,
        "truncated": truncated,
        "max_chars": cap,
        "fallback": "content/read" if used_fallback else None,
        "audit": {
            "tool": "viking_read",
            "llm_calls": 0,
            "root_exposed": False,
            "endpoint_local_private": True,
            "url_exposed": False,
        },
    }


def _reject_disabled_powerful_tool_after_context(
    tool_name: str,
    workspace_id: str,
    *,
    context_token: str | None = None,
) -> NoReturn:
    """Require fresh context before rejecting a not-yet-exposed powerful tool.

    The profile router deliberately keeps write/patch/terminal execution disabled
    in the public MCP surface. These direct wrappers still enforce the Phase 4.5
    ordering contract now: stale or missing SOUL/AGENTS context fails closed
    before any future implementation could touch the filesystem or terminal.
    """

    require_fresh_workspace_context(workspace_id, context_token=context_token)
    raise ProfileRouterError(
        "tool_disabled",
        (
            f"{tool_name} is disabled until explicit no-model policy, context, "
            "containment, audit, and focused tests are implemented"
        ),
    )


def profiles_list(active_only: bool = True) -> str:
    """MCP-ready wrapper: list local profiles without invoking any model."""

    try:
        return _tool_envelope(
            "profiles_list",
            {"ok": True, "profiles": list_local_profiles(active_only=active_only)},
        )
    except ProfileRouterError as exc:
        return _tool_error("profiles_list", exc)


def profile_get(profile_ref: str) -> str:
    """MCP-ready wrapper: get one local profile without invoking any model."""

    try:
        return _tool_envelope(
            "profile_get",
            {"ok": True, "profile": get_local_profile(profile_ref)},
        )
    except ProfileRouterError as exc:
        return _tool_error("profile_get", exc)


def profile_health(profile_ref: str) -> str:
    """MCP-ready wrapper: health summary for one local profile."""

    try:
        profile = get_local_profile(profile_ref)
        return _tool_envelope(
            "profile_health",
            {
                "ok": True,
                "profile_ref": profile["profile_ref"],
                "health": {
                    "status": "ok",
                    "host": profile["host"],
                    "profile": profile["profile"],
                    "gateway_running": profile["gateway_running"],
                },
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("profile_health", exc)


def workspace_open(profile_ref: str, root: str, mode: str = "checkout") -> str:
    """MCP-ready wrapper: open a read-only workspace without invoking a model."""

    try:
        workspace = open_workspace(profile_ref, root, mode=mode)
        return _tool_envelope(
            "workspace_open",
            {
                "ok": True,
                "workspace": _public_workspace_dict(workspace),
                "context": {
                    "required_before_powerful_tools": True,
                    "state": "not_loaded",
                    "next_tool": "workspace_instructions_get",
                },
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_open", exc)


def profile_context_get(profile_ref: str) -> str:
    """MCP-ready wrapper: load profile SOUL/policy context with no LLM calls."""

    try:
        return _tool_envelope(
            "profile_context_get",
            {"ok": True, "context": build_profile_context(profile_ref)},
        )
    except ProfileRouterError as exc:
        return _tool_error("profile_context_get", exc)


def skills_list(profile_ref: str, limit: int | None = MAX_SKILLS_LIST_RESULTS) -> str:
    """MCP-ready wrapper: list bounded profile skills without invoking a model."""

    try:
        return _tool_envelope(
            "skills_list",
            {"ok": True, **list_profile_skills(profile_ref, limit=limit)},
        )
    except ProfileRouterError as exc:
        return _tool_error("skills_list", exc)


def skill_view(profile_ref: str, name: str, file_path: str | None = None) -> str:
    """MCP-ready wrapper: view a bounded profile skill file without a model."""

    try:
        return _tool_envelope(
            "skill_view",
            {"ok": True, **view_profile_skill(profile_ref, name, file_path=file_path)},
        )
    except ProfileRouterError as exc:
        return _tool_error("skill_view", exc)


def session_search(
    profile_ref: str,
    query: str | None = None,
    limit: int | None = 3,
    sort: str | None = None,
) -> str:
    """MCP-ready wrapper: search bounded profile session snippets without a model."""

    try:
        return _tool_envelope(
            "session_search",
            {
                "ok": True,
                **search_profile_sessions(
                    profile_ref,
                    query=query,
                    limit=limit,
                    sort=sort,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("session_search", exc)


def viking_search(
    query: str,
    mode: str = "auto",
    scope: str | None = None,
    limit: int | None = 10,
) -> str:
    """MCP-ready wrapper: search local/private OpenViking context without a model."""

    try:
        return _tool_envelope(
            "viking_search",
            {
                "ok": True,
                **search_openviking_context(
                    query,
                    mode=mode,
                    scope=scope,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("viking_search", exc)


def viking_read(uri: str, level: str = "overview") -> str:
    """MCP-ready wrapper: read local/private OpenViking context without a model."""

    try:
        return _tool_envelope(
            "viking_read",
            {"ok": True, **read_openviking_context(uri, level=level)},
        )
    except ProfileRouterError as exc:
        return _tool_error("viking_read", exc)


def workspace_instructions_get(workspace_id: str) -> str:
    """MCP-ready wrapper: hydrate workspace instructions and record freshness."""

    try:
        return _tool_envelope(
            "workspace_instructions_get",
            {"ok": True, "context": hydrate_workspace_context(workspace_id)},
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_instructions_get", exc)


def workspace_context_status(workspace_id: str) -> str:
    """MCP-ready wrapper: report whether workspace context is loaded/stale."""

    try:
        return _tool_envelope(
            "workspace_context_status",
            {"ok": True, "context_status": get_workspace_context_status(workspace_id)},
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_context_status", exc)


def workspace_get(workspace_id: str) -> str:
    """MCP-ready wrapper: inspect an opened workspace by opaque ID."""

    try:
        workspace = get_workspace(workspace_id)
        return _tool_envelope(
            "workspace_get",
            {"ok": True, "workspace": _public_workspace_dict(workspace)},
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_get", exc)


def workspace_close(workspace_id: str) -> str:
    """MCP-ready wrapper: close an opened workspace registry entry."""

    try:
        workspace = close_workspace(workspace_id)
        return _tool_envelope(
            "workspace_close",
            {
                "ok": True,
                "closed": True,
                "workspace": _public_workspace_dict(workspace),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_close", exc)


def workspace_file_list(
    workspace_id: str,
    path: str | None = None,
    file_glob: str | None = None,
    limit: int | None = MAX_FILE_LIST_RESULTS,
    context_token: str | None = None,
) -> str:
    """MCP-ready wrapper: list bounded sanitized files after context hydration."""

    try:
        return _tool_envelope(
            "workspace_file_list",
            {
                "ok": True,
                "file_list": list_workspace_files(
                    workspace_id,
                    path=path,
                    file_glob=file_glob,
                    limit=limit,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_file_list", exc)


def workspace_file_read(
    workspace_id: str,
    path: str,
    offset: int | None = 1,
    limit: int | None = MAX_FILE_READ_LINES,
    context_token: str | None = None,
) -> str:
    """MCP-ready wrapper: read a bounded sanitized file slice after context hydration."""

    try:
        require_fresh_workspace_context(workspace_id, context_token=context_token)
        return _tool_envelope(
            "workspace_file_read",
            {
                "ok": True,
                "file": read_workspace_file(
                    workspace_id,
                    path,
                    offset=offset,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_file_read", exc)


def workspace_file_stat(
    workspace_id: str,
    path: str,
    context_token: str | None = None,
) -> str:
    """MCP-ready wrapper: stat a bounded sanitized path after context hydration."""

    try:
        return _tool_envelope(
            "workspace_file_stat",
            {
                "ok": True,
                "stat": stat_workspace_file(
                    workspace_id,
                    path,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_file_stat", exc)


def workspace_file_search(
    workspace_id: str,
    pattern: str,
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "content",
    limit: int | None = MAX_FILE_SEARCH_RESULTS,
    context_token: str | None = None,
) -> str:
    """MCP-ready wrapper: search bounded text files after context hydration."""

    try:
        require_fresh_workspace_context(workspace_id, context_token=context_token)
        return _tool_envelope(
            "workspace_file_search",
            {
                "ok": True,
                "search": search_workspace_files(
                    workspace_id,
                    pattern,
                    path=path,
                    file_glob=file_glob,
                    output_mode=output_mode,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_file_search", exc)


def file_read(
    workspace_id: str,
    path: str,
    offset: int | None = 1,
    limit: int | None = MAX_FILE_READ_LINES,
    context_token: str | None = None,
) -> str:
    """Legacy direct wrapper: read a bounded text slice from a workspace."""

    try:
        require_fresh_workspace_context(workspace_id, context_token=context_token)
        return _tool_envelope(
            "file_read",
            {
                "ok": True,
                "file": read_workspace_file(
                    workspace_id,
                    path,
                    offset=offset,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_read", exc)


def file_search(
    workspace_id: str,
    pattern: str,
    path: str | None = None,
    file_glob: str | None = None,
    output_mode: str = "content",
    limit: int | None = MAX_FILE_SEARCH_RESULTS,
    context_token: str | None = None,
) -> str:
    """Legacy direct wrapper: search bounded text files in a workspace."""

    try:
        require_fresh_workspace_context(workspace_id, context_token=context_token)
        return _tool_envelope(
            "file_search",
            {
                "ok": True,
                "search": search_workspace_files(
                    workspace_id,
                    pattern,
                    path=path,
                    file_glob=file_glob,
                    output_mode=output_mode,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_search", exc)


def workspace_status_probe(
    workspace_id: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: fixed workspace status probe for ChatGPT-safe action smoke."""

    try:
        return _tool_envelope(
            "workspace_status_probe",
            {
                "ok": True,
                "status_probe": probe_workspace_status(
                    workspace_id,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_status_probe", exc)


def workspace_scratch_smoke(
    workspace_id: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: fixed scratch write/read/patch/delete smoke."""

    try:
        return _tool_envelope(
            "workspace_scratch_smoke",
            {
                "ok": True,
                "scratch_smoke": run_workspace_scratch_smoke(
                    workspace_id,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_scratch_smoke", exc)


def file_patch(
    workspace_id: str,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: patch text after context/write-policy gates.

    The tool remains disabled in default metadata and is not registered on the
    public MCP surface yet; direct calls exercise the fail-closed write path for
    focused Phase 5 tests.
    """

    try:
        return _tool_envelope(
            "file_patch",
            {
                "ok": True,
                "patch": patch_workspace_file(
                    workspace_id,
                    path,
                    old_string,
                    new_string,
                    replace_all=replace_all,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_patch", exc)


def patch_apply(
    workspace_id: str,
    patches: list[Mapping[str, Any]],
    context_token: str | None = None,
) -> str:
    """Direct wrapper: apply a bounded multi-file literal patch batch."""

    try:
        return _tool_envelope(
            "patch_apply",
            {
                "ok": True,
                "patch_apply": apply_workspace_patch(
                    workspace_id,
                    patches,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("patch_apply", exc)


def file_write(
    workspace_id: str,
    path: str,
    content: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: write text after context/write-policy gates.

    The tool remains disabled in default metadata and is not registered on the
    public MCP surface yet; direct calls exercise the fail-closed write path for
    focused Phase 5 tests.
    """

    try:
        return _tool_envelope(
            "file_write",
            {
                "ok": True,
                "write": write_workspace_file(
                    workspace_id,
                    path,
                    content,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_write", exc)


def file_move(
    workspace_id: str,
    source_path: str,
    destination_path: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: move/rename one file after context/write-policy gates."""

    try:
        return _tool_envelope(
            "file_move",
            {
                "ok": True,
                "move": move_workspace_file(
                    workspace_id,
                    source_path,
                    destination_path,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_move", exc)


def file_delete(
    workspace_id: str,
    path: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: delete one file after context/write-policy gates."""

    try:
        return _tool_envelope(
            "file_delete",
            {
                "ok": True,
                "delete": delete_workspace_file(
                    workspace_id,
                    path,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("file_delete", exc)


def directory_create(
    workspace_id: str,
    path: str,
    parents: bool = False,
    exist_ok: bool = False,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: create a directory after context/write-policy gates."""

    try:
        return _tool_envelope(
            "directory_create",
            {
                "ok": True,
                "directory": create_workspace_directory(
                    workspace_id,
                    path,
                    parents=parents,
                    exist_ok=exist_ok,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("directory_create", exc)


def workspace_diff(
    workspace_id: str,
    context_token: str | None = None,
    max_files: int | None = MAX_WORKSPACE_DIFF_FILES,
) -> str:
    """Direct wrapper: return bounded Git diff/audit after fresh context.

    This remains absent from default public MCP registration until the
    write-surface HTTP/auth/config UX review is complete. It performs only
    read-only Git subprocess calls and reports ``llm_calls=0``.
    """

    try:
        return _tool_envelope(
            "workspace_diff",
            {
                "ok": True,
                "workspace_diff": diff_workspace(
                    workspace_id,
                    context_token=context_token,
                    max_files=max_files,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("workspace_diff", exc)


def git_status(
    workspace_id: str,
    context_token: str | None = None,
    limit: int | None = MAX_GIT_STATUS_ENTRIES,
) -> str:
    """Direct wrapper: return read-only Git status after context/git policy."""

    try:
        return _tool_envelope(
            "git_status",
            {
                "ok": True,
                "git_status": read_workspace_git_status(
                    workspace_id,
                    context_token=context_token,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("git_status", exc)


def git_diff(
    workspace_id: str,
    context_token: str | None = None,
    max_files: int | None = MAX_WORKSPACE_DIFF_FILES,
) -> str:
    """Direct wrapper: return read-only Git diff after context/git policy."""

    try:
        return _tool_envelope(
            "git_diff",
            {
                "ok": True,
                "git_diff": read_workspace_git_diff(
                    workspace_id,
                    context_token=context_token,
                    max_files=max_files,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("git_diff", exc)


def git_log(
    workspace_id: str,
    context_token: str | None = None,
    limit: int | None = MAX_GIT_LOG_COUNT,
) -> str:
    """Direct wrapper: return read-only Git log after context/git policy."""

    try:
        return _tool_envelope(
            "git_log",
            {
                "ok": True,
                "git_log": read_workspace_git_log(
                    workspace_id,
                    context_token=context_token,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("git_log", exc)


def git_branch(
    workspace_id: str,
    context_token: str | None = None,
    limit: int | None = MAX_GIT_BRANCH_COUNT,
) -> str:
    """Direct wrapper: return read-only Git branch metadata after context/git policy."""

    try:
        return _tool_envelope(
            "git_branch",
            {
                "ok": True,
                "git_branch": read_workspace_git_branch(
                    workspace_id,
                    context_token=context_token,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("git_branch", exc)


def cron_list(
    workspace_id: str,
    context_token: str | None = None,
    include_disabled: bool = False,
    limit: int | None = MAX_CRON_LIST_RESULTS,
) -> str:
    """Direct wrapper: list sanitized script/no-agent cron status after context/cron policy."""

    try:
        return _tool_envelope(
            "cron_list",
            {
                "ok": True,
                "cron": list_workspace_cron_jobs(
                    workspace_id,
                    context_token=context_token,
                    include_disabled=include_disabled,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("cron_list", exc)


def cron_pause(
    workspace_id: str,
    job_ref: str,
    context_token: str | None = None,
    reason: str | None = None,
) -> str:
    """Direct wrapper: pause a script-only no_agent cron job after context/cron policy."""

    try:
        return _tool_envelope(
            "cron_pause",
            {
                "ok": True,
                "cron": pause_workspace_cron_job(
                    workspace_id,
                    job_ref,
                    context_token=context_token,
                    reason=reason,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("cron_pause", exc)


def cron_resume(
    workspace_id: str,
    job_ref: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: resume an allowlisted script-only no_agent cron job."""

    try:
        return _tool_envelope(
            "cron_resume",
            {
                "ok": True,
                "cron": resume_workspace_cron_job(
                    workspace_id,
                    job_ref,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("cron_resume", exc)


def cron_run(
    workspace_id: str,
    job_ref: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: trigger an allowlisted script-only no_agent cron job next tick."""

    try:
        return _tool_envelope(
            "cron_run",
            {
                "ok": True,
                "cron": trigger_workspace_cron_job(
                    workspace_id,
                    job_ref,
                    context_token=context_token,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("cron_run", exc)


def cron_create_script_only(
    workspace_id: str,
    schedule: str,
    script: str,
    context_token: str | None = None,
    name: str | None = None,
    repeat: int | None = None,
) -> str:
    """Direct wrapper: create only allowlisted script-only no_agent cron jobs."""

    try:
        return _tool_envelope(
            "cron_create_script_only",
            {
                "ok": True,
                "cron": create_workspace_cron_script_job(
                    workspace_id,
                    schedule,
                    script,
                    context_token=context_token,
                    name=name,
                    repeat=repeat,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("cron_create_script_only", exc)


def message_send(
    workspace_id: str,
    destination: str,
    message: str,
    context_token: str | None = None,
    dry_run: bool = True,
) -> str:
    """Direct wrapper: validate an allowlisted messaging dry-run without delivery."""

    try:
        return _tool_envelope(
            "message_send",
            {
                "ok": True,
                **prepare_workspace_message_send(
                    workspace_id,
                    destination,
                    message,
                    context_token=context_token,
                    dry_run=dry_run,
                    tool_name="message_send",
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("message_send", exc)


def telegram_send(
    workspace_id: str,
    recipient: str,
    message: str,
    context_token: str | None = None,
    dry_run: bool = True,
) -> str:
    """Direct wrapper: validate an allowlisted Telegram dry-run without delivery."""

    try:
        return _tool_envelope(
            "telegram_send",
            {
                "ok": True,
                **prepare_workspace_message_send(
                    workspace_id,
                    f"telegram:{recipient}",
                    message,
                    context_token=context_token,
                    dry_run=dry_run,
                    required_platform="telegram",
                    tool_name="telegram_send",
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("telegram_send", exc)


def process_start(
    workspace_id: str,
    command: str,
    timeout: int = 30,
    working_directory: str = ".",
    context_token: str | None = None,
    max_output_chars: int | None = MAX_PROCESS_LOG_CHARS,
) -> str:
    """Direct wrapper: start an allowlisted runtime-owned background process."""

    try:
        return _tool_envelope(
            "process_start",
            start_workspace_process(
                workspace_id,
                command,
                timeout=timeout,
                working_directory=working_directory,
                context_token=context_token,
                max_output_chars=max_output_chars,
            ),
        )
    except ProfileRouterError as exc:
        return _tool_error("process_start", exc)


def process_list(
    workspace_id: str,
    context_token: str | None = None,
    limit: int | None = MAX_PROCESS_LIST_RESULTS,
) -> str:
    """Direct wrapper: list only runtime-tracked workspace processes."""

    try:
        return _tool_envelope(
            "process_list",
            {
                "ok": True,
                "processes": list_workspace_processes(
                    workspace_id,
                    context_token=context_token,
                    limit=limit,
                ),
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("process_list", exc)


def process_poll(
    workspace_id: str,
    process_id: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: poll only runtime-tracked workspace processes."""

    try:
        return _tool_envelope(
            "process_poll",
            poll_workspace_process(
                workspace_id,
                process_id,
                context_token=context_token,
            ),
        )
    except ProfileRouterError as exc:
        return _tool_error("process_poll", exc)


def process_log(
    workspace_id: str,
    process_id: str,
    context_token: str | None = None,
    max_chars: int | None = MAX_PROCESS_LOG_CHARS,
) -> str:
    """Direct wrapper: read bounded logs only for runtime-tracked processes."""

    try:
        return _tool_envelope(
            "process_log",
            read_workspace_process_log(
                workspace_id,
                process_id,
                context_token=context_token,
                max_chars=max_chars,
            ),
        )
    except ProfileRouterError as exc:
        return _tool_error("process_log", exc)


def process_kill(
    workspace_id: str,
    process_id: str,
    context_token: str | None = None,
) -> str:
    """Direct wrapper: kill only runtime-tracked workspace processes."""

    try:
        return _tool_envelope(
            "process_kill",
            kill_workspace_process(
                workspace_id,
                process_id,
                context_token=context_token,
            ),
        )
    except ProfileRouterError as exc:
        return _tool_error("process_kill", exc)


def terminal_run(
    workspace_id: str,
    command: str,
    timeout: int = 30,
    working_directory: str = ".",
    context_token: str | None = None,
    max_output_chars: int | None = MAX_TERMINAL_OUTPUT_CHARS,
) -> str:
    """Direct wrapper: run an allowlisted no-shell command after context gates."""

    try:
        classification = preflight_terminal_command(
            workspace_id,
            command,
            timeout=timeout,
            working_directory=working_directory,
            max_output_chars=max_output_chars,
            context_token=context_token,
        )
        if classification["blocked"]:
            return _tool_envelope(
                "terminal_run",
                {
                    "ok": False,
                    "error": {
                        "code": "terminal_command_blocked",
                        "message": "terminal command is blocked by profile-router no-model policy",
                    },
                    "terminal_command": classification,
                },
            )
        if not classification["execution_plan"]["available"]:
            return _tool_envelope(
                "terminal_run",
                {
                    "ok": False,
                    "error": {
                        "code": "tool_disabled",
                        "message": (
                            "terminal_run requires explicit terminal.execution "
                            "allowlist policy before private direct execution"
                        ),
                    },
                    "terminal_command": classification,
                },
            )

        terminal_result = _run_preflighted_terminal_command(
            workspace_id,
            command,
            preflight=classification,
            context_token=context_token,
        )
        if terminal_result["status"] == "success":
            return _tool_envelope(
                "terminal_run",
                {
                    "ok": True,
                    "terminal_command": classification,
                    "terminal_result": terminal_result,
                },
            )
        error_code = (
            "terminal_command_timeout"
            if terminal_result["status"] == "timeout"
            else "terminal_command_failed"
        )
        return _tool_envelope(
            "terminal_run",
            {
                "ok": False,
                "error": {
                    "code": error_code,
                    "message": "allowlisted terminal command completed without success",
                },
                "terminal_command": classification,
                "terminal_result": terminal_result,
            },
        )
    except ProfileRouterError as exc:
        return _tool_error("terminal_run", exc)
