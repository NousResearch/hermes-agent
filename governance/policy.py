"""Central Policy Gate for Hermes tool execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .backup import has_recent_verified_backup
from .evidence import append_hash_chained_event


class ActionClass(str, Enum):
    READ_ONLY_LOCAL_SAFE = "read_only_local_safe"
    READ_ONLY_SECRET_ADJACENT = "read_only_secret_adjacent"
    PROCESS_ENV_READ = "process_env_read"
    NETWORK_READ = "network_read"
    TEST_ONLY = "test_only"
    REVERSIBLE_EDIT = "reversible_edit"
    DEPENDENCY_CHANGE = "dependency_change"
    MEMORY_WRITE = "memory_write"
    SERVICE_RUNTIME_CHANGE = "service_runtime_change"
    DURABLE_SCHEDULING_CHANGE = "durable_scheduling_change"
    REMOTE_WRITE = "remote_write"
    PUBLIC_EXPOSURE_CHANGE = "public_exposure_change"
    CREDENTIAL_OR_AUTH_CHANGE = "credential_or_auth_change"
    LIVE_DATA_MIGRATION = "live_data_migration"
    DESTRUCTIVE = "destructive"
    UNKNOWN = "unknown"


class Decision(str, Enum):
    ALLOW = "allow"
    ALLOW_AFTER_BACKUP = "allow_after_backup"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass
class PolicyGateRequest:
    requested_action: str
    action_class: ActionClass
    profile: str = "default"
    tool_name: Optional[str] = None
    command: Optional[str] = None
    risk_tier: int = 1
    affected_paths: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    affected_memory_stores: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityRules:
    profile: str
    allowed_tools: Optional[List[str]] = None
    denied_tools: List[str] = field(default_factory=list)
    denied_action_classes: List[ActionClass] = field(default_factory=list)
    max_auto_tier: int = 2
    standing_approval_action_classes: List[ActionClass] = field(default_factory=lambda: [
        ActionClass.READ_ONLY_LOCAL_SAFE,
        ActionClass.NETWORK_READ,
        ActionClass.TEST_ONLY,
        ActionClass.REVERSIBLE_EDIT,
        ActionClass.MEMORY_WRITE,
    ])


@dataclass
class PolicyDecision:
    decision: Decision
    reason: str
    request: PolicyGateRequest
    capability_check: str = "not_checked"
    backup_check: str = "not_required"
    logged: bool = False
    log_ref: Optional[str] = None
    enforcement_mode: str = "audit"

    def to_log_event(self) -> Dict[str, Any]:
        return {
            "event_type": "policy_decision",
            "decision": self.decision.value,
            "reason": self.reason,
            "profile": self.request.profile,
            "requested_action": self.request.requested_action,
            "tool_name": self.request.tool_name,
            "action_class": self.request.action_class.value,
            "risk_tier": self.request.risk_tier,
            "affected_paths": list(self.request.affected_paths),
            "affected_services": list(self.request.affected_services),
            "affected_memory_stores": list(self.request.affected_memory_stores),
            "capability_check": self.capability_check,
            "backup_check": self.backup_check,
            "enforcement_mode": self.enforcement_mode,
        }


def default_capabilities_for_profile(profile: str) -> CapabilityRules:
    return CapabilityRules(profile=profile)


_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+(-[\w-]*[rf][\w-]*|-[\w-]*[fr][\w-]*)\b",
    r"\bshutil\.rmtree\s*\(",
    r"\bmkfs(?:\.|\s)",
    r"\bdd\b.*\bof=",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[\w-]*f",
    r"\btruncate\s+-s\s+0\b",
]
_SECRET_READ_PATTERNS = [
    r"\b(cat|less|more|tail|head|grep|rg)\b.*(~?/\.env|\.netrc|auth\.json|api-keys|credentials|secret|token)",
    r"\b(pass|op|security)\s+show\b",
]
_PROCESS_ENV_PATTERNS = [r"\bps\s+[^\n;|&]*\beww\b", r"\bprintenv\b", r"\benv\b"]
_DEPENDENCY_PATTERNS = [
    r"\b(pip|pip3|uv)\s+install\b",
    r"\bpython\s+-m\s+pip\s+install\b",
    r"\b(npm|pnpm|yarn)\s+(install|add|update)\b",
    r"\b(apt|apt-get|dnf|yum|pacman|brew)\s+.*\b(install|upgrade|remove)\b",
    r"\bcargo\s+install\b",
]
_SERVICE_PATTERNS = [r"\bsystemctl\b.*\b(start|stop|restart|reload|enable|disable|mask|unmask)\b", r"\bservice\s+\S+\s+(start|stop|restart|reload)\b"]
_PUBLIC_EXPOSURE_PATTERNS = [r"\btailscale\s+funnel\b", r"\bngrok\b", r"\bcloudflared\s+tunnel\b"]
_CREDENTIAL_PATTERNS = [r"\bhermes\s+login\b", r"\bhermes\s+logout\b", r"\bgh\s+auth\b", r"\bssh-keygen\b", r"\bchmod\s+.*(auth|secret|key)"]
_TEST_PATTERNS = [r"\b(pytest|python3?\s+-m\s+pytest)\b", r"\bnpm\s+test\b", r"\bcargo\s+test\b", r"\bgo\s+test\b"]

_READ_ONLY_TOOL_PREFIXES = (
    "get",
    "list",
    "read",
    "search",
    "probe",
    "related",
    "reason",
    "contradict",
    "debug",
    "ping",
    "status",
)
_READ_ONLY_TOOL_SUFFIXES = ("_read", "_read_file", "_list", "_search", "_status", "_summary")
_REMOTE_MUTATION_TERMS = {
    "add",
    "apply",
    "close",
    "commit",
    "create",
    "delete",
    "edit",
    "fork",
    "merge",
    "patch",
    "push",
    "remove",
    "replace",
    "restore",
    "resume",
    "run",
    "send",
    "start",
    "stop",
    "submit",
    "update",
    "upload",
    "write",
}
_MIGRATION_OR_SQL_TERMS = {"migration", "migrate", "sql", "execute", "exec"}


def _matches_any(command: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, command, re.IGNORECASE) for pattern in patterns)


def _tool_tokens(tool_name: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", (tool_name or "").lower()) if token]


def _looks_like_read_only_tool(tool_name: str) -> bool:
    lowered = (tool_name or "").lower()
    tokens = _tool_tokens(lowered)
    if not tokens:
        return False
    if lowered.endswith(_READ_ONLY_TOOL_SUFFIXES):
        return True
    return tokens[-1] in _READ_ONLY_TOOL_PREFIXES or tokens[0] in _READ_ONLY_TOOL_PREFIXES


def _looks_like_live_data_migration(tool_name: str) -> bool:
    lowered = (tool_name or "").lower()
    tokens = set(_tool_tokens(lowered))
    if "sql" in tokens and "read" not in tokens:
        return True
    if "migration" in tokens or "migrate" in tokens:
        return True
    if {"execute", "sql"}.issubset(tokens) or {"exec", "sql"}.issubset(tokens):
        return True
    if lowered.endswith("_migration_apply") or lowered.endswith("_sql_execute"):
        return True
    return False


def _looks_like_remote_mutation_tool(tool_name: str) -> bool:
    tokens = set(_tool_tokens(tool_name))
    return bool(tokens & _REMOTE_MUTATION_TERMS)


def classify_command(command: str) -> ActionClass:
    """Classify shell command semantics instead of trusting tool name alone."""
    command = command or ""
    if _matches_any(command, _DESTRUCTIVE_PATTERNS):
        return ActionClass.DESTRUCTIVE
    if _matches_any(command, _PUBLIC_EXPOSURE_PATTERNS):
        return ActionClass.PUBLIC_EXPOSURE_CHANGE
    if _matches_any(command, _CREDENTIAL_PATTERNS):
        return ActionClass.CREDENTIAL_OR_AUTH_CHANGE
    if _matches_any(command, _SERVICE_PATTERNS):
        return ActionClass.SERVICE_RUNTIME_CHANGE
    if _matches_any(command, _SECRET_READ_PATTERNS):
        return ActionClass.READ_ONLY_SECRET_ADJACENT
    if _matches_any(command, _PROCESS_ENV_PATTERNS):
        return ActionClass.PROCESS_ENV_READ
    if _matches_any(command, _DEPENDENCY_PATTERNS):
        return ActionClass.DEPENDENCY_CHANGE
    if _matches_any(command, _TEST_PATTERNS):
        return ActionClass.TEST_ONLY
    return ActionClass.READ_ONLY_LOCAL_SAFE


def _risk_tier(action_class: ActionClass) -> int:
    if action_class in {ActionClass.READ_ONLY_LOCAL_SAFE, ActionClass.NETWORK_READ, ActionClass.TEST_ONLY}:
        return 1
    if action_class in {ActionClass.REVERSIBLE_EDIT, ActionClass.DEPENDENCY_CHANGE, ActionClass.READ_ONLY_SECRET_ADJACENT, ActionClass.PROCESS_ENV_READ}:
        return 2
    if action_class in {ActionClass.MEMORY_WRITE, ActionClass.SERVICE_RUNTIME_CHANGE, ActionClass.DURABLE_SCHEDULING_CHANGE, ActionClass.REMOTE_WRITE}:
        return 3
    return 4


def _affected_services(command: str) -> List[str]:
    services: List[str] = []
    for match in re.finditer(r"\b(?:systemctl(?:\s+--user)?|service)\s+(?:--user\s+)?(?:start|stop|restart|reload|enable|disable|mask|unmask)?\s*([\w@.\-]+\.service|[\w@.\-]+)", command or ""):
        candidate = match.group(1)
        if candidate not in {"start", "stop", "restart", "reload", "enable", "disable", "mask", "unmask"}:
            services.append(candidate)
    # systemctl syntax usually puts verb before service; the regex above is
    # intentionally broad but can miss when flags appear between.  Add a small
    # fallback for unit-looking tokens.
    for token in re.findall(r"[\w@.\-]+\.service", command or ""):
        if token not in services:
            services.append(token)
    return services


def _path_arg(args: Mapping[str, Any], *names: str) -> List[str]:
    found: List[str] = []
    for name in names:
        value = args.get(name)
        if isinstance(value, str) and value:
            found.append(value)
    return found


def classify_tool_call(function_name: str, function_args: Mapping[str, Any] | None, *, profile: str = "default") -> PolicyGateRequest:
    args = dict(function_args or {})
    tool = function_name
    action_class = ActionClass.UNKNOWN
    affected_paths: List[str] = []
    affected_services: List[str] = []
    affected_memory_stores: List[str] = []
    command: Optional[str] = None

    if tool == "terminal":
        command = str(args.get("command", ""))
        action_class = classify_command(command)
        affected_services = _affected_services(command) if action_class == ActionClass.SERVICE_RUNTIME_CHANGE else []
    elif tool in {"read_file", "search_files", "browser_snapshot", "browser_console", "browser_get_images"}:
        action_class = ActionClass.READ_ONLY_LOCAL_SAFE
        affected_paths = _path_arg(args, "path")
    elif tool in {"write_file", "patch", "skill_manage"}:
        action_class = ActionClass.REVERSIBLE_EDIT
        affected_paths = _path_arg(args, "path", "file_path")
    elif tool == "memory" or tool == "fact_store":
        action_class = ActionClass.MEMORY_WRITE
        target = args.get("target") or args.get("category") or "memory"
        affected_memory_stores = [str(target)]
    elif tool == "cronjob":
        action_class = ActionClass.DURABLE_SCHEDULING_CHANGE
    elif tool in {"send_message", "mcp_github_create_or_update_file", "mcp_github_push_files", "mcp_github_create_pull_request", "mcp_github_merge_pull_request", "mcp_github_create_issue", "mcp_github_update_issue", "mcp_github_add_issue_comment"}:
        action_class = ActionClass.REMOTE_WRITE
    elif tool.startswith("browser_") or tool in {"web_search", "web_extract"}:
        action_class = ActionClass.NETWORK_READ
    elif "supabase_migration_apply" in tool or tool.endswith("_migration_apply"):
        action_class = ActionClass.LIVE_DATA_MIGRATION
    elif tool in {"image_generate", "text_to_speech", "vision_analyze"}:
        action_class = ActionClass.REMOTE_WRITE if tool == "image_generate" else ActionClass.READ_ONLY_LOCAL_SAFE
    elif _looks_like_live_data_migration(tool):
        action_class = ActionClass.LIVE_DATA_MIGRATION
    elif _looks_like_remote_mutation_tool(tool):
        action_class = ActionClass.REMOTE_WRITE
    elif tool.startswith("mcp_") and _looks_like_read_only_tool(tool):
        action_class = ActionClass.NETWORK_READ
    else:
        action_class = ActionClass.UNKNOWN

    return PolicyGateRequest(
        requested_action=f"tool:{tool}",
        action_class=action_class,
        profile=profile,
        tool_name=tool,
        command=command,
        risk_tier=_risk_tier(action_class),
        affected_paths=affected_paths,
        affected_services=affected_services,
        affected_memory_stores=affected_memory_stores,
        metadata={"raw_args_keys": sorted(args.keys())},
    )


class PolicyGate:
    """Evaluate and log governance decisions for tool execution."""

    def __init__(self, *, profile: str = "default", capabilities: CapabilityRules | None = None, enforcement_mode: str | None = None):
        self.profile = profile
        self.capabilities = capabilities or default_capabilities_for_profile(profile)
        self.enforcement_mode = (enforcement_mode or os.getenv("HERMES_POLICY_GATE_MODE", "audit")).strip().lower() or "audit"

    def evaluate(self, request: PolicyGateRequest) -> PolicyDecision:
        request.profile = request.profile or self.profile
        capability_check = self._capability_check(request)
        backup_check = "not_required"

        if capability_check != "passed":
            decision = PolicyDecision(
                decision=Decision.DENY,
                reason="Capability check failed for this profile/tool/action class.",
                request=request,
                capability_check=capability_check,
                enforcement_mode=self.enforcement_mode,
            )
            return self._log(decision)

        if request.action_class == ActionClass.DESTRUCTIVE:
            decision = PolicyDecision(
                decision=Decision.DENY,
                reason="Destructive action is denied by hardened governance policy.",
                request=request,
                capability_check=capability_check,
                enforcement_mode=self.enforcement_mode,
            )
            return self._log(decision)

        if request.action_class == ActionClass.REVERSIBLE_EDIT:
            backup_check = self._backup_check(request)
            if backup_check == "failed":
                decision = PolicyDecision(
                    decision=Decision.ALLOW_AFTER_BACKUP,
                    reason="Reversible edit requires a verified backup before live mutation.",
                    request=request,
                    capability_check=capability_check,
                    backup_check=backup_check,
                    enforcement_mode=self.enforcement_mode,
                )
                return self._log(decision)

        if request.action_class in {
            ActionClass.UNKNOWN,
            ActionClass.PUBLIC_EXPOSURE_CHANGE,
            ActionClass.CREDENTIAL_OR_AUTH_CHANGE,
            ActionClass.LIVE_DATA_MIGRATION,
            ActionClass.SERVICE_RUNTIME_CHANGE,
            ActionClass.DURABLE_SCHEDULING_CHANGE,
        }:
            decision = PolicyDecision(
                decision=Decision.REQUIRE_APPROVAL,
                reason=f"{request.action_class.value} requires explicit approval and verification gates.",
                request=request,
                capability_check=capability_check,
                backup_check=backup_check,
                enforcement_mode=self.enforcement_mode,
            )
            return self._log(decision)

        if request.risk_tier > self.capabilities.max_auto_tier and request.action_class not in self.capabilities.standing_approval_action_classes:
            decision = PolicyDecision(
                decision=Decision.REQUIRE_APPROVAL,
                reason="Action risk exceeds automatic standing approval for this profile.",
                request=request,
                capability_check=capability_check,
                backup_check=backup_check,
                enforcement_mode=self.enforcement_mode,
            )
            return self._log(decision)

        decision = PolicyDecision(
            decision=Decision.ALLOW,
            reason="Action allowed by current profile capability rules.",
            request=request,
            capability_check=capability_check,
            backup_check=backup_check,
            enforcement_mode=self.enforcement_mode,
        )
        return self._log(decision)

    def _capability_check(self, request: PolicyGateRequest) -> str:
        if request.tool_name and request.tool_name in self.capabilities.denied_tools:
            return "failed"
        if self.capabilities.allowed_tools is not None and request.tool_name not in self.capabilities.allowed_tools:
            return "failed"
        if request.action_class in self.capabilities.denied_action_classes:
            return "failed_denied_action_class"
        return "passed"

    def _backup_check(self, request: PolicyGateRequest) -> str:
        existing_paths = [Path(p).expanduser() for p in request.affected_paths if p]
        existing_files = [p for p in existing_paths if p.exists()]
        if not existing_files:
            return "not_required"
        if all(has_recent_verified_backup(p) for p in existing_files):
            return "passed"
        return "failed"

    def _log(self, decision: PolicyDecision) -> PolicyDecision:
        try:
            row = append_hash_chained_event("policy_decisions", decision.to_log_event())
            decision.logged = True
            decision.log_ref = row.get("entry_hash")
        except Exception:
            decision.logged = False
        return decision

    def should_block_dispatch(self, decision: PolicyDecision) -> bool:
        """Return whether handle_function_call should block execution now.

        Denials are always hard blocks.  Other gated states are hard blocks only
        in strict/enforce mode so current live Hermes behavior can be audited
        before a deployment flips stricter runtime enforcement on.
        """
        if decision.decision == Decision.DENY:
            return True
        if self.enforcement_mode in {"strict", "enforce", "enforced"} and decision.decision in {Decision.REQUIRE_APPROVAL, Decision.ALLOW_AFTER_BACKUP}:
            return True
        return False
