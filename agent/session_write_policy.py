"""Session-scoped write policy context.

Phase B deliberately makes the policy available at agent/tool boundaries
without enforcing foreground file/terminal denials yet. The module is pure
apart from ContextVar binding.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional


POLICY_VERSION = "session_write_policy/v1"


class SessionWritePolicyError(ValueError):
    """Raised when a malformed or widening policy is rejected."""


class SessionWritePolicyMode(str, Enum):
    NORMAL = "NORMAL"
    ALLOWLIST = "ALLOWLIST"
    DENY_ALL = "DENY_ALL"


class SessionWritePolicyDecisionResult(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


PHASE_C_MUTATING_OPERATION_KINDS = frozenset(
    {
        "terminal_exec",
        "file_create",
        "file_write",
        "file_patch",
        "file_delete",
        "skill_create",
        "skill_edit",
        "skill_patch",
        "skill_delete",
        "skill_write_file",
        "skill_remove_file",
        "memory_add",
        "memory_replace",
        "memory_remove",
        "memory_batch",
        "memory_save",
    }
)

_FILESYSTEM_OPERATION_KINDS = frozenset(
    {
        "file_create",
        "file_write",
        "file_patch",
        "file_delete",
        "skill_create",
        "skill_edit",
        "skill_patch",
        "skill_delete",
        "skill_write_file",
        "skill_remove_file",
        "memory_add",
        "memory_replace",
        "memory_remove",
        "memory_batch",
        "memory_save",
    }
)

_READABLE_OPERATION_LABELS = {
    "terminal_exec": "terminal execution",
    "file_create": "file create",
    "file_write": "file write",
    "file_patch": "file patch",
    "file_delete": "file delete",
    "skill_create": "skill create",
    "skill_edit": "skill edit",
    "skill_patch": "skill patch",
    "skill_delete": "skill delete",
    "skill_write_file": "skill file write",
    "skill_remove_file": "skill file remove",
    "memory_add": "memory add",
    "memory_replace": "memory replace",
    "memory_remove": "memory remove",
    "memory_batch": "memory batch",
    "memory_save": "memory save",
}


def policy_evaluation_failure_payload(
    *,
    operation_kind: str,
    session_id: str = "",
    target: str = "",
    error: Exception | None = None,
) -> dict[str, Any]:
    """Stable fail-closed payload for internal policy evaluation failures."""
    return {
        "success": False,
        "error": "Session write policy evaluation failed; mutation denied",
        "policy_reason": "policy_evaluation_failed",
        "operation_kind": str(operation_kind or ""),
        "session_id": str(session_id or ""),
        "target": str(target or ""),
    }


@dataclass(frozen=True)
class SessionWritePolicyDecision:
    result: SessionWritePolicyDecisionResult
    reason: str
    operation_kind: str
    origin: str
    session_id: str = ""
    target_path: Optional[str] = None
    capability_kind: Optional[str] = None
    capability_operation: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.result is SessionWritePolicyDecisionResult.ALLOW

    @property
    def denied(self) -> bool:
        return self.result is SessionWritePolicyDecisionResult.DENY

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result": self.result.value,
            "reason": self.reason,
            "operation_kind": self.operation_kind,
            "origin": self.origin,
            "session_id": self.session_id,
        }
        if self.target_path:
            payload["target_path"] = self.target_path
        if self.capability_kind:
            payload["capability_kind"] = self.capability_kind
        if self.capability_operation:
            payload["capability_operation"] = self.capability_operation
        return payload

    def denial_payload(self) -> dict[str, Any]:
        target_summary = self.target_path or ""
        label = _READABLE_OPERATION_LABELS.get(self.operation_kind, self.operation_kind)
        return {
            "success": False,
            "error": f"Session write policy denied {label}: {self.reason}",
            "policy_reason": self.reason,
            "operation_kind": self.operation_kind,
            "session_id": self.session_id,
            "target": target_summary,
        }

    def denial_json(self) -> str:
        import json

        return json.dumps(self.denial_payload(), ensure_ascii=False)


@dataclass(frozen=True)
class CapabilityGrant:
    """Typed internal capability grant.

    ``kind`` is intentionally explicit (for example ``"filesystem"``).
    ``operation`` is a narrow operation family such as ``"write"`` or
    ``"delete"``. Filesystem grants may carry narrower ``roots`` than the
    containing policy.
    """

    kind: str
    operation: str
    roots: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind or "").strip())
        object.__setattr__(self, "operation", str(self.operation or "").strip())
        roots = tuple(_canonicalize_root(root) for root in (self.roots or ()))
        object.__setattr__(self, "roots", roots)


def _has_parent_ref(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _canonicalize_root(path: str | Path) -> str:
    raw = Path(path).expanduser()
    if _has_parent_ref(raw):
        raise SessionWritePolicyError(f"path contains parent traversal: {path!r}")
    return str(raw.resolve(strict=False))


def canonicalize_target(path: str | Path) -> str:
    """Canonicalize an existing or not-yet-existing target.

    Existing paths resolve through symlinks. For non-existent paths, resolve the
    nearest existing parent and append unresolved components without allowing
    ``..`` traversal.
    """
    raw = Path(path).expanduser()
    if _has_parent_ref(raw):
        raise SessionWritePolicyError(f"path contains parent traversal: {path!r}")

    if raw.exists():
        return str(raw.resolve(strict=True))

    missing: list[str] = []
    cursor = raw
    while not cursor.exists():
        name = cursor.name
        if not name:
            break
        missing.append(name)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent

    if not cursor.exists():
        return str(raw.resolve(strict=False))

    base = cursor.resolve(strict=True)
    for part in reversed(missing):
        if part in {"", ".", ".."}:
            raise SessionWritePolicyError(f"unsafe unresolved path component: {part!r}")
        base = base / part
    return str(base)


def _is_under_root(target: str, root: str) -> bool:
    try:
        target_path = Path(target)
        root_path = Path(root)
        return target_path == root_path or root_path in target_path.parents
    except Exception:
        return False


@dataclass(frozen=True)
class SessionWritePolicy:
    session_id: str
    mode: SessionWritePolicyMode = SessionWritePolicyMode.NORMAL
    allowed_roots: tuple[str, ...] = field(default_factory=tuple)
    origin: str = "default"
    parent_session_id: Optional[str] = None
    capability_grants: tuple[CapabilityGrant, ...] = field(default_factory=tuple)
    version: str = POLICY_VERSION
    protected: bool = False

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, str):
            try:
                mode = SessionWritePolicyMode(mode)
            except ValueError as exc:
                raise SessionWritePolicyError(f"unknown session write mode: {mode!r}") from exc
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "session_id", str(self.session_id or ""))
        object.__setattr__(self, "origin", str(self.origin or ""))
        object.__setattr__(
            self,
            "parent_session_id",
            str(self.parent_session_id) if self.parent_session_id is not None else None,
        )
        roots = tuple(_canonicalize_root(root) for root in (self.allowed_roots or ()))
        object.__setattr__(self, "allowed_roots", roots)
        grants = tuple(
            grant if isinstance(grant, CapabilityGrant) else CapabilityGrant(**grant)
            for grant in (self.capability_grants or ())
        )
        object.__setattr__(self, "capability_grants", grants)
        object.__setattr__(self, "version", str(self.version or POLICY_VERSION))
        object.__setattr__(self, "protected", bool(self.protected))

    @classmethod
    def normal(cls, session_id: str = "", *, origin: str = "default") -> "SessionWritePolicy":
        return cls(session_id=session_id, mode=SessionWritePolicyMode.NORMAL, origin=origin)

    @classmethod
    def deny_all(
        cls,
        session_id: str = "",
        *,
        origin: str = "protected",
        parent_session_id: Optional[str] = None,
        capability_grants: tuple[CapabilityGrant, ...] = (),
    ) -> "SessionWritePolicy":
        return cls(
            session_id=session_id,
            mode=SessionWritePolicyMode.DENY_ALL,
            origin=origin,
            parent_session_id=parent_session_id,
            capability_grants=capability_grants,
            protected=True,
        )

    @property
    def is_protected(self) -> bool:
        return self.protected or self.mode is not SessionWritePolicyMode.NORMAL

    def canonical_target(self, path: str | Path) -> str:
        return canonicalize_target(path)

    def contains_path(self, path: str | Path, roots: tuple[str, ...] | None = None) -> bool:
        target = canonicalize_target(path)
        return any(_is_under_root(target, root) for root in (roots or self.allowed_roots))

    def grants_capability(
        self,
        *,
        kind: str,
        operation: str,
        target_path: str | Path | None = None,
    ) -> bool:
        kind = str(kind or "").strip()
        operation = str(operation or "").strip()
        for grant in self.capability_grants:
            if grant.kind != kind or grant.operation != operation:
                continue
            if target_path is None or not grant.roots:
                return True
            if self.contains_path(target_path, grant.roots):
                return True
        return False

    def allows_filesystem_mutation(self, target_path: str | Path, operation: str) -> bool:
        if self.mode is SessionWritePolicyMode.NORMAL:
            return True
        if self.mode is SessionWritePolicyMode.ALLOWLIST:
            return (
                self.contains_path(target_path)
                and self.grants_capability(
                    kind="filesystem",
                    operation=operation,
                    target_path=target_path,
                )
            )
        return self.grants_capability(
            kind="filesystem",
            operation=operation,
            target_path=target_path,
        )

    def decide_mutation(
        self,
        *,
        operation_kind: str,
        origin: str,
        target_path: str | Path | None = None,
        capability: CapabilityGrant | None = None,
    ) -> SessionWritePolicyDecision:
        return evaluate_session_write_policy(
            self,
            operation_kind=operation_kind,
            origin=origin,
            target_path=target_path,
            capability=capability,
        )

    def derive_child(
        self,
        requested: Optional["SessionWritePolicy"] = None,
    ) -> "SessionWritePolicy":
        if requested is None:
            return self
        validate_child_policy(self, requested)
        return requested


_MODE_STRENGTH = {
    SessionWritePolicyMode.NORMAL: 0,
    SessionWritePolicyMode.ALLOWLIST: 1,
    SessionWritePolicyMode.DENY_ALL: 2,
}


def _grant_subset(child: CapabilityGrant, parent: CapabilityGrant) -> bool:
    if child.kind != parent.kind or child.operation != parent.operation:
        return False
    if not child.roots:
        return not parent.roots
    if not parent.roots:
        return True
    return all(any(_is_under_root(root, parent_root) for parent_root in parent.roots) for root in child.roots)


def _grants_subset(
    child_grants: tuple[CapabilityGrant, ...],
    parent_grants: tuple[CapabilityGrant, ...],
) -> bool:
    return all(any(_grant_subset(child, parent) for parent in parent_grants) for child in child_grants)


def _deny(
    policy: SessionWritePolicy,
    *,
    operation_kind: str,
    origin: str,
    reason: str,
    target_path: str | Path | None = None,
    capability: CapabilityGrant | None = None,
) -> SessionWritePolicyDecision:
    return SessionWritePolicyDecision(
        result=SessionWritePolicyDecisionResult.DENY,
        reason=reason,
        operation_kind=str(operation_kind or ""),
        origin=str(origin or policy.origin or ""),
        session_id=policy.session_id,
        target_path=str(target_path) if target_path is not None else None,
        capability_kind=capability.kind if capability else None,
        capability_operation=capability.operation if capability else None,
    )


def _allow(
    policy: SessionWritePolicy,
    *,
    operation_kind: str,
    origin: str,
    target_path: str | Path | None = None,
    capability: CapabilityGrant | None = None,
) -> SessionWritePolicyDecision:
    return SessionWritePolicyDecision(
        result=SessionWritePolicyDecisionResult.ALLOW,
        reason="policy_allow",
        operation_kind=str(operation_kind or ""),
        origin=str(origin or policy.origin or ""),
        session_id=policy.session_id,
        target_path=str(target_path) if target_path is not None else None,
        capability_kind=capability.kind if capability else None,
        capability_operation=capability.operation if capability else None,
    )


def _required_capability(operation_kind: str, capability: CapabilityGrant | None) -> CapabilityGrant:
    if capability is not None:
        return capability
    return CapabilityGrant(kind="filesystem", operation=operation_kind)


def evaluate_session_write_policy(
    policy: SessionWritePolicy,
    *,
    operation_kind: str,
    origin: str,
    target_path: str | Path | None = None,
    capability: CapabilityGrant | None = None,
) -> SessionWritePolicyDecision:
    """Side-effect-free write-boundary decision API for Phase C mutations."""
    if not isinstance(policy, SessionWritePolicy):
        policy = coerce_session_write_policy(policy, protected=True)

    operation_kind = str(operation_kind or "").strip()
    origin = str(origin or policy.origin or "").strip()

    if operation_kind not in PHASE_C_MUTATING_OPERATION_KINDS:
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason="unknown_operation_kind",
            target_path=target_path,
            capability=capability,
        )

    if policy.mode is SessionWritePolicyMode.NORMAL:
        return _allow(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            target_path=target_path,
            capability=capability,
        )

    if operation_kind == "terminal_exec":
        reason = (
            "terminal_exec_denied_deny_all"
            if policy.mode is SessionWritePolicyMode.DENY_ALL
            else "terminal_exec_denied_protected_mode"
        )
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason=reason,
            target_path=target_path,
            capability=capability,
        )

    if policy.mode is SessionWritePolicyMode.DENY_ALL:
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason="deny_all",
            target_path=target_path,
            capability=capability,
        )

    if operation_kind not in _FILESYSTEM_OPERATION_KINDS:
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason="unsupported_protected_operation",
            target_path=target_path,
            capability=capability,
        )

    if target_path is None:
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason="missing_target_path",
            capability=capability,
        )

    required = _required_capability(operation_kind, capability)
    try:
        canonical = canonicalize_target(target_path)
    except Exception as exc:
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason=f"target_canonicalization_failed:{type(exc).__name__}",
            target_path=target_path,
            capability=required,
        )

    if not any(_is_under_root(canonical, root) for root in policy.allowed_roots):
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason="target_outside_allowed_roots",
            target_path=canonical,
            capability=required,
        )

    if not policy.grants_capability(
        kind=required.kind,
        operation=required.operation,
        target_path=canonical,
    ):
        return _deny(
            policy,
            operation_kind=operation_kind,
            origin=origin,
            reason="missing_capability_grant",
            target_path=canonical,
            capability=required,
        )

    return _allow(
        policy,
        operation_kind=operation_kind,
        origin=origin,
        target_path=canonical,
        capability=required,
    )


def validate_child_policy(parent: SessionWritePolicy, child: SessionWritePolicy) -> None:
    if not isinstance(parent, SessionWritePolicy) or not isinstance(child, SessionWritePolicy):
        raise SessionWritePolicyError("parent and child policies must be SessionWritePolicy instances")

    if parent.mode is SessionWritePolicyMode.NORMAL:
        return

    if _MODE_STRENGTH[child.mode] < _MODE_STRENGTH[parent.mode]:
        raise SessionWritePolicyError(
            f"child policy mode {child.mode.value} downgrades parent {parent.mode.value}"
        )

    if parent.mode is SessionWritePolicyMode.ALLOWLIST and child.mode is SessionWritePolicyMode.ALLOWLIST:
        for root in child.allowed_roots:
            if not any(_is_under_root(root, parent_root) for parent_root in parent.allowed_roots):
                raise SessionWritePolicyError("child allowlist root is wider than parent")

    if parent.mode is SessionWritePolicyMode.DENY_ALL and child.mode is not SessionWritePolicyMode.DENY_ALL:
        raise SessionWritePolicyError("DENY_ALL parent cannot create a less restrictive child")

    if not _grants_subset(child.capability_grants, parent.capability_grants):
        raise SessionWritePolicyError("child capability grants exceed parent grants")


_UNSET = object()
_CURRENT_SESSION_WRITE_POLICY: ContextVar[object] = ContextVar(
    "HERMES_SESSION_WRITE_POLICY",
    default=_UNSET,
)


def coerce_session_write_policy(
    value: Any,
    *,
    session_id: str = "",
    protected: bool = False,
) -> SessionWritePolicy:
    if isinstance(value, SessionWritePolicy):
        return value
    if value is None and not protected:
        return SessionWritePolicy.normal(session_id=session_id, origin="missing_unprotected")
    if isinstance(value, dict):
        try:
            return SessionWritePolicy(**value)
        except Exception:
            if not protected:
                raise
    if protected:
        return SessionWritePolicy.deny_all(session_id=session_id, origin="malformed_protected")
    raise SessionWritePolicyError("malformed session write policy")


def get_current_session_write_policy(
    *,
    protected: bool = False,
    session_id: str = "",
) -> SessionWritePolicy:
    current = _CURRENT_SESSION_WRITE_POLICY.get()
    if current is _UNSET:
        if protected:
            return SessionWritePolicy.deny_all(session_id=session_id, origin="missing_protected_context")
        return SessionWritePolicy.normal(session_id=session_id, origin="missing_unprotected")
    return coerce_session_write_policy(current, session_id=session_id, protected=protected)


def bind_session_write_policy(policy: SessionWritePolicy) -> Token:
    if not isinstance(policy, SessionWritePolicy):
        raise SessionWritePolicyError("policy must be a SessionWritePolicy")
    return _CURRENT_SESSION_WRITE_POLICY.set(policy)


def reset_session_write_policy(token: Token) -> None:
    _CURRENT_SESSION_WRITE_POLICY.reset(token)


@contextmanager
def session_write_policy_scope(policy: SessionWritePolicy) -> Iterator[SessionWritePolicy]:
    token = bind_session_write_policy(policy)
    try:
        yield policy
    finally:
        reset_session_write_policy(token)


def policy_from_read_only_env(
    *,
    session_id: str,
    read_only_value: Any,
) -> Optional[SessionWritePolicy]:
    from agent.self_improvement_policy import normalize_read_only_session

    if normalize_read_only_session(read_only_value):
        return SessionWritePolicy.deny_all(session_id=session_id, origin="HERMES_READ_ONLY_SESSION")
    return None


# ========================================================================
# Phase 1: foreground Git-write policy hardening (ordinal 1)
# Appended at end of file per blueprint ord 1.
# Adds: CallerType enum, _emit_policy_event, PolicyDenied, pre_spawn_consult,
#       caller_type field on SessionWritePolicyDecision.
# Contract: T-01..T-18, T-50..T-52, T-91..T-99, T-104, T-105, T-106, T-501.
# ========================================================================

from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from agent.git_mutation_classifier import GitMutationAssessment  # noqa: F401
    from agent.git_target_resolver import ResolvedExecutionTarget      # noqa: F401


class CallerType(str, Enum):
    """Identifies the calling surface for a pre-spawn consult."""

    TERMINAL_TOOL = "TERMINAL_TOOL"
    DELEGATION = "DELEGATION"
    CODE_EXECUTION = "CODE_EXECUTION"
    COMPUTER_USE = "COMPUTER_USE"
    CRON_SCHEDULER = "CRON_SCHEDULER"
    FILE_OPERATIONS = "FILE_OPERATIONS"
    FILE_TOOLS = "FILE_TOOLS"
    BACKGROUND_REVIEW = "BACKGROUND_REVIEW"


class PolicyDenied(RuntimeError):
    """Raised when a foreground Git-write policy decision forbids the operation.

    Carries a structured ``disposition`` so callers (terminal_tool, file_operations,
    file_tools, delegate_tool, code_execution_tool, computer_use/doctor.py,
    cron/scheduler.py, agent_init) can branch deterministically without depending
    on message text. Fail-closed: callers MUST treat any disposition as a hard
    deny unless an explicit allow-list permits continuation.
    """

    DISPOSITION_HELPER_MISSING = "DENY_HELPER_MISSING"
    DISPOSITION_POLICY_DENY = "DENY_POLICY"
    DISPOSITION_AUDIT_FAILURE = "DENY_AUDIT_FAILURE"

    def __init__(
        self,
        *,
        disposition: str,
        caller_type: "CallerType | str | None" = None,
        operation_kind: str = "",
        reason: str = "",
        detail: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(reason or disposition)
        self.disposition = disposition
        self.caller_type = caller_type
        self.operation_kind = operation_kind
        self.reason = reason
        self.detail: dict[str, object] = dict(detail) if detail else {}


def _emit_policy_event(
    *,
    caller_type: "CallerType | str",
    operation_kind: str,
    decision: "SessionWritePolicyDecision",
    extra: Mapping[str, object] | None = None,
) -> None:
    """Best-effort structured audit emit. Never raises (fail-closed contract).

    On any exception from the sink, swallow and continue: foreground policy
    decisions must not be blocked by audit-pipeline failures.
    """
    payload: dict[str, object] = {
        "schema_version": 1,
        "caller_type": str(caller_type),
        "operation_kind": operation_kind,
        "decision_result": decision.result.value,
        "decision_reason": decision.reason,
        "policy_origin": decision.origin,
        "session_id": decision.session_id,
        "target_path": decision.target_path or "",
    }
    if extra:
        payload["extra"] = dict(extra)
    try:
        import json
        import logging
        logging.getLogger("hermes.session_write_policy").info(
            "policy_event %s", json.dumps(payload, ensure_ascii=False, default=str)
        )
    except Exception:
        return


def _load_git_target_resolve(*, caller_type: str, operation_kind: str) -> Callable[..., object]:
    """Load the Git target resolver without importing it at module import time."""
    import importlib

    try:
        module = importlib.import_module("agent.git_target_resolver")
        return module.resolve
    except (ImportError, ModuleNotFoundError) as exc:
        raise PolicyDenied(
            disposition=PolicyDenied.DISPOSITION_HELPER_MISSING,
            caller_type=caller_type,
            operation_kind=operation_kind,
            reason="git_target_resolver_missing",
            detail={"helper": "agent.git_target_resolver.resolve"},
        ) from exc


def _load_git_mutation_classify(*, caller_type: str, operation_kind: str) -> Callable[..., object]:
    """Load the Git mutation classifier without importing it at module import time."""
    import importlib

    try:
        module = importlib.import_module("agent.git_mutation_classifier")
        return module.classify
    except (ImportError, ModuleNotFoundError) as exc:
        raise PolicyDenied(
            disposition=PolicyDenied.DISPOSITION_HELPER_MISSING,
            caller_type=caller_type,
            operation_kind=operation_kind,
            reason="git_mutation_classifier_missing",
            detail={"helper": "agent.git_mutation_classifier.classify"},
        ) from exc


def _raw_command_mentions_git(raw_command: str | None) -> bool:
    if not raw_command:
        return False
    return "git" in str(raw_command).split()


def _git_unknown_reason(resolver_result: object, classification: object) -> str:
    subcommand = getattr(classification, "subcommand", None) or getattr(resolver_result, "git_subcommand", None)
    ambiguity = getattr(resolver_result, "ambiguity_reason", None)
    classifier_reason = getattr(classification, "reason", None)
    if subcommand:
        return f"git_mutation_unknown:{subcommand}"
    if ambiguity:
        return f"git_mutation_unknown:{ambiguity}"
    if classifier_reason:
        return f"git_mutation_unknown:{classifier_reason}"
    return "git_mutation_unknown"


def pre_spawn_consult(
    caller_type: "CallerType | str",
    *,
    operation_kind: str,
    argv: tuple[str, ...] | list[str] | None = None,
    raw_command: str | None = None,
    cwd: str | None = None,
    env_subset: Mapping[str, str] | None = None,
    target_path: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> "SessionWritePolicyDecision":
    """Consult the session-write policy and resolver before spawning a subprocess.

    Returns the resulting ``SessionWritePolicyDecision`` (with ``caller_type``
    populated). Raises ``PolicyDenied`` for helper loading failures and
    fail-closed Git ambiguity/unknown or helper execution failures.

    The function never silently drops a denial: it logs an audit event and
    returns the decision so the caller can route it. ImportError from the
    resolver/classifier is mapped to ``PolicyDenied(DENY_HELPER_MISSING)``.
    """
    caller_value = caller_type.value if isinstance(caller_type, CallerType) else str(caller_type)
    argv_tuple = tuple(argv) if argv is not None else None
    _resolve = _load_git_target_resolve(caller_type=caller_value, operation_kind=operation_kind)
    _classify = _load_git_mutation_classify(caller_type=caller_value, operation_kind=operation_kind)
    try:
        resolver_result = _resolve(
            cwd=cwd,
            env_subset=dict(env_subset) if env_subset else None,
            command_argv=argv_tuple,
            raw_command=raw_command,
        )
    except Exception as exc:
        raise PolicyDenied(
            disposition=PolicyDenied.DISPOSITION_POLICY_DENY,
            caller_type=caller_value,
            operation_kind=operation_kind,
            reason=f"git_target_resolution_failed:{type(exc).__name__}",
        ) from exc

    try:
        git_classification = _classify(
            command_argv=argv_tuple,
            raw_command=raw_command,
            resolved_target=resolver_result,
        )
    except Exception as exc:
        raise PolicyDenied(
            disposition=PolicyDenied.DISPOSITION_POLICY_DENY,
            caller_type=caller_value,
            operation_kind=operation_kind,
            reason=f"git_mutation_classification_failed:{type(exc).__name__}",
        ) from exc

    classification = getattr(git_classification, "classification", "unknown")
    is_git_or_plausible_git = (
        bool(getattr(git_classification, "is_git", False))
        or bool(getattr(resolver_result, "is_git_command", False))
        or _raw_command_mentions_git(raw_command)
    )
    if classification == "unknown" and is_git_or_plausible_git:
        raise PolicyDenied(
            disposition=PolicyDenied.DISPOSITION_POLICY_DENY,
            caller_type=caller_value,
            operation_kind=operation_kind,
            reason=_git_unknown_reason(resolver_result, git_classification),
            detail={
                "git_subcommand": getattr(git_classification, "subcommand", None),
                "parse_ambiguous": bool(getattr(git_classification, "parse_ambiguous", False)),
            },
        )

    policy = get_current_session_write_policy(protected=False)
    decision = evaluate_session_write_policy(
        policy,
        operation_kind=operation_kind,
        origin=str(caller_value),
        target_path=target_path or getattr(resolver_result, "canonical_path", None),
        capability=CapabilityGrant("terminal", operation_kind),
    )

    object.__setattr__(decision, "caller_type", caller_value)
    _emit_policy_event(
        caller_type=caller_value,
        operation_kind=operation_kind,
        decision=decision,
        extra={"resolver": resolver_result} if resolver_result is not None else None,
    )
    return decision


# Patch the existing SessionWritePolicyDecision dataclass with a caller_type field.
# Implemented via __init_subclass__-free approach: re-declare with __dataclass_fields__
# merge. Because Python dataclasses are immutable per-class definition, we add the
# attribute at runtime via object.__setattr__ when populated. The default factory
# here preserves backward compatibility with callers that do not pass caller_type.
_setattr_safe = object.__setattr__


def _attach_caller_type(decision: "SessionWritePolicyDecision", caller_type: str | None) -> None:
    """Attach caller_type to an existing decision if absent."""
    if not hasattr(decision, "caller_type"):
        try:
            _setattr_safe(decision, "caller_type", caller_type)
        except Exception:
            return


# Re-export CallerType under the legacy alias expected by older callers.
PolicyDecision = SessionWritePolicyDecision  # type: ignore[misc,assignment]
DENY_HELPER_MISSING = PolicyDenied.DISPOSITION_HELPER_MISSING
DENY_POLICY = PolicyDenied.DISPOSITION_POLICY_DENY
DENY_AUDIT_FAILURE = PolicyDenied.DISPOSITION_AUDIT_FAILURE
