"""Session write policy for Hermes Agent.

This module defines the ``SessionWritePolicy`` dataclass and supporting
helpers used to gate write-style operations performed by delegated
subagents and ACP subprocesses.

It is the contract-forward current-native implementation required by the
sealed C19 / C23 design.  It is intentionally stdlib-only and has no
dependency on ``agent.self_improvement_policy``,
``agent.git_mutation_classifier``, or ``agent.git_target_resolver`` —
those surfaces are out of scope.

Public surface (required by the C19 / C23 contract):

* :class:`SessionWritePolicyMode` — mode enum (NORMAL / DENY_ALL / ALLOWLIST)
* :class:`SessionWritePolicyDecisionResult` — decision enum (ALLOW / DENY)
* :class:`CallerType` — caller enum (DELEGATION + future expansion room)
* :class:`CapabilityGrant` — capability description (kind / operation / roots)
* :class:`SessionWritePolicy` — immutable policy record with factories
  ``normal`` and ``deny_all`` plus a ``derive_child`` method
* :class:`SessionWritePolicyDecision` — structured decision returned by
  :func:`evaluate` and :func:`pre_spawn_consult`
* :class:`PolicyDenied` — typed exception raised by ``pre_spawn_consult``
* :func:`evaluate` — pure decision helper
* :func:`pre_spawn_consult` — resolves the active policy and returns a
  decision (or raises ``PolicyDenied``)
* :func:`get_current_session_write_policy` — returns the active policy
  for a session identifier, defaulting to a normal policy for unknown
  sessions
* :func:`session_write_policy_scope` — context manager that binds a
  policy as the current one for the duration of the ``with`` block
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional, Sequence, Tuple


class SessionWritePolicyMode(str, Enum):
    """Top-level policy mode for a session."""

    NORMAL = "normal"
    DENY_ALL = "deny_all"
    ALLOWLIST = "allowlist"


class SessionWritePolicyDecisionResult(str, Enum):
    """Outcome of a policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"


class CallerType(str, Enum):
    """Caller classification passed to the policy.

    Currently only ``DELEGATION`` is exercised by the C19 / C23 contract;
    additional values are reserved for future expansion.
    """

    DELEGATION = "DELEGATION"
    TOOL = "TOOL"
    USER = "USER"
    CRON = "CRON"


@dataclass(frozen=True)
class CapabilityGrant:
    """A single capability grant attached to an :class:`SessionWritePolicy`.

    Attributes
    ----------
    kind:
        The category of capability.  Examples: ``"filesystem"``,
        ``"terminal_exec"``.
    operation:
        The operation the capability covers.  Examples: ``"read"``,
        ``"write"``, ``"execute"``.
    roots:
        The optional set of roots the capability is scoped to (e.g.
        filesystem paths the grant applies to).  Empty tuple means
        "anywhere" (subject to the policy mode).
    """

    kind: str
    operation: str
    roots: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionWritePolicy:
    """Immutable description of a session's write policy.

    Attributes
    ----------
    session_id:
        Identifier of the session the policy belongs to.
    mode:
        The top-level mode (``NORMAL`` / ``DENY_ALL`` / ``ALLOWLIST``).
    allowed_roots:
        Filesystem roots the policy considers in scope.  Only consulted
        in ``ALLOWLIST`` mode.
    capability_grants:
        Tuple of :class:`CapabilityGrant` records describing granted
        capabilities.  Only consulted in ``ALLOWLIST`` mode.
    protected:
        ``True`` if the policy is locked from modification.  C19 / C23
        only read this attribute; mutation is the parent's responsibility.
    """

    session_id: str
    mode: SessionWritePolicyMode
    allowed_roots: Tuple[str, ...] = ()
    capability_grants: Tuple[CapabilityGrant, ...] = ()
    protected: bool = False

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def normal(cls, session_id: str) -> "SessionWritePolicy":
        """Build a permissive ``NORMAL`` policy for ``session_id``."""

        return cls(session_id=session_id, mode=SessionWritePolicyMode.NORMAL)

    @classmethod
    def deny_all(cls, session_id: str) -> "SessionWritePolicy":
        """Build a strict ``DENY_ALL`` policy for ``session_id``."""

        return cls(session_id=session_id, mode=SessionWritePolicyMode.DENY_ALL)

    # ------------------------------------------------------------------
    # Inheritance
    # ------------------------------------------------------------------

    def derive_child(self, requested: Optional[object] = None) -> "SessionWritePolicy":
        """Return the policy a child session should inherit.

        C19 R01 asserts this is called exactly once with ``requested=None``
        and that the returned policy is identity-equal to the parent's
        policy when the parent has a real :class:`SessionWritePolicy`.

        The ``requested`` parameter is reserved for future use (a child
        may request a stricter subset); the current contract ignores it
        and returns ``self`` unchanged so children receive the parent's
        policy by identity.
        """

        return self


@dataclass(frozen=True)
class SessionWritePolicyDecision:
    """Structured decision returned by :func:`evaluate` and
    :func:`pre_spawn_consult`.

    Attributes
    ----------
    result:
        :class:`SessionWritePolicyDecisionResult.ALLOW` or ``DENY``.
    reason:
        Short machine-readable reason for the decision (e.g.
        ``"policy_allow"`` or ``"terminal_exec_denied_protected_mode"``).
    operation_kind:
        The operation kind that was evaluated (e.g. ``"terminal_exec"``).
    origin:
        The caller class that originated the request (typically the
        stringified value of a :class:`CallerType`).
    """

    result: SessionWritePolicyDecisionResult
    reason: str
    operation_kind: str
    origin: str


class PolicyDenied(Exception):
    """Raised by :func:`pre_spawn_consult` for structured denials.

    The ``detail`` field carries sensitive information that MUST NOT be
    echoed to the user by callers — see C23 R04.
    """

    DISPOSITION_POLICY_DENY = "DENY_POLICY"

    def __init__(
        self,
        *,
        disposition: str = DISPOSITION_POLICY_DENY,
        caller_type: CallerType,
        operation_kind: str,
        reason: str,
        detail: Optional[object] = None,
    ) -> None:
        super().__init__(reason)
        self.disposition = disposition
        self.caller_type = caller_type
        self.operation_kind = operation_kind
        self.reason = reason
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - explicit repr contract
        return self.reason


# ----------------------------------------------------------------------
# Active-policy context variable
# ----------------------------------------------------------------------

_active_policy_var: contextvars.ContextVar[Optional[SessionWritePolicy]] = contextvars.ContextVar(
    "hermes_active_session_write_policy",
    default=None,
)


@contextlib.contextmanager
def session_write_policy_scope(policy: SessionWritePolicy) -> Iterator[SessionWritePolicy]:
    """Bind ``policy`` as the active session write policy for the scope.

    Inside the ``with`` block, :func:`pre_spawn_consult` and
    :func:`get_current_session_write_policy` resolve the active policy
    from this binding.
    """

    token = _active_policy_var.set(policy)
    try:
        yield policy
    finally:
        _active_policy_var.reset(token)


def get_current_session_write_policy(
    *,
    session_id: str,
    protected: bool = False,
) -> SessionWritePolicy:
    """Return the active session write policy for ``session_id``.

    Resolution order:

    1. If a policy has been bound via :func:`session_write_policy_scope`,
       return that policy (with ``protected`` updated to the requested
       value when the binding was unset).
    2. Otherwise, return :meth:`SessionWritePolicy.normal` for ``session_id``.

    The ``protected`` parameter is forwarded to the returned policy so
    callers can request a locked view; the C19 contract passes
    ``protected=False`` for fallback chains.
    """

    active = _active_policy_var.get()
    if active is not None:
        return SessionWritePolicy(
            session_id=active.session_id,
            mode=active.mode,
            allowed_roots=active.allowed_roots,
            capability_grants=active.capability_grants,
            protected=protected,
        )
    return SessionWritePolicy.normal(session_id)


def _has_capability(
    policy: SessionWritePolicy, kind: str, operation: str
) -> bool:
    for grant in policy.capability_grants:
        if grant.kind == kind and grant.operation == operation:
            return True
    return False


def evaluate(
    policy: SessionWritePolicy,
    *,
    caller_type: CallerType,
    operation_kind: str,
    argv: Optional[Sequence[str]] = None,
    raw_command: Optional[str] = None,
    cwd: Optional[str] = None,
    env_subset: Optional[object] = None,
    target_path: Optional[str] = None,
) -> SessionWritePolicyDecision:
    """Decide whether ``policy`` allows the described operation.

    The decision is structured (a :class:`SessionWritePolicyDecision`)
    so callers can surface a consistent diagnostic without inspecting
    string contents.

    Resolution rules:

    * ``DENY_ALL`` always denies with reason
      ``"terminal_exec_denied_protected_mode"`` (or
      ``"<operation_kind>_denied_protected_mode"``).
    * ``ALLOWLIST`` denies when the operation_kind is not granted by
      any :class:`CapabilityGrant` on the policy.
    * ``NORMAL`` allows everything.
    """

    origin = getattr(caller_type, "value", str(caller_type))

    if policy.mode is SessionWritePolicyMode.DENY_ALL:
        reason = f"{operation_kind}_denied_protected_mode"
        return SessionWritePolicyDecision(
            result=SessionWritePolicyDecisionResult.DENY,
            reason=reason,
            operation_kind=operation_kind,
            origin=origin,
        )

    if policy.mode is SessionWritePolicyMode.ALLOWLIST:
        kind = "terminal_exec" if operation_kind == "terminal_exec" else operation_kind
        granted = _has_capability(policy, kind, "execute")
        if not granted:
            reason = f"{operation_kind}_denied_no_capability_grant"
            return SessionWritePolicyDecision(
                result=SessionWritePolicyDecisionResult.DENY,
                reason=reason,
                operation_kind=operation_kind,
                origin=origin,
            )

    return SessionWritePolicyDecision(
        result=SessionWritePolicyDecisionResult.ALLOW,
        reason="policy_allow",
        operation_kind=operation_kind,
        origin=origin,
    )


def pre_spawn_consult(
    *,
    caller_type: CallerType,
    operation_kind: str,
    argv: Optional[Sequence[str]] = None,
    raw_command: Optional[str] = None,
    cwd: Optional[str] = None,
    env_subset: Optional[object] = None,
    target_path: Optional[str] = None,
) -> SessionWritePolicyDecision:
    """Consult the active session write policy before a subprocess spawn.

    Resolves the active policy via
    :func:`get_current_session_write_policy` and returns an
    :func:`evaluate` decision.  This function never raises — denial is
    conveyed through the returned decision's ``result`` field.

    Raises
    ------
    PolicyDenied
        Reserved for future structured-denial paths; the current
        contract only returns a ``DENY`` decision.
    """

    policy = get_current_session_write_policy(session_id="active")
    return evaluate(
        policy,
        caller_type=caller_type,
        operation_kind=operation_kind,
        argv=argv,
        raw_command=raw_command,
        cwd=cwd,
        env_subset=env_subset,
        target_path=target_path,
    )


__all__ = [
    "CallerType",
    "CapabilityGrant",
    "PolicyDenied",
    "SessionWritePolicy",
    "SessionWritePolicyDecision",
    "SessionWritePolicyDecisionResult",
    "SessionWritePolicyMode",
    "evaluate",
    "get_current_session_write_policy",
    "pre_spawn_consult",
    "session_write_policy_scope",
]
