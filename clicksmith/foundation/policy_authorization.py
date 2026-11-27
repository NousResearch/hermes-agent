"""FND-04 Policy & Authorization Boundary — Phase 1.

Standalone, deterministic authorization semantics for the Clicksmith Control
Plane. FND-04 evaluates only explicitly supplied inputs and does not
authenticate principals, persist decisions, or perform infrastructure I/O.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from .execution_identity import ExecutionIdentity, ExecutionIdentityError

SCHEMA_VERSION = "1.0"
DECISIONS = frozenset({"ALLOW", "DENY"})
ERROR_CODES = frozenset({
    "INVALID_IDENTITY", "INVALID_TASK_CONTRACT", "INVALID_POLICY_INPUT",
    "INVALID_AUTHORITY_CONTEXT", "POLICY_EVALUATION_ERROR", "UNSUPPORTED_POLICY",
    "SCHEMA_ERROR", "AMBIGUOUS_AUTHORIZATION",
})


class PolicyAuthorizationError(ValueError):
    """Base FND-04 boundary error."""
    code = "POLICY_EVALUATION_ERROR"


class InvalidIdentityError(PolicyAuthorizationError):
    code = "INVALID_IDENTITY"


class InvalidTaskContractError(PolicyAuthorizationError):
    code = "INVALID_TASK_CONTRACT"


class InvalidPolicyInputError(PolicyAuthorizationError):
    code = "INVALID_POLICY_INPUT"


class InvalidAuthorityContextError(PolicyAuthorizationError):
    code = "INVALID_AUTHORITY_CONTEXT"


class PolicyEvaluationError(PolicyAuthorizationError):
    code = "POLICY_EVALUATION_ERROR"


class UnsupportedPolicyError(PolicyAuthorizationError):
    code = "UNSUPPORTED_POLICY"


class SchemaError(PolicyAuthorizationError):
    code = "SCHEMA_ERROR"


class AmbiguousAuthorizationError(PolicyAuthorizationError):
    code = "AMBIGUOUS_AUTHORIZATION"


@dataclasses.dataclass(frozen=True)
class Policy:
    """Explicit Phase 1 policy configuration; no external discovery."""
    policy_id: str
    policy_version: str
    allowed: frozenset[str]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        _nonempty("policy_id", self.policy_id)
        _nonempty("policy_version", self.policy_version)
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}.")
        if not isinstance(self.allowed, frozenset):
            raise InvalidPolicyInputError("policy.allowed must be a frozenset.")
        if any(not isinstance(item, str) or not item.strip() for item in self.allowed):
            raise InvalidPolicyInputError("policy.allowed contains an invalid action.")


@dataclasses.dataclass(frozen=True)
class AuthorityContext:
    """Authority facts supplied by an upstream boundary."""
    authority_reference: str
    granted_actions: frozenset[str]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        _nonempty("authority_reference", self.authority_reference)
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}.")
        if not isinstance(self.granted_actions, frozenset):
            raise InvalidAuthorityContextError("granted_actions must be a frozenset.")
        if any(not isinstance(item, str) or not item.strip() for item in self.granted_actions):
            raise InvalidAuthorityContextError("granted_actions contains an invalid action.")


@dataclasses.dataclass(frozen=True)
class AuthorizationRequest:
    task: Any
    identity: ExecutionIdentity
    policy: Policy
    authority_context: AuthorityContext

    def validate(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise InvalidIdentityError("identity must be an ExecutionIdentity.")
        try:
            self.identity.validate()
        except ExecutionIdentityError as exc:
            raise InvalidIdentityError(str(exc)) from exc
        _validate_task(self.task, self.identity)
        if not isinstance(self.policy, Policy):
            raise InvalidPolicyInputError("policy must be a Policy.")
        self.policy.validate()
        if not isinstance(self.authority_context, AuthorityContext):
            raise InvalidAuthorityContextError(
                "authority_context must be an AuthorityContext."
            )
        self.authority_context.validate()


@dataclasses.dataclass(frozen=True)
class AuthorizationDecision:
    identity: ExecutionIdentity
    decision: str
    policy_id: str
    policy_version: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise InvalidIdentityError("decision identity must be an ExecutionIdentity.")
        try:
            self.identity.validate()
        except ExecutionIdentityError as exc:
            raise InvalidIdentityError(str(exc)) from exc
        if self.decision not in DECISIONS:
            raise SchemaError("decision must be exactly ALLOW or DENY.")
        _nonempty("policy_id", self.policy_id)
        _nonempty("policy_version", self.policy_version)
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "decision": self.decision,
            "identity": self.identity.to_dict(),
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evaluate(request: AuthorizationRequest) -> AuthorizationDecision:
    """Evaluate one explicit request; errors never become decisions."""
    if not isinstance(request, AuthorizationRequest):
        raise SchemaError("request must be an AuthorizationRequest.")
    try:
        request.validate()
        policy_actions = request.policy.allowed
        authority_actions = request.authority_context.granted_actions
        if not policy_actions:
            raise UnsupportedPolicyError("empty allowed policy is unsupported in Phase 1.")
        if not policy_actions.issubset(authority_actions):
            result = "DENY"
        else:
            result = "ALLOW"
        decision = AuthorizationDecision(
            identity=request.identity,
            decision=result,
            policy_id=request.policy.policy_id,
            policy_version=request.policy.policy_version,
        )
        decision.validate()
        return decision
    except PolicyAuthorizationError:
        raise
    except Exception as exc:
        raise PolicyEvaluationError("policy evaluation failed.") from exc


def _validate_task(task: Any, identity: ExecutionIdentity) -> None:
    if task is None:
        raise InvalidTaskContractError("task is required.")
    task_id = getattr(task, "task_id", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise InvalidTaskContractError("task must expose a non-empty task_id.")
    if task_id != identity.task_id:
        raise InvalidTaskContractError("task.task_id must match identity.task_id.")
    validate = getattr(task, "validate", None)
    if not callable(validate):
        raise InvalidTaskContractError("task must expose validate().")
    try:
        validate()
    except Exception as exc:
        raise InvalidTaskContractError("task contract validation failed.") from exc


def _nonempty(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string.")
