"""Consumer-neutral contracts for plugin-authorized API credentials."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

_IDENTIFIER_RE = re.compile(r"[^\x00-\x1f\x7f]{1,256}")
_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_ROUTE_RE = re.compile(r"/[A-Za-z0-9_{}./:-]*")
_BEARER_RE = re.compile(r"[^\x00-\x20\x7f]{1,8192}")


class APIServerOperation(StrEnum):
    """Finite transient-credential surface; all omitted routes stay administrator-only."""

    CAPABILITIES_READ = "capabilities.read"
    SESSIONS_RESOLVE = "sessions.resolve"
    SESSIONS_CREATE = "sessions.create"
    RUNS_CREATE = "runs.create"
    RUN_STATUS_READ = "runs.status.read"
    RUN_EVENTS_READ = "runs.events.read"


@dataclass(frozen=True, slots=True)
class AgentProfileId:
    value: str

    def __post_init__(self) -> None:
        _validate_identifier("agent profile ID", self.value)


@dataclass(frozen=True, slots=True)
class CredentialScopeId:
    value: str

    def __post_init__(self) -> None:
        _validate_identifier("credential scope ID", self.value)


def _validate_identifier(label: str, value: object) -> None:
    if not isinstance(value, str) or value != value.strip() or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} is invalid")


@dataclass(frozen=True, slots=True)
class CredentialAuthorizationRequest:
    """Bounded server-derived request metadata passed to the plugin authorizer."""

    bearer: str = field(repr=False)
    method: str
    canonical_route: str
    operation: APIServerOperation

    def __post_init__(self) -> None:
        if not isinstance(self.bearer, str) or not _BEARER_RE.fullmatch(self.bearer):
            raise ValueError("bearer credential is invalid")
        if self.method not in {"GET", "POST"}:
            raise ValueError("method is not normalized")
        if not isinstance(self.canonical_route, str) or not _ROUTE_RE.fullmatch(self.canonical_route):
            raise ValueError("canonical route is invalid")
        if type(self.operation) is not APIServerOperation:
            raise TypeError("operation must be an APIServerOperation")


@dataclass(frozen=True, slots=True)
class AuthorizedAPICredential:
    """Strict immutable server-derived identity accepted by the API server."""

    principal_id: str
    runtime_profile: str
    agent_profile_id: AgentProfileId
    credential_scope_id: CredentialScopeId
    allowed_operations: frozenset[APIServerOperation]

    def __post_init__(self) -> None:
        _validate_identifier("principal ID", self.principal_id)
        if not isinstance(self.runtime_profile, str) or not _PROFILE_RE.fullmatch(self.runtime_profile):
            raise ValueError("runtime profile is invalid")
        if type(self.agent_profile_id) is not AgentProfileId:
            raise TypeError("agent_profile_id must be an AgentProfileId")
        if type(self.credential_scope_id) is not CredentialScopeId:
            raise TypeError("credential_scope_id must be a CredentialScopeId")
        if type(self.allowed_operations) is not frozenset or not self.allowed_operations:
            raise TypeError("allowed_operations must be a non-empty frozenset")
        if any(type(operation) is not APIServerOperation for operation in self.allowed_operations):
            raise TypeError("allowed_operations must contain only APIServerOperation values")


class APIServerCredentialAuthorizer(Protocol):
    """Plugin contract; ``authorize`` must be an async method (``async def authorize``)."""

    async def authorize(
        self, request: CredentialAuthorizationRequest
    ) -> AuthorizedAPICredential | None: ...
