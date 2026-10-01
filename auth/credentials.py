"""Credential request and selection identities, independent of pool/storage I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from auth.context import AuthContext, CredentialScope, _require_identifier
from auth.status import CredentialFailure, CredentialStatus


@dataclass(frozen=True)
class EndpointScope:
    """Exact canonical route identity supplied by the routing owner.

    endpoint is the resolved base URL for HTTP, or an explicit SDK/process
    route identity for a non-HTTP provider. No URL inference or normalization
    happens here. custom_provider_id is the durable configured identity,
    not a display name: two custom providers can share an endpoint.
    """

    endpoint: str = field(repr=False)
    custom_provider_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.endpoint, "endpoint")
        if self.custom_provider_id is not None:
            _require_identifier(self.custom_provider_id, "custom_provider_id")


@dataclass(frozen=True)
class CredentialRequest:
    """Credentials for an already-selected provider/route, never provider='auto'.

    provider_id is the canonical providers/ identity supplied by the caller.
    Unknown plugin IDs remain valid: constructing a request must not discover
    plugins, snapshot their registry, resolve aliases or select a provider.
    """

    provider_id: str
    endpoint: EndpointScope
    context: AuthContext
    model_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.provider_id, "provider_id")
        if self.provider_id.lower() == "auto":
            raise ValueError("provider_id must already be selected")
        if not isinstance(self.endpoint, EndpointScope):
            raise TypeError("endpoint must be an explicit EndpointScope")
        if not isinstance(self.context, AuthContext):
            raise TypeError("context must be AuthContext")
        if self.provider_id == "custom" or self.provider_id.startswith("custom:"):
            if self.endpoint.custom_provider_id is None:
                raise ValueError("custom routes require custom_provider_id")
        if self.model_id is not None:
            _require_identifier(self.model_id, "model_id")

    @property
    def scope(self) -> CredentialScope:
        return self.context.scope


@dataclass(frozen=True)
class CredentialIdentity:
    """Non-secret handle for feedback, independent of the pool's current cursor."""

    provider_id: str
    endpoint: EndpointScope
    scope: CredentialScope
    credential_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.provider_id, "provider_id")
        _require_identifier(self.credential_id, "credential_id")
        if not isinstance(self.endpoint, EndpointScope):
            raise TypeError("endpoint must be EndpointScope")
        if not isinstance(self.scope, CredentialScope):
            raise TypeError("scope must be CredentialScope")

    def matches(self, request: CredentialRequest) -> bool:
        return (
            self.provider_id == request.provider_id
            and self.endpoint == request.endpoint
            and self.scope == request.scope
        )


@dataclass(frozen=True)
class CredentialMaterial:
    """Request-time material, not an OAuth grant or a persisted pool row.

    Provider-specific options (e.g. a token callback or SDK credentials) remain
    opaque. Mapping copies prevent top-level caller mutation; nested provider
    objects retain their existing semantics. Never log/serialize this object.
    """

    api_key: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    client_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        object.__setattr__(
            self, "client_options", MappingProxyType(dict(self.client_options))
        )


@dataclass(frozen=True)
class CredentialLease:
    """Handle allocated by the existing pool, not a second lease allocator.

    lease_id may be the credential ID used by today's soft-lease API.
    Acquiring/releasing remains the pool's responsibility.
    """

    identity: CredentialIdentity
    lease_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CredentialIdentity):
            raise TypeError("identity must be CredentialIdentity")
        _require_identifier(self.lease_id, "lease_id")


@dataclass(frozen=True)
class CredentialSelection:
    """A successful selection captures identity, model context and runtime material.

    Construct only after required refresh persistence succeeds. A failed write
    must return PERSISTENCE_FAILED, never a success tagged as durable.
    """

    request: CredentialRequest
    identity: CredentialIdentity
    material: CredentialMaterial = field(repr=False, compare=False)
    source: str
    lease: CredentialLease | None = None

    def __post_init__(self) -> None:
        if not self.identity.matches(self.request):
            raise ValueError("credential identity does not match the request scope")
        if not isinstance(self.material, CredentialMaterial):
            raise TypeError("material must be CredentialMaterial")
        _require_identifier(self.source, "source")
        if self.lease is not None and self.lease.identity != self.identity:
            raise ValueError("lease belongs to a different credential identity")

    @property
    def status(self) -> CredentialStatus:
        return CredentialStatus.SELECTED


CredentialResult = CredentialSelection | CredentialFailure
