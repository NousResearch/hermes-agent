"""Driver / Adapter contracts for Hermes Enterprise.

These ABCs are the extension seams of the platform:

    ComputeDriver  - provisions/observes the workload for an admitted revision
    SandboxDriver  - establishes + verifies containment before harness start
    SecretDriver   - performs brokered operations against a secret backend
    IAMAdapter     - authorizes exact actions on exact resources
    IdentityVerifier - verifies external identity evidence (OAG boundary)

Ownership rules (enforced by the controller, documented here for
implementers): drivers and adapters consume admitted intent at their
boundary. They never own platform resources, select themselves, grant
permissions, or rewrite revisions. Any failure or inability to verify is a
denial — implementations raise, they do not degrade.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .resources import Resource

# ---------------------------------------------------------------------------
# Identity / authorization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedIdentity:
    """Output of the OAG boundary: verified identity + admitted scope.

    Carries evidence, not permission. OCC authorization is always a separate,
    independent decision.
    """

    issuer: str
    subject: str            # immutable subject from the identity provider
    installation: str       # server-selected, never caller-selected
    namespace: str | None   # admitted namespace, when the request required one
    claims: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuthzRequest:
    """One exact authorization question."""

    principal: str          # resolved platform identity (never raw claims)
    principal_kind: str     # principal | service-principal | workload-identity
    action: str             # e.g. "openclaw.agents.deploy" -> ours: "hermes.agents.deploy"
    kind: str               # resource kind
    namespace: str | None
    resource: str | None    # exact resource name; None only for create/list scope checks


class IdentityVerifier(ABC):
    """Verifies external identity evidence and admits exact scope (OAG)."""

    @abstractmethod
    def verify(self, token: str, *, require_namespace: str | None = None) -> VerifiedIdentity:
        """Return a VerifiedIdentity or raise AdmissionError. Fail closed."""


class IAMAdapter(ABC):
    """Authoritative authorization for the resource kinds assigned to it."""

    name: str = "abstract"

    @abstractmethod
    def authorize(self, request: AuthzRequest) -> None:
        """Return None on allow; raise AuthorizationError (or
        RestrictionError) on deny. An unavailable authority must raise —
        never default-allow."""


# ---------------------------------------------------------------------------
# Compute / sandbox
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadRef:
    """Handle to one provisioned candidate workload."""

    revision_uid: str
    namespace: str
    workload_identity: str
    driver: str
    handle: dict[str, Any] = field(default_factory=dict)  # driver-specific ids


class ComputeDriver(ABC):
    """Provisions and observes infrastructure for an admitted AgentRevision."""

    name: str = "abstract"

    @abstractmethod
    def provision_candidate(self, revision: Resource) -> WorkloadRef:
        """Create a separate, non-serving candidate workload whose harness
        cannot start yet. Must not mutate the previously active workload."""

    @abstractmethod
    def workload_ready(self, ref: WorkloadRef) -> bool:
        """Infrastructure readiness for the candidate (not harness health)."""

    @abstractmethod
    def start_harness(self, ref: WorkloadRef) -> None:
        """Permit the harness to execute. Called only after containment is
        verified and the previous revision is retired."""

    @abstractmethod
    def stop_harness(self, ref: WorkloadRef) -> None:
        """Stop the harness and verify it has stopped."""

    @abstractmethod
    def teardown(self, ref: WorkloadRef) -> None:
        """Remove the workload's infrastructure."""


class SandboxDriver(ABC):
    """Enforces the exact admitted SandboxPolicy for one revision."""

    name: str = "abstract"

    @abstractmethod
    def supports(self, policy: dict[str, Any]) -> bool:
        """True only if the ENTIRE policy can be enforced. Partial support
        is unsupported; the controller rejects deployment."""

    @abstractmethod
    def enforce(self, ref: WorkloadRef, policy: dict[str, Any]) -> None:
        """Establish containment for the candidate workload. Raise
        DriverError if enforcement cannot be established."""

    @abstractmethod
    def verify(self, ref: WorkloadRef, policy: dict[str, Any]) -> None:
        """Independently verify enforcement is in place. Raise DriverError
        when verification is unavailable or ambiguous — unverifiable
        containment blocks activation."""


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


class SecretDriver(ABC):
    """Performs permitted operations against an external secret backend.

    The driver mediates; it never returns raw secret values to workloads.
    ``use`` executes an operation that needs the secret (e.g. signing a
    request, minting a scoped short-lived token) backend-side and returns
    only the operation result.
    """

    name: str = "abstract"

    @abstractmethod
    def exists(self, backend: dict[str, Any], key: str) -> bool:
        """Whether the backend can serve this key (no value retrieval)."""

    @abstractmethod
    def use(self, backend: dict[str, Any], key: str, operation: str,
            params: dict[str, Any]) -> dict[str, Any]:
        """Execute one permitted, secret-backed operation. The result must
        not contain the secret value or reusable backend credentials."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DriverRegistry:
    """Installation-owned selection of exactly one implementation per
    capability. Nothing can register over an existing selection, and lookup
    of an unselected capability fails closed."""

    def __init__(self) -> None:
        self._impls: dict[tuple[str, str], Any] = {}

    def select(self, capability: str, impl: Any) -> None:
        name = getattr(impl, "name", None)
        if not name or name == "abstract":
            raise ValueError("implementation must carry a concrete .name")
        key = (capability, name)
        if any(cap == capability for cap, _ in self._impls):
            raise ValueError(
                f"capability {capability!r} already has a selected "
                "implementation; changing selection requires explicit "
                "reconfiguration, not re-registration"
            )
        self._impls[key] = impl

    def get(self, capability: str) -> Any:
        matches = [impl for (cap, _), impl in self._impls.items() if cap == capability]
        if not matches:
            from .errors import DriverError

            raise DriverError(f"no implementation selected for {capability!r}")
        return matches[0]

    def selected_name(self, capability: str) -> str:
        return str(getattr(self.get(capability), "name"))
