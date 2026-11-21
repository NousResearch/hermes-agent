"""Platform resource model for Hermes Enterprise.

v1 resources (mirroring the platform spec):

    Installation-scoped:  Harness, Restriction (optionally), IAM config
    Namespace-scoped:     Configuration, Agent, AgentRevision, Channel,
                          Secret, SecretBroker, SandboxPolicy, Restriction

Design rules encoded here:
  * Every resource belongs to the Installation or to exactly one Namespace.
  * Namespace-scoped references never cross the Namespace boundary
    (validated in the store, not just documented).
  * AgentRevision is an immutable snapshot: it embeds resolved copies of the
    configuration and policy that were admitted, not live references.
  * Secret resources carry no secret value, ever.
  * Resources are plain dataclasses serialized to/from JSON dicts so the
    store, CLI, and controller share one wire shape.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .errors import ValidationError

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")  # DNS-1123 label


def validate_name(name: str, kind: str = "resource") -> str:
    """Names are DNS-1123 labels so they can back Kubernetes objects 1:1."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValidationError(
            f"invalid {kind} name {name!r}: must be a lowercase DNS-1123 label "
            "(a-z, 0-9, '-', max 63 chars, no leading/trailing '-')"
        )
    return name


def new_uid() -> str:
    return uuid.uuid4().hex


def now_ts() -> float:
    return time.time()


class Scope(str, Enum):
    INSTALLATION = "installation"
    NAMESPACE = "namespace"


class Kind(str, Enum):
    NAMESPACE = "Namespace"
    CONFIGURATION = "Configuration"
    AGENT = "Agent"
    AGENT_REVISION = "AgentRevision"
    HARNESS = "Harness"
    CHANNEL = "Channel"
    SECRET = "Secret"
    SECRET_BROKER = "SecretBroker"
    SANDBOX_POLICY = "SandboxPolicy"
    RESTRICTION = "Restriction"


#: Which scope each kind lives in. Restriction may live in either.
KIND_SCOPES: dict[Kind, tuple[Scope, ...]] = {
    Kind.NAMESPACE: (Scope.INSTALLATION,),
    Kind.CONFIGURATION: (Scope.NAMESPACE,),
    Kind.AGENT: (Scope.NAMESPACE,),
    Kind.AGENT_REVISION: (Scope.NAMESPACE,),
    Kind.HARNESS: (Scope.INSTALLATION,),
    Kind.CHANNEL: (Scope.NAMESPACE,),
    Kind.SECRET: (Scope.NAMESPACE,),
    Kind.SECRET_BROKER: (Scope.NAMESPACE,),
    Kind.SANDBOX_POLICY: (Scope.NAMESPACE,),
    Kind.RESTRICTION: (Scope.INSTALLATION, Scope.NAMESPACE),
}


@dataclass
class ResourceMeta:
    """Identity + lifecycle metadata shared by every platform resource."""

    kind: str
    name: str
    uid: str = field(default_factory=new_uid)
    namespace: str | None = None  # None => installation-scoped
    generation: int = 1  # bumped on every spec change (optimistic concurrency)
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    labels: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        kind = Kind(self.kind)
        validate_name(self.name, kind.value)
        scopes = KIND_SCOPES[kind]
        if self.namespace is None and Scope.INSTALLATION not in scopes:
            raise ValidationError(f"{kind.value} {self.name!r} requires a namespace")
        if self.namespace is not None:
            if Scope.NAMESPACE not in scopes:
                raise ValidationError(
                    f"{kind.value} {self.name!r} is installation-scoped and "
                    "cannot carry a namespace"
                )
            validate_name(self.namespace, "Namespace")


@dataclass
class Resource:
    """Envelope: metadata + kind-specific spec + controller-owned status."""

    meta: ResourceMeta
    spec: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"meta": asdict(self.meta), "spec": self.spec, "status": self.status}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Resource":
        return cls(
            meta=ResourceMeta(**data["meta"]),
            spec=dict(data.get("spec") or {}),
            status=dict(data.get("status") or {}),
        )

    def validate(self) -> None:
        self.meta.validate()
        validator = _SPEC_VALIDATORS.get(Kind(self.meta.kind))
        if validator is not None:
            validator(self)


# ---------------------------------------------------------------------------
# Per-kind spec validation
# ---------------------------------------------------------------------------

def _require(spec: dict[str, Any], key: str, typ: type, kind: str) -> Any:
    val = spec.get(key)
    if not isinstance(val, typ) or (typ is str and not val):
        raise ValidationError(f"{kind}.spec.{key} must be a non-empty {typ.__name__}")
    return val


def _validate_namespace(res: Resource) -> None:
    # Namespace spec is minimal in v1; phase/readiness live in status.
    phase = res.status.get("phase", NamespacePhase.PENDING.value)
    NamespacePhase(phase)


def _validate_configuration(res: Resource) -> None:
    _require(res.spec, "config", dict, "Configuration")
    forbidden = _find_secretlike_keys(res.spec["config"])
    if forbidden:
        raise ValidationError(
            "Configuration must not embed secret values; found secret-like "
            f"keys: {sorted(forbidden)}. Reference a Secret resource instead."
        )


_SECRETLIKE = re.compile(r"(api[_-]?key|token|password|secret|credential)", re.I)
_REFERENCE_KEYS = re.compile(r"^(secretRef|secret_ref|secretName|secret_name)$")


def _find_secretlike_keys(cfg: Any, path: str = "") -> set[str]:
    hits: set[str] = set()
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            p = f"{path}.{k}" if path else str(k)
            # {"secretRef": "<resource name>"} is the sanctioned reference
            # shape — the value is a resource NAME, not a secret.
            if isinstance(v, str) and _REFERENCE_KEYS.match(str(k)):
                continue
            # A *string value* under a secret-like key is an embedded secret.
            # A dict/reference shape ({"secretRef": name}) is fine.
            if isinstance(v, str) and v and _SECRETLIKE.search(str(k)):
                hits.add(p)
            hits |= _find_secretlike_keys(v, p)
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            hits |= _find_secretlike_keys(v, f"{path}[{i}]")
    return hits


def _validate_agent(res: Resource) -> None:
    spec = res.spec
    _require(spec, "harness", str, "Agent")            # Harness name (installation)
    _require(spec, "configuration", str, "Agent")      # Configuration name (ns)
    for key in ("channels", "secrets"):
        vals = spec.get(key, [])
        if not isinstance(vals, list) or not all(isinstance(v, str) and v for v in vals):
            raise ValidationError(f"Agent.spec.{key} must be a list of resource names")
    sandbox = spec.get("sandboxPolicy")
    if sandbox is not None and (not isinstance(sandbox, str) or not sandbox):
        raise ValidationError("Agent.spec.sandboxPolicy must be a resource name")


def _validate_agent_revision(res: Resource) -> None:
    spec = res.spec
    _require(spec, "agent", str, "AgentRevision")
    _require(spec, "agentUid", str, "AgentRevision")
    _require(spec, "workloadIdentity", str, "AgentRevision")
    _require(spec, "harness", dict, "AgentRevision")          # snapshot {name, version}
    _require(spec, "configuration", dict, "AgentRevision")    # snapshot of config contents
    _require(spec, "computeDriver", str, "AgentRevision")     # pinned implementation
    _require(spec, "sandboxDriver", str, "AgentRevision")     # pinned implementation
    if "sandboxPolicy" in spec and not isinstance(spec["sandboxPolicy"], dict):
        raise ValidationError("AgentRevision.spec.sandboxPolicy must be a snapshot dict")
    phase = res.status.get("phase", RevisionPhase.CANDIDATE.value)
    RevisionPhase(phase)


def _validate_harness(res: Resource) -> None:
    _require(res.spec, "version", str, "Harness")
    _require(res.spec, "image", str, "Harness")  # OCI image ref for the runtime


def _validate_channel(res: Resource) -> None:
    _require(res.spec, "platform", str, "Channel")  # telegram/discord/slack/...


def _validate_secret(res: Resource) -> None:
    _require(res.spec, "broker", str, "Secret")     # SecretBroker name (same ns)
    _require(res.spec, "key", str, "Secret")        # backend key/path identifier
    if "value" in res.spec or "data" in res.spec:
        raise ValidationError("Secret resources must not contain secret values")


def _validate_secret_broker(res: Resource) -> None:
    _require(res.spec, "driver", str, "SecretBroker")   # selected SecretDriver
    backend = res.spec.get("backend", {})
    if not isinstance(backend, dict):
        raise ValidationError("SecretBroker.spec.backend must be a dict")
    if _find_secretlike_keys(backend):
        raise ValidationError(
            "SecretBroker.spec.backend must reference backend credentials "
            "out-of-band; it cannot embed them"
        )


def _validate_sandbox_policy(res: Resource) -> None:
    spec = res.spec
    network = spec.get("network", "isolated")
    if network not in ("isolated", "egress-allowlist", "open"):
        raise ValidationError(
            "SandboxPolicy.spec.network must be one of "
            "'isolated', 'egress-allowlist', 'open'"
        )
    if network == "egress-allowlist":
        allow = spec.get("egressAllow", [])
        if not isinstance(allow, list) or not allow:
            raise ValidationError(
                "SandboxPolicy with network=egress-allowlist requires a "
                "non-empty spec.egressAllow list"
            )
    for key in ("readOnlyRootFilesystem", "allowPrivilegeEscalation"):
        if key in spec and not isinstance(spec[key], bool):
            raise ValidationError(f"SandboxPolicy.spec.{key} must be a bool")


def _validate_restriction(res: Resource) -> None:
    _require(res.spec, "rule", dict, "Restriction")
    rule = res.spec["rule"]
    if not isinstance(rule.get("deny"), list) or not rule["deny"]:
        raise ValidationError(
            "Restriction.spec.rule.deny must be a non-empty list of "
            "'<action>:<kind>[:<name>]' patterns; Restrictions can only "
            "narrow, never grant"
        )


_SPEC_VALIDATORS = {
    Kind.NAMESPACE: _validate_namespace,
    Kind.CONFIGURATION: _validate_configuration,
    Kind.AGENT: _validate_agent,
    Kind.AGENT_REVISION: _validate_agent_revision,
    Kind.HARNESS: _validate_harness,
    Kind.CHANNEL: _validate_channel,
    Kind.SECRET: _validate_secret,
    Kind.SECRET_BROKER: _validate_secret_broker,
    Kind.SANDBOX_POLICY: _validate_sandbox_policy,
    Kind.RESTRICTION: _validate_restriction,
}


# ---------------------------------------------------------------------------
# Lifecycle phases
# ---------------------------------------------------------------------------

class NamespacePhase(str, Enum):
    PENDING = "Pending"        # created, backing infra not ready
    READY = "Ready"            # backing k8s namespace + gateway ready
    FAILED = "Failed"          # reconcile failed; cannot admit deployments
    TERMINATING = "Terminating"


class RevisionPhase(str, Enum):
    CANDIDATE = "Candidate"    # created; workload may be provisioning
    CONTAINED = "Contained"    # sandbox enforcement verified
    ACTIVE = "Active"          # sole active revision; serving
    RETIRED = "Retired"        # superseded; harness stopped
    FAILED = "Failed"          # never activated (previous revision unchanged)


#: Namespace-scoped reference fields per kind: (spec key, target kind, is_list)
REFERENCE_FIELDS: dict[Kind, tuple[tuple[str, Kind, bool], ...]] = {
    Kind.AGENT: (
        ("configuration", Kind.CONFIGURATION, False),
        ("channels", Kind.CHANNEL, True),
        ("secrets", Kind.SECRET, True),
        ("sandboxPolicy", Kind.SANDBOX_POLICY, False),
    ),
    Kind.SECRET: (("broker", Kind.SECRET_BROKER, False),),
}
