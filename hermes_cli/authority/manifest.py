"""Authority policy manifest compiler and admission evaluator.

This module is deliberately transport-free. It compiles a closed policy
manifest into a normalized representation and evaluates whether an operation
is policy-admitted. It does not execute effects, mint carriers, or settle
operations; those belong to later Authority Execution Layer shards.

The canonical S1 policy is embedded below as one immutable runtime artifact.
Consumers import that artifact instead of locating and reparsing a repository
file, so packaged Python installations retain the exact bytes, digest, sink
ownership, and compiled policy that later shards bind into proof objects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

KNOWN_DOMAINS = frozenset(
    {
        "github.operation",
        "sandbox.execution",
        "child_process.operation",
        "gateway.mutation",
    }
)
KNOWN_ACTOR_CLASSES = frozenset(
    {"human", "github_app", "workflow", "automation", "external_bot"}
)
_TOP_LEVEL_KEYS = frozenset({"schema_version", "policy_version", "domains"})
_DOMAIN_KEYS = frozenset({"sinks", "operation_classes"})
_SINK_KEYS = frozenset({"broker", "direct_symbols"})
_OPERATION_KEYS = frozenset(
    {
        "sink_class",
        "required_capabilities",
        "allowed_actor_classes",
        "allowed_resource_states",
    }
)
_REQUEST_KEYS = frozenset(
    {"domain", "operation_class", "actor_class", "resource_state", "capabilities"}
)
_CAPABILITY_GRANT_KEYS = frozenset(
    {"capability", "granted", "source", "generation"}
)

_CANONICAL_MANIFEST_BYTES = b'''{
  "schema_version": 1,
  "policy_version": "2026.08.25-v2",
  "domains": {
    "github.operation": {
      "sinks": {
        "github.issue.metadata.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "issues.addLabels",
            "issues.removeLabel",
            "issues.update"
          ]
        },
        "github.comment.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "issues.createComment",
            "pulls.createReview",
            "pulls.createReviewComment"
          ]
        },
        "github.contents.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "repos.createOrUpdateFileContents",
            "repos.deleteFile"
          ]
        },
        "github.gitdata.write": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "git.createBlob",
            "git.createCommit",
            "git.createRef",
            "git.createTree",
            "git.updateRef"
          ]
        },
        "github.pull_request.create": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "pulls.create"
          ]
        },
        "github.actions.dispatch": {
          "broker": "githubMutationBroker",
          "direct_symbols": [
            "actions.createWorkflowDispatch"
          ]
        }
      },
      "operation_classes": {
        "github.issue.metadata.write": {
          "sink_class": "github.issue.metadata.write",
          "required_capabilities": [
            "issues:metadata:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "open",
            "closed"
          ]
        },
        "github.comment.write": {
          "sink_class": "github.comment.write",
          "required_capabilities": [
            "comments:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "open",
            "closed"
          ]
        },
        "github.contents.write": {
          "sink_class": "github.contents.write",
          "required_capabilities": [
            "contents:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        },
        "github.gitdata.write": {
          "sink_class": "github.gitdata.write",
          "required_capabilities": [
            "git_objects:write",
            "refs:write"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        },
        "github.pull_request.create": {
          "sink_class": "github.pull_request.create",
          "required_capabilities": [
            "pull_requests:create"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        },
        "github.actions.dispatch": {
          "sink_class": "github.actions.dispatch",
          "required_capabilities": [
            "actions:dispatch"
          ],
          "allowed_actor_classes": [
            "human",
            "github_app",
            "workflow",
            "automation"
          ],
          "allowed_resource_states": [
            "current"
          ]
        }
      }
    }
  }
}
'''
_CANONICAL_MANIFEST_SHA256 = (
    "d427e3cd580f97a18103cbb73c04dc0baf2565f8509bcd298ee7aff049588d8e"
)


class ManifestValidationError(ValueError):
    """Raised when a manifest is not closed, explicit, and internally valid."""


class AdmissionRequestValidationError(ValueError):
    """Raised when an admission request is structurally malformed."""


@dataclass(frozen=True, slots=True)
class CapabilityGrant:
    capability: str
    granted: bool
    source: str
    generation: str


@dataclass(frozen=True, slots=True)
class CapabilityProof:
    capability: str
    source: str
    generation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "capability": self.capability,
            "source": self.source,
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    domain: str
    operation_class: str
    actor_class: str
    resource_state: str
    capabilities: tuple[CapabilityGrant, ...]


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    allowed: bool
    operation_class: str
    reason_code: str
    matched_rule: str | None
    capability_proofs: tuple[CapabilityProof, ...]
    policy_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "operation_class": self.operation_class,
            "reason_code": self.reason_code,
            "matched_rule": self.matched_rule,
            "capability_proofs": [proof.as_dict() for proof in self.capability_proofs],
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True, slots=True)
class CompiledSink:
    broker: str
    direct_symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledOperation:
    sink_class: str
    required_capabilities: tuple[str, ...]
    allowed_actor_classes: frozenset[str]
    allowed_resource_states: frozenset[str]


@dataclass(frozen=True, slots=True)
class CompiledDomain:
    sinks: Mapping[str, CompiledSink]
    operations: Mapping[str, CompiledOperation]


@dataclass(frozen=True, slots=True)
class CompiledManifest:
    schema_version: int
    policy_version: str
    domains: Mapping[str, CompiledDomain]


@dataclass(frozen=True, slots=True)
class AuthorityManifestArtifact:
    """Exact packaged bytes, digest, and the sole canonical compiled policy."""

    manifest_bytes: bytes
    manifest_sha256: str
    manifest: CompiledManifest


def _closed_key_error(
    value: Mapping[str, Any], expected: frozenset[str], where: str
) -> str | None:
    actual = frozenset(value)
    if actual == expected:
        return None
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    details: list[str] = []
    if unknown:
        details.append(f"unknown={unknown}")
    if missing:
        details.append(f"missing={missing}")
    return f"{where} must be closed ({', '.join(details)})"


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], where: str
) -> None:
    if error := _closed_key_error(value, expected, where):
        raise ManifestValidationError(error)


def _require_nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{where} must be a non-empty string")
    return value


def _require_unique_strings(
    value: Any, where: str, *, nonempty: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManifestValidationError(f"{where} must be an array")
    if nonempty and not value:
        raise ManifestValidationError(f"{where} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ManifestValidationError(f"{where} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise ManifestValidationError(f"{where} must not contain duplicates")
    return tuple(value)


def compile_authority_manifest(raw: Mapping[str, Any]) -> CompiledManifest:
    """Validate and compile a closed Authority Policy Manifest v1.

    Closed means no unknown top-level/domain/operation/sink fields, no unknown
    domains, no implicit capabilities, no wildcard grants, and no operation
    that points at an unregistered sink. Sink ownership metadata survives
    compilation so later shards never need a second raw-JSON parser.
    """

    if not isinstance(raw, Mapping):
        raise ManifestValidationError("manifest must be an object")
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, "manifest")

    schema_version = raw["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, (int, float))
        or schema_version != 1
    ):
        raise ManifestValidationError("schema_version must equal 1")
    policy_version = _require_nonempty_string(raw["policy_version"], "policy_version")

    raw_domains = raw["domains"]
    if not isinstance(raw_domains, Mapping) or not raw_domains:
        raise ManifestValidationError("domains must be a non-empty object")
    unknown_domains = sorted(set(raw_domains) - KNOWN_DOMAINS)
    if unknown_domains:
        raise ManifestValidationError(f"unknown domains: {unknown_domains}")

    domains: dict[str, CompiledDomain] = {}
    for domain_name in sorted(raw_domains):
        raw_domain = raw_domains[domain_name]
        if not isinstance(raw_domain, Mapping):
            raise ManifestValidationError(f"domains.{domain_name} must be an object")
        _require_exact_keys(raw_domain, _DOMAIN_KEYS, f"domains.{domain_name}")

        raw_sinks = raw_domain["sinks"]
        raw_operations = raw_domain["operation_classes"]
        if not isinstance(raw_sinks, Mapping) or not raw_sinks:
            raise ManifestValidationError(f"domains.{domain_name}.sinks must be non-empty")
        if not isinstance(raw_operations, Mapping) or not raw_operations:
            raise ManifestValidationError(
                f"domains.{domain_name}.operation_classes must be non-empty"
            )

        sinks: dict[str, CompiledSink] = {}
        for sink_name, sink_spec in raw_sinks.items():
            _require_nonempty_string(sink_name, f"domains.{domain_name}.sink name")
            if not isinstance(sink_spec, Mapping):
                raise ManifestValidationError(f"sink {sink_name} must be an object")
            _require_exact_keys(sink_spec, _SINK_KEYS, f"sink {sink_name}")
            broker = _require_nonempty_string(
                sink_spec["broker"], f"sink {sink_name}.broker"
            )
            direct_symbols = _require_unique_strings(
                sink_spec["direct_symbols"],
                f"sink {sink_name}.direct_symbols",
                nonempty=False,
            )
            sinks[sink_name] = CompiledSink(
                broker=broker,
                direct_symbols=tuple(sorted(direct_symbols)),
            )

        operations: dict[str, CompiledOperation] = {}
        for operation_name, operation_spec in raw_operations.items():
            _require_nonempty_string(
                operation_name, f"domains.{domain_name}.operation class name"
            )
            if not isinstance(operation_spec, Mapping):
                raise ManifestValidationError(f"operation {operation_name} must be an object")
            _require_exact_keys(
                operation_spec, _OPERATION_KEYS, f"operation {operation_name}"
            )

            sink_class = _require_nonempty_string(
                operation_spec["sink_class"],
                f"operation {operation_name}.sink_class",
            )
            if sink_class not in sinks:
                raise ManifestValidationError(
                    f"operation {operation_name} references unregistered sink {sink_class}"
                )

            required_capabilities = _require_unique_strings(
                operation_spec["required_capabilities"],
                f"operation {operation_name}.required_capabilities",
            )
            if any("*" in capability for capability in required_capabilities):
                raise ManifestValidationError(
                    f"operation {operation_name} contains an ambient wildcard capability"
                )

            actor_classes = _require_unique_strings(
                operation_spec["allowed_actor_classes"],
                f"operation {operation_name}.allowed_actor_classes",
            )
            unknown_actors = sorted(set(actor_classes) - KNOWN_ACTOR_CLASSES)
            if unknown_actors:
                raise ManifestValidationError(
                    f"operation {operation_name} has unknown actor classes {unknown_actors}"
                )

            resource_states = _require_unique_strings(
                operation_spec["allowed_resource_states"],
                f"operation {operation_name}.allowed_resource_states",
            )

            operations[operation_name] = CompiledOperation(
                sink_class=sink_class,
                required_capabilities=required_capabilities,
                allowed_actor_classes=frozenset(actor_classes),
                allowed_resource_states=frozenset(resource_states),
            )

        domains[domain_name] = CompiledDomain(
            sinks=MappingProxyType(sinks),
            operations=MappingProxyType(operations),
        )

    return CompiledManifest(
        schema_version=1,
        policy_version=policy_version,
        domains=MappingProxyType(domains),
    )


def admission_request_from_mapping(raw: Mapping[str, Any]) -> AdmissionRequest:
    """Parse a closed request shape without weakening proof semantics."""

    if not isinstance(raw, Mapping):
        raise AdmissionRequestValidationError("admission request must be an object")
    if error := _closed_key_error(raw, _REQUEST_KEYS, "admission request"):
        raise AdmissionRequestValidationError(error)

    string_fields: dict[str, str] = {}
    for field in ("domain", "operation_class", "actor_class", "resource_state"):
        value = raw[field]
        if not isinstance(value, str):
            raise AdmissionRequestValidationError(f"{field} must be a string")
        string_fields[field] = value

    capabilities_raw = raw["capabilities"]
    if not isinstance(capabilities_raw, Sequence) or isinstance(
        capabilities_raw, (str, bytes)
    ):
        raise AdmissionRequestValidationError("capabilities must be an array")

    capabilities: list[CapabilityGrant] = []
    for index, item in enumerate(capabilities_raw):
        if not isinstance(item, Mapping):
            raise AdmissionRequestValidationError(
                f"capabilities[{index}] must be an object"
            )
        if error := _closed_key_error(
            item, _CAPABILITY_GRANT_KEYS, f"capabilities[{index}]"
        ):
            raise AdmissionRequestValidationError(error)
        capability = item["capability"]
        source = item["source"]
        generation = item["generation"]
        granted = item["granted"]
        if not isinstance(capability, str):
            raise AdmissionRequestValidationError(
                f"capabilities[{index}].capability must be a string"
            )
        if not isinstance(source, str):
            raise AdmissionRequestValidationError(
                f"capabilities[{index}].source must be a string"
            )
        if not isinstance(generation, str):
            raise AdmissionRequestValidationError(
                f"capabilities[{index}].generation must be a string"
            )
        if not isinstance(granted, bool):
            raise AdmissionRequestValidationError(
                f"capabilities[{index}].granted must be a boolean"
            )
        capabilities.append(
            CapabilityGrant(
                capability=capability,
                granted=granted,
                source=source,
                generation=generation,
            )
        )

    return AdmissionRequest(
        domain=string_fields["domain"],
        operation_class=string_fields["operation_class"],
        actor_class=string_fields["actor_class"],
        resource_state=string_fields["resource_state"],
        capabilities=tuple(capabilities),
    )


def _decision(
    manifest: CompiledManifest,
    request: AdmissionRequest,
    reason_code: str,
    matched_rule: str | None,
    *,
    allowed: bool = False,
    capability_proofs: tuple[CapabilityProof, ...] = (),
) -> AuthorityDecision:
    return AuthorityDecision(
        allowed=allowed,
        operation_class=request.operation_class,
        reason_code=reason_code,
        matched_rule=matched_rule,
        capability_proofs=capability_proofs,
        policy_version=manifest.policy_version,
    )


def evaluate_authority_operation(
    manifest: CompiledManifest, request: AdmissionRequest
) -> AuthorityDecision:
    """Evaluate policy admission without performing any external effect.

    Refusal precedence is unsupported operation, forbidden actor, denied
    resource state, invalid capability proof, then missing capability.
    """

    domain = manifest.domains.get(request.domain)
    operation = domain.operations.get(request.operation_class) if domain else None
    matched_rule = (
        f"{request.domain}.{request.operation_class}" if operation is not None else None
    )

    if operation is None:
        return _decision(manifest, request, "unsupported_operation", None)

    if request.actor_class not in operation.allowed_actor_classes:
        return _decision(manifest, request, "actor_forbidden", matched_rule)

    if request.resource_state not in operation.allowed_resource_states:
        return _decision(manifest, request, "resource_state_denied", matched_rule)

    grants: dict[str, CapabilityGrant] = {}
    for grant in request.capabilities:
        if (
            not grant.capability.strip()
            or not grant.source.strip()
            or not grant.generation.strip()
            or grant.capability in grants
        ):
            return _decision(
                manifest,
                request,
                "invalid_capability_proof",
                matched_rule,
            )
        grants[grant.capability] = grant

    if any(
        capability not in grants or not grants[capability].granted
        for capability in operation.required_capabilities
    ):
        return _decision(manifest, request, "missing_capability", matched_rule)

    proofs = tuple(
        CapabilityProof(
            capability=capability,
            source=grants[capability].source,
            generation=grants[capability].generation,
        )
        for capability in operation.required_capabilities
    )
    return _decision(
        manifest,
        request,
        "allowed",
        matched_rule,
        allowed=True,
        capability_proofs=proofs,
    )


def _build_canonical_manifest_artifact() -> AuthorityManifestArtifact:
    digest = hashlib.sha256(_CANONICAL_MANIFEST_BYTES).hexdigest()
    if digest != _CANONICAL_MANIFEST_SHA256:
        raise RuntimeError("embedded authority manifest bytes do not match their digest")

    raw = json.loads(_CANONICAL_MANIFEST_BYTES)
    if not isinstance(raw, Mapping):  # pragma: no cover - guarded by literal + tests
        raise RuntimeError("embedded authority manifest must decode to an object")

    return AuthorityManifestArtifact(
        manifest_bytes=_CANONICAL_MANIFEST_BYTES,
        manifest_sha256=digest,
        manifest=compile_authority_manifest(raw),
    )


CANONICAL_AUTHORITY_MANIFEST = _build_canonical_manifest_artifact()
