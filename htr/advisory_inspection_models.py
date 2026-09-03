"""Task 29 — advisory artifact/link inspection result models (R5-10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from htr.advisory_inspection_constants import PROTOCOL_VERSION, SCHEMA_VERSION

RecordKind = Literal[
    "run_execution_request_record",
    "run_post_verification_execution_request_record",
]

DigestPattern = r"^sha256:[0-9a-f]{64}$"


@dataclass(frozen=True)
class ArtifactReferenceSelector:
    run_id: str
    task_id: str
    attempt_id: str
    manifest_raw_digest: str
    entry_index: int


@dataclass(frozen=True)
class LinkReferenceSelector:
    run_id: str
    record_kind: RecordKind
    record_raw_digest: str
    item_index: int


@dataclass
class AuthorityFlags:
    """Fourteen advisory authority booleans — all default false (R5-10)."""

    may_execute: bool = False
    may_retry: bool = False
    may_repair: bool = False
    may_complete: bool = False
    may_finalize: bool = False
    may_mutate_lifecycle: bool = False
    may_approve: bool = False
    may_claim: bool = False
    may_copy_artifact: bool = False
    may_adopt_artifact: bool = False
    may_create_task23_marker: bool = False
    may_mutate_task24_28_evidence: bool = False
    may_fetch_remote: bool = False
    may_grant_successor_mutation: bool = False


@dataclass
class DerivedAlignment:
    role: str
    applicable: bool
    match_status: str
    derived_index: int | None
    candidate_derived_indexes: list[int] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


@dataclass
class UnreferencedObservation:
    name: str
    hashed: bool = False
    findings: list[str] = field(default_factory=list)


@dataclass
class RecordLoadStatus:
    filename: str
    status: str


@dataclass
class ArtifactInspectionResult(AuthorityFlags):
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    publication: str = "none"
    atime_may_have_changed: bool = True

    run_id: str = ""
    task_id: str = ""
    attempt_id: str = ""
    entry_index: int | None = None
    manifest_raw_digest: str | None = None
    manifest_semantic_digest: str | None = None
    path_identity_digest: str | None = None
    declared_path: str | None = None
    validated_components: tuple[str, ...] | None = None

    authority_status: str = "advisory_only"
    aggregate_completeness: str = "aggregate_not_applicable"
    reference_status: str = "reference_not_applicable"
    manifest_status: str = "manifest_absent"
    path_status: str = "path_not_applicable"
    filesystem_status: str = "filesystem_not_attempted"
    file_type_status: str = "file_type_not_inspected"
    hardlink_status: str = "hardlink_not_inspected"
    identity_status: str = "identity_not_applicable"
    size_status: str = "size_not_inspected"
    digest_status: str = "digest_not_inspected"
    media_type_status: str = "media_type_not_inspected"
    semantic_status: str = "artifact_semantics_not_inspected"
    stability_status: str = "stability_not_applicable"
    budget_status: str = "budget_within_limits"
    run_context_status: str = "run_not_finalized"

    extras_unprocessed_count: int = 0
    findings: list[str] = field(default_factory=list)
    decoded_manifest: dict[str, Any] | None = None
    entry: dict[str, Any] | None = None


@dataclass
class ArtifactAggregateResult(AuthorityFlags):
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    publication: str = "none"
    atime_may_have_changed: bool = True

    run_id: str = ""
    task_id: str | None = None
    attempt_id: str | None = None

    authority_status: str = "advisory_only"
    aggregate_completeness: str = "aggregate_empty"
    budget_status: str = "budget_within_limits"
    run_context_status: str = "run_not_finalized"

    items: list[ArtifactInspectionResult] = field(default_factory=list)
    unreferenced: list[UnreferencedObservation] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


@dataclass
class LinkInspectionResult(AuthorityFlags):
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    publication: str = "none"
    atime_may_have_changed: bool = True

    run_id: str = ""
    record_kind: str | None = None
    item_index: int | None = None
    record_raw_digest: str | None = None
    record_semantic_digest: str | None = None

    authority_status: str = "advisory_only"
    aggregate_completeness: str = "aggregate_not_applicable"
    link_record_status: str = "link_record_not_attempted"
    link_item_status: str = "link_item_not_applicable"
    link_scheme_status: str = "link_scheme_not_applicable"
    link_host_status: str = "link_host_not_applicable"
    link_port_status: str = "link_port_not_applicable"
    link_structure_status: str = "link_structure_not_applicable"
    link_fetch_status: str = "link_fetch_not_applicable"
    budget_status: str = "budget_within_limits"
    run_context_status: str = "run_not_finalized"

    derived_alignments: list[DerivedAlignment] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    item: dict[str, Any] | None = None


@dataclass
class LinkAggregateResult(AuthorityFlags):
    protocol_version: str = PROTOCOL_VERSION
    schema_version: str = SCHEMA_VERSION
    publication: str = "none"
    atime_may_have_changed: bool = True

    run_id: str = ""

    authority_status: str = "advisory_only"
    aggregate_completeness: str = "aggregate_empty"
    budget_status: str = "budget_within_limits"
    run_context_status: str = "run_not_finalized"

    items: list[LinkInspectionResult] = field(default_factory=list)
    records_loaded: list[RecordLoadStatus] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


def sort_findings(findings: list[str]) -> list[str]:
    """Unique supplemental findings in Unicode code-point lexicographic order."""
    return sorted(set(findings))
