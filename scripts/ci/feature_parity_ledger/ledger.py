"""Per-campaign ledger validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .core import (
    CAMPAIGN_ID,
    CAPABILITY_ID,
    CONSUMER_REQUIRED_STATES,
    DECISION_GATED_PRODUCT_STATES,
    DELIVERY_STATES,
    FORBIDDEN_DELIVERY_STATES,
    GITHUB_REPOSITORY,
    HEX40,
    HEX64,
    LedgerValidationError,
    MAIN_REQUIRED_STATES,
    NON_PROMOTABLE_PRODUCT_STATES,
    PRODUCT_STATES,
    SCHEMA_VERSION,
    _append_required_string,
    _canonical_repo_path,
    _consumer_path,
    _is_int,
    _non_empty_string,
    _parse_utc_timestamp,
    _required_list,
    _required_string_list,
    canonical_contract_digest,
)
from .evidence import _validate_publications, _validate_release_evidence
from .registry import _contract_revision, validate_contract_registry

def validate_ledger(
    document: Mapping[str, Any],
    *,
    contract_registry: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return every structural and semantic validation error for one ledger."""
    errors: list[str] = []
    if not isinstance(document, Mapping):
        return ["ledger root must be an object"]
    schema_version = document.get("schema_version")
    if not _is_int(schema_version) or schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    campaign = document.get("campaign")
    if not isinstance(campaign, Mapping):
        errors.append("campaign must be an object")
        campaign = {}

    campaign_id = _append_required_string(
        campaign, "id", "campaign.id", errors
    )
    if campaign_id and not CAMPAIGN_ID.fullmatch(campaign_id):
        errors.append("campaign.id must be lowercase kebab-case")

    repository = _append_required_string(
        campaign, "repository", "campaign.repository", errors
    )
    if repository and not GITHUB_REPOSITORY.fullmatch(repository):
        errors.append("campaign.repository must be an owner/name repository")

    tracker = campaign.get("tracker")
    if not _is_int(tracker) or tracker <= 0:
        errors.append("campaign.tracker must be a positive GitHub issue number")

    revision_number = campaign.get("contract_revision")
    if not _is_int(revision_number) or revision_number <= 0:
        errors.append("campaign.contract_revision must be a positive integer")
        revision_number = 0

    expected_ids = _required_string_list(
        campaign,
        "expected_capability_ids",
        "campaign.expected_capability_ids",
        errors,
    )
    if not expected_ids:
        errors.append("campaign.expected_capability_ids must not be empty")
    invalid_expected = [item for item in expected_ids if not CAPABILITY_ID.fullmatch(item)]
    if invalid_expected:
        errors.append(
            f"campaign.expected_capability_ids contains invalid ids: {invalid_expected}"
        )

    forbidden_paths = _required_string_list(
        campaign,
        "forbidden_growth_paths",
        "campaign.forbidden_growth_paths",
        errors,
    )
    canonical_forbidden: list[str] = []
    for index, path in enumerate(forbidden_paths):
        canonical = _canonical_repo_path(
            path,
            f"campaign.forbidden_growth_paths[{index}]",
            errors,
        )
        if canonical:
            canonical_forbidden.append(canonical)

    declared_digest = campaign.get("contract_sha256")
    if not isinstance(declared_digest, str) or not HEX64.fullmatch(declared_digest):
        errors.append("campaign.contract_sha256 must be lowercase 64-hex")
        declared_digest = ""

    snapshot = document.get("snapshot")
    if not isinstance(snapshot, Mapping):
        errors.append("snapshot must be an object")
        snapshot = {}
    upstream_sha = snapshot.get("upstream_sha")
    if not isinstance(upstream_sha, str) or not HEX40.fullmatch(upstream_sha):
        errors.append("snapshot.upstream_sha must be lowercase 40-hex")
    _parse_utc_timestamp(snapshot.get("captured_at"), "snapshot.captured_at", errors)

    capabilities = document.get("capabilities")
    if not isinstance(capabilities, list):
        errors.append("capabilities must be a list")
        capabilities = []
    if not capabilities:
        errors.append("capabilities must not be empty")

    ids: list[str] = []
    authoritative_prs: list[tuple[int, str]] = []
    digest_rows: list[Mapping[str, Any]] = []

    for index, raw_row in enumerate(capabilities):
        prefix = f"capabilities[{index}]"
        if not isinstance(raw_row, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        row = raw_row
        digest_rows.append(row)

        capability_id = row.get("id")
        if not isinstance(capability_id, str) or not CAPABILITY_ID.fullmatch(capability_id):
            errors.append(f"{prefix}.id must be a canonical capability id")
            capability_id = f"<row-{index}>"
        ids.append(capability_id)

        _append_required_string(row, "name", f"{capability_id}.name", errors)
        _append_required_string(
            row,
            "source_anchor",
            f"{capability_id}.source_anchor",
            errors,
        )

        product_state = row.get("product_state")
        if product_state not in PRODUCT_STATES:
            errors.append(
                f"{capability_id}.product_state must be one of {sorted(PRODUCT_STATES)}"
            )

        delivery_state = row.get("delivery_state")
        if delivery_state in FORBIDDEN_DELIVERY_STATES:
            errors.append(
                f"{capability_id}.delivery_state={delivery_state!r} confuses artifact evidence with delivery"
            )
        elif delivery_state not in DELIVERY_STATES:
            errors.append(
                f"{capability_id}.delivery_state must be one of {sorted(DELIVERY_STATES)}"
            )

        if product_state in DECISION_GATED_PRODUCT_STATES:
            _append_required_string(
                row,
                "decision",
                f"{capability_id}.decision",
                errors,
            )
        if (
            product_state in NON_PROMOTABLE_PRODUCT_STATES
            and delivery_state in {"candidate_open", "on_main_unverified", "released"}
        ):
            errors.append(
                f"{capability_id} product_state={product_state!r} cannot advance to {delivery_state!r}"
            )

        implementation_paths = _required_string_list(
            row,
            "implementation_paths",
            f"{capability_id}.implementation_paths",
            errors,
        )
        test_paths = _required_string_list(
            row,
            "test_paths",
            f"{capability_id}.test_paths",
            errors,
        )
        consumers = _required_string_list(
            row,
            "consumers",
            f"{capability_id}.consumers",
            errors,
        )
        _required_list(
            row,
            "artifact_evidence",
            f"{capability_id}.artifact_evidence",
            errors,
        )

        canonical_implementation: list[str] = []
        for path_index, path in enumerate(implementation_paths):
            canonical = _canonical_repo_path(
                path,
                f"{capability_id}.implementation_paths[{path_index}]",
                errors,
            )
            if canonical:
                canonical_implementation.append(canonical)
        for path_index, path in enumerate(test_paths):
            _canonical_repo_path(
                path,
                f"{capability_id}.test_paths[{path_index}]",
                errors,
            )
        for consumer_index, consumer in enumerate(consumers):
            _consumer_path(
                consumer,
                f"{capability_id}.consumers[{consumer_index}]",
                errors,
            )

        for path in canonical_implementation:
            for forbidden in canonical_forbidden:
                if path == forbidden or path.startswith(f"{forbidden}/"):
                    errors.append(
                        f"{capability_id}.implementation_paths grows forbidden surface {forbidden!r}"
                    )

        if product_state in {"rejected", "deferred"}:
            if implementation_paths or test_paths or consumers:
                errors.append(
                    f"{capability_id} product_state={product_state!r} cannot declare production, test, or consumer paths"
                )
            if delivery_state not in {"gap", "superseded"}:
                errors.append(
                    f"{capability_id} product_state={product_state!r} requires gap or superseded delivery"
                )

        if delivery_state == "gap":
            if not _non_empty_string(row.get("gap_reason")) and not _non_empty_string(
                row.get("decision")
            ):
                errors.append(f"{capability_id}.gap_reason is required for a gap")
        elif delivery_state == "candidate_blocked":
            _append_required_string(
                row,
                "blocker",
                f"{capability_id}.blocker",
                errors,
            )
        elif delivery_state == "candidate_unwired":
            _append_required_string(
                row,
                "wiring_gap",
                f"{capability_id}.wiring_gap",
                errors,
            )
            if consumers:
                errors.append(f"{capability_id} candidate_unwired must have no consumers")
            if not implementation_paths or not test_paths:
                errors.append(
                    f"{capability_id} candidate_unwired requires implementation_paths and test_paths"
                )
        elif delivery_state in CONSUMER_REQUIRED_STATES:
            if not implementation_paths:
                errors.append(
                    f"{capability_id} delivery_state={delivery_state!r} requires implementation_paths"
                )
            if not test_paths:
                errors.append(
                    f"{capability_id} delivery_state={delivery_state!r} requires test_paths"
                )
            if not consumers:
                errors.append(
                    f"{capability_id} delivery_state={delivery_state!r} requires runtime consumers"
                )
        elif delivery_state == "superseded":
            _append_required_string(
                row,
                "superseded_by",
                f"{capability_id}.superseded_by",
                errors,
            )

        merged_sha = ""
        merged = row.get("merged")
        if delivery_state in MAIN_REQUIRED_STATES:
            if not isinstance(merged, Mapping):
                errors.append(f"{capability_id}.merged is required on main")
            else:
                commit_sha = merged.get("commit_sha")
                if not isinstance(commit_sha, str) or not HEX40.fullmatch(commit_sha):
                    errors.append(
                        f"{capability_id}.merged.commit_sha must be lowercase 40-hex"
                    )
                else:
                    merged_sha = commit_sha
                if merged.get("repository") != repository:
                    errors.append(
                        f"{capability_id}.merged.repository must equal campaign.repository"
                    )
        elif merged is not None:
            errors.append(f"{capability_id}.merged is only valid on main delivery states")

        row_prs, authoritative = _validate_publications(
            row,
            capability_id,
            delivery_state,
            repository,
            merged_sha,
            errors,
        )
        authoritative_prs.extend(row_prs)

        if delivery_state == "released":
            _validate_release_evidence(
                row,
                capability_id,
                repository,
                merged_sha,
                authoritative,
                errors,
            )
        elif "release_evidence" in row:
            errors.append(
                f"{capability_id}.release_evidence is only valid for released delivery"
            )

    duplicate_ids = sorted(name for name, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate capability ids: {duplicate_ids}")
    if ids != expected_ids:
        missing = [item for item in expected_ids if item not in ids]
        unexpected = [item for item in ids if item not in expected_ids]
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if not details:
            details.append("order differs")
        errors.append(
            "capability ids do not exactly match campaign.expected_capability_ids: "
            + ", ".join(details)
        )

    duplicate_prs = sorted(
        number
        for number, count in Counter(number for number, _ in authoritative_prs).items()
        if count > 1
    )
    for number in duplicate_prs:
        owners = sorted(
            capability_id
            for pr_number, capability_id in authoritative_prs
            if pr_number == number
        )
        errors.append(
            f"authoritative pull request #{number} is claimed by multiple capabilities: {owners}"
        )

    if digest_rows:
        try:
            calculated_digest = canonical_contract_digest(digest_rows)
        except LedgerValidationError as exc:
            errors.extend(exc.errors)
        else:
            if declared_digest != calculated_digest:
                errors.append(
                    "campaign.contract_sha256 does not match canonical "
                    f"(id, name, product_state, source_anchor) payload: expected {calculated_digest}"
                )

    if contract_registry is not None:
        registry_errors = validate_contract_registry(contract_registry)
        errors.extend(f"contract registry: {error}" for error in registry_errors)
        if campaign_id and revision_number:
            revision = _contract_revision(
                contract_registry,
                campaign_id,
                revision_number,
            )
            if revision is None:
                errors.append(
                    f"campaign contract {campaign_id!r} revision {revision_number} is not registered"
                )
            else:
                if revision.get("contract_sha256") != declared_digest:
                    errors.append(
                        "campaign.contract_sha256 does not match the external registry revision"
                    )
                if revision.get("tracker") != tracker:
                    errors.append("campaign.tracker does not match the external registry")
                if revision.get("repository") != repository:
                    errors.append(
                        "campaign.repository does not match the external registry"
                    )

    return errors
