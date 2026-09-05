"""Append-only semantic contract registry validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core import (
    CAMPAIGN_ID,
    GITHUB_REPOSITORY,
    HEX64,
    REGISTRY_SCHEMA_VERSION,
    _github_url,
    _is_int,
    _non_empty_string,
)

def _registry_contracts(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    contracts = registry.get("contracts")
    return contracts if isinstance(contracts, Mapping) else {}


def validate_contract_registry(document: Mapping[str, Any]) -> list[str]:
    """Validate the external, append-only semantic contract registry."""
    errors: list[str] = []
    if not isinstance(document, Mapping):
        return ["contract registry root must be an object"]
    registry_version = document.get("schema_version")
    if not _is_int(registry_version) or registry_version != REGISTRY_SCHEMA_VERSION:
        errors.append(f"registry.schema_version must be {REGISTRY_SCHEMA_VERSION}")

    contracts = document.get("contracts")
    if not isinstance(contracts, Mapping):
        errors.append("registry.contracts must be an object")
        return errors

    for campaign_id, raw_revisions in contracts.items():
        prefix = f"registry.contracts[{campaign_id!r}]"
        if not isinstance(campaign_id, str) or not CAMPAIGN_ID.fullmatch(campaign_id):
            errors.append(f"{prefix} has an invalid campaign id")
        if not isinstance(raw_revisions, list) or not raw_revisions:
            errors.append(f"{prefix} must be a non-empty revision list")
            continue

        seen_digests: set[str] = set()
        prior_digest: str | None = None
        stable_repository: str | None = None
        stable_tracker: int | None = None
        for index, revision in enumerate(raw_revisions, start=1):
            revision_prefix = f"{prefix}[{index - 1}]"
            if not isinstance(revision, Mapping):
                errors.append(f"{revision_prefix} must be an object")
                continue

            number = revision.get("revision")
            if number != index:
                errors.append(
                    f"{revision_prefix}.revision must be contiguous and equal {index}"
                )

            repository = revision.get("repository")
            if not isinstance(repository, str) or not GITHUB_REPOSITORY.fullmatch(
                repository
            ):
                errors.append(
                    f"{revision_prefix}.repository must be an owner/name repository"
                )
                repository = ""
            if stable_repository is None:
                stable_repository = repository
            elif repository != stable_repository:
                errors.append(f"{revision_prefix}.repository cannot change across revisions")

            tracker = revision.get("tracker")
            if not _is_int(tracker) or tracker <= 0:
                errors.append(f"{revision_prefix}.tracker must be a positive issue number")
            if stable_tracker is None and _is_int(tracker):
                stable_tracker = tracker
            elif _is_int(tracker) and tracker != stable_tracker:
                errors.append(f"{revision_prefix}.tracker cannot change across revisions")

            digest = revision.get("contract_sha256")
            if not isinstance(digest, str) or not HEX64.fullmatch(digest):
                errors.append(
                    f"{revision_prefix}.contract_sha256 must be lowercase 64-hex"
                )
                digest = ""
            elif digest in seen_digests:
                errors.append(f"{revision_prefix}.contract_sha256 is duplicated")
            else:
                seen_digests.add(digest)

            previous = revision.get("previous_contract_sha256")
            if index == 1:
                if previous is not None:
                    errors.append(
                        f"{revision_prefix}.previous_contract_sha256 must be null for revision 1"
                    )
            elif previous != prior_digest:
                errors.append(
                    f"{revision_prefix}.previous_contract_sha256 must match revision {index - 1}"
                )

            authority = revision.get("authority")
            if not isinstance(authority, Mapping):
                errors.append(f"{revision_prefix}.authority must be an object")
            else:
                authority_kind = authority.get("kind")
                if authority_kind not in {"issue", "pull_request"}:
                    errors.append(
                        f"{revision_prefix}.authority.kind must be issue or pull_request"
                    )
                authority_number = authority.get("number")
                if not _is_int(authority_number) or authority_number <= 0:
                    errors.append(
                        f"{revision_prefix}.authority.number must be positive"
                    )
                if repository:
                    expected = (
                        f"issues/{authority_number}"
                        if authority_kind == "issue"
                        else f"pull/{authority_number}"
                    )
                    _github_url(
                        authority.get("url"),
                        f"{revision_prefix}.authority.url",
                        repository,
                        errors,
                        expected_path_prefix=expected,
                    )

            if index > 1 and not _non_empty_string(revision.get("reason")):
                errors.append(f"{revision_prefix}.reason is required after revision 1")

            prior_digest = digest or prior_digest

    return errors


def _contract_revision(
    registry: Mapping[str, Any] | None,
    campaign_id: str,
    revision_number: int,
) -> Mapping[str, Any] | None:
    if registry is None:
        return None
    revisions = _registry_contracts(registry).get(campaign_id)
    if not isinstance(revisions, list):
        return None
    for revision in revisions:
        if isinstance(revision, Mapping) and revision.get("revision") == revision_number:
            return revision
    return None
