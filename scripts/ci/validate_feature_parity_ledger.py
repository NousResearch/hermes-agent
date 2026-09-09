#!/usr/bin/env python3
"""Validate Feature Parity & Alignment campaign ledgers.

The ledger is an executable publication contract. It separates:
- canonical capability identity and product disposition;
- implementation/publication state;
- runtime consumer wiring;
- release evidence.

A packet, patch, branch, or green focused suite is evidence, not delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
HEX40 = re.compile(r"^[0-9a-f]{40}$")
CAPABILITY_ID = re.compile(r"^[A-Z][A-Z0-9]*[0-9]+$")

PRODUCT_STATES = {
    "accepted",
    "existing",
    "pair_gap",
    "conditional",
    "deferred",
    "rejected",
}
DELIVERY_STATES = {
    "gap",
    "candidate_blocked",
    "candidate_unwired",
    "candidate_open",
    "on_main_unverified",
    "released",
    "superseded",
}
FORBIDDEN_DELIVERY_STATES = {
    "implemented_in_packet",
    "implemented_locally",
    "package_green",
    "patch_ready",
    "branch_exists",
}
PUBLICATION_ROLES = {"authoritative", "dependency", "superseded"}
PUBLICATION_KINDS = {"issue", "pull_request", "commit", "release"}

ACTIVE_PUBLICATION_STATES = {
    "candidate_blocked",
    "candidate_unwired",
    "candidate_open",
    "on_main_unverified",
    "released",
}
CONSUMER_REQUIRED_STATES = {
    "candidate_open",
    "on_main_unverified",
    "released",
}
MAIN_REQUIRED_STATES = {"on_main_unverified", "released"}
TERMINAL_REQUIRED_STATES = {"released"}


class LedgerValidationError(ValueError):
    """Raised when a campaign ledger violates one or more invariants."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


def _strings(value: Any, field: str, errors: list[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{field} must be a list of strings")
        return []
    return value


def canonical_contract_payload(capabilities: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return the immutable semantic identity used for the contract digest."""
    rows = []
    for row in capabilities:
        rows.append(
            {
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "product_state": str(row.get("product_state", "")),
            }
        )
    return rows


def canonical_contract_digest(capabilities: Iterable[Mapping[str, Any]]) -> str:
    payload = canonical_contract_payload(capabilities)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_ledger(document: Mapping[str, Any]) -> list[str]:
    """Return every validation error; an empty list means the ledger is valid."""
    errors: list[str] = []

    if not isinstance(document, Mapping):
        return ["ledger root must be an object"]

    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    campaign = document.get("campaign")
    if not isinstance(campaign, Mapping):
        errors.append("campaign must be an object")
        campaign = {}

    campaign_id = campaign.get("id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        errors.append("campaign.id must be a non-empty string")

    tracker = campaign.get("tracker")
    if not isinstance(tracker, int) or isinstance(tracker, bool) or tracker <= 0:
        errors.append("campaign.tracker must be a positive GitHub issue number")

    expected_ids = _strings(
        campaign.get("expected_capability_ids"),
        "campaign.expected_capability_ids",
        errors,
    )
    if len(expected_ids) != len(set(expected_ids)):
        errors.append("campaign.expected_capability_ids contains duplicates")

    forbidden_paths = _strings(
        campaign.get("forbidden_growth_paths", []),
        "campaign.forbidden_growth_paths",
        errors,
    )
    forbidden_paths = [path.rstrip("/") for path in forbidden_paths if path]

    snapshot = document.get("snapshot")
    if not isinstance(snapshot, Mapping):
        errors.append("snapshot must be an object")
        snapshot = {}
    upstream_sha = snapshot.get("upstream_sha")
    if not isinstance(upstream_sha, str) or not HEX40.fullmatch(upstream_sha):
        errors.append("snapshot.upstream_sha must be a lowercase 40-hex commit SHA")
    captured_at = snapshot.get("captured_at")
    if not isinstance(captured_at, str) or not captured_at.endswith("Z"):
        errors.append("snapshot.captured_at must be an explicit UTC timestamp ending in Z")

    capabilities = document.get("capabilities")
    if not isinstance(capabilities, list):
        errors.append("capabilities must be a list")
        capabilities = []

    ids: list[str] = []
    authoritative_prs: list[tuple[int, str]] = []
    for index, raw_row in enumerate(capabilities):
        prefix = f"capabilities[{index}]"
        if not isinstance(raw_row, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        row = raw_row
        capability_id = row.get("id")
        if not isinstance(capability_id, str) or not CAPABILITY_ID.fullmatch(capability_id):
            errors.append(f"{prefix}.id must be a canonical capability id")
            capability_id = f"<row-{index}>"
        ids.append(capability_id)

        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{capability_id}.name must be a non-empty string")

        source_anchor = row.get("source_anchor")
        if not isinstance(source_anchor, str) or not source_anchor.strip():
            errors.append(
                f"{capability_id}.source_anchor must identify the canonical spec section"
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

        decision = row.get("decision")
        if product_state in {"pair_gap", "conditional", "deferred", "rejected"}:
            if not isinstance(decision, str) or not decision.strip():
                errors.append(
                    f"{capability_id}.decision is required for product_state={product_state}"
                )

        implementation_paths = _strings(
            row.get("implementation_paths", []),
            f"{capability_id}.implementation_paths",
            errors,
        )
        test_paths = _strings(
            row.get("test_paths", []),
            f"{capability_id}.test_paths",
            errors,
        )
        consumers = _strings(
            row.get("consumers", []),
            f"{capability_id}.consumers",
            errors,
        )
        evidence = row.get("artifact_evidence", [])
        if not isinstance(evidence, list):
            errors.append(f"{capability_id}.artifact_evidence must be a list")

        if product_state == "rejected":
            if implementation_paths:
                errors.append(
                    f"{capability_id} is rejected but declares implementation_paths"
                )
            if delivery_state not in {"gap", "superseded"}:
                errors.append(
                    f"{capability_id} is rejected but delivery_state={delivery_state!r}"
                )

        for path in implementation_paths:
            normalized = path.strip("/")
            for forbidden in forbidden_paths:
                if normalized == forbidden or normalized.startswith(f"{forbidden}/"):
                    errors.append(
                        f"{capability_id}.implementation_paths grows forbidden surface {forbidden!r}"
                    )

        publications = row.get("publications", [])
        if not isinstance(publications, list):
            errors.append(f"{capability_id}.publications must be a list")
            publications = []
        authoritative = 0
        for pub_index, publication in enumerate(publications):
            pub_prefix = f"{capability_id}.publications[{pub_index}]"
            if not isinstance(publication, Mapping):
                errors.append(f"{pub_prefix} must be an object")
                continue
            role = publication.get("role")
            kind = publication.get("kind")
            if role not in PUBLICATION_ROLES:
                errors.append(
                    f"{pub_prefix}.role must be one of {sorted(PUBLICATION_ROLES)}"
                )
            if kind not in PUBLICATION_KINDS:
                errors.append(
                    f"{pub_prefix}.kind must be one of {sorted(PUBLICATION_KINDS)}"
                )
            number = publication.get("number")
            if kind in {"issue", "pull_request"}:
                if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                    errors.append(f"{pub_prefix}.number must be a positive integer")
            if role == "authoritative":
                authoritative += 1
                if kind == "pull_request" and isinstance(number, int):
                    authoritative_prs.append((number, capability_id))

        if delivery_state in ACTIVE_PUBLICATION_STATES and authoritative != 1:
            errors.append(
                f"{capability_id} delivery_state={delivery_state!r} requires exactly one authoritative publication"
            )
        if delivery_state in {"gap", "superseded"} and authoritative > 1:
            errors.append(f"{capability_id} has multiple authoritative publications")

        if delivery_state in CONSUMER_REQUIRED_STATES:
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

        merged = row.get("merged")
        if delivery_state in MAIN_REQUIRED_STATES:
            if not isinstance(merged, Mapping):
                errors.append(f"{capability_id}.merged is required on main")
            else:
                sha = merged.get("commit_sha")
                if not isinstance(sha, str) or not HEX40.fullmatch(sha):
                    errors.append(
                        f"{capability_id}.merged.commit_sha must be a 40-hex SHA"
                    )

        if delivery_state in TERMINAL_REQUIRED_STATES:
            release = row.get("release_evidence")
            if not isinstance(release, Mapping):
                errors.append(
                    f"{capability_id}.release_evidence is required for released"
                )
            else:
                for key in ("ci_url", "live_receipt", "review_a", "review_b"):
                    value = release.get(key)
                    if not isinstance(value, str) or not value.strip():
                        errors.append(
                            f"{capability_id}.release_evidence.{key} is required for released"
                        )

    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate capability ids: {', '.join(duplicate_ids)}")

    if expected_ids and ids != expected_ids:
        missing = [item for item in expected_ids if item not in ids]
        unexpected = [item for item in ids if item not in expected_ids]
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if unexpected:
            detail.append(f"unexpected={unexpected}")
        if not detail:
            detail.append("order differs")
        errors.append(
            "capability ids do not exactly match campaign.expected_capability_ids: "
            + ", ".join(detail)
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

    declared_digest = campaign.get("contract_sha256")
    calculated_digest = canonical_contract_digest(capabilities)
    if not isinstance(declared_digest, str) or declared_digest != calculated_digest:
        errors.append(
            "campaign.contract_sha256 does not match canonical "
            f"(id, name, product_state) payload: expected {calculated_digest}"
        )

    return errors


def load_ledger(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LedgerValidationError([f"cannot read {path}: {exc}"]) from exc
    except json.JSONDecodeError as exc:
        raise LedgerValidationError([f"{path}: invalid JSON: {exc}"]) from exc
    if not isinstance(document, Mapping):
        raise LedgerValidationError(["ledger root must be an object"])
    return document


def validate_path(path: Path) -> list[str]:
    try:
        document = load_ledger(path)
    except LedgerValidationError as exc:
        return list(exc.errors)
    return validate_ledger(document)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledgers", nargs="+", type=Path)
    args = parser.parse_args(argv)

    failed = False
    for path in args.ledgers:
        errors = validate_path(path)
        if errors:
            failed = True
            print(f"{path}: INVALID", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"{path}: VALID")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
