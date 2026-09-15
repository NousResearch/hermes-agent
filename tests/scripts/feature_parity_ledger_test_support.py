"""Behavioral tests for the Feature Parity campaign ledger contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "ci" / "validate_feature_parity_ledger.py"
_SPEC = importlib.util.spec_from_file_location("validate_feature_parity_ledger", VALIDATOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load Feature Parity ledger validator")
_validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_validator)

LedgerValidationError = _validator.LedgerValidationError
canonical_contract_digest = _validator.canonical_contract_digest
load_json_document = _validator.load_json_document
validate_contract_registry = _validator.validate_contract_registry
validate_ledger = _validator.validate_ledger

SHA_A = "a" * 40
SHA_B = "b" * 40
DIGEST_A = "a" * 64


def _publication(
    pr: int,
    *,
    state: str = "open",
    author: str = "contributor",
    head_sha: str | None = SHA_A,
    merge_commit_sha: str | None = None,
) -> dict:
    publication = {
        "kind": "pull_request",
        "number": pr,
        "role": "authoritative",
        "state": state,
        "author": author,
        "url": f"https://github.com/example/project/pull/{pr}",
    }
    if head_sha is not None:
        publication["head_sha"] = head_sha
    if merge_commit_sha is not None:
        publication["merge_commit_sha"] = merge_commit_sha
    return publication


def _row(
    capability_id: str,
    *,
    name: str | None = None,
    product_state: str = "accepted",
    delivery_state: str = "candidate_open",
    pr: int | None = None,
) -> dict:
    pr_number = pr if pr is not None else 1000 + int("".join(filter(str.isdigit, capability_id)) or 1)
    row = {
        "id": capability_id,
        "name": name or f"Capability {capability_id}",
        "source_anchor": f"Canonical / {capability_id}",
        "product_state": product_state,
        "delivery_state": delivery_state,
        "implementation_paths": [f"plugins/example/{capability_id.lower()}.py"],
        "test_paths": [f"tests/example/test_{capability_id.lower()}.py"],
        "consumers": [f"plugins/example/runtime.py:{capability_id}"],
        "publications": [_publication(pr_number)],
        "artifact_evidence": [],
    }
    if product_state in {"pair_gap", "conditional", "deferred", "rejected"}:
        row["decision"] = "Explicit product decision."
    if delivery_state == "candidate_blocked":
        row["blocker"] = "Explicit dependency gate."
    elif delivery_state == "candidate_unwired":
        row["consumers"] = []
        row["wiring_gap"] = "No production caller owns the effect path."
    elif delivery_state == "gap":
        row["implementation_paths"] = []
        row["test_paths"] = []
        row["consumers"] = []
        row["publications"] = []
        row["gap_reason"] = "No active implementation authority."
    elif delivery_state == "superseded":
        row["implementation_paths"] = []
        row["test_paths"] = []
        row["consumers"] = []
        row["publications"] = []
        row["superseded_by"] = "M2"
    if product_state in {"deferred", "rejected"}:
        row["delivery_state"] = "gap"
        row["implementation_paths"] = []
        row["test_paths"] = []
        row["consumers"] = []
        row["publications"] = []
        row["gap_reason"] = "Product decision excludes implementation."
    return row


def _released_row(capability_id: str = "M1", pr: int = 1001) -> dict:
    row = _row(capability_id, pr=pr)
    row["delivery_state"] = "released"
    row["publications"] = [
        _publication(
            pr,
            state="merged",
            head_sha=None,
            merge_commit_sha=SHA_B,
        )
    ]
    row["merged"] = {
        "repository": "example/project",
        "commit_sha": SHA_B,
    }
    row["release_evidence"] = {
        "ci": {
            "url": "https://github.com/example/project/actions/runs/123",
            "commit_sha": SHA_B,
        },
        "live_receipt": {
            "path": "receipts/example.json",
            "sha256": DIGEST_A,
            "commit_sha": SHA_B,
        },
        "reviews": [
            {
                "reviewer": "reviewer-one",
                "url": f"https://github.com/example/project/pull/{pr}#pullrequestreview-1",
                "commit_sha": SHA_B,
            },
            {
                "reviewer": "reviewer-two",
                "url": f"https://github.com/example/project/pull/{pr}#pullrequestreview-2",
                "commit_sha": SHA_B,
            },
        ],
    }
    return row


def _ledger(*rows: dict, revision: int = 1) -> dict:
    capabilities = list(rows)
    return {
        "schema_version": 1,
        "campaign": {
            "id": "example-feature-parity",
            "repository": "example/project",
            "tracker": 123,
            "contract_revision": revision,
            "expected_capability_ids": [row["id"] for row in capabilities],
            "forbidden_growth_paths": ["plugins/example/adapter.py"],
            "contract_sha256": canonical_contract_digest(capabilities),
        },
        "snapshot": {
            "upstream_sha": SHA_A,
            "captured_at": "2026-08-19T21:45:14Z",
        },
        "capabilities": capabilities,
    }


def _registry(document: dict, *, revision: int = 1) -> dict:
    digest = document["campaign"]["contract_sha256"]
    entry = {
        "revision": revision,
        "repository": "example/project",
        "tracker": 123,
        "contract_sha256": digest,
        "previous_contract_sha256": None,
        "authority": {
            "kind": "issue",
            "number": 123,
            "url": "https://github.com/example/project/issues/123",
        },
    }
    return {
        "schema_version": 1,
        "contracts": {"example-feature-parity": [entry]},
    }


def _errors(document: dict, registry: dict | None = None) -> list[str]:
    return validate_ledger(document, contract_registry=registry)
