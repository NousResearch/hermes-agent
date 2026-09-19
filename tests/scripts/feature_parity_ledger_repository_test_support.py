"""Repository-wide tests for Feature Parity ledger ownership and receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "ci" / "validate_feature_parity_ledger.py"
_SPEC = importlib.util.spec_from_file_location("validate_feature_parity_ledger_repo", VALIDATOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load Feature Parity ledger validator")
_validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_validator)

canonical_contract_digest = _validator.canonical_contract_digest
discover_ledgers = _validator.discover_ledgers
validate_repository = _validator.validate_repository

SHA_A = "a" * 40
SHA_B = "b" * 40


def _write_registry(root: Path, *entries: tuple[str, str, int, int]) -> None:
    contracts: dict[str, list[dict]] = {}
    for campaign_id, digest, tracker, authority in entries:
        contracts[campaign_id] = [
            {
                "revision": 1,
                "repository": "example/project",
                "tracker": tracker,
                "contract_sha256": digest,
                "previous_contract_sha256": None,
                "authority": {
                    "kind": "issue",
                    "number": authority,
                    "url": f"https://github.com/example/project/issues/{authority}",
                },
            }
        ]
    path = root / "docs" / "architecture" / "feature-parity" / "contracts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "contracts": contracts}), encoding="utf-8")


def _ledger(
    campaign_id: str,
    tracker: int,
    capability_id: str,
    pr: int,
    *,
    released: bool = False,
    receipt_path: str = "receipts/live.json",
    receipt_digest: str = "a" * 64,
) -> dict:
    row = {
        "id": capability_id,
        "name": f"Capability {capability_id}",
        "source_anchor": f"Canonical / {capability_id}",
        "product_state": "accepted",
        "delivery_state": "candidate_open",
        "implementation_paths": [f"plugins/example/{capability_id.lower()}.py"],
        "test_paths": [f"tests/example/test_{capability_id.lower()}.py"],
        "consumers": [f"plugins/example/runtime.py:{capability_id}"],
        "publications": [
            {
                "kind": "pull_request",
                "number": pr,
                "role": "authoritative",
                "state": "open",
                "author": "contributor",
                "url": f"https://github.com/example/project/pull/{pr}",
                "head_sha": SHA_A,
            }
        ],
        "artifact_evidence": [],
    }
    if released:
        row["delivery_state"] = "released"
        row["publications"][0].update(
            {
                "state": "merged",
                "merge_commit_sha": SHA_B,
            }
        )
        row["publications"][0].pop("head_sha")
        row["merged"] = {
            "repository": "example/project",
            "commit_sha": SHA_B,
        }
        row["release_evidence"] = {
            "ci": {
                "url": "https://github.com/example/project/actions/runs/1",
                "commit_sha": SHA_B,
            },
            "live_receipt": {
                "path": receipt_path,
                "sha256": receipt_digest,
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
    digest = canonical_contract_digest([row])
    return {
        "schema_version": 1,
        "campaign": {
            "id": campaign_id,
            "repository": "example/project",
            "tracker": tracker,
            "contract_revision": 1,
            "expected_capability_ids": [capability_id],
            "forbidden_growth_paths": ["plugins/example/adapter.py"],
            "contract_sha256": digest,
        },
        "snapshot": {
            "upstream_sha": SHA_A,
            "captured_at": "2026-08-19T21:45:14Z",
        },
        "capabilities": [row],
    }


def _write_ledger(root: Path, name: str, document: dict) -> Path:
    path = root / "docs" / "architecture" / "feature-parity" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
