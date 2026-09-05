"""Terminal evidence tests for Feature Parity ledgers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_parity_ledger_test_support import (  # noqa: E402
    SHA_A,
    _errors,
    _ledger,
    _released_row,
    _registry,
    _row,

)

def test_released_row_with_structured_exact_head_evidence_passes() -> None:
    document = _ledger(_released_row())
    assert _errors(document, _registry(document)) == []


def test_released_requires_structured_terminal_evidence() -> None:
    document = _ledger(_released_row())
    document["capabilities"][0]["release_evidence"] = {
        "ci_url": "anything",
        "live_receipt": "anything",
        "review_a": "anything",
        "review_b": "anything",
    }
    errors = _errors(document)
    assert any("release_evidence.ci must be an object" in error for error in errors)
    assert any("reviews requires at least two" in error for error in errors)


def test_ci_evidence_is_repository_and_commit_bound() -> None:
    document = _ledger(_released_row())
    ci = document["capabilities"][0]["release_evidence"]["ci"]
    ci["url"] = "https://github.com/other/project/actions/runs/1"
    ci["commit_sha"] = SHA_A
    errors = _errors(document)
    assert any("must belong to example/project" in error for error in errors)
    assert any("ci.commit_sha" in error for error in errors)


def test_live_receipt_requires_canonical_path_hash_and_commit() -> None:
    document = _ledger(_released_row())
    receipt = document["capabilities"][0]["release_evidence"]["live_receipt"]
    receipt.update({"path": "../receipt.json", "sha256": "bad", "commit_sha": SHA_A})
    errors = _errors(document)
    assert any("live_receipt.path" in error for error in errors)
    assert any("live_receipt.sha256" in error for error in errors)
    assert any("live_receipt.commit_sha" in error for error in errors)


def test_reviews_must_be_distinct_and_independent() -> None:
    document = _ledger(_released_row())
    reviews = document["capabilities"][0]["release_evidence"]["reviews"]
    reviews[0]["reviewer"] = "contributor"
    reviews[1]["reviewer"] = "contributor"
    errors = _errors(document)
    assert any("independent of the PR author" in error for error in errors)
    assert any("reviews must be independent" in error for error in errors)


def test_review_url_must_identify_submitted_review_on_same_pr() -> None:
    document = _ledger(_released_row())
    review = document["capabilities"][0]["release_evidence"]["reviews"][0]
    review["url"] = "https://github.com/example/project/pull/999#issuecomment-1"
    errors = _errors(document)
    assert any("must point to pull/1001" in error for error in errors)
    assert any("submitted pull request review" in error for error in errors)


def test_merged_publication_sha_must_match_merged_record() -> None:
    document = _ledger(_released_row())
    document["capabilities"][0]["publications"][0]["merge_commit_sha"] = SHA_A
    assert any("merge_commit_sha" in error for error in _errors(document))


def test_release_evidence_is_rejected_before_release() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["release_evidence"] = {}
    assert any("only valid for released" in error for error in _errors(document))
