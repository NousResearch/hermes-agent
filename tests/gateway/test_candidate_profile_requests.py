"""Safety contracts for inert local specialist-candidate requests."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest


def _registry_api():
    from gateway.capability_registry import (
        CapabilityRegistry,
        CapabilitySignature,
        RegistryResolution,
    )

    return CapabilityRegistry, CapabilitySignature, RegistryResolution


def _candidate_api():
    try:
        from gateway.candidate_profile_requests import (
            CandidateProfileRequests,
            OpaqueEvidenceReference,
            SanitizedTaskEnvelope,
        )
    except ImportError as exc:  # RED: the split starts without this capability.
        pytest.fail(f"specialist candidate requests are unavailable: {exc}")
    return CandidateProfileRequests, OpaqueEvidenceReference, SanitizedTaskEnvelope


def _kanban_db():
    from hermes_cli import kanban_db

    return kanban_db


def _repository_review():
    _, CapabilitySignature, _ = _registry_api()
    return CapabilitySignature(
        domain="repository-evidence",
        actions=("read", "review"),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:read",),
    )


def _opaque(label: str):
    _, OpaqueEvidenceReference, _ = _candidate_api()
    return OpaqueEvidenceReference(hashlib.sha256(label.encode()).hexdigest())


def test_local_no_match_creates_one_inert_candidate(tmp_path):
    CandidateProfileRequests, _, SanitizedTaskEnvelope = _candidate_api()
    requests = CandidateProfileRequests(db_path=tmp_path / "candidates.db")

    result = requests.open_or_reuse(
        _repository_review(),
        source_key="gateway:request-1",
        envelope=SanitizedTaskEnvelope(evidence_refs=(_opaque("request-1"),)),
    )

    assert result.status == "candidate"
    assert result.profile_id is None
    assert result.request_id.startswith("cpr_")


def test_caller_supplied_no_match_cannot_override_local_active_match(tmp_path):
    CapabilityRegistry, _, RegistryResolution = _registry_api()
    CandidateProfileRequests, _, _ = _candidate_api()
    db_path = tmp_path / "candidates.db"
    signature = _repository_review()
    registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"repository-reviewer": signature},
    )
    registry.register_configured_profile("repository-reviewer")
    requests = CandidateProfileRequests(db_path=db_path)

    result = requests.open_or_reuse(
        signature,
        source_key="gateway:forged",
        resolution=RegistryResolution("no_match", None, "caller assertion"),
    )

    assert result.status == "rejected"
    assert result.request_id == ""
    with _kanban_db().connect_closing(db_path) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidate_profile_requests'"
        ).fetchone()
    assert table is None


def test_local_ambiguity_is_preserved_without_candidate_side_effect(tmp_path):
    CapabilityRegistry, CapabilitySignature, _ = _registry_api()
    CandidateProfileRequests, _, _ = _candidate_api()
    db_path = tmp_path / "candidates.db"
    broad = CapabilitySignature(
        domain="repository-evidence",
        actions=("read",),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:read",),
    )
    registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"one": broad, "two": broad},
    )
    registry.register_configured_profile("one")
    registry.register_configured_profile("two")

    result = CandidateProfileRequests(db_path=db_path).open_or_reuse(
        broad,
        source_key="gateway:ambiguous",
    )

    assert result.status == "rejected"
    assert result.request_id == ""
    assert "ambiguous" in result.reason
    with _kanban_db().connect_closing(db_path) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidate_profile_requests'"
        ).fetchone()
    assert table is None


def test_duplicate_source_and_scope_reuses_candidate(tmp_path):
    CandidateProfileRequests, _, _ = _candidate_api()
    requests = CandidateProfileRequests(db_path=tmp_path / "candidates.db")

    first = requests.open_or_reuse(_repository_review(), source_key="gateway:request-1")
    repeated = requests.open_or_reuse(_repository_review(), source_key="gateway:request-1")

    assert repeated.status == "duplicate"
    assert repeated.request_id == first.request_id
    assert repeated.profile_id is None


def test_write_scope_is_rejected_then_bounded_by_cooldown(tmp_path):
    _, CapabilitySignature, _ = _registry_api()
    CandidateProfileRequests, _, _ = _candidate_api()
    requests = CandidateProfileRequests(
        db_path=tmp_path / "candidates.db",
        cooldown_seconds=30,
        clock=lambda: 1_000,
    )
    write_scope = CapabilitySignature(
        domain="repository-evidence",
        actions=("read", "write"),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:read", "repository-evidence:write"),
    )

    rejected = requests.open_or_reuse(write_scope, source_key="gateway:request-2")
    repeated = requests.open_or_reuse(write_scope, source_key="gateway:request-2")

    assert rejected.status == "rejected"
    assert repeated.status == "cooldown"
    assert repeated.request_id == rejected.request_id


def test_raw_evidence_is_rejected_and_never_persisted(tmp_path):
    CandidateProfileRequests, _, SanitizedTaskEnvelope = _candidate_api()
    requests = CandidateProfileRequests(db_path=tmp_path / "candidates.db")

    result = requests.open_or_reuse(
        _repository_review(),
        source_key="gateway:unsafe-evidence",
        envelope=SanitizedTaskEnvelope(evidence_refs=("raw payload",)),
    )

    assert result.status == "rejected"
    with _kanban_db().connect_closing(tmp_path / "candidates.db") as conn:
        row = conn.execute(
            "SELECT evidence_ref_hashes_json FROM candidate_profile_requests WHERE request_id = ?",
            (result.request_id,),
        ).fetchone()
    assert "raw payload" not in row["evidence_ref_hashes_json"]


def test_candidate_rows_are_append_only(tmp_path):
    CandidateProfileRequests, _, _ = _candidate_api()
    requests = CandidateProfileRequests(db_path=tmp_path / "candidates.db")
    result = requests.open_or_reuse(_repository_review(), source_key="gateway:request-1")

    with _kanban_db().connect_closing(tmp_path / "candidates.db") as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE candidate_profile_requests SET lifecycle_status = 'active' WHERE request_id = ?",
                (result.request_id,),
            )
