from datetime import datetime, timezone

import pytest

from hermes_state import SessionDB
from research.evidence_fabric import (
    ClaimStatus,
    EvidenceFabricService,
    EvidenceIntegrityError,
    EvidenceLifecycleError,
    EvidenceScope,
    EvidenceValidationError,
    ResearchRunStatus,
    canonicalize_uri,
    content_sha256,
)


def _service(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    scope = EvidenceScope("scope", "profile", "connection", "agent")
    return db, EvidenceFabricService(db, scope)


def test_hash_and_uri_helpers_are_deterministic():
    assert content_sha256("café") == content_sha256("cafe\u0301")
    assert content_sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert canonicalize_uri("HTTP://Example.TEST:80") == "http://example.test/"
    assert canonicalize_uri("https://Example.TEST:443/a#fragment") == "https://example.test/a"
    assert canonicalize_uri("https://example.test/a?x=1&x=2&Track=Yes") == "https://example.test/a?x=1&x=2&Track=Yes"


def test_service_creates_run_evidence_claim_link_and_status_provenance(tmp_path):
    db, service = _service(tmp_path)
    try:
        run = service.create_research_run("Find the answer")
        evidence = service.add_evidence(
            run.id,
            source_type="WEB_PAGE",
            retrieval_method="DIRECT_HTTP",
            content="source text",
            source_uri="https://Example.test#x",
        ).evidence
        claim = service.create_claim(run.id, "The answer is supported")
        link = service.link_evidence_to_claim(claim.id, evidence.id, "SUPPORTS")
        updated = service.set_claim_status(claim.id, ClaimStatus.SUPPORTED)
        assert run.status is ResearchRunStatus.OPEN
        assert link.created_by_agent == "agent"
        assert link.created_by_profile == "profile"
        assert updated.status is ClaimStatus.SUPPORTED
        assert updated.updated_by_agent == "agent"
        assert updated.updated_by_profile == "profile"
    finally:
        db.close()


def test_scope_is_runtime_owned_and_other_scope_cannot_read_or_mutate(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    owner = EvidenceFabricService(db, EvidenceScope("scope-a", "p", "c", "agent-a"))
    other = EvidenceFabricService(db, EvidenceScope("scope-b", "p", "c", "agent-b"))
    try:
        run = owner.create_research_run("private objective")
        from research.evidence_fabric import EvidenceNotFoundError, EvidenceScopeError
        with pytest.raises(EvidenceScopeError):
            other.get_research_run(run.id)
        assert other.list_research_runs() == ()
        with pytest.raises(EvidenceScopeError):
            other.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content="x", raw_reference="artifact:x")
    finally:
        db.close()


def test_all_terminal_run_graph_mutations_are_rejected(tmp_path):
    db, service = _service(tmp_path)
    try:
        run = service.create_research_run("objective")
        evidence = service.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content="x", raw_reference="artifact:x").evidence
        claim = service.create_claim(run.id, "claim")
        service.transition_research_run(run.id, ResearchRunStatus.CANCELLED)
        with pytest.raises(EvidenceLifecycleError):
            service.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content="y", raw_reference="artifact:y")
        with pytest.raises(EvidenceLifecycleError):
            service.create_claim(run.id, "late claim")
        with pytest.raises(EvidenceLifecycleError):
            service.link_evidence_to_claim(claim.id, evidence.id, "CONTEXT")
        with pytest.raises(EvidenceLifecycleError):
            service.set_claim_status(claim.id, ClaimStatus.SUPPORTED)
    finally:
        db.close()


def test_evidence_deduplicates_by_run_uri_or_raw_reference_and_service_hash(tmp_path):
    db, service = _service(tmp_path)
    try:
        run = service.create_research_run("objective")
        first = service.add_evidence(run.id, source_type="WEB_PAGE", retrieval_method="DIRECT_HTTP", content="same", source_uri="https://example.test")
        duplicate = service.add_evidence(run.id, source_type="WEB_PAGE", retrieval_method="DIRECT_HTTP", content="same", source_uri="HTTPS://EXAMPLE.TEST:443/")
        assert first.created is True
        assert duplicate.created is False
        assert duplicate.evidence.id == first.evidence.id
        distinct = service.add_evidence(run.id, source_type="WEB_PAGE", retrieval_method="DIRECT_HTTP", content="different", source_uri="https://example.test")
        assert distinct.created is True
        raw1 = service.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content=b"bytes", raw_reference="artifact:x")
        raw2 = service.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content=b"bytes", raw_reference="artifact:x")
        assert raw2.created is False and raw2.evidence.id == raw1.evidence.id
        verified = service.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content="verified", raw_reference="artifact:y", expected_content_hash=content_sha256("verified"))
        assert verified.created is True
        with pytest.raises(EvidenceValidationError):
            service.add_evidence(run.id, source_type="FILE", retrieval_method="FILE_READ", content="wrong", raw_reference="artifact:w", expected_content_hash=content_sha256("verified"))
    finally:
        db.close()


def test_validation_rejects_bad_hash_uri_and_terminal_mutation(tmp_path):
    db, service = _service(tmp_path)
    try:
        run = service.create_research_run("objective")
        with pytest.raises(EvidenceValidationError):
            service.add_evidence(run.id, source_type="WEB_PAGE", retrieval_method="DIRECT_HTTP", content="x", expected_content_hash="x")
        with pytest.raises(EvidenceValidationError):
            canonicalize_uri("relative/path")
        service.transition_research_run(run.id, ResearchRunStatus.COMPLETED)
        with pytest.raises(EvidenceLifecycleError):
            service.create_claim(run.id, "too late")
    finally:
        db.close()


def test_research_run_rejects_open_to_open_and_terminal_transitions(tmp_path):
    db, service = _service(tmp_path)
    try:
        run = service.create_research_run("objective")
        with pytest.raises(EvidenceLifecycleError):
            service.transition_research_run(run.id, ResearchRunStatus.OPEN)
        service.transition_research_run(run.id, ResearchRunStatus.COMPLETED)
        with pytest.raises(EvidenceLifecycleError):
            service.transition_research_run(run.id, ResearchRunStatus.OPEN)
        with pytest.raises(EvidenceLifecycleError):
            service.transition_research_run(run.id, ResearchRunStatus.FAILED)
    finally:
        db.close()


def test_cross_run_link_is_a_domain_integrity_error_not_a_lifecycle_error(tmp_path):
    db, service = _service(tmp_path)
    try:
        run_a = service.create_research_run("a")
        run_b = service.create_research_run("b")
        evidence = service.add_evidence(run_a.id, source_type="FILE", retrieval_method="FILE_READ", content="a", raw_reference="artifact:a").evidence
        claim = service.create_claim(run_b.id, "b")
        with pytest.raises(EvidenceIntegrityError):
            service.link_evidence_to_claim(claim.id, evidence.id, "CONTEXT")
    finally:
        db.close()
