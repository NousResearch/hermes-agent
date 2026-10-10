"""Authenticated bounded reads must preserve the canonical file's raw bytes."""
from hashlib import sha256
from agent.context_references import preprocess_context_references
from agent.source_provenance import DEFAULT_POLICY_DIGEST, SourceProvenanceRegistry


def test_crlf_bounded_reference_issues_grant_for_exact_raw_slice(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"first = 1\r\nsecond = 2\r\n")
    registry = SourceProvenanceRegistry()
    result = preprocess_context_references(
        "Inspect @file:source.py:1-1", cwd=tmp_path, context_length=100_000,
        source_provenance_registry=registry, session_id="session", turn_id="turn",
        request_id="turn:api:1", policy_digest=DEFAULT_POLICY_DIGEST,
    )
    grants = registry.grants_for_request("turn:api:1")
    assert len(grants) == 1, result.warnings
    assert grants[0].content_sha256 == sha256(b"first = 1\r\n").hexdigest()
    assert "first = 1\r\n" in result.message
    assert "second = 2" not in result.message
