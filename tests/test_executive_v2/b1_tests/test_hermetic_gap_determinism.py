"""Hermetic gap tests — determinism (2 tests).

Implements the determinism sub-section of the hermetic_test_gap_analysis.md:

* test_ep_det_08_summary_fingerprint_changes_with_hits_change
* test_ep_det_09_two_pass_sha256_deterministic_across_instances
"""
from __future__ import annotations

def test_ep_det_08_summary_fingerprint_changes_with_hits_change(hermetic_evidence_pack_engine):
    """If the hit set changes (different sources/snippets), the summary_fingerprint
    must change accordingly — determinism is *input-sensitive*, not random.
    """
    engine, _ = hermetic_evidence_pack_engine
    pack_a = engine.dry_run(objective_id='b1-det-08-a', objective_text='discovery policy contract')
    pack_b = engine.dry_run(objective_id='b1-det-08-b', objective_text='completely different gbrain obsidian keywords')
    assert pack_a.query_fingerprint != pack_b.query_fingerprint
    assert pack_a.summary_fingerprint != pack_b.summary_fingerprint, 'summary_fingerprint did not change between distinct queries'

def test_ep_det_09_two_pass_sha256_deterministic_across_instances(frozen_time, in_memory_storage, audit_capture, provider_bundle):
    """Two engine instances built with the same bundle yield identical fingerprints.

    The 2-pass sha256 strategy (canonical JSON + sort_keys + ensure_ascii=False)
    must be reproducible across instantiations.
    """
    from agent.executive.knowledge_discovery import EvidencePackEngine
    eng1 = EvidencePackEngine(sources=provider_bundle, storage=in_memory_storage, audit_sink=audit_capture)
    eng2 = EvidencePackEngine(sources=provider_bundle, storage=in_memory_storage, audit_sink=audit_capture)
    p1 = eng1.dry_run(objective_id='b1-det-09', objective_text='discovery policy contract report')
    p2 = eng2.dry_run(objective_id='b1-det-09', objective_text='discovery policy contract report')
    assert p1.query_fingerprint == p2.query_fingerprint
    assert p1.summary_fingerprint == p2.summary_fingerprint
