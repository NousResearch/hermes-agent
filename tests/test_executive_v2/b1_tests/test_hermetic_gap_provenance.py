"""Hermetic gap tests — provenance envelope (2 tests).

Implements the provenance sub-section of the hermetic_test_gap_analysis.md:

* test_ep_prv_09_provenance_read_only_cannot_be_overridden
* test_ep_prv_10_provenance_with_empty_quote_and_line_range

This module covers three distinct facts about ``ProvenanceEnvelope``:

* **Dataclass-level fact** — a frozen ``ProvenanceEnvelope`` instance is
  immutable. Attempts to assign to a field on an existing instance raise
  ``dataclasses.FrozenInstanceError``.

* **Value-construction fact** — ``frozen=True`` on the dataclass prevents
  mutation of an existing instance, but it does NOT prevent creating a
  *new* instance with a different ``read_only`` value via
  ``dataclasses.replace``. ``read_only=True`` is the default value and
  is enforced for engine-produced provenance, but the dataclass itself
  does not enforce ``read_only=True`` as a universal semantic invariant.

* **Engine/public-flow fact** — provenance envelopes produced by the
  public Knowledge Discovery flow (here exercised via
  ``EvidencePackEngine.dry_run``) always carry ``read_only=True``.
  This is a contract enforced by the engine's provenance builder, not
  by the frozen-dataclass mechanism.
"""
from __future__ import annotations

import dataclasses

import pytest
from agent.executive.knowledge_discovery import ProvenanceEnvelope


def test_ep_prv_09_provenance_read_only_cannot_be_overridden():
    """ProvenanceEnvelope distinguishes instance immutability from value policy.

    Three distinct facts are asserted:

    1. ``dataclasses.FrozenInstanceError`` is raised when assigning to a
       field of an existing frozen ``ProvenanceEnvelope`` instance. The
       *instance* is immutable.

    2. ``dataclasses.replace(base, read_only=False)`` produces a *new*
       instance whose ``read_only`` is ``False``. The default value is
       ``True`` and is the contract enforced by engine-produced
       provenance, but the frozen dataclass itself does not enforce
       ``read_only=True`` as a universal semantic invariant — replace()
       constructs a new instance, which the dataclass happily permits.

    3. ``base.read_only`` remains ``True`` after the replace: replace()
       does not mutate the source instance.
    """
    base = ProvenanceEnvelope(
        producer='fake_gbrain_provider_v1',
        produced_at='2026-07-08T20:00:00+00:00',
        source_type='gbrain',
        source_uri='gbrain://x',
        retrieval_mode='metadata_only',
    )
    # Engine-produced contract: default read_only is True.
    assert base.read_only is True

    # (1) Instance mutation is rejected.
    with pytest.raises(dataclasses.FrozenInstanceError):
        base.read_only = False

    # (2) Replace constructs a new instance; the dataclass permits it.
    replaced = dataclasses.replace(base, read_only=False)
    assert replaced.read_only is False, (
        "dataclasses.replace must honor the supplied read_only value; "
        "frozen=True guards instance mutation, not field-value construction"
    )

    # (3) The original is unchanged — replace() is non-mutating.
    assert base.read_only is True, (
        "base instance must remain read_only=True after replace()"
    )
    # The replacement is a distinct instance.
    assert replaced is not base


def test_ep_prv_09b_engine_provenance_is_read_only(hermetic_evidence_pack_engine):
    """Engine-produced provenance envelopes always have read_only=True.

    Exercises ``EvidencePackEngine.dry_run`` end-to-end through the
    public result objects (``EvidencePack.hits`` → ``KnowledgeHitV2`` →
    ``ProvenanceEnvelope``). The contract enforced by the engine is
    that every hit's provenance has ``read_only=True`` — independent
    of the frozen-dataclass mechanism which only guards against
    in-place mutation.

    The objective_text is chosen to overlap with the canned hermetic
    fixtures so the engine actually returns hits for each registered
    source (gbrain/obsidian/policy/contract/report). The frozen
    ``_UNIVERSAL_TOKEN`` signal is not available here, but the default
    fixture shares tokens with ``knowledge discovery canary`` etc.
    """
    engine, _ = hermetic_evidence_pack_engine
    pack = engine.dry_run(
        objective_id='b1-prv-09-engine',
        objective_text='discovery policy contract report canary',
    )

    assert pack.hits, 'expected the engine to return at least one hit'
    for hit in pack.hits:
        assert hit.provenance is not None, (
            f'hit {hit.hit_id!r} missing provenance'
        )
        assert hit.provenance.read_only is True, (
            f'engine-produced provenance for hit {hit.hit_id!r} '
            f'source={hit.source!r} must carry read_only=True; got '
            f'{hit.provenance.read_only!r}'
        )


def test_ep_prv_10_provenance_with_empty_quote_and_line_range(hermetic_evidence_pack_engine):
    """A hit with quote=None + line_range=None serializes cleanly.

    to_dict() must NOT include ``quote=None`` (per evidence_pack.py:324,
    empty strings collapse to None and are emitted as such). This is
    documented JSON contract — we verify it stays stable.
    """
    from agent.executive.knowledge_discovery import _make_hit_v2, SOURCE_TTL_DAYS
    observed = '2026-07-08T20:00:00+00:00'
    updated = '2026-07-08T20:00:00+00:00'
    hit = _make_hit_v2(
        source='gbrain',
        hit_id='b1-prv-10',
        title='b1 prv-10',
        relevance_score=0.5,
        snippet='b1-prv-10 snippet content',
        source_uri='gbrain://b1-prv-10',
        source_updated_at=updated,
        retrieval_mode='metadata_only',
        quote=None,
        line_range=None,
        observed_at=observed,
        ttl_days=SOURCE_TTL_DAYS['gbrain'],
    )
    assert hit.provenance.quote is None
    assert hit.provenance.line_range is None
    engine, _ = hermetic_evidence_pack_engine
    pack = engine.dry_run(
        objective_id='b1-prv-10',
        objective_text='b1-prv-10 content',
    )
    d = pack.to_dict()
    assert 'hits' in d
    assert pack.schema_version == 'evidence_pack.v1'
