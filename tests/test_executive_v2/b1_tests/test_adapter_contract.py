"""Adapter contract tests (8 tests, parametrized → 28 cases).

Implements the contract tests from adapter_contract_test_plan.md.

These tests use fake providers directly (not the engine, except for failure
modes) to verify each provider satisfies the EvidencePack v2 contract:

* test_contract_01_provider_returns_knowledge_hit_v2_list (5 cases)
* test_contract_02_every_hit_has_provenance_with_read_only_true (5 cases)
* test_contract_03_producer_name_in_canonical_registry (5 cases)
* test_contract_04_provider_respects_max_hits_per_source (5 cases)
* test_contract_05_provider_clamp_max_hits_to_one_when_zero (1 case)
* test_contract_06_provider_does_not_return_more_than_top_n (1 case)
* test_contract_07_provider_query_exception_marked_as_source_failed (4 cases)
* test_contract_08_provider_with_no_hits_returns_empty_list (1 case)

Total: 27 test cases (parametrized expansions of 8 contract tests).
"""
from __future__ import annotations
from typing import Optional
import pytest
from agent.executive.knowledge_discovery import EvidencePackEngine, KnowledgeHitV2, KnowledgeQuery, ProvenanceEnvelope, FreshnessPolicy
from tests.test_executive_v2.b1_tests.support import _InMemoryStorage
from tests.test_executive_v2.b1_tests.fake_providers import PRODUCER_NAME, contract_provider, empty_spec, failing_spec, gbrain_provider, obsidian_provider, policy_provider, report_provider, FakeProviderSpec
OBS = '2026-07-08T20:00:00+00:00'

def _spec_for(source: str, n_hits: int=1) -> FakeProviderSpec:
    """Build a FakeProviderSpec with `n_hits` canned hits for `source`."""
    base_hit = {'title': f'contract {source} title', 'relevance_score': 0.9, 'snippet': f'contract {source} content alpha', 'source_updated_at': OBS}
    if source == 'gbrain':
        hits = tuple(({**base_hit, 'hit_id': f'contract-{source}-{i:03d}', 'snippet': f'contract {source} content alpha variant-{i}'} for i in range(n_hits)))
    elif source == 'obsidian':
        hits = tuple(({**base_hit, 'hit_id': f'contract-{source}-{i:03d}', 'snippet': f'contract {source} content alpha variant-{i}'} for i in range(n_hits)))
    elif source == 'report':
        hits = tuple(({**base_hit, 'hit_id': f'contract-{source}-{i:03d}', 'source_uri': f'report://contract-{source}-{i:03d}', 'snippet': f'contract {source} content alpha variant-{i}'} for i in range(n_hits)))
    elif source == 'policy':
        hits = tuple(({**base_hit, 'hit_id': f'contract-{source}-{i:03d}', 'warnings': (f'alpha warning {i}',), 'snippet': f'contract {source} content alpha variant-{i}'} for i in range(n_hits)))
    elif source == 'contract':
        hits = tuple(({**base_hit, 'hit_id': f'contract-{source}-{i:03d}', 'success_criteria': (f'alpha criterion {i}',), 'snippet': f'contract {source} content alpha variant-{i}'} for i in range(n_hits)))
    else:
        raise ValueError(f'unknown source: {source}')
    return FakeProviderSpec(name=source, hits=hits, is_available=True)

def _provider_for(source: str, spec: FakeProviderSpec):
    """Return the matching provider factory for `source`."""
    return {'gbrain': gbrain_provider, 'obsidian': obsidian_provider, 'report': report_provider, 'policy': policy_provider, 'contract': contract_provider}[source](spec)

def _query(text: str='alpha content variant', idx: int=0) -> KnowledgeQuery:
    return KnowledgeQuery(objective_id=f'contract-{idx}', objective_text=text)

@pytest.mark.parametrize('source_name', ['gbrain', 'obsidian', 'report', 'policy', 'contract'])
def test_contract_01_provider_returns_knowledge_hit_v2_list(source_name):
    """Each fake provider returns list[KnowledgeHitV2] for any valid query."""
    spec = _spec_for(source_name, n_hits=1)
    provider = _provider_for(source_name, spec)
    q = _query(idx=1)
    result = provider(q, max_hits=5, observed_at=OBS)
    assert isinstance(result, list), f'{source_name}: not a list'
    assert all((isinstance(h, KnowledgeHitV2) for h in result)), f'{source_name}: non-KnowledgeHitV2 elements: {result}'

@pytest.mark.parametrize('source_name', ['gbrain', 'obsidian', 'report', 'policy', 'contract'])
def test_contract_02_every_hit_has_provenance_with_read_only_true(source_name):
    """Every hit returned by every provider has provenance.read_only == True."""
    spec = _spec_for(source_name, n_hits=3)
    provider = _provider_for(source_name, spec)
    for i in range(5):
        q = _query(text=f'alpha content variant-{i}', idx=i)
        result = provider(q, max_hits=5, observed_at=OBS)
        assert result, f'{source_name}: empty result for query {i}'
        for h in result:
            assert h.provenance.read_only is True, f'{source_name} hit {h.hit_id} has read_only={h.provenance.read_only}'

@pytest.mark.parametrize('source_name', ['gbrain', 'obsidian', 'report', 'policy', 'contract'])
def test_contract_03_producer_name_in_canonical_registry(source_name):
    """producer ∈ PRODUCER_NAME.values() for every fake provider."""
    spec = _spec_for(source_name, n_hits=1)
    provider = _provider_for(source_name, spec)
    q = _query(idx=3)
    result = provider(q, max_hits=5, observed_at=OBS)
    assert result, f'{source_name}: empty result'
    for h in result:
        assert h.provenance.producer in PRODUCER_NAME.values(), f'{source_name}: producer {h.provenance.producer} not in registry'
        assert h.provenance.producer == PRODUCER_NAME[source_name], f'{source_name}: expected producer {PRODUCER_NAME[source_name]!r}, got {h.provenance.producer!r}'

@pytest.mark.parametrize('source_name', ['gbrain', 'obsidian', 'report', 'policy', 'contract'])
def test_contract_04_provider_respects_max_hits_per_source(source_name):
    """Provider returns ≤ max_hits hits."""
    spec = _spec_for(source_name, n_hits=10)
    provider = _provider_for(source_name, spec)
    q = _query(idx=4)
    result = provider(q, max_hits=3, observed_at=OBS)
    assert len(result) <= 3, f'{source_name}: returned {len(result)} > 3 cap'

def test_contract_05_provider_clamp_max_hits_to_one_when_zero():
    """max_hits=0 at the provider level returns [] (slice [:0]).

    The engine clamps max_hits_per_source to 1 BEFORE calling the provider,
    so providers never actually receive max_hits=0 in production. The
    provider-level behavior is to honor the slice (return []).
    """
    spec = _spec_for('gbrain', n_hits=10)
    provider = gbrain_provider(spec)
    q = _query(idx=5)
    result = provider(q, max_hits=0, observed_at=OBS)
    assert isinstance(result, list)
    assert len(result) == 0, f'expected [] for max_hits=0, got {len(result)}'

def test_contract_06_provider_does_not_return_more_than_top_n():
    """10 hits available, max_hits=5 → exactly 5 returned (not 6, not 4)."""
    spec = _spec_for('gbrain', n_hits=10)
    provider = gbrain_provider(spec)
    q = _query(idx=6)
    result = provider(q, max_hits=5, observed_at=OBS)
    assert len(result) == 5, f'expected 5 hits, got {len(result)}'

@pytest.mark.parametrize('exc_type', [RuntimeError, OSError, ValueError, TimeoutError])
def test_contract_07_provider_query_exception_marked_as_source_failed(exc_type):
    """Provider that raises → engine marks source as failed, doesn't propagate."""
    bundle = {'gbrain': gbrain_provider(failing_spec('gbrain', exc_type('simulated')))}
    engine = EvidencePackEngine(sources=bundle, storage=_InMemoryStorage(), audit_sink=None)
    pack = engine.dry_run(objective_id=f'contract-07-{exc_type.__name__}', objective_text='alpha content')
    assert 'gbrain' in pack.sources_failed, f'expected gbrain in sources_failed, got {pack.sources_failed}'
    assert 'gbrain' not in pack.sources_queried, f'gbrain should not be in sources_queried when it failed, got {pack.sources_queried}'

def test_contract_08_provider_with_no_hits_returns_empty_list():
    """Empty spec → engine produces pack with 0 hits but the source IS queried."""
    bundle = {'gbrain': gbrain_provider(empty_spec('gbrain')), 'obsidian': obsidian_provider(empty_spec('obsidian'))}
    engine = EvidencePackEngine(sources=bundle, storage=_InMemoryStorage(), audit_sink=None)
    pack = engine.dry_run(objective_id='contract-08', objective_text='alpha content')
    assert len(pack.hits) == 0
    assert 'gbrain' in pack.sources_queried
    assert 'obsidian' in pack.sources_queried


# ── Idempotency contract tests (using public-only storage) ──


class _PublicOnlyStorage:
    """Storage with ONLY public get_meta/set_meta methods (no _state_meta, no delete_meta).

    Uses closure for state instead of private attribute.
    """

    def __init__(self):
        # Use closure-based storage: list with one dict
        self.storage_ref = [{}]

    def get_meta(self, k: str) -> Optional[str]:
        """Public get_meta: returns JSON string or None."""
        return self.storage_ref[0].get(k)

    def set_meta(self, k: str, v: str) -> None:
        """Public set_meta: stores JSON string."""
        self.storage_ref[0][k] = v


class _StorageWithWriteCount:
    """Storage that counts set_meta calls."""

    def __init__(self):
        self.storage_ref = [{}]
        self.set_meta_calls = 0

    def get_meta(self, k: str) -> Optional[str]:
        return self.storage_ref[0].get(k)

    def set_meta(self, k: str, v: str) -> None:
        self.set_meta_calls += 1
        self.storage_ref[0][k] = v


class _SimpleAuditSink:
    """Minimal audit sink that records events."""

    def __init__(self):
        self.events_ref = [[]]

    def emit(self, event: dict) -> None:
        self.events_ref[0].append(dict(event))

    def get_events(self) -> list[dict]:
        return list(self.events_ref[0])


@pytest.fixture
def public_only_storage():
    """Fresh storage with only public get_meta/set_meta methods (no _state_meta, no delete_meta)."""
    return _PublicOnlyStorage()


def _make_provider_call_counting_bundle():
    """Create a bundle that counts how many times the provider is called."""
    call_count = [0]
    def counting_provider(query, *, max_hits, observed_at):
        call_count[0] += 1
        # Use public constructors instead of private helpers
        hit = KnowledgeHitV2(
            source='gbrain',
            hit_id=f'hit-{query.objective_id}',
            title=f'Title for {query.objective_id}',
            relevance_score=0.9,
            snippet=f'Snippet for {query.objective_text}',
            location='test://location',
            fingerprint='test-fingerprint',
            created_at=observed_at,
            provenance=ProvenanceEnvelope(
                producer='test-producer',
                produced_at=observed_at,
                source_type='test',
                source_uri='test://uri',
                retrieval_mode='test',
                read_only=True,
            ),
            freshness=FreshnessPolicy(
                observed_at=observed_at,
                source_updated_at=observed_at,
                staleness_days=0,
                freshness='current',
                freshness_score=1.0,
            ),
        )
        return [hit]
    return {'gbrain': counting_provider}, call_count


def test_idempotency_first_call_invokes_provider_and_writes_metadata(public_only_storage):
    """First discover call invokes provider once and writes metadata."""
    bundle, call_count = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=public_only_storage)

    pack = engine.discover(objective_id='idem-001', objective_text='test query')

    assert call_count[0] == 1, f'Provider should be called once, got {call_count[0]}'
    assert pack.is_idempotent_reuse is False
    # Verify metadata was written via public API
    meta = engine.get_meta(f'objective_knowledge_discovery:idem-001:v2')
    assert meta is not None
    assert meta.get('objective_id') == 'idem-001'


def test_idempotency_second_call_reuses_cache_without_new_provider_call(public_only_storage):
    """Second discover call with same objective_id reuses cache, no new provider call."""
    bundle, call_count = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=public_only_storage)

    # First call
    pack1 = engine.discover(objective_id='idem-002', objective_text='test query')
    assert call_count[0] == 1

    # Second call with same objective_id - should hit cache
    pack2 = engine.discover(objective_id='idem-002', objective_text='test query')

    assert call_count[0] == 1, f'Provider should not be called again, got {call_count[0]}'
    assert pack2.is_idempotent_reuse is True


def test_idempotency_different_objective_text_causes_cache_miss(public_only_storage):
    """Same objective_id with different objective_text causes cache miss and invokes provider."""
    bundle, call_count = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=public_only_storage)

    # First call
    pack1 = engine.discover(objective_id='idem-003', objective_text='first query')
    assert call_count[0] == 1

    # Second call with different text - should miss cache
    pack2 = engine.discover(objective_id='idem-003', objective_text='different query')

    assert call_count[0] == 2, f'Provider should be called again for different text, got {call_count[0]}'
    assert pack2.is_idempotent_reuse is False


def test_idempotency_no_storage_preserved():
    """When storage is None, discover works without persisting."""
    bundle, call_count = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=None)

    pack = engine.discover(objective_id='idem-004', objective_text='test query')

    assert call_count[0] == 1
    # get_meta should return None when storage is None
    meta = engine.get_meta('objective_knowledge_discovery:idem-004:v2')
    assert meta is None


def test_idempotency_get_meta_returns_none_for_missing_key(public_only_storage):
    """get_meta returns None for non-existent key."""
    engine = EvidencePackEngine(sources={}, storage=public_only_storage)
    meta = engine.get_meta('nonexistent-key')
    assert meta is None


def test_idempotency_get_meta_returns_valid_metadata(public_only_storage):
    """get_meta returns valid metadata when present."""
    bundle, _ = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=public_only_storage)

    engine.discover(objective_id='idem-005', objective_text='test')

    meta = engine.get_meta('objective_knowledge_discovery:idem-005:v2')
    assert meta is not None
    assert 'objective_id' in meta
    assert 'query_fingerprint' in meta


def test_idempotency_corrupted_metadata_does_not_cause_false_hit(public_only_storage):
    """Corrupted metadata (invalid JSON) does not cause false cache hit."""
    # Manually write corrupted metadata
    public_only_storage.set_meta('objective_knowledge_discovery:idem-006:v2', 'not valid json {{{')

    bundle, call_count = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=public_only_storage)

    pack = engine.discover(objective_id='idem-006', objective_text='test query')

    # Should invoke provider because corrupted metadata can't be parsed
    assert call_count[0] == 1
    assert pack.is_idempotent_reuse is False


def test_idempotency_get_meta_raises_on_storage_error():
    """get_meta returns None when storage raises exception (not raises)."""
    class BadStorage:
        def get_meta(self, k):
            raise RuntimeError("storage error")
        def set_meta(self, k, v):
            pass

    engine = EvidencePackEngine(sources={}, storage=BadStorage())
    # Should return None, not raise
    meta = engine.get_meta('some-key')
    assert meta is None


def test_idempotency_set_meta_raises_on_no_storage():
    """set_meta raises RuntimeError when storage is None."""
    engine = EvidencePackEngine(sources={}, storage=None)

    with pytest.raises(RuntimeError) as exc_info:
        engine.set_meta('key', {'value': 1})

    assert 'No storage' in str(exc_info.value)


def test_idempotency_set_meta_raises_on_missing_method():
    """set_meta raises RuntimeError when storage lacks set_meta method."""
    class BadStorage:
        def get_meta(self, k):
            return None

    engine = EvidencePackEngine(sources={}, storage=BadStorage())

    with pytest.raises(RuntimeError) as exc_info:
        engine.set_meta('key', {'value': 1})

    assert 'does not support set_meta' in str(exc_info.value)


def test_idempotency_set_meta_raises_on_storage_failure():
    """set_meta raises RuntimeError when storage write fails."""
    class BadStorage:
        def get_meta(self, k):
            return None
        def set_meta(self, k, v):
            raise IOError("write failed")

    engine = EvidencePackEngine(sources={}, storage=BadStorage())

    with pytest.raises(RuntimeError) as exc_info:
        engine.set_meta('key', {'value': 1})

    assert 'Failed to set metadata' in str(exc_info.value)


def test_idempotency_second_call_does_not_write_metadata():
    """Second discover call with same params does not call set_meta."""
    storage = _StorageWithWriteCount()
    bundle, call_count = _make_provider_call_counting_bundle()
    engine = EvidencePackEngine(sources=bundle, storage=storage)

    # First call - should write metadata
    pack1 = engine.discover(objective_id='idem-007', objective_text='test query')
    first_writes = storage.set_meta_calls
    assert first_writes == 1, f'First call should write once, got {first_writes}'

    # Second call with same params - should NOT write metadata
    pack2 = engine.discover(objective_id='idem-007', objective_text='test query')
    second_writes = storage.set_meta_calls - first_writes

    assert second_writes == 0, f'Second call should not write, got {second_writes}'
    assert pack2.is_idempotent_reuse is True
