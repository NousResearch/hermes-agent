"""Phase 4 — Memory API abstraction contract tests.

Verifies the stable front door (MemoryAPI) and the structural Provider Protocol
without depending on any specific storage backend beyond the existing SQLite
index. Asserts invariants, not snapshots:

- Every API result carries provenance (source/provider/layer/retrieval_method).
- context() is STRUCTURED (categorized), not a raw concatenated string.
- Writes NEVER report silent success: unsupported writes raise a typed error.
- The IndexMemoryProvider is structurally compatible (Protocol, no base class).
- No existing memory behavior is changed (router still routes; index unchanged).
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from hermes_cli.memory_api import (
    CapabilityError,
    ContextBundle,
    IndexMemoryProvider,
    MemoryAPI,
    MemoryResult,
    UnsupportedCapability,
)
from hermes_cli.memory_api.protocols import MemoryProvider


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "sessions").mkdir(parents=True)
    sp = h / "sessions" / "s1.jsonl"
    sp.write_text(
        "\n".join([
            '{"role":"session_meta","session_id":"s1"}',
            '{"role":"user","content":"the lighthouse keeps the archive stable"}',
            '{"role":"assistant","content":"indexed via on_session_end lifecycle"}',
        ])
    )
    # a markdown note so L5 / identity search has something to hit
    (h / "memories").mkdir()
    (h / "memories" / "note.md").write_text("lighthouse archive policy\n")
    # Point the whole memory subsystem at the test home so the router's default
    # index (and the seeded provider) read from here, not real ~/.hermes.
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def api(home):
    from hermes_cli.memory_index.indexer import MemoryIndex

    idx = MemoryIndex(hermes_home=home)
    idx.build(home)
    # Construct AFTER HERMES_HOME is set, so router + default provider use it.
    return MemoryAPI()


# --------------------------------------------------------------------------- #
# Structural Protocol compatibility (no base class)
# --------------------------------------------------------------------------- #
def test_provider_is_structurally_compatible():
    # IndexMemoryProvider must satisfy the MemoryProvider Protocol by SHAPE,
    # not by inheritance. isinstance against a runtime_checkable Protocol.
    assert isinstance(IndexMemoryProvider(), MemoryProvider)


def test_provider_has_no_memoryprovider_base_class():
    # Guard the directive: providers MUST NOT require a shared base class.
    bases = IndexMemoryProvider.__mro__
    assert MemoryProvider not in bases
    assert all("memory_api.protocols" not in str(b) or "MemoryProvider" not in str(b) for b in bases[1:])


# --------------------------------------------------------------------------- #
# Read operations return normalized, provenance-bearing results
# --------------------------------------------------------------------------- #
def test_search_returns_memory_results_with_provenance(api):
    results = api.search("lighthouse")
    assert results, "expected at least one hit"
    assert all(isinstance(r, MemoryResult) for r in results)
    for r in results:
        assert r.source
        assert r.provider
        assert r.layer
        assert r.retrieval_method
        assert r.content


def test_archive_returns_l3_chunks(api):
    results = api.archive(session_id="s1")
    assert len(results) == 2
    assert all(r.layer == "L3-archive" for r in results)
    assert any("lighthouse" in r.content for r in results)


def test_recent_returns_results(api):
    results = api.recent(limit=5)
    assert results
    assert all(isinstance(r, MemoryResult) for r in results)


def test_search_empty_query_is_graceful(api):
    # No crash, returns a (possibly empty) list — never raises on no match.
    assert isinstance(api.search("zzz-no-such-token-zzz"), list)


# --------------------------------------------------------------------------- #
# Context is STRUCTURED (categorized), not concatenated
# --------------------------------------------------------------------------- #
def test_context_is_structured_bundle(api):
    bundle = api.context("lighthouse")
    assert isinstance(bundle, ContextBundle)
    # Results are bucketed by category, each entry provenance-bearing.
    assert isinstance(bundle.identity, list)
    assert isinstance(bundle.recent, list)
    assert isinstance(bundle.decision, list)
    assert isinstance(bundle.project, list)
    assert isinstance(bundle.other, list)
    # No LLM reasoning/ranking performed: the bundle is just categorized data.
    assert bundle.identity or bundle.recent
    for cat in (bundle.identity, bundle.recent, bundle.decision, bundle.project, bundle.other):
        for r in cat:
            assert isinstance(r, MemoryResult)
            assert r.source and r.provider and r.layer and r.retrieval_method


def test_context_does_not_concatenate_to_string(api):
    bundle = api.context("lighthouse")
    # The structured merge must NOT collapse everything into one opaque string.
    assert not isinstance(bundle, str)
    assert not isinstance(bundle.all_results(), str)


# --------------------------------------------------------------------------- #
# Writes are NEVER silent no-ops — they raise typed errors
# --------------------------------------------------------------------------- #
def test_remember_raises_on_read_only_provider(api):
    with pytest.raises(CapabilityError):
        api.remember("some content", layer="L5")


def test_decision_is_wired_after_phase5(api):
    # Phase 5 built L4: decision() no longer raises UnsupportedCapability.
    # With no ADRs it returns an empty list gracefully (not a fake success,
    # and not a crash) — the trust boundary is enforced at the read path,
    # not by raising. (Proposed-vs-accepted enforcement lives in
    # test_phase5_adr.py.)
    assert api.decision() == []


def test_project_absent_returns_none_gracefully(api):
    # Phase 6 implemented: project() is wired and returns None (no exception)
    # when no STATUS.md exists — it never fabricates project state.
    assert api.project("nonexistent-project") is None


# --------------------------------------------------------------------------- #
# Graceful degradation: capability status is queryable
# --------------------------------------------------------------------------- #
def test_capability_status_exposes_provider_readiness(api):
    status = api.capability_status()
    # Capability names come from the Router's registry (backend-agnostic),
    # not the old facade provider registry. L2-project is the Phase 6 owner.
    assert "L2-project" in status
    assert status["L2-project"] in ("available", "unavailable")
