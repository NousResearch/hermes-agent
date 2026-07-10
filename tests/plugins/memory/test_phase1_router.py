"""Tests for the Phase 1 Memory Router (classify / registry / dispatch / log).

These tests are hermetic: HERMES_HOME is redirected to tmp_path by the
autouse ``_hermetic_environment`` fixture in tests/conftest.py. We additionally
monkeypatch ``hermes_constants.get_hermes_home`` to be explicit and safe.
"""

from __future__ import annotations

import logging

import pytest

from hermes_cli.memory_index.indexer import MemoryIndex
from hermes_cli.memory_router import MemoryRouter, classify
from hermes_cli.memory_router.classify import tokenize
from hermes_cli.memory_router.intents import Intent
from hermes_cli.memory_router.provenance import SearchResult, format_results
from hermes_cli.memory_router.registry import Capability, CapabilityRegistry


@pytest.fixture
def router():
    return MemoryRouter()


# --------------------------------------------------------------------------- #
# classify() — rule-based, deterministic intent classification
# --------------------------------------------------------------------------- #
def test_classify_identity():
    assert classify("who am i") == Intent.IDENTITY
    assert classify("tell me about joe") == Intent.IDENTITY
    assert classify("about hermes") == Intent.IDENTITY
    assert classify("what is my identity") == Intent.IDENTITY


def test_classify_project_state():
    assert classify("what is the project roadmap") == Intent.PROJECT_STATE
    assert classify("show project progress") == Intent.PROJECT_STATE


def test_classify_decision():
    assert classify("what was the decision about X") == Intent.DECISION
    assert classify("show me the ADR for auth") == Intent.DECISION


def test_classify_historical():
    assert classify("what did we discuss about cats") == Intent.HISTORICAL
    assert classify("search for the meeting notes") == Intent.HISTORICAL
    assert classify("history of the billing system") == Intent.HISTORICAL


def test_classify_relationship():
    assert classify("who is related to the payment module") == Intent.RELATIONSHIP
    assert classify("what is the connection between A and B") == Intent.RELATIONSHIP


def test_classify_recent():
    assert classify("what happened in the last session") == Intent.RECENT
    assert classify("show me recent activity") == Intent.RECENT


def test_classify_context():
    assert classify("give me context on the deploy pipeline", intent_hint="context") == Intent.CONTEXT


def test_classify_low_confidence_defaults_to_historical():
    # No matching keyword -> safe default (broad index).
    assert classify("some random thing about cats") == Intent.HISTORICAL
    assert classify("") == Intent.HISTORICAL


def test_classify_intent_hint_override():
    # Explicit hint is honored deterministically.
    assert classify("who am i", intent_hint="historical") == Intent.HISTORICAL
    assert classify("random text", intent_hint="decision") == Intent.DECISION


def test_classify_is_deterministic():
    q = "the quick brown fox jumped over the lazy dog"
    assert classify(q) == classify(q)
    assert classify(q) == Intent.HISTORICAL  # no keyword -> default


def test_tokenize_drops_stopwords():
    toks = tokenize("the cat and the dog")
    assert "the" not in toks
    assert "cat" in toks and "dog" in toks


# --------------------------------------------------------------------------- #
# Registry — only routes to available, registered capabilities
# --------------------------------------------------------------------------- #
def test_registry_only_routes_to_available(router):
    # Default registry has L5 (historical) + L1 (identity) registered.
    cap = router.registry.get_available(Intent.HISTORICAL)
    assert cap is not None and cap.name == "L5-index"

    # project_state IS wired (Phase 6: L2-project capability registered by default).
    assert router.registry.get_available(Intent.PROJECT_STATE) is not None
    res = router.project(project="hermes-aios")
    # No STATUS.md exists for that project -> graceful None, not a crash.
    assert res.ok is True
    assert res.capability == "L2-project"
    # No STATUS.md -> empty/absent (never fabricated).
    assert res.results in (None, [])


def test_registry_set_availability():
    reg = CapabilityRegistry()
    reg.register("X", [Intent.RECENT], True, lambda *a, **k: [])
    assert reg.get_available(Intent.RECENT) is not None
    reg.set_availability("X", False)
    assert reg.get_available(Intent.RECENT) is None
    reg.set_availability("X", True)
    assert reg.get_available(Intent.RECENT) is not None


def test_registry_list_and_names():
    reg = CapabilityRegistry()
    reg.register("A", [Intent.IDENTITY], True, lambda *a, **k: [])
    reg.register("B", [Intent.HISTORICAL], False, lambda *a, **k: [])
    assert set(reg.names()) == {"A", "B"}
    assert len(reg.list()) == 2


# --------------------------------------------------------------------------- #
# Routing logs every decision (caplog on hermes.memory_router)
# --------------------------------------------------------------------------- #
def test_routing_logs_decision(router, caplog):
    with caplog.at_level(logging.INFO, logger="hermes.memory_router"):
        router.search("who am i")  # identity -> L1
        router.search("history of billing")  # historical -> L5
    records = [r.getMessage() for r in caplog.records]
    joined = "\n".join(records)
    assert "route intent=identity" in joined
    assert "capability=L1-identity" in joined
    assert "route intent=historical" in joined
    assert "capability=L5-index" in joined


def test_routing_records_last_routes(router):
    router.search("who am i")
    assert router.last_routes
    rec = router.last_routes[-1]
    assert rec.intent == "identity"
    assert rec.capability == "L1-identity"
    assert rec.method == "search"
    assert rec.fallback_used is False


def test_routing_unavailable_records_fallback_flag(router, caplog):
    # RELATIONSHIP has no backend registered -> genuinely unavailable.
    with caplog.at_level(logging.INFO, logger="hermes.memory_router"):
        res = router.route(Intent.RELATIONSHIP, "search", query="x")
    assert res.ok is False
    assert "NO CAPABILITY" in "\n".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# Router does not depend on an LLM (determinism + no openai/anthropic import)
# --------------------------------------------------------------------------- #
def test_router_does_not_depend_on_llm(router):
    # Determinism across repeated calls with identical input.
    a = router.search("decision about caching")
    b = router.search("decision about caching")
    assert str(a.intent) == str(b.intent)
    assert a.capability == b.capability


def test_router_modules_no_llm_imports():
    import importlib
    import inspect

    for mod_name in [
        "hermes_cli.memory_router.router",
        "hermes_cli.memory_router.classify",
        "hermes_cli.memory_router.registry",
        "hermes_cli.memory_router.capabilities",
        "hermes_cli.memory_index.indexer",
    ]:
        mod = importlib.import_module(mod_name)
        src = inspect.getsource(mod).lower()
        for forbidden in ("openai", "anthropic", "import llm", "from llm"):
            assert forbidden not in src, f"{mod_name} references {forbidden!r}"


# --------------------------------------------------------------------------- #
# Provenance present on every search result
# --------------------------------------------------------------------------- #
def test_provenance_present(router):
    for q in ["who am i", "history of billing", "search for cats"]:
        res = router.search(q)
        for r in (res.results or []):
            assert isinstance(r, SearchResult)
            assert r.source_file
            assert r.memory_layer
            assert r.retrieval_method


def test_provenance_format_results_roundtrip():
    r = SearchResult(
        source_file="memories/MEMORY.md",
        memory_layer="L5-notes",
        retrieval_method="fts5",
        content="hello world",
        intent="historical",
        capability="L5-index",
    )
    d = format_results([r])[0]
    assert d["source_file"] == "memories/MEMORY.md"
    assert d["memory_layer"] == "L5-notes"
    assert d["retrieval_method"] == "fts5"
    assert d["snippet"] == "hello world"


def test_context_fanout_only_l5_in_phase1(router):
    res = router.context("search for billing")
    assert res.intent == Intent.CONTEXT
    # In Phase 1 only L5 participates.
    assert res.capability is None or "L5-index" in res.capability
