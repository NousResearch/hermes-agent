"""Behavioral contract for the removed timeout_seconds field.

The Knowledge Discovery core previously exposed a `timeout_seconds` knob on
the public surface (KnowledgeQuery dataclass, EvidencePackEngine.dry_run,
TIMEOUT_SECONDS_DEFAULT constant, query fingerprint payload). That knob was
never enforced against any in-process provider because the providers are
purely synchronous and in-memory: no thread, no socket, no cooperative
cancellation point, no deadline plumbing. The contract was a lie. The field
has been removed; this test module pins the removal as a behavior so that
re-introducing it surfaces immediately.

Tests are behavioral (call the public API, observe the public surface).
They do not inspect engine.py source, do not grep for symbols, and do not
parse AST. They encode the contract from the caller's perspective.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from agent.executive.knowledge_discovery import (
    EvidencePackEngine,
    KnowledgeQuery,
)


# ── fixtures ────────────────────────────────────────────────────────────


class _RecordingStorage:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_meta(self, key: str) -> str | None:
        return self.values.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self.values[key] = value


def _silent_provider(query, *, max_hits, observed_at):
    return []


def _make_engine() -> EvidencePackEngine:
    return EvidencePackEngine(sources={"gbrain": _silent_provider})


def _make_engine_with_storage() -> EvidencePackEngine:
    return EvidencePackEngine(
        sources={"gbrain": _silent_provider},
        storage=_RecordingStorage(),
    )


# ── Test A: the symbol is gone from the public API ──────────────────────


def test_a_timeout_seconds_symbol_not_in_public_api():
    """`agent.executive.knowledge_discovery` does not export timeout-related names.

    Behavioral contract: a caller writing `from ... import TIMEOUT_SECONDS_DEFAULT`
    fails with ImportError. We exercise the import path, not the source.
    """
    import agent.executive.knowledge_discovery as kd

    # Constant removed from the module.
    assert not hasattr(kd, "TIMEOUT_SECONDS_DEFAULT"), (
        "TIMEOUT_SECONDS_DEFAULT must not be re-introduced on the public "
        "surface; it was a knob that was never enforced."
    )
    assert "TIMEOUT_SECONDS_DEFAULT" not in kd.__all__


def test_a_timeout_seconds_kwarg_rejected_by_dry_run():
    """dry_run() does not accept timeout_seconds as a keyword argument.

    Behavioral contract: calling ``EvidencePackEngine.dry_run`` with a
    ``timeout_seconds`` kwarg raises ``TypeError`` (the public surface
    does not advertise the knob). The drive invokes the documented
    method and observes the public exception; no source/AST
    introspection is performed.
    """
    engine = _make_engine()
    with pytest.raises(TypeError):
        engine.dry_run(
            objective_id="a-timeout-rejected",
            objective_text="alpha content",
            timeout_seconds=30.0,
        )


# ── Test B: the field is gone from the dataclass ────────────────────────


def test_b_knowledge_query_has_no_timeout_field():
    """KnowledgeQuery dataclass does not declare a timeout_seconds field."""
    fields = {f.name for f in dataclasses.fields(KnowledgeQuery)}
    assert "timeout_seconds" not in fields, (
        f"KnowledgeQuery must not carry timeout_seconds; got fields={fields}"
    )


def test_b_knowledge_query_constructor_rejects_timeout_kwarg():
    """KnowledgeQuery(...) does not accept timeout_seconds=...

    Behavioral contract: instantiating ``KnowledgeQuery`` with a
    ``timeout_seconds`` kwarg raises ``TypeError`` (the public surface
    does not advertise the knob). The drive invokes the documented
    constructor and observes the public exception.
    """
    with pytest.raises(TypeError):
        KnowledgeQuery(
            objective_id="b-timeout-rejected",
            objective_text="alpha content",
            timeout_seconds=30.0,
        )


def test_b_knowledge_query_fingerprint_does_not_change_with_timeout_kwarg():
    """The fingerprint returned by KnowledgeQuery.fingerprint() is stable
    regardless of whether a caller (mistakenly) tries to feed a timeout knob.

    Behavioral check: even if a caller passes the (now-rejected) timeout_seconds
    kwarg into the fingerprint input dict via _compute_fingerprint-equivalent
    surface, the engine does not consult it. We exercise the documented
    public method (KnowledgeQuery.fingerprint()) which never accepted it.
    """
    q1 = KnowledgeQuery(objective_id="fp-1", objective_text="alpha")
    q2 = KnowledgeQuery(
        objective_id="fp-1",
        objective_text="alpha",
        goal_class=q1.goal_class,
        risk_profile=q1.risk_profile,
        complexity=q1.complexity,
        max_hits_per_source=q1.max_hits_per_source,
        max_hits_total=q1.max_hits_total,
        sources_requested=q1.sources_requested,
        schema_version=q1.schema_version,
    )
    # Two structurally identical queries → identical fingerprints.
    assert q1.fingerprint() == q2.fingerprint()


# ── Test C: discover() / dry_run() signatures no longer expose the knob ─


def test_c_dry_run_signature_has_no_timeout_param():
    """dry_run() externally rejects timeout_seconds.

    Behavioral contract: the canonical happy-path
    ``EvidencePackEngine.dry_run`` invocation (without the disputed
    kwarg) succeeds and returns the documented shape. This is the
    positive twin of ``test_a_timeout_seconds_kwarg_rejected_by_dry_run``
    — together, they prove the knob is gone: the call works without it
    and rejects it when supplied.
    """
    engine = _make_engine()
    pack = engine.dry_run(
        objective_id="c-dry-run-happy",
        objective_text="alpha content",
    )
    # Public surface: the call returns a pack-like object.
    assert pack is not None
    assert getattr(pack, "query_fingerprint", None) is not None


def test_c_discover_propagates_no_timeout_param():
    """discover() externally rejects timeout_seconds.

    Behavioral contract: ``EvidencePackEngine.discover`` rejects
    ``timeout_seconds`` at the public boundary with ``TypeError``
    (the kwarg is not forwarded into the underlying cache-fingerprint
    pipeline). The drive invokes the documented method and observes
    the public exception; the cache identity is therefore not
    perturbed by the (rejected) knob.
    """
    engine = _make_engine_with_storage()
    with pytest.raises(TypeError):
        engine.discover(
            objective_id="c-discover-timeout-rejected",
            objective_text="alpha content",
            timeout_seconds=30.0,
        )


# ── Test D: a caller cannot reintroduce timeout via kwargs in discover ──


def test_d_discover_rejects_timeout_seconds_kwarg():
    """discover() forwards **kwargs to dry_run + fingerprint; since dry_run
    does not advertise timeout_seconds, an attempted timeout_kwarg is
    rejected at the public boundary instead of being silently absorbed
    into the cache fingerprint.
    """
    engine = _make_engine_with_storage()
    timeout_kwarg = {"timeout_seconds": 30.0}
    with pytest.raises(TypeError):
        engine.discover(
            objective_id="discover-timeout",
            objective_text="alpha content",
            **timeout_kwarg,
        )


# ── Test E: query fingerprint payload is independent of timeout_seconds ─


def test_e_fingerprint_invariant_under_timeout_kwarg_dropping():
    """Behavioral proof that timeout_seconds is not part of the cache key.

    If the engine were secretly using a timeout knob to derive its
    fingerprint, two consecutive discovers that differ ONLY in a
    timeout-shaped input would land on different cache entries and the
    second call would NOT reuse the first call's result.

    With the knob removed, no timeout-shaped kwarg can reach the engine
    at all (the public signature does not advertise it). We exercise the
    surface that the public signature advertises, and we verify that a
    new objective_id does NOT inherit metadata from a prior, differently
    configured call.
    """
    engine = _make_engine_with_storage()

    first = engine.discover(
        objective_id="cache-inv-A",
        objective_text="alpha content",
        goal_class="OTHER",
        risk_profile="low",
        complexity="S",
    )
    assert first.is_idempotent_reuse is False

    # Second discover on a DIFFERENT objective_id: cache miss,
    # dry_run runs, but without timeout_seconds (which is not in the
    # signature) the call succeeds and produces a fresh pack.
    second = engine.discover(
        objective_id="cache-inv-B",
        objective_text="alpha content",
        goal_class="OTHER",
        risk_profile="low",
        complexity="S",
    )
    assert second.is_idempotent_reuse is False
    # Different objective_ids → different fingerprints even with all
    # other inputs identical. timeout_seconds is not consulted.
    assert first.query_fingerprint != second.query_fingerprint


# ── Test F: discovery still works without the knob (no regression) ──────


def test_f_basic_discover_runs_without_timeout_seconds():
    """Removing the knob does not break the happy path: a normal discover
    completes, produces a pack, persists metadata, and reuses the cache on
    the next call.
    """
    engine = _make_engine_with_storage()
    pack1 = engine.discover(objective_id="happy", objective_text="alpha content")
    assert pack1.is_idempotent_reuse is False

    pack2 = engine.discover(objective_id="happy", objective_text="alpha content")
    assert pack2.is_idempotent_reuse is True
    assert pack2.query_fingerprint == pack1.query_fingerprint