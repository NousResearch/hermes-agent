"""Pilot G1 — run_pipeline under B1-on, hermetic, evidence_pack=True.

Validates that ObjectiveEngine.run_pipeline completes the chain
submit -> normalize -> classify -> discover -> [discover_evidence_pack]
-> generate_contract without requiring network, subprocess, or LLM,
when constructed with in-memory storage and B1 enabled (env flag +
injected evidence_engine).
"""

from __future__ import annotations

import pytest


def test_g1_run_pipeline_returns_objective_id(objective_engine_in_memory):
    engine = objective_engine_in_memory
    oid = engine.run_pipeline(
        "pilot canary: hermetic evidence-pack dryrun",
        constraints=("forbidden:network", "limit:subprocess"),
        evidence_pack=True,
    )
    assert isinstance(oid, str) and oid
    state = engine.get_state(oid)
    assert state.state.value == "CONTRACT_DRAFT", (
        f"expected CONTRACT_DRAFT, got {state.state.value}"
    )


def test_g1_run_pipeline_emits_contract_dict_with_evidence_key(
    objective_engine_in_memory,
):
    engine = objective_engine_in_memory
    oid = engine.run_pipeline(
        "pilot canary: evidence fields",
        evidence_pack=True,
    )
    state = engine.get_state(oid)
    assert state.contract is not None, "contract must be generated"
    # When B1 is on, the contract dict carries evidence_pack_ref /
    # evidence_pack_summary keys (default-off contracts still carry
    # them with empty/default values, but the keys are present).
    assert "evidence_pack_ref" in state.contract, (
        f"contract missing evidence_pack_ref; keys={list(state.contract.keys())}"
    )


def test_g1_run_pipeline_does_not_persist_by_default(
    objective_engine_in_memory,
):
    engine = objective_engine_in_memory
    oid = engine.run_pipeline(
        "pilot canary: no persist",
        evidence_pack=True,
        persist_to_state_meta=False,
    )
    assert list(engine.list_persisted()) == [], (
        f"pilot must not persist by default; "
        f"got persisted={list(engine.list_persisted())!r}"
    )
    # OID is still tracked in-memory for the duration of the engine.
    assert oid in engine.list_active()


def test_g1_run_pipeline_preserves_forbidden_constraint(objective_engine_in_memory):
    """The 'forbidden:network' constraint must reach the normalized dict
    (so downstream gates can enforce it)."""
    engine = objective_engine_in_memory
    oid = engine.run_pipeline(
        "pilot canary: no network",
        constraints=("forbidden:network",),
        evidence_pack=True,
    )
    state = engine.get_state(oid)
    assert state.normalized is not None
    norm_constraints = state.normalized.get("constraints", ())
    assert "forbidden:network" in norm_constraints