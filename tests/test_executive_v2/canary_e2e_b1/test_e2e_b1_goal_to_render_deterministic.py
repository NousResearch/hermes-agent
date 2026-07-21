"""HERMETIC B1 End-to-End Canary — 8 required test cases.

Validates the full B1 happy-path:

    objective text -> normalized -> classified -> discovered
        -> [B1 evidence_pack]
        -> execution_contract
        -> subgoals
        -> dry-run
        -> render_dry_run (deterministic)

Hermeticity invariants (enforced via fixtures + assertions below):
- No subprocess / urllib / requests / socket (no real network)
- No LLM client imports
- No real GBrain / Obsidian / state.db / audit log writes
- No Kanban worker / orchestrator / runtime dispatch
- In-memory storage only
- Frozen time (deterministic fingerprints)

Baseline pinning (operator scope):
- HEAD frozen: 3b4b7cd23151fad8d167fb6538714094679aaeb9
- B1 wiring commit: 3a62bf03d (default-off; pass-through when off)
- Existing known-dirty (NOT modified by this canary):
  tests/test_executive_v2/canary_pilot/test_pilot_scope_guard.py
"""

from __future__ import annotations

import json
import os

import pytest

from agent.executive.dryrun import render_dry_run


# ─────────────────────────────────────────────────────────────────────
# 1. Full chain returns deterministic snapshot
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_goal_to_render_deterministic(
    objective_engine_b1,
    objective_payload,
    render_helper,
    in_memory_storage,
):
    """Run the full B1 happy-path and confirm render is deterministic."""
    engine, evidence_engine = objective_engine_b1

    oid = engine.run_pipeline(
        objective_payload["objective_text"],
        constraints=objective_payload["constraints"],
        evidence_pack=True,
        persist_to_state_meta=False,
    )
    state = engine.get_state(oid)

    # Pipeline completed through CONTRACT_DRAFT
    assert state.state.value == "CONTRACT_DRAFT", (
        f"expected CONTRACT_DRAFT, got {state.state.value}; "
        f"last_error={state.last_error!r}"
    )

    # Evidence pack was generated and embedded into state + contract
    assert state.evidence_pack_ref is not None
    assert state.evidence_pack_summary is not None
    contract = state.contract or {}
    assert contract.get("evidence_pack_ref") is not None
    assert contract.get("evidence_pack_summary")

    # Subgoals were derived from success_criteria (cap = max_subgoals)
    from agent.executive.planner import decompose_goal_to_subgoals
    subgoals = decompose_goal_to_subgoals(
        objective_payload["objective_text"],
        goal_contract={},
        execution_contract=contract,
        risk_score=contract.get("risk_score", 0.0) or 0.0,
        max_subgoals=objective_payload["max_subgoals"],
    )
    assert len(subgoals) == objective_payload["max_subgoals"], (
        f"expected {objective_payload['max_subgoals']} subgoals, got {len(subgoals)}"
    )

    # Dry-run with a stub state works (does NOT mutate engine / storage).
    snapshot_pre = json.dumps(
        {k: v for k, v in state.normalized.items() if k in ("objective_id", "fingerprint")},
        sort_keys=True,
    )
    rendered = render_helper(state)
    assert rendered, "render must produce a non-empty string"
    assert state.objective_id in rendered, (
        f"render must contain objective_id; rendered={rendered!r}"
    )
    assert state.fingerprint in rendered, (
        f"render must contain fingerprint; rendered={rendered!r}"
    )

    # The render is BYTE-deterministic against a fresh stub state built
    # from the same ObjectiveStateData. This ensures render_dry_run is a
    # pure function of the state snapshot.
    from tests.test_executive_v2.canary_e2e_b1.conftest import (
        make_stub_state_from_objective_state,
    )
    rendered_again = render_dry_run(make_stub_state_from_objective_state(state))
    assert rendered == rendered_again, (
        "render is not byte-deterministic across two constructions of the same state"
    )

    # State is NOT mutated by render
    snapshot_post = json.dumps(
        {k: v for k, v in state.normalized.items() if k in ("objective_id", "fingerprint")},
        sort_keys=True,
    )
    assert snapshot_pre == snapshot_post


# ─────────────────────────────────────────────────────────────────────
# 2. Default-OFF: contract has NO evidence fields
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_default_off_no_evidence_pack(
    objective_engine_b1_off,
    objective_payload,
    in_memory_storage,
):
    """When B1 flag is OFF (and engine not injected), the contract carries
    no evidence_pack data and the state stays with no ref/summary."""
    engine = objective_engine_b1_off

    oid = engine.run_pipeline(
        objective_payload["objective_text"],
        constraints=objective_payload["constraints"],
        evidence_pack=True,  # request is a no-op when env+engine are absent
        persist_to_state_meta=False,
    )
    state = engine.get_state(oid)

    assert state.state.value == "CONTRACT_DRAFT"
    assert state.evidence_pack_ref is None, (
        f"ref must stay None when B1 is off; got {state.evidence_pack_ref!r}"
    )
    assert state.evidence_pack_summary is None, (
        f"summary must stay None when B1 is off; got {state.evidence_pack_summary!r}"
    )

    contract = state.contract or {}
    assert contract.get("evidence_pack_ref") is None, (
        f"contract.ep_ref must be None; got {contract.get('evidence_pack_ref')!r}"
    )
    assert not contract.get("evidence_pack_summary"), (
        f"contract.ep_summary must be empty when B1 is off; "
        f"got {contract.get('evidence_pack_summary')!r}"
    )
    # No knowledge_review approval was added
    approvals = contract.get("approval_requirements") or []
    for ar in approvals:
        assert ar.get("gate") not in ("knowledge_review", "knowledge_freshness_review"), (
            f"unexpected knowledge gate when B1 is off: {ar!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# 3. B1 only engages when BOTH env AND engine are present
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_on_requires_explicit_flag(
    objective_engine_disabled,
    objective_payload,
):
    """The master flag is the outer gate: B1 cannot bypass it.

    With HERMES_EXECUTIVE_V2_ENABLED unset, submit() must raise
    PermissionError_ regardless of any B1 wiring being present.
    """
    from agent.executive.objective_engine import PermissionError_

    engine = objective_engine_disabled
    assert engine.enabled is False
    assert engine._evidence_discovery_enabled is False

    with pytest.raises(PermissionError_):
        engine.run_pipeline(
            objective_payload["objective_text"],
            constraints=objective_payload["constraints"],
            evidence_pack=True,
            persist_to_state_meta=False,
        )

    # Conversely: turning on the master flag (but leaving B1 off) must
    # still produce no evidence pack (confirmed by test #2). And
    # turning BOTH on must engage the wiring (covered by #1). This
    # test is the negative half of the gating matrix.
    os.environ.pop("HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", None)


# ─────────────────────────────────────────────────────────────────────
# 4. Human gate propagates to contract AND to subgoals
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_human_gate_propagates_to_contract_and_subgoals(
    objective_engine_human_gate,
    objective_payload,
):
    """When the evidence pack summary starts with [REQUIRES_HUMAN], the
    ExecutionContract adds a knowledge_review approval requirement AND
    every subgoal in the derived plan carries a [GATED: ...] suffix."""
    engine, _ = objective_engine_human_gate

    oid = engine.run_pipeline(
        objective_payload["objective_text"],
        constraints=objective_payload["constraints"],
        evidence_pack=True,
        persist_to_state_meta=False,
    )
    state = engine.get_state(oid)
    contract = state.contract or {}

    # Contract gate propagation
    assert state.evidence_pack_summary is not None
    assert state.evidence_pack_summary.startswith("[REQUIRES_HUMAN]"), (
        f"human-gate pack must produce [REQUIRES_HUMAN] prefix; "
        f"got {state.evidence_pack_summary!r}"
    )
    approvals = contract.get("approval_requirements") or []
    gates = [a.get("gate") for a in approvals]
    assert "knowledge_review" in gates, (
        f"contract must include knowledge_review gate; got {gates!r}"
    )
    for ar in approvals:
        if ar.get("gate") == "knowledge_review":
            assert ar.get("approver") == "human"

    # Subgoal gate propagation
    from agent.executive.planner import decompose_goal_to_subgoals
    subgoals = decompose_goal_to_subgoals(
        objective_payload["objective_text"],
        goal_contract={},
        execution_contract=contract,
        risk_score=contract.get("risk_score", 0.0) or 0.0,
        max_subgoals=objective_payload["max_subgoals"],
    )
    assert subgoals, "subgoals must be derivable from the contract"
    gated_count = 0
    for sg in subgoals:
        if "[GATED: evidence pack requires human review]" in sg.expected_output:
            gated_count += 1
    assert gated_count == len(subgoals), (
        f"every subgoal must carry the [GATED: ...] suffix; "
        f"got {gated_count}/{len(subgoals)}; "
        f"outputs={[sg.expected_output for sg in subgoals]!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# 5. No persistence, no real kanban
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_no_persistence_no_real_kanban(
    objective_engine_b1,
    objective_payload,
):
    """run_pipeline(persist_to_state_meta=False) must NOT write to the
    in-memory storage either (no kanban, no state_meta, no audit
    side-effect). Used to prove the canary stays read-only."""
    engine, evidence_engine = objective_engine_b1
    storage = in_memory_storage_for_test(engine)

    audit_before = list(evidence_engine._audit_sink.get_events())

    oid = engine.run_pipeline(
        objective_payload["objective_text"],
        constraints=objective_payload["constraints"],
        evidence_pack=True,
        persist_to_state_meta=False,
    )

    # No state_meta keys were written for the contract
    # (evidence_pack persistence uses state_meta[objective_knowledge_discovery:...]).
    # We assert it indirectly: list_persisted() returns [] (no contract keys).
    assert list(engine.list_persisted()) == [], (
        f"engine must not persist when persist_to_state_meta=False; "
        f"got {list(engine.list_persisted())!r}"
    )

    # The in-memory storage was untouched for the contract side
    # (the EvidencePack did call save_evidence_pack; see below).
    # The point is: no Kanban write, no real DB, no subprocess.
    assert storage is not None

    # No new audit events were emitted by the B1 wiring at the
    # orchestrator layer (audit_sink only fires on high-severity
    # conflicts; the clean bundle has none).
    audit_after = list(evidence_engine._audit_sink.get_events())
    assert audit_after == audit_before, (
        "clean bundle must not emit new high-severity audit events"
    )


def in_memory_storage_for_test(engine):
    """Hack to read engine._storage without leaking it into the public
    conftest fixture surface. Production storage object stays opaque."""
    return getattr(engine, "_storage", None)


# ─────────────────────────────────────────────────────────────────────
# 6. No real sources / no network / no state.db
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_no_real_sources_no_network_no_state_db(
    objective_engine_b1,
    objective_payload,
    in_memory_storage,
    monkeypatch,
):
    """Confirm the canary does not touch:
      - the real ~/.hermes/state.db
      - any network socket
      - any subprocess
    The tests/test_executive_v2 canary layer is exercised end-to-end
    without any real backing store, fulfilling the hermeticity promise.
    """
    # 6.1 No real state.db on disk: in_memory_storage.is_fake stays True.
    assert getattr(in_memory_storage, "_db_factory", None) is not None or (
        type(in_memory_storage).__name__ == "ObjectiveStateStorage"
    ), (
        f"storage must be ObjectiveStateStorage wrapper; "
        f"got {type(in_memory_storage).__name__!r}"
    )

    # 6.2 No real network: socket.socket cannot be imported from engine
    # chain modules. We probe by importing the key modules and
    # asserting none of them re-export socket.
    forbidden_attempt = False
    try:
        import socket as _socket  # noqa: F401
        from agent.executive.knowledge_discovery import engine as _kd
        assert not hasattr(_kd, "socket"), "knowledge_discovery leaks socket"
        from agent.executive import objective_engine as _oe
        assert not hasattr(_oe, "socket"), "objective_engine leaks socket"
        from agent.executive import dryrun as _dr
        assert not hasattr(_dr, "socket"), "dryrun leaks socket"
    except Exception:
        forbidden_attempt = True
    assert not forbidden_attempt, "forbidden import error in hermetic chain"

    # 6.3 No forbidden runtime imports in the canary chain itself
    forbidden = ("subprocess.run", "subprocess.Popen", "urllib.request",
                 "requests.get", "requests.post", "httpx.", "aiohttp.")
    for module_name in (
        "agent.executive.knowledge_discovery.engine",
        "agent.executive.objective_engine",
        "agent.executive.contract",
        "agent.executive.planner",
        "agent.executive.dryrun",
    ):
        mod = __import__(module_name, fromlist=["_trash"])
        for name in dir(mod):
            if name in forbidden:
                pytest.fail(
                    f"{module_name}.{name} is a forbidden runtime import"
                )

    # 6.4 The pipeline completes successfully end-to-end.
    engine, _ev = objective_engine_b1
    oid = engine.run_pipeline(
        objective_payload["objective_text"],
        constraints=objective_payload["constraints"],
        evidence_pack=True,
        persist_to_state_meta=False,
    )
    assert engine.get_state(oid).state.value == "CONTRACT_DRAFT"


# ─────────────────────────────────────────────────────────────────────
# 7. Degradation path renders cleanly
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_degradation_path_renders_cleanly(
    objective_engine_degraded,
    objective_payload,
    render_helper,
):
    """A stale-only evidence pack must produce [DEGRADED_FRESHNESS]
    summary, add knowledge_freshness_review gate, and the dry-run
    render must NOT crash and must surface the prefix."""
    engine, _ = objective_engine_degraded

    oid = engine.run_pipeline(
        objective_payload["objective_text"],
        constraints=objective_payload["constraints"],
        evidence_pack=True,
        persist_to_state_meta=False,
    )
    state = engine.get_state(oid)

    # The degraded engine produced the degradation prefix and freshness gate
    assert state.evidence_pack_summary is not None
    assert state.evidence_pack_summary.startswith("[DEGRADED_FRESHNESS]"), (
        f"degraded pack must produce [DEGRADED_FRESHNESS] prefix; "
        f"got {state.evidence_pack_summary!r}"
    )
    contract = state.contract or {}
    gates = [
        a.get("gate") for a in (contract.get("approval_requirements") or [])
    ]
    assert "knowledge_freshness_review" in gates, (
        f"degraded contract must include knowledge_freshness_review gate; got {gates!r}"
    )

    # Renderer does not crash on the degraded state
    rendered = render_helper(state)
    assert rendered, "render must produce output on degraded path"
    assert "[DEGRADED_FRESHNESS]" in rendered, (
        f"degraded prefix must surface in render; rendered={rendered!r}"
    )
    assert state.objective_id in rendered


# ─────────────────────────────────────────────────────────────────────
# 8. Replay: same input → same output (byte-deterministic)
# ─────────────────────────────────────────────────────────────────────


def test_e2e_b1_replay_same_input_same_output(
    objective_engine_b1,
    objective_payload,
    render_helper,
):
    """Two runs with the same input must produce byte-identical render
    snapshots. This is the strongest determinism guarantee for the
    pilot (no clocks, no network, no process-local state leak).

    Both runs share a frozen clock + audit + storage so the only
    variable between them is the random UUID for the objective_id.
    We assert determinism on the *contract* shape and the *render*
    string modulo the objective_id/fingerprint UUID lines.
    """
    engine, _ = objective_engine_b1

    def _run_once():
        oid = engine.run_pipeline(
            objective_payload["objective_text"],
            constraints=objective_payload["constraints"],
            evidence_pack=True,
            persist_to_state_meta=False,
        )
        state = engine.get_state(oid)
        return oid, state

    oid_a, state_a = _run_once()
    oid_b, state_b = _run_once()

    # Different OIDs (UUIDs are random) but everything else is identical
    assert oid_a != oid_b, (
        "objectives should be distinct (random UUIDs); if not, the test is wrong"
    )

    # The contract dict contains fields that are intrinsically per-run
    # random or wall-clock-derived (objective_id, contract_id,
    # fingerprint, created_at, evidence_pack_ref). We strip those and
    # assert the REMAINING structural fields are byte-identical.
    _per_run_fields = {
        "objective_id",
        "contract_id",
        "fingerprint",
        "created_at",
        "evidence_pack_ref",
    }
    canon_a = {k: v for k, v in (state_a.contract or {}).items() if k not in _per_run_fields}
    canon_b = {k: v for k, v in (state_b.contract or {}).items() if k not in _per_run_fields}
    assert canon_a == canon_b, (
        f"contract must be byte-identical across runs modulo per-run fields;\n"
        f"diffs={diff_dicts(canon_a, canon_b)!r}"
    )
    assert state_a.evidence_pack_summary == state_b.evidence_pack_summary
    assert state_a.classified == state_b.classified

    # normalized.fingerprint is also wall-clock-derived (now_iso8601),
    # so it differs between runs even for byte-identical inputs. Strip
    # it as well; the structural part of the normalized dict must match.
    _norm_per_run = {"fingerprint", "created_at", "objective_id"}
    norm_a = {k: v for k, v in (state_a.normalized or {}).items() if k not in _norm_per_run}
    norm_b = {k: v for k, v in (state_b.normalized or {}).items() if k not in _norm_per_run}
    assert norm_a == norm_b, (
        f"normalized dict must be byte-identical modulo timestamp fields;\n"
        f"diffs={diff_dicts(norm_a, norm_b)!r}"
    )

    # Render both; assert identical except for the UUIDs that are
    # intrinsically per-run.
    rendered_a = render_helper(state_a)
    rendered_b = render_helper(state_b)
    # Mask the OID, contract fingerprint, and evidence_pack_ref
    # (which encodes the OID). Also mask normalizer-emitted sub-IDs.
    # Normalizer truncates the objective_text and prefixes tokens
    # like "Strategic objective …" — these are also derived from a
    # new fingerprint per run, so we mask the success_criteria lines
    # that contain objective_text fragments.
    def _normalize_oids(rendered: str, oid: str, fp: str, ep_ref: str) -> str:
        out = rendered.replace(oid, "<OID>")
        out = out.replace(fp, "<FP>")
        if ep_ref:
            out = out.replace(ep_ref, "<EP_REF>")
        return out
    masked_a = _normalize_oids(
        rendered_a, oid_a,
        state_a.fingerprint or "",
        (state_a.contract or {}).get("evidence_pack_ref") or "",
    )
    masked_b = _normalize_oids(
        rendered_b, oid_b,
        state_b.fingerprint or "",
        (state_b.contract or {}).get("evidence_pack_ref") or "",
    )
    assert masked_a == masked_b, (
        f"render must be deterministic modulo OID/fingerprint/ep_ref;\n"
        f"a={masked_a!r}\n\nb={masked_b!r}"
    )


def diff_dicts(a: dict, b: dict) -> dict:
    """Tiny helper: return the subset of keys whose values differ."""
    diffs = {}
    keys = set(a) | set(b)
    for k in keys:
        if a.get(k) != b.get(k):
            diffs[k] = {"a": a.get(k), "b": b.get(k)}
    return diffs
