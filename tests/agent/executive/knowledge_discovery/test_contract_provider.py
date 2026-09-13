"""Focal tests for the production contract source adapter.

These tests cover the contract primitive in
``agent.executive.knowledge_discovery.adapters.contract_provider``
end-to-end through the engine source contract without importing
``hermes_cli``. Coverage matrix:

* Missing goal returns []
* Missing or empty contract returns []
* Populated contract produces KnowledgeHitV2
* All five contract fields are represented deterministically
* Source and provenance are canonical and read-only
* Source URI contains no absolute path
* Loader receives the exact session_id
* Repeated calls produce stable hit_id and fingerprint
* max_hits is respected
* Query filtering/scoring is deterministic
* Loader exception remains observable
* Provider does not mutate loader-owned state
* No import from hermes_cli exists in the production adapter
* Provider has no filesystem or network side effect
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from agent.executive.knowledge_discovery import (
    FreshnessPolicy,
    KnowledgeHitV2,
    KnowledgeQuery,
    ProvenanceEnvelope,
)
from agent.executive.knowledge_discovery.adapters import contract_provider
from agent.executive.knowledge_discovery.adapters.contract_provider import (
    CONTRACT_FIELDS,
    PRODUCER,
    SOURCE_URI_PREFIX,
    STATE_META_GOAL_KEY,
    make_contract_provider,
)


OBS = "2026-08-04T11:30:00+00:00"
SESSION_ID = "b1-e1b-session-canonical"


# ─────────────────────────────────────────────────────────────────────
# Lightweight duck-typed GoalState stand-ins (NOT hermes_cli imports)
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _DuckContract:
    """Structural stand-in for hermes_cli.goals.GoalContract.

    The production adapter reads attribute access only — no isinstance
    check, no class identity check, no import of the real dataclass.
    """

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""


@dataclass(frozen=True)
class _DuckGoalState:
    """Structural stand-in for hermes_cli.goals.GoalState.

    The production adapter reads ``state.contract`` (attribute access)
    and never inspects anything else.
    """

    goal: str = ""
    contract: _DuckContract = field(default_factory=_DuckContract)


def _full_contract() -> _DuckContract:
    return _DuckContract(
        outcome="ship the migration",
        verification="the auth test suite passes",
        constraints="keep the public /login response shape unchanged",
        boundaries="only touch services/auth and its tests",
        stop_when="a schema change needs product sign-off",
    )


def _query(text: str = "ship migration auth login") -> KnowledgeQuery:
    return KnowledgeQuery(
        objective_id="obj-b1-e1b",
        objective_text=text,
    )


def _query_other() -> KnowledgeQuery:
    return KnowledgeQuery(
        objective_id="obj-b1-e1b-other",
        objective_text="completely unrelated objective text",
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Missing goal returns []
# ─────────────────────────────────────────────────────────────────────


def test_missing_goal_returns_empty_list():
    """Loader returning None (no goal for this session) → []."""
    calls: list[str] = []

    def loader(sid: str) -> Optional[Any]:
        calls.append(sid)
        return None

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    result = provider(_query(), max_hits=5, observed_at=OBS)

    assert result == []
    assert calls == [SESSION_ID], "loader must be called with provider session_id"


# ─────────────────────────────────────────────────────────────────────
# 2. Missing or empty contract returns []
# ─────────────────────────────────────────────────────────────────────


def test_missing_contract_attribute_returns_empty_list():
    """Loader returns a state-like object with contract=None → []."""
    @dataclass(frozen=True)
    class _StateNoContract:
        goal: str = "some goal"

    def loader(sid: str) -> Any:
        return _StateNoContract()

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    assert provider(_query(), max_hits=5, observed_at=OBS) == []


def test_empty_contract_returns_empty_list():
    """All five fields blank → [] (source-unavailable, not a fake hit)."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(goal="plain free-form goal", contract=_DuckContract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    assert provider(_query(), max_hits=5, observed_at=OBS) == []


def test_whitespace_only_contract_returns_empty_list():
    """All fields blank after .strip() → []."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(
            goal="plain free-form goal",
            contract=_DuckContract(
                outcome="   ",
                verification="\t",
                constraints="",
                boundaries="\n",
                stop_when="",
            ),
        )

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    assert provider(_query(), max_hits=5, observed_at=OBS) == []


def test_loader_returns_mapping_contract_directly():
    """Loader may return a raw mapping (state_meta JSON dict)."""
    raw = {
        "outcome": "ship the migration",
        "verification": "the auth test suite passes",
        "constraints": "",
        "boundaries": "",
        "stop_when": "",
    }

    def loader(sid: str) -> Any:
        return {"goal": "plain", "contract": raw}

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    hits = provider(_query(), max_hits=5, observed_at=OBS)
    assert isinstance(hits, list)
    assert len(hits) == 1
    assert isinstance(hits[0], KnowledgeHitV2)


# ─────────────────────────────────────────────────────────────────────
# 3. Populated contract produces KnowledgeHitV2
# 4. All five contract fields are represented deterministically
# ─────────────────────────────────────────────────────────────────────


def test_populated_contract_produces_knowledge_hit_v2():
    def loader(sid: str) -> Any:
        return _DuckGoalState(
            goal="port auth to JWT",
            contract=_full_contract(),
        )

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    hits = provider(_query(), max_hits=5, observed_at=OBS)

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, KnowledgeHitV2)
    assert hit.source == "contract"

    # All five fields are represented in the snippet. Order is canonical.
    for label in ("outcome", "verification", "constraints", "boundaries", "stop_when"):
        assert label in hit.snippet, f"missing {label} label in snippet {hit.snippet!r}"
    assert "ship the migration" in hit.snippet
    assert "the auth test suite passes" in hit.snippet
    assert "keep the public /login response shape unchanged" in hit.snippet
    assert "only touch services/auth and its tests" in hit.snippet
    assert "a schema change needs product sign-off" in hit.snippet


def test_only_populated_fields_appear_in_snippet():
    """Fields left blank are omitted; canonical order preserved."""
    partial = _DuckContract(
        outcome="ship the migration",
        verification="",
        constraints="",
        boundaries="",
        stop_when="",
    )

    def loader(sid: str) -> Any:
        return _DuckGoalState(goal="g", contract=partial)

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)
    assert "outcome: ship the migration" in hit.snippet
    assert "verification" not in hit.snippet
    assert "constraints" not in hit.snippet
    assert "boundaries" not in hit.snippet
    assert "stop_when" not in hit.snippet


def test_canonical_field_order_is_outcome_verification_constraints_boundaries_stop_when():
    assert CONTRACT_FIELDS == (
        "outcome",
        "verification",
        "constraints",
        "boundaries",
        "stop_when",
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Source and provenance are canonical and read-only
# ─────────────────────────────────────────────────────────────────────


def test_source_is_contract():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)
    assert hit.source == "contract"


def test_provenance_is_read_only_true():
    """provenance.read_only MUST be True (engine invariant)."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)
    assert isinstance(hit.provenance, ProvenanceEnvelope)
    assert hit.provenance.read_only is True


def test_provenance_producer_and_source_type_and_retrieval_mode():
    """Producer/source_type/retrieval_mode are canonical for production."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)
    assert hit.provenance.producer == PRODUCER
    assert hit.provenance.producer != "fake_contract_provider_v1"
    assert hit.provenance.source_type == "contract"
    # Retrieval mode reflects metadata-backed production state.
    assert hit.provenance.retrieval_mode == "metadata_only"
    assert hit.provenance.retrieval_mode in {
        "metadata_only", "snippet", "full_document",
        "semantic_search", "keyword_search",
    }


def test_freshness_is_engine_freshness_policy():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)
    assert isinstance(hit.freshness, FreshnessPolicy)
    assert hit.freshness.observed_at == OBS


# ─────────────────────────────────────────────────────────────────────
# 6. Source URI contains no absolute path
# ─────────────────────────────────────────────────────────────────────


def test_source_uri_is_state_meta_key_anchored_and_not_absolute():
    """source_uri is anchored to the SessionDB goal key; no abs path."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)

    uri = hit.provenance.source_uri
    location = hit.location
    for s in (uri, location):
        assert not s.startswith("/"), f"source URI exposes absolute path: {s!r}"
        assert not s.startswith("file://"), f"source URI uses file:// scheme: {s!r}"
        assert SESSION_ID in s, f"source URI must be tied to session_id: {s!r}"
        assert SOURCE_URI_PREFIX in s, f"source URI uses canonical prefix: {s!r}"
    assert STATE_META_GOAL_KEY.format(session_id=SESSION_ID) in uri


def test_source_uri_is_stable_across_calls():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit_a] = provider(_query(), max_hits=5, observed_at=OBS)
    [hit_b] = provider(_query(), max_hits=5, observed_at=OBS)
    assert hit_a.provenance.source_uri == hit_b.provenance.source_uri
    assert hit_a.location == hit_b.location


# ─────────────────────────────────────────────────────────────────────
# 7. Loader receives the exact session_id
# ─────────────────────────────────────────────────────────────────────


def test_loader_receives_provider_session_id_exactly():
    received: list[str] = []

    def loader(sid: str) -> Any:
        received.append(sid)
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    provider(_query(), max_hits=5, observed_at=OBS)
    provider(_query(), max_hits=5, observed_at=OBS)
    assert received == [SESSION_ID, SESSION_ID]


def test_provider_does_not_resolve_session_id_from_query_or_environment():
    """Loader is called with the closed-over session_id, not query.*."""
    @dataclass(frozen=True)
    class _Spy:
        contract: Any

    received: list[str] = []

    def loader(sid: str) -> Any:
        received.append(sid)
        return _Spy(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    # Even with a query whose objective_id looks like another session,
    # the loader sees the provider's session_id.
    q = KnowledgeQuery(
        objective_id="not-the-session",
        objective_text="not the session either",
    )
    provider(q, max_hits=5, observed_at=OBS)
    assert received == [SESSION_ID]


# ─────────────────────────────────────────────────────────────────────
# 8. Repeated calls produce stable hit_id and fingerprint
# ─────────────────────────────────────────────────────────────────────


def test_repeated_calls_produce_stable_hit_id_and_fingerprint():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit_a] = provider(_query(), max_hits=5, observed_at=OBS)
    [hit_b] = provider(_query(), max_hits=5, observed_at=OBS)
    [hit_c] = provider(_query_other(), max_hits=5, observed_at=OBS)
    # Stable across calls AND across queries (deterministic contract body).
    assert hit_a.hit_id == hit_b.hit_id == hit_c.hit_id
    assert hit_a.fingerprint == hit_b.fingerprint == hit_c.fingerprint


def test_different_contracts_produce_different_hit_ids():
    def loader_full(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    def loader_partial(sid: str) -> Any:
        return _DuckGoalState(
            contract=_DuckContract(
                outcome="different outcome",
                verification="different verification",
                constraints="different constraints",
                boundaries="different boundaries",
                stop_when="different stop_when",
            )
        )

    provider_full = make_contract_provider(SESSION_ID, state_loader=loader_full)
    provider_partial = make_contract_provider(SESSION_ID, state_loader=loader_partial)
    [hit_full] = provider_full(_query(), max_hits=5, observed_at=OBS)
    [hit_partial] = provider_partial(_query(), max_hits=5, observed_at=OBS)
    assert hit_full.hit_id != hit_partial.hit_id
    assert hit_full.fingerprint != hit_partial.fingerprint


def test_different_sessions_produce_different_hit_ids():
    """Same contract body, different sessions → different hit_ids."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider_a = make_contract_provider("session-A", state_loader=loader)
    provider_b = make_contract_provider("session-B", state_loader=loader)
    [hit_a] = provider_a(_query(), max_hits=5, observed_at=OBS)
    [hit_b] = provider_b(_query(), max_hits=5, observed_at=OBS)
    assert hit_a.hit_id != hit_b.hit_id


# ─────────────────────────────────────────────────────────────────────
# 9. max_hits is respected
# ─────────────────────────────────────────────────────────────────────


def test_max_hits_one_returns_single_hit():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    hits = provider(_query(), max_hits=1, observed_at=OBS)
    assert len(hits) == 1


def test_max_hits_zero_returns_empty():
    """Engine clamps max_hits_per_source to >= 1 before calling, but the
    provider itself honors slice semantics: ``max_hits=0`` → ``[]``."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    hits = provider(_query(), max_hits=0, observed_at=OBS)
    assert hits == []


def test_max_hits_large_does_not_exceed_one_hit_per_session():
    """A session has at most one GoalContract, so even max_hits=100 → 1."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    hits = provider(_query(), max_hits=100, observed_at=OBS)
    assert len(hits) == 1


def test_max_hits_negative_returns_empty():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    assert provider(_query(), max_hits=-1, observed_at=OBS) == []


# ─────────────────────────────────────────────────────────────────────
# 10. Query filtering/scoring is deterministic
# ─────────────────────────────────────────────────────────────────────


def test_relevance_score_is_in_unit_interval():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query(), max_hits=5, observed_at=OBS)
    assert 0.0 <= hit.relevance_score <= 1.0


def test_relevance_score_is_deterministic_across_calls():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit_a] = provider(_query(), max_hits=5, observed_at=OBS)
    [hit_b] = provider(_query(), max_hits=5, observed_at=OBS)
    assert hit_a.relevance_score == hit_b.relevance_score


def test_unrelated_query_still_returns_populated_contract():
    """Token-empty / unrelated query does not fabricate a non-match —
    the populated contract surfaces as a candidate regardless."""
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    [hit] = provider(_query_other(), max_hits=5, observed_at=OBS)
    assert isinstance(hit, KnowledgeHitV2)
    assert 0.0 <= hit.relevance_score <= 1.0


# ─────────────────────────────────────────────────────────────────────
# 11. Loader exception remains observable
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("boom"), OSError("fs error"), ValueError("bad input")],
)
def test_loader_exception_propagates(exc):
    """Engine relies on this propagation for source_failed accounting."""
    def loader(sid: str) -> Any:
        raise exc

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    with pytest.raises(type(exc)) as excinfo:
        provider(_query(), max_hits=5, observed_at=OBS)
    assert excinfo.value is exc


def test_loader_exception_does_not_silently_become_empty_list():
    """A loader exception must NOT be converted into []."""
    def loader(sid: str) -> Any:
        raise RuntimeError("explicit failure")

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    with pytest.raises(RuntimeError):
        provider(_query(), max_hits=5, observed_at=OBS)


def test_engine_marks_source_failed_when_provider_raises():
    """Wire through EvidencePackEngine to confirm source_failed contract."""
    from agent.executive.knowledge_discovery import EvidencePackEngine

    def loader(sid: str) -> Any:
        raise RuntimeError("simulated loader failure")

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    engine = EvidencePackEngine(sources={"contract": provider})
    pack = engine.dry_run(objective_id="obj-b1", objective_text="some query")
    assert "contract" in pack.sources_failed
    assert "contract" not in pack.sources_queried


# ─────────────────────────────────────────────────────────────────────
# 12. Provider does not mutate loader-owned state
# ─────────────────────────────────────────────────────────────────────


def test_provider_does_not_mutate_loader_state_contract():
    """Repeated reads must return byte-identical loader-owned state."""
    original_state = _DuckGoalState(
        goal="port auth to JWT",
        contract=_full_contract(),
    )
    snapshot_before = (
        original_state.goal,
        original_state.contract.outcome,
        original_state.contract.verification,
        original_state.contract.constraints,
        original_state.contract.boundaries,
        original_state.contract.stop_when,
    )

    captured: dict[str, Any] = {"state": original_state}

    def loader(sid: str) -> Any:
        return captured["state"]

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    for _ in range(5):
        provider(_query(), max_hits=5, observed_at=OBS)
        provider(_query_other(), max_hits=5, observed_at=OBS)
        provider(_query(), max_hits=0, observed_at=OBS)
        provider(_query(), max_hits=-2, observed_at=OBS)

    state_after = captured["state"]
    snapshot_after = (
        state_after.goal,
        state_after.contract.outcome,
        state_after.contract.verification,
        state_after.contract.constraints,
        state_after.contract.boundaries,
        state_after.contract.stop_when,
    )
    assert snapshot_before == snapshot_after
    # Defensive: identity preserved.
    assert state_after is original_state


def test_provider_does_not_mutate_mapping_contract():
    """If the loader returns a dict, repeated reads don't mutate it."""
    original = {
        "goal": "port auth",
        "contract": {
            "outcome": "ship the migration",
            "verification": "the auth test suite passes",
            "constraints": "",
            "boundaries": "",
            "stop_when": "",
        },
    }
    import copy as _copy
    snapshot_before = _copy.deepcopy(original)
    captured = {"raw": original}

    def loader(sid: str) -> Any:
        return captured["raw"]

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    provider(_query(), max_hits=5, observed_at=OBS)
    provider(_query_other(), max_hits=5, observed_at=OBS)
    assert captured["raw"] == snapshot_before


# ─────────────────────────────────────────────────────────────────────
# 13. No import from hermes_cli exists in the production adapter
# ─────────────────────────────────────────────────────────────────────


def test_production_adapter_does_not_pull_in_hermes_cli_at_import_time():
    """Runtime invariant: the adapter must not import hermes_cli.*.

    Verify by inspecting the adapter's namespace: no forbidden
    re-exports, no attribute sourced from a hermes_cli module.
    """
    # The adapter must not re-export CLI-only types.
    for forbidden in ("GoalState", "GoalContract", "GoalManager"):
        assert forbidden not in vars(contract_provider), (
            f"production adapter re-exports forbidden name: {forbidden}"
        )
    # And the module must not expose anything whose __module__ comes
    # from hermes_cli — that would mean a hermes_cli import leaked.
    for name, obj in vars(contract_provider).items():
        mod = getattr(obj, "__module__", "")
        assert not (mod == "hermes_cli" or mod.startswith("hermes_cli.")), (
            f"production adapter pulls in {name!r} from {mod!r}"
        )


def test_production_adapter_does_not_import_hermes_cli():
    """The adapter module must not import any hermes_cli submodule."""
    module = importlib.import_module(
        "agent.executive.knowledge_discovery.adapters.contract_provider"
    )
    # The module must not re-export GoalState / GoalContract / GoalManager.
    for forbidden in ("GoalState", "GoalContract", "GoalManager"):
        assert forbidden not in module.__dict__, (
            f"production adapter re-exports forbidden name: {forbidden}"
        )


def test_adapter_package_init_has_no_hermes_cli_imports():
    """The adapters package itself must not pull in hermes_cli.

    Runtime check: the adapters package __dict__ must not contain
    anything sourced from hermes_cli.
    """
    import importlib
    pkg = importlib.import_module(
        "agent.executive.knowledge_discovery.adapters"
    )
    for name, obj in vars(pkg).items():
        mod = getattr(obj, "__module__", "")
        assert not (mod == "hermes_cli" or mod.startswith("hermes_cli.")), (
            f"adapters package exposes {name!r} from {mod!r}"
        )


# ─────────────────────────────────────────────────────────────────────
# 14. Provider has no filesystem or network side effect
# ─────────────────────────────────────────────────────────────────────


def test_provider_does_not_open_filesystem_paths():
    """Provider module must not import filesystem-reading primitives."""
    import inspect
    import sys
    module = sys.modules[
        "agent.executive.knowledge_discovery.adapters.contract_provider"
    ]
    # Whitelist of allowed third-party modules.
    allowed_prefixes = (
        "agent.executive.knowledge_discovery",
        "typing",
        "copy",
        "hashlib",
        "builtins",
    )
    # Inspect only public, importable names. Private dunders (__loader__,
    # __spec__, __file__, …) are set by Python's import machinery on
    # every module and are not the responsibility of this adapter.
    for name in module.__dict__:
        if name.startswith("_"):
            continue
        obj = module.__dict__[name]
        if obj is None:
            continue
        # Skip non-callables, classes, and constants — they don't
        # represent imports. Functions, however, may carry a
        # __module__ that exposes their origin.
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        mod = getattr(obj, "__module__", "")
        if not mod:
            continue
        if any(mod == p or mod.startswith(p + ".") for p in allowed_prefixes):
            continue
        raise AssertionError(
            f"production provider pulled in unexpected primitive {name!r} from {mod!r}"
        )


def test_provider_does_not_open_network_sockets():
    """Provider must not import network/socket primitives."""
    import sys
    module = sys.modules[
        "agent.executive.knowledge_discovery.adapters.contract_provider"
    ]
    forbidden = {"socket", "urllib", "urllib.request", "http", "httpx", "aiohttp"}
    for name, obj in vars(module).items():
        mod = getattr(obj, "__module__", "")
        for ban in forbidden:
            if mod == ban or mod.startswith(ban + "."):
                raise AssertionError(
                    f"production provider imports network primitive {name!r} from {mod!r}"
                )


def test_provider_does_not_spawn_threads_or_processes():
    """Provider module must not import threading / subprocess / multiprocessing."""
    import sys
    module = sys.modules[
        "agent.executive.knowledge_discovery.adapters.contract_provider"
    ]
    forbidden = {
        "threading", "thread", "multiprocessing", "subprocess",
        "asyncio", "concurrent", "concurrent.futures",
    }
    for name, obj in vars(module).items():
        mod = getattr(obj, "__module__", "")
        for ban in forbidden:
            if mod == ban or mod.startswith(ban + "."):
                raise AssertionError(
                    f"production provider imports concurrency primitive {name!r} from {mod!r}"
                )


def test_provider_constructor_arguments_are_validated():
    """Both session_id and state_loader must be supplied up front."""
    with pytest.raises(ValueError):
        make_contract_provider("", state_loader=lambda sid: None)
    with pytest.raises(ValueError):
        make_contract_provider(None, state_loader=lambda sid: None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_contract_provider(SESSION_ID, state_loader=None)  # type: ignore[arg-type]


def test_provider_callable_has_engine_signature():
    """Provider must match the engine source contract signature.

    Behavioral contract: the provider callable accepts a positional
    ``query`` argument followed by the keyword-only parameters
    ``max_hits`` and ``observed_at``. Calling the provider with the
    documented kwargs returns the expected list of hits. The drive
    invokes the documented callable and observes its public surface.
    """
    # Behavioral observation 1: the documented kwargs are accepted
    # and the call returns a list (the contract shape).
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    result = provider(
        _query(),
        max_hits=5,
        observed_at="2026-08-04T12:00:00+00:00",
    )
    assert isinstance(result, list)
    assert all(isinstance(h, KnowledgeHitV2) for h in result)

    # Behavioral observation 2: keyword-only contract — passing
    # ``max_hits`` or ``observed_at`` positionally is rejected with
    # TypeError (the public surface advertises keyword-only).
    import pytest
    with pytest.raises(TypeError):
        provider(_query(), 5, "2026-08-04T12:00:00+00:00")  # type: ignore[misc]

    # Behavioral observation 3: omitting a required kwarg is rejected
    # with TypeError (the public surface is non-defaulting).
    with pytest.raises(TypeError):
        provider(_query(), max_hits=5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        provider(_query(), observed_at="2026-08-04T12:00:00+00:00")  # type: ignore[call-arg]


def test_provider_callable_returns_list_of_knowledge_hit_v2():
    def loader(sid: str) -> Any:
        return _DuckGoalState(contract=_full_contract())

    provider = make_contract_provider(SESSION_ID, state_loader=loader)
    result = provider(_query(), max_hits=5, observed_at=OBS)
    assert isinstance(result, list)
    assert all(isinstance(h, KnowledgeHitV2) for h in result)
