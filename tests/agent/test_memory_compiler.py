from copy import deepcopy

from agent.memory_bitemporal_fact_graph import BITEMPORAL_FACT_GRAPH_POLICY
from agent.memory_compiler import (
    MEMORY_COMPILER_POLICY,
    compile_memory_patterns,
    compile_methodology_candidate,
    compile_procedure_block_candidate,
    explain_compilation_trace,
    recommend_compilation_action,
)


def _fact(**overrides):
    base = {
        "fact_id": "fact-1",
        "subject": "Hermes Memory Fabric",
        "predicate": "procedure",
        "object": "compile review-only procedure candidates",
        "project_id": "memory-fabric",
        "source_episode_id": "episode-1",
        "provenance": {"source": "bench"},
        "confidence": 0.9,
        "valid_from": "2026-05-23T00:00:00Z",
        "system_created_at": "2026-05-23T08:00:00Z",
        "governance": BITEMPORAL_FACT_GRAPH_POLICY,
    }
    base.update(overrides)
    return base


def test_stable_facts_compile_into_methodology_candidate():
    facts = [
        _fact(fact_id="stable-1", source_episode_id="episode-1"),
        _fact(fact_id="stable-2", source_episode_id="episode-2"),
    ]

    result = compile_memory_patterns(facts, project_scope="memory-fabric")

    assert result["project_scope"] == "memory-fabric"
    assert result["input_fact_count"] == 2
    assert result["current_fact_count"] == 1
    assert [pattern["pattern_type"] for pattern in result["patterns"]] == ["stable_repeated_claim"]
    assert result["methodology_candidate"]["status"] == "review_required"
    assert "has not been applied" in result["methodology_candidate"]["summary"]


def test_superseded_facts_compile_into_evolution_pattern():
    old = _fact(fact_id="old", object="use contradiction review only")
    new = _fact(
        fact_id="new",
        object="compile review-only procedure candidates",
        valid_from="2026-05-24T00:00:00Z",
        system_created_at="2026-05-24T08:00:00Z",
        supersedes=["old"],
    )

    result = compile_memory_patterns([old, new], project_scope="memory-fabric")

    evolution = [pattern for pattern in result["patterns"] if pattern["pattern_type"] == "superseded_lineage_evolution"]
    assert len(evolution) == 1
    assert evolution[0]["current_fact_id"] == "new"
    assert evolution[0]["superseded_fact_ids"] == ["old"]


def test_contradictions_compile_into_review_required_pattern():
    facts = [
        _fact(fact_id="allowed", object="allow external memory exposure"),
        _fact(fact_id="blocked", object="block external memory exposure"),
    ]

    result = compile_memory_patterns(facts, project_scope="memory-fabric")

    contradictions = [pattern for pattern in result["patterns"] if pattern["pattern_type"] == "contradiction_review_required"]
    assert len(contradictions) == 1
    assert contradictions[0]["status"] == "review_required"
    assert result["review_recommendation"]["action"] == "review_contradictions_before_methodology"


def test_project_scope_isolation():
    facts = [
        _fact(fact_id="memory-1", project_id="memory-fabric"),
        _fact(fact_id="memory-2", project_id="memory-fabric"),
        _fact(fact_id="openclaw-1", project_id="openclaw", object="route external recall"),
        _fact(fact_id="openclaw-2", project_id="openclaw", object="route external recall"),
    ]

    result = compile_memory_patterns(facts, project_scope="openclaw")

    assert result["input_fact_count"] == 4
    assert all(pattern["project_id"] == "openclaw" for pattern in result["patterns"])
    assert result["patterns"][0]["fact_ids"] == ["openclaw-1", "openclaw-2"]


def test_compiler_does_not_mutate_inputs():
    facts = [_fact(fact_id="stable-1"), _fact(fact_id="stable-2")]
    before = deepcopy(facts)

    compile_memory_patterns(facts, project_scope="memory-fabric")

    assert facts == before


def test_output_policy_proves_no_memory_config_graph_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = compile_memory_patterns([_fact(fact_id="stable-1"), _fact(fact_id="stable-2")])

    assert result["policy"] == MEMORY_COMPILER_POLICY
    assert result["policy"]["read_only"] is True
    assert result["policy"]["would_write_memory"] is False
    assert result["policy"]["would_modify_config"] is False
    assert result["policy"]["would_write_graph"] is False
    assert result["policy"]["does_not_create_operation_events"] is True
    assert result["policy"]["creates_review_candidates_only"] is True
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_procedure_block_candidate_has_review_required_status():
    methodology = compile_methodology_candidate(
        [
            {
                "pattern_id": "stable|memory-fabric|Hermes_Memory_Fabric|procedure|candidate",
                "pattern_type": "stable_repeated_claim",
            }
        ],
        project_scope="memory-fabric",
    )

    procedure = compile_procedure_block_candidate(methodology, project_scope="memory-fabric")

    assert procedure["block_type"] == "procedure_candidate"
    assert procedure["status"] == "review_required"
    assert procedure["project_scope"] == "memory-fabric"
    assert procedure["source_pattern_ids"] == methodology["source_pattern_ids"]
    assert procedure["policy"] == MEMORY_COMPILER_POLICY


def test_trace_and_recommendation_are_review_only():
    result = compile_memory_patterns([_fact(fact_id="stable-1"), _fact(fact_id="stable-2")])

    trace = explain_compilation_trace(result)
    recommendation = recommend_compilation_action(result)

    assert [step["step"] for step in trace] == [
        "normalize_facts",
        "select_current_facts",
        "group_contradictions",
        "compile_candidates",
    ]
    assert recommendation["creates_review_candidates_only"] is True
