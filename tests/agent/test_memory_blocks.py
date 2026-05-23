from copy import deepcopy

from agent.memory_bitemporal_fact_graph import BITEMPORAL_FACT_GRAPH_POLICY
from agent.memory_blocks import (
    MEMORY_BLOCK_POLICY,
    SUPPORTED_MEMORY_BLOCK_TYPES,
    compile_blocks_from_compiler_result,
    create_memory_block_candidate,
    explain_memory_block_candidate,
    normalize_memory_block,
    recommend_memory_block_action,
    validate_memory_block_candidate,
)
from agent.memory_compiler import compile_memory_patterns


def _fact(**overrides):
    base = {
        "fact_id": "fact-1",
        "subject": "Hermes Memory Fabric",
        "predicate": "procedure",
        "object": "create review-only memory block candidates",
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


def test_all_supported_block_types_validate():
    for block_type in SUPPORTED_MEMORY_BLOCK_TYPES:
        block = create_memory_block_candidate(block_type, {"text": block_type}, project_scope="memory-fabric")

        assert block["validity"]["valid"] is True
        assert validate_memory_block_candidate(block) == {"valid": True, "errors": []}


def test_unsupported_block_type_fails_validation():
    block = create_memory_block_candidate("unsupported", {"text": "invalid"})

    assert block["validity"]["valid"] is False
    assert "unsupported_block_type:unsupported" in block["validity"]["errors"]


def test_block_candidate_is_review_required_and_proposal_only():
    block = create_memory_block_candidate("persona", "Prefer terse engineering updates.")

    assert block["status"] == "review_required"
    assert block["mutation_policy"] == "proposal_only"
    assert block["direct_write_allowed"] is False
    assert block["last_reviewed_at"] is None


def test_compile_blocks_from_compiler_result_creates_procedural_rules_block():
    compiler_result = compile_memory_patterns(
        [_fact(fact_id="stable-1"), _fact(fact_id="stable-2", source_episode_id="episode-2")],
        project_scope="memory-fabric",
    )

    blocks = compile_blocks_from_compiler_result(compiler_result, project_scope="memory-fabric")

    assert len(blocks) == 1
    block = blocks[0]
    assert block["block_type"] == "procedural_rules"
    assert block["status"] == "review_required"
    assert block["content"]["applied"] is False
    assert block["content"]["compiler_candidate_type"] == "procedure_candidate"
    assert block["mutation_policy"] == "proposal_only"


def test_source_pattern_ids_are_preserved():
    compiler_result = compile_memory_patterns([_fact(fact_id="stable-1"), _fact(fact_id="stable-2")])

    block = compile_blocks_from_compiler_result(compiler_result)[0]

    assert block["source_pattern_ids"] == compiler_result["procedure_block_candidate"]["source_pattern_ids"]


def test_source_fact_ids_are_preserved_when_provided():
    compiler_result = compile_memory_patterns([_fact(fact_id="stable-1"), _fact(fact_id="stable-2")])
    compiler_result["procedure_block_candidate"]["source_fact_ids"] = ["provided-2", "provided-1"]

    block = compile_blocks_from_compiler_result(compiler_result)[0]

    assert block["source_fact_ids"] == ["provided-1", "provided-2"]


def test_input_data_is_not_mutated():
    raw_block = {
        "block_type": "project_context",
        "content": {"nested": ["value"]},
        "source_pattern_ids": ["pattern-b", "pattern-a"],
        "metadata": {"source_event_id": "event-1"},
    }
    raw_before = deepcopy(raw_block)
    compiler_result = compile_memory_patterns([_fact(fact_id="stable-1"), _fact(fact_id="stable-2")])
    compiler_before = deepcopy(compiler_result)

    normalize_memory_block(raw_block)
    compile_blocks_from_compiler_result(compiler_result)

    assert raw_block == raw_before
    assert compiler_result == compiler_before


def test_policy_proves_no_memory_config_graph_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    block = create_memory_block_candidate("safety_policy", "Never write memory directly.")
    explanation = explain_memory_block_candidate(block)
    recommendation = recommend_memory_block_action(block)

    assert block["policy"] == MEMORY_BLOCK_POLICY
    assert block["policy"]["read_only"] is True
    assert block["policy"]["would_write_memory"] is False
    assert block["policy"]["would_modify_config"] is False
    assert block["policy"]["would_write_graph"] is False
    assert block["policy"]["does_not_create_operation_events"] is True
    assert block["policy"]["creates_review_candidates_only"] is True
    assert explanation["applied"] is False
    assert recommendation["creates_review_candidates_only"] is True
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()
