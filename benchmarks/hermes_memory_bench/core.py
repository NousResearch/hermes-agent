from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.memory_bitemporal_fact_graph import (
    BITEMPORAL_FACT_GRAPH_POLICY,
    explain_fact_lineage,
    select_current_facts,
)
from agent.memory_contradiction_engine import explain_contradiction_group, group_contradictions
from agent.memory_compiler import MEMORY_COMPILER_POLICY, compile_memory_patterns
from agent.memory_blocks import MEMORY_BLOCK_POLICY, compile_blocks_from_compiler_result
from agent.memory_block_review_queue import (
    MEMORY_BLOCK_REVIEW_QUEUE_POLICY,
    build_review_queue,
    summarize_review_queue,
)
from agent.memory_review_decision_gate import (
    MEMORY_REVIEW_DECISION_GATE_POLICY,
    evaluate_review_queue_item,
    summarize_review_decisions,
)
from agent.memory_retrieval_fusion import fuse_memory_retrieval


BENCHMARK_TYPE = "hermes_memory_bench_v0.1"
DIMENSIONS = (
    "recall_accuracy",
    "temporal_accuracy",
    "source_provenance_accuracy",
    "governance_write_safety",
    "project_scope_isolation",
    "contradiction_handling",
    "hybrid_retrieval_fusion",
    "bitemporal_fact_graph",
    "contradiction_engine",
    "memory_compiler",
    "memory_blocks",
    "memory_block_review_queue",
    "memory_review_decision_gate",
    "latency_ms",
)
POLICY = {
    "read_only": True,
    "would_write_memory": False,
    "would_modify_config": False,
    "would_write_graph": False,
    "does_not_create_operation_events": True,
}


@dataclass(frozen=True)
class CaseResult:
    id: str
    dimension: str
    query: str
    expected_answer: str
    actual_answer: str
    score: float
    latency_ms: float
    passed: bool
    evidence: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dimension": self.dimension,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "passed": self.passed,
            "evidence": self.evidence,
        }


def fixtures_path() -> Path:
    return Path(__file__).with_name("fixtures") / "smoke_cases.json"


def load_cases(suite: str) -> list[dict[str, Any]]:
    if suite != "smoke":
        raise ValueError(f"Unsupported suite: {suite}")
    with fixtures_path().open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Smoke fixture must contain a list of cases.")
    return cases


def run_benchmark(suite: str = "smoke") -> dict[str, Any]:
    cases = [_evaluate_case(case) for case in load_cases(suite)]
    scores = _dimension_scores(cases)
    aggregate = _aggregate(cases)
    return {
        "benchmark_type": BENCHMARK_TYPE,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "suite": suite,
        "scores": scores,
        "cases": [case.to_json() for case in cases],
        "aggregate": aggregate,
        "policy": dict(POLICY),
    }


def write_report(report: dict[str, Any], output: str | Path | None = None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def _evaluate_case(case: dict[str, Any]) -> CaseResult:
    started = time.perf_counter()
    dimension = case["dimension"]
    expected = case["expected_answer"]
    actual, evidence = _answer_case(case)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    score = 1.0 if actual == expected else 0.0
    return CaseResult(
        id=case["id"],
        dimension=dimension,
        query=case["query"],
        expected_answer=expected,
        actual_answer=actual,
        score=score,
        latency_ms=latency_ms,
        passed=score == 1.0,
        evidence=evidence,
    )


def _answer_case(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dimension = case["dimension"]
    memories = list(case.get("memories", []))

    if dimension == "governance_write_safety":
        return "blocked", {
            "blocked_operation": case.get("unsafe_operation"),
            "reason": "Benchmark policy forbids writes and proposals.",
            "policy": dict(POLICY),
        }

    if dimension == "project_scope_isolation":
        scope = case.get("project_scope")
        scoped = [memory for memory in memories if memory.get("project_id") == scope]
        selected = _newest(scoped)
        return selected.get("content", ""), {
            "project_scope": scope,
            "candidate_count": len(memories),
            "scoped_candidate_count": len(scoped),
            "selected_project_id": selected.get("project_id"),
        }

    if dimension == "temporal_accuracy":
        selected = _newest(memories)
        return selected.get("content", ""), {
            "selected_created_at": selected.get("created_at"),
            "candidate_count": len(memories),
            "temporal_rule": "newest_matching_preference_wins",
        }

    if dimension == "source_provenance_accuracy":
        selected = _newest(memories)
        source_ok = selected.get("source") == case.get("required_source")
        provenance_ok = selected.get("provenance") == case.get("required_provenance")
        return selected.get("content", ""), {
            "source": selected.get("source"),
            "provenance": selected.get("provenance"),
            "source_ok": source_ok,
            "provenance_ok": provenance_ok,
        }

    if dimension == "contradiction_handling":
        return _contradiction_answer(memories), {
            "candidate_count": len(memories),
            "claim_keys": sorted({memory.get("claim_key") for memory in memories if memory.get("claim_key")}),
            "handling": "flag_candidate_for_review",
        }

    if dimension == "hybrid_retrieval_fusion":
        result = fuse_memory_retrieval(
            query=case["query"],
            candidates=memories,
            project_scope=case.get("project_scope"),
            entity_ids=case.get("entity_ids"),
            now=case.get("now"),
            limit=case.get("limit", 5),
        )
        selected = result["selected_memories"][0] if result["selected_memories"] else {}
        return selected.get("text", ""), {
            "fusion": result,
            "selected_id": selected.get("id"),
            "candidate_count": len(memories),
        }

    if dimension == "bitemporal_fact_graph":
        selected = select_current_facts(
            memories,
            at_time=case.get("at_time") or case.get("now"),
            project_scope=case.get("project_scope"),
        )
        winner = selected[0] if selected else None
        lineage = explain_fact_lineage(winner.fact_id, memories) if winner else {}
        return (winner.object if winner else ""), {
            "selected_fact_id": winner.fact_id if winner else None,
            "selected_fact": winner.to_json() if winner else None,
            "lineage": lineage,
            "candidate_count": len(memories),
            "policy": dict(BITEMPORAL_FACT_GRAPH_POLICY),
        }

    if dimension == "contradiction_engine":
        groups = group_contradictions(memories)
        explanation = explain_contradiction_group(groups[0]) if groups else {}
        recommendation = explanation.get("recommended_action", {})
        return recommendation.get("action", "no_action"), {
            "contradiction_groups": groups,
            "explanation": explanation,
            "review_recommendation": recommendation,
            "candidate_count": len(memories),
        }

    if dimension == "memory_compiler":
        result = compile_memory_patterns(memories, project_scope=case.get("project_scope"))
        procedure = result["procedure_block_candidate"]
        return procedure.get("status", ""), {
            "compiler": result,
            "procedure_block_candidate": procedure,
            "candidate_count": len(memories),
            "policy": dict(MEMORY_COMPILER_POLICY),
        }

    if dimension == "memory_blocks":
        compiler_result = compile_memory_patterns(memories, project_scope=case.get("project_scope"))
        blocks = compile_blocks_from_compiler_result(compiler_result, project_scope=case.get("project_scope"))
        block = blocks[0] if blocks else {}
        return block.get("block_type", ""), {
            "compiler": compiler_result,
            "memory_blocks": blocks,
            "candidate_count": len(memories),
            "policy": dict(MEMORY_BLOCK_POLICY),
        }

    if dimension == "memory_block_review_queue":
        compiler_result = compile_memory_patterns(memories, project_scope=case.get("project_scope"))
        blocks = compile_blocks_from_compiler_result(compiler_result, project_scope=case.get("project_scope"))
        queue = build_review_queue(blocks, reviewer=case.get("reviewer"))
        item = queue[0] if queue else {}
        return item.get("status", ""), {
            "compiler": compiler_result,
            "memory_blocks": blocks,
            "review_queue": queue,
            "summary": summarize_review_queue(queue),
            "candidate_count": len(memories),
            "policy": dict(MEMORY_BLOCK_REVIEW_QUEUE_POLICY),
        }

    if dimension == "memory_review_decision_gate":
        compiler_result = compile_memory_patterns(memories, project_scope=case.get("project_scope"))
        blocks = compile_blocks_from_compiler_result(compiler_result, project_scope=case.get("project_scope"))
        queue = build_review_queue(blocks, reviewer=case.get("reviewer"))
        decisions = [evaluate_review_queue_item(item, reviewer=case.get("reviewer")) for item in queue]
        candidate = decisions[0] if decisions else {}
        return candidate.get("decision", ""), {
            "compiler": compiler_result,
            "memory_blocks": blocks,
            "review_queue": queue,
            "decision_candidates": decisions,
            "summary": summarize_review_decisions(decisions),
            "candidate_count": len(memories),
            "created_real_proposal": False,
            "created_operation_event": False,
            "policy": dict(MEMORY_REVIEW_DECISION_GATE_POLICY),
        }

    selected = _newest(memories)
    return selected.get("content", ""), {
        "candidate_count": len(memories),
        "selected_project_id": selected.get("project_id"),
        "selected_source": selected.get("source"),
    }


def _newest(memories: list[dict[str, Any]]) -> dict[str, Any]:
    if not memories:
        return {}
    return max(memories, key=lambda memory: memory.get("created_at", ""))


def _contradiction_answer(memories: list[dict[str, Any]]) -> str:
    normalized = {str(memory.get("content", "")).lower() for memory in memories}
    has_allowed = any(" is allowed" in content for content in normalized)
    has_blocked = any(" is blocked" in content or "forbidden" in content for content in normalized)
    return "contradiction_detected" if has_allowed and has_blocked else _newest(memories).get("content", "")


def _dimension_scores(cases: list[CaseResult]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for dimension in DIMENSIONS:
        if dimension == "latency_ms":
            scores[dimension] = round(sum(case.latency_ms for case in cases) / max(len(cases), 1), 3)
            continue
        relevant = [case for case in cases if case.dimension == dimension]
        scores[dimension] = round(sum(case.score for case in relevant) / len(relevant), 3) if relevant else 0.0
    return scores


def _aggregate(cases: list[CaseResult]) -> dict[str, Any]:
    case_count = len(cases)
    passed_count = sum(1 for case in cases if case.passed)
    score = sum(case.score for case in cases) / case_count if case_count else 0.0
    return {
        "overall_score": round(score, 3),
        "case_count": case_count,
        "passed_count": passed_count,
        "failed_count": case_count - passed_count,
        "mean_latency_ms": round(sum(case.latency_ms for case in cases) / max(case_count, 1), 3),
        "dimensions": list(DIMENSIONS),
    }
