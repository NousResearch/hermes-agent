"""Offline contracts for original-evidence augmentation and frozen-summary pairs."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from evals.compaction.evidence import (
    Evidence, collect_evidence, evidence_asker, rank_evidence, render_evidence, select_evidence,
)
from evals.compaction.fixtures import estimate_tokens
from evals.compaction import runner


def test_native_question_bank_accepts_qwen_jsonl_without_changing_questions():
    questions = [{"q": "code?", "gold": "Z1234", "where": "region code Z1234"},
                 {"q": "error?", "gold": "E42", "where": "error E42"}]
    assert runner.parse_question_bank(json.dumps(questions)) == questions
    assert runner.parse_question_bank("\n".join(json.dumps(q) for q in questions)) == questions
    with pytest.raises(ValueError, match="q/gold/where"):
        runner.parse_question_bank('[{"q": "missing gold"}]')


def test_collects_pre_pruning_bodies_and_assistant_facts_without_mutation():
    messages = [
        {"role": "system", "content": "system secret"},
        {"role": "user", "content": "investigate the failure"},
        {"role": "assistant", "content": "The root cause is stale routing; commit abc123 fixes it."},
        {"role": "tool", "tool_call_id": "c1", "content": "exact output: " + "xyz " * 30},
        {"role": "assistant", "content": "recent text"},
    ]
    compressed = [{"role": "assistant", "content": "summary"}, messages[-1]]
    before = deepcopy(messages)
    units = collect_evidence(messages, compressed, chunk_chars=40)
    assert {u.message_index for u in units} == {2, 3}
    assert "".join(u.text for u in units if u.message_index == 3) == messages[3]["content"]
    assert all(u.text == messages[u.message_index]["content"][u.start:u.start + len(u.text)] for u in units)
    assert messages == before
    assert "recent text" not in render_evidence(units)


def test_redacts_candidate_bodies_before_ranking():
    units = collect_evidence([{"role": "tool", "content": "api_key=sk-1234567890abcdefghijklmnopqrstuvwxyz"}], [])
    assert units and "sk-1234567890abcdefghijklmnopqrstuvwxyz" not in units[0].text


def test_ranked_prefix_budget_counts_framing_and_restores_source_order():
    units = [Evidence("a", 1, 0, "tool", "old fact"), Evidence("b", 2, 0, "assistant", "new fact")]
    scores = {"a": 0.6, "b": 0.9}
    budget = estimate_tokens({"content": render_evidence([units[1]])})
    assert select_evidence(units, scores, 2, budget) == [units[1]]
    assert select_evidence(units, scores, 2, budget - 1) == []
    selected = select_evidence(units, scores, 2, 1000)
    text = render_evidence(selected)
    assert text.index("old fact") < text.index("new fact")
    assert select_evidence(units, {}, 2, 1000) == []


def test_batches_visible_bodies_and_reuses_only_valid_probabilities():
    units = [Evidence(str(i), i, 0, "tool", "x" * 400) for i in range(4)]
    requests = []

    def ask(state, questions):
        requests.append((state, questions))
        assert {u["id"] for u in state["evidence_units"]} == set(questions)
        assert all(u["text"] == "x" * 400 for u in state["evidence_units"])
        return {"answers": {k: {"type": "noul", "noul": 0.8} for k in questions},
                "model": "fake", "usage": {"input_tokens": 10, "cost": 0.01}}

    scores, stats = rank_evidence(units, "investigate", ask, max_request_tokens=600)
    assert len(requests) == 4
    assert set(scores) == {u.id for u in units}
    assert stats["input_tokens"] == 40 and stats["cost_usd"] == pytest.approx(0.04)
    assert all(state["task"] == "investigate" for state, _ in requests)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.5", None])
def test_invalid_judgment_never_selects_evidence(value):
    unit = Evidence("a", 1, 0, "tool", "fact")
    scores, stats = rank_evidence([unit], "task", lambda *_: {"answers": {"a": {"type": "noul", "noul": value}}})
    assert scores == {} and stats["invalid_answers"] == 1
    assert select_evidence([unit], scores, 1, 1000) == []


def test_failed_requests_and_oversized_units_remain_observable():
    def fail(*_):
        raise TimeoutError("offline")

    unit = Evidence("a", 1, 0, "tool", "fact")
    scores, stats = rank_evidence([unit], "task", fail)
    assert scores == {} and stats["requests"] == stats["failed_batches"] == 1
    scores, stats = rank_evidence([unit], "x" * 10000, fail, max_request_tokens=600)
    assert scores == {} and stats["requests"] == 0 and stats["skipped"] == ["a"]


def test_first_party_transport_uses_its_own_key_and_records_unknown_cost(monkeypatch):
    calls = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    monkeypatch.setattr("evals.compaction.jev_arm.openrouter_asker", lambda **kw: calls.append(kw) or (lambda *_: {
        "answers": {"a": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 10},
    }))
    ask = evidence_asker("typesafe", "pinned-model")
    assert calls[0]["api_key"] == "typesafe-test"
    assert calls[0]["model"] == "pinned-model"
    assert calls[0]["url"] == "https://api.typesafe.ai/v1/systemone"
    _, stats = rank_evidence([Evidence("a", 1, 0, "tool", "fact")], "task", ask)
    assert stats["cost_usd"] is None
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        evidence_asker("typesafe")


def test_noop_native_compression_is_not_an_augmentation_experiment(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "short"}]
    monkeypatch.setattr(runner, "_compress_with_policy", lambda *_: (messages, SimpleNamespace(), {}))
    with pytest.raises(ValueError, match="no-op"):
        runner.run_evidence_pairs("current", {}, messages, [], tmp_path, [1], 1000, "task", 2000)


def test_paired_run_freezes_one_compression_one_ranking_and_one_recovery_archive(tmp_path, monkeypatch):
    original = [
        {"role": "user", "content": "continue the deployment"},
        {"role": "assistant", "content": "The deployment identifier is deploy-123; keep it exact."},
        {"role": "tool", "content": "The old log has an exact error: E_ROUTE_42."},
    ]
    compressed = [{"role": "assistant", "content": "FROZEN SUMMARY"}]
    saved = deepcopy(compressed)
    calls, requests, searches, answers = [], [], [], []

    def compress(*_):
        calls.append(1)
        return compressed, SimpleNamespace(_last_summary_error=None), {"compaction_calls": 1}

    def ask(state, questions):
        requests.append((state, questions))
        assert "EXAM_SENTINEL" not in json.dumps((state, questions))
        return {"answers": {k: {"type": "noul", "noul": 0.9} for k in questions}}

    def llm(prompt, **_):
        if prompt.startswith("Score this answer"):
            return '{"score": 2, "why": "offline"}'
        if prompt.startswith("You are an AI agent resuming"):
            answers.append(prompt)
        return "deploy-123"

    def search(archive, query):
        searches.append(deepcopy(archive))
        return "archive hit"

    monkeypatch.setattr(runner, "_compress_with_policy", compress)
    monkeypatch.setattr(runner, "_call", llm)
    monkeypatch.setattr(runner, "keyword_search", search)
    monkeypatch.setattr("evals.compaction.evidence.evidence_asker", lambda *_: ask)
    results = runner.run_evidence_pairs(
        "current", {}, original, [{"q": "EXAM_SENTINEL", "gold": "deploy-123"}],
        tmp_path, [1, 2], 1000, "", 2000,
    )
    assert len(calls) == len(requests) == 1
    assert len(results) == 10 and len(searches) == 5
    assert all(archive == original for archive in searches)
    assert compressed == saved
    assert all("FROZEN SUMMARY" in prompt for prompt in answers)
    assert len({r["evidence"]["frozen_sha256"] for r in results}) == 1
    assert {r["evidence"]["actual_n"] for r in results if "actual_n" in r["evidence"]} == {1, 2}
    assert (tmp_path / "current-frozen.json").exists()


def test_recovery_keeps_messages_whose_only_difference_is_beyond_200_chars(tmp_path, monkeypatch):
    prefix = "shared prefix " * 20
    original = [{"role": "assistant", "content": prefix + "lost fact"}]
    compressed = [{"role": "assistant", "content": prefix + "different fact"}]
    archives = []
    monkeypatch.setattr(runner, "keyword_search", lambda archive, _: archives.append(archive) or "hit")
    monkeypatch.setattr(runner, "_call", lambda prompt, **_: '{"score": 2}' if prompt.startswith("Score") else "answer")
    runner.run_policy("current", {}, original, [{"q": "fact?", "gold": "lost fact"}], tmp_path, True,
                      prepared=(compressed, SimpleNamespace(), {}))
    assert archives == [original]
