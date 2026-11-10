"""Frozen summary/evidence pairing, exact grading and source provenance."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from evals.compaction import study
from evals.compaction.study_cases import event_units, exam, grade, write


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=text.split())


def sample():
    history = [{"role": "system", "content": "real system"}, {"role": "user", "content": "investigate task"}]
    for index in range(3):
        history.extend([
            {"role": "assistant", "tool_calls": [{"id": str(index), "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": f"f{index}"}),
            }}]},
            {"role": "tool", "tool_call_id": str(index), "content": json.dumps({"content": f"fact {index}"})},
        ])
    return {"id": "example", "messages": history, "exam": "EXAM_SENTINEL",
            "gold": {"integer": 7, "flag": True, "mapping": {"x": 2}, "list": ["a", "b"]},
            "fields": {"integer": "integer value"}, "unordered": [], "control_reason": None}


@pytest.mark.parametrize("value", [True, 7.0, "7", None])
def test_grader_requires_exact_nested_types_and_question_does_not_reveal_gold(value):
    case = sample()
    assert all(grade(case, {"facts": case["gold"]}).values())
    assert not grade(case, {"facts": {**case["gold"], "integer": value}})["integer"]
    assert not grade(case, {"facts": {"mapping": {"x": True}}})["mapping"]
    assert not grade(case, {"facts": {"list": ["b", "a"]}})["list"]
    case["gold"] = {"hidden": "GOLD_SENTINEL"}
    assert "GOLD_SENTINEL" not in exam(case)


def test_preparation_reuses_one_ranking_and_two_frozen_summaries_across_top_n(tmp_path, monkeypatch):
    case = sample()
    original = deepcopy(case)
    calls = []

    def ask(state, questions):
        calls.append(deepcopy((state, questions)))
        assert "EXAM_SENTINEL" not in json.dumps((state, questions))
        assert "gold" not in state and "summary" not in state
        return {"answers": {k: {"type": "noul", "noul": 0.5} for k in questions}}

    generations = []
    monkeypatch.setattr(study.Experiment, "summarize", lambda self, *args:
                        generations.append("S") or "S frozen exact")
    monkeypatch.setattr(study.Experiment, "native_summary", lambda self, *args:
                        generations.append("H") or "H frozen exact")
    rows = study.prepare(case, tmp_path, Tokenizer(), ask)
    assert len(calls) == 3 and generations == ["S", "H"] * 3
    assert len(rows) == 39 and case == original
    for repeat in range(3):
        for n in range(1, 6):
            s, h = [next(r for r in rows if r["arm"] == f"{kind}+top{n}" and r["repeat"] == repeat)
                    for kind in ("S", "H")]
            assert s["selected"] == h["selected"] and s["actual_n"] == min(n, 3)
            assert s["summary"] == "S frozen exact" and h["summary"] == "H frozen exact"
    assert {r["allowance"] for r in rows} == {64}


def test_hermes_generator_uses_configured_transport_and_preserves_provenance(tmp_path, monkeypatch):
    prompts = []

    def ask(self, label, context, **kwargs):
        prompts.append(context[0]["content"])
        assert kwargs == {"summary": True, "native": True}
        return ("## Historical Task Snapshot\nNone. This session contains no user-authored turns."
                "\n\n## Critical Context\nfact 1", {})

    monkeypatch.setattr(study.Experiment, "ask", ask)
    case = sample()
    before = deepcopy(case)
    summary = study.Experiment(tmp_path, Tokenizer()).native_summary("c0", case["messages"], 64)
    assert "Critical Context" in summary and "fact 1" in summary
    assert len(prompts) == 1 and "EXAM_SENTINEL" not in prompts[0]
    assert "at most 22 English words TOTAL" in prompts[0]
    assert case == before


def test_source_loader_checks_hashes_excludes_report_and_private_metadata(tmp_path, monkeypatch):
    root = tmp_path / "example"
    root.mkdir()
    case = sample()
    history = deepcopy(case["messages"]) + [{"role": "assistant", "content": "FINAL_SENTINEL"}]
    history[1]["_db_row_snapshot"] = "PRIVATE_SENTINEL"
    write(root / "trajectory.full.json", {"messages": history})
    write(root / "collection.json", {"completed": True, "trajectory_sha256": hashlib.sha256(
        (root / "trajectory.full.json").read_bytes()).hexdigest()})
    definition = {k: v for k, v in case.items() if k not in {"messages", "exam"}}
    write(tmp_path / "corpus.json", {"cases": [{**definition, "source_hashes": {}}]})
    monkeypatch.setattr(study, "CORPUS_SHA", hashlib.sha256((tmp_path / "corpus.json").read_bytes()).hexdigest())
    loaded = study.load_cases(tmp_path / "corpus.json")
    assert "FINAL_SENTINEL" not in json.dumps(loaded) and "PRIVATE_SENTINEL" not in json.dumps(loaded)
    assert loaded[0]["messages"] == case["messages"]
    (root / "trajectory.full.json").write_text("changed")
    with pytest.raises(ValueError, match="changed trajectory"):
        study.load_cases(tmp_path / "corpus.json")


def test_strict_prefix_retains_low_scores_and_never_trims_source_blocks():
    units = event_units(sample()["messages"])
    scores = {u["id"]: 0.01 for u in units}
    assert study.top_n(units, scores, 3) == units
    assert study.top_n(units, scores, 1) == units[:1]
    with pytest.raises(ValueError, match="valid probability"):
        study.top_n(units, {**scores, units[0]["id"]: float("nan")}, 1)


def test_failed_contexts_keep_fixed_task_and_trial_denominators():
    cases = [{"id": "a", "control_reason": None}, {"id": "b", "control_reason": None},
             {"id": "control", "control_reason": "short_fixture"}]
    rows = [{"case": c["id"], "arm": arm, "repeat": i, "valid": c["id"] != "b",
             "trials": [{"checks": {"fact": True}, "usage": {"prompt_tokens": 20}}] * 2}
            for c in cases for arm in study.ARMS for i in range(3)]
    result = study.report(rows, cases)
    assert result["primary"][0]["facts"] == 0.5
    assert result["controls"][0]["facts"] == 1
    rows[0].update(valid=False, trials=[])
    assert study.report(rows, cases)["primary"][0]["facts"] == pytest.approx(1 / 3)
