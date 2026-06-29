#!/usr/bin/env python3
"""adjudicator.py — decide whether an information-gain run's output is acceptable.

Two layers:
  * structural_checks()  — deterministic, no model call. Schema/threshold/diversity
                           invariants plus per-case calibration (bucket size in the
                           expected band for an underspecified vs well-specified case).
  * adjudicate()         — an LLM judge (a DIFFERENT, stronger model than the one that
                           generated the questions, to reduce self-judging bias; CLAMBER
                           warns LLMs are weak self-judges) scores qualitative criteria.

A case is ACCEPTABLE iff structural checks pass AND the judge's required criteria
clear the floor. The judge reuses pipeline.raw_chat / extract_json.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import pipeline  # noqa: E402
from pipeline import resolve_alias  # noqa: E402

# Criteria the judge scores 0-1; these three must clear ACCEPT_FLOOR to pass.
REQUIRED_FOR_ACCEPT = ("framing_accuracy", "question_relevance", "calibration")
ADVISORY_CRITERIA = ("value_justified", "diversity")
ALL_CRITERIA = REQUIRED_FOR_ACCEPT + ADVISORY_CRITERIA
ACCEPT_FLOOR = 0.6


# ── deterministic structural checks ──────────────────────────────────────────


def structural_checks(result, case):
    """Schema/threshold/diversity/calibration invariants. Returns {passed, failures}."""
    failures = []
    fr = result.get("framing") or {}
    if not (fr.get("goal") or "").strip():
        failures.append("framing.goal is empty")
    if not (fr.get("baseline_plan") or "").strip():
        failures.append("framing.baseline_plan is empty")

    bucket = result.get("bucket") or []
    cfg = result.get("config") or {}
    hard_cap = cfg.get("hard_cap", 7)
    pre = cfg.get("pre_answer_threshold", 0.60)
    disc = cfg.get("discard_threshold", 0.40)

    if len(bucket) > hard_cap:
        failures.append(f"bucket {len(bucket)} exceeds hard_cap {hard_cap}")

    last = 2.0
    seen_targets = set()
    for i, r in enumerate(bucket):
        v = r.get("value")
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            failures.append(f"q{i} value out of [0,1]: {v}")
        elif v > last + 1e-9:
            failures.append(f"bucket not sorted by value at q{i} ({v} > {last})")
        else:
            last = v
        rec = r.get("recommendation")
        if rec not in ("PRE_ANSWER", "ASSUME_DEFAULT"):
            failures.append(f"q{i} bad recommendation: {rec}")
        if rec == "PRE_ANSWER" and isinstance(v, (int, float)) and v < pre - 1e-9:
            failures.append(f"q{i} PRE_ANSWER but value {v} < {pre}")
        if isinstance(v, (int, float)) and v < disc - 1e-9:
            failures.append(f"q{i} kept but value {v} < discard {disc}")
        tgt = (r.get("target") or "").strip().lower()
        if tgt and tgt in seen_targets:
            failures.append(f"q{i} duplicate target '{tgt}' survived diversity")
        seen_targets.add(tgt)

    lo, hi = case.get("expect_min_bucket", 0), case.get("expect_max_bucket", hard_cap)
    if not (lo <= len(bucket) <= hi):
        failures.append(
            f"calibration: bucket {len(bucket)} outside expected [{lo},{hi}] "
            f"for {case.get('expectation')} problem")

    return {"passed": not failures, "failures": failures}


# ── LLM adjudicator ──────────────────────────────────────────────────────────


def _summarize_result(result):
    fr = result.get("framing") or {}
    lines = [
        f"GOAL: {fr.get('goal', '')}",
        f"DECISION: {fr.get('decision', '')}",
        f"BASELINE PLAN: {fr.get('baseline_plan', '')}",
        f"RANKED QUESTIONS ({len(result.get('bucket') or [])} kept, "
        f"{result.get('discarded_count', 0)} discarded):",
    ]
    for i, r in enumerate(result.get("bucket") or []):
        m = (r.get("modal_answer") or {}).get("answer", "")
        lines.append(
            f"  {i + 1}. [{r.get('recommendation')}, value={r.get('value', 0):.2f}] "
            f"{r.get('question')}  (resolves: {r.get('target', '')}; "
            f"most-likely answer: {m[:80]})")
    return "\n".join(lines)


def judge_prompt(case, result):
    expectation = case.get("expectation")
    exp_text = (
        "This problem is UNDERSPECIFIED; a good analysis surfaces several clarifying "
        "questions that are genuinely specific to it and would change the approach."
        if expectation == "underspecified" else
        "This problem is ALREADY WELL-SPECIFIED; a good analysis surfaces FEW or NO "
        "high-value questions and must NOT manufacture generic ones.")
    return (
        "You are an exacting, skeptical evaluator of an 'information-gain' tool that "
        "ranks clarifying questions by their value of information for a problem.\n\n"
        f"PROBLEM:\n{case.get('problem')}\n\n"
        f"EXPECTATION: {exp_text}\n\n"
        f"TOOL OUTPUT:\n{_summarize_result(result)}\n\n"
        "Score each criterion from 0.0 to 1.0 with a one-line reason:\n"
        "- framing_accuracy: does GOAL/BASELINE PLAN correctly capture the problem "
        "(not off-topic or hallucinated)?\n"
        "- question_relevance: are the kept questions genuinely SPECIFIC to THIS problem "
        "(not generic filler that would fit any project)?\n"
        "- value_justified: do the PRE_ANSWER questions plausibly change the right "
        "approach (is the high value warranted)?\n"
        "- diversity: do the kept questions cover DISTINCT concerns (not near-duplicates)?\n"
        "- calibration: does the NUMBER of high-value questions match the expectation "
        "above (underspecified→several; well-specified→few/none)?\n\n"
        "Return ONLY a JSON object:\n"
        '{"criteria": {"framing_accuracy": {"score": 0.0, "reason": ""}, '
        '"question_relevance": {"score": 0.0, "reason": ""}, '
        '"value_justified": {"score": 0.0, "reason": ""}, '
        '"diversity": {"score": 0.0, "reason": ""}, '
        '"calibration": {"score": 0.0, "reason": ""}}, "summary": ""}\n'
        "Respond ONLY with the JSON object."
    )


def _coerce_criteria(parsed):
    """Normalize the judge's criteria into {name: {score, reason}} with floats."""
    crit = (parsed or {}).get("criteria") or {}
    out = {}
    for name in ALL_CRITERIA:
        c = crit.get(name) or {}
        if isinstance(c, (int, float)):
            c = {"score": c, "reason": ""}
        try:
            score = float(c.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        out[name] = {"score": max(0.0, min(1.0, score)), "reason": c.get("reason", "")}
    return out


def adjudicate(case, result, judge_model="deepseek", timeout=180):
    """LLM-judge the result. Returns {criteria, acceptable, summary, error}.

    `acceptable` here is the JUDGE's verdict only (required criteria ≥ floor); the
    overall accept (structural AND judge) is combined in evaluate_case().
    """
    model = resolve_alias(judge_model)
    parsed, err = pipeline._call_json(model, judge_prompt(case, result), timeout,
                                      num_predict=700)
    if parsed is None:
        return {"criteria": {}, "acceptable": False, "summary": "",
                "error": err or "judge returned no JSON"}
    criteria = _coerce_criteria(parsed)
    acceptable = all(criteria[c]["score"] >= ACCEPT_FLOOR for c in REQUIRED_FOR_ACCEPT)
    return {"criteria": criteria, "acceptable": acceptable,
            "summary": (parsed.get("summary") or "")[:300], "error": None}


def evaluate_case(case, result, judge_model="deepseek", timeout=180):
    """Combine structural + judge into an overall acceptance verdict."""
    structural = structural_checks(result, case)
    judged = adjudicate(case, result, judge_model, timeout)
    acceptable = structural["passed"] and judged["acceptable"] and not judged.get("error")
    return {
        "id": case.get("id"),
        "expectation": case.get("expectation"),
        "bucket_size": len(result.get("bucket") or []),
        "structural": structural,
        "judged": judged,
        "acceptable": acceptable,
    }
