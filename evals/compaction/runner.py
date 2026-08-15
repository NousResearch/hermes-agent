"""Compaction eval runner.

Pipeline per transcript:
  1. Load + cap the transcript.
  2. Generate (or load cached) recall questions from the region that will be
     summarized away under the CURRENT policy (the most conservative boundary:
     anything the current policy summarizes is fair game for every policy).
  3. For each policy: compress, then answer each question with ONLY the
     compressed context, using a single LLM call per question.
  4. Judge answers against gold with an LLM judge (sees gold; answerer
     does not).
  5. Write per-policy results JSON for report.py.

Run from repo root with the project venv (needs a configured provider).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.compaction.fixtures import (  # noqa: E402
    estimate_tokens,
    load_transcript,
    total_tokens,
)
from evals.compaction.policies import EVAL_MODEL, POLICIES, apply_policy  # noqa: E402

QUESTION_PROMPT = """You are building a factual recall exam from an AI-agent work session transcript.

Write {n} questions that test SPECIFIC, VERIFIABLE facts from the transcript below: identifiers (PR numbers, file paths, error messages, commit subjects), decisions and their reasons, user instructions, and outcomes. Rules:
- Every answer must appear literally in the transcript.
- No questions about the system prompt or generic behavior.
- Spread questions across the WHOLE span (early, middle, late).
- Prefer facts that matter for continuing the work (what was decided, what failed, what the user asked for).

Return STRICT JSON: a list of {{"q": "...", "gold": "...", "where": "<short quote locating the answer>"}}.

TRANSCRIPT:
{transcript}
"""

ANSWER_PROMPT = """You are an AI agent resuming a work session. Below is your CURRENT conversation context (it may include a compaction summary of earlier work). Answer the question using ONLY this context. If the context does not contain the answer, say exactly "NOT IN CONTEXT" and give your best guess after a semicolon.

CONTEXT:
{context}

QUESTION: {question}

Answer in one or two sentences."""

JUDGE_PROMPT = """Score this answer against the gold answer. Reply with STRICT JSON: {{"score": 2|1|0, "why": "..."}}.
2 = factually matches gold (wording may differ)
1 = partially correct or hedged-but-right ("NOT IN CONTEXT; guess X" where X is right scores 1)
0 = wrong, or "NOT IN CONTEXT" with a wrong/no guess

QUESTION: {question}
GOLD: {gold}
ANSWER: {answer}"""


def _call(prompt: str, max_tokens: int = 2000) -> str:
    from agent.auxiliary_client import call_llm

    resp = call_llm(
        messages=[{"role": "user", "content": prompt}],
        task="compression",
        max_tokens=max_tokens,
    )
    if hasattr(resp, "choices"):
        return resp.choices[0].message.content or ""
    return str(resp)


def _extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=0)
    return json.loads(text[start:])


def serialize_for_exam(messages, char_cap: int = 600_000) -> str:
    parts = []
    for m in messages:
        role = m.get("role")
        c = m.get("content")
        if not isinstance(c, str) or not c:
            continue
        if role == "system":
            continue
        parts.append(f"[{role}] {c}")
    text = "\n\n".join(parts)
    if len(text) > char_cap:
        half = char_cap // 2
        text = text[:half] + "\n\n...[middle elided for exam generation]...\n\n" + text[-half:]
    return text


def summarized_region(compressor_module, messages):
    """The middle region the current policy would summarize: everything
    between the protected head and the tail cut. Questions come from here."""
    from agent.context_compressor import ContextCompressor

    comp = ContextCompressor(model=EVAL_MODEL, quiet_mode=True)
    head_end = comp.protect_first_n
    tail_start = comp._find_tail_cut_by_tokens(messages, head_end)
    return messages[head_end:tail_start]


def generate_questions(messages, n: int, cache_path: Path) -> list:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    import agent.context_compressor as cc

    region = summarized_region(cc, messages)
    text = serialize_for_exam(region)
    raw = _call(QUESTION_PROMPT.format(n=n, transcript=text), max_tokens=4000)
    questions = _extract_json(raw)[:n]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(questions, indent=1))
    return questions


def run_policy(name: str, spec: dict, messages, questions, out_dir: Path) -> dict:
    from agent.context_compressor import ContextCompressor

    before = copy.deepcopy(messages)
    comp = apply_policy(ContextCompressor(model=EVAL_MODEL, quiet_mode=True), spec)
    for key, value in (spec.get("ctor") or {}).items():
        setattr(comp, key, value)
    t0 = time.time()
    compressed = comp.compress(copy.deepcopy(messages), current_tokens=total_tokens(messages), force=True)
    elapsed = time.time() - t0

    context_text = serialize_for_exam(compressed, char_cap=700_000)
    results = []
    for qa in questions:
        answer = _call(ANSWER_PROMPT.format(context=context_text, question=qa["q"]), max_tokens=400)
        verdict_raw = _call(JUDGE_PROMPT.format(question=qa["q"], gold=qa["gold"], answer=answer), max_tokens=300)
        try:
            verdict = _extract_json(verdict_raw)
        except Exception:
            verdict = {"score": 0, "why": f"judge parse failure: {verdict_raw[:100]}"}
        results.append({"q": qa["q"], "gold": qa["gold"], "answer": answer, **verdict})

    scored = [r["score"] for r in results]
    summary = {
        "policy": name,
        "before_tokens": total_tokens(before),
        "after_tokens": total_tokens(compressed),
        "after_msgs": len(compressed),
        "compress_seconds": round(elapsed, 1),
        "recall_pct": round(100 * sum(scored) / (2 * len(scored)), 1) if scored else 0.0,
        "scores": scored,
        "summary_error": getattr(comp, "_last_summary_error", None),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--cap-tokens", type=int, default=500_000)
    ap.add_argument("--policies", default="current,tail25k,codex_style")
    ap.add_argument("--questions", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--also-uncompacted", action="store_true")
    args = ap.parse_args()

    messages = load_transcript(args.transcript, cap_tokens=args.cap_tokens)
    out_dir = Path(args.out)
    tid = hashlib.md5(args.transcript.encode()).hexdigest()[:10]
    qcache = out_dir / f"questions-{tid}.json"
    questions = generate_questions(messages, args.questions, qcache)
    print(f"{len(questions)} questions ready ({qcache})")

    summaries = []
    if args.also_uncompacted:
        spec = {"ctor": {}, "attrs": {"tail_token_budget": 10**9}}
        # control: no compression at all — answer from the full transcript
        context_text = serialize_for_exam(messages, char_cap=900_000)
        results = []
        for qa in questions:
            answer = _call(ANSWER_PROMPT.format(context=context_text, question=qa["q"]), max_tokens=400)
            verdict_raw = _call(JUDGE_PROMPT.format(question=qa["q"], gold=qa["gold"], answer=answer), max_tokens=300)
            try:
                verdict = _extract_json(verdict_raw)
            except Exception:
                verdict = {"score": 0, "why": "judge parse failure"}
            results.append({"q": qa["q"], **verdict, "answer": answer})
        scored = [r["score"] for r in results]
        ctl = {
            "policy": "uncompacted_control",
            "before_tokens": total_tokens(messages),
            "after_tokens": total_tokens(messages),
            "recall_pct": round(100 * sum(scored) / (2 * len(scored)), 1),
            "scores": scored,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "uncompacted_control.json").write_text(json.dumps({"summary": ctl, "results": results}, indent=1))
        summaries.append(ctl)
        print(json.dumps(ctl, indent=1))

    for name in args.policies.split(","):
        name = name.strip()
        if name not in POLICIES:
            print(f"unknown policy {name}, skipping"); continue
        s = run_policy(name, POLICIES[name], messages, questions, out_dir)
        summaries.append(s)
        print(json.dumps(s, indent=1))

    (out_dir / "scorecard.json").write_text(json.dumps(summaries, indent=1))
    print(f"\nscorecard -> {out_dir}/scorecard.json")


if __name__ == "__main__":
    main()
