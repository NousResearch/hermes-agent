#!/usr/bin/env python3
"""Run the codex CLI as an eval arm on the same transcripts + question banks.

Per transcript:
  1. Split the 500K-token prefix into ~150KB chunk files in a work dir.
  2. `codex exec` reads every file (2-3 sentence summary each) — the read
     volume exceeds codex's 258K window, so its auto-compaction fires
     naturally (verified via token_count drops / compacted events in the
     rollout jsonl).
  3. `codex exec resume --last` asks the SAME 15 exam questions; answers are
     judged by the same LLM judge against the same golds.

Usage: codex_arm.py <lineage_json> <questions_json> <workdir> <out_json>
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "main-co"))

LINEAGE = sys.argv[1]
QUESTIONS = sys.argv[2]
WORKDIR = Path(sys.argv[3])
OUT = sys.argv[4]

JUDGE_PROMPT = """Score this answer against the gold answer. Reply with STRICT JSON: {{"score": 2|1|0, "why": "..."}}.
2 = factually matches gold (wording may differ)
1 = partially correct or hedged-but-right
0 = wrong, or refuses/says it doesn't know with a wrong/no guess

QUESTION: {question}
GOLD: {gold}
ANSWER: {answer}"""


def prepare_chunks() -> int:
    from evals.compaction.fixtures import load_transcript

    WORKDIR.mkdir(parents=True, exist_ok=True)
    msgs = load_transcript(LINEAGE, cap_tokens=500_000)
    chunk, size, idx = [], 0, 0
    for m in msgs:
        c = m.get("content") or ""
        if not isinstance(c, str) or not c:
            continue
        chunk.append(f"--- {m['role']} ---\n{c}\n")
        size += len(c)
        if size > 150_000:
            (WORKDIR / f"transcript_{idx:02d}.txt").write_text(
                "\n".join(chunk), encoding="utf-8")
            chunk, size = [], 0
            idx += 1
    if chunk:
        (WORKDIR / f"transcript_{idx:02d}.txt").write_text(
            "\n".join(chunk), encoding="utf-8")
        idx += 1
    return idx


def newest_rollout() -> str:
    files = sorted(
        glob.glob(os.path.expanduser("~/.codex/sessions/*/*/*/rollout-*.jsonl")),
        key=os.path.getmtime,
    )
    return files[-1] if files else ""


def rollout_session_id(path: str) -> str:
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "session_meta":
            return d.get("payload", {}).get("session_id", "")
    return ""


def last_agent_message(path: str) -> str:
    msgs = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        p = d.get("payload", {})
        if p.get("type") == "agent_message":
            msgs.append(p.get("message", ""))
    return msgs[-1] if msgs else ""


def rollout_stats(path: str) -> dict:
    compacted = 0
    peak = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        p = d.get("payload", {})
        if d.get("type") == "compacted" or p.get("type") == "compacted":
            compacted += 1
        if p.get("type") == "token_count" and p.get("info"):
            last = p["info"].get("last_token_usage") or {}
            ctx = last.get("input_tokens", 0) + last.get("cached_input_tokens", 0)
            peak = max(peak, ctx)
    return {"compaction_events": compacted, "peak_context_tokens": peak}


def codex(args: list, prompt: str, timeout: int = 3600) -> str:
    proc = subprocess.run(
        ["codex", "exec", *args, "--skip-git-repo-check", prompt],
        cwd=str(WORKDIR), capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout + proc.stderr


def judge(question: str, gold: str, answer: str) -> dict:
    from agent.auxiliary_client import call_llm

    resp = call_llm(
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, gold=gold, answer=answer)}],
        task="compression", max_tokens=300,
    )
    text = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"score": 0, "why": f"judge parse failure: {text[:80]}"}


def main():
    n = prepare_chunks()
    print(f"[codex-arm] {WORKDIR.name}: {n} chunk files", flush=True)
    t0 = time.time()
    codex(
        ["-s", "read-only"],
        f"This directory contains transcript_00.txt through transcript_{n-1:02d}.txt. "
        "Read EVERY file COMPLETELY one at a time using 'cat transcript_NN.txt' "
        "(full file, do not use head/tail/grep). After each file, write a 2-3 "
        "sentence summary of what happened in that portion. Do not skip any file.",
    )
    rollout = newest_rollout()
    session_id = rollout_session_id(rollout)
    stats = rollout_stats(rollout)
    # Codex auto-compacts at ~90% of its 258K window. If one read pass didn't
    # trigger it, re-read files in the SAME session until it does (max 3
    # extra passes) — the comparison requires post-compaction state.
    passes = 0
    while stats["compaction_events"] == 0 and passes < 3:
        passes += 1
        print(f"[codex-arm] no compaction yet (peak={stats['peak_context_tokens']:,}) — re-read pass {passes}", flush=True)
        codex(
            ["resume", session_id],
            "Re-read ALL transcript files again completely with 'cat', one at a "
            "time, and refine each of your per-file summaries with any details "
            "you missed. Do not skip any file.",
        )
        stats = rollout_stats(rollout)
    read_s = time.time() - t0
    print(f"[codex-arm] read phase {read_s:.0f}s, {stats}", flush=True)
    if stats["compaction_events"] == 0:
        print("[codex-arm] WARNING: compaction never fired — arm invalid", flush=True)

    questions = json.loads(Path(QUESTIONS).read_text(encoding="utf-8"))
    qlist = "\n".join(f"{i+1}. {q['q']}" for i, q in enumerate(questions))
    codex(
        ["resume", session_id],
        "Based on everything you learned from the transcript files earlier in "
        "this session, answer the following questions from memory. Do NOT "
        "re-read any files — answer only from what you currently retain in "
        "context. If you don't know, say 'UNKNOWN' and give your best guess. "
        "Reply with a numbered list, one concise answer per question.\n\n" + qlist,
    )
    quiz_text = last_agent_message(rollout)
    print(f"[codex-arm] quiz reply: {len(quiz_text)} chars", flush=True)
    answers = {}
    for m in re.finditer(r"(?m)^\s*\**(\d{1,2})[.)]\**\s+(.+?)(?=^\s*\**\d{1,2}[.)]\**\s|\Z)",
                         quiz_text, re.S):
        answers[int(m.group(1))] = m.group(2).strip()[:600]

    results = []
    for i, q in enumerate(questions):
        ans = answers.get(i + 1, "(no answer parsed)")
        verdict = judge(q["q"], q["gold"], ans)
        results.append({"q": q["q"], "gold": q["gold"], "answer": ans, **verdict})
        print(f"  Q{i+1}: {verdict['score']}", flush=True)

    scored = [r["score"] for r in results]
    summary = {
        "policy": "codex_real",
        "recall_pct": round(100 * sum(scored) / (2 * len(scored)), 1),
        "scores": scored,
        "read_seconds": round(read_s),
        **stats,
        "rollout": rollout,
    }
    Path(OUT).write_text(json.dumps({"summary": summary, "results": results}, indent=1),
                         encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
