"""Benchmark harness for Level 2 experience learning.

Measures the four things that can be measured deterministically, without an
LLM and without spending API credits:

  A. Retrieval quality  — precision@k, recall, false-positive rate on a
     labeled corpus of task/query pairs.
  B. Latency            — write and retrieval cost at realistic store sizes.
  C. Context overhead   — characters and estimated tokens added per turn.
  D. Closed-loop A/B    — a SIMULATED agent policy replayed twice over the
     same workload, once without retrieval (baseline) and once with it
     (level 2), reporting task success / failure / recovery / correction
     rates and iterations-to-solution.

What phase D is and is not
--------------------------
Phase D replaces the model with a deterministic policy: pick the tool a
retrieved experience recommends, else try tools in a fixed default order until
one works. It measures whether retrieval carries *actionable signal* through
the whole record→store→retrieve→render path. It does NOT measure a real
model's success rate — that needs live provider calls and is reported as
NOT AVAILABLE below.

Run::

    venv/Scripts/python.exe scripts/bench_experience.py
    venv/Scripts/python.exe scripts/bench_experience.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.experience import (  # noqa: E402
    Experience,
    format_experience_block,
    normalize_task,
    rank_rows,
    task_fingerprint,
)
from hermes_state import SessionDB  # noqa: E402

RNG = random.Random(20260819)


def _est_tokens(text: str) -> int:
    try:
        from agent.model_metadata import estimate_tokens_rough

        return estimate_tokens_rough(text)
    except Exception:
        return (len(text) + 3) // 4


def _fresh_db(tmp: Path, name: str) -> SessionDB:
    return SessionDB(db_path=tmp / f"{name}-{uuid.uuid4().hex[:8]}.db")


def _store(db: SessionDB, task: str, outcome: str, *, tools=(), recovery="",
           workspace="/bench", verification="", session_id="bench") -> str:
    exp = Experience(
        task=task,
        task_norm=normalize_task(task),
        task_hash=task_fingerprint(normalize_task(task)),
        outcome=outcome,
        strategy="used " + " → ".join(tools) if tools else "",
        tools=list(tools),
        recovery=recovery,
        session_id=session_id,
        cwd=workspace,
        workspace=workspace,
        verification=verification,
    )
    return db.record_experience(exp.to_row()) or ""


# ── A. Retrieval quality ────────────────────────────────────────────────

# (stored task, paraphrase that SHOULD retrieve it)
RELEVANT_PAIRS: List[Tuple[str, str]] = [
    ("fix the failing build in the payment module",
     "the payment module build is broken again"),
    ("sửa lỗi timeout khi gọi API thanh toán",
     "api thanh toan bi timeout, sua giup"),
    ("add pagination to the users endpoint",
     "users endpoint needs pagination"),
    ("migrate the sessions table to the new schema",
     "run the sessions table schema migration"),
    ("deploy the gateway service to staging",
     "staging deploy of the gateway service"),
    ("write unit tests for the retry helper",
     "retry helper needs unit tests"),
    ("debug the memory leak in the browser tool",
     "browser tool leaking memory"),
    ("update the OAuth callback URL for Slack",
     "slack oauth callback url update"),
    ("compress the session transcript before persisting",
     "transcript compression before persist"),
    ("rotate the leaked telegram bot token",
     "telegram bot token rotation"),
]

# Queries that must retrieve NOTHING from the corpus above.
IRRELEVANT_QUERIES = [
    "what is the weather in Paris tomorrow",
    "recommend a book about ancient Rome",
    "convert 42 kilometres to miles",
    "who won the 1998 world cup",
    "explain the difference between a monad and a functor",
    "draft a birthday message for my sister",
    "what time does the pharmacy close",
    "translate hello world into Japanese",
]


def bench_retrieval_quality(tmp: Path) -> Dict[str, Any]:
    db = _fresh_db(tmp, "quality")
    for task, _ in RELEVANT_PAIRS:
        _store(db, task, "success", tools=["read_file", "patch"])
    # Distractors: unrelated stored tasks that must not be returned.
    for i in range(40):
        _store(db, f"unrelated background chore number {i} about logs and cleanup",
               "success", tools=["terminal"])

    cands = db.fetch_experience_candidates(workspace="/bench")
    hits = misses = 0
    ranks: List[int] = []
    for task, query in RELEVANT_PAIRS:
        top = rank_rows(cands, query, limit=3)
        ids = [r["task"] for r in top]
        if task in ids:
            hits += 1
            ranks.append(ids.index(task) + 1)
        else:
            misses += 1

    false_positives = sum(1 for q in IRRELEVANT_QUERIES if rank_rows(cands, q, limit=3))

    # Precision@3: of everything returned for a relevant query, how much was
    # the right row (the corpus has exactly one correct answer per query).
    returned = sum(len(rank_rows(cands, q, limit=3)) for _, q in RELEVANT_PAIRS)
    db.close()
    return {
        "corpus_rows": len(RELEVANT_PAIRS) + 40,
        "relevant_queries": len(RELEVANT_PAIRS),
        "recall_at_3": round(hits / len(RELEVANT_PAIRS), 4),
        "misses": misses,
        "precision_at_3": round(hits / returned, 4) if returned else 0.0,
        "mean_rank_of_correct_hit": round(statistics.mean(ranks), 2) if ranks else None,
        "irrelevant_queries": len(IRRELEVANT_QUERIES),
        "false_positive_rate": round(false_positives / len(IRRELEVANT_QUERIES), 4),
    }


# ── B. Latency ──────────────────────────────────────────────────────────


def bench_latency(tmp: Path, sizes=(100, 500, 2000)) -> List[Dict[str, Any]]:
    out = []
    for n in sizes:
        db = _fresh_db(tmp, f"lat{n}")
        write_ms: List[float] = []
        for i in range(n):
            t0 = time.perf_counter()
            _store(db, f"benchmark task {i} touching module {i % 37} and helper {i % 11}",
                   "success" if i % 4 else "failure", tools=["read_file", "patch"])
            write_ms.append((time.perf_counter() - t0) * 1000)

        read_ms: List[float] = []
        for i in range(60):
            q = f"module {i % 37} helper {i % 11} benchmark task"
            t0 = time.perf_counter()
            rows = db.fetch_experience_candidates(workspace="/bench")
            top = rank_rows(rows, q, limit=3)
            format_experience_block(top)
            read_ms.append((time.perf_counter() - t0) * 1000)
        db.close()
        out.append({
            "rows": n,
            "write_p50_ms": round(statistics.median(write_ms), 3),
            "write_p95_ms": round(sorted(write_ms)[int(len(write_ms) * 0.95)], 3),
            "retrieve_p50_ms": round(statistics.median(read_ms), 3),
            "retrieve_p95_ms": round(sorted(read_ms)[int(len(read_ms) * 0.95)], 3),
            "retrieve_max_ms": round(max(read_ms), 3),
        })
    return out


# ── C. Context overhead ─────────────────────────────────────────────────


def bench_context_overhead(tmp: Path) -> Dict[str, Any]:
    db = _fresh_db(tmp, "overhead")
    for task, _ in RELEVANT_PAIRS:
        _store(db, task, "partial", tools=["read_file", "patch", "terminal"],
               recovery="retried after failure and succeeded: patch")
    cands = db.fetch_experience_candidates(workspace="/bench")
    sizes, empties = [], 0
    for _, query in RELEVANT_PAIRS:
        block = format_experience_block(rank_rows(cands, query, limit=3))
        if not block:
            empties += 1
        sizes.append(len(block))
    for query in IRRELEVANT_QUERIES:
        block = format_experience_block(rank_rows(cands, query, limit=3))
        sizes.append(len(block))
        if not block:
            empties += 1
    db.close()
    non_empty = [s for s in sizes if s]
    return {
        "turns_measured": len(sizes),
        "turns_with_no_injection": empties,
        "chars_mean_when_injected": round(statistics.mean(non_empty), 1) if non_empty else 0,
        "chars_max": max(sizes) if sizes else 0,
        "tokens_mean_when_injected": (
            round(statistics.mean(_est_tokens("x" * int(s)) for s in non_empty), 1)
            if non_empty else 0
        ),
        "tokens_max": _est_tokens("x" * max(sizes)) if sizes else 0,
        "amortized_tokens_per_turn": (
            round(sum(_est_tokens("x" * s) for s in sizes) / len(sizes), 1) if sizes else 0
        ),
    }


# ── D. Simulated closed-loop A/B ────────────────────────────────────

# Each scenario has exactly one tool that solves it. The agent gets a small
# iteration budget and must find that tool.
#
# Without experience, tool choice is unguided: the policy tries tools in a
# random order (the honest stand-in for "the model guesses"). With experience,
# a tool a prior turn recorded as working is tried FIRST.
#
# The workload repeats scenarios, so the interesting number is not the overall
# rate but the FIRST-ENCOUNTER vs REPEAT-ENCOUNTER split: that is where
# learning either shows up or does not.
SCENARIOS: List[Dict[str, Any]] = [
    {"task": "fix the failing build in the payment module", "solver": "patch"},
    {"task": "sửa lỗi timeout khi gọi API thanh toán", "solver": "terminal"},
    {"task": "add pagination to the users endpoint", "solver": "write_file"},
    {"task": "migrate the sessions table to the new schema", "solver": "terminal"},
    {"task": "deploy the gateway service to staging", "solver": "browser"},
    {"task": "write unit tests for the retry helper", "solver": "write_file"},
    {"task": "debug the memory leak in the browser tool", "solver": "browser"},
    {"task": "update the OAuth callback URL for Slack", "solver": "web_search"},
    {"task": "compress the session transcript before persisting", "solver": "patch"},
    {"task": "rotate the leaked telegram bot token", "solver": "terminal"},
]

TOOLBOX = ["read_file", "search_files", "web_search", "browser",
           "write_file", "patch", "terminal"]
MAX_ATTEMPTS = 3          # iteration budget per task
CORRECTION_PROB = 0.4     # chance the user pushes back after a failed task


def _paraphrase(task: str, rng: random.Random) -> str:
    """A re-ask of the same task in different words/order."""
    words = task.split()
    rng.shuffle(words)
    return " ".join(words)


def _simulate(db: SessionDB, *, use_retrieval: bool, episodes: int,
              seed: int) -> Dict[str, Any]:
    """Replay the workload under one policy and return outcome rates.

    Both arms are driven by the SAME seed, so they see an identical task order
    and identical unguided exploration order. The only difference between the
    arms is whether retrieval reorders the toolbox.
    """
    rng = random.Random(seed)
    seen_before: Dict[str, int] = {}
    tally = {
        "tasks": 0, "success": 0, "failure": 0, "recovered": 0,
        "corrections": 0, "attempts": 0, "retrieval_hits": 0,
        "first_seen": 0, "first_success": 0,
        "repeat_seen": 0, "repeat_success": 0,
    }
    curve: List[int] = []

    for ep in range(episodes):
        scen = SCENARIOS[rng.randrange(len(SCENARIOS))]
        key = scen["task"]
        task = key if rng.random() < 0.5 else _paraphrase(key, rng)
        solver = scen["solver"]
        is_repeat = key in seen_before
        seen_before[key] = seen_before.get(key, 0) + 1
        tally["tasks"] += 1
        tally["repeat_seen" if is_repeat else "first_seen"] += 1

        # Unguided exploration order — identical across arms for this episode.
        order = TOOLBOX[:]
        rng.shuffle(order)

        recommended = None
        if use_retrieval:
            cands = db.fetch_experience_candidates(workspace="/bench")
            top = rank_rows(cands, task, limit=1)
            if top:
                tally["retrieval_hits"] += 1
                try:
                    tools = json.loads(top[0].get("tools") or "[]")
                except Exception:
                    tools = []
                # Reuse only what a prior turn recorded as actually working:
                # the LAST tool of a successful run is the one that solved it.
                if tools and top[0].get("outcome") in ("success", "partial"):
                    recommended = tools[-1]
        if recommended in order:
            order.remove(recommended)
            order.insert(0, recommended)

        used: List[str] = []
        solved = False
        for tool in order[:MAX_ATTEMPTS]:
            used.append(tool)
            tally["attempts"] += 1
            if tool == solver:
                solved = True
                break

        recovery = ""
        if solved and len(used) > 1:
            recovery = "switched away from failing " + used[0] + " to " + solver
            tally["recovered"] += 1
        outcome = "success" if solved else "failure"
        tally["success" if solved else "failure"] += 1
        if solved:
            tally["repeat_success" if is_repeat else "first_success"] += 1
        curve.append(1 if solved else 0)

        # Draw the correction die UNCONDITIONALLY so both arms consume the
        # same rng stream regardless of outcome. Without this the streams
        # diverge at the first differing episode and the arms stop being
        # paired — which would silently invalidate the comparison.
        corr_draw = rng.random()

        exp_id = _store(db, task, outcome, tools=used, recovery=recovery,
                        session_id="ep" + str(ep))
        if not solved and corr_draw < CORRECTION_PROB and exp_id:
            db.record_experience_correction(exp_id, "still not working")
            tally["corrections"] += 1

    n = max(1, tally["tasks"])
    third = max(1, n // 3)
    return {
        "tasks": tally["tasks"],
        "success_rate": round(tally["success"] / n, 4),
        "failure_rate": round(tally["failure"] / n, 4),
        "first_encounter_success_rate": round(
            tally["first_success"] / max(1, tally["first_seen"]), 4),
        "repeat_encounter_success_rate": round(
            tally["repeat_success"] / max(1, tally["repeat_seen"]), 4),
        "retry_needed_rate": round(tally["recovered"] / max(1, tally["success"]), 4),
        "user_correction_rate": round(tally["corrections"] / n, 4),
        "mean_tool_attempts": round(tally["attempts"] / n, 3),
        "retrieval_hit_rate": round(tally["retrieval_hits"] / n, 4),
        "success_rate_first_third": round(sum(curve[:third]) / third, 4),
        "success_rate_last_third": round(sum(curve[-third:]) / third, 4),
    }


def bench_ab(tmp: Path, episodes: int = 300, seed: int = 4242) -> Dict[str, Any]:
    baseline_db = _fresh_db(tmp, "baseline")
    baseline = _simulate(baseline_db, use_retrieval=False, episodes=episodes, seed=seed)
    baseline["store_stats"] = baseline_db.experience_stats()
    baseline_db.close()

    level2_db = _fresh_db(tmp, "level2")
    level2 = _simulate(level2_db, use_retrieval=True, episodes=episodes, seed=seed)
    level2["store_stats"] = level2_db.experience_stats()
    level2_db.close()

    return {
        "episodes": episodes,
        "seed": seed,
        "baseline": baseline,
        "level2": level2,
    }


# ── Report ──────────────────────────────────────────────────────────────

NOT_AVAILABLE = {
    "real_model_task_success_rate":
        "NOT AVAILABLE — requires live provider calls over a labeled task set; "
        "no such harness exists in this repo and fabricating numbers is not an option.",
    "end_to_end_turn_latency":
        "NOT AVAILABLE — dominated by provider latency, which varies per call. "
        "The retrieval component is measured in section B.",
    "real_token_cost_delta":
        "NOT AVAILABLE as billed tokens — the injected-context size is measured "
        "exactly in section C; converting it to cost needs a live run.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write the full report as JSON")
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        report = {
            "A_retrieval_quality": bench_retrieval_quality(tmp),
            "B_latency": bench_latency(tmp),
            "C_context_overhead": bench_context_overhead(tmp),
            "D_simulated_ab": bench_ab(tmp, episodes=args.episodes),
            "NOT_AVAILABLE": NOT_AVAILABLE,
        }

    print("=" * 72)
    print("HERMES LEVEL 2 EXPERIENCE LEARNING — BENCHMARK")
    print("=" * 72)

    a = report["A_retrieval_quality"]
    print("\n[A] RETRIEVAL QUALITY")
    print(f"  corpus rows              : {a['corpus_rows']}")
    print(f"  recall@3                 : {a['recall_at_3']:.1%} "
          f"({a['relevant_queries'] - a['misses']}/{a['relevant_queries']})")
    print(f"  precision@3              : {a['precision_at_3']:.1%}")
    print(f"  mean rank of correct hit : {a['mean_rank_of_correct_hit']}")
    print(f"  false-positive rate      : {a['false_positive_rate']:.1%} "
          f"(on {a['irrelevant_queries']} unrelated queries)")

    print("\n[B] LATENCY")
    print(f"  {'rows':>6} {'write p50':>10} {'write p95':>10} "
          f"{'read p50':>9} {'read p95':>9} {'read max':>9}")
    for r in report["B_latency"]:
        print(f"  {r['rows']:>6} {r['write_p50_ms']:>9.2f}m {r['write_p95_ms']:>9.2f}m "
              f"{r['retrieve_p50_ms']:>8.2f}m {r['retrieve_p95_ms']:>8.2f}m "
              f"{r['retrieve_max_ms']:>8.2f}m")

    c = report["C_context_overhead"]
    print("\n[C] CONTEXT OVERHEAD")
    print(f"  turns measured           : {c['turns_measured']}")
    print(f"  turns with no injection  : {c['turns_with_no_injection']}")
    print(f"  chars when injected (avg): {c['chars_mean_when_injected']}")
    print(f"  est. tokens when injected: {c['tokens_mean_when_injected']}")
    print(f"  est. tokens (max)        : {c['tokens_max']}")
    print(f"  amortized tokens / turn  : {c['amortized_tokens_per_turn']}")

    d = report["D_simulated_ab"]
    print(f"\n[D] SIMULATED A/B  (deterministic policy, {d['episodes']} episodes)")
    print(f"  {'metric':<24} {'baseline':>10} {'level 2':>10} {'delta':>10}")
    for key, label in (
        ("success_rate", "task success rate"),
        ("failure_rate", "task failure rate"),
        ("first_encounter_success_rate", "  first encounter"),
        ("repeat_encounter_success_rate", "  repeat encounter"),
        ("success_rate_first_third", "  first third of run"),
        ("success_rate_last_third", "  last third of run"),
        ("retry_needed_rate", "successes needing retry"),
        ("user_correction_rate", "user correction rate"),
        ("mean_tool_attempts", "mean tool attempts"),
        ("retrieval_hit_rate", "retrieval hit rate"),
    ):
        b_v, l_v = d["baseline"][key], d["level2"][key]
        print(f"  {label:<24} {b_v:>10.4f} {l_v:>10.4f} {l_v - b_v:>+10.4f}")

    print("\n[!] NOT AVAILABLE")
    for k, v in NOT_AVAILABLE.items():
        print(f"  {k}:\n      {v}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
