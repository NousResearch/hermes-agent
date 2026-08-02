#!/usr/bin/env python3
"""
Autoreason: Self-Refinement That Knows When to Stop

A standalone CLI tool implementing the autoreason pipeline from
NousResearch/autoreason (SHL0MS, 2026).

Usage:
    python run_autoreason.py --task "Your task prompt here"
    python run_autoreason.py --task "Write a go-to-market strategy" --model openrouter/anthropic/claude-sonnet-4-20250514 --judges 3
    python run_autoreason.py --file task_prompt.txt --output ./my_results

Requires: litellm, and API keys in environment (e.g. ANTHROPIC_API_KEY, OPENROUTER_API_KEY)
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path


# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"
DEFAULT_AUTHOR_TEMP = 0.8
DEFAULT_JUDGE_TEMP = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_NUM_JUDGES = 3
DEFAULT_CONVERGENCE = 2  # A must win this many consecutive times
DEFAULT_MAX_PASSES = 30
DEFAULT_OUTPUT_DIR = "autoreason_output"


# ── Prompt templates (from paper experiments) ─────────────────────────────
AUTHOR_SYSTEM = (
    "You are a senior consultant producing professional deliverables. "
    "Be specific, concrete, and practical. Avoid generic advice. "
    "Tailor everything to the constraints stated in the task."
)

CRITIC_SYSTEM = (
    "You are a critical reviewer. Your only job is to find real problems. "
    "Be specific and concrete. Do not suggest fixes."
)

AUTHOR_B_SYSTEM = (
    "You are a senior consultant revising a proposal based on specific criticisms. "
    "Address each valid criticism directly. Do not make changes that aren't "
    "motivated by an identified problem."
)

SYNTHESIZER_SYSTEM = (
    "You are a senior consultant. You are given two versions as equal inputs. "
    "Take the strongest elements from each and produce a coherent synthesis. "
    "This is not a compromise — pick the best answer per dimension."
)

JUDGE_SYSTEM = (
    "You are an independent evaluator. You have no authorship stake in any "
    "version. Evaluate which version best accomplishes the original task."
)

GENERATE_A = "{task_prompt}\n\nProduce a complete, detailed proposal."

CRITIC_PROMPT = """Here is a proposal:

---
{version_a}
---

Find real problems with this proposal. Focus on:
- Things that won't work as described
- Complexity that doesn't pay for itself
- Assumptions that are wrong
- Missing pieces that block the design

Do NOT propose fixes. Just the problems."""

AUTHOR_B_PROMPT = """ORIGINAL TASK:
---
{task_prompt}
---

Here is a proposal and the problems identified with it.

CURRENT PROPOSAL:
---
{version_a}
---

PROBLEMS FOUND:
---
{critic}
---

Revise the proposal to address these problems.
For each change, state which problem it fixes.
Do not make changes that aren't motivated by an identified problem."""

SYNTHESIZER_PROMPT = """ORIGINAL TASK:
---
{task_prompt}
---

Here are two versions of a proposal. Treat them as equal inputs.

VERSION {vx_label}:
---
{version_x}
---

VERSION {vy_label}:
---
{version_y}
---

Produce a synthesis that keeps the strongest elements from both.
Pick the best version of each section and make them cohere."""

JUDGE_RANK_3_PROMPT = """ORIGINAL TASK:
---
{task_prompt}
---

Three proposals have been produced independently. Evaluate how well each accomplishes the stated task.

{judge_proposals}

For each proposal, state what it gets right and what it gets wrong.
Then rank all three from best to worst:

RANKING: [best], [second], [worst]

Where each slot is 1, 2, or 3."""


# ── LLM wrapper (paper's retry/rate-limit logic) ──────────────────────────
async def call_llm(system: str, user: str, model: str, temperature: float,
                   max_tokens: int, max_retries: int = 8) -> str:
    """Call LLM with exponential backoff rate-limit handling (from paper code)."""
    import litellm
    litellm.suppress_debug_info = True

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "overloaded" in err or "529" in err:
                wait = min((2 ** attempt) * 5, 120)
                print(f"      [Rate limited, retry {attempt+1}/{max_retries} in {wait}s]")
                await asyncio.sleep(wait)
            else:
                if attempt < max_retries - 1:
                    wait = 10
                    print(f"      [Error: {str(e)[:80]}, retry in {wait}s]")
                    await asyncio.sleep(wait)
                else:
                    raise
    raise RuntimeError(f"Failed after {max_retries} retries")


# ── Judging helpers ───────────────────────────────────────────────────────
def randomize_for_judge(va: str, vb: str, vab: str) -> tuple[str, dict]:
    """Randomly shuffle A, B, AB so judges see them blind. Returns (formatted, order_map)."""
    versions = [("A", va), ("B", vb), ("AB", vab)]
    random.shuffle(versions)
    order = {}
    parts = []
    for i, (label, content) in enumerate(versions, 1):
        order[str(i)] = label
        parts.append(f"PROPOSAL {i}:\n---\n{content}\n---")
    return "\n\n".join(parts), order


def parse_ranking(text: str, valid_chars: str = "123") -> list[str] | None:
    """Parse 'RANKING: [best], [second], [worst]' from judge output."""
    for line in reversed(text.split("\n")):
        line = line.strip().strip("*").strip().lstrip("#").strip()
        if line.upper().startswith("RANKING:"):
            raw = line.split(":", 1)[1].strip()
            items = [c for c in raw if c in valid_chars]
            if len(items) >= 2:
                return items
    return None


def aggregate_rankings(rankings: list[list[str] | None],
                       labels: list[str],
                       tiebreak_winner: str | None = None) -> tuple[str, dict, int]:
    """Borda count: each judge allocates (n-pos) points. Returns (winner, scores, valid_count)."""
    scores = {l: 0 for l in labels}
    n = len(labels)
    valid = [r for r in rankings if r is not None]
    for ranking in valid:
        for pos, label in enumerate(ranking):
            if label in scores and pos < n:
                scores[label] += (n - pos)
    if tiebreak_winner:
        priority = {l: (0 if l == tiebreak_winner else i+1) for i, l in enumerate(labels)}
    else:
        priority = {l: i for i, l in enumerate(labels)}
    ranked = sorted(scores.keys(), key=lambda k: (-scores[k], priority[k]))
    return ranked[0], scores, len(valid)


# ── Autoreason pass ───────────────────────────────────────────────────────
async def run_autoreason_pass(task_prompt: str, current_a: str, pass_num: int,
                               pass_dir: Path, model: str, author_temp: float,
                               judge_temp: float, max_tokens: int,
                               num_judges: int) -> tuple[str, str, dict]:
    """Single autoreason iteration: Critic → Author B → Synthesizer → Judge panel."""
    pass_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Save incumbent
    (pass_dir / "version_a.md").write_text(current_a)

    # 1. Critic
    print(f"    [Pass {pass_num}] Critic...", end=" ", flush=True)
    critic = await call_llm(CRITIC_SYSTEM, CRITIC_PROMPT.format(
        version_a=current_a), model, author_temp, max_tokens)
    (pass_dir / "critic.md").write_text(critic)
    print(f"ok")

    # 2. Author B (revision)
    print(f"    [Pass {pass_num}] Author B...", end=" ", flush=True)
    vb = await call_llm(AUTHOR_B_SYSTEM, AUTHOR_B_PROMPT.format(
        task_prompt=task_prompt, version_a=current_a, critic=critic),
        model, author_temp, max_tokens)
    (pass_dir / "version_b.md").write_text(vb)
    print(f"ok")

    # 3. Synthesizer (AB) — randomize input order to avoid position bias
    print(f"    [Pass {pass_num}] Synthesizer...", end=" ", flush=True)
    if random.random() < 0.5:
        vx_label, vx, vy_label, vy = "X", current_a, "Y", vb
    else:
        vx_label, vx, vy_label, vy = "X", vb, "Y", current_a
    vab = await call_llm(SYNTHESIZER_SYSTEM, SYNTHESIZER_PROMPT.format(
        task_prompt=task_prompt, vx_label=vx_label, version_x=vx,
        vy_label=vy_label, version_y=vy),
        model, author_temp, max_tokens)
    (pass_dir / "version_ab.md").write_text(vab)
    print(f"ok")

    # 4. Judge panel
    print(f"    [Pass {pass_num}] Judges ({num_judges}x)...", end=" ", flush=True)
    jtasks, jorders = [], []
    for _ in range(num_judges):
        proposals, order = randomize_for_judge(current_a, vb, vab)
        jorders.append(order)
        jtasks.append(call_llm(
            JUDGE_SYSTEM,
            JUDGE_RANK_3_PROMPT.format(task_prompt=task_prompt, judge_proposals=proposals),
            model, judge_temp, max_tokens))

    jresps = await asyncio.gather(*jtasks, return_exceptions=True)
    rankings, jdetails = [], []
    for j, (resp, order) in enumerate(zip(jresps, jorders)):
        if isinstance(resp, Exception):
            rankings.append(None)
            jdetails.append({"error": str(resp)})
        else:
            raw_ranking = parse_ranking(resp, "123")
            mapped = [order.get(r, r) for r in raw_ranking] if raw_ranking else None
            rankings.append(mapped)
            jdetails.append({"ranking": mapped, "order": order})

    winner, scores, valid = aggregate_rankings(rankings, ["A", "B", "AB"], tiebreak_winner="A")
    elapsed = time.time() - t0

    vmap = {"A": current_a, "B": vb, "AB": vab}
    result = {
        "pass": pass_num, "winner": winner, "scores": scores,
        "valid_judges": valid, "elapsed": round(elapsed, 1),
        "judge_details": jdetails,
    }
    (pass_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Winner: {winner} (A={scores.get('A',0)}, B={scores.get('B',0)}, AB={scores.get('AB',0)}) [{elapsed:.0f}s]")
    return winner, vmap[winner], result


# ── Full autoreason run ───────────────────────────────────────────────────
async def run_autoreason(task_prompt: str, output_dir: Path,
                          model: str = DEFAULT_MODEL,
                          author_temp: float = DEFAULT_AUTHOR_TEMP,
                          judge_temp: float = DEFAULT_JUDGE_TEMP,
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          num_judges: int = DEFAULT_NUM_JUDGES,
                          convergence: int = DEFAULT_CONVERGENCE,
                          max_passes: int = DEFAULT_MAX_PASSES) -> tuple[str, list[dict]]:
    """Run the full autoreason pipeline until convergence or max passes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Initial generation
    print(f"\n  Generating initial version A...", end=" ", flush=True)
    current_a = await call_llm(AUTHOR_SYSTEM, GENERATE_A.format(
        task_prompt=task_prompt), model, author_temp, max_tokens)
    (output_dir / "initial_a.md").write_text(current_a)
    print(f"{len(current_a.split())} words\n")

    # Phase 2: Iterative refinement
    streak = 0
    history = []
    for p in range(1, max_passes + 1):
        pass_dir = output_dir / f"pass_{p:02d}"
        winner, winner_text, result = await run_autoreason_pass(
            task_prompt, current_a, p, pass_dir, model,
            author_temp, judge_temp, max_tokens, num_judges)

        entry = {
            "pass": p, "winner": winner,
            "scores": result.get("scores", {}),
            "words": len(winner_text.split()),
            "elapsed": result.get("elapsed", 0),
        }
        history.append(entry)

        if winner == "A":
            streak += 1
        else:
            streak = 0
            current_a = winner_text
            (output_dir / f"incumbent_after_{p:02d}.md").write_text(current_a)

        if streak >= convergence:
            print(f"\n  ✔ Converged at pass {p} (A won {convergence}x consecutively)")
            break

    # Final output
    (output_dir / "final_output.md").write_text(current_a)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    trajectory = " → ".join(h["winner"] for h in history)
    print(f"\n  Final: {len(current_a.split())} words")
    print(f"  Trajectory: {trajectory}")
    return current_a, history


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Autoreason: Self-Refinement That Knows When to Stop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task "Write a go-to-market strategy for a developer tool"
  %(prog)s --file task_prompt.txt --model openrouter/anthropic/claude-sonnet-4-20250514
  %(prog)s --task "Design an API" --judges 7 --max-passes 20 --output ./results

Based on NousResearch/autoreason by SHL0MS (2026).
        """,
    )
    parser.add_argument("--task", type=str, help="Task prompt (inline)")
    parser.add_argument("--file", type=str, help="Task prompt file path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--author-temp", type=float, default=DEFAULT_AUTHOR_TEMP,
                        help=f"Author temperature (default: {DEFAULT_AUTHOR_TEMP})")
    parser.add_argument("--judge-temp", type=float, default=DEFAULT_JUDGE_TEMP,
                        help=f"Judge temperature (default: {DEFAULT_JUDGE_TEMP})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per call (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--judges", type=int, default=DEFAULT_NUM_JUDGES,
                        help=f"Number of judges per pass (default: {DEFAULT_NUM_JUDGES})")
    parser.add_argument("--convergence", type=int, default=DEFAULT_CONVERGENCE,
                        help=f"Consecutive A wins to converge (default: {DEFAULT_CONVERGENCE})")
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES,
                        help=f"Max iterations (default: {DEFAULT_MAX_PASSES})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    if not args.task and not args.file:
        parser.error("Either --task or --file is required")

    task_prompt = args.task if args.task else Path(args.file).read_text().strip()

    output_dir = Path(args.output)
    print(f"\n{'='*60}")
    print(f"  Autoreason: Self-Refinement That Knows When to Stop")
    print(f"  Model: {args.model}")
    print(f"  Judges: {args.judges} per pass, convergence: {args.convergence}x A win")
    print(f"  Max passes: {args.max_passes}")
    print(f"  Output: {output_dir.absolute()}")
    print(f"{'='*60}\n")

    start = time.time()
    asyncio.run(run_autoreason(
        task_prompt=task_prompt,
        output_dir=output_dir,
        model=args.model,
        author_temp=args.author_temp,
        judge_temp=args.judge_temp,
        max_tokens=args.max_tokens,
        num_judges=args.judges,
        convergence=args.convergence,
        max_passes=args.max_passes,
    ))
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} minutes")
    print(f"  Results: {output_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
