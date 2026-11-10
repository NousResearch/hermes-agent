#!/usr/bin/env python3
"""Battery runner + readout for the deferred-catalog form A/B/C.

    CF_TREE=<checkout with tools.tool_search.listing_servers> CF_PYTHON=<hermes env python> \\
    python3 run.py <model> <reps> [--parallel=N] [--only=task1,task2] [--arms=full,names,summary]
    python3 run.py --report [--md]

Inference: CF_INFERENCE_URL + CF_INFERENCE_KEY, or (default) the Nous Portal agent key read fresh from
$HERMES_HOME/auth.json for every cell. Read-only: the harness never refreshes or copies the OAuth grant.
Resume-safe per (model, task, arm, rep). Errored, timed-out and noise cells score 0 and stay in the denominator;
exit-3 cells (no model turn) are infra, written nowhere, and rerun on the next invocation.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tasks import TASK_ORDER, TASKS

RESULTS = Path(os.environ.get("CF_RESULTS") or HERE / "results")
ARMS = ["full", "names", "summary"]
# USD per token, from the Nous model list on 2026-11-09 (input, cache read, cache write, output).
PRICES = {
    "anthropic/claude-fable-5.2": (8e-6, 0.2e-6, 10e-6, 40e-6),
    "qwen/qwen3.8-27b": (0.15e-6, 0.0375e-6, 0.15e-6, 1.875e-6),
    "openai/gpt-5.6-luna": (0.2e-6, 0.02e-6, 0.25e-6, 1.2e-6),
}


def _inference_env():
    env = dict(os.environ)
    if not env.get("CF_INFERENCE_KEY"):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        nous = json.loads((home / "auth.json").read_text())["providers"]["nous"]
        env["CF_INFERENCE_URL"], env["CF_INFERENCE_KEY"] = nous["inference_base_url"], nous["agent_key"]
    return env


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    py = os.environ.get("CF_PYTHON") or sys.executable
    try:
        p = subprocess.run([py, str(HERE / "worker.py"), arm, model, task, str(rep), str(out)],
                           capture_output=True, text=True, timeout=420, env=_inference_env(), check=False)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line if p.returncode == 0 else f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "kind": TASKS[task]["kind"], "rep": rep,
                                   "score": 0.0, "error": "wall timeout"}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)


def dollars(r):
    """Cost from the session's own usage split; the provider-reported figure wins when present."""
    if r.get("cost_usd"):
        return float(r["cost_usd"])
    p = PRICES.get(r["model"])
    if not p or r.get("input_tokens") is None:
        return 0.0
    return (r["input_tokens"] * p[0] + (r.get("cache_read_tokens") or 0) * p[1]
            + (r.get("cache_write_tokens") or 0) * p[2] + (r.get("output_tokens") or 0) * p[3])


def all_in_tokens(r):
    return sum(r.get(k) or 0 for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"))


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _f(v, spec=",.0f"):
    return "-" if v is None else format(v, spec)


def report():
    rows = [json.loads(p.read_text()) for p in RESULTS.rglob("*.json") if not p.name.endswith(".transcript.json")]
    print(f"cells: {len(rows)}   billed (sum of per-cell dollars): ${sum(dollars(r) for r in rows):.2f}")
    for model in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == model]
        print(f"\n## {model}  (${sum(dollars(r) for r in mr):.2f})\n")
        print("| arm | deferred+continuation: task done | right tool reached | wrong-tool calls | model calls to first "
              "correct use | controls done | distractors done | false-discovery bridge calls | noise/error cells | "
              "tokens per cell (all-in) | fresh+write input per cell | $ per cell | $ per completed task |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            ar = [r for r in mr if r["arm"] == arm]
            d = [r for r in ar if r.get("kind") in ("deferred", "continuation")]
            c = [r for r in ar if r.get("kind") == "control"]
            x = [r for r in ar if r.get("kind") == "distractor"]
            done = sum(r["score"] for r in ar)
            cost = sum(dollars(r) for r in ar)
            print("| " + " | ".join([
                arm, f"{sum(r['score'] for r in d):g}/{len(d)}", f"{sum(1 for r in d if r.get('target_reached'))}/{len(d)}",
                str(sum(r.get("wrong_tool_calls") or 0 for r in d)),
                _f(_mean([r.get("first_correct_model_call") for r in d]), ".2f"),
                f"{sum(r['score'] for r in c):g}/{len(c)}", f"{sum(r['score'] for r in x):g}/{len(x)}",
                str(sum(r.get("false_discovery") or 0 for r in c + x)),
                str(sum(1 for r in ar if r.get("error") or r.get("raw_xml_noise"))),
                _f(_mean([all_in_tokens(r) for r in ar if r.get("input_tokens") is not None])),
                _f(_mean([(r.get("input_tokens") or 0) + (r.get("cache_write_tokens") or 0) for r in ar if r.get("input_tokens") is not None])),
                f"{cost / max(len(ar), 1):.4f}", f"{cost / done:.4f}" if done else "-"]) + " |")
        print("\n| task | kind | " + " | ".join(f"{a} done" for a in ARMS) + " | " + " | ".join(f"{a} reached" for a in ARMS)
              + " | " + " | ".join(f"{a} wrong" for a in ARMS) + " | " + " | ".join(f"{a} calls" for a in ARMS)
              + " | " + " | ".join(f"{a} first-correct" for a in ARMS) + " | " + " | ".join(f"{a} tok" for a in ARMS) + " |")
        print("|---|---|" + "---|" * 18)
        for task in TASK_ORDER:
            cells = {a: [r for r in mr if r["task"] == task and r["arm"] == a] for a in ARMS}
            if not any(cells.values()):
                continue
            has_target = bool(TASKS[task].get("targets"))
            line = [task, TASKS[task]["kind"]]
            line += [f"{sum(r['score'] for r in cells[a]):g}/{len(cells[a])}" for a in ARMS]
            line += [f"{sum(1 for r in cells[a] if r.get('target_reached'))}/{len(cells[a])}" if has_target else
                     f"bridge {sum(r.get('false_discovery') or 0 for r in cells[a])}" for a in ARMS]
            line += [str(sum(r.get("wrong_tool_calls") or 0 for r in cells[a])) for a in ARMS]
            line += [_f(_mean([r.get("model_calls") for r in cells[a]]), ".1f") for a in ARMS]
            line += [_f(_mean([r.get("first_correct_model_call") for r in cells[a]]), ".1f") for a in ARMS]
            line += [_f(_mean([all_in_tokens(r) for r in cells[a] if r.get("input_tokens") is not None])) for a in ARMS]
            print("| " + " | ".join(line) + " |")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)
    model, reps = sys.argv[1], int(sys.argv[2])
    par = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--parallel=")), "6"))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), TASK_ORDER)
    arms = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--arms=")), ARMS)
    jobs = [(model, t, arm, rep) for rep in range(1, reps + 1) for t in only for arm in arms]
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda j: cell(*j), jobs))
    report()
