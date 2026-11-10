#!/usr/bin/env python3
"""Battery runner + readout for the budgeted-skills-index A/B.

    SIB_TREE=<checkout> SIB_LIBRARY=<library.json> SIB_TASKS=<tasks.json> SIB_PYTHON=<hermes env python> \\
    python3 run.py <model> <reps> [--parallel=N] [--only=task1,task2]
    python3 run.py --report

Inference: SIB_INFERENCE_URL + SIB_INFERENCE_KEY, or (default) the Nous Portal agent key read fresh from
$HERMES_HOME/auth.json for every cell. Read-only: the harness never refreshes or copies the OAuth grant.
Resume-safe per (model, task, arm, rep); errored cells score 0 and stay in the denominator.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SIB_RESULTS") or HERE / "results")
ARMS = ("full", "tiered")
KINDS = ("scrubbed", "kept", "none")


def _tasks():
    return json.loads(Path(os.environ["SIB_TASKS"]).read_text(encoding="utf-8"))


def _inference_env():
    env = dict(os.environ)
    if not env.get("SIB_INFERENCE_KEY"):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        nous = json.loads((home / "auth.json").read_text())["providers"]["nous"]
        env["SIB_INFERENCE_URL"], env["SIB_INFERENCE_KEY"] = nous["inference_base_url"], nous["agent_key"]
    return env


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    py = os.environ.get("SIB_PYTHON") or sys.executable
    try:
        p = subprocess.run([py, str(HERE / "worker.py"), arm, model, task, str(rep), str(out)],
                           capture_output=True, text=True, timeout=300, env=_inference_env(), check=False)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line if p.returncode == 0 else f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "rep": rep, "score": 0.0,
                                   "kind": _tasks()[task]["kind"], "error": "wall timeout"}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)


def _mean(cells, key):
    vals = [r[key] for r in cells if r.get(key) is not None]
    mean = sum(vals) / len(vals) if vals else None
    return "-" if mean is None else f"{mean:,.0f}" if mean >= 100 else f"{mean:.3g}"


def _row(label, by_arm):
    line = [label] + [f"{sum(r['score'] for r in by_arm[a]):g}/{len(by_arm[a])}" for a in ARMS]
    line += [str(sum(len(r.get("wrong_loads") or []) for r in by_arm[a])) for a in ARMS]
    for key in ("api_calls", "prompt_tokens", "cost_usd"):
        line += [_mean(by_arm[a], key) for a in ARMS]
    return "| " + " | ".join(line) + " |"


def report():
    rows = [json.loads(p.read_text()) for p in RESULTS.rglob("*.json")]
    head = ("| {} | right full | right tiered | wrong loads full | wrong loads tiered | calls full | calls tiered | "
            "prompt tok full | prompt tok tiered | $ full | $ tiered |\n|" + "---|" * 11)
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        print(f"\n## {model}\n\n" + head.format("kind"))
        for kind in KINDS:
            print(_row(kind, {a: [r for r in mine if r["kind"] == kind and r["arm"] == a] for a in ARMS}))
        print(_row("**all**", {a: [r for r in mine if r["arm"] == a] for a in ARMS}))
        print("\n" + head.format("task"))
        for task in sorted({r["task"] for r in mine}, key=lambda t: (KINDS.index(next(r["kind"] for r in mine if r["task"] == t)), t)):
            print(_row(task, {a: [r for r in mine if r["task"] == task and r["arm"] == a] for a in ARMS}))
        for arm in ARMS:
            cells = [r for r in mine if r["arm"] == arm]
            spent = sum(r.get("cost_usd") or 0 for r in cells)
            print(f"\n{arm}: {len(cells)} cells, ${spent:.2f}, names-only skills in the index: "
                  f"{sorted({r.get('names_only_count') for r in cells if r.get('names_only_count') is not None})}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)
    model, reps = sys.argv[1], int(sys.argv[2])
    par = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--parallel=")), "6"))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), list(_tasks()))
    jobs = [(model, t, arm, rep) for rep in range(1, reps + 1) for t in only for arm in ARMS]
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda j: cell(*j), jobs))
    report()
