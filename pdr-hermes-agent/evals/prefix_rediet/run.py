#!/usr/bin/env python3
"""Battery runner + readout for the fixed-prefix re-diet A/B.

    PRD_BASE_TREE=<checkout of the base SHA> PRD_PR_TREE=<checkout of the PR SHA> PRD_PYTHON=<hermes env python> \\
    python3 run.py <model> <reps> [--parallel=N] [--only=task1,task2]

Inference: PRD_INFERENCE_URL + PRD_INFERENCE_KEY, or (default) the Nous Portal agent key read fresh from
$HERMES_HOME/auth.json for every cell. Read-only: the harness never refreshes or copies the OAuth grant.
    python3 run.py --report

Resume-safe per (model, task, arm, rep); errored cells score 0 and stay in the denominator.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("PRD_RESULTS") or HERE / "results")
TASKS = ["vault_saved_login", "vault_first_login", "term_slow_tests", "term_dev_server", "term_wait_then_write",
         "search_cron_backup", "search_recall_dose"]
TIMEOUT = {"term_slow_tests": 600}


def _inference_env():
    env = dict(os.environ)
    if not env.get("PRD_INFERENCE_KEY"):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        nous = json.loads((home / "auth.json").read_text())["providers"]["nous"]
        env["PRD_INFERENCE_URL"], env["PRD_INFERENCE_KEY"] = nous["inference_base_url"], nous["agent_key"]
    return env


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    py = os.environ.get("PRD_PYTHON") or sys.executable
    try:
        p = subprocess.run([py, str(HERE / "worker.py"), arm, model, task, str(rep), str(out)],
                           capture_output=True, text=True, timeout=TIMEOUT.get(task, 420), env=_inference_env(), check=False)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line if p.returncode == 0 else f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "rep": rep, "score": 0.0,
                                   "error": "wall timeout", "api_turns": None, "total_tokens": None}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)


def report():
    rows = [json.loads(p.read_text()) for p in RESULTS.rglob("*.json") if not p.name.endswith(".transcript.json")]
    for model in sorted({r["model"] for r in rows}):
        print(f"\n## {model}\n\n| task | base | pr | base turns | pr turns | base tok | pr tok | pr bridge calls |\n|---|---|---|---|---|---|---|---|")
        tot = {"base": [0, 0], "pr": [0, 0]}
        for task in TASKS:
            cells = {arm: [r for r in rows if r["model"] == model and r["task"] == task and r["arm"] == arm] for arm in ("base", "pr")}
            if not any(cells.values()):
                continue

            def stat(arm, key):
                vals = [r[key] for r in cells[arm] if r.get(key) is not None and not r.get("error")]
                return f"{sum(vals) / len(vals):,.0f}" if vals else "-"

            line = [task]
            for arm in ("base", "pr"):
                s = sum(r["score"] for r in cells[arm])
                tot[arm][0] += s
                tot[arm][1] += len(cells[arm])
                line.append(f"{s:g}/{len(cells[arm])}")
            line += [stat("base", "api_turns"), stat("pr", "api_turns"), stat("base", "total_tokens"),
                     stat("pr", "total_tokens"), stat("pr", "bridge_calls")]
            print("| " + " | ".join(line) + " |")
        print(f"| **total** | **{tot['base'][0]:g}/{tot['base'][1]}** | **{tot['pr'][0]:g}/{tot['pr'][1]}** | | | | | |")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)
    model, reps = sys.argv[1], int(sys.argv[2])
    par = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--parallel=")), "6"))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), TASKS)
    jobs = [(model, t, arm, rep) for rep in range(1, reps + 1) for t in only for arm in ("base", "pr")]
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda j: cell(*j), jobs))
    report()
