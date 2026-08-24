"""Local-CDP battery orchestrator: tasks x arms x models x reps.

Resume-safe: completed cells in results.jsonl are skipped, so a killed
battery continues where it left off (same pattern as scripts/toolperf_abeval).

Usage:
    # start a headless Chrome first:
    #   google-chrome --headless=new --remote-debugging-port=9333 \
    #     --user-data-dir=/tmp/bubench-chrome --no-first-run --disable-gpu about:blank
    BUBENCH_BASE_TREE=... BUBENCH_PR_TREE=... BENCH_CDP_URL=http://127.0.0.1:9333 \
        python3 orchestrate.py [--tasks tasks/hard.json] [--models m1,m2] \
                               [--arms base,pr,prns] [--reps 3]
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("BUBENCH_ROOT", os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = {**os.environ}
ENV["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + ENV.get("PATH", "")

parser = argparse.ArgumentParser()
parser.add_argument("--tasks", default=os.path.join(ROOT, "tasks", "hard.json"))
parser.add_argument("--models", default="anthropic/claude-opus-4.8,moonshotai/kimi-k3")
parser.add_argument("--arms", default="base,pr,prns")
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--results", default=os.path.join(ROOT, "results", "results.jsonl"))
parser.add_argument("--run-timeout", type=int, default=1200)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.results), exist_ok=True)
ARMS = args.arms.split(",")
MODELS = args.models.split(",")
TASKS = list(json.load(open(args.tasks, encoding="utf-8")).keys())
REPS = list(range(1, args.reps + 1))

done = set()
if os.path.exists(args.results):
    for line in open(args.results, encoding="utf-8"):
        try:
            r = json.loads(line)
            done.add((r["arm"], r["task"], r["model"], r["rep"]))
        except Exception:
            pass


def reset_browser_state():
    """Kill lingering drivers and clear cookies between cells."""
    subprocess.run(["pkill", "-f", "agent-browser"], capture_output=True)
    code = "cdp('Network.clearBrowserCookies')\nprint('cleared')\n"
    try:
        subprocess.run(
            ["browser-use"],
            input=code,
            text=True,
            capture_output=True,
            timeout=120,
            env=ENV,
        )
    except Exception:
        pass


cells = [
    (arm, task, model, rep)
    for model, task, rep, arm in itertools.product(MODELS, TASKS, REPS, ARMS)
]
total = len(cells)
n = 0
for arm, task, model, rep in cells:
    n += 1
    if (arm, task, model, rep) in done:
        continue
    print(f"[{n}/{total}] {arm} {task} {model} rep{rep}", flush=True)
    reset_browser_state()
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PY, os.path.join(ROOT, "single_run.py"), arm, task, model, str(rep)],
            capture_output=True,
            text=True,
            timeout=args.run_timeout,
            env={**ENV, "BUBENCH_TASKS": args.tasks},
        )
        rec = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT_JSON:"):
                rec = json.loads(line[len("RESULT_JSON:") :])
        if rec is None:
            rec = {
                "arm": arm,
                "task": task,
                "model": model,
                "rep": rep,
                "ok": False,
                "error": "no-result",
                "stderr_tail": (proc.stderr or "")[-800:],
                "stdout_tail": (proc.stdout or "")[-400:],
            }
    except subprocess.TimeoutExpired:
        rec = {
            "arm": arm,
            "task": task,
            "model": model,
            "rep": rep,
            "ok": False,
            "error": f"orchestrator-timeout-{args.run_timeout}s",
        }
    rec["cell_wall_s"] = round(time.time() - t0, 1)
    with open(args.results, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"  -> ok={rec.get('ok')} err={rec.get('error')} wall={rec.get('cell_wall_s')}s",
        flush=True,
    )

print("BATTERY COMPLETE", flush=True)
