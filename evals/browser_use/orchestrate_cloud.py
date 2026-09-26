"""Cloud-backend battery: pr arm attached to a provisioned cloud browser per cell.

Backends:
    nous-cloud   - Nous Portal-provisioned Browser Use cloud browser
                   (plugins.browser.browser_use provider; needs gateway access)
    browserbase  - fresh Browserbase session per cell
                   (needs BROWSERBASE_API_KEY + BROWSERBASE_PROJECT_ID)

Usage:
    BUBENCH_BASE_TREE=... BUBENCH_PR_TREE=... \
        python3 orchestrate_cloud.py --backend nous-cloud [--reps 2]
        python3 orchestrate_cloud.py --backend browserbase [--reps 1]

Per cell: provision a session, export its CDP endpoint via BENCH_CDP_URL /
BU_CDP_WS, run single_run.py (pr arm), close the session. Resume-safe.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.environ.get("BUBENCH_ROOT", os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV_BASE = {**os.environ}
ENV_BASE["PATH"] = (
    os.path.expanduser("~/.local/bin") + os.pathsep + ENV_BASE.get("PATH", "")
)

parser = argparse.ArgumentParser()
parser.add_argument("--backend", required=True, choices=["nous-cloud", "browserbase"])
parser.add_argument("--tasks", default=os.path.join(ROOT, "tasks", "hard.json"))
parser.add_argument("--models", default="anthropic/claude-opus-4.8,moonshotai/kimi-k3")
parser.add_argument("--reps", type=int, default=2)
parser.add_argument("--results", default=None)
parser.add_argument("--run-timeout", type=int, default=1200)
args = parser.parse_args()

RESULTS = args.results or os.path.join(ROOT, "results", f"results_{args.backend}.jsonl")
os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
MODELS = args.models.split(",")
TASKS = list(json.load(open(args.tasks, encoding="utf-8")).keys())
REPS = list(range(1, args.reps + 1))

done = set()
if os.path.exists(RESULTS):
    for line in open(RESULTS, encoding="utf-8"):
        try:
            r = json.loads(line)
            done.add((r["task"], r["model"], r["rep"]))
        except Exception:
            pass


class NousCloud:
    def __init__(self):
        sys.path.insert(0, os.environ["BUBENCH_PR_TREE"])
        import importlib

        self._mod = importlib.import_module("plugins.browser.browser_use.provider")

    def create(self, name):
        self._prov = self._mod.BrowserUseBrowserProvider()
        sess = self._prov.create_session(name)
        return sess, {"BENCH_CDP_URL": sess["cdp_url"]}

    def close(self, sess):
        self._prov.close_session(
            sess.get("bb_session_id") or sess.get("session_name", "")
        )


class Browserbase:
    def create(self, name):
        req = urllib.request.Request(
            "https://api.browserbase.com/v1/sessions",
            data=json.dumps({
                "projectId": os.environ["BROWSERBASE_PROJECT_ID"]
            }).encode(),
            headers={
                "x-bb-api-key": os.environ["BROWSERBASE_API_KEY"],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            sess = json.load(resp)
        return sess, {
            "BU_CDP_WS": sess["connectUrl"],
            "BENCH_CDP_URL": sess["connectUrl"],
        }

    def close(self, sess):
        req = urllib.request.Request(
            f"https://api.browserbase.com/v1/sessions/{sess['id']}",
            data=json.dumps({
                "projectId": os.environ["BROWSERBASE_PROJECT_ID"],
                "status": "REQUEST_RELEASE",
            }).encode(),
            headers={
                "x-bb-api-key": os.environ["BROWSERBASE_API_KEY"],
                "Content-Type": "application/json",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=30)


provider = NousCloud() if args.backend == "nous-cloud" else Browserbase()

cells = [(t, m, rep) for m, t, rep in itertools.product(MODELS, TASKS, REPS)]
total = len(cells)
n = 0
for task, model, rep in cells:
    n += 1
    if (task, model, rep) in done:
        continue
    print(f"[{n}/{total}] {args.backend} pr {task} {model} rep{rep}", flush=True)
    sess = None
    t0 = time.time()
    try:
        sess, extra_env = provider.create(f"bubench-{task}-{rep}")
    except Exception as e:  # noqa: BLE001
        rec = {
            "arm": f"pr-{args.backend}",
            "task": task,
            "model": model,
            "rep": rep,
            "ok": False,
            "error": f"session-create: {e}",
        }
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        continue
    env = {**ENV_BASE, **extra_env, "BUBENCH_TASKS": args.tasks}
    subprocess.run(["pkill", "-f", "browser_harness"], capture_output=True)
    try:
        proc = subprocess.run(
            [PY, os.path.join(ROOT, "single_run.py"), "pr", task, model, str(rep)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.run_timeout,
            env=env,
        )
        rec = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT_JSON:"):
                rec = json.loads(line[len("RESULT_JSON:") :])
        if rec is None:
            rec = {
                "task": task,
                "model": model,
                "rep": rep,
                "ok": False,
                "error": "no-result",
                "stderr_tail": (proc.stderr or "")[-600:],
            }
    except subprocess.TimeoutExpired:
        rec = {
            "task": task,
            "model": model,
            "rep": rep,
            "ok": False,
            "error": f"timeout-{args.run_timeout}s",
        }
    finally:
        try:
            provider.close(sess)
        except Exception as e:  # noqa: BLE001
            print(f"  close warning: {e}", flush=True)
    rec["arm"] = f"pr-{args.backend}"
    rec["cell_wall_s"] = round(time.time() - t0, 1)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"  -> ok={rec.get('ok')} err={rec.get('error')} wall={rec.get('cell_wall_s')}s",
        flush=True,
    )

print("CLOUD BATTERY COMPLETE", flush=True)
