#!/usr/bin/env python3
"""Battery runner + readout for the text_to_speech / manage_connections deferral A/B.

    TCD_BASE_TREE=<checkout of the base SHA> TCD_PR_TREE=<checkout with the change> TCD_PYTHON=<hermes env python> \\
    python3 run.py <model> <reps> [--parallel=N] [--only=task1,task2]
    python3 run.py --report

Inference: TCD_INFERENCE_URL + TCD_INFERENCE_KEY, or (default) the Nous Portal agent key read fresh from
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
RESULTS = Path(os.environ.get("TCD_RESULTS") or HERE / "results")
TASKS = ["tts_read_aloud", "tts_voice_note", "tts_indirect_driving", "tts_indirect_mp3", "tts_negative",
         "conn_connect_gmail", "conn_status_notion", "conn_required_recovery", "conn_mcp_install", "conn_second_account",
         "conn_negative", "ctl_write_file", "ctl_count_lines"]
TIMEOUT = {}


def _inference_env():
    env = dict(os.environ)
    if not env.get("TCD_INFERENCE_KEY"):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        nous = json.loads((home / "auth.json").read_text())["providers"]["nous"]
        env["TCD_INFERENCE_URL"], env["TCD_INFERENCE_KEY"] = nous["inference_base_url"], nous["agent_key"]
    return env


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    py = os.environ.get("TCD_PYTHON") or sys.executable
    try:
        p = subprocess.run([py, str(HERE / "worker.py"), arm, model, task, str(rep), str(out)],
                           capture_output=True, text=True, timeout=TIMEOUT.get(task, 420), env=_inference_env(), check=False)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line if p.returncode == 0 else f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "rep": rep, "score": 0.0,
                                   "error": "wall timeout", "reached": False, "first_call_valid": False, "total_tokens": None}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)


def _mean(cells, key):
    vals = [r[key] for r in cells if r.get(key) is not None and not r.get("error")]
    return f"{sum(vals) / len(vals):,.0f}" if vals else "-"


def report():
    """Fixed denominators: every cell that ran is in x/N; errored and timed-out cells score 0."""
    rows = [json.loads(p.read_text()) for p in RESULTS.rglob("*.json") if not p.name.endswith(".transcript.json")]
    for model in sorted({r["model"] for r in rows}):
        print(f"\n## {model}\n\n| task | base score | pr score | base reached | pr reached | base 1st-call valid | pr 1st-call valid "
              "| base calls | pr calls | base tok | pr tok | pr bridge |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        tot = {arm: {"score": 0.0, "n": 0, "tok": 0, "usd": 0.0, "done": 0.0} for arm in ("base", "pr")}
        for task in TASKS:
            cells = {arm: [r for r in rows if r["model"] == model and r["task"] == task and r["arm"] == arm] for arm in ("base", "pr")}
            if not any(cells.values()):
                continue
            line = [task]
            for arm in ("base", "pr"):
                t = tot[arm]
                t["score"] += sum(r["score"] for r in cells[arm])
                t["done"] += sum(1 for r in cells[arm] if r["score"] == 1.0)
                t["n"] += len(cells[arm])
                t["tok"] += sum(r.get("total_tokens") or 0 for r in cells[arm])
                t["usd"] += sum(r.get("cost_usd") or 0 for r in cells[arm])
                line.append(f"{sum(r['score'] for r in cells[arm]):g}/{len(cells[arm])}")
            for key in ("reached", "first_call_valid"):
                line += [f"{sum(bool(r.get(key)) for r in cells[arm])}/{len(cells[arm])}" for arm in ("base", "pr")]
            line += [_mean(cells["base"], "api_calls"), _mean(cells["pr"], "api_calls"), _mean(cells["base"], "total_tokens"),
                     _mean(cells["pr"], "total_tokens"), _mean(cells["pr"], "bridge_calls")]
            print("| " + " | ".join(line) + " |")
        for arm, t in tot.items():
            per = f"{t['tok'] / t['done']:,.0f} tok, ${t['usd'] / t['done']:.4f}" if t["done"] else "-"
            print(f"\n{arm}: score {t['score']:g}/{t['n']}, completed {t['done']:g}/{t['n']}, total {t['tok']:,} tok, "
                  f"${t['usd']:.2f}; per completed task: {per}")


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
