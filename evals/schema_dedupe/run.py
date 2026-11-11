#!/usr/bin/env python3
"""Battery runner + readout for the tool-schema dedupe A/B.

    SDD_BASE_TREE=<checkout of the base SHA> SDD_PR_TREE=<checkout of the PR SHA> SDD_PYTHON=<hermes env python> \\
    python3 run.py <model> <reps> [--parallel=N] [--only=task1,task2]
    python3 run.py --report

Inference: SDD_INFERENCE_URL + SDD_INFERENCE_KEY, or (default) the Nous Portal agent key read fresh from
$HERMES_HOME/auth.json for every cell. Read-only: the harness never refreshes or copies the OAuth grant.
Resume-safe per (model, task, arm, rep); errored cells score 0 and stay in the denominator; exit 3 (no model
turn) is infra and is never written.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SDD_RESULTS") or HERE / "results")
TASKS = ["mem_swap_stale", "mem_full_add", "mem_add_overflow", "mem_targets", "mem_procedure_to_skill",
         "conn_hosted_and_mcp", "conn_ambiguous_hosted", "conn_mcp_words", "conn_second_account",
         "conn_switch_account", "conn_skip_no_reask", "conn_no_disconnect",
         "search_content", "search_newest", "search_count", "term_wait_then_write", "term_dev_server"]
ARMS = ("base", "pr")


def _inference_env():
    env = dict(os.environ)
    if not env.get("SDD_INFERENCE_KEY"):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        nous = json.loads((home / "auth.json").read_text())["providers"]["nous"]
        env["SDD_INFERENCE_URL"], env["SDD_INFERENCE_KEY"] = nous["inference_base_url"], nous["agent_key"]
    return env


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    py = os.environ.get("SDD_PYTHON") or sys.executable
    try:
        p = subprocess.run([py, str(HERE / "worker.py"), arm, model, task, str(rep), str(out)],
                           capture_output=True, text=True, timeout=420, env=_inference_env(), check=False)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line if p.returncode == 0 else f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "rep": rep, "score": 0.0,
                                   "error": "wall timeout"}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)


def _mean(cells, key):
    vals = [r[key] for r in cells if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(value, spec=",.0f"):
    return "-" if value is None else format(value, spec)


def report():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.rglob("*.json")) if not p.name.endswith(".transcript.json")]
    spend = sum(r.get("estimated_cost_usd") or 0 for r in rows)
    for model in sorted({r["model"] for r in rows}):
        by = {(t, a): [r for r in rows if r["model"] == model and r["task"] == t and r["arm"] == a]
              for t in TASKS for a in ARMS}
        print(f"\n## {model}\n\n| task | base | pr | base calls | pr calls | base tok | pr tok | base $ | pr $ | notes (base / pr) |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
        tot = {a: [0.0, 0, 0.0, 0.0] for a in ARMS}
        for task in TASKS:
            if not any(by[(task, a)] for a in ARMS):
                continue
            line, extra = [task], []
            for arm in ARMS:
                cells = by[(task, arm)]
                tot[arm][0] += sum(r["score"] for r in cells)
                tot[arm][1] += len(cells)
                tot[arm][2] += sum(r.get("total_tokens") or 0 for r in cells)
                tot[arm][3] += sum(r.get("estimated_cost_usd") or 0 for r in cells)
                line.append(f"{sum(r['score'] for r in cells):g}/{len(cells)}")
                if task.startswith("mem_"):
                    mem = [r["memory"] for r in cells if r.get("memory")]
                    extra.append("calls {} rejected {} repeat {}".format(
                        sum(m["memory_calls"] for m in mem), sum(m["memory_rejected"] for m in mem),
                        sum(m["repeat_writes"] for m in mem)))
                elif task.startswith("conn_"):
                    extra.append(f"refused {sum(r.get('conn_refused') or 0 for r in cells)}")
                else:
                    extra.append("")
            line += [_fmt(_mean(by[(task, a)], k), s) for k, s in (("api_calls", ".1f"), ("total_tokens", ",.0f"),
                                                                    ("estimated_cost_usd", ".3f")) for a in ARMS]
            print("| " + " | ".join(line + [" / ".join(extra)]) + " |")
        print("| **total** | **{:g}/{}** | **{:g}/{}** | | | {:,.0f} | {:,.0f} | {:.2f} | {:.2f} | summed over all cells |".format(
            tot["base"][0], tot["base"][1], tot["pr"][0], tot["pr"][1], tot["base"][2], tot["pr"][2], tot["base"][3], tot["pr"][3]))
        for arm in ARMS:
            cells = [r for r in rows if r["model"] == model and r["arm"] == arm and not r.get("error")]
            print(f"- {arm}: input {sum(r.get('input_tokens') or 0 for r in cells):,} · cache write "
                  f"{sum(r.get('cache_write_tokens') or 0 for r in cells):,} · cache read "
                  f"{sum(r.get('cache_read_tokens') or 0 for r in cells):,} · output "
                  f"{sum(r.get('completion_tokens') or 0 for r in cells):,} · errored cells "
                  f"{sum(1 for r in rows if r['model'] == model and r['arm'] == arm and r.get('error'))}")
    print(f"\nEstimated spend across every recorded cell: ${spend:.2f}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)
    model, reps = sys.argv[1], int(sys.argv[2])
    par = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--parallel=")), "6"))
    only = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), TASKS)
    jobs = [(model, t, arm, rep) for rep in range(1, reps + 1) for t in only for arm in ARMS]
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda j: cell(*j), jobs))
    report()
