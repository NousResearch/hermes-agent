#!/usr/bin/env python3
"""Battery runner + readout for the delegation result-grouping A/B.

    DG_BASE_TREE=<checkout of the base SHA> DG_CAND_TREE=<checkout of the candidate SHA> DG_PYTHON=<hermes env python> \\
    DG_CHILD_MODEL=<cheap model> python3 run.py <parent_model> <reps> [--parallel=N] [--only=task1,task2] [--arms=base,cand]
    python3 run.py --report
    python3 run.py --schema <tree> <on|off>      # the delegate_task definition a tree builds, as OpenAI tool JSON

Inference: DG_INFERENCE_URL + DG_INFERENCE_KEY, or (default) the Nous Portal agent key read fresh from
$HERMES_HOME/auth.json for every cell. Read-only: the harness never refreshes or copies the OAuth grant.
Resume-safe per (model, task, arm, rep). A cell that errors or times out scores 0 and stays in the denominator; a
cell with no model turn (worker exit 3) is infra and is rerun, never scored.
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("DG_RESULTS") or HERE / "results")
TASKS = ["all_join_ids", "all_slowest", "all_summary_file", "all_implicit_table", "each_write_as_lands", "each_stream_to_user", "single_control"]
KINDS = {"all": "answer after all (1 unit expected)", "each": "act on each (3 units expected)", "single": "single task (control)"}


def _inference_env():
    env = dict(os.environ)
    if not env.get("DG_INFERENCE_KEY"):
        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        nous = json.loads((home / "auth.json").read_text())["providers"]["nous"]
        env["DG_INFERENCE_URL"], env["DG_INFERENCE_KEY"] = nous["inference_base_url"], nous["agent_key"]
    return env


def _pricing():
    """List prices per token for every model the endpoint serves (cached next to the results)."""
    cache = RESULTS / "pricing.json"
    if cache.exists():
        return json.loads(cache.read_text())
    env = _inference_env()
    req = urllib.request.Request(env["DG_INFERENCE_URL"].rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer " + env["DG_INFERENCE_KEY"]})
    with urllib.request.urlopen(req, timeout=30) as resp:
        prices = {m["id"]: m.get("pricing") or {} for m in json.load(resp)["data"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(prices))
    return prices


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    py = os.environ.get("DG_PYTHON") or sys.executable
    try:
        p = subprocess.run([py, str(HERE / "worker.py"), arm, model, task, str(rep), str(out)],
                           capture_output=True, text=True, timeout=480, env=_inference_env(), check=False)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        print(line if p.returncode == 0 else f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-400:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "rep": rep, "score": 0.0, "success": False,
                                   "kind": task.split("_", 1)[0], "delivery_ok": False, "error": "wall timeout"}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)


def dollars(row, prices):
    """Usage x list price: parent on its model (cache reads and writes at their own rates), children on theirs."""
    u, p = row.get("parent_usage") or {}, prices.get(row["model"]) or {}
    rate = lambda key, fallback="prompt": float(p.get(key) or p.get(fallback) or 0)
    read, write = u.get("cache_read_tokens", 0), u.get("cache_write_tokens", 0)
    fresh = max(u.get("prompt_tokens", 0) - read - write, 0)
    parent = (fresh * rate("prompt") + read * rate("input_cache_read") + write * rate("input_cache_write")
              + u.get("completion_tokens", 0) * rate("completion"))
    c, cp = row.get("child_tokens") or {}, prices.get(row.get("child_model")) or {}
    child = c.get("input", 0) * float(cp.get("prompt") or 0) + c.get("output", 0) * float(cp.get("completion") or 0)
    return parent + child


def report():
    rows = [json.loads(p.read_text()) for p in RESULTS.rglob("*.json") if p.name != "pricing.json"]
    prices = _pricing()
    arms = sorted({r["arm"] for r in rows})
    mean = lambda vals: f"{sum(vals) / len(vals):,.1f}" if vals else "-"
    spent = 0.0
    for model in sorted({r["model"] for r in rows}):
        print(f"\n## parent {model}\n")
        print("| task | arm | delivered as expected | task success | units | parent turns after dispatch | "
              "parent model calls | tokens per cell (parent + children) | $ per cell |\n|---|---|---|---|---|---|---|---|---|")
        for task in TASKS + ["**all tasks**"]:
            for arm in arms:
                cells = [r for r in rows if r["model"] == model and r["arm"] == arm and (task.startswith("**") or r["task"] == task)]
                if not cells:
                    continue
                ran = [r for r in cells if r.get("parent_usage")]
                tok = [r["parent_usage"]["total_tokens"] + sum(r["child_tokens"].values()) for r in ran]
                usd = [dollars(r, prices) for r in ran]
                if task.startswith("**"):
                    spent += sum(usd)
                print(f"| {task} | {arm} | {sum(bool(r.get('delivery_ok')) for r in cells)}/{len(cells)} | "
                      f"{sum(bool(r.get('success')) for r in cells)}/{len(cells)} | {mean([r['units'] for r in ran])} | "
                      f"{mean([r['parent_turns_after_dispatch'] for r in ran])} | {mean([r['parent_calls_total'] for r in ran])} | "
                      f"{mean(tok)} | {(sum(usd) / len(usd)) if usd else 0:.4f} |")
        print("\n| class | arm | delivered as expected | task success | tokens per cell | $ per cell |\n|---|---|---|---|---|---|")
        for kind, label in KINDS.items():
            for arm in arms:
                cells = [r for r in rows if r["model"] == model and r["arm"] == arm and r.get("kind") == kind]
                ran = [r for r in cells if r.get("parent_usage")]
                if not cells:
                    continue
                tok = [r["parent_usage"]["total_tokens"] + sum(r["child_tokens"].values()) for r in ran]
                usd = [dollars(r, prices) for r in ran]
                print(f"| {label} | {arm} | {sum(bool(r.get('delivery_ok')) for r in cells)}/{len(cells)} | "
                      f"{sum(bool(r.get('success')) for r in cells)}/{len(cells)} | {mean(tok)} | {(sum(usd) / len(usd)) if usd else 0:.4f} |")
    print(f"\nUsage x list price over every recorded cell: ${spent:.2f}")


def schema(tree, mode):
    """Print the delegate_task definition *tree* builds with independent_completions on/off (no network, temp home)."""
    code = (
        "import json, os, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from tools.registry import registry\n"
        "import tools.delegate_tool\n"
        "print(json.dumps(registry.get_definitions({'delegate_task'})[0]))\n"
    )
    with tempfile.TemporaryDirectory(prefix="dg-schema-") as home:
        Path(home, "config.yaml").write_text(
            "delegation:\n  independent_completions: %s\n" % ("true" if mode == "on" else "false"), encoding="utf-8")
        env = {"HOME": home, "HERMES_HOME": home, "PATH": os.environ.get("PATH", ""), "HERMES_PLATFORM": "cli"}
        out = subprocess.run([os.environ.get("DG_PYTHON") or sys.executable, "-c", code, tree], capture_output=True,
                             text=True, env=env, cwd=home, check=True, timeout=120).stdout
    print(out.strip().splitlines()[-1])


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)
    if "--schema" in sys.argv:
        i = sys.argv.index("--schema")
        schema(sys.argv[i + 1], sys.argv[i + 2])
        sys.exit(0)
    model, reps = sys.argv[1], int(sys.argv[2])
    opt = lambda name, default: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)
    only, arms = opt("only", ",".join(TASKS)).split(","), opt("arms", "base,cand").split(",")
    _pricing()
    jobs = [(model, t, arm, rep) for rep in range(1, reps + 1) for t in only for arm in arms]
    with ThreadPoolExecutor(int(opt("parallel", "4"))) as ex:
        list(ex.map(lambda j: cell(*j), jobs))
    report()
