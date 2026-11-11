#!/usr/bin/env python3
"""Sandboxed multi-arm battery: can manage_connections sit behind tool_search?

    relay.py <sock> <log> &                                   # the only network exit for a cell
    TCD_TREE=<checkout> TCD_PYTHON=<hermes env python> TCD_RESULTS=<dir> TCD_CELLS=<scratch dir> TCD_RELAY_SOCK=<sock> \\
      python3 battery.py <model> --arms=A,B,C [--parallel=N] [--only=t1,t2] [--reps=N] [--dry]
    TCD_RESULTS=<dir> python3 battery.py --report

Arms (one tree, see worker.py): A eager, B deferred, C deferred + reworded catalog line, D eager + reworded.
Every cell runs in bubblewrap: its own network namespace (loopback + the relay socket only), /home, /tmp,
/run and /mnt replaced by empty tmpfs, the tree and the interpreter bound read-only, one writable cell
dir at /home/cell. The operator's Hermes home, tokens and CLI do not exist inside, and the agent key
stays in the relay. ``--dry`` runs the no-model proof cell for each arm.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("TCD_RESULTS") or HERE / "results")
MCP_TASKS = ["conn_mcp_install", "mcp_add_linear_server", "mcp_setup_github", "mcp_talk_sentry"]
CONN_TASKS = ["conn_connect_gmail", "conn_status_notion", "conn_required_recovery", "conn_second_account"]
NEGATIVES = ["conn_negative", "mcp_negative"]
CONTROLS = ["ctl_write_file", "ctl_count_lines"]
TASKS = MCP_TASKS + CONN_TASKS + NEGATIVES + CONTROLS
REPS = {**dict.fromkeys(MCP_TASKS, 5), **dict.fromkeys(CONN_TASKS + NEGATIVES + CONTROLS, 3)}
INFER_HOST = "inference-api.nousresearch.com"
# $ per million tokens (fresh input, cache read, cache write, output), from the live Nous model list.
PRICES = {"anthropic/claude-fable-5.2": (8.0, 0.2, 10.0, 40.0), "qwen/qwen3.8-27b": (0.15, 0.0375, 0.15, 1.875),
          "openai/gpt-5.6-luna": (0.2, 0.02, 0.25, 1.2)}


def _sandbox_cmd(cell_dir, worker_args):
    tree, py = os.environ["TCD_TREE"], Path(os.environ["TCD_PYTHON"])
    venv, base_py = py.parent.parent, py.resolve().parent.parent
    hosts = Path(os.environ["TCD_CELLS"]) / "hosts"
    hosts.write_text(f"127.0.0.1 localhost {INFER_HOST}\n", encoding="utf-8")
    cmd = ["bwrap", "--die-with-parent", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--clearenv",
           "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    # The real Hermes home is hidden wherever it lives (HERMES_HOME may sit outside /home).
    real_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").resolve()
    for masked in ("/home", "/tmp", "/run", "/mnt", "/media", "/srv", "/var/tmp", str(real_home)):
        cmd += ["--tmpfs", masked]
    cmd += ["--ro-bind", str(base_py), str(base_py), "--ro-bind", str(venv), str(venv), "--ro-bind", tree, "/home/tree",
            "--bind", str(cell_dir), "/home/cell", "--ro-bind", str(hosts), "/etc/hosts",
            "--bind", os.environ["TCD_RELAY_SOCK"], "/run/relay.sock", "--chdir", "/home/cell"]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/cell", "LANG": "C.UTF-8", "TMPDIR": "/tmp",
           "PYTHONDONTWRITEBYTECODE": "1", "TCD_TREE": "/home/tree", "TCD_CELL_ROOT": "/home/cell",
           "TCD_INFERENCE_URL": f"http://{INFER_HOST}:8080/v1", "TCD_INFERENCE_KEY": "held-by-the-relay",
           "TCD_REAL_HERMES_HOME": str(real_home)}
    for k, v in env.items():
        cmd += ["--setenv", k, v]
    script = ("socat TCP-LISTEN:8080,bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:/run/relay.sock & sleep 0.3; "
              f'exec {shlex.quote(str(py))} /home/tree/evals/tts_connections_deferral/worker.py "$@"')
    return cmd + ["/bin/sh", "-c", script, "sh", *worker_args, "/home/cell/out.json"]


def cell(model, task, arm, rep):
    out = RESULTS / model.replace("/", "__") / f"{task}.{arm}.{rep}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cell_dir = Path(os.environ["TCD_CELLS"]) / uuid.uuid4().hex[:12]
    cell_dir.mkdir(parents=True)
    try:
        p = subprocess.run(_sandbox_cmd(cell_dir, [arm, model, task, str(rep)]), capture_output=True, text=True,
                           timeout=480, check=False)
        if p.returncode == 0 and (cell_dir / "out.json").exists():
            shutil.copy(cell_dir / "out.json", out)
            if (cell_dir / "out.json.transcript.json").exists():
                shutil.copy(cell_dir / "out.json.transcript.json", str(out) + ".transcript.json")
            print((p.stdout.strip().splitlines() or [f"{task} {arm} {rep} done"])[-1], flush=True)
        else:
            print(f"[{task} {arm} {rep}] exit {p.returncode}: {p.stderr[-500:]}", flush=True)
    except subprocess.TimeoutExpired:
        out.write_text(json.dumps({"arm": arm, "model": model, "task": task, "rep": rep, "score": 0.0, "error": "wall timeout",
                                   "reached": False, "first_call_valid": False, "wrong_route": [], "kind": None}))
        print(f"[{task} {arm} {rep}] wall timeout", flush=True)
    finally:
        shutil.rmtree(cell_dir, ignore_errors=True)


def _tokens(r):
    """Fresh input, cache read, cache write, output. ``input_tokens`` is the fresh (uncached) part."""
    return tuple(r.get(k) or 0 for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"))


def _usd(r):
    return sum(t * p for t, p in zip(_tokens(r), PRICES.get(r["model"], (0, 0, 0, 0)))) / 1e6


def _agg(cells):
    n = len(cells)
    done = sum(1 for r in cells if r["score"] == 1.0)
    tok = sum(sum(_tokens(r)) for r in cells)
    return {"n": n, "done": done, "wrong": sum(1 for r in cells if r.get("wrong_route")),
            "wrong_calls": sum(len(r.get("wrong_route") or []) for r in cells),
            "calls": sum(r.get("api_calls") or 0 for r in cells) / n if n else 0.0,
            "tok": tok, "tok_task": tok / n if n else 0.0, "tok_done": tok / done if done else float("inf"),
            "usd": sum(_usd(r) for r in cells), "parts": [sum(_tokens(r)[i] for r in cells) for i in range(4)],
            "false": sum(1 for r in cells if r.get("expect") is False and r.get("reached")),
            "noise": sum(1 for r in cells if r.get("error"))}


def report():
    """Fixed denominators: every cell that ran is in x/N; errored and timed-out cells score 0 on every arm."""
    rows = [json.loads(p.read_text()) for p in RESULTS.rglob("*.json") if not p.name.endswith(".transcript.json")]
    rows = [r for r in rows if "score" in r]
    arms = sorted({r["arm"] for r in rows})
    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        print(f"\n## {model}\n\n| arm | done | wrong-route cells (calls) | false triggers | model calls/task | tokens/task "
              "| tokens/completed task | fresh in | cache read | cache write | output | $/task | errored |\n" + "|---" * 13 + "|")
        agg = {arm: _agg([r for r in mine if r["arm"] == arm]) for arm in arms}
        for arm, a in agg.items():
            print(f"| {arm} | {a['done']}/{a['n']} | {a['wrong']} ({a['wrong_calls']}) | {a['false']} | {a['calls']:.2f} | {a['tok_task']:,.0f} "
                  f"| {a['tok_done']:,.0f} | {a['parts'][0]:,} | {a['parts'][1]:,} | {a['parts'][2]:,} | {a['parts'][3]:,} "
                  f"| {a['usd'] / a['n'] if a['n'] else 0:.4f} | {a['noise']} |")
        base = agg.get("A")
        for arm in [x for x in arms if x != "A"] if base else []:
            a = agg[arm]
            checks = {"tasks done": a["done"] >= base["done"], "wrong route": a["wrong"] <= base["wrong"],
                      "calls/task (+0.2 max)": a["calls"] <= base["calls"] + 0.2, "tokens/task": a["tok_task"] <= base["tok_task"]}
            verdict = "PASS" if all(checks.values()) and a["n"] == base["n"] else "FAIL"
            print(f"\nArm {arm} vs A: {verdict} (" + ", ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items())
                  + f"; n {a['n']} vs {base['n']})")
        print("\n| task | " + " | ".join(f"{arm} done | {arm} reached | {arm} searched | {arm} wrong-route | {arm} calls | {arm} tok"
                                          for arm in arms) + " |\n|---" + "|---" * (6 * len(arms)) + "|")
        for task in [t for t in TASKS if any(r["task"] == t for r in mine)]:
            line = [task]
            for arm in arms:
                c = [r for r in mine if r["arm"] == arm and r["task"] == task]
                a = _agg(c)
                line += [f"{a['done']}/{a['n']}", f"{sum(bool(r.get('reached')) for r in c)}/{a['n']}",
                         f"{sum(bool(r.get('searched')) for r in c)}/{a['n']}", f"{a['wrong']}/{a['n']}",
                         f"{a['calls']:.1f}", f"{a['tok_task']:,.0f}"]
            print("| " + " | ".join(line) + " |")
    print(f"\ntotal priced from token counts: ${sum(_usd(r) for r in rows):.2f}; "
          f"Hermes's own estimate: ${sum(r.get('cost_usd') or 0 for r in rows):.2f}; cells: {len(rows)}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit(0)

    def opt(name, default):
        return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)

    model, arms = sys.argv[1], opt("arms", "A,B,C").split(",")
    if "--dry" in sys.argv:
        jobs = [(model, "__dry__", arm, 0) for arm in arms]
    else:
        only, cap = opt("only", ",".join(TASKS)).split(","), int(opt("reps", "99"))
        jobs = [(model, t, arm, rep) for rep in range(1, 6) for t in only for arm in arms if rep <= min(REPS[t], cap)]
    with ThreadPoolExecutor(int(opt("parallel", "6"))) as ex:
        list(ex.map(lambda j: cell(*j), jobs))
