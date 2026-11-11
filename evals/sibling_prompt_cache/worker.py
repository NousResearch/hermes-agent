"""Do sibling subagents read each other's prompt cache? Real delegate_task children, one tree.

  SC_TREE=<tree> SC_URL=... SC_KEY=... SC_PROVIDER=... SC_API_MODE=... SC_MODEL=... python worker.py <arm> <out.jsonl>
A real parent AIAgent in a temp HERMES_HOME calls the real ``delegate_task`` three times, one after
another: child A, child B with different context, child C with A's context. Each child does a small
real task with real tools and must follow an instruction that is given only in its context. Per-call
prompt and cache-read tokens come from the agent's own usage accounting.
"""
import json, os, sys, tempfile, time, uuid
from pathlib import Path

ARM, OUT = sys.argv[1], sys.argv[2]
TREE, URL, KEY = os.environ["SC_TREE"], os.environ["SC_URL"], os.environ.pop("SC_KEY")
PROVIDER, API_MODE, MODEL = os.environ["SC_PROVIDER"], os.environ["SC_API_MODE"], os.environ["SC_MODEL"]
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")) or var.startswith(("HERMES_", "TERMINAL_")):
        os.environ.pop(var, None)
sys.path.insert(0, os.path.join(TREE, "evals", "core_tool_deferral"))
from sandbox import isolate_host

root = tempfile.mkdtemp(prefix=f"sc-{ARM}-")
home, ws = os.path.join(root, ".hermes"), os.path.join(root, "ws")
os.makedirs(home); os.makedirs(ws)
isolate_host(root, home)
NONCE = uuid.uuid4().hex[:10]
with open(os.path.join(home, "config.yaml"), "w") as f:
    f.write("delegation:\n  max_iterations: 14\n")
# A per-run SOUL line keeps every other live session's cache entries out of this run.
with open(os.path.join(home, "SOUL.md"), "w") as f:
    f.write(f"You are Hermes. Probe run {NONCE}.\n")
# A repo-shaped workspace: the child prompt carries its AGENTS.md, as in real in-repo delegation.
with open(os.path.join(ws, "AGENTS.md"), "w") as f:
    f.write(open(os.path.join(TREE, "AGENTS.md")).read())
for name, n in (("alpha", 4127), ("beta", 9310), ("gamma", 5586)):
    with open(os.path.join(ws, f"{name}.txt"), "w") as f:
        f.write(f"The {name} number is {n}.\n")
os.environ["HERMES_HOME"] = home
os.environ["TERMINAL_CWD"] = ws
os.chdir(ws)
sys.path.insert(0, TREE)

import agent.turn_response_check as trc
from hermes_state import SessionDB
from run_agent import AIAgent
from tools.delegate_tool import delegate_task

calls = []
_orig = trc.record_response_usage


def _rec(agent, *a, **k):
    before = (agent.session_prompt_tokens, agent.session_cache_read_tokens)
    out = _orig(agent, *a, **k)
    calls.append({"sid": agent.session_id, "prompt": agent.session_prompt_tokens - before[0],
                  "cache_read": agent.session_cache_read_tokens - before[1]})
    return out


trc.record_response_usage = _rec

parent = AIAgent(base_url=URL, api_key=KEY, provider=PROVIDER, api_mode=API_MODE, model=MODEL, quiet_mode=True,
                 platform="cli", session_db=SessionDB(db_path=Path(home) / "state.db"),
                 session_id=f"sc-parent-{NONCE}", skip_memory=True, skip_background_review=True,
                 enabled_toolsets=["terminal", "file", "skills", "delegation"])
parent._ensure_db_session()  # children reference the parent's session row
CHILDREN = [
    ("A", "The parent is auditing numbers. End your final reply with the exact word AUDIT-OK.", "alpha", "4127", "AUDIT-OK"),
    ("B", "The parent is fixing a flaky gateway test. End your final reply with the exact word FLAKY-OK.", "beta", "9310", "FLAKY-OK"),
    ("C", "The parent is auditing numbers. End your final reply with the exact word AUDIT-OK.", "gamma", "5586", "AUDIT-OK"),
]
def _ends_with(final, marker):
    """The context asks for the reply to END with the marker: nothing but whitespace may follow it."""
    return str(final or "").rstrip().endswith(marker)


GOAL = "Use the terminal to list this directory, then read {}.txt and reply with the number in it."
if os.environ.get("SC_LONG") == "1":  # several rounds per child, to see in-session reads past the shared prefix
    GOAL = ("Run these one per terminal call, in this order, never combined: `ls -la`, `cat alpha.txt`, `cat beta.txt`, "
            "`cat gamma.txt`, `wc -c *.txt`, `head -c 3000 AGENTS.md`, `tail -c 3000 AGENTS.md`. "
            "Then reply with the number in {}.txt.")
if os.environ.get("SC_PARALLEL") == "1":  # one batch: do concurrent siblings under one scope still cache in-session?
    raw = delegate_task(tasks=[{"goal": GOAL.format(c[2]), "context": c[1]} for c in CHILDREN], parent_agent=parent)
    results = json.loads(raw).get("results") or []
    by_sid = {}
    for c in calls:
        by_sid.setdefault(c["sid"], []).append((c["prompt"], c["cache_read"]))
    rec = {"arm": ARM, "route": PROVIDER, "model": MODEL, "parallel": True, "nonce": NONCE,
           "task_ok": [c[3] in str(r.get("summary") or "") for c, r in zip(CHILDREN, results)],
           "context_followed": [_ends_with(r.get("summary"), c[4]) for c, r in zip(CHILDREN, results)],
           "n_results": len(results), "finals": [str(r.get("summary") or "")[-200:] for r in results],
           "calls": list(by_sid.values())}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    sys.exit(0)
for name, context, fname, want, marker in CHILDREN:
    calls.clear()
    t0 = time.time()
    raw = delegate_task(goal=GOAL.format(fname), context=context, parent_agent=parent)
    try:
        entry = (json.loads(raw).get("results") or [{}])[0]
    except ValueError:
        entry = {"summary": raw}
    final = str(entry.get("summary") or "")
    rec = {"arm": ARM, "route": PROVIDER, "model": MODEL, "child": name, "task_ok": want in final, "context_followed": _ends_with(final, marker), "final": final[-200:],
           "status": entry.get("status"), "calls": [{"prompt": c["prompt"], "cache_read": c["cache_read"]} for c in calls], "nonce": NONCE, "wall": round(time.time() - t0, 1),
           "error": str(entry.get("error") or "")[:200]}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    time.sleep(3)
