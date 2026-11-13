"""Real-loop check: three real subagent-shaped AIAgents, run one after another, on one tree.

  RL_TREE=<tree> RL_URL=... RL_KEY=... python worker.py <arm> <out.jsonl> <rep>
Each child is a subagent-shaped AIAgent (platform="subagent", no context files, no memory) whose
ephemeral system prompt is a per-session tail that differs between A/C and B, and does
a small real task with real tools in a temp HERMES_HOME. Per-call usage is taken from the agent's own
usage accounting.
"""
import json, os, sys, tempfile, time, uuid

ARM, OUT, REP = sys.argv[1], sys.argv[2], int(sys.argv[3])
TREE, URL, KEY = os.environ["RL_TREE"], os.environ["RL_URL"], os.environ.pop("RL_KEY")
MODEL = os.environ.get("RL_MODEL", "anthropic/claude-fable-5.2")
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")) or var.startswith(("HERMES_", "TERMINAL_")):
        os.environ.pop(var, None)
sys.path.insert(0, os.path.join(TREE, "evals", "core_tool_deferral"))
from sandbox import isolate_host

root = tempfile.mkdtemp(prefix=f"rl-{ARM}-")
home, ws = os.path.join(root, ".hermes"), os.path.join(root, "ws")
os.makedirs(home); os.makedirs(ws)
isolate_host(root, home)
NONCE = uuid.uuid4().hex[:10]
with open(os.path.join(home, "config.yaml"), "w") as f:
    f.write(f"model:\n  provider: custom\n  base_url: {URL}\n  model: {MODEL}\n")
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

import hermes_agent.agent.turn_response_check as trc
from hermes_agent.run_agent import AIAgent

calls = []


def task_tail(context):
    """A per-session tail appended to the system prompt at request time (a task brief, a personality)."""
    return ("You are a focused subagent working on a specific delegated task.\n\nCONTEXT:\n" + context
            + "\n\n# Project Context\n\n" + open(os.path.join(ws, "AGENTS.md")).read()
            + "\n\nComplete this task using the tools available to you. When finished, provide a clear, concise summary.")

_orig = trc.record_response_usage


def _rec(agent, *a, **k):
    before = (agent.session_prompt_tokens, agent.session_cache_read_tokens, agent.session_cache_write_tokens)
    out = _orig(agent, *a, **k)
    calls.append({"prompt": agent.session_prompt_tokens - before[0], "cache_read": agent.session_cache_read_tokens - before[1],
                  "cache_write": agent.session_cache_write_tokens - before[2]})
    return out


trc.record_response_usage = _rec

CHILDREN = [
    ("A", "The parent is auditing numbers; report digits only.", "alpha", "4127"),
    ("B", "The parent is fixing a flaky gateway test; be brief and exact.", "beta", "9310"),
    ("C", "The parent is auditing numbers; report digits only.", "gamma", "5586"),
]
for name, context, fname, want in CHILDREN:
    calls.clear()
    child = AIAgent(base_url=URL, api_key=KEY, provider="custom", model=MODEL, quiet_mode=True, platform="subagent",
                    skip_context_files=True, skip_memory=True, skip_background_review=True,
                    enabled_toolsets=["terminal", "file", "skills"], max_iterations=8,
                    ephemeral_system_prompt=task_tail(context))
    assert child._use_prompt_caching and not child._use_native_cache_layout, "caching route not active"
    t0 = time.time()
    res = child.run_conversation(user_message=f"Use the terminal to list this directory, then read {fname}.txt and reply with the number in it.")
    final = str(res.get("final_response") or "")
    rec = {"arm": ARM, "rep": REP, "child": name, "ok": want in final, "n_calls": len(calls), "calls": list(calls),
           "session_prompt_tokens_est": len(child._cached_system_prompt or "") // 4, "nonce": NONCE,
           "wall": round(time.time() - t0, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    child.close()
    time.sleep(3)
