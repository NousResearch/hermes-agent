#!/usr/bin/env python3
"""Run ONE (arm, model, task, rep) cell of the budgeted-skills-index A/B in an isolated process.

Usage: worker.py <arm:full|tiered> <model_slug> <task_id> <rep> <out_json>
Env: SIB_INFERENCE_URL + SIB_INFERENCE_KEY (any OpenAI-compatible endpoint), SIB_TREE (checkout under
test), SIB_LIBRARY (library JSON, see library.py), SIB_TASKS (tasks JSON, see tasks.example.json).
Exit 3 = infra/config error (never scored).

Both arms run the same tree and the same rebuilt library; the only difference is
``skills.index_budget_percent`` (0 vs 1 of a pinned 200K window), i.e. whether the least-used skills
are listed with or without their description. Score = which skills the model loaded. The terminal
and the file-writing tools are stubbed to refuse: no cell can act on the host.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

ARM, MODEL, TASK_ID, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
TREE, LIBRARY, TASKS_FILE = (os.environ.get(k) or "" for k in ("SIB_TREE", "SIB_LIBRARY", "SIB_TASKS"))
if not (os.path.isdir(TREE) and os.path.isfile(LIBRARY) and os.path.isfile(TASKS_FILE)):
    print("ABORT: SIB_TREE / SIB_LIBRARY / SIB_TASKS not set", file=sys.stderr)
    sys.exit(3)
INFER_URL, INFER_KEY = os.environ.get("SIB_INFERENCE_URL") or "", os.environ.pop("SIB_INFERENCE_KEY", "")
if not (INFER_URL and INFER_KEY):
    print("ABORT: SIB_INFERENCE_URL / SIB_INFERENCE_KEY missing", file=sys.stderr)
    sys.exit(3)
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(var, None)
os.environ.pop("HERMES_PROFILE", None)
with open(TASKS_FILE, encoding="utf-8") as f:
    TASK = json.load(f)[TASK_ID]

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS, "..", "core_tool_deferral"))
sys.path.insert(0, HARNESS)
from library import build_home
from sandbox import isolate_host

tmp_root = tempfile.mkdtemp(prefix=f"sib-{ARM}-{TASK_ID}-")
hermes_home = os.path.join(tmp_root, ".hermes")
workspace = os.path.join(tmp_root, "ws")
os.makedirs(hermes_home)
os.makedirs(workspace)
isolate_host(tmp_root, hermes_home)
with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(f"model:\n  provider: custom\n  base_url: {INFER_URL}\n  model: {MODEL}\n  context_length: 200000\n"
            f"skills:\n  index_budget_percent: {1 if ARM == 'tiered' else 0}\n")
os.environ["HERMES_HOME"] = hermes_home
os.environ["TERMINAL_CWD"] = workspace
os.chdir(workspace)
build_home(hermes_home, LIBRARY)
sys.path.insert(0, TREE)

from tools.registry import registry


def _refuse(tool):
    """The question is which skill gets loaded, so nothing may act on the host: every tool that runs
    a command or writes a file answers with a fixed refusal on both arms."""
    entry = registry.get_entry(tool)
    if entry is not None:
        entry.handler = lambda args, **kw: json.dumps({"success": False, "exit_code": 126, "error": (
            f"{tool} is switched off in this sandbox. Say what you would have done and stop.")})
        entry.check_fn, entry.is_async = None, False


import model_tools  # registers the tools before they are stubbed
for _tool in ("terminal", "write_file", "patch", "execute_code", "process_manage"):
    _refuse(_tool)

from run_agent import AIAgent

agent = AIAgent(base_url=INFER_URL, api_key=INFER_KEY, provider="custom", model=MODEL, quiet_mode=True,
                skip_context_files=True, skip_memory=True, skip_background_review=True,
                enabled_toolsets=["skills", "terminal", "file"], max_iterations=3)

t0, error, convo = time.time(), None, None
try:
    convo = agent.run_conversation(TASK["prompt"])
except SystemExit:
    raise
except BaseException as e:  # health: allow BLE001 -- a crashed run is a scored data point; traceback printed
    error = f"{type(e).__name__}: {e}"
    traceback.print_exc()
wall = time.time() - t0
messages = (convo or {}).get("messages") or getattr(agent, "messages", []) or []
if (error and any(s in error for s in ("Authentication", "No LLM provider", "401"))) or not any(
        m.get("role") == "assistant" for m in messages):
    print(f"ABORT: no model turn (auth/provider failure): {error or (convo or {}).get('error')}", file=sys.stderr)
    sys.exit(3)

from agent.skills_index_delta import index_entries

index = index_entries(getattr(agent, "_cached_system_prompt", "") or "")
calls, loads = [], []
for m in messages:
    for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
        fn = tc.get("function", {})
        calls.append(fn.get("name", ""))
        if fn.get("name") == "skill_view":
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            loads.append(str((args if isinstance(args, dict) else {}).get("name") or "").rsplit("/", 1)[-1])
expect = TASK.get("skill")
wrong = sorted({n for n in loads if n != expect})
score = (1.0 if expect in loads else 0.0) if expect else (1.0 if not loads else 0.0)
record = {
    "arm": ARM, "model": MODEL, "task": TASK_ID, "kind": TASK["kind"], "rep": REP, "score": score, "error": error,
    "expected": expect, "loaded": loads, "wrong_loads": wrong, "tool_calls": calls,
    # Evidence the arm did what it says: was the expected skill listed, and with a description?
    "expected_listed": expect in index if expect else None,
    "expected_described": (": " in index.get(expect, "")) if expect else None,
    "names_only_count": sum(1 for row in index.values() if ": " not in row), "index_skills": len(index),
    "api_calls": getattr(agent, "session_api_calls", None),
    "input_tokens": getattr(agent, "session_input_tokens", None),
    "cache_write_tokens": getattr(agent, "session_cache_write_tokens", None),
    "cache_read_tokens": getattr(agent, "session_cache_read_tokens", None),
    "output_tokens": getattr(agent, "session_output_tokens", None),
    "prompt_tokens": getattr(agent, "session_prompt_tokens", None),
    "total_tokens": getattr(agent, "session_total_tokens", None),
    "cost_usd": getattr(agent, "session_estimated_cost_usd", None),
    "wall_s": round(wall, 1),
}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1, default=str)
print(json.dumps({k: record[k] for k in ("arm", "model", "task", "rep", "score", "loaded", "api_calls", "prompt_tokens",
                                         "cache_read_tokens", "cost_usd", "names_only_count", "error")}))
agent.close()
shutil.rmtree(tmp_root, ignore_errors=True)
