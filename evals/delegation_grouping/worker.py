#!/usr/bin/env python3
"""Run ONE (arm, model, task, rep) cell of the delegation result-grouping A/B in an isolated process.

Usage: worker.py <arm> <parent_model> <task_id> <rep> <out_json>
Env: DG_<ARM>_TREE (checkout of the arm's SHA), DG_INFERENCE_URL + DG_INFERENCE_KEY (OpenAI-compatible endpoint),
     DG_CHILD_MODEL (cheap model every child runs on, both arms).
Exit 3 = infra/config error (never scored).

A real parent ``AIAgent`` dispatches real background children with ``delegation.independent_completions`` on. The
harness plays the CLI's part between turns: every drained completion unit re-enters the parent as its own new
turn. Scores come from runtime accounting (completion units delivered, parent turns and model calls after the
dispatch, usage) and from file/answer oracles, never from how the model describes what it did.
"""
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback

ARM, MODEL, TASK_ID, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
TREE = os.environ.get(f"DG_{ARM.upper()}_TREE") or ""
if not TREE or not os.path.isdir(TREE):
    print(f"ABORT: DG_{ARM.upper()}_TREE not set or not a directory", file=sys.stderr)
    sys.exit(3)
INFER_URL, INFER_KEY = os.environ.get("DG_INFERENCE_URL") or "", os.environ.pop("DG_INFERENCE_KEY", "")
CHILD_MODEL = os.environ.get("DG_CHILD_MODEL") or ""
if not (INFER_URL and INFER_KEY and CHILD_MODEL):
    print("ABORT: DG_INFERENCE_URL / DG_INFERENCE_KEY / DG_CHILD_MODEL missing", file=sys.stderr)
    sys.exit(3)
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(var, None)
os.environ.pop("HERMES_PROFILE", None)

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS, "..", "core_tool_deferral"))
from sandbox import isolate_host

tmp_root = tempfile.mkdtemp(prefix=f"dg-{ARM}-{TASK_ID}-")
hermes_home = os.path.join(tmp_root, ".hermes")
ws = os.path.join(tmp_root, "ws")
os.makedirs(hermes_home)
os.makedirs(os.path.join(ws, "out"))
isolate_host(tmp_root, hermes_home)
with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(
        f"model:\n  provider: custom\n  base_url: {INFER_URL}\n  model: {MODEL}\n"
        "delegation:\n  independent_completions: true\n  subagent_auto_approve: true\n"
        f"  max_concurrent_children: 5\n  model: {CHILD_MODEL}\n"
    )
os.environ["HERMES_HOME"] = hermes_home
os.environ["TERMINAL_CWD"] = ws
os.chdir(ws)
sys.path.insert(0, TREE)

# --------------------------------------------------------------------------- fixtures
NAMES = ["alpha", "beta", "gamma"]
rng = random.Random(f"{TASK_ID}-{REP}")
FACTS = {}
for name in NAMES:
    build, latency, errors = f"{rng.randrange(1000, 9999)}", rng.randrange(100, 900), rng.randrange(2, 6)
    FACTS[name] = {"build": build, "latency": latency, "errors": errors}
    lines = [f"2026-11-09T10:{i:02d}:00 INFO {name} heartbeat ok seq={i}" for i in range(14)]
    lines[4] = f"BUILD_ID={build}"
    lines[9] = f"LATENCY_MS={latency}"
    lines += [f"2026-11-09T11:{i:02d}:00 ERROR {name} upstream timeout attempt={i}" for i in range(errors)]
    with open(os.path.join(ws, f"{name}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
SLOWEST = max(NAMES, key=lambda n: FACTS[n]["latency"])
TOTAL_ERRORS = sum(FACTS[n]["errors"] for n in NAMES)
FILES = ", ".join(f"{ws}/{n}.log" for n in NAMES)
NO_SELF = "Do not read the files yourself."

# kind: "all" = the parent answers once, after every child (one completion unit expected);
#       "each" = the parent acts on every result as it lands (one unit per child expected); "single" = control.
TASKS = {
    "all_join_ids": {
        "kind": "all", "children": 3,
        "prompt": (f"Three log files: {FILES}. Each has exactly one line starting with BUILD_ID=. Use three parallel "
                   f"subagents, one per file, to find the ids. {NO_SELF} Then give me one reply with the three ids "
                   "joined by dashes in the order alpha-beta-gamma."),
    },
    "all_slowest": {
        "kind": "all", "children": 3,
        "prompt": (f"Each of {FILES} has one line LATENCY_MS=<n>. Have a subagent check each file in parallel. "
                   f"{NO_SELF} I only care which service is slowest: tell me its name and its number."),
    },
    "all_summary_file": {
        "kind": "all", "children": 3,
        "prompt": (f"Fan out three subagents to review {FILES} (one each): each reports how many lines in its file "
                   f"contain the word ERROR. {NO_SELF} When the review is done, write {ws}/out/summary.md with one "
                   "row per file and a final line 'TOTAL=<sum>'."),
    },
    "all_implicit_table": {
        "kind": "all", "children": 3,
        "prompt": (f"Have subagents pull the BUILD_ID and the number of ERROR lines out of {FILES}, one subagent per "
                   f"file. {NO_SELF} I want a small table when it is done."),
    },
    "each_write_as_lands": {
        "kind": "each", "children": 3,
        "prompt": (f"Three log files: {FILES}. Each has one line starting with BUILD_ID=. Use three parallel "
                   f"subagents, one per file, to find the ids. {NO_SELF} I am watching {ws}/out/: the moment any "
                   f"one subagent reports, write its id into {ws}/out/<name>.txt (alpha.txt, beta.txt, gamma.txt) "
                   "right then. Do not wait for the others before writing; each file must appear as its result lands."),
    },
    "each_stream_to_user": {
        "kind": "each", "children": 3,
        "prompt": (f"Each of {FILES} has one line LATENCY_MS=<n>. Have a subagent check each file in parallel. "
                   f"{NO_SELF} Some of these may take a while, so tell me each number as soon as that subagent "
                   "finishes, in its own message; I do not want to wait for the slowest one to hear about the others."),
    },
    "single_control": {
        "kind": "single", "children": 1,
        "prompt": (f"Have a subagent find the line starting with BUILD_ID= in {ws}/alpha.log and tell me the id. "
                   f"{NO_SELF}"),
    },
}
TASK = TASKS[TASK_ID]

import hermes_agent.tools.process_registry as pr

pr._SYSTEMD_SCOPE_AVAILABLE = False
from hermes_agent.tools.process_registry import process_registry
import hermes_agent.tools.async_delegation as ad
from run_agent import AIAgent

agent = AIAgent(base_url=INFER_URL, api_key=INFER_KEY, provider="custom", model=MODEL, quiet_mode=True,
                skip_context_files=True, skip_memory=True, skip_background_review=True,
                enabled_toolsets=["delegation", "file"], max_iterations=20)
schema_chars = next((len(json.dumps(t)) for t in agent.tools or [] if t["function"]["name"] == "delegate_task"), None)


def _usage():
    return {k: int(getattr(agent, f"session_{k}", 0) or 0) for k in
            ("api_calls", "prompt_tokens", "input_tokens", "cache_read_tokens", "cache_write_tokens",
             "completion_tokens", "total_tokens")}


t0, error, convo, units, child_entries, turn_usage = time.time(), None, None, [], [], []
try:
    convo = agent.run_conversation(user_message=TASK["prompt"])
    turn_usage.append(_usage())
    deadline = time.time() + 300
    idle_since = None
    while time.time() < deadline and len(units) < 8:
        drained = []
        for evt, text in process_registry.drain_notifications(owns_event=lambda e: True):
            # Claim and acknowledge like the CLI drain: an unclaimed durable completion is offered again.
            claim = ad.claim_event_delivery(evt, "eval") if evt.get("type") == "async_delegation" else None
            if claim is None:
                continue
            ad.complete_event_delivery(evt, claim)
            drained.append((evt, text))
        for evt, text in drained:  # the CLI queues each completion unit as its own turn
            results = evt.get("results") or []
            units.append({"delegation_id": evt.get("delegation_id"), "group": evt.get("group"),
                          "task_indexes": [r.get("task_index") for r in results], "at_s": round(time.time() - t0, 1)})
            child_entries += results
            convo = agent.run_conversation(user_message=text, conversation_history=convo["messages"])
            turn_usage.append(_usage())
        if drained:
            idle_since = None
            continue
        if ad.active_count() == 0:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 3:
                break
        time.sleep(0.5)
except SystemExit:
    raise
except BaseException as e:  # health: allow BLE001 -- a crashed run is a scored data point; traceback printed
    error = f"{type(e).__name__}: {e}"
    traceback.print_exc()
wall = time.time() - t0
messages = (convo or {}).get("messages") or []
if (error and any(s in error for s in ("Authentication", "No LLM provider", "401"))) or not any(
        m.get("role") == "assistant" for m in messages):
    print(f"ABORT: no model turn (auth/provider failure): {error or (convo or {}).get('error')}", file=sys.stderr)
    sys.exit(3)

spawns, other_calls, replies = [], [], []
for m in messages:
    if m.get("role") != "assistant":
        continue
    if (m.get("content") or "").strip():
        replies.append(m["content"])
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            fargs = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            fargs = {}
        fargs = fargs if isinstance(fargs, dict) else {}
        if fn.get("name") == "delegate_task" and fargs.get("action") in (None, "spawn"):
            tasks = fargs.get("tasks") if isinstance(fargs.get("tasks"), list) else []
            spawns.append([t.get("group") if isinstance(t, dict) else None for t in tasks])
        else:
            other_calls.append(fn.get("name") + (f":{fargs.get('action')}" if fn.get("name") == "delegate_task" else ""))
final, all_text = (replies[-1] if replies else ""), "\n".join(replies)


def _read(rel):
    try:
        with open(os.path.join(ws, rel), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def success():
    if TASK_ID == "all_join_ids":
        return "-".join(FACTS[n]["build"] for n in NAMES) in final
    if TASK_ID == "all_slowest":
        return SLOWEST in final.lower() and str(FACTS[SLOWEST]["latency"]) in final
    if TASK_ID == "all_summary_file":
        return f"TOTAL={TOTAL_ERRORS}" in _read("out/summary.md").replace(" ", "")
    if TASK_ID == "all_implicit_table":
        return all(FACTS[n]["build"] in final for n in NAMES)
    if TASK_ID == "each_write_as_lands":
        return all(FACTS[n]["build"] in _read(f"out/{n}.txt") for n in NAMES)
    if TASK_ID == "each_stream_to_user":
        return all(str(FACTS[n]["latency"]) in all_text for n in NAMES)
    return FACTS["alpha"]["build"] in final


expected_units = TASK["children"] if TASK["kind"] == "each" else 1
ok = False if error else bool(success())
end = turn_usage[-1] if turn_usage else _usage()
first = turn_usage[0] if turn_usage else end
child_tok = {k: sum(int((r.get("tokens") or {}).get(k) or 0) for r in child_entries) for k in ("input", "output")}
record = {
    "arm": ARM, "model": MODEL, "child_model": CHILD_MODEL, "task": TASK_ID, "kind": TASK["kind"], "rep": REP,
    "error": error, "success": ok, "units": len(units), "expected_units": expected_units,
    "delivery_ok": len(units) == expected_units, "score": 1.0 if ok and len(units) == expected_units else 0.0,
    "spawn_calls": len(spawns), "spawn_groups": spawns, "children_run": len(child_entries),
    "parent_turns_after_dispatch": len(units), "parent_calls_total": end["api_calls"],
    "parent_calls_after_dispatch": end["api_calls"] - first["api_calls"], "other_calls": other_calls,
    "parent_usage": end, "child_tokens": child_tok, "child_api_calls": sum(int(r.get("api_calls") or 0) for r in child_entries),
    "child_cost_usd_estimate": round(sum(float(r.get("cost_usd") or 0) for r in child_entries), 6),
    "delegate_schema_chars": schema_chars, "unit_log": units, "wall_s": round(wall, 1),
}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
if os.environ.get("DG_TRANSCRIPTS"):  # local diagnosis only; transcripts are never checked in
    os.makedirs(os.environ["DG_TRANSCRIPTS"], exist_ok=True)
    with open(os.path.join(os.environ["DG_TRANSCRIPTS"], os.path.basename(OUT)), "w", encoding="utf-8") as f:
        json.dump(messages, f, default=str)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1, default=str)
print(json.dumps({k: record[k] for k in ("arm", "model", "task", "rep", "success", "units", "expected_units",
                                         "spawn_groups", "parent_calls_total", "wall_s", "error")}))
process_registry.kill_all(source="dg-end")
agent.close()
shutil.rmtree(tmp_root, ignore_errors=True)
os._exit(0)
