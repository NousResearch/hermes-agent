#!/usr/bin/env python3
"""Run ONE (arm, model, task, rep) cell of the deferred-catalog form A/B/C in an isolated process.

Usage: worker.py <arm:full|names|summary> <model_slug> <task_id> <rep> <out_json>
Env: CF_INFERENCE_URL + CF_INFERENCE_KEY (any OpenAI-compatible endpoint), CF_TREE (hermes-agent checkout
carrying the ``tools.tool_search.listing_servers`` knob). CF_DUMP_PREFIX=1 writes the assembled tool_search
description next to the result and exits before any model call. Exit 3 = infra/config error (never scored).

The only thing that differs between arms is the listing embedded in the tool_search description:
  full    - one "name: description" line per deferred tool (today's form)
  names   - full lines for single-tool groups (deferred built-ins), names only for plugin/MCP servers
  summary - full lines for single-tool groups, one "N tools, names not listed" line per server
The 73 plugin/MCP tools of the reference catalog (catalog.txt: blender 30, meshy 28, elevenlabs_music 8,
nous-support 6, elevenlabs_sound_effect) are registry stubs with the catalog's names and descriptions and
canned results; the deferred built-ins are the tree's own. Terminal and file tools are real.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback

ARM, MODEL, TASK_ID, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
ARM_FORM = {"full": "auto", "names": "names", "summary": "summary"}
if ARM not in ARM_FORM:
    print(f"ABORT: unknown arm {ARM!r}", file=sys.stderr)
    sys.exit(3)
TREE = os.environ.get("CF_TREE") or ""
if not TREE or not os.path.isdir(TREE):
    print("ABORT: CF_TREE not set or not a directory", file=sys.stderr)
    sys.exit(3)
DUMP = os.environ.get("CF_DUMP_PREFIX") == "1"
INFER_URL, INFER_KEY = os.environ.get("CF_INFERENCE_URL") or "", os.environ.pop("CF_INFERENCE_KEY", "")
if not DUMP and not (INFER_URL and INFER_KEY):
    print("ABORT: CF_INFERENCE_URL / CF_INFERENCE_KEY missing", file=sys.stderr)
    sys.exit(3)
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(var, None)
os.environ.pop("HERMES_PROFILE", None)
# A real Blender on the operator's PATH would let the agent bypass the stubs with the terminal.
os.environ["PATH"] = os.pathsep.join(
    p for p in os.environ.get("PATH", "").split(os.pathsep) if not os.path.exists(os.path.join(p, "blender")))

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS, "..", "core_tool_deferral"))
from sandbox import isolate_host

sys.path.insert(0, HARNESS)  # this harness's tasks.py, not core_tool_deferral's

tmp_root = tempfile.mkdtemp(prefix=f"cf-{ARM}-{TASK_ID}-")
hermes_home = os.path.join(tmp_root, ".hermes")
workspace = os.path.join(tmp_root, "ws")
os.makedirs(hermes_home)
os.makedirs(workspace)
isolate_host(tmp_root, hermes_home)
with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(f"model:\n  provider: custom\n  base_url: {INFER_URL or 'http://127.0.0.1:9'}\n  model: {MODEL}\n"
            f"tools:\n  tool_search:\n    listing_servers: {ARM_FORM[ARM]}\n")
os.environ["HERMES_HOME"] = hermes_home
os.environ["TERMINAL_CWD"] = workspace
os.chdir(workspace)
sys.path.insert(0, TREE)

from tasks import NEUTRAL, TASKS, canned_result, params_for, write_fixtures

TASK = TASKS[TASK_ID]
REPLIES = list(TASK.get("replies") or [])
write_fixtures(workspace)

# --------------------------------------------------------------------------- stub the catalog
LOG = []  # every stub invocation, in order


def parse_catalog():
    """catalog.txt -> [(toolset, name, description)] for the plugin/MCP tools (built-ins are the tree's own)."""
    rows, group = [], None
    with open(os.path.join(HARNESS, "catalog.txt"), encoding="utf-8") as fh:
        for line in fh:
            head = re.match(r"^(\S+) tools \(\d+\):$", line.strip())
            if head:
                group = head.group(1)
                continue
            item = re.match(r"^- ([A-Za-z0-9_]+): (.*)$", line.strip())
            if not item:
                continue
            name, desc = item.group(1), item.group(2)
            if group is None:
                if name == "elevenlabs_sound_effect":
                    rows.append(("elevenlabs_sound_effects", name, desc))
                continue
            rows.append((group if group == "elevenlabs_music" else f"mcp-{group}", name, desc))
    return rows


import model_tools  # registers the built-in tools
from tools.registry import registry

STUBS = parse_catalog()
STUB_NAMES = {name for _, name, _ in STUBS}


def make_handler(name):
    def handler(args, **kw):
        args = args if isinstance(args, dict) else {}
        LOG.append({"tool": name, "args": args})
        return json.dumps(canned_result(name, args, TASK_ID, workspace))
    return handler


for toolset, name, desc in STUBS:
    registry.register(
        name=name, toolset=toolset, handler=make_handler(name), description=desc,
        schema={"name": name, "description": desc, "parameters": params_for(name)})
for name in registry.get_all_tool_names():  # keep the eager surface to terminal + files in every arm
    entry = registry.get_entry(name)
    if entry is not None and name.startswith("browser_") and name != "browser_vault":
        entry.check_fn = lambda: False
for name in ("browser_vault", "image_generate"):
    entry = registry.get_entry(name)
    if entry is not None:
        entry.check_fn, entry.is_async = None, False
        entry.handler = lambda a, _n=name, **kw: (LOG.append({"tool": _n, "args": a}) or json.dumps({"success": True}))

TOOLSETS = ["terminal", "file", "browser", "cronjob", "session_search", "todo", "image_gen"] + sorted(
    {ts for ts, _, _ in STUBS})

from run_agent import AIAgent

assert model_tools  # imported for its registration side effect
agent = AIAgent(base_url=INFER_URL or "http://127.0.0.1:9", api_key=INFER_KEY or "none", provider="custom", model=MODEL,
                quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                enabled_toolsets=TOOLSETS, max_iterations=20)



def budget_ladder():
    """What the stock budget ladder (no listing_servers pin) renders for this catalog at each budget."""
    from model_tools import get_tool_definitions
    from tools.tool_search import ToolSearchConfig, classify_tools
    from tools.tool_search_catalog import build_catalog_listing_with_form
    cfg = ToolSearchConfig.from_raw({})
    deferrable = classify_tools(get_tool_definitions(enabled_toolsets=TOOLSETS, quiet_mode=True, skip_tool_search_assembly=True),
                                cfg.effective_defer_tools)[1]
    rows = []
    for budget in (4000, 2200, 2100, 1500, 1000, 940, 900, 800, 700, 600, 500, 420, 380, 360, 340, 300, 200):
        text, form = build_catalog_listing_with_form(deferrable, max_tokens=budget)
        text = text or ""
        rows.append({"budget": budget, "form": form, "chars": len(text),
                     "builtin_lines_have_descriptions": "- browser_vault: " in text,
                     "servers_summarized": text.count("names not listed"),
                     "servers_names_only": sum(1 for ln in text.splitlines() if ln.count(", ") > 3 and not ln.startswith("- "))})
    return rows


if DUMP:
    tools = getattr(agent, "tools", None) or []
    names = [t["function"]["name"] for t in tools]
    desc = next((t["function"]["description"] for t in tools if t["function"]["name"] == "tool_search"), "")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"arm": ARM, "visible_tools": names, "tool_search_description": desc, "tools": tools,
                   "budget_ladder": budget_ladder()}, f, indent=1)
    print(json.dumps({"arm": ARM, "visible": names, "desc_chars": len(desc)}))
    shutil.rmtree(tmp_root, ignore_errors=True)
    sys.exit(0)

PREAMBLE = f"Your working directory (the workspace) is: {workspace}\n\nTask: "
t0, error, convo, roundtrips = time.time(), None, None, 0
try:
    convo = agent.run_conversation(PREAMBLE + TASK["prompt"].replace("{ws}", workspace))
    for _ in range(2):  # interactive fairness: answer a plain-text question like a user would
        msgs = (convo or {}).get("messages") or []
        last = next((m.get("content") or "" for m in reversed(msgs)
                     if m.get("role") == "assistant" and (m.get("content") or "").strip()), "")
        # A question anywhere in the reply counts: long clarifying replies put it first, options after.
        if "?" not in last or not REPLIES or any(e["tool"] in (TASK.get("targets") or ()) for e in LOG):
            break
        roundtrips += 1
        convo = agent.run_conversation(REPLIES.pop(0))
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


def _loads(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


# Model calls in order; each is the list of (kind, name) it requested. kind: eager | search | describe | call.
steps, searches, describes, attempted = [], 0, 0, []
for m in messages:
    if m.get("role") != "assistant":
        continue
    step = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        fargs = _loads(fn.get("arguments") or "{}")
        fargs = fargs if isinstance(fargs, dict) else {}
        if name == "tool_search":
            searches += 1
            step.append(("search", json.dumps(fargs.get("queries"))[:200]))
        elif name == "tool_describe":
            describes += 1
            step.append(("describe", json.dumps(fargs.get("names"))[:300]))
        elif name == "tool_call":
            entries = fargs.get("calls") or ([fargs] if fargs.get("name") else [])
            entries = _loads(entries) if isinstance(entries, str) else entries
            for entry in entries if isinstance(entries, list) else []:
                entry = _loads(entry) if isinstance(entry, str) else entry
                if isinstance(entry, dict):
                    step.append(("call", str(entry.get("name") or "")))
                    attempted.append(str(entry.get("name") or ""))
        else:
            step.append(("eager", name))
    steps.append(step)
final = next((m.get("content") or "" for m in reversed(messages)
              if m.get("role") == "assistant" and (m.get("content") or "").strip()), "")

targets = set(TASK.get("targets") or [])
executed = [e["tool"] for e in LOG]  # deferred tools that really ran (the oracle)
arg_ok = TASK.get("arg_ok") or (lambda a: True)
reached = any(e["tool"] in targets and arg_ok(e["args"]) for e in LOG)
first_correct_step = next((i + 1 for i, st in enumerate(steps) if any(k == "call" and n in targets for k, n in st)), None)
wrong = [n for n in executed if n not in targets and n not in NEUTRAL and n not in (TASK.get("neutral") or ())]
facts = TASK.get("facts") or []
fact_ok = all(any(alt.lower() in final.lower() for alt in (f if isinstance(f, (list, tuple)) else [f])) for f in facts)
file_ok = TASK["file_check"](workspace) if TASK.get("file_check") else True
if targets:
    score = 1.0 if (reached and fact_ok and file_ok) else 0.0
else:
    score = 1.0 if (fact_ok and file_ok) else 0.0
if error:
    score = 0.0
raw_xml_noise = "<function=" in final or "<tool_call>" in final

record = {
    "arm": ARM, "model": MODEL, "task": TASK_ID, "kind": TASK["kind"], "rep": REP, "score": score, "error": error,
    "target_reached": bool(reached) if targets else None, "fact_ok": fact_ok, "file_ok": file_ok,
    "first_correct_model_call": first_correct_step, "model_calls": len(steps),
    "wrong_tool_calls": len(wrong), "wrong_tools": wrong, "deferred_executed": executed, "attempted": attempted,
    "searches": searches, "describes": describes, "bridge_calls": searches + describes + len(attempted),
    "false_discovery": (searches + describes + len(attempted)) if not targets else None,
    "eager_calls": [n for st in steps for k, n in st if k == "eager"],
    "input_tokens": getattr(agent, "session_input_tokens", None),
    "cache_read_tokens": getattr(agent, "session_cache_read_tokens", None),
    "cache_write_tokens": getattr(agent, "session_cache_write_tokens", None),
    "output_tokens": getattr(agent, "session_output_tokens", None),
    "prompt_tokens": getattr(agent, "session_prompt_tokens", None),
    "completion_tokens": getattr(agent, "session_completion_tokens", None),
    "total_tokens": getattr(agent, "session_total_tokens", None),
    "cost_usd": getattr(agent, "session_estimated_cost_usd", None),
    "wall_s": round(wall, 1), "user_roundtrips": roundtrips, "raw_xml_noise": raw_xml_noise, "final": final[:1200],
}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT + ".transcript.json", "w", encoding="utf-8") as f:
    json.dump({"messages": messages, "log": LOG, "steps": steps}, f, default=str)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1, default=str)
print(json.dumps({k: record[k] for k in ("arm", "model", "task", "rep", "score", "target_reached", "model_calls",
                                         "first_correct_model_call", "wrong_tools", "searches", "prompt_tokens",
                                         "cache_read_tokens", "cost_usd", "wall_s", "error")}))
agent.close()
shutil.rmtree(tmp_root, ignore_errors=True)
