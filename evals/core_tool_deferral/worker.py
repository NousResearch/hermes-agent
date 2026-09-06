#!/usr/bin/env python3
"""Run ONE (arm, model, task, rep) cell of the PR #97979 A/B in an isolated process.

Usage: worker.py <arm:base|pr> <model_slug> <task_id> <rep> <out_json>
Env: OPENROUTER_API_KEY must be set. Exit 3 = infra/config error (do not score).
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

ARM, MODEL, TASK_ID, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
# Arm trees: plain checkouts of the two SHAs under test (git worktree/clone —
# NEVER `pip install -e .` from them). Set both env vars before running:
#   ABDEFER_BASE_TREE=/path/to/checkout-of-baseline-sha
#   ABDEFER_PR_TREE=/path/to/checkout-of-pr-sha
TREE = os.environ.get(f"ABDEFER_{ARM.upper()}_TREE") or ""
if not TREE or not os.path.isdir(TREE):
    print(f"ABORT: ABDEFER_{ARM.upper()}_TREE not set or not a directory", file=sys.stderr)
    sys.exit(3)
HARNESS = os.path.dirname(os.path.abspath(__file__))

if not os.environ.get("OPENROUTER_API_KEY"):
    print("ABORT: OPENROUTER_API_KEY missing", file=sys.stderr)
    sys.exit(3)

# --- hermetic env BEFORE any hermes import -------------------------------
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")) and var != "OPENROUTER_API_KEY":
        os.environ.pop(var, None)
os.environ.pop("FAL_KEY", None)
os.environ.pop("HERMES_PROFILE", None)

tmp_root = tempfile.mkdtemp(prefix=f"ab-{ARM}-{TASK_ID}-")
hermes_home = os.path.join(tmp_root, ".hermes")
workspace = os.path.join(tmp_root, "ws")
os.makedirs(hermes_home)
os.makedirs(workspace)
with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
    f.write("model:\n  provider: openrouter\n  model: %s\n" % MODEL)

os.environ["HERMES_HOME"] = hermes_home
os.environ["TERMINAL_CWD"] = workspace
os.chdir(workspace)
sys.path.insert(0, HARNESS)
sys.path.insert(0, TREE)

import tasks as taskmod  # noqa: E402
TASK = taskmod.TASKS_BY_ID[TASK_ID]

# --- seed session DB for recall tasks (both arms, always — cheap) ---------
def seed_sessions():
    from hermes_state import SessionDB
    db = SessionDB()
    month_ago = time.time() - 30 * 86400
    def sess(sid, msgs, t0):
        db.create_session(sid, source="cli")
        t = t0
        for role, content in msgs:
            db.append_message(sid, role, content=content, timestamp=t)
            t += 60
    sess("seed_shadow_vet", [
        ("user", "Back from the vet with Shadow. They put him on carprofen for the leg inflammation."),
        ("assistant", "Got it — what dose did they prescribe for Shadow?"),
        ("user", "Shadow's carprofen dose is 12.5 mg, twice a day with food. Two week course."),
        ("assistant", "Noted: Shadow takes 12.5 mg carprofen twice daily with food, for two weeks."),
    ], month_ago)
    sess("seed_backup_talk", [
        ("user", "I keep worrying about my repos. The sparks-data repo really needs nightly backups, it has irreplaceable training data."),
        ("assistant", "Agreed — sparks-data should get a nightly backup job. The toybox repo is scratch space so it can be skipped."),
        ("user", "Right, toybox doesn't matter. Just sparks-data."),
    ], month_ago + 3 * 86400)
    sess("seed_decoy_cat", [
        ("user", "My cat Biscuit is on 5 mg cetirizine for allergies."),
        ("assistant", "Noted — Biscuit: 5 mg cetirizine daily."),
    ], month_ago + 5 * 86400)
    sess("seed_decoy_dose", [
        ("user", "I bumped the server worker count from 8 to 25 mg— sorry, to 25 workers. Typo."),
        ("assistant", "25 workers, got it."),
    ], month_ago + 6 * 86400)
    db.close()

seed_sessions()

if TASK.get("fixtures"):
    TASK["fixtures"](workspace)

# --- stub the desktop / external surfaces ---------------------------------
EVENTS = []
CALLBACK_LOG = []

from tools import desktop_ui  # noqa: E402
desktop_ui.set_emitter(lambda sid, event, payload: EVENTS.append(
    {"sid": sid, "event": event, "payload": payload}))

FOCUSED = taskmod.FOCUSED_APP
PREVIEW_TITLE = taskmod.PREVIEW_TITLE
TERMINAL_TAIL = taskmod.TERMINAL_TAIL
WINDOW_BELOW = taskmod.WINDOW_BELOW_TEXT
IMG_URL = taskmod.IMG_URL

_clarify_answers = list(TASK.get("clarify_answers") or [])

def clarify_cb(question, choices, multi_select=False):
    CALLBACK_LOG.append({"name": "clarify", "question": question, "choices": choices})
    if _clarify_answers:
        ans = _clarify_answers.pop(0)
    else:
        ans = "Use your best judgement."
    if choices:
        for c in choices:
            if ans.lower() in str(c).lower():
                return str(c)
    return ans

def tour_cb(payload):
    CALLBACK_LOG.append({"name": "tour", "payload": payload})
    action = payload.get("action", "")
    if action == "targets":
        return json.dumps({"success": True, "targets": [
            {"selector": "[data-tour='settings']", "label": "Settings button", "stable": True},
            {"selector": "[data-tour='composer']", "label": "Message composer", "stable": True},
            {"selector": "[data-tour='sidebar']", "label": "Session sidebar", "stable": True},
            {"selector": "[data-tour='model-picker']", "label": "Model picker", "stable": True},
        ]})
    if action in ("start", "steps", "show"):
        return json.dumps({"success": True, "shown": True,
                           "steps_total": len(payload.get("steps") or []) or 1,
                           "completed": True})
    return json.dumps({"success": True, "action": action})

def read_terminal_cb(start=None, count=None):
    CALLBACK_LOG.append({"name": "read_terminal", "start": start, "count": count})
    lines = ["$ make build", "compiling core...", "linking...", TERMINAL_TAIL]
    return json.dumps({"total_lines": 4, "start": 0, "end": 3,
                       "viewport_rows": 24, "cursor_row": 3,
                       "text": "\n".join(lines)})

def read_preview_cb(start=None, count=None):
    CALLBACK_LOG.append({"name": "read_preview", "start": start, "count": count})
    return json.dumps({"title": PREVIEW_TITLE, "url": "https://example.com/docs/",
                       "text": ("Example Domain\nThis domain is for use in documents.\n"
                                "[Docs] link -> /docs/\nSearch: input#docs-search [ref=e12]\n")})

def drive_preview_cb(payload):
    CALLBACK_LOG.append({"name": "drive_preview", "payload": payload})
    action = payload.get("action", "")
    if "annotate" in json.dumps(payload) or action in ("highlight", "point", "underline", "clear", "hold"):
        return json.dumps({"success": True, "annotated": payload.get("selector") or payload.get("ref")})
    if action in ("click", "goto", "navigate"):
        return json.dumps({"success": True, "title": PREVIEW_TITLE,
                           "url": "https://example.com/docs/",
                           "text": "Docs index. Search box: input#docs-search [ref=e12]"})
    if action in ("snapshot", "read", "links"):
        return json.dumps({"success": True, "title": PREVIEW_TITLE,
                           "url": "https://example.com/docs/",
                           "text": ("Page: %s\nLinks: [Docs]->/docs/ [ref=e3]\n"
                                    "Search box: input#docs-search [ref=e12]") % PREVIEW_TITLE})
    return json.dumps({"success": True, "action": action, "title": PREVIEW_TITLE})

def read_window_below_cb(**kw):
    CALLBACK_LOG.append({"name": "read_window_below", "kw": kw})
    return json.dumps({"title": "Invoices — draft", "text": WINDOW_BELOW})

def setup_mcp_cb(name, action, reason):
    CALLBACK_LOG.append({"name": "setup_mcp", "server": name, "action": action})
    return json.dumps({"success": True, "server": name, "status": "installed"})

# --- import the tree's model_tools + patch registry stubs ------------------
import model_tools  # noqa: E402  (triggers registrations + plugin discovery)
from tools.registry import registry  # noqa: E402

def _stub_entry(name, handler):
    entry = registry.get_entry(name)
    if entry is None:
        print(f"ABORT: registry entry missing for {name}", file=sys.stderr)
        sys.exit(3)
    entry.handler = handler
    entry.check_fn = None
    entry.is_async = False

def computer_use_stub(args, **kw):
    CALLBACK_LOG.append({"name": "computer_use", "args": args})
    action = (args or {}).get("action", "screenshot")
    shot = os.path.join(tmp_root, "screen.png")
    with open(shot, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nstub")
    return json.dumps({
        "success": True, "action": action, "screenshot": shot,
        "analysis": ("Focused window: %s. It shows a note titled 'Shadow feeding "
                     "schedule' with a table of meal times. No error dialogs visible." % FOCUSED),
    })

def image_generate_stub(args, **kw):
    CALLBACK_LOG.append({"name": "image_generate", "args": args})
    return json.dumps({"success": True, "image": IMG_URL,
                       "prompt_used": (args or {}).get("prompt", "")})

_stub_entry("computer_use", computer_use_stub)
_stub_entry("image_generate", image_generate_stub)

# --- build agent -----------------------------------------------------------
TOOLSETS = ["file", "terminal", "search", "web", "todo", "session_search",
            "clarify", "image_gen", "computer_use", "cronjob", "memory",
            "desktop_ui", "project", "code_execution"]

from run_agent import AIAgent  # noqa: E402

agent = AIAgent(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    provider="openrouter",
    model=MODEL,
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    skip_background_review=True,
    enabled_toolsets=TOOLSETS,
    max_iterations=40,
    clarify_callback=clarify_cb,
    tour_callback=tour_cb,
    read_terminal_callback=read_terminal_cb,
    read_preview_callback=read_preview_cb,
    drive_preview_callback=drive_preview_cb,
    read_window_below_callback=read_window_below_cb,
    setup_mcp_callback=setup_mcp_cb,
)

PREAMBLE = ("You are running inside the Hermes desktop app on the user's machine. "
            "Your working directory (the workspace) is: %s\n\nTask: " % workspace)

t0 = time.time()
error = None
convo = None
user_roundtrips = 0
try:
    convo = agent.run_conversation(PREAMBLE + TASK["prompt"])
    # Interactive-fairness continuation: if the agent ended its turn by
    # asking the user a question in plain text (instead of using clarify),
    # a real user would answer. Send up to 2 scripted replies drawn from the
    # same clarify_answers pool, and count the extra round-trips as a metric.
    for _ in range(2):
        _msgs = (convo or {}).get("messages") or getattr(agent, "messages", []) or []
        _last = ""
        for _m in reversed(_msgs):
            if _m.get("role") == "assistant" and (_m.get("content") or "").strip():
                _last = _m["content"].strip()
                break
        if "?" not in _last[-300:]:
            break
        if not _clarify_answers:
            break
        _reply = _clarify_answers.pop(0)
        user_roundtrips += 1
        convo = agent.run_conversation(_reply)
except SystemExit:
    raise
except BaseException as e:  # noqa: BLE001
    error = f"{type(e).__name__}: {e}"
    traceback.print_exc()
wall = time.time() - t0

msg_txt = ""
if error and any(s in error for s in ("auth", "Authentication", "No LLM provider", "401")):
    print("ABORT: auth/config error: " + error, file=sys.stderr)
    sys.exit(3)

messages = (convo or {}).get("messages") or getattr(agent, "messages", []) or []

# --- metrics ----------------------------------------------------------------
LEGACY = {"todo": "todo_list", "cronjob": "cronjob_manage", "process": "process_manage",
          "tour": "gui_tour", "tip": "show_tip"}
tool_counts = {}
tool_args = {}
bridge_calls = 0
api_turns = 0
raw_xml_noise = False
for m in messages:
    if m.get("role") == "assistant":
        api_turns += 1
        if "<function=" in (m.get("content") or ""):
            raw_xml_noise = True
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                fargs = json.loads(fn.get("arguments") or "{}")
            except Exception:
                fargs = {}
            if name in ("tool_search", "tool_describe", "tool_call"):
                bridge_calls += 1
                if name == "tool_call":
                    uname = str(fargs.get("name") or "")
                    uargs = fargs.get("arguments") or {}
                    if isinstance(uargs, str):
                        try:
                            uargs = json.loads(uargs)
                        except Exception:
                            uargs = {}
                    uname = LEGACY.get(uname, uname)
                    if uname:
                        tool_counts[uname] = tool_counts.get(uname, 0) + 1
                        tool_args.setdefault(uname, []).append(uargs)
                continue
            cname = LEGACY.get(name, name)
            tool_counts[cname] = tool_counts.get(cname, 0) + 1
            tool_args.setdefault(cname, []).append(fargs)

final_answer = ""
for m in reversed(messages):
    if m.get("role") == "assistant" and (m.get("content") or "").strip():
        final_answer = m["content"]
        break

todo_dump = []
try:
    todo_dump = list(getattr(agent._todo_store, "_items", []))
except Exception:
    pass

ctx = {
    "workspace": workspace, "hermes_home": hermes_home,
    "events": EVENTS, "callback_log": CALLBACK_LOG,
    "tool_counts": tool_counts, "messages_tool_args": tool_args,
    "messages": messages, "final_answer": final_answer,
    "todo_dump": todo_dump,
}

score, notes = 0.0, ["run errored: %s" % error] if error else (0.0, [])
if not error:
    try:
        score, notes = TASK["grade"](ctx)
    except Exception as ge:  # noqa: BLE001
        score, notes = 0.0, [f"grader crashed: {ge}"]
else:
    score, notes = 0.0, ["run errored: %s" % error]

record = {
    "arm": ARM, "model": MODEL, "task": TASK_ID, "rep": REP,
    "score": round(float(score), 3), "notes": notes, "error": error,
    "api_turns": api_turns,
    "tool_calls_total": int(sum(tool_counts.values())) + bridge_calls,
    "bridge_calls": bridge_calls,
    "tool_counts": tool_counts,
    "prompt_tokens": getattr(agent, "session_prompt_tokens", None),
    "completion_tokens": getattr(agent, "session_completion_tokens", None),
    "total_tokens": getattr(agent, "session_total_tokens", None),
    "wall_s": round(wall, 1),
    "raw_xml_noise": raw_xml_noise,
    "user_roundtrips": user_roundtrips,
    "clarify_invocations": len([c for c in CALLBACK_LOG if c["name"] == "clarify"]),
    "final_answer": (final_answer or "")[:2000],
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT + ".transcript.json", "w", encoding="utf-8") as f:
    json.dump({"messages": messages, "events": EVENTS, "callback_log": CALLBACK_LOG},
              f, default=str)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1, default=str)
print(json.dumps({k: record[k] for k in ("arm", "model", "task", "rep", "score",
                                          "api_turns", "total_tokens", "wall_s",
                                          "bridge_calls", "error")}))
try:
    agent.close()
except Exception:
    pass
shutil.rmtree(tmp_root, ignore_errors=True)
