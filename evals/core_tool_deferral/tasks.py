"""Task battery for PR #97979 core-tool-deferral A/B.

Covers all 19 deferred tools:
  computer_use, session_search, clarify, image_generate, todo_list,
  process_manage, cronjob_manage, drive_preview, gui_tour, desktop_preview,
  annotate_preview, show_tip, setup_mcp, desktop_project, close_terminal,
  apply_layout, read_terminal, read_window_below, focus_pane
plus an eager-surface control and a false-discovery distractor.

Each task: id, prompt, fixtures(workspace), grade(ctx) -> (score 0..1, notes).
ctx keys: workspace, hermes_home, events (desktop emit records),
callback_log (agent-callback invocations), tool_counts, messages,
final_answer, todo_dump.
"""
import json
import os
import re


# Planted ground truths
FOCUSED_APP = "Obsidian — Shadow Care Notes"
MED_FACT = "12.5"
BACKUP_REPO = "sparks-data"
TERMINAL_TAIL = "BUILD OK #4471"
WINDOW_BELOW_TEXT = "Draft invoice for Fern & Forge — total $1,240"
PREVIEW_TITLE = "Example Domain — Docs"
IMG_URL = "https://img.eval.local/fern-forge.png"


def _w(ws, rel, content):
    p = os.path.join(ws, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)


def _read(ws, rel):
    p = os.path.join(ws, rel)
    try:
        with open(p, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _called(ctx, name):
    return ctx["tool_counts"].get(name, 0) > 0


def _grep_home(ctx, needle):
    """Binary-safe grep of the entire HERMES_HOME tree (sqlite included)."""
    nb = needle.encode()
    for root, _dirs, files in os.walk(ctx["hermes_home"]):
        for fn in files:
            try:
                with open(os.path.join(root, fn), "rb") as f:
                    if nb in f.read():
                        return True
            except OSError:
                continue
    return False


def _events(ctx, ev_prefix):
    return [e for e in ctx["events"] if e["event"].startswith(ev_prefix)]


def _cb(ctx, name):
    return [c for c in ctx["callback_log"] if c["name"] == name]


# ---------------------------------------------------------------------------
# graders
# ---------------------------------------------------------------------------

def g_screenshot(ctx):
    score, notes = 0.0, []
    if _called(ctx, "computer_use"):
        score += 0.5
    else:
        notes.append("computer_use never called")
    if FOCUSED_APP.split(" — ")[0].lower() in (ctx["final_answer"] or "").lower():
        score += 0.5
    else:
        notes.append("answer does not name focused app")
    return score, notes


def g_recall(ctx):
    score, notes = 0.0, []
    if _called(ctx, "session_search"):
        score += 0.4
    else:
        notes.append("session_search never called")
    ans = ctx["final_answer"] or ""
    if MED_FACT in ans and "carprofen" in ans.lower():
        score += 0.6
    elif MED_FACT in ans:
        score += 0.45
        notes.append("dose found, drug name missing")
    else:
        notes.append("dose not in answer")
    return score, notes


def g_clarify(ctx):
    ws = ctx["workspace"]
    staging_gone = _read(ws, "configs/staging.yaml") is None
    prod_intact = _read(ws, "configs/prod.yaml") is not None
    asked = _called(ctx, "clarify")
    score, notes = 0.0, []
    if asked:
        score += 0.4
    else:
        notes.append("clarify never called — acted without asking")
    if staging_gone and prod_intact:
        score += 0.6
    elif not prod_intact:
        score = 0.0
        notes.append("FATAL: prod.yaml deleted")
        return score, notes
    else:
        notes.append("staging.yaml still present")
    return score, notes


def g_image(ctx):
    score, notes = 0.0, []
    if _called(ctx, "image_generate"):
        score += 0.5
    else:
        notes.append("image_generate never called")
    logo = _read(ctx["workspace"], "logo.txt") or ""
    if IMG_URL in logo:
        score += 0.5
    else:
        notes.append("logo.txt missing planted URL")
    return score, notes


def g_todo_multi(ctx):
    ws = ctx["workspace"]
    score, notes = 0.0, []
    if _called(ctx, "todo_list"):
        score += 0.4
    else:
        notes.append("todo_list never called")
    td = json.dumps(ctx.get("todo_dump") or [])
    if td.count("completed") >= 3:
        score += 0.15
    else:
        notes.append("fewer than 3 completed todo items")
    checks = [
        (_read(ws, "greet.py") or ""), (_read(ws, "notes/summary.md") or ""),
        (_read(ws, "data/rows.csv") or ""),
    ]
    if "def greet" in checks[0] and "hello" in checks[0].lower():
        score += 0.15
    else:
        notes.append("greet.py wrong")
    if "3 files" in checks[1] or "three" in checks[1].lower() or "3" in checks[1]:
        score += 0.15
    else:
        notes.append("summary.md wrong")
    if checks[2].strip().count("\n") == 2 and "widget" in checks[2]:
        score += 0.15
    else:
        notes.append("rows.csv wrong")
    return score, notes


def g_cron(ctx):
    score, notes = 0.0, []
    if _called(ctx, "cronjob_manage"):
        score += 0.4
    else:
        notes.append("cronjob_manage never called")
    if _grep_home(ctx, "15 7 * * 1-5"):
        score += 0.4
    else:
        notes.append("weekday 7:15 cron expression not persisted")
    if _grep_home(ctx, "inbox"):
        score += 0.2
    else:
        notes.append("job prompt does not reference inbox")
    return score, notes


def g_process(ctx):
    import socket
    score, notes = 0.0, []
    used_pm = _called(ctx, "process_manage")
    if used_pm:
        score += 0.3
    else:
        notes.append("process_manage never called (may have used raw shell)")
    ans = (ctx["final_answer"] or "").lower()
    if any(k in ans for k in ("dead", "killed", "terminated", "stopped", "no longer running")):
        score += 0.3
    else:
        notes.append("answer does not confirm termination")
    s = socket.socket()
    try:
        s.settimeout(1.0)
        s.connect(("127.0.0.1", 8123))
        notes.append("port 8123 STILL LISTENING")
        alive = True
    except OSError:
        alive = False
    finally:
        s.close()
    if not alive:
        score += 0.4
    return score, notes


def g_tour(ctx):
    score, notes = 0.0, []
    tour_used = _called(ctx, "gui_tour") or bool(_cb(ctx, "tour"))
    tip_used = _called(ctx, "show_tip") or bool(_events(ctx, "tip.show"))
    if tour_used:
        score += 0.45
    else:
        notes.append("gui_tour never used")
    if tip_used:
        score += 0.35
    else:
        notes.append("show_tip never used")
    if "settings" in (ctx["final_answer"] or "").lower():
        score += 0.2
    else:
        notes.append("answer does not mention settings")
    return score, notes


def g_layout(ctx):
    score, notes = 0.0, []
    if _called(ctx, "apply_layout") or _events(ctx, "layout"):
        score += 0.25
    else:
        notes.append("apply_layout never used")
    if _called(ctx, "focus_pane") or _events(ctx, "focus"):
        score += 0.2
    else:
        notes.append("focus_pane never used")
    if _called(ctx, "read_terminal") or _cb(ctx, "read_terminal"):
        score += 0.2
    else:
        notes.append("read_terminal never used")
    if TERMINAL_TAIL in (ctx["final_answer"] or ""):
        score += 0.2
    else:
        notes.append("terminal tail not reported")
    if _called(ctx, "close_terminal") or _events(ctx, "terminal.close"):
        score += 0.15
    else:
        notes.append("close_terminal never used")
    return score, notes


def g_preview(ctx):
    score, notes = 0.0, []
    if _called(ctx, "desktop_preview") or _events(ctx, "preview"):
        score += 0.25
    else:
        notes.append("desktop_preview never used")
    if _called(ctx, "drive_preview") or _cb(ctx, "drive_preview"):
        score += 0.25
    else:
        notes.append("drive_preview never used")
    if _called(ctx, "annotate_preview") or _events(ctx, "annotate"):
        score += 0.15
    else:
        notes.append("annotate_preview never used")
    if _called(ctx, "read_window_below") or _cb(ctx, "read_window_below"):
        score += 0.15
    else:
        notes.append("read_window_below never used")
    ans = ctx["final_answer"] or ""
    if PREVIEW_TITLE in ans:
        score += 0.1
    else:
        notes.append("page title not reported")
    if "1,240" in ans or "1240" in ans:
        score += 0.1
    else:
        notes.append("window-below content not reported")
    return score, notes


def g_project(ctx):
    score, notes = 0.0, []
    proj_calls = [c for c in ctx["messages_tool_args"].get("desktop_project", [])
                  if "apollo" in json.dumps(c).lower()]
    if _called(ctx, "desktop_project"):
        score += 0.3
        if proj_calls:
            score += 0.2
        else:
            notes.append("desktop_project called but not with 'apollo'")
    else:
        notes.append("desktop_project never called")
    mcp_calls = [c for c in ctx["messages_tool_args"].get("setup_mcp", [])
                 if "github" in json.dumps(c).lower()]
    if _called(ctx, "setup_mcp"):
        score += 0.3
        if mcp_calls:
            score += 0.2
        else:
            notes.append("setup_mcp called but not for github")
    else:
        notes.append("setup_mcp never called")
    return score, notes


def g_longrange(ctx):
    ws = ctx["workspace"]
    score, notes = 0.0, []
    if _called(ctx, "session_search"):
        score += 0.15
    else:
        notes.append("session_search never called")
    sh = _read(ws, "backup.sh") or ""
    if BACKUP_REPO in sh and ("tar" in sh or "rsync" in sh or "zip" in sh):
        score += 0.25
    elif BACKUP_REPO in sh:
        score += 0.15
        notes.append("backup.sh names repo but no archive command")
    else:
        notes.append("backup.sh missing or wrong repo")
    if _called(ctx, "cronjob_manage") and (_grep_home(ctx, "0 2 * * *") or _grep_home(ctx, "2am") or _grep_home(ctx, "02:00")):
        score += 0.25
    elif _called(ctx, "cronjob_manage"):
        score += 0.1
        notes.append("cron created but 2am schedule not found")
    else:
        notes.append("cronjob_manage never called")
    if _called(ctx, "todo_list"):
        score += 0.15
    else:
        notes.append("todo_list never used")
    if BACKUP_REPO in (ctx["final_answer"] or ""):
        score += 0.2
    else:
        notes.append("answer does not name the repo")
    return score, notes


def g_control(ctx):
    ws = ctx["workspace"]
    score, notes = 0.0, []
    svc = _read(ws, "src/service.py") or ""
    if "timeout=45" in svc.replace(" ", ""):
        score += 0.4
    else:
        notes.append("timeout not updated to 45")
    if "timeout=30" in svc.replace(" ", ""):
        notes.append("old timeout=30 still present")
        score -= 0.1
    changelog = _read(ws, "CHANGELOG.md") or ""
    if "45" in changelog and ("timeout" in changelog.lower()):
        score += 0.3
    else:
        notes.append("CHANGELOG entry missing")
    ans = ctx["final_answer"] or ""
    if "3" in ans:  # three call sites
        score += 0.3
    else:
        notes.append("call-site count not reported")
    return max(0.0, score), notes


def g_distractor(ctx):
    score, notes = 0.0, []
    ans = ctx["final_answer"] or ""
    if "7" in ans:
        score += 1.0
    else:
        notes.append("retry_limit value not found")
    bridge = sum(ctx["tool_counts"].get(n, 0) for n in ("tool_search", "tool_describe", "tool_call"))
    if bridge:
        notes.append(f"bridge_calls={bridge} (false-discovery overhead)")
    return score, notes


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def f_clarify(ws):
    _w(ws, "configs/staging.yaml", "env: staging\nreplicas: 1\n# unused since migration\n")
    _w(ws, "configs/prod.yaml", "env: prod\nreplicas: 8\n")


def f_todo(ws):
    _w(ws, "README.md", "workspace for the three-file task\n")


def f_cron(ws):
    _w(ws, "notes/inbox.md", "- call vet\n- renew domain\n")


def f_longrange(ws):
    _w(ws, "repos/sparks-data/data.txt", "important\n")
    _w(ws, "repos/toybox/junk.txt", "junk\n")


def f_control(ws):
    _w(ws, "src/service.py",
       "import requests\n\n"
       "def fetch_a(url):\n    return requests.get(url, timeout=30)\n\n"
       "def fetch_b(url):\n    return requests.post(url, timeout=30)\n\n"
       "def fetch_c(url):\n    return requests.head(url, timeout=30)\n")
    _w(ws, "src/util.py", "def helper():\n    return 1\n")
    _w(ws, "CHANGELOG.md", "# Changelog\n\n## 1.0\n- initial\n")


def f_distractor(ws):
    _w(ws, "conf/app.ini", "[net]\nretry_limit = 7\nbackoff = 2\n")
    _w(ws, "conf/old.ini", "[net]\n# retry_limit moved to app.ini\n")


TASKS = [
    dict(id="screenshot_ambiguous", timeout=420,
         prompt=("Something looks off on my machine right now — take a look at my screen "
                 "and tell me which application window is focused and what it's showing."),
         fixtures=None, grade=g_screenshot,
         clarify_answers=["Just tell me what's focused."]),
    dict(id="recall_med_dose", timeout=420,
         prompt=("I can't remember what I told you last month about my dog Shadow's medication dose. "
                 "Look it up — don't guess, I need the exact dose from our past conversation."),
         fixtures=None, grade=g_recall,
         clarify_answers=["The medication dose we discussed before."]),
    dict(id="ambiguous_delete_ask", timeout=420,
         prompt=("In configs/ there are two config files; one of them is obsolete and must be deleted. "
                 "You cannot tell which from the files alone — check with me first, then delete the right one."),
         fixtures=f_clarify, grade=g_clarify,
         clarify_answers=["staging", "staging.yaml is the obsolete one — delete it."]),
    dict(id="logo_image", timeout=420,
         prompt=("Create a square logo image for the 'Fern & Forge' bakery (rustic, fern motif). "
                 "Then save the resulting image URL into a file named logo.txt in the workspace."),
         fixtures=None, grade=g_image,
         clarify_answers=["Rustic green, no text in the image."]),
    dict(id="todo_three_files", timeout=600,
         prompt=("This is a multi-step job — track it with your todo checklist tool and keep it updated as you go, "
                 "marking each step completed when verified. Steps: (1) create greet.py containing a greet(name) "
                 "function that returns 'hello <name>'; (2) create data/rows.csv with header 'item,qty' and exactly two "
                 "data rows for widgets; (3) create notes/summary.md stating how many files you created. "
                 "Finish only when all three are done and checked off."),
         fixtures=f_todo, grade=g_todo_multi,
         clarify_answers=["Whatever sensible defaults."]),
    dict(id="weekday_cron", timeout=420,
         prompt=("Every weekday at 7:15am I want a summary of what's in notes/inbox.md sent to me. "
                 "Set that up so it actually happens on schedule."),
         fixtures=f_cron, grade=g_cron,
         clarify_answers=["Weekdays only, 7:15am local time."]),
    dict(id="bg_server_lifecycle", timeout=600,
         prompt=("Start `python3 -m http.server 8123` as a background process, verify it responds on "
                 "http://127.0.0.1:8123/, then shut it down and prove to me it is no longer running."),
         fixtures=None, grade=g_process,
         clarify_answers=["Yes, kill it after verifying."]),
    dict(id="gui_onboarding", timeout=420,
         prompt=("I'm brand new to this desktop app. Point out the Settings button for me right on the screen, "
                 "and then walk me through a short 2-step guided tour of the composer and the sidebar."),
         fixtures=None, grade=g_tour,
         clarify_answers=["Just the composer and sidebar."]),
    dict(id="layout_terminal_readout", timeout=420,
         prompt=("Switch my workspace to the split layout, focus the terminal pane, tell me the last line the "
                 "terminal printed, and then close that terminal pane."),
         fixtures=None, grade=g_layout,
         clarify_answers=["The embedded terminal pane in the app."]),
    dict(id="preview_inspect_chain", timeout=600,
         prompt=("Open https://example.com/docs in the app's preview pane and click the 'Docs' link. "
                 "Tell me the page title you end up on. Then draw a highlight around the search box on that page. "
                 "Finally, check the window right below our chat and tell me what it says."),
         fixtures=None, grade=g_preview,
         clarify_answers=["The in-app preview pane, not an external browser."]),
    dict(id="project_mcp_setup", timeout=420,
         prompt=("Set up a fresh desktop project workspace named 'apollo', and get the github MCP server "
                 "installed for me so it's available in that project."),
         fixtures=None, grade=g_project,
         clarify_answers=["Name it exactly apollo, lowercase."]),
    dict(id="longrange_backup_pipeline", timeout=900,
         prompt=("A while back I told you one of my repos needed nightly backups — find which repo that was in our "
                 "past conversations (do not guess). Then: write backup.sh in the workspace that archives that repo "
                 "directory under repos/, schedule it to run nightly at 2am, and track the whole job with your todo "
                 "checklist. Report back which repo it was and what you set up."),
         fixtures=f_longrange, grade=g_longrange,
         clarify_answers=["Trust what you find in our history."]),
    dict(id="eager_refactor_control", timeout=600,
         prompt=("In src/, every requests call uses timeout=30. Bump them all to timeout=45, add a CHANGELOG.md "
                 "entry describing the change, and tell me exactly how many call sites you changed."),
         fixtures=f_control, grade=g_control,
         clarify_answers=["All of them."]),
    dict(id="config_grep_distractor", timeout=420,
         prompt=("Search this workspace for wherever the retry_limit setting is configured and tell me its "
                 "current value."),
         fixtures=f_distractor, grade=g_distractor,
         clarify_answers=["The active config, not the old one."]),
]

TASKS_BY_ID = {t["id"]: t for t in TASKS}
