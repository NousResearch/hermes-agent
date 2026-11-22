#!/usr/bin/env python3
"""Run ONE (arm, model, task, rep) cell of the text_to_speech / manage_connections deferral A/B.

Usage: worker.py <arm:base|pr|A|B|C|D> <model_slug> <task_id> <rep> <out_json>
Env: TCD_INFERENCE_URL + TCD_INFERENCE_KEY (any OpenAI-compatible endpoint), TCD_BASE_TREE, TCD_PR_TREE.
Exit 3 = infra/config error (never scored).

Arms base/pr compare two trees. Arms A-D run ONE tree (TCD_TREE) and differ only in the cell's own
config and one sentence: A = manage_connections eager, B = deferred, C = deferred with the reworded
first sentence (the catalog line), D = eager with the reworded sentence. Task id ``__dry__`` makes no
model call: it proves the stubs and the sandbox and dumps the tool surface.

The only thing that differs between the arms is whether the two tools are sent as schemas (base) or
sit behind tool_search (pr). Both tools, the hosted connector search/describe legs and the hosted
connector call are deterministic stubs, so the score measures whether the model REACHES the tool and
with which arguments, not speech synthesis or OAuth plumbing. Terminal and files are real.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

ARM, MODEL, TASK_ID, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
# arm -> (manage_connections deferred, first sentence reworded)
ARMS = {"A": (False, False), "B": (True, False), "C": (True, True), "D": (False, True)}
OLD_LEAD, NEW_LEAD = "Connect the user to apps.", "Connect the user's app accounts and install MCP servers."
TREE = os.environ.get("TCD_TREE" if ARM in ARMS else f"TCD_{ARM.upper()}_TREE") or ""
if not TREE or not os.path.isdir(TREE):
    print("ABORT: the arm's tree env var is not set or not a directory", file=sys.stderr)
    sys.exit(3)
INFER_URL, INFER_KEY = os.environ.get("TCD_INFERENCE_URL") or "", os.environ.pop("TCD_INFERENCE_KEY", "")
if not (INFER_URL and INFER_KEY):
    print("ABORT: TCD_INFERENCE_URL / TCD_INFERENCE_KEY missing", file=sys.stderr)
    sys.exit(3)
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(var, None)
os.environ.pop("HERMES_PROFILE", None)

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS, "..", "core_tool_deferral"))
from sandbox import isolate_host

tmp_root = os.environ.get("TCD_CELL_ROOT") or tempfile.mkdtemp(prefix="tcd-cell-")  # never the arm name: the model sees it
hermes_home = os.path.join(tmp_root, ".hermes")
workspace = os.path.join(tmp_root, "ws")
os.makedirs(hermes_home, exist_ok=True)
os.makedirs(workspace, exist_ok=True)
isolate_host(tmp_root, hermes_home)
# A model that cannot see manage_connections reaches for the host's `hermes` CLI (`hermes mcp install`,
# `hermes pm repair`); keep that out of the operator's install.
with open(os.path.join(tmp_root, "bin", "hermes"), "w", encoding="utf-8") as f:
    f.write("#!/bin/sh\necho 'hermes: the CLI is not available inside this eval sandbox' >&2\nexit 127\n")
os.chmod(os.path.join(tmp_root, "bin", "hermes"), 0o755)
os.environ["HERMES_HOME"] = hermes_home
os.environ["TERMINAL_CWD"] = workspace
os.chdir(workspace)
sys.path.insert(0, TREE)
defer_yaml = ""
if ARM in ARMS:  # every arm pins the list, so the only difference is the one name
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    defer = [n for n in DEFAULT_CONFIG["tools"]["tool_search"]["defer"] if n != "manage_connections"]
    defer_yaml = "tools:\n  tool_search:\n    defer: [" + ", ".join(defer + ["manage_connections"] * ARMS[ARM][0]) + "]\n"
with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(f"model:\n  provider: custom\n  base_url: {INFER_URL}\n  model: {MODEL}\n{defer_yaml}")

NOTES = "Dentist moved to Thursday 3pm.\nPick up the rye bread from Larder.\nCall Priya about the cabin keys.\n"
POEM = "The fog comes\non little cat feet.\nIt sits looking\nover harbor and city\non silent haunches\nand then moves on.\n"

# --------------------------------------------------------------------------- tasks
# kind: which stub the task is about ("tts" / "conn"), None for controls. expect: must the stub be reached.
TASKS = {
    "tts_read_aloud": {"kind": "tts", "expect": True, "needle": "quick brown fox",
                       "prompt": "Read this aloud for me: 'The quick brown fox jumps over the lazy dog.'"},
    "tts_voice_note": {"kind": "tts", "expect": True, "needle": "birthday",
                       "prompt": "Send me a voice note wishing my mom a happy birthday. Keep it to one sentence."},
    "tts_indirect_driving": {"kind": "tts", "expect": True, "needle": "thursday",
                             "prompt": "I'm driving and can't look at the screen. Tell me what's in notes.txt so I can hear it."},
    "tts_indirect_mp3": {"kind": "tts", "expect": True, "needle": "cat feet",
                         "prompt": "Can you make me an audio file of the poem in poem.txt? I want to listen to it later."},
    "tts_negative": {"kind": "tts", "expect": False, "final": "thursday",
                     "prompt": "When is the dentist appointment according to notes.txt? One sentence."},
    "conn_connect_gmail": {"kind": "conn", "expect": True, "prompt": "Connect my Gmail.", "connected": []},
    "conn_status_notion": {"kind": "conn", "expect": True, "prompt": "Is my Notion connected?", "connected": ["gmail"]},
    "conn_required_recovery": {
        "kind": "conn", "expect": True, "connected": [],
        "prompt": "Send an email from my Gmail to sam@example.com, subject 'Lunch', body 'Noon works for me.'"},
    "conn_mcp_install": {"kind": "conn", "expect": True, "prompt": "Install the Linear MCP server.", "connected": [],
                         "mcp": "linear"},
    # MCP installs that do not say "install ... MCP server": the catalog line has to carry them.
    "mcp_add_linear_server": {"kind": "conn", "expect": True, "prompt": "Add the Linear server.", "connected": [],
                              "mcp": "linear"},
    "mcp_setup_github": {"kind": "conn", "expect": True, "prompt": "Set up the GitHub MCP.", "connected": [], "mcp": "github"},
    # Either side is a right answer here (hosted account or MCP server); the terminal is not.
    "mcp_talk_sentry": {"kind": "conn", "expect": True, "prompt": "I want Hermes to talk to Sentry.", "connected": [],
                        "mcp": "sentry", "hosted_ok": True},
    "conn_second_account": {
        "kind": "conn", "expect": True, "connected": ["gmail"], "replies": ["Call it work.", "Yes, go ahead."],
        "prompt": "My personal Gmail is already connected. Add my work Gmail as a second account."},
    # No scripted reply: a connection string ends in "?sslmode=..." and would read as a question.
    "conn_negative": {"kind": "conn", "expect": False, "final": "postgres", "connected": [], "replies": [],
                      "prompt": "What does a PostgreSQL connection string look like? Answer in two lines, no tools needed."},
    "mcp_negative": {"kind": "conn", "expect": False, "final": "model context protocol", "connected": [], "replies": [],
                     "prompt": "In one sentence: what does MCP stand for, and what is an MCP server? No tools needed."},
    "ctl_write_file": {"kind": None, "prompt": "Create hello.txt in the workspace containing just the word hello."},
    "ctl_count_lines": {"kind": None, "final": "3", "prompt": "How many lines are in notes.txt? Check with the terminal."},
}
DRY = TASK_ID == "__dry__"
TASK = {"kind": "conn", "expect": True, "prompt": "", "connected": []} if DRY else TASKS[TASK_ID]
REPLIES = list(TASK["replies"] if "replies" in TASK else ["Yes, go ahead."])

with open(os.path.join(workspace, "notes.txt"), "w", encoding="utf-8") as f:
    f.write(NOTES)
with open(os.path.join(workspace, "poem.txt"), "w", encoding="utf-8") as f:
    f.write(POEM)

# --------------------------------------------------------------------------- stubs
STATE = {"connected": set(TASK.get("connected") or []), "email_sent": False, "connection_required_seen": False,
         "accounts": {app: ["tek@nous.test"] for app in TASK.get("connected") or []}}
TTS_CALLS, CONN_CALLS, LOG = [], [], []


def tts_stub(args, **kw):
    args = args or {}
    text = args.get("text")
    ok = isinstance(text, str) and bool(text.strip())
    TTS_CALLS.append({"args": args, "valid": ok})
    LOG.append({"tool": "text_to_speech", "args": args})
    if not ok:
        return json.dumps({"success": False, "error": "text is required"})
    path = os.path.join(workspace, f"speech_{len(TTS_CALLS)}.ogg")
    with open(path, "wb") as fh:
        fh.write(b"OggS-eval-stub")
    return json.dumps({"success": True, "file_path": path, "media_tag": f"[[audio_as_voice]]\nMEDIA:{path}",
                       "provider": "edge", "voice_compatible": True})


def conn_stub(args, **kw):
    """Arguments are checked by the arm's own validators; the lifecycle is scripted."""
    from tools.connectors.targets import MCP_ACTIONS, normalize_targets, validate_action
    args = args or {}
    action = str(args.get("action") or "status").strip().lower()
    managed, mcp_targets, err = normalize_targets(args.get("connectors"))
    err = err or validate_action(action, managed, mcp_targets)
    CONN_CALLS.append({"args": args, "valid": not err, "action": action})
    LOG.append({"tool": "manage_connections", "args": args, "error": err})
    if err:
        return json.dumps({"error": err})
    names = [getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else str(t)) for t in managed]
    if action in MCP_ACTIONS:
        return json.dumps({"targets": [{"name": str(n), "kind": "mcp", "state": "connected"} for n in mcp_targets]})
    if action in ("status", "display"):
        apps = names or ["gmail", "notion", "slack", "googlecalendar"]
        return json.dumps({"connectors": [{"name": n, "connected": n in STATE["connected"],
                                           "accounts": [{"name": a} for a in STATE["accounts"].get(n, [])]}
                                          for n in apps]})
    if action in ("connect", "reconnect"):
        STATE["connected"].update(names)
        for target in managed:  # a new alias is a second account, as in the real tool
            label = getattr(target, "alias", None) or "tek@nous.test"
            accounts = STATE["accounts"].setdefault(target.name, [])
            if label not in accounts:
                accounts.append(label)
        return json.dumps({"targets": [{"name": n, "kind": "hosted", "state": "connected"} for n in names]})
    return json.dumps({"targets": [{"name": n, "kind": "hosted", "state": "renamed"} for n in names]})


GMAIL_SEND = {"connector": "gmail", "description": "Send an email from the user's Gmail account.",
              "input_schema": {"type": "object", "properties": {
                  "recipient_email": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                  "required": ["recipient_email", "subject", "body"]}}
NOTION_SEARCH = {"connector": "notion", "description": "Search pages in the user's Notion workspace.",
                 "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
REMOTE = {"GMAIL_SEND_EMAIL": GMAIL_SEND, "NOTION_SEARCH_PAGES": NOTION_SEARCH}


def search_stub(queries, **kw):
    from tools.connectors.gateway.bridge import ConnectorLeg
    results, schemas = [], {}
    for q in queries:
        text = str(q.get("use_case") or "").lower()
        slugs = [s for s, sch in REMOTE.items()
                 if sch["connector"] in text or (sch["connector"] == "gmail" and ("email" in text or "mail" in text))]
        schemas.update({s: REMOTE[s] for s in slugs})
        results.append({"use_case": q.get("use_case"), "tools": slugs})
    return ConnectorLeg(payload={"schemas": schemas, "results": results})


def describe_stub(names, **kw):
    from tools.connectors.gateway.bridge import ConnectorLeg
    known = {"connectors__gmail__SEND_EMAIL": GMAIL_SEND, "connectors__notion__SEARCH_PAGES": NOTION_SEARCH}
    return ConnectorLeg(payload={"tools": {n: {"description": known[n]["description"], "parameters": known[n]["input_schema"]}
                                           for n in names if n in known}})


def connector_call_stub(name, arguments, tool_call_id=None, turn_id=None, task_id=None):
    from tools.connectors.gateway.errors import render_connection_required
    from tools.connectors.turn import CARD
    app = name.split("__")[1]
    LOG.append({"tool": name, "args": arguments, "connected": app in STATE["connected"]})
    if app not in STATE["connected"]:
        STATE["connection_required_seen"] = True
        return json.dumps({"error": render_connection_required(connector=app, surface=CARD)})
    if name.endswith("SEND_EMAIL"):
        STATE["email_sent"] = "sam@example.com" in json.dumps(arguments)
        return json.dumps({"response": {"id": "msg_1", "status": "sent"}})
    return json.dumps({"response": {"results": []}})


import model_tools  # registers the built-in tools
import tools.connectors as connectors_pkg
import tools.connectors.dispatch as connectors_dispatch
import tools.connectors.gateway.bridge as connectors_bridge
import tools.connectors.gateway.config as gateway_config
import tools.connectors.tool as connectors_tool
from tools.registry import registry

for name, handler in (("text_to_speech", tts_stub), ("manage_connections", conn_stub)):
    entry = registry.get_entry(name)
    entry.handler, entry.check_fn, entry.is_async = handler, None, False
    # The ElevenLabs schema variant depends on the host's keys; pin the default schema on both arms.
    if getattr(entry, "dynamic_schema_overrides", None) is not None:
        entry.dynamic_schema_overrides = None
gateway_config.connectors_available = lambda: True
# manage_connections runs inline in the agent loop (never through the registry handler).
connectors_tool.manage_connections = connectors_pkg.manage_connections = conn_stub
connectors_dispatch.dispatch_connector_call = connectors_pkg.dispatch_connector_call = connector_call_stub
connectors_bridge.connector_search_hits = connectors_pkg.connector_search_hits = search_stub
connectors_bridge.connector_describe = connectors_pkg.connector_describe = describe_stub



def _real_backend_reached(*a, **kw):
    raise RuntimeError("eval sandbox: the real connection backend must never run")


# Backstop: if a caller ever bypasses the stub above, the real managed/MCP backends raise instead of acting.
connectors_tool.run_managed_action = connectors_tool.run_mcp_operation = _real_backend_reached
if ARM in ARMS:  # every arm pins its sentence: the tree's own default may be either one
    want = NEW_LEAD if ARMS[ARM][1] else OLD_LEAD
    for schema in (connectors_tool.MANAGE_CONNECTIONS_SCHEMA, registry.get_entry("manage_connections").schema):
        for lead in (NEW_LEAD, OLD_LEAD):
            if schema["description"].startswith(lead):
                schema["description"] = want + schema["description"][len(lead):]
                break
    if not registry.get_entry("manage_connections").schema["description"].startswith(want):
        print("ABORT: the arm's first sentence did not land in the registry schema", file=sys.stderr)
        sys.exit(3)

TOOLSETS = ["clarify", "code_execution", "cronjob", "file", "memory", "session_search", "skills", "terminal",
            "todo", "vision", "image_gen", "tts", "connections"]

from run_agent import AIAgent

agent = AIAgent(base_url=INFER_URL, api_key=INFER_KEY, provider="custom", model=MODEL, quiet_mode=True, skip_context_files=True,
                skip_memory=True, skip_background_review=True, enabled_toolsets=TOOLSETS, max_iterations=20,
                clarify_callback=lambda qs: {
                    "answers": {q["qid"]: (REPLIES[0] if REPLIES else "Use your best judgement.") for q in qs},
                    "outcome": "submitted"})
sent_names = [(t.get("function") or {}).get("name") for t in (agent.tools or [])]
_fn = {(t.get("function") or {}).get("name"): (t.get("function") or {}) for t in (agent.tools or [])}
SURFACE = {"tools_sent": sent_names,
           "catalog_line": next((ln for ln in (_fn.get("tool_search", {}).get("description") or "").splitlines()
                                 if "manage_connections:" in ln), None),
           "eager_lead": (_fn.get("manage_connections", {}).get("description") or "")[:80] or None}

if DRY:
    import socket
    from types import SimpleNamespace

    from agent.inline_tool_executors import _manage_connections
    from tools.terminal_tool import terminal_tool

    real_home = os.environ.get("TCD_REAL_HERMES_HOME") or "/nonexistent"
    proof = {"arm": ARM, "surface": SURFACE, "tools": agent.tools, "hermes_home": hermes_home, "env_home": os.environ["HOME"],
             "real_home_visible": {n: os.path.exists(os.path.join(real_home, n))
                                   for n in ("config.yaml", "auth.json", ".env", "state.db", "mcp-tokens", "hermes-agent")},
             "real_home_listing": sorted(os.listdir(real_home)) if os.path.isdir(real_home) else None, "net": {}}
    for host in ("api.notion.com", "mcp.linear.app", "backend.composio.dev", "1.1.1.1", "github.com"):
        try:
            socket.create_connection((host, 443), timeout=4).close()
            proof["net"][host] = "CONNECTED"
        except OSError as exc:
            proof["net"][host] = f"blocked: {type(exc).__name__}: {exc}"
    ctx = SimpleNamespace(effective_task_id="dry", tool_call_id="dry", messages=None)
    proof["inline_install"] = _manage_connections(agent, {"action": "install", "connectors": [{"name": "linear", "mcp": True}]}, ctx)
    proof["inline_connect"] = _manage_connections(agent, {"action": "connect", "connectors": ["notion"]}, ctx)
    proof["registry_connect"] = model_tools.handle_function_call("manage_connections", {"action": "connect", "connectors": ["gmail"]})
    proof["bridge_search"] = model_tools.handle_function_call("tool_search", {"queries": ["send gmail email"]}, enabled_toolsets=TOOLSETS)
    proof["bridge_describe"] = model_tools.handle_function_call("tool_describe", {"names": ["connectors__notion__SEARCH_PAGES"]}, enabled_toolsets=TOOLSETS)
    proof["bridge_call"] = model_tools.handle_function_call(
        "tool_call", {"calls": [{"name": "connectors__notion__SEARCH_PAGES", "arguments": {"query": "x"}}]}, enabled_toolsets=TOOLSETS)
    try:
        connectors_tool.run_mcp_operation([], "install")
        proof["real_backend"] = "RAN"
    except RuntimeError as exc:
        proof["real_backend"] = f"raises: {exc}"
    proof["terminal"] = terminal_tool(
        "command -v hermes; hermes mcp install linear; echo rc=$?; echo HOME=$HOME; ls -a " + real_home + "; cat " + real_home
        + "/config.yaml; curl -sS -m 5 -o /dev/null -w '%{http_code}' https://mcp.linear.app/mcp; echo curl_rc=$?", task_id="dry")
    proof["stub_log"], proof["conn_calls"] = LOG, len(CONN_CALLS)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(proof, f, indent=1, default=str)
    # The dry cell is the gate: a visible credential file, a reachable host or a live backend fails it.
    leaks = ([n for n, seen in proof["real_home_visible"].items() if seen]
             + [h for h, state in proof["net"].items() if state == "CONNECTED"] + ["real_backend"] * (proof["real_backend"] == "RAN"))
    if leaks:
        print(f"ABORT: sandbox leak in the dry cell: {leaks}", file=sys.stderr)
        sys.exit(4)
    sys.exit(0)

PREAMBLE = f"Your working directory (the workspace) is: {workspace}\n\nTask: "
t0, error, convo, roundtrips = time.time(), None, None, 0
try:
    convo = agent.run_conversation(PREAMBLE + TASK["prompt"])
    for _ in range(2):  # interactive fairness: answer a plain-text question like a user would
        msgs = (convo or {}).get("messages") or []
        last = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "assistant" and (m.get("content") or "").strip()), "")
        if "?" not in last or not REPLIES:
            break
        roundtrips += 1
        convo = agent.run_conversation(REPLIES.pop(0), conversation_history=msgs)
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


calls, bridge, turns = [], 0, 0
for m in messages:
    if m.get("role") != "assistant":
        continue
    turns += 1
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        fargs = _loads(fn.get("arguments") or "{}")
        fargs = fargs if isinstance(fargs, dict) else {}
        if name in ("tool_search", "tool_describe", "tool_call"):
            bridge += 1
            if name == "tool_call":
                entries = fargs.get("calls") or ([fargs] if fargs.get("name") else [])
                entries = _loads(entries) if isinstance(entries, str) else entries
                for entry in entries if isinstance(entries, list) else []:
                    entry = _loads(entry) if isinstance(entry, str) else entry
                    if isinstance(entry, dict):
                        calls.append(str(entry.get("name") or ""))
            continue
        calls.append(name)
final = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "assistant" and (m.get("content") or "").strip()), "")
# Bridge-side schema refusals never reach the stub; count them so "valid on first call" is honest.
_REFUSED = ("failed argument validation", "The tool was NOT invoked", "must be an object")
refusals = sum(1 for m in messages if m.get("role") == "tool" and any(s in str(m.get("content")) for s in _REFUSED))
stub_calls = TTS_CALLS if TASK["kind"] == "tts" else CONN_CALLS if TASK["kind"] == "conn" else []
if calls.count("manage_connections") > len(CONN_CALLS) + refusals:
    print("ABORT: manage_connections was called more often than the stub recorded", file=sys.stderr)
    sys.exit(3)
# Wrong route: on a connection task nothing but the connection tools (and asking the user) is needed.
WRONG_ROUTE_TOOLS = ("terminal", "execute_code", "write_file", "patch", "process_manage", "cronjob_manage")
wrong_route = [c for c in calls if c in WRONG_ROUTE_TOOLS] if TASK["kind"] == "conn" and TASK["expect"] else []
reached = bool(stub_calls)
first_valid = reached and stub_calls[0]["valid"] and refusals == 0


def grade():
    notes = []
    if TASK["kind"] is None:
        if TASK_ID == "ctl_write_file":
            try:
                with open(os.path.join(workspace, "hello.txt"), encoding="utf-8") as fh:
                    return (1.0 if fh.read().strip() == "hello" else 0.0), notes
            except OSError:
                return 0.0, notes
        return (1.0 if TASK["final"] in final else 0.0), notes
    if not TASK["expect"]:  # negative: answer correctly without touching the tool
        notes.append(f"false_trigger={reached}")
        return (1.0 if not reached and TASK["final"] in final.lower() else 0.0), notes
    if TASK["kind"] == "tts":
        spoken = " ".join(str(c["args"].get("text") or "") for c in TTS_CALLS).lower()
        notes.append(f"tts_calls={len(TTS_CALLS)}")
        return (1.0 if reached and TASK["needle"] in spoken else 0.5 if reached else 0.0), notes
    valid = [c for c in CONN_CALLS if c["valid"]]
    targets = json.dumps([c["args"].get("connectors") for c in valid]).lower()
    notes.append(f"conn_actions={[c['action'] for c in CONN_CALLS]}")
    if TASK_ID == "conn_connect_gmail":
        return (1.0 if any(c["action"] == "connect" for c in valid) and "gmail" in STATE["connected"] else 0.5 if reached else 0.0), notes
    if TASK_ID == "conn_status_notion":
        low = final.lower()
        told = "not connected" in low or "isn't connected" in low or "is not" in low or "not yet" in low or low.startswith("no")
        return (1.0 if valid and told else 0.5 if valid else 0.0), notes
    if TASK_ID == "conn_required_recovery":
        notes.append(f"required_seen={STATE['connection_required_seen']} sent={STATE['email_sent']}")
        connected = any(c["action"] in ("connect", "reconnect") for c in valid) and "gmail" in STATE["connected"]
        return (1.0 if connected and STATE["email_sent"] else 0.5 if connected else 0.0), notes
    if TASK.get("mcp"):
        def _is_mcp(t):
            return isinstance(t, dict) and t.get("mcp") is True and TASK["mcp"] in str(t.get("name")).lower()

        def _is_hosted(t):
            return TASK["mcp"] in str(t.get("name") if isinstance(t, dict) else t).lower() and not (isinstance(t, dict) and t.get("mcp"))
        mcp_ok = any(c["action"] == "install" and any(_is_mcp(t) for t in c["args"].get("connectors") or []) for c in valid)
        hosted_ok = TASK.get("hosted_ok") and any(
            c["action"] == "connect" and any(_is_hosted(t) for t in c["args"].get("connectors") or []) for c in valid)
        notes.append(f"shape={'mcp' if mcp_ok else 'hosted' if hosted_ok else 'none'}")
        return (1.0 if mcp_ok or hosted_ok else 0.5 if reached else 0.0), notes
    if TASK_ID == "conn_second_account":
        ok = any(c["action"] == "connect" and "alias" in json.dumps(c["args"].get("connectors")) for c in valid)
        return (1.0 if ok and "gmail" in targets else 0.5 if reached else 0.0), notes
    return 0.0, ["no grader"]


score, notes = (0.0, [f"run errored: {error}"]) if error else grade()
record = {"arm": ARM, "model": MODEL, "task": TASK_ID, "rep": REP, "score": score, "notes": notes, "error": error,
          "kind": TASK["kind"], "expect": TASK.get("expect"), "reached": reached, "first_call_valid": first_valid,
          "eager": {n: n in sent_names for n in ("text_to_speech", "manage_connections")},
          "catalog_line": SURFACE["catalog_line"], "eager_lead": SURFACE["eager_lead"],
          "wrong_route": wrong_route, "searched": any(
              tc.get("function", {}).get("name") == "tool_search" for m in messages for tc in (m.get("tool_calls") or [])),
          "api_turns": turns, "api_calls": getattr(agent, "session_api_calls", None), "bridge_calls": bridge, "tool_calls": calls,
          "input_tokens": getattr(agent, "session_input_tokens", None),
          "cache_read_tokens": getattr(agent, "session_cache_read_tokens", None),
          "cache_write_tokens": getattr(agent, "session_cache_write_tokens", None),
          "output_tokens": getattr(agent, "session_output_tokens", None),
          "prompt_tokens": getattr(agent, "session_prompt_tokens", None),
          "total_tokens": getattr(agent, "session_total_tokens", None),
          "cost_usd": getattr(agent, "session_estimated_cost_usd", None),
          "cost_status": getattr(agent, "session_cost_status", None),
          "wall_s": round(wall, 1), "user_roundtrips": roundtrips, "final": final[:800]}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT + ".transcript.json", "w", encoding="utf-8") as f:
    json.dump({"messages": messages, "log": LOG}, f, default=str)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1, default=str)
print(json.dumps({k: record[k] for k in ("arm", "model", "task", "rep", "score", "reached", "api_calls", "total_tokens",
                                         "cost_usd", "wall_s", "bridge_calls", "error")}))
agent.close()
if not os.environ.get("TCD_CELL_ROOT"):
    shutil.rmtree(tmp_root, ignore_errors=True)
