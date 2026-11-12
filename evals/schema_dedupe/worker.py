#!/usr/bin/env python3
"""Run ONE (arm, model, task, rep) cell of the tool-schema dedupe A/B in an isolated process.

Usage: worker.py <arm:base|pr> <model_slug> <task_id> <rep> <out_json>
Env: SDD_INFERENCE_URL + SDD_INFERENCE_KEY (any OpenAI-compatible endpoint), SDD_BASE_TREE, SDD_PR_TREE.
Exit 3 = infra/config error (never scored).

The arms differ only in schema text. memory, skill_manage, search_files and terminal are the real tools
against a temp HERMES_HOME and workspace. manage_connections is stubbed at the registry handler: the real
target normalisation and action validation from the arm's tree run first (so a wrong-side call is refused
with the real message), then a canned settled result is returned. Scores come from the stores, the
workspace and the recorded calls, never from model prose alone.
"""
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
import urllib.request

ARM, MODEL, TASK_ID, REP, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
TREE = os.environ.get(f"SDD_{ARM.upper()}_TREE") or ""
if not TREE or not os.path.isdir(TREE):
    print(f"ABORT: SDD_{ARM.upper()}_TREE not set or not a directory", file=sys.stderr)
    sys.exit(3)
INFER_URL, INFER_KEY = os.environ.get("SDD_INFERENCE_URL") or "", os.environ.pop("SDD_INFERENCE_KEY", "")
if not (INFER_URL and INFER_KEY):
    print("ABORT: SDD_INFERENCE_URL / SDD_INFERENCE_KEY missing", file=sys.stderr)
    sys.exit(3)
for var in list(os.environ):
    if var.endswith(("_API_KEY", "_TOKEN")):
        os.environ.pop(var, None)
os.environ.pop("HERMES_PROFILE", None)

HARNESS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HARNESS, "..", "core_tool_deferral"))
from sandbox import isolate_host

# The arm name must not reach the model: the workspace path is in the prompt, and an early run of this
# harness with "base"/"pr" in the path saw two byte-identical schemas behave differently.
tmp_root = tempfile.mkdtemp(prefix="sdd-cell-")
hermes_home = os.path.join(tmp_root, ".hermes")
workspace = os.path.join(tmp_root, "ws")
os.makedirs(hermes_home)
os.makedirs(workspace)
isolate_host(tmp_root, hermes_home)
with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(f"model:\n  provider: custom\n  base_url: {INFER_URL}\n  model: {MODEL}\n")
os.environ["HERMES_HOME"] = hermes_home
# A model that reads "the user works inside tmux" starts one: keep its server on a socket inside the cell.
os.environ["TMUX_TMPDIR"] = tmp_root
os.environ.pop("TMUX", None)
os.environ["TERMINAL_CWD"] = workspace
os.chdir(workspace)
sys.path.insert(0, TREE)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
DELIM = "\n§\n"
MEM_DIR = os.path.join(hermes_home, "memories")

# Durable entries every memory task must leave in place.
CANARIES = ["Postgres 16", "ruff", "tmux"]
FULL_MEMORY = [  # ~2,150 of the 2,200-char limit
    "Production deploys go to Heroku (app acme-prod): `git push heroku main`, then check `heroku logs --tail` for boot errors.",
    "Primary datastore is Postgres 16 behind pgbouncer; migrations run with `alembic upgrade head` and must be backwards compatible for one release.",
    "Python style in this org: ruff with line length 120, no black; type hints required on public functions; pytest for tests.",
    "User works inside tmux on the dev box; long jobs go in a window named `jobs` so they survive SSH drops.",
    "TEMP 2025-03: team offsite planning — venue shortlist is Lisbon or Porto, budget cap 40k EUR, survey closes March 14. Remind about visa letters for two attendees.",
    "Old Jenkins box (jenkins-01.acme.internal) still runs the nightly perf suite; credentials are in the legacy vault path `ci/jenkins`; the box is slated for decommission.",
    "Frontend lives in `web/` (Vite + React 19); run `pnpm dev` there; API proxy points at localhost:8080; Storybook on 6006.",
    "The monorepo uses `just` recipes: `just test`, `just lint`, `just db-reset`; CI mirrors those recipes exactly, so reproduce CI failures with the same recipe.",
    "Feature flags are in `config/flags.yaml`; a flag must have an owner and an expiry date or the lint step fails. Expired flags are removed in the first week of each quarter.",
    "Incident notes go in `docs/incidents/YYYY-MM-DD-slug.md` using the five-section template (summary, timeline, impact, root cause, follow-ups); link the PR that fixed it.",
    "Observability: Grafana dashboards under the `acme-core` folder; alerts route to #ops-alerts; p95 latency SLO is 400 ms for the public API and 1.5 s for exports.",
    "Release cadence is weekly on Tuesdays; the release branch is cut Monday 16:00 UTC and only cherry-picks tagged `release-blocker` go in after the cut.",
    "Secrets come from Vault via `vault kv get secret/acme/<service>`; never commit `.env` files; the pre-commit hook scans for AWS and Stripe key shapes and blocks the commit when it finds one in a staged file.",
    "Docs site builds with Docusaurus from `website/`; broken links fail the build; screenshots live in `website/static/img` and must be under 300 KB each.",
]
LIGHT_MEMORY = FULL_MEMORY[1:4]
USER_SEED = ["User is a backend engineer at Acme; timezone is US Pacific."]

TASKS = {
    # ---- memory: nearly full store
    "mem_swap_stale": {
        "memory": FULL_MEMORY,
        "prompt": "Heads up: we moved production deploys off Heroku to Fly.io last week (app acme-prod, region iad, "
                  "deploy with `fly deploy`). Update your memory so it is right.",
    },
    "mem_full_add": {
        "memory": FULL_MEMORY,
        "prompt": "Save to your memory: the staging database is on host db-staging.acme.internal port 6432, and CI "
                  "runners are self-hosted on the 'forge' cluster. Your memory is nearly full; the offsite note and "
                  "the old Jenkins box note are obsolete.",
    },
    "mem_add_overflow": {
        "memory": FULL_MEMORY,
        "prompt": "Remember this for future sessions: the on-call runbook lives at https://wiki.acme.internal/oncall "
                  "and pager escalation is 15 minutes.",
    },
    # ---- memory: which store, and what does not belong in memory
    "mem_targets": {
        "memory": LIGHT_MEMORY,
        "prompt": "Two things to remember from now on: I go by Tek and I want terse answers with no preamble. Also, "
                  "on this dev box docker needs sudo.",
    },
    "mem_procedure_to_skill": {
        "memory": LIGHT_MEMORY,
        "prompt": "That release went fine. For next time, the docs-site release procedure is: 1) run `npm run build` in "
                  "website/, 2) run `npm run linkcheck`, 3) tag `docs-vX.Y`, 4) push the tag, 5) purge the CDN with "
                  "`./scripts/purge.sh docs`. Save it so you can do the whole thing next time without asking me.",
    },
    # ---- manage_connections (stubbed backend, real validation)
    "conn_hosted_and_mcp": {
        "prompt": "Connect my Notion account, and separately install the Linear MCP server locally.",
    },
    "conn_ambiguous_hosted": {"prompt": "Hook up GitHub for me."},
    "conn_mcp_words": {"prompt": "I want the GitHub MCP server set up."},
    "conn_second_account": {
        "prompt": "My personal Gmail is already connected. Add my work Gmail as a second account.",
    },
    "conn_switch_account": {
        "prompt": "Gmail is connected, but to the wrong Google account. Switch it to the right one.",
    },
    "conn_skip_no_reask": {
        "prompt": "Connect Slack, then post 'build is green' to the #eng channel.",
        "skip": ["slack"],
    },
    "conn_no_disconnect": {"prompt": "Disconnect my Notion account."},
    # ---- search_files / terminal: no-regression
    "search_content": {
        "prompt": "Somewhere under src/ a function named reconcile_ledger is defined. Which file is it in, and what "
                  "is the code on the line right after its def line?",
    },
    "search_newest": {"prompt": "Which .log file under logs/ was modified most recently? Give me just the file name."},
    "search_count": {"prompt": "How many lines under src/ contain the exact text TODO(tek)? Give me the total."},
    "term_wait_then_write": {
        "prompt": "Wait 10 seconds, then write the word done into status.txt in the workspace.",
    },
    "term_dev_server": {
        "prompt": (f"Start a static dev server for ./site on port {PORT} (python3 -m http.server is fine), leave it "
                   "running, and confirm it actually serves index.html."),
    },
}
TASK = TASKS[TASK_ID]


def fixtures():
    os.makedirs(MEM_DIR, exist_ok=True)
    with open(os.path.join(MEM_DIR, "MEMORY.md"), "w", encoding="utf-8") as f:
        f.write(DELIM.join(TASK.get("memory") or LIGHT_MEMORY))
    with open(os.path.join(MEM_DIR, "USER.md"), "w", encoding="utf-8") as f:
        f.write(DELIM.join(USER_SEED))
    src = os.path.join(workspace, "src")
    for pkg in ("billing", "ledger", "reports", "api"):
        os.makedirs(os.path.join(src, pkg))
        for i in range(6):
            body = [f'"""{pkg} module {i}."""', "", f"def {pkg}_helper_{i}(x):", f"    return x + {i}", ""]
            if (pkg, i) in {("billing", 1), ("billing", 4), ("reports", 2), ("api", 0), ("api", 5)}:
                body += [f"# TODO(tek): tighten validation in {pkg} {i}", "# TODO(sam): unrelated", ""]
            if (pkg, i) in {("billing", 4), ("api", 5)}:
                body += ["# TODO(tek): second note"]
            if (pkg, i) == ("ledger", 3):
                body += ["def reconcile_ledger(entries, tolerance=0.005):",
                         "    drift = sum(e.amount for e in entries) - entries.expected_total",
                         "    return abs(drift) <= tolerance", ""]
            if (pkg, i) == ("reports", 5):
                body += ["# calls reconcile_ledger(entries) before export", ""]
            with open(os.path.join(src, pkg, f"mod_{i}.py"), "w", encoding="utf-8") as f:
                f.write("\n".join(body) + "\n")
    logs = os.path.join(workspace, "logs")
    os.makedirs(os.path.join(logs, "archive"))
    now = time.time()
    for i in range(12):
        sub = "archive" if i % 3 == 0 else ""
        path = os.path.join(logs, sub, f"worker-{i:02d}.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"worker {i} started\n")
        age = 600 if i == 7 else 86400 * (2 + (i * 5) % 11)  # worker-07.log is the newest, not the last by name
        os.utime(path, (now - age, now - age))
    os.makedirs(os.path.join(workspace, "site"))
    with open(os.path.join(workspace, "site", "index.html"), "w", encoding="utf-8") as f:
        f.write("<h1>schema-dedupe fixture page</h1>\n")


fixtures()
TODO_TEK_LINES = 7  # 5 files with one line + 2 files with a second line

# --------------------------------------------------------------------------- manage_connections stub
CONN_LOG = []  # every call: {"args", "error"} in order


def manage_connections_stub(args, **kw):
    from hermes_agent.tools.connectors.managed import NOTE as HOSTED_NOTE
    from hermes_agent.tools.connectors.mcp import NOTE as MCP_NOTE
    from hermes_agent.tools.connectors.targets import MCP_ACTIONS, normalize_targets, validate_action
    from hermes_agent.tools.registry import tool_error

    args = args if isinstance(args, dict) else {}
    action = str(args.get("action") or "status").strip().lower()
    managed, mcp_targets, err = normalize_targets(args.get("connectors"))
    err = err or validate_action(action, managed, mcp_targets)
    CONN_LOG.append({"args": args, "error": err})
    if err:
        return tool_error(err)
    if action in ("status", "display"):
        return json.dumps({"connectors": [
            {"name": "gmail", "connected": True, "accounts": [{"label": "tek.personal@gmail.com", "alias": None}]},
            {"name": "notion", "connected": True, "accounts": [{"label": "Tek's workspace", "alias": None}]},
            {"name": "slack", "connected": False, "accounts": []},
            {"name": "github", "connected": False, "accounts": []},
            {"name": "linear", "connected": False, "accounts": []},
        ]})
    if action in MCP_ACTIONS:
        return json.dumps({"targets": [{"name": n, "kind": "mcp", "state": "connected",
                                        "tools": [f"mcp__{n}__search", f"mcp__{n}__create_issue"]} for n in mcp_targets],
                           "note": MCP_NOTE})
    targets = []
    for t in managed:
        state = "skipped" if t.name in (TASK.get("skip") or []) else "connected"
        row = {"name": t.name, "kind": "connector", "state": state}
        if t.alias:
            row["alias"] = t.alias
        if action == "reconnect" and not args.get("force") and state == "connected":
            row["detail"] = "already connected; nothing to repair"
        targets.append(row)
    return json.dumps({"targets": targets, "note": HOSTED_NOTE})


from hermes_agent import model_tools  # registers the built-in tools
import hermes_agent.tools.connectors as connectors_pkg
import hermes_agent.tools.connectors.tool as connectors_tool
from hermes_agent.tools.connectors.gateway import config as gateway_config
from hermes_agent.tools.registry import registry


def _blocked(*a, **kw):
    raise RuntimeError("schema_dedupe eval: the real connector backend is blocked")


# The agent loop runs manage_connections inline (hermes_agent/agent/inline_tool_executors.py imports it from
# tools.connectors at call time), so the package attribute is the seam; the registry handler is the
# fallback path. The real backends are blocked outright: a missed seam must fail the cell, never mint
# a connect link on the operator's account.
connectors_pkg.manage_connections = manage_connections_stub
connectors_tool.run_managed_action = connectors_tool.run_mcp_operation = _blocked
gateway_config.connectors_available = lambda: True
entry = registry.get_entry("manage_connections")
if entry is None:
    print("ABORT: manage_connections is not registered in this tree", file=sys.stderr)
    sys.exit(3)
entry.handler, entry.check_fn, entry.is_async = manage_connections_stub, None, False

# Each family gets the tools it needs plus file/terminal as competition; a full default surface would
# only add the same ~10k tokens to every call of both arms.
FAMILY_TOOLSETS = {"mem": ["memory", "skills", "file", "terminal"],
                   "conn": ["connections", "file", "terminal", "clarify"],
                   "search": ["file", "terminal"], "term": ["file", "terminal"]}
TOOLSETS = FAMILY_TOOLSETS[TASK_ID.split("_", 1)[0]]

from hermes_agent.run_agent import AIAgent

agent = AIAgent(base_url=INFER_URL, api_key=INFER_KEY, provider="custom", model=MODEL, quiet_mode=True,
                skip_context_files=True, skip_memory=False, skip_background_review=True, enabled_toolsets=TOOLSETS,
                max_iterations=20, clarify_callback=lambda qs: {
                    "answers": {q["qid"]: "Use your best judgement." for q in qs}, "outcome": "submitted"})
SCHEMA_CHARS = {t["function"]["name"]: len(json.dumps(t["function"])) for t in agent.tools or []}

PREAMBLE = f"Your working directory (the workspace) is: {workspace}\n\nTask: "
t0, error, convo = time.time(), None, None
try:
    convo = agent.run_conversation(PREAMBLE + TASK["prompt"])
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
    except (TypeError, ValueError):
        return None


calls, results, turns = [], {}, 0
for m in messages:
    if m.get("role") == "tool":
        results[m.get("tool_call_id")] = m.get("content") or ""
    if m.get("role") != "assistant":
        continue
    turns += 1
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        fargs = _loads(fn.get("arguments") or "{}")
        calls.append((fn.get("name", ""), fargs if isinstance(fargs, dict) else {}, tc.get("id")))
final = next((m.get("content") or "" for m in reversed(messages)
              if m.get("role") == "assistant" and (m.get("content") or "").strip()), "")
if sum(1 for n, _, _ in calls if n == "manage_connections") != len(CONN_LOG):
    print("ABORT: a manage_connections call bypassed the stub", file=sys.stderr)
    sys.exit(3)


def read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _ok(call_id):
    parsed = _loads(results.get(call_id) or "")
    return isinstance(parsed, dict) and parsed.get("success") is True


def memory_stats():
    mem = [(a, cid) for n, a, cid in calls if n == "memory"]
    ok = [(a, cid) for a, cid in mem if _ok(cid)]
    ops = lambda a: a.get("operations") if isinstance(a.get("operations"), list) else []
    first_ok = ok[0][0] if ok else {}
    return {"memory_calls": len(mem), "memory_ok_calls": len(ok), "memory_rejected": len(mem) - len(ok),
            "first_ok_ops": [o.get("action") for o in ops(first_ok) if isinstance(o, dict)],
            # a second successful write to the SAME target after one already succeeded
            "repeat_writes": sum(1 for i, (a, _) in enumerate(ok) if any(b.get("target") == a.get("target") for b, _ in ok[:i]))}


def conn_targets(args):
    raw = args.get("connectors")
    raw = [raw] if isinstance(raw, (str, dict)) else raw if isinstance(raw, list) else []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append({"name": str(item.get("name") or "").lower(), "mcp": bool(item.get("mcp")),
                        "alias": item.get("alias")})
        else:
            out.append({"name": str(item or "").lower(), "mcp": False, "alias": None})
    return out


def conn_accepted(action, name, *, mcp):
    """Accepted calls with this action naming this target on this side."""
    return [c["args"] for c in CONN_LOG if not c["error"] and str(c["args"].get("action") or "status").lower() == action
            and any(t["name"] == name and t["mcp"] == mcp for t in conn_targets(c["args"]))]


def grade():  # health: allow CC -- one flat oracle per task; splitting it hides the battery
    notes = []
    memory, user = read(os.path.join(MEM_DIR, "MEMORY.md")), read(os.path.join(MEM_DIR, "USER.md"))
    kept = all(c in memory for c in CANARIES)
    if TASK_ID.startswith("mem_"):
        notes.append(f"memory_chars={len(memory)} canaries_kept={kept} {memory_stats()}")
    if TASK_ID == "mem_swap_stale":
        # A note that the app moved OFF Heroku is fine; the stale deploy instruction must be gone.
        stale = "deploys go to Heroku" in memory or "git push heroku" in memory
        return (1.0 if "Fly" in memory and not stale and kept else 0.0), notes
    if TASK_ID == "mem_full_add":
        ok = "6432" in memory and "forge" in memory and "Jenkins" not in memory and "offsite" not in memory and kept
        return (1.0 if ok else 0.5 if "6432" in memory and "forge" in memory and kept else 0.0), notes
    if TASK_ID == "mem_add_overflow":
        return (1.0 if "wiki.acme.internal/oncall" in memory and kept else 0.0), notes
    if TASK_ID == "mem_targets":
        user_ok = "tek" in user.lower() and "terse" in user.lower()
        mem_ok = "sudo" in memory.lower() and "docker" in memory.lower()
        notes.append(f"user_ok={user_ok} mem_ok={mem_ok}")
        return (1.0 if user_ok and mem_ok else 0.5 if user_ok or mem_ok else 0.0), notes
    if TASK_ID == "mem_procedure_to_skill":
        in_skill = False
        for root, _, files in os.walk(os.path.join(hermes_home, "skills")):
            in_skill = in_skill or any("purge" in read(os.path.join(root, fn)) for fn in files)
        in_memory = "purge" in memory or "purge" in user
        notes.append(f"in_skill={in_skill} in_memory={in_memory}")
        return (1.0 if in_skill and not in_memory else 0.5 if in_skill else 0.0), notes

    if TASK_ID.startswith("conn_"):
        bad = sum(1 for c in CONN_LOG if c["error"])
        notes.append(f"calls={[(str(c['args'].get('action') or 'status'), conn_targets(c['args']), bool(c['error'])) for c in CONN_LOG]} refused={bad}")
    if TASK_ID == "conn_hosted_and_mcp":
        hosted, mcp = bool(conn_accepted("connect", "notion", mcp=False)), bool(conn_accepted("install", "linear", mcp=True))
        return (1.0 if hosted and mcp else 0.5 if hosted or mcp else 0.0), notes
    if TASK_ID == "conn_ambiguous_hosted":
        return (1.0 if conn_accepted("connect", "github", mcp=False) and not conn_accepted("install", "github", mcp=True)
                else 0.0), notes
    if TASK_ID == "conn_mcp_words":
        return (1.0 if conn_accepted("install", "github", mcp=True) and not conn_accepted("connect", "github", mcp=False)
                else 0.0), notes
    if TASK_ID == "conn_second_account":
        aliased = [a for a in conn_accepted("connect", "gmail", mcp=False)
                   if any(t["name"] == "gmail" and t["alias"] for t in conn_targets(a))]
        return (1.0 if aliased else 0.0), notes
    if TASK_ID == "conn_switch_account":
        forced = [a for a in conn_accepted("reconnect", "gmail", mcp=False) if a.get("force") is True]
        return (1.0 if forced else 0.0), notes
    if TASK_ID == "conn_skip_no_reask":
        asks = len(conn_accepted("connect", "slack", mcp=False)) + len(conn_accepted("reconnect", "slack", mcp=False))
        return (1.0 if asks == 1 else 0.0), notes
    if TASK_ID == "conn_no_disconnect":
        acted = [c for c in CONN_LOG if str(c["args"].get("action") or "status").lower() not in ("status", "display")]
        pointed = any(w in final.lower() for w in ("portal", "desktop app", "connectors page"))
        return (1.0 if not acted and pointed else 0.5 if not acted else 0.0), notes

    if TASK_ID == "search_content":
        return (1.0 if "mod_3.py" in final and "ledger" in final and "expected_total" in final else 0.0), notes
    if TASK_ID == "search_newest":
        others = [f"worker-{i:02d}.log" for i in range(12) if i != 7]
        return (1.0 if "worker-07.log" in final and not any(o in final for o in others) else 0.0), notes
    if TASK_ID == "search_count":
        import re
        return (1.0 if re.search(rf"(?<![\d.]){TODO_TEK_LINES}(?![\d.])", final) else 0.0), notes
    if TASK_ID == "term_wait_then_write":
        bg_sleep = any(n == "terminal" and a.get("background") and "sleep" in str(a.get("command")) for n, a, _ in calls)
        notes.append(f"background_sleep={bg_sleep}")
        done = "done" in read(os.path.join(workspace, "status.txt"))
        return (1.0 if done and not bg_sleep else 0.5 if done else 0.0), notes
    if TASK_ID == "term_dev_server":
        try:
            body = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/index.html", timeout=5).read().decode()
        except OSError as e:
            notes.append(f"server not serving after the run: {e}")
            return 0.0, notes
        return (1.0 if "schema-dedupe fixture page" in body else 0.0), notes
    return 0.0, ["no grader"]


score, notes = (0.0, [f"run errored: {error}"]) if error else grade()
usage = {k: getattr(agent, f"session_{k}", None) for k in (
    "api_calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "prompt_tokens", "completion_tokens",
    "total_tokens", "estimated_cost_usd")}
record = {"arm": ARM, "model": MODEL, "task": TASK_ID, "rep": REP, "score": score, "notes": notes, "error": error,
          "api_turns": turns, "tool_calls": [n for n, _, _ in calls], **usage,
          "memory": memory_stats() if TASK_ID.startswith("mem_") else None,
          "conn_refused": sum(1 for c in CONN_LOG if c["error"]), "schema_chars": SCHEMA_CHARS,
          "wall_s": round(wall, 1), "final": final[:600]}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT + ".transcript.json", "w", encoding="utf-8") as f:
    json.dump({"messages": messages, "conn_log": CONN_LOG}, f, default=str)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1, default=str)
print(json.dumps({k: record[k] for k in ("arm", "model", "task", "rep", "score", "api_turns", "total_tokens",
                                         "estimated_cost_usd", "wall_s", "error")}))
agent.close()
# The dev-server task leaves real processes behind (background terminal, nohup, tmux). Kill, by PID,
# everything still running from inside this cell's directory.
for pid in [int(d) for d in os.listdir("/proc") if d.isdigit() and int(d) != os.getpid()]:
    try:
        if os.readlink(f"/proc/{pid}/cwd").startswith(tmp_root):
            os.kill(pid, 15)
    except OSError:
        continue
shutil.rmtree(tmp_root, ignore_errors=True)
