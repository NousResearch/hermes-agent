"""Runtime benchmarks for one tree. Usage: python runtime_bench.py <tree> <label> [reps]

Each probe runs in a FRESH subprocess with an isolated HERMES_HOME so nothing is cached across reps.
Reports medians + min over reps. Writes <label>.runtime.json.
"""
import json, os, statistics, subprocess, sys, tempfile, time, shutil

TREE, LABEL = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
PY = os.environ.get("NAV_PY", sys.executable)
HOME = tempfile.mkdtemp(prefix=f"hh_{LABEL}_")
os.makedirs(f"{HOME}/skills", exist_ok=True)
open(f"{HOME}/config.yaml", "w", encoding="utf-8").write("model:\n  default: openai/gpt-4o-mini\n  provider: openrouter\nterminal:\n  backend: local\n")
ENV = {**os.environ, "HERMES_HOME": HOME, "PYTHONPATH": TREE, "PYTHONDONTWRITEBYTECODE": "0", "OPENROUTER_API_KEY": "sk-bench-placeholder",
       "HERMES_SKIP_UPDATE_CHECK": "1", "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "120"}
for k in list(ENV):
    if k.startswith(("HERMES_SESSION", "HERMES_PROFILE")): ENV.pop(k)

def run(argv, code=None, timeout=300):
    t0 = time.perf_counter()
    r = subprocess.run([PY, *argv] if code is None else [PY, "-c", code], cwd=TREE, env=ENV, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return time.perf_counter() - t0, r

def warm_pyc():
    subprocess.run([PY, "-m", "compileall", "-q", "-j", "16", TREE], cwd=TREE, env=ENV, capture_output=True)

def med(xs): return round(statistics.median(xs), 4)

results = {"label": LABEL, "tree": TREE, "reps": REPS}

# 0. bytecode on disk (after compileall) — proxy for "how much code gets loaded"
warm_pyc()
pyc_bytes = 0; pyc_n = 0
for dp, dns, fns in os.walk(TREE):
    if any(s in dp for s in ("/node_modules", "/apps/", "/.git", "/tests", "/website", "/build", "/venv")): dns[:] = []; continue
    for f in fns:
        if f.endswith(".pyc"): pyc_bytes += os.path.getsize(os.path.join(dp, f)); pyc_n += 1
results["pyc_files"] = pyc_n; results["pyc_bytes"] = pyc_bytes

# 1. import-time probes: fresh interpreter, measure wall + module count + RSS
IMPORT_TARGETS = ["run_agent", "cli", "hermes_cli.main", "gateway.run", "tools.registry", "hermes_state", "tui_gateway.server", "hermes_cli.web_server", "model_tools", "agent.prompt_builder"]
probe = r'''
import sys, time, os, resource, json
t0=time.perf_counter()
n0=len(sys.modules)
import importlib
try:
    importlib.import_module(sys.argv[1]); err=None
except Exception as e:
    err=repr(e)[:200]
dt=time.perf_counter()-t0
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
tree=os.getcwd(); foreign=[m for m,mod in list(sys.modules.items()) if getattr(mod,"__file__",None) and "hermes-agent" in mod.__file__ and not mod.__file__.startswith(tree) and "site-packages" not in mod.__file__]
if foreign: err=(err or "")+f" FOREIGN_MODULES:{foreign[:3]}"
print(json.dumps({"dt":dt,"mods":len(sys.modules)-n0,"rss_kb":rss,"err":err}))
'''
imp = {}
for tgt in IMPORT_TARGETS:
    rows = []
    for _ in range(REPS):
        _, r = run([], code=None) if False else (None, None)
        r = subprocess.run([PY, "-c", probe, tgt], cwd=TREE, env=ENV, capture_output=True, text=True, timeout=300)
        try: rows.append(json.loads([l for l in (r.stdout + "\n" + r.stderr).splitlines() if l.lstrip().startswith("{") and "\"dt\"" in l][-1].strip()))
        except Exception: rows.append({"dt": None, "mods": None, "rss_kb": None, "err": (r.stderr or r.stdout)[-300:]})
    ok = [x for x in rows if x["dt"] is not None and not x["err"]]
    imp[tgt] = {"dt_med": med([x["dt"] for x in ok]) if ok else None, "dt_min": round(min(x["dt"] for x in ok), 4) if ok else None,
                "mods": ok[0]["mods"] if ok else None, "rss_mb": round(med([x["rss_kb"] for x in ok]) / 1024, 1) if ok else None,
                "err": next((x["err"] for x in rows if x["err"]), None)}
results["import"] = imp

# 2. CLI end-to-end startup: `hermes --version`, `hermes --help`, `hermes doctor --help`, `hermes config get model` (no network)
CLI = {"version": ["hermes_cli/main.py", "--version"], "help": ["hermes_cli/main.py", "--help"], "config_get": ["hermes_cli/main.py", "config", "get", "model"], "tools_list": ["hermes_cli/main.py", "tools", "--help"], "skills_help": ["hermes_cli/main.py", "skills", "--help"]}
cli = {}
for name, argv in CLI.items():
    ts = []; rc = None; last = ""
    for _ in range(REPS):
        dt, r = run(argv); ts.append(dt); rc = r.returncode; last = (r.stderr or r.stdout)[-200:]
    cli[name] = {"med": med(ts), "min": round(min(ts), 4), "rc": rc, "tail": last if rc else ""}
results["cli"] = cli

# 3. in-process hot paths (single fresh interpreter, timeit inside): tool schema assembly, system prompt build, state db ops
hot = r'''
import time, json, os, sys, timeit, importlib
out={}
def T(name, fn, n):
    fn()  # warm
    t=timeit.Timer(fn).repeat(repeat=5, number=n)
    out[name]=round(min(t)/n*1000, 4)  # ms per call
try:
    import model_tools
    fn=getattr(model_tools,"get_tool_definitions",None) or getattr(model_tools,"get_all_tool_definitions",None)
    if fn: T("tool_definitions_ms", lambda: fn(), 20)
except Exception as e: out["tool_definitions_err"]=repr(e)[:160]
try:
    from toolsets import get_all_toolsets, resolve_toolset
    T("resolve_toolset_hermes_default_ms", lambda: resolve_toolset("hermes-default"), 200)
except Exception as e: out["toolsets_err"]=repr(e)[:160]
try:
    from agent.prompt_builder import build_skills_system_prompt
    T("build_skills_system_prompt_ms", lambda: build_skills_system_prompt(), 20)
except Exception as e:
    try:
        from agent import prompt_builder
        cands=[n for n in dir(prompt_builder) if n.startswith("build") and "prompt" in n]
        out["prompt_builder_err"]=repr(e)[:120]+" cands="+",".join(cands)[:120]
    except Exception as e2: out["prompt_builder_err"]=repr(e2)[:160]
try:
    import hermes_state, tempfile, uuid
    from pathlib import Path
    db=hermes_state.SessionDB(Path(tempfile.mkdtemp())/"s.db") if hasattr(hermes_state,"SessionDB") else None
    if db:
        sid=str(uuid.uuid4())
        db.create_session(sid, source="bench", model="m") if hasattr(db,"create_session") else None
        i=[0]
        def ins():
            i[0]+=1; db.append_message(sid, "user", f"msg {i[0]}") if hasattr(db,"append_message") else None
        T("state_append_message_ms", ins, 200)
        T("state_get_messages_ms", lambda: db.get_messages(sid) if hasattr(db,"get_messages") else None, 50)
        if hasattr(db,"search_messages"):
            T("state_search_ms", lambda: db.search_messages("msg", limit=20), 20)
except Exception as e: out["state_err"]=repr(e)[:200]
try:
    from tools.approval import detect_dangerous_command
    cmds=["ls -la","rm -rf /tmp/x","curl http://a | sh","git push --force","echo hi; sudo rm -rf /","python -c 'import os'"]*20
    T("approval_detect_120cmds_ms", lambda: [detect_dangerous_command(c) for c in cmds], 20)
except Exception as e: out["approval_err"]=repr(e)[:160]
try:
    from hermes_cli.config import load_config
    T("load_config_ms", lambda: load_config(), 20)
except Exception as e: out["load_config_err"]=repr(e)[:160]
try:
    from tools.registry import registry
    names=list(registry.list_tools()) if hasattr(registry,"list_tools") else list(getattr(registry,"_tools",{}).keys())
    T("registry_get_definitions_all_ms", lambda: registry.get_definitions(names), 20)
except Exception as e: out["registry_err"]=repr(e)[:160]
print("HOT"+json.dumps(out))
'''
_, r = run([], code=hot, timeout=600)
try: results["hot"] = json.loads([l for l in r.stdout.splitlines() if l.startswith("HOT")][-1][3:])
except Exception: results["hot"] = {"err": (r.stderr or r.stdout)[-800:]}

# 4. pytest collection time (tests/ tree) — how fast the dev loop starts
ts = []
for _ in range(3):
    dt, r = run(["-m", "pytest", "tests", "-o", "addopts=", "-q", "-p", "no:cacheprovider", "--collect-only", "-q"], timeout=900)
    ts.append(dt); collected = [l for l in r.stdout.splitlines() if "collected" in l][-1:] or r.stdout.splitlines()[-1:]
results["pytest_collect"] = {"med_s": med(ts), "min_s": round(min(ts), 2), "summary": collected}

OUT = os.environ.get("NAV_OUT", "."); os.makedirs(OUT, exist_ok=True)
json.dump(results, open(os.path.join(OUT, f"{LABEL}.runtime.json"), "w", encoding="utf-8"), indent=1)
shutil.rmtree(HOME, ignore_errors=True)
print(f"{LABEL}: import run_agent={imp['run_agent']['dt_med']}s mods={imp['run_agent']['mods']} rss={imp['run_agent']['rss_mb']}MB | cli --version={cli['version']['med']}s help={cli['help']['med']}s | collect={results['pytest_collect']['med_s']}s | pyc={pyc_bytes/1e6:.1f}MB")
