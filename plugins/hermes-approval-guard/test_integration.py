"""Integration test for hermes-approval-guard plugin."""

import sys, os, importlib.util

PLUGIN_DIR = os.path.expanduser("~/.hermes/plugins/hermes-approval-guard")
passed = failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1; print(f"  ✅ {name}")
    else:
        failed += 1; print(f"  ❌ {name} — {detail}")

def _load(name, filename):
    full = f"hermes_approval_guard.{name}"
    spec = importlib.util.spec_from_file_location(full, os.path.join(PLUGIN_DIR, filename), submodule_search_locations=[PLUGIN_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    if "hermes_approval_guard" not in sys.modules:
        pkg = type(sys)("hermes_approval_guard")
        pkg.__file__ = PLUGIN_DIR; pkg.__path__ = [PLUGIN_DIR]; pkg.__package__ = "hermes_approval_guard"
        sys.modules["hermes_approval_guard"] = pkg
    mod.__package__ = "hermes_approval_guard"
    spec.loader.exec_module(mod)
    return mod

# Load modules
feedback    = _load("feedback", "feedback.py")
stage1_rules = _load("stage1_rules", "stage1_rules.py")
stage1_llm   = _load("stage1_llm", "stage1_llm.py")
guard        = _load("guard", "guard.py")
pre_tool_call_handler = guard.pre_tool_call_handler
fast_path = stage1_rules.fast_path

# ═══ TESTS 1-8: Rule engine and delegation ═══

print("═══ 1. SAFE_TOOLS ═══")
for t in ["read_file","web_search","skill_view","clarify","session_search","hindsight_recall","vision_analyze"]:
    check(t, pre_tool_call_handler(t, {}) is None)

print("\n═══ 2. HARDLINE path ═══")
for t,a in [("write_file","/etc/shadow"),("write_file","/boot/grub"),("patch","~/.ssh/config"),
            ("write_file","/proc/cpuinfo"),("patch","~/.gnupg/key")]:
    r = fast_path(t, {"path": a}, {"enabled": True})
    check(f"{t}({a[:25]})", r and r.get("action")=="block")

print("\n═══ 3. HARDLINE filename ═══")
for p,blk in [("/x/.env",True),("/x/config.yaml",True),("/x/id_rsa",True),
              ("/x/id_ed25519",True),("/x/authorized_keys",True),("/x/readme.md",False),("/x/foo.py",False)]:
    r = fast_path("write_file", {"path": p}, {"enabled": True})
    if blk: check(p.split("/")[-1], r and r["action"]=="block")
    else:   check(p.split("/")[-1], r is None, f"spurious block: {r}")

print("\n═══ 4. SAFE path → LLM ═══")
for p in ["/home/atlasengine/t.py","/tmp/f.py","~/p/readme.md","./local/f.txt"]:
    check(f"write_file({p[:30]})", fast_path("write_file",{"path":p},{"enabled":True}) is None)

print("\n═══ 5. DELEGATE danger ═══")
for g,blk in [("delete all user files",True),("format disk completely",True),
              ("wipe system clean",True),("refactor auth module",False),("write unit tests",False)]:
    r = fast_path("delegate_task",{"goal":g},{"enabled":True})
    if blk: check(f"'{g}'", r and r["action"]=="block")
    else:   check(f"'{g}'", r is None)

print("\n═══ 6. TERMINAL delegation ═══")
for t,a in [("terminal",{"command":"ls"}),("terminal",{"command":"sudo rm -rf /"}),
            ("process",{"action":"kill","session_id":"x"}),("process",{"action":"poll"})]:
    check(f"{t}({str(a)[:30]})", pre_tool_call_handler(t, a) is None)

print("\n═══ 7. DISABLED → no-op ═══")
guard._config_disable = True; guard._config_cache = {"enabled": False}
for t,a in [("write_file",{"path":"/etc/passwd"}),("delegate_task",{"goal":"rm -rf /"}),("terminal",{"command":"reboot"})]:
    check(f"off: {t}", pre_tool_call_handler(t, a) is None)
guard._config_disable = False; guard._config_cache = None

print("\n═══ 8. Feedback quality ═══")
r = fast_path("write_file", {"path": "/etc/hosts"}, {"enabled": True})
if r and r["action"] == "block":
    check("action field", "action" in r)
    check("message field", "message" in r)
    check("message non-empty", len(r["message"]) > 20)
    check("Chinese text", any('\u4e00' <= c <= '\u9fff' for c in r["message"]))

# ═══ TEST 9: LLM classification pipeline ═══
print("\n═══ 9. LLM classification ═══")

import agent.auxiliary_client as aac
_orig = aac.call_llm

class MC:  # mock choice
    def __init__(s, c): s.message = type("m",(),{"content":c})()

class MR:  # mock response
    def __init__(s, c): s.choices = [MC(c)]

_q = []
_fail = False
def mock_llm(*a, **kw):
    if _fail: raise RuntimeError("LLM down")
    return MR(_q.pop(0)) if _q else MR("ALLOW")

aac.call_llm = mock_llm
clf = stage1_llm.llm_classify
cfg = {"enabled":True,"fail_open":True}

# ALLOW
_q = ["ALLOW"]; check("ALLOW", clf("write_file",{"path":"/tmp/t.py"},cfg,"","")=="ALLOW")
# DENY
_q = ["DENY"]; check("DENY", clf("write_file",{"path":"/tmp/c.py"},cfg,"","")=="DENY")
# ESCALATE
_q = ["ESCALATE"]; check("ESCALATE", clf("execute_code",{"code":"import os"},cfg,"","")=="ESCALATE")
# fail_open
_fail = True; check("fail→ALLOW", clf("write_file",{"path":"/tmp/t.py"},cfg,"","")=="ALLOW")
# fail_close
check("fail→DENY", clf("write_file",{"path":"/tmp/t.py"},{"enabled":True,"fail_open":False},"","")=="DENY")

aac.call_llm = _orig

# ═══ SUMMARY ═══
tot = passed + failed
print(f"\n{'='*50}\n  {passed}/{tot} 通过, {failed}/{tot} 失败")
sys.exit(0 if not failed else 1)
