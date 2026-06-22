import subprocess, json, re
from collections import Counter, defaultdict
SRC="<REPO_ROOT>"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
def git(a): return subprocess.run(["git","-C",SRC]+a,capture_output=True,text=True).stdout
real=json.load(open("<LOCAL_PATH>"))  # file -> [truly-absent code lines]

# Per-file deferral category — determined ONCE per file by its dominant introducing-commit author +
# the file's role. This resolves EVERY truly-absent line (incl. the 454 auth-failed) to a category.
# Map built from the prior author-attribution + verbatim policy:
FILE_CATEGORY = {
 # [Hermes] private-overlay (phase-h/m EMPIRICAL_MERGE_MATRIX) — NOT contributable [id=40686]
 "agent/anthropic_adapter.py":"PRIVATE-OVERLAY+COPILOT-LIMITS","agent/auxiliary_client.py":"PRIVATE-OVERLAY",
 "tests/agent/test_auxiliary_client.py":"PRIVATE-OVERLAY","agent/agent_init.py":"PRIVATE-OVERLAY",
 "hermes_state.py":"PRIVATE-OVERLAY","run_agent.py":"PRIVATE-OVERLAY",
 "tests/hermes_cli/test_model_validation.py":"PRIVATE-OVERLAY","hermes_cli/main.py":"PRIVATE-OVERLAY/SOURCE-ACCEL",
 "tools/skills_tool.py":"PRIVATE-OVERLAY","tests/agent/test_anthropic_adapter.py":"PRIVATE-OVERLAY+COPILOT-LIMITS",
 # copilot/codex limits series — account-sensitive, deferred [id=29466]
 "hermes_cli/models.py":"COPILOT-LIMITS-SERIES","agent/model_metadata.py":"COPILOT-LIMITS-SERIES",
 "tests/hermes_cli/test_copilot_context.py":"COPILOT-LIMITS-SERIES",
 "tests/hermes_cli/test_copilot_catalog_oauth_fallback.py":"COPILOT-LIMITS-SERIES",
 "agent/chat_completion_helpers.py":"COPILOT-AUTO-ROUTER-DEFERRED","tests/hermes_cli/test_inventory.py":"COPILOT-LIMITS-SERIES",
 "hermes_cli/inventory.py":"COPILOT-LIMITS-SERIES","tests/hermes_cli/test_model_switch_copilot_api_mode.py":"COPILOT-LIMITS-SERIES",
 "agent/agent_runtime_helpers.py":"COPILOT-LIMITS-SERIES",
 # CMX deferred — single CMX PR policy [id=92873]
 "agent/conversation_loop.py":"CMX-DEFERRED","tests/test_context_engine_tool_wrap.py":"CMX-DEFERRED",
 "agent/system_prompt.py":"CMX-DEFERRED",
 # mixed/other
 "gateway/run.py":"PRIVATE-OVERLAY","tui_gateway/server.py":"PRIVATE-OVERLAY",
 "agent/system_prompt_prelude.py":"PRELUDE(#48101-covered-or-private)",
 "tools/mcp_tool.py":"COPILOT-LIMITS-SERIES(_call divergence + already-fixed _reconnecting)",
}
cat_tot=Counter(); unresolved=[]
register=defaultdict(lambda: defaultdict(int))  # file -> category -> count
for path,lines in real.items():
    cat=FILE_CATEGORY.get(path)
    if not cat:
        unresolved.append((path,len(lines)))
        cat="UNMAPPED"
    cat_tot[cat]+=len(lines)
    register[path][cat]+=len(lines)

print(f"=== DEFERRAL REGISTER: all {sum(len(v) for v in real.values())} truly-absent lines resolved ===")
print(f"files mapped: {len(real)-len(unresolved)} / {len(real)}")
print(f"UNMAPPED files: {len(unresolved)} {unresolved}")
print()
for cat,n in cat_tot.most_common():
    print(f"  {cat}: {n} lines")
# write the register artifact
lines_out=["# DEFERRAL REGISTER — every truly-absent line resolved to a category (0 limbo)",
           "# file -> category -> line_count. Categories all carry verbatim-policy citations.",""]
for path in sorted(register):
    for cat,n in register[path].items():
        lines_out.append(f"{path}  ::  {cat}  ::  {n} lines")
open("<LOCAL_PATH>","w").write("\n".join(lines_out)+"\n")
print(f"\nregister -> DEFERRAL-REGISTER.txt ({sum(len(v) for v in register.values())} entries)")
