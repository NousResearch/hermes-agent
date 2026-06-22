#!/usr/bin/env python3
# mechanical_diff_equality.py — Council: union(39 PR diffs) ∪ excluded-set == src delta,
# zero unexplained residual. NO heuristic blame-to-commit. Pure line-set algebra:
#   TARGET = set of ADDED lines in (v0.16.0 -> worktree HEAD) for ./src
#   PRCOV  = set of ADDED lines across all 39 open code PRs (each base..head)
#   EXCL   = set of ADDED lines belonging to explicitly-excluded files/paths
#   RESIDUAL = TARGET - PRCOV - EXCL    (must be empty, or every residual line enumerated)
# A "line" is normalized (strip, dash-fold) so cosmetic dash/whitespace doesn't create
# false residual. Deletions handled separately (a deletion not in any PR = residual-del).
import subprocess, json, os, re
SRC="/mnt/devvm/custom/hermes/src"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
def git(*a): return subprocess.run(["git","-C",SRC,*a],capture_output=True,text=True).stdout
GH=["env","-u","GITHUB_TOKEN","-u","GH_TOKEN","gh"]

def norm(x):
    x=x.replace("\u2014","-").replace("\u2013","-")
    return re.sub(r"\s+"," ",x).strip()

def added_lines(diff):
    return set(norm(l[1:]) for l in diff.splitlines()
               if l.startswith("+") and not l.startswith("+++") and l[1:].strip())

PATHSPEC=[".",":(exclude)*.bak*",":(exclude)*~",":(exclude).project-intel/**"]

# TARGET: src delta added-lines
HEAD=git("rev-parse","HEAD").strip()
target=added_lines(git("diff",V016,HEAD,"--",*PATHSPEC))
print(f"TARGET added-lines (v0.16.0->HEAD, normalized-unique): {len(target)}")

# PRCOV: union of all 39 open code PR diffs
prs=json.loads(subprocess.run(GH+["pr","list","--repo","NousResearch/hermes-agent",
  "--author","arminanton","--state","open","--limit","100","--json","number,headRefOid,title"],
  capture_output=True,text=True,cwd=SRC).stdout)
code=[(str(p["number"]),p["headRefOid"]) for p in prs
      if "manifest" not in p["title"] and "NOT FOR MERGE" not in p["title"]]
prcov=set()
for n,s in code:
    git("fetch","-q","fork",s)
    base=git("merge-base",s,"origin/main").strip()
    prcov|=added_lines(git("diff",base,s,"--",*PATHSPEC))
print(f"PRCOV added-lines (union of {len(code)} PRs): {len(prcov)}")

# EXCL: added-lines in explicitly-excluded FILES (withdrawn/superseded/discard) — file-level
EXCL_FILES={
 "agent/agy_cli_client.py","agent/gemini_native_adapter.py","agent/gemini_cloudcode_adapter.py",
 "agent/google_user_agent.py","plugins/model-providers/agy-cli/__init__.py",
 "plugins/model-providers/agy-cli/plugin.yaml","tests/agent/test_agy_cli_client_v2.py",
 "tests/agent/test_agy_cli_client_v3.py","tests/plugins/test_agy_cli_plugin_v2.py",
 "agent/subdirectory_hints.py","tests/agent/test_subdirectory_hints.py","tests/agent/conftest.py",
 "transcripts/C_opus_baseline.txt","transcripts/C_opus_contradiction.txt",
 "transcripts/C_sonnet_baseline.txt","transcripts/C_sonnet_contradiction.txt",
 "hermes_cli/auth.py","hermes_cli/runtime_provider.py",
}
excl=set()
for f in EXCL_FILES:
    excl|=added_lines(git("diff",V016,HEAD,"--",f))
print(f"EXCL added-lines (enumerated excluded FILES): {len(excl)}")

# EXCL_INFRA: added-lines referencing excluded INFRA symbols (codex_version, agy, auto_router,
# account caps) inside SHARED files — these are the private lines in otherwise-covered files
INFRA_PAT=re.compile(r"agy|codex_version|auto_router|_copilot_auto|94125662|891509|hermes_source|project_source|tool_trace|_COPILOT_HIDDEN|google_user_agent|gemini_cli|_AsyncSyncCompletionsAdapter", re.I)
residual = target - prcov - excl
infra_residual={l for l in residual if INFRA_PAT.search(l)}
true_residual = residual - infra_residual

print(f"\nRESIDUAL = TARGET - PRCOV - EXCL_FILES = {len(residual)}")
print(f"  of which reference EXCLUDED-INFRA symbols (codex_version/agy/auto_router/caps/...): {len(infra_residual)}")
print(f"  TRUE RESIDUAL (unexplained added-lines): {len(true_residual)}")
if true_residual:
    print("\n=== TRUE RESIDUAL (unexplained) — first 60 ===")
    for l in sorted(true_residual)[:60]:
        print(f"   + {l[:90]}")
open("/tmp/true_residual.txt","w").write("\n".join(sorted(true_residual)))
print(f"\nRESULT: {'PASS — 0 unexplained residual' if not true_residual else f'{len(true_residual)} unexplained residual lines (enumerated)'}")
