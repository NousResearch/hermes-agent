#!/usr/bin/env python3
# exclusion_manifest_and_coverage.py — OPTION B mechanical proof.
# Goal: TARGET_added_lines − EXCL  ⊆  (PR diffs ∪ PR contents).  Residual must be 0,
# or every residual line is surfaced. EXCL is a PRINCIPLED manifest (rules below), NOT
# circular (NOT defined as "everything not covered").
import subprocess, re, json
SRC="/mnt/devvm/custom/hermes/src"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
def git(*a): return subprocess.run(["git","-C",SRC,*a],capture_output=True,text=True).stdout
GH=["env","-u","GITHUB_TOKEN","-u","GH_TOKEN","gh"]
def norm(x):
    x=x.replace("\u2014","-").replace("\u2013","-"); return re.sub(r"\s+"," ",x).strip()
def added(diff):
    return set(norm(l[1:]) for l in diff.splitlines()
               if l.startswith("+") and not l.startswith("+++") and l[1:].strip())
PS=[".",":(exclude)*.bak*",":(exclude)*~",":(exclude).project-intel/**"]
HEAD=git("rev-parse","HEAD").strip()

# ---- TARGET: per-file added lines (keep file association for line-range manifest) ----
files=[f for f in git("diff","--name-only",V016,HEAD,"--",*PS).splitlines() if f.strip()]
target={}  # file -> set(norm lines)
for f in files:
    a=added(git("diff",V016,HEAD,"--",f))
    if a: target[f]=a
TARGET=set().union(*target.values()) if target else set()
print(f"TARGET files={len(target)}  added-lines(uniq)={len(TARGET)}")

# ---- PR coverage: diffs + CONTENT (the honest membership test) ----
prs=json.loads(subprocess.run(GH+["pr","list","--repo","NousResearch/hermes-agent","--author",
  "arminanton","--state","open","--limit","100","--json","number,headRefOid,title"],
  capture_output=True,text=True,cwd=SRC).stdout)
code=[(str(p["number"]),p["headRefOid"]) for p in prs
      if "manifest" not in p["title"] and "NOT FOR MERGE" not in p["title"]]
for _,s in code: git("fetch","-q","fork",s)
PRCOV=set()
for n,s in code:
    base=git("merge-base",s,"origin/main").strip()
    PRCOV|=added(git("diff",base,s,"--",*PS))
# PR content (per delta-file)
PRCONTENT=set()
for f in target:
    for n,s in code:
        c=git("show",f"{s}:{f}")
        if c: PRCONTENT|={norm(l) for l in c.splitlines() if norm(l)}
COVERAGE = PRCOV | PRCONTENT
print(f"PR coverage (diff∪content) lines={len(COVERAGE)}")

# ---- EXCLUSION MANIFEST (principled rules) ----
# R1: fully-excluded FILES (withdrawn/superseded/discard) — file-level
R1_FILES={"agent/agy_cli_client.py","agent/gemini_native_adapter.py","agent/gemini_cloudcode_adapter.py",
 "agent/google_user_agent.py","plugins/model-providers/agy-cli/__init__.py","plugins/model-providers/agy-cli/plugin.yaml",
 "tests/agent/test_agy_cli_client_v2.py","tests/agent/test_agy_cli_client_v3.py","tests/plugins/test_agy_cli_plugin_v2.py",
 "agent/subdirectory_hints.py","tests/agent/test_subdirectory_hints.py","tests/agent/conftest.py",
 "transcripts/C_opus_baseline.txt","transcripts/C_opus_contradiction.txt","transcripts/C_sonnet_baseline.txt",
 "transcripts/C_sonnet_contradiction.txt","hermes_cli/auth.py","hermes_cli/runtime_provider.py"}
# R2: private-overlay INFRA files entirely absent from PRs (impersonation/accelerator never-PR list)
R2_FILES={"agent/auto_router.py","agent/codex_version.py","agent/copilot_acp_client.py",
 "agent/transports/codex_app_server.py","agent/transports/codex_app_server_session.py",
 "agent/google_user_agent.py","tools/tool_trace_sidecar.py"}
# R3: private-CONTENT patterns inside SHARED files (account caps, hidden catalog, codex/agy/auto,
#     opus-context phase work, fable/mythos effort tables) — the entangled/private LINES
R3_PAT=re.compile(r"\bagy\b|codex_version|codex_cli_rs|auto_router|_copilot_auto|94125662|891509|900K|"
 r"_COPILOT_HIDDEN|hermes_source|project_source|tool_trace|google_user_agent|gemini_cli|"
 r"_AsyncSync|antigravity|Phase A[0-9]|opus.context|fable|mythos|_XHIGH_EFFORT|_ADAPTIVE_THINKING|"
 r"_NO_SAMPLING|EMPIRICAL_MERGE|chatgpt_account_id|Honcho|honcho|catalog.first|TRUE enforced|"
 r"TRUE usable|INPUT-token budget|effort allow-list|supportedReasoningEfforts|context_length_cache",re.I)

EXCL=set(); excl_reason={}
for f,lines in target.items():
    for l in lines:
        if f in R1_FILES: EXCL.add(l); excl_reason.setdefault(l,("R1-withdrawn/superseded/discard file",f))
        elif f in R2_FILES: EXCL.add(l); excl_reason.setdefault(l,("R2-private-infra file",f))
        elif R3_PAT.search(l): EXCL.add(l); excl_reason.setdefault(l,("R3-private-content line",f))
print(f"EXCL manifest lines={len(EXCL)}  (R1 files + R2 infra files + R3 private-content lines)")

# ---- RESIDUAL ----
UNCOVERED = TARGET - COVERAGE
RESIDUAL = UNCOVERED - EXCL
# cosmetic recovery: a residual line whose dash/space-stripped form matches a coverage line
def strip2(x): return re.sub(r"[\s\-_]","",x).lower()
cov_strip={strip2(l) for l in COVERAGE}
cosmetic={l for l in RESIDUAL if strip2(l) in cov_strip}
RESIDUAL2 = RESIDUAL - cosmetic
print(f"\nUNCOVERED (TARGET − PRcoverage) = {len(UNCOVERED)}")
print(f"  − EXCL manifest               = {len(RESIDUAL)}")
print(f"  − cosmetic(dash/ws match)      = {len(RESIDUAL2)}  <-- TRUE unexplained residual")
if RESIDUAL2:
    # group by file
    from collections import Counter
    bf=Counter()
    for f,lines in target.items():
        bf[f]=len(lines & RESIDUAL2)
    print("\n=== TRUE residual by file (genuinely unexplained) ===")
    for f,n in bf.most_common():
        if n: print(f"  {n:>4}  {f}")
    open("/tmp/residual2.txt","w").write("\n".join(sorted(RESIDUAL2)))
print(f"\nRESULT: {'PASS — residual 0 under the OPTION-B manifest' if not RESIDUAL2 else f'{len(RESIDUAL2)} unexplained (surfaced)'}")
