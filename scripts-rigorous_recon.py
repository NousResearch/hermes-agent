#!/usr/bin/env python3
# RIGOROUS reconciliation of EVERY "MISSING clean" line — no generalization.
# For each our-added line not in (owning-PR added-diff) and not in (drift-diff) and not policy-private:
#   classify by checking main's FULL file content (whitespace-normalized) + ALL-PR full content +
#   author of the introducing overlay commit. 0 unexplained remainder is the goal.
import subprocess, json, re
from collections import defaultdict
SRC="<REPO_ROOT>"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
MAIN="f57ff7aef1d3d447e159511f3a3e9ed8ae0c7298"
def git(a): return subprocess.run(["git","-C",SRC]+a,capture_output=True,text=True).stdout
def wsk(l): return "".join(l.split())
def addedlines(diff):
    out=[]
    for l in diff.splitlines():
        if l.startswith("+") and not l.startswith("+++") and l[1:].strip():
            out.append(l[1:])
    return out
def wsset_blob(ref,path):
    r=subprocess.run(["git","-C",SRC,"show",f"{ref}:{path}"],capture_output=True,text=True)
    return set(wsk(l) for l in r.stdout.splitlines() if l.split()) if r.returncode==0 else set()

br={}
for line in open("<LOCAL_PATH>"):
    n,b,o=line.split(); br[int(n)]=o
def prtip(n): return "4a1fbe9e1" if n==48069 else br[n]
om=json.load(open("<LOCAL_PATH>"))
PRIV=re.compile(r"agy[-_]?cli|agy://|antigravity|auto_rout|copilot_auto|_maybe_apply_copilot_auto|autopilot|_autopilot|\bcmx\b|_cmx|prefetch_all|_materialize_data_url|Phase A[0-9]|/mnt/devvm|review-sysprompt|antigravity-core|copilot_inventory|opus.context|mythos|fable|codex_version|hermes_source|project_source|tool_trace|stable_update|google_user_agent|_CODEX_OAUTH_CONTEXT_EMPIRICAL|_COPILOT_HIDDEN_USABLE|resolve_copilot_identity_audit|_ANTHROPIC_OUTPUT_LIMITS_COPILOT|_AsyncSyncCompletionsAdapter|AsyncCopilotACP|_is_missing_trigram|_HONCHO_CACHE|HERMES_PREFETCH",re.I)

# all-PR full content cache (whitespace-normalized) per file
allpr_cache={}
def allpr_full(path):
    if path not in allpr_cache:
        s=set()
        for n in br: s|=wsset_blob(prtip(n),path)
        allpr_cache[path]=s
    return allpr_cache[path]

buckets=defaultdict(int)
unexplained=defaultdict(list)
for path,owners in om.items():
    our=addedlines(git(["diff",V016,"--",path]))
    owners_diff=set()
    for n in owners: owners_diff|=set(wsk(x) for x in addedlines(git(["diff",MAIN,prtip(n),"--",path])))
    drift=set(wsk(x) for x in addedlines(git(["diff",V016,MAIN,"--",path])))
    mainfull=wsset_blob(MAIN,path)
    allprfull=allpr_full(path)
    for l in our:
        k=wsk(l)
        if k in owners_diff: buckets["covered-owning-PR-diff"]+=1
        elif k in allprfull: buckets["covered-some-PR-fullcontent"]+=1
        elif k in mainfull or k in drift: buckets["already-upstream"]+=1
        elif PRIV.search(l): buckets["policy-deferred"]+=1
        else:
            buckets["UNEXPLAINED"]+=1
            unexplained[path].append(l.strip())

total=sum(buckets.values())
print(f"=== RIGOROUS per-line reconciliation of ALL our-added lines ({total}) ===")
for k in ["covered-owning-PR-diff","covered-some-PR-fullcontent","already-upstream","policy-deferred","UNEXPLAINED"]:
    print(f"  {k:30s}: {buckets[k]}")
print(f"\nUNEXPLAINED remainder: {buckets['UNEXPLAINED']} across {len(unexplained)} files")
for f,ls in sorted(unexplained.items(),key=lambda x:-len(x[1]))[:25]:
    print(f"  {f}: {len(ls)}")
json.dump({f:ls for f,ls in unexplained.items()}, open("<LOCAL_PATH>","w"))
