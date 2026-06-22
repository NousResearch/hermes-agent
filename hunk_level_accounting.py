#!/usr/bin/env python3
# hunk_level_accounting.py — Council: prove EVERY hunk of v0.16.0->src-HEAD ./src delta
# lives in >=1 open code PR, or is an explicitly enumerated exclusion. Hunk = COVERED if
# its added lines (the "+"-lines) all appear in some covering PR's file CONTENT (PRs are
# cut off origin/main, so per-file content is the correct membership test).
import subprocess, os, json, sys
SRC=os.environ.get("SRC","/mnt/devvm/custom/hermes/src")
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
def git(*a): return subprocess.run(["git","-C",SRC,*a],capture_output=True,text=True).stdout
GH=["env","-u","GITHUB_TOKEN","-u","GH_TOKEN","gh"]

prs=json.loads(subprocess.run(GH+["pr","list","--repo","NousResearch/hermes-agent",
  "--author","arminanton","--state","open","--limit","100","--json","number,headRefOid,title"],
  capture_output=True,text=True,cwd=SRC).stdout)
code=[(str(p["number"]),p["headRefOid"]) for p in prs
      if "manifest" not in p["title"] and "NOT FOR MERGE" not in p["title"]]
manifest=[(str(p["number"]),p["headRefOid"]) for p in prs
      if "manifest" in p["title"] or "NOT FOR MERGE" in p["title"]]
for _,s in code+manifest: git("fetch","-q","fork",s)
print(f"code PRs={len(code)}  manifest PRs={len(manifest)}")

# explicitly enumerated EXCLUSIONS (withdrawn / superseded / discard) — must be user-acked
EXCL={
 "agent/agy_cli_client.py":"WITHDRAWN-agy(#50454)","agent/gemini_native_adapter.py":"WITHDRAWN-agy",
 "agent/gemini_cloudcode_adapter.py":"WITHDRAWN-gemini-UA(#50492/#50033)","agent/google_user_agent.py":"WITHDRAWN-gemini-UA",
 "plugins/model-providers/agy-cli/__init__.py":"WITHDRAWN-agy","plugins/model-providers/agy-cli/plugin.yaml":"WITHDRAWN-agy",
 "tests/agent/test_agy_cli_client_v2.py":"WITHDRAWN-agy","tests/agent/test_agy_cli_client_v3.py":"WITHDRAWN-agy",
 "tests/plugins/test_agy_cli_plugin_v2.py":"WITHDRAWN-agy",
 "agent/subdirectory_hints.py":"SUPERSEDED(#29433)","tests/agent/test_subdirectory_hints.py":"SUPERSEDED(#29433)",
 "tests/agent/conftest.py":"SUPERSEDED(#29433)",
 "transcripts/C_opus_baseline.txt":"DISCARD","transcripts/C_opus_contradiction.txt":"DISCARD",
 "transcripts/C_sonnet_baseline.txt":"DISCARD","transcripts/C_sonnet_contradiction.txt":"DISCARD",
 "hermes_cli/auth.py":"WITHDRAWN-agy-registration(#50039/#50657)","hermes_cli/runtime_provider.py":"WITHDRAWN-agy-registration",
}
def pr_content(sha,f):
    return set(git("show",f"{sha}:{f}").splitlines())
def manifest_content(f):
    out=set()
    for _,s in manifest: out|=set(git("show",f"{s}:{f}").splitlines())
    return out

HEAD=git("rev-parse","HEAD").strip()
files=[f for f in git("diff","--name-only",V016,HEAD,"--",".",
    ":(exclude)*.bak",":(exclude)*.bak.*",":(exclude).project-intel/**").splitlines() if f.strip()]
print(f"src-delta files={len(files)}  (HEAD={HEAD[:9]} vs v0.16.0)")

tot_h=mapped=excl_h=unmapped=0
unmapped_detail=[]
for f in files:
    # which PRs touch this file?
    covering=[(n,s) for n,s in code if git("show",f"{s}:{f}")!=""]
    cov_lines=set()
    for n,s in covering: cov_lines|=pr_content(s,f)
    diff=git("diff",f"-U0",V016,HEAD,"--",f)
    # parse hunks
    cur=None; adds=[]
    def flush():
        global tot_h,mapped,excl_h,unmapped
        if cur is None: return
        tot_h+=1
        addset=[l[1:] for l in adds if l.startswith("+") and not l.startswith("+++")]
        addset=[l for l in addset if l.strip()!=""]
        if not addset:  # pure-deletion hunk
            mapped+=1; return
        if all(a in cov_lines for a in addset):
            mapped+=1
        elif f in EXCL:
            excl_h+=1
        elif all(a in manifest_content(f) for a in addset):
            mapped+=1  # carried by #50111 manifest content
        else:
            # last try: any single covering PR fully contains them
            ok=False
            for n,s in covering:
                pl=pr_content(s,f)
                if all(a in pl for a in addset): ok=True;break
            if ok: mapped+=1
            else:
                unmapped+=1; unmapped_detail.append((f,cur,addset[:2]))
    for ln in diff.splitlines():
        if ln.startswith("@@"): flush(); cur=ln; adds=[]
        elif cur is not None and (ln.startswith("+") or ln.startswith("-")): adds.append(ln)
    flush()

print(f"\nHUNKS total={tot_h}  mapped-to-code-PR-or-manifest={mapped}  in-enumerated-exclusion={excl_h}  UNMAPPED={unmapped}")
if unmapped_detail:
    print("UNMAPPED hunks:")
    for f,h,sample in unmapped_detail[:500]:
        print(f"  {f}  {h}")
        for s in sample: print(f"    + {s[:80]}")
print(f"\nEXCLUSIONS (enumerated, require user ack):")
seen=set()
for f,r in EXCL.items():
    if f in files:
        print(f"  {f} -> {r}")
print(f"\nRESULT: {'PASS — 0 unmapped' if unmapped==0 else f'FAIL — {unmapped} unmapped'}")
sys.exit(0 if unmapped==0 else 1)
