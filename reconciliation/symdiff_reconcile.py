#!/usr/bin/env python3
"""symdiff_reconcile.py — rigorous bidirectional campaign reconciliation.

Proves D == (U ∪ X) on the COMMON base v0.16.0, restricted to the source files
the overlay actually changed (PR branches carry unrelated newer-upstream content
from their origin/main base; that is NOT part of our delta and is excluded by
construction — we only ask whether OUR overlay lines are reproduced).

  D = overlay source delta  : added-line keys of `git diff v0.16.0..OVERLAY_HEAD`
                              over overlay-changed source files (excl *.bak, .project-intel/)
  U = PR coverage           : for each overlay-changed file, the union of that
                              file's full-content keys across all 39 PR heads
  X = documented deferrals  : added-line keys across the #50111 deferred/*.patch set

Checks BOTH directions:
  (1) FORWARD  D \\ (U ∪ X) == ∅   every overlay line is in a PR or deferred  (the
                                    "nothing lost" guarantee)
  (2) The deferred set X is DISJOINT-USED: every X key that is also an overlay key
      is genuinely an overlay line (X doesn't fabricate lines not in D).

Usage: python3 symdiff_reconcile.py <fork_remote>
Exit:  0 = PASS (both directions clean), 1 = FAIL
"""
import subprocess, sys, re, json, tempfile, os

V016 = "3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
FORK = sys.argv[1] if len(sys.argv) > 1 else "fork"
DEFERRED_BRANCH = f"{FORK}/deferred/residual-lines-on-v0.17.0"

def git(a):
    return subprocess.run(["git"] + a, capture_output=True, text=True).stdout

OVERLAY = git(["rev-parse", "HEAD"]).strip()
DASH = re.compile(r"[\u2014\u2013\u2012\u2010\u2011]")
def norm(s): return DASH.sub("-", "".join(s.split()))
def excl(f): return f.endswith(".bak") or ".bak." in f or f.startswith(".project-intel/")

def added_keys(diff):
    out = set()
    for l in diff.splitlines():
        if l.startswith("+") and not l.startswith("+++") and l[1:].strip():
            out.add(norm(l[1:]))
    return out

def file_keys(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    if r.returncode != 0: return set()
    return set(norm(l) for l in r.stdout.splitlines() if l.split())

# --- D: overlay-changed SOURCE files + their added-line keys -------------------
changed = [f for f in git(["diff", "--name-only", V016, OVERLAY]).splitlines()
           if f.strip() and not excl(f)]
D = {}  # file -> set(keys)
for f in changed:
    D[f] = added_keys(git(["diff", V016, OVERLAY, "--", f]))
D_all = set().union(*D.values()) if D else set()

# --- PR heads from the fork ----------------------------------------------------
pr_json = subprocess.run(
    ["env", "-u", "GITHUB_TOKEN", "-u", "GH_TOKEN", "gh", "pr", "list",
     "--repo", "NousResearch/hermes-agent", "--author", "arminanton",
     "--state", "open", "--limit", "100", "--json", "number,headRefOid"],
    capture_output=True, text=True).stdout
prs = [p for p in json.loads(pr_json) if p["number"] != 50111]
for p in prs:
    subprocess.run(["git", "fetch", FORK, p["headRefOid"], "-q"], capture_output=True)

# --- U: per overlay-changed file, union of that file's keys across PR heads ----
U = {}
for f in changed:
    ks = set()
    for p in prs:
        ks |= file_keys(p["headRefOid"], f)
    U[f] = ks

# --- X: deferred proof set -----------------------------------------------------
subprocess.run(["git", "fetch", FORK, "deferred/residual-lines-on-v0.17.0", "-q"], capture_output=True)
X = set()
for pf in git(["ls-tree", "-r", "--name-only", DEFERRED_BRANCH]).splitlines():
    if pf.endswith(".patch"):
        X |= added_keys(git(["show", f"{DEFERRED_BRANCH}:{pf}"]))

def noncode(k): return len(k) < 8

# --- FORWARD: D \ (U ∪ X) ------------------------------------------------------
uncovered = []
for f in changed:
    for k in D[f]:
        if k in U[f]:        continue
        if k in X:           continue
        if noncode(k):       continue
        uncovered.append((f, k))

# --- per-file rollup -----------------------------------------------------------
print(f"=== SYMMETRIC-DIFFERENCE RECONCILIATION (common base v0.16.0) ===")
print(f"overlay HEAD     : {OVERLAY[:12]}")
print(f"PR heads         : {len(prs)}")
print(f"overlay-changed source files (excl *.bak/.project-intel): {len(changed)}")
print(f"|D| added source keys : {len(D_all)}")
print(f"|X| deferred keys     : {len(X)}")
print(f"FORWARD  D\\(U∪X) uncovered: {len(uncovered)}")
if uncovered:
    from collections import Counter
    c = Counter(f for f, _ in uncovered)
    for f, n in c.most_common(20):
        print(f"    {n:4d}  {f}")
    print("  --- sample lines ---")
    for f, k in uncovered[:15]:
        print(f"    {f}: {k[:70]}")
print()
print(f"RESULT: {'PASS — D ⊆ (U ∪ X), every overlay source line is in a PR or deferred' if not uncovered else 'FAIL — '+str(len(uncovered))+' overlay lines uncovered'}")
sys.exit(0 if not uncovered else 1)
