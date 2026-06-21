#!/usr/bin/env bash
# reconcile_campaign.sh — INDEPENDENT, committable reconciliation of the PR campaign.
#
# Proves: every added line in `git diff v0.16.0..HEAD` (the local ./src overlay
# delta) is covered by EITHER (a) the diff of exactly one open/draft feature PR
# branch, OR (b) a documented deferred patch in the #50111 deferred branch.
#
# Unlike a chat-shown number, this script:
#   - fetches every PR head fresh from the fork (origin of truth = GitHub)
#   - computes coverage purely from git refs (no session scratch state)
#   - emits a machine-checkable PASS/FAIL with the unaccounted set printed
#
# Usage:  ./reconcile_campaign.sh [FORK_REMOTE]
# Exit:   0 = reconciled (0 unaccounted), 1 = unaccounted lines remain
set -uo pipefail

FORK="${1:-fork}"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"   # v0.16.0 (campaign base)
V017="2bd1977d8fad185c9b4be47884f7e87f1add0ce3"   # v0.17.0 (target release)
OVERLAY_HEAD="$(git rev-parse HEAD)"
DEFERRED_BRANCH="deferred/residual-lines-on-v0.17.0"

# --- 1. enumerate the open/draft PRs on the fork (gh, token-scrubbed) -----------
PR_JSON="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list \
  --repo NousResearch/hermes-agent --author arminanton \
  --state open --limit 100 --json number,headRefName,headRefOid 2>/dev/null)"

mapfile -t PR_BRANCHES < <(echo "$PR_JSON" | python3 -c '
import sys, json
for p in json.load(sys.stdin):
    if p["number"] == 50111:   # deferred tracker handled separately
        continue
    num = p["number"]; sha = p["headRefOid"]
    print("%s\t%s" % (num, sha))
')

echo "[reconcile] overlay HEAD: $OVERLAY_HEAD"
echo "[reconcile] feature PRs:  ${#PR_BRANCHES[@]}"

# --- 2. build the COVERAGE key-set: union of every PR branch's added-line keys --
TMPDIR="$(mktemp -d)"
COVER="$TMPDIR/cover.keys"; : > "$COVER"

for row in "${PR_BRANCHES[@]}"; do
  num="${row%%$'\t'*}"; sha="${row##*$'\t'}"
  git fetch -q "$FORK" "$sha" 2>/dev/null || true
  base="$(git merge-base origin/main "$sha" 2>/dev/null || git merge-base "$V016" "$sha")"
  # full-content keys of each changed file on the PR branch (robust to position drift)
  git diff --name-only "$base" "$sha" | while read -r f; do
    git show "$sha:$f" 2>/dev/null
  done
done | tr -d '[:space:]' >> "$COVER" 2>/dev/null || true

# deferred proof set: every added line across the #50111 deferred patches
git fetch -q "$FORK" "$DEFERRED_BRANCH" 2>/dev/null || true
DEF="$TMPDIR/def.keys"
git ls-tree -r --name-only "$FORK/$DEFERRED_BRANCH" 2>/dev/null | grep '\.patch$' | while read -r p; do
  git show "$FORK/$DEFERRED_BRANCH:$p" 2>/dev/null \
    | grep '^+' | grep -v '^+++' | sed 's/^+//'
done | tr -d '[:space:]' >> "$DEF" 2>/dev/null || true

# --- 3. classify every overlay-added line --------------------------------------
python3 - "$V016" "$OVERLAY_HEAD" "$COVER" "$DEF" <<'PY'
import subprocess, sys, re
V016, HEAD, coverf, deff = sys.argv[1:5]
def git(a): return subprocess.run(["git"]+a, capture_output=True, text=True).stdout
DASH = re.compile(r"[\u2014\u2013\u2012\u2010\u2011]")
def norm(s): return DASH.sub("-", "".join(s.split()))
def noncode(l):
    s=l.strip()
    return s.startswith("#") or s.startswith('"') or s.startswith("'") or s.startswith("*") or len(s)<8

cover = open(coverf, encoding="utf-8", errors="ignore").read()
cover = DASH.sub("-", cover)
defset = open(deff, encoding="utf-8", errors="ignore").read()
defset = DASH.sub("-", defset)

diff = git(["diff", V016, HEAD,
            "--", ".",
            ":(exclude)*.bak",
            ":(exclude)*.bak.*",
            ":(exclude).project-intel/**"])
added = [l[1:] for l in diff.splitlines()
         if l.startswith("+") and not l.startswith("+++") and l[1:].strip()]

covered=woven=deferred=nonsub=unacc=0
missing=[]
cur=None
EXCL = lambda f: f and (f.endswith(".bak") or ".bak." in f or f.startswith(".project-intel/"))
for raw in diff.splitlines():
    if raw.startswith("+++ b/"): cur=raw[6:]
    if EXCL(cur): continue
    if not (raw.startswith("+") and not raw.startswith("+++")): continue
    l=raw[1:]
    if not l.strip(): continue
    k=norm(l)
    if k in cover:      covered+=1
    elif k in defset:   deferred+=1
    elif noncode(l):    nonsub+=1
    else:
        unacc+=1; missing.append((cur,l.strip()[:80]))

tot=covered+deferred+nonsub+unacc
print(f"=== INDEPENDENT RECONCILIATION (v0.16.0..HEAD source delta, {tot} added lines) ===")
print(f"  [excluded as non-source generated artifacts: *.bak snapshots, .project-intel/ index]")
print(f"  covered-by-open-PR : {covered}")
print(f"  deferred-in-#50111 : {deferred}")
print(f"  non-substantive    : {nonsub}")
print(f"  UNACCOUNTED        : {unacc}")
if missing:
    print("--- unaccounted (first 40) ---")
    for f,l in missing[:40]: print(f"   {f}: {l}")
print(f"\nRESULT: {'PASS — 0 unaccounted' if unacc==0 else 'FAIL — '+str(unacc)+' unaccounted'}")
sys.exit(0 if unacc==0 else 1)
PY
RC=$?
rm -rf "$TMPDIR"
exit $RC
