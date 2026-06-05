#!/usr/bin/env bash
# find-preview-url.sh — Discover preview / deploy URLs for a PR from GitHub metadata.
#
# Surfaces candidate URLs from four independent sources so you don't have to guess:
#   1. Deployment statuses   (environment_url / target_url) — GitHub Deployments API
#   2. Commit statuses       (target_url)                    — Vercel/Netlify/etc. post here
#   3. Check runs            (details_url)                    — CI / preview check runs
#   4. PR bot comments       (URLs in the body)              — Vercel/Netlify/Cloudflare bots
#
# It only REPORTS what GitHub returns — it never invents a URL. Each line is tagged
# with its source so you can judge trustworthiness. Verify the URL actually loads
# (see the github-pr-visual-qa skill) before reporting any visual state.
#
# Deps: bash + curl + python3. Uses `gh` automatically when authenticated (no token needed).
#
# Usage:
#   ./find-preview-url.sh                 # current branch's open PR
#   ./find-preview-url.sh 1234            # PR number
#   ./find-preview-url.sh --sha <SHA>     # a specific commit SHA
#
set -euo pipefail

PR_ARG=""
SHA_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sha) SHA_OVERRIDE="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) PR_ARG="$1"; shift ;;
  esac
done

# --- Auth detection (mirrors github-pr-workflow) -----------------------------
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  AUTH="gh"
else
  AUTH="curl"
  if [ -z "${GITHUB_TOKEN:-}" ]; then
    if [ -f "$HOME/.hermes/.env" ] && grep -q "^GITHUB_TOKEN=" "$HOME/.hermes/.env"; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$HOME/.hermes/.env" | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" "$HOME/.git-credentials" 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" "$HOME/.git-credentials" 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
  if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "ERROR: no gh auth and no GITHUB_TOKEN found. Authenticate first (see github-auth skill)." >&2
    exit 1
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)

# api PATH -> JSON on stdout. Works with gh or curl.
api() {
  local path="$1"
  if [ "$AUTH" = "gh" ]; then
    gh api "$path" 2>/dev/null || echo '{}'
  else
    curl -s -H "Authorization: token $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github+json" \
         "https://api.github.com/$path" 2>/dev/null || echo '{}'
  fi
}

# --- Resolve PR number + head SHA --------------------------------------------
BRANCH=$(git branch --show-current 2>/dev/null || echo "")
PR_NUMBER="$PR_ARG"

if [ -z "$PR_NUMBER" ] && [ -n "$BRANCH" ]; then
  PR_NUMBER=$(api "repos/$OWNER/$REPO/pulls?head=$OWNER:$BRANCH&state=all" \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
print(d[0]['number'] if isinstance(d,list) and d else '')" 2>/dev/null || echo "")
fi

SHA="$SHA_OVERRIDE"
if [ -z "$SHA" ] && [ -n "$PR_NUMBER" ]; then
  SHA=$(api "repos/$OWNER/$REPO/pulls/$PR_NUMBER" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('head',{}).get('sha',''))" 2>/dev/null || echo "")
fi
[ -z "$SHA" ] && SHA=$(git rev-parse HEAD 2>/dev/null || echo "")

echo "# Preview/deploy URL candidates"
echo "repo: $OWNER/$REPO   pr: ${PR_NUMBER:-<none>}   sha: ${SHA:-<none>}   auth: $AUTH"
echo

FOUND=0
emit() { echo "[$1] $2"; FOUND=$((FOUND+1)); }

# --- 1. Deployment statuses ---------------------------------------------------
if [ -n "$SHA" ]; then
  api "repos/$OWNER/$REPO/deployments?sha=$SHA" \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
print('\n'.join(str(x['id']) for x in d) if isinstance(d,list) else '')" 2>/dev/null \
    | while read -r DID; do
        [ -z "$DID" ] && continue
        api "repos/$OWNER/$REPO/deployments/$DID/statuses" \
          | python3 -c "import sys,json
for s in json.load(sys.stdin):
    u=s.get('environment_url') or s.get('target_url')
    if u: print(f\"deployment:{s.get('state','?')} {u}\")" 2>/dev/null
      done | sort -u | while read -r line; do emit deployment "$line"; done
fi

# --- 2. Commit statuses (Vercel/Netlify post target_url here) -----------------
if [ -n "$SHA" ]; then
  api "repos/$OWNER/$REPO/commits/$SHA/status" \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
for s in d.get('statuses',[]):
    u=s.get('target_url')
    if u and 'github.com' not in u: print(f\"{s.get('context','?')}:{s.get('state','?')} {u}\")" 2>/dev/null \
    | sort -u | while read -r line; do emit status "$line"; done
fi

# --- 3. Check runs ------------------------------------------------------------
if [ -n "$SHA" ]; then
  api "repos/$OWNER/$REPO/commits/$SHA/check-runs" \
    | python3 -c "import sys,json
d=json.load(sys.stdin)
for c in d.get('check_runs',[]):
    u=c.get('details_url')
    if u and 'github.com' not in u: print(f\"{c.get('name','?')}:{c.get('conclusion') or c.get('status')} {u}\")" 2>/dev/null \
    | sort -u | while read -r line; do emit check-run "$line"; done
fi

# --- 4. PR bot comments -------------------------------------------------------
if [ -n "$PR_NUMBER" ]; then
  api "repos/$OWNER/$REPO/issues/$PR_NUMBER/comments" \
    | python3 -c "import sys,json,re
d=json.load(sys.stdin)
seen=set()
for c in d if isinstance(d,list) else []:
    login=(c.get('user') or {}).get('login','')
    for u in re.findall(r'https?://[^\s)\]\"<>]+', c.get('body','') or ''):
        if 'github.com' in u or u in seen: continue
        seen.add(u); print(f\"by {login}: {u}\")" 2>/dev/null \
    | while read -r line; do emit pr-comment "$line"; done
fi

echo
if [ "$FOUND" -eq 0 ]; then
  echo "No preview/deploy URLs found in PR metadata."
  echo "Fallbacks: ask the user for the URL, check the PR description, or use the known"
  echo "live/staging URL for this app. Do NOT guess a URL pattern and report on it."
fi
