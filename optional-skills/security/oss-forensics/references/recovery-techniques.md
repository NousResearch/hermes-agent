# Deleted Content Recovery Techniques

## Key Insight: GitHub Never Fully Deletes Force-Pushed Commits

Force-pushed commits are removed from the branch history but REMAIN on GitHub's servers until garbage collection runs (which can take weeks to months). This is the foundation of deleted commit recovery.

---

## Method 1: Direct GitHub URL (Fastest — No Auth Required)

If you have a commit SHA, access it directly even if it was force-pushed off a branch:

```bash
# View commit metadata
curl -s "https://github.com/OWNER/REPO/commit/SHA"

# Download as patch (includes full diff)
curl -s "https://github.com/OWNER/REPO/commit/SHA.patch" > recovered_commit.patch

# Download as diff
curl -s "https://github.com/OWNER/REPO/commit/SHA.diff" > recovered_commit.diff

# Example (Istio credential leak - real incident):
curl -s "https://github.com/istio/istio/commit/FORCE_PUSHED_SHA.patch"
```

**When this works**: SHA is known (from GH Archive, Wayback Machine, or `git fsck`)
**When this fails**: GitHub has already garbage-collected the object (rare, typically 30–90 days post-force-push)

---

## Method 2: GitHub REST API

```bash
# Works for commits force-pushed off branches but still on server
# Note: /commits/SHA may 404, but /git/commits/SHA often succeeds for orphaned commits
curl -s "https://api.github.com/repos/OWNER/REPO/git/commits/SHA" | jq .

# Get the tree (file listing) of a force-pushed commit
curl -s "https://api.github.com/repos/OWNER/REPO/git/trees/SHA?recursive=1" | jq .

# Get a specific file from a force-pushed commit
curl -s "https://api.github.com/repos/OWNER/REPO/contents/PATH?ref=SHA" | jq .content | base64 -d
```

---

## Method 3: Git Fetch by SHA (Local — Requires Clone)

```bash
# Fetch an orphaned commit directly by SHA into local repo
cd target_repo
git fetch origin SHA
git log FETCH_HEAD -1   # view the commit
git diff FETCH_HEAD~1 FETCH_HEAD  # view the diff

# If the SHA was recently force-pushed it will still be fetchable
# This stops working once GitHub GC runs
```

---

## Method 4: Dangling Commits via git fsck

```bash
cd target_repo

# Find all unreachable objects (includes force-pushed commits)
git fsck --unreachable --no-reflogs 2>&1 | grep "unreachable commit" | awk '{print $3}' > dangling_shas.txt

# For each dangling commit, get its metadata
while read sha; do
  echo "=== $sha ===" >> dangling_details.txt
  git show --stat "$sha" >> dangling_details.txt 2>&1
done < dangling_shas.txt

# Note: dangling objects only exist in LOCAL clone — not the same as GitHub's copies
# GitHub's copies are accessible via Methods 1-3 until GC runs
```

---

## Recovering Deleted GitHub Issues and PRs

### Via Wayback Machine CDX API

```bash
# Find all archived snapshots of a specific issue
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/issues/NUMBER&output=json&limit=50&fl=timestamp,statuscode,original" | python3 -m json.tool

# Fetch the best snapshot
# Use the timestamp from the CDX result:
# https://web.archive.org/web/TIMESTAMP/https://github.com/OWNER/REPO/issues/NUMBER
curl -s "https://web.archive.org/web/TIMESTAMP/https://github.com/OWNER/REPO/issues/NUMBER" > issue_NUMBER_archived.html

# Find all snapshots of the repo in a date range
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO*&output=json&from=20240101&to=20240201&limit=200&fl=timestamp,urlkey,statuscode" | python3 -m json.tool
```

### Via GitHub API (Limited — Only Non-Deleted Content)

```bash
# Closed issues (not deleted) are retrievable
curl -s "https://api.github.com/repos/OWNER/REPO/issues?state=closed&per_page=100" | jq '.[].number'

# Note: DELETED issues/PRs do NOT appear in the API. Use Wayback Machine or GH Archive for those.
```

### Via GitHub Archive (For Event History — Not Content)

```sql
-- Find all IssueEvents for a repo in a date range
SELECT created_at, actor.login, payload.action, payload.issue.number, payload.issue.title
FROM `githubarchive.day.*`
WHERE _TABLE_SUFFIX BETWEEN '20240101' AND '20240201'
  AND type = 'IssuesEvent'
  AND repo.name = 'OWNER/REPO'
ORDER BY created_at
```

---

## Recovering Deleted Files from a Known Commit

```bash
# If you have the commit SHA (even force-pushed):
git show SHA:path/to/file.py > recovered_file.py

# Or via API (base64 encoded content):
curl -s "https://api.github.com/repos/OWNER/REPO/contents/path/to/file.py?ref=SHA" | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
print(base64.b64decode(d['content']).decode())
"
```

---

## Evidence Recording

After recovering any deleted content, immediately record it:

```bash
python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json add \
  --source "git fetch origin FORCE_PUSHED_SHA" \
  --content "Recovered commit: FORCE_PUSHED_SHA | Author: attacker@example.com | Date: 2024-01-15 | Added file: malicious.sh" \
  --type git \
  --actor "attacker-handle" \
  --url "https://github.com/OWNER/REPO/commit/FORCE_PUSHED_SHA.patch" \
  --timestamp "2024-01-15T00:00:00Z" \
  --verification single_source \
  --notes "Commit force-pushed off main branch on 2024-01-16. Recovered via direct fetch."
```

---

## Recovery Failure Modes

| Failure | Cause | Workaround |
|---------|-------|------------|
| `git fetch origin SHA` returns "not our ref" | GitHub GC already ran | Try Method 1/2, search Wayback Machine |
| `github.com/OWNER/REPO/commit/SHA` returns 404 | GC ran or SHA is wrong | Verify SHA via GH Archive; try partial SHA search |
| Wayback Machine has no snapshots | Page was never crawled by IA | Check `commoncrawl.org`, check Google Cache |
| BigQuery shows event but no content | GH Archive stores event metadata, not file contents | Recovery only reveals the event occurred, not the content |
