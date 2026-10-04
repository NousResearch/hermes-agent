---
name: github-pages-workflow-mode-debug
description: Debug and fix GitHub Pages when configured in workflow mode. Covers common deploy.yml errors, REST API file update without git push, and CDN propagation delays.
---

# GitHub Pages Workflow Mode Debug

## Key Distinction
When Pages `build_type` is `"workflow"`, Pages serves artifacts from the most recent successful `deploy-pages` step — NOT the `gh-pages` branch content. This causes massive confusion.

## Verify Mode
```bash
gh api repos/{owner}/{repo}/pages
# {"build_type": "workflow", ...}
```

## Debugging Sequence

### 1. Latest workflow run
```bash
gh api repos/{owner}/{repo}/actions/runs?per_page=3
```
Must show `completed/success`.

### 2. Pages build history
```bash
gh api repos/{owner}/{repo}/pages/builds
```
Each entry has `commit` SHA + `status` (built/pending/failed).

### 3. Served content
```bash
curl -s https://{owner}.github.io/{repo}/ | grep 'script.*\.js'
```

### 4. CI job logs
```bash
gh api repos/{owner}/{repo}/actions/jobs/{job_id}/logs > /tmp/logs.txt
grep -i 'error' /tmp/logs.txt
```

## Common deploy.yml Errors

### Wrong id-token value
```yaml
# WRONG — "grant" is not valid
permissions:
  id-token: grant

# CORRECT — valid: none | read | write
permissions:
  id-token: write
```

### Wrong trigger branch
```yaml
# Must match actual repo default branch
on:
  push:
    branches: [master]   # or [main]
```

### npm peer dependency conflict
```yaml
# Use --legacy-peer-deps instead of npm ci
- run: npm install --legacy-peer-deps
```
Symptoms: build fails with "Could not resolve entry module index.html" AFTER npm install succeeds.

### Missing source file on trigger branch
If CI fails with missing file but it exists on another branch, use the REST API method below.

## REST API File Update (7 Steps Without Git Push)

When git push is blocked and you need to update one or few files:

```python
import urllib.request, base64, json, subprocess

token = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True).stdout.strip()
owner, repo = 'OWNER', 'REPO'

# Step 1: Get current branch SHA
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/git/refs/heads/master',
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
)
sha = json.loads(urllib.request.urlopen(req).read())['object']['sha']

# Step 2: Get file content
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/contents/path?ref=feature/branch',
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
)
content = base64.b64decode(json.loads(urllib.request.urlopen(req).read())['content']).decode()

# Step 3: Create blob
data = json.dumps({'content': content, 'encoding': 'utf-8'}).encode()
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/git/blobs',
    data=data,
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json', 'Content-Type': 'application/json'}
)
blob_sha = json.loads(urllib.request.urlopen(req).read())['sha']

# Step 4: Get tree SHA from current commit
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/git/commits/{sha}',
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
)
tree_sha = json.loads(urllib.request.urlopen(req).read())['tree']['sha']

# Step 5: Create new tree
data = json.dumps({
    'tree': [{'path': 'path', 'mode': '100644', 'type': 'blob', 'sha': blob_sha}],
    'base_tree': tree_sha
}).encode()
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/git/trees',
    data=data,
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json', 'Content-Type': 'application/json'}
)
new_tree_sha = json.loads(urllib.request.urlopen(req).read())['sha']

# Step 6: Create commit
data = json.dumps({
    'message': 'chore: update file',
    'tree': new_tree_sha,
    'parents': [sha]
}).encode()
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/git/commits',
    data=data,
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json', 'Content-Type': 'application/json'}
)
new_sha = json.loads(urllib.request.urlopen(req).read())['sha']

# Step 7: Update branch ref (PATCH)
data = json.dumps({'sha': new_sha, 'force': False}).encode()
req = urllib.request.Request(
    f'https://api.github.com/repos/{owner}/{repo}/git/refs/heads/master',
    data=data,
    headers={'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json', 'Content-Type': 'application/json'}
)
req.get_method = lambda: 'PATCH'
urllib.request.urlopen(req)
```

**Critical**: Use FULL 40-char SHA. Contents API PUT with `?ref=branch` returns 404 if file doesn't exist — tree approach works in all cases.

## CDN Still Stale After Success

Wait 5 minutes. If still stale, manually trigger workflow_dispatch in GitHub Actions UI.
