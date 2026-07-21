---
name: github-repo-rename-with-rebase
description: Rename a GitHub repository via REST API while handling git rebase conflicts from concurrent remote changes
---

# GitHub Repository Rename with Concurrent Remote Changes

## Scenario
Rename a GitHub repo while remote has new commits, requiring git rebase + conflict resolution before push.

## Steps

### 1. Rename remote via GitHub REST API
```bash
curl -s -X PATCH "https://api.github.com/repos/{owner}/{old-name}" \
  -H "Authorization: token {PAT}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/vnd.github.v3+json" \
  -d '{"name":"new-repo-name"}'
```

### 2. Update local git remote URL
```bash
git remote set-url origin https://github.com/{owner}/new-repo-name.git
```

### 3. Handle push rejection (fetch-first rule)
If remote has new commits and push is rejected:
```bash
git fetch origin {branch}
git rebase origin/{branch}
```

### 4. Resolve merge conflicts during rebase
When rebase reports conflicts, resolve each file manually:
```bash
# Check which files have conflicts
git diff --name-only --diff-filter=U

# Edit and fix each conflicted file, then:
git add <fixed-file>
git rebase --continue
```

Note: Use explicit `git commit -m "message"` then `git rebase --continue` — don't rely on interactive editor.

### 5. Push after successful rebase
```bash
git push
```

## Pitfalls
- Conflict resolution during rebase: must use `-m` flag on commit to avoid editor prompt in non-interactive sessions
