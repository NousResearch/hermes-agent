---
name: github-force-rollback-via-api
description: Force rollback a GitHub branch to a previous commit via REST API when git push is blocked
---

# GitHub Force Rollback via API

## Problem
When `git push` is blocked (SSH/HTTPS ports timeout) and you need to roll back a protected or non-fast-forward branch to a previous commit, standard `gh api -f force=true` may fail with `"force" is not a boolean` (HTTP 422).

## Solution
Use `--input -` (stdin) with a raw JSON body:

```bash
echo '{"sha":"<OLD_COMMIT_SHA>","force":true}' | gh api repos/{owner}/{repo}/git/refs/heads/{branch} -X PATCH --input -
```

## Why --input Works
When using `-f force=true`, gh CLI sends it as a URL-encoded form field (`force=true`), but the GitHub API expects a proper JSON boolean. Piping raw JSON via `--input -` sends the correct Content-Type: application/json with a real boolean.

## Example
```bash
# Get current SHA first
gh api repos/YeLuo45/pixel-pal-web/git/refs/heads/master --jq '.object.sha'

# Roll back to a known good commit
echo '{"sha":"b970db7a3bc60f67e2f20d138d22a458fe2967d1","force":true}' | gh api repos/YeLuo45/pixel-pal-web/git/refs/heads/master -X PATCH --input -
```

## When to Use
- git push is blocked (SSH port 22, HTTPS 443 all timeout)
- Branch is protected (non-fast-forward)
- Standard `gh api -f force=true` returns 422
