---
name: ghp-404-recovery
description: GitHub Pages 404 after force push - verify + workflow dispatch recovery
---
# GitHub Pages 404 After Force Push Recovery

## Problem
After force pushing to `gh-pages` branch, GitHub Pages returns 404 even though:
- `gh api repos/:owner/:repo/pages` shows the page is configured
- The workflow shows "success" in GitHub Actions
- `curl` returns HTTP 200

## Root Cause
Force pushing to `gh-pages` with `git push origin master:gh-pages -f` can corrupt the GitHub Pages deployment state when GitHub Actions is also managing deployments.

## Recovery Steps

### 1. Verify the site is actually running
```bash
curl -s -o /dev/null -w "%{http_code}" https://yeluo45.github.io/pixel-pal-web/
# Should return 200
```

### 2. If 404, trigger workflow_dispatch to force redeploy
```bash
cd /home/hermes/pixel-pal-web
gh workflow run deploy.yml
```

### 3. Wait for deployment to complete
```bash
gh run list --limit 1  # Wait for "completed" + "success"
```

### 4. Verify via API
```bash
gh api repos/:owner/:repo/pages/builds
# Should show a recent build, not []
```

## Prevention
- Avoid `git push origin master:gh-pages -f` when GitHub Actions manages the deployment
- If force push is needed, immediately trigger `workflow_dispatch` to rebuild
- Use `gh pages enable` / `gh pages disable` to reset if corruption persists

## Key Distinction
- `browser_navigate` tool can timeout (60s) even when site returns HTTP 200
- Always verify with `curl` first when browser navigation fails
- GitHub Actions workflow-based Pages uses a different deployment mechanism than static gh-pages branch
