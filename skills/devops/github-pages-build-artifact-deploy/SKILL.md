---
name: github-pages-build-artifact-deploy
description: Deploy build artifacts to GitHub Pages gh-pages branch via git clone-replace-commit-push workflow
tags: [github, deploy, gh-pages]
---

# GitHub Pages Build Artifact Deploy

## Problem
Deploy build artifacts to GitHub Pages (gh-pages branch) when:
- Standard `git push origin gh-pages` silently fails (exit 0, no actual push)
- Embedded PAT in git URL (`https://{token}@github.com/...`) also fails silently
- gh api blob upload is too slow for large file sets (114 files → 300s timeout)
- gh-pages branch was accidentally corrupted with source code instead of build artifacts

## Solution: Clone → Replace → Commit → Push

```bash
# 1. Shallow clone gh-pages branch to temp dir
git clone --depth=1 https://github.com/{owner}/{repo}.git --branch gh-pages /tmp/ghpages-repo

# 2. Remove old/corrupted contents
cd /tmp/ghpages-repo
rm -rf .github .gitignore CHANGELOG.md README.md assets audits.html dev.ps1 docs \
  index.html login-prototype.html manifest.json package-lock.json package.json \
  pages.json playwright-test.js project.config.json src static vite.config.js vue.config.js

# 3. Copy fresh build artifacts
cp -r /path/to/project/dist/build/h5/. .

# 4. Commit and force push
git add -A
git commit -m "Deploy {version}: {changes}"
git push origin gh-pages --force
```

## Why This Works
- Fresh clone uses credential helper correctly (no embedded token needed)
- `git push origin gh-pages --force` works reliably when credential helper is properly invoked
- No API rate limits or blob upload overhead
- Direct git transport (not HTTPS API)

## Key Lessons
- `git push` returning exit 0 without pushing = git thinks remote already has that SHA. Force push needed.
- gh-pages stores **build artifacts only** (dist/build/h5/), never source code
- GitHub Pages source mode = serve branch content as-is
- Deploy = replace gh-pages branch content with build output, not source

## Verification
```bash
# Should show /{repo}/assets/... paths, NOT /src/main.js
curl -s https://{user}.github.io/{repo}/ | grep -o 'href="[^"]*"' | head -3

# Check specific JS files exist (200 = deployed, 404 = not yet)
curl -sI "https://{user}.github.io/{repo}/assets/pages-level-level.{hash}.js" | head -1
```

## Fallback: gh api for tiny patches
If only 1-2 files need patching:
```bash
SHA=$(gh api repos/{owner}/{repo}/git/refs/heads/gh-pages --jq '.object.sha')
gh api repos/{owner}/{repo}/contents/{path} \
  --method PUT \
  -f message="fix" \
  -f content=@/tmp/file.js \
  -f sha="$SHA"
```
Only viable for 1-2 files; slow/brittle for full build deployment.
