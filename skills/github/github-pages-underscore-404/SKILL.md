---
name: github-pages-underscore-404
description: Fix GitHub Pages 404 for files with underscore prefix (Jekyll ignores _ files)
---

# GitHub Pages Underscore File 404 Fix

## Problem
Files in a GitHub Pages site return 404 even though they exist in the git repository with correct content. Specifically affects files whose names start with underscore (`_`).

## Root Cause
GitHub Pages uses Jekyll as its site generator by default. Jekyll ignores (does not publish) any files or directories whose names begin with underscore (`_`), treating them as "draft" or "partial" files. This is a standard Jekyll convention.

## Diagnosis Checklist
1. File exists in git: `git ls-tree HEAD -- path/to/file`
2. File SHA matches local content (not a corruption issue)
3. Raw GitHub URL works: `https://raw.githubusercontent.com/{user}/{repo}/{branch}/path/to/file` returns 200
4. GitHub Pages URL returns 404: `https://{user}.github.io/{repo}/path/to/file`
5. GitHub Pages build status shows "built" (not errored) in repo Settings → Pages

## Solution
Add a `.nojekyll` file to the root of the publishing branch. This disables Jekyll processing entirely.

```bash
echo "" > .nojekyll
git add .nojekyll
git commit -m "Disable Jekyll to prevent underscore files from being filtered"
git push origin gh-pages
```

Alternatively, if you want Jekyll enabled but only for specific directories, you can configure `_config.yml` with `include: [assets]` to force Jekyll to process certain underscore directories.

## Verification
After pushing, wait ~30-60 seconds for GitHub Pages to rebuild:
```bash
curl -sI "https://{user}.github.io/{repo}/path/to/_file.js" | head -1
# Should return HTTP/2 200 (was 404)
```

## Relevant Files
- `.nojekyll` — created at repo root
- `_config.yml` — Jekyll config (if using)

## Context
This commonly affects:
- Vite/uni-app builds with `_plugin-*` files
- Webpack chunk files with underscore prefixes
- Any generated asset files that start with `_`
