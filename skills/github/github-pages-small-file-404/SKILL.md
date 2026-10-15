---
name: github-pages-small-file-404
description: GitHub Pages CDN returns HTTP/2 404 for small static files despite correct origin content — bypass via fresh subdirectory deployment
---

# GitHub Pages Small File 404 Diagnostic

## Symptom

All or specific static assets return `HTTP/2 404` from CDN, but:
- `curl --http1.1` returns `200`
- GitHub API confirms file content exists and is correct
- `git clone --depth=1 --branch <branch>` confirms files are in the repo

## Root Cause

GitHub Pages CDN occasionally caches a **404 response for small files** (< ~100 bytes) and continues serving the cached 404 even after the file content is corrected. The CDN `age:` header in 404 responses confirms caching behavior.

## Diagnostic Steps

```bash
# Check CDN caching status
curl -si "https://<org>.github.io/<repo>/<path>" 2>&1 | grep -E "HTTP|age|x-proxy-cache"

# Compare HTTP/2 vs HTTP/1.1 behavior
curl -s -o /dev/null -w "%{http_code}" "https://<org>.github.io/<repo>/<path>"        # HTTP/2
curl -s --http1.1 -o /dev/null -w "%{http_code}" "https://<org>.github.io/<repo>/<path>"  # HTTP/1.1

# Verify file exists via GitHub API
curl -s "https://api.github.com/repos/<org>/<repo>/contents/<path>?ref=<branch>"
```

## Solution: Deploy to Fresh Subdirectory

Move the entire build to a new subdirectory path that has never been published:

```bash
# Clone gh-pages branch
git clone --depth=1 --branch gh-pages https://github.com/<org>/<repo>.git /tmp/<repo>-ghpages

# Remove old assets, keep only build output in new subdir
rm -rf assets/ index.html
mkdir -p dist/
cp -r /path/to/project/dist/build/h5/* dist/

# Verify index.html paths reference the new subdirectory
grep "href\|src" dist/index.html
# Expected: /<repo>/dist/assets/...

# Commit and push
git add -A
git commit -m "Deploy to dist/ subdirectory - bypass CDN small-file 404"
git push origin gh-pages
```

The new path (e.g., `/future-little-leaders/dist/index.html`) has no cached 404s and works immediately.

## What DOESN'T Work

- **Renaming files**: CDN serves cached 404 even for new filenames (old cache key may be reused)
- **Patching file size**: Making a 91-byte file 100+ bytes does NOT bypass the cached 404
- **Waiting**: CDN 404s can persist for extended periods (hours to days)

## Prevention

Avoid committing very small (<100 byte) JavaScript files to GitHub Pages branches. Instead:
- Concatenate small helpers into larger chunks
- Or ensure build output places all assets in a fresh subdirectory path per deployment
