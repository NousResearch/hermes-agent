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

- **Renaming files within the same path**: CDN serves cached 404 even for new filenames
- **Padding file size alone without URL change**: Padding a 91-byte file to 200 bytes in the same location still returns 404 because the cached 404 key is the URL path
- **Waiting**: CDN 404s can persist for extended periods (hours to days) — observed 13+ seconds of cached 404 during testing

## What CAN Work (Partial)

- **Padding + fresh subdirectory**: Pad small files to >100 bytes AND deploy to a new subdirectory path. Both conditions needed. The padding alone (without URL change) does NOT bypass the cache.
- **HTTP/1.1 bypass**: `curl --http1.1` often returns 200 while HTTP/2 returns 404. Browser tools that use HTTP/1.1 internally may load assets successfully even when `curl` with HTTP/2 fails.

## Vite/Subdirectory Deployment Pattern

When deploying a Vite build to a GitHub Pages subdirectory (e.g., `/future-little-leaders/dist/`):

1. **Vite generates relative chunk references** — chunks are referenced via `__vite__mapDeps` array with relative paths like `"assets/chunk-xxx.js"`. These resolve correctly from the HTML file's location, so `dist/index.html` referencing `"assets/chunk.js"` correctly resolves to `dist/assets/chunk.js`.
2. **Only HTML entry paths need fixing** — the `<link>` and `<script>` tags in `dist/index.html` need their base path updated from `/future-little-leaders/assets/` to `/future-little-leaders/dist/assets/`.
3. **Chunk JS files do NOT need individual path patching** — Vite's relative path mechanism handles them automatically.
4. **Recommended workflow**:
   ```bash
   # Clone gh-pages
   git clone --depth=1 --branch gh-pages https://github.com/<org>/<repo>.git /tmp/<repo>-ghpages
   
   # Replace entire dist/ with new build
   rm -rf dist/ assets/ index.html
   cp -r /path/to/project/dist/build/h5/. dist/
   
   # Fix HTML asset paths if needed
   # (e.g., /<repo>/assets/ → /<repo>/dist/assets/)
   
   git add -A && git commit -m "deploy" && git push
   ```

## Prevention

- Configure Vite's `base` path in `vite.config.js` to the full subdirectory path (e.g., `'/future-little-leaders/dist/'`) before building — this avoids manual HTML path patching
- Avoid committing very small (<100 byte) JavaScript files to GitHub Pages. Concatenate small helpers into larger chunks.
- Deploy to a fresh subdirectory path per major version to avoid CDN cache issues
