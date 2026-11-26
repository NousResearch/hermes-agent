---
name: github-actions-artifact-to-release
description: Upload GitHub Actions artifact to GitHub Release when local network can't download artifacts. Two-workflow pattern with intermediate upload step.
triggers:
  - "GitHub Actions artifact too large to download locally"
  - "WSL/network cannot reach GitHub artifact storage"
  - "need to distribute binary releases via GitHub"
---

# GitHub Actions Artifact → Release Upload Pipeline

## Problem
When `actions/upload-artifact` creates an artifact, downloading it locally (via `gh run download` or API) often fails due to network issues, especially from WSL/China networks. The artifact sits on GitHub servers but is inaccessible from local environment.

## Solution: Trigger a second workflow on GitHub runners to handle the upload

### Architecture
```
build-windows.yml (run #1)
  └─→ creates artifact "PlantsVsZombies-win64" (14MB on GitHub servers)
  
upload-release.yml (run #2, triggered via API dispatch)
  └─→ runs on GitHub's ubuntu-latest runner (fast internal network)
      └─→ gh run download (fetches artifact from GitHub servers)
      └─→ softprops/action-gh-release (uploads to Release)
```

### build-windows.yml
```yaml
name: Build Windows EXE
on: workflow_dispatch  # or push tags

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install pygame pyinstaller
      - run: pyinstaller --onefile --windowed main.py
      - uses: actions/upload-artifact@v4
        with:
          name: PlantsVsZombies-win64
          path: dist/PlantsVsZombies.exe
```

### upload-release.yml
```yaml
name: Upload EXE to Release
on:
  workflow_dispatch:
    inputs:
      run_id:
        description: 'Build run ID'
        required: true

permissions:
  contents: write  # REQUIRED for gh-release action to work

jobs:
  upload:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { ref: master }
      - name: Download Artifact
        run: gh run download ${{ inputs.run_id }} --name PlantsVsZombies-win64 -D ./exe
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Upload to Release
        uses: softprops/action-gh-release@v2
        with:
          files: ./exe/PlantsVsZombies.exe
          tag_name: v1.0.0          # IMPORTANT: tag_name, NOT tag
          draft: false
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Key Pitfalls Discovered (Trial & Error)

### 1. Wrong parameter name for tag
- **Wrong**: `tag: v1.0.0` → "invalid inputs: 'tag'"
- **Correct**: `tag_name: v1.0.0`

### 2. Release needs a git tag backing it
- If release was created via API without a git tag, `tag_name: v1.0.0` still fails
- Fix: Create an annotated git tag first:
  ```python
  # Create tag object
  tag_obj = api('POST', '/repos/{owner}/{repo}/git/tags', {
      "tag": "v1.0.0",
      "message": "Release v1.0.0",
      "object": commit_sha,
      "type": "commit",
      "tagger": {"name": "Bot", "email": "bot@local"}
  })
  # Create tag ref
  api('POST', '/repos/{owner}/{repo}/git/refs', {
      "ref": "refs/tags/v1.0.0",
      "sha": tag_obj['sha']
  })
  ```

### 3. Permissions error: "Resource not accessible by integration"
- **Cause**: `softprops/action-gh-release@v2` needs `contents: write` permission to update releases
- **Fix**: Add `permissions: contents: write` at workflow level (or job level)

### 4. gh run download needs git context
- **Wrong**: `gh run download` in a fresh workflow without checkout
- **Correct**: Add `actions/checkout@v4` step first to provide git context for `gh`

### 5. Artifact ID vs Run ID
- `gh run download <run_id>` takes the **run ID** (number like 25275778727), NOT the artifact ID
- Artifact ID is for API calls; Run ID is for `gh run download`

## Triggering workflow via REST API (when git push is blocked)
```python
import urllib.request, json

token = "<PAT>"
headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}

# Trigger workflow dispatch
data = json.dumps({
    "ref": "master",
    "inputs": {"run_id": "25275778727"}
}).encode()

req = urllib.request.Request(
    "https://api.github.com/repos/{owner}/{repo}/actions/workflows/upload-release.yml/dispatches",
    method='POST', headers=headers, data=data
)
with urllib.request.urlopen(req, timeout=10) as resp:
    print(f"Status: {resp.status}")  # 204 = success
```

## Checking workflow run status
```bash
gh run list --workflow=upload-release.yml --limit 1
gh run logs <run_id>  # full logs
```

## When to use this pattern
- Artifact size > 5MB (too large for base64 encoding in API calls)
- Local network cannot reliably reach GitHub's artifact storage
- Need to distribute binary releases to end users
- CI/CD pipeline where artifact needs to become a downloadable release
