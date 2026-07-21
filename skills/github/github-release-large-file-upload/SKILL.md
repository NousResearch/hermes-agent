---
name: github-release-large-file-upload
description: Upload large files (>100MB) to GitHub Releases when network is slow, using chunked uploads via cron job
---

# GitHub Releases Large File Upload (Slow Network)

## Problem
Upload a large file (e.g., 168MB .exe) to a GitHub Release, but:
- Network to GitHub is slow (~69KB/s)
- Direct upload times out at tool limit (~300s)
- Large file in single curl times out
- Parallel uploads cause `starter` state (partial upload garbage)

## Solution: Chunked Upload with Cron

### Step 1: Split file into ~5MB chunks
```bash
mkdir -p /tmp/chunks && split -b 5M -a 3 -d /path/to/file.exe /tmp/chunks/part_
# Results in part_000, part_001, ... (33 chunks for 168MB)
```

### Step 2: Check existing assets
```python
import subprocess, json
TOKEN = "ghp_YOUR_TOKEN"
REPO = "owner/repo"
RELEASE_ID = "123456789"

result = subprocess.run(
    ["curl", "-s", "-H", f"Authorization: token {TOKEN}",
     f"https://api.github.com/repos/{REPO}/releases/{RELEASE_ID}/assets"],
    capture_output=True, text=True, timeout=30
)
assets = json.loads(result.stdout)
uploaded = {a['name'] for a in assets if a['state'] == 'uploaded' and 'filename' in a['name']}
```

### Step 3: Delete stale `starter` assets before retrying
Starter state = upload was interrupted, chunk is garbage. Must delete before re-uploading.
```python
for a in assets:
    if a['state'] == 'starter' and 'filename' in a['name']:
        subprocess.run(
            ["curl", "-s", "-X", "DELETE", "-H", f"Authorization: token {TOKEN}",
             f"https://api.github.com/repos/{REPO}/releases/assets/{a['id']}"],
            capture_output=True, timeout=15
        )
```

### Step 4: Upload one chunk at a time
```python
result = subprocess.run(
    [
        "curl", "-s", "-w", "HTTP:%{http_code}",
        "-X", "POST",
        "-H", f"Authorization: token {TOKEN}",
        "-H", "Content-Type: application/octet-stream",
        "--upload-file", chunk_path,
        "--max-time", "240",  # 4 min per 5MB chunk at slow speed
        f"https://uploads.github.com/repos/{REPO}/releases/{RELEASE_ID}/assets?name={chunk_name}"
    ],
    capture_output=True, text=True, timeout=260
)
```

### Step 5: Use cron to automate repeated uploads
```python
# Cron job: every 5min, upload one chunk
# 34 chunks × 5min = ~3 hours for 168MB at 69KB/s
```

### Key Insights
- 5MB chunks upload in ~75s at 69KB/s, fits within tool timeout (~300s)
- `starter` state = incomplete upload, MUST delete before retry
- `HTTP:201` = success; `HTTP:422` or "already_exists" means name reserved but not uploaded yet
- Parallel uploads cause starter states, always use sequential uploads
- Draft releases (`draft: True`) work for testing; switch to正式 before announcing
- Asset names are release-scoped, not repo-scoped — a "already_exists" error may mean the name was used in a different release that was deleted
- When "already_exists" but asset not in list: the old release still holds it; create a fresh release
- Browser navigation to GitHub releases times out (~60s); use API via curl for verification
- Cron script path must use absolute path: `/usr/bin/python3 /home/hermes/.hermes/scripts/upload-exe-chunks.py`
- Script must use `if __name__ == "__main__":` guard, otherwise subprocess imports cause module-level execution

## Ghost Asset Problem (Critical)

When a previous upload times out, GitHub may:
1. Create an asset with `state=starter` (incomplete) — easy to see in API
2. **OR**: Reserve the filename globally, causing future uploads to fail with `HTTP:422 {"message":"already_exists"}` — but the asset does NOT appear in subsequent API listings

**Symptom**: Upload returns `HTTP:422 already_exists` but `GET /releases/{id}/assets` shows the asset is NOT present.

**Root Cause**: The filename was used in a deleted release (or interrupted release) and GitHub's internal state still reserves that name.

**Solution**: Delete the entire release and create a fresh one with a different tag. The name reservation is scoped to the release.

```python
# Delete old release
subprocess.run(
    ["curl", "-s", "-X", "DELETE", "-H", f"Authorization: token {TOKEN}",
     f"https://api.github.com/repos/{REPO}/releases/{old_release_id}"],
    capture_output=True, timeout=15
)
# Create new release with new tag
# Then retry uploads
```

## Related Tools
- `gh release upload` - simpler but `gh` CLI auth sometimes fails (401)
- Git LFS - alternative but requires git-lfs install
- Git push to branch - alternative but push also blocked in some environments
