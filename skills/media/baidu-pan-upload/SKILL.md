---
name: baidu-pan-upload
description: Use when asked to upload, download, list, or sync files with Baidu Netdisk (百度网盘) using the bypy Python library. Supports single file upload, batch upload, directory sync, and directory listing via Baidu PCS API.
version: 1.0.0
author: Hermes Agent Community
license: MIT
metadata:
  hermes:
    tags: [baidu, pan, netdisk, bypy, upload, download, sync, cloud-storage]
    related_skills: [chevereto-upload, gif-search]
---

# Baidu Netdisk (百度网盘) — bypy

Upload, download, list, and sync files with Baidu Netdisk using [`bypy`](https://github.com/houtianze/bypy), a Python library that wraps the Baidu PCS (Personal Cloud Storage) API.

## When to Use

- User asks to upload files to Baidu Netdisk
- User asks to download files from Baidu Netdisk
- User asks to sync a local folder with Baidu Netdisk
- User asks to list or browse Baidu Netdisk directory contents
- User asks to check Baidu Netdisk quota usage
- Don't use for: Google Drive, OneDrive, Dropbox, or other cloud storage (use their respective tools instead)

## Prerequisites

### 1. bypy must be installed and authenticated

```bash
pip install bypy
bypy info        # First time: opens browser for OAuth, then saves token to ~/.bypy/bypy.json
```

### 2. Known bypy installation paths

| Environment | Path |
|-------------|------|
| User's venv | `/home/shangyi/python/venv/bypy_venv/bin/python3 -m bypy` |
| System-wide | `bypy` (if installed globally) |

### 3. Baidu Netdisk quota

- Anonymous: 2GB upload / 5GB total
- Authorized: **2TB upload / 5TB total** (OAuth via bypy)
- Current usage shown by `bypy info`

## Common Operations

### Check quota and account info

```bash
bypy info
```

Output example:
```
Quota: 14.008TB
Used: 13.449TB
```

### List root directory

```bash
bypy list /
```

### List a specific directory

```bash
bypy list /syncup
bypy list /syncbooks
```

### Upload a single file

```bash
bypy upload /local/path/to/file.jpg /remote/path/
```

The remote path is relative to `/apps/bypy/`.

Example — upload to `/apps/bypy/test_hermes/`:
```bash
bypy upload /home/shangyi/disk0/aicg/workspace/team45/images/0.jpg /test_hermes/
```

### Download a directory

```bash
bypy downdir /remote/path /local/destination/
```

Downloads recursively. Progress shown with `[====================]`.

### Sync local folder UP to Baidu Netdisk

```bash
bypy syncup /local/folder /remote/folder
```

Example — scheduled in crontab (每周三凌晨):
```bash
0 1 * * 3 /home/shangyi/python/venv/bypy_venv/bin/python3 -m bypy syncup /home/shangyi/disk0/books /syncbooks >> /tmp/cronjob.log 2>&1
```

### Sync Baidu Netdisk folder DOWN to local

```bash
bypy syncdown /remote/folder /local/folder
```

### Compare local and remote (diff)

```bash
bypy compare /local/folder /remote/folder
```

## Python one-liner (no shell loop needed)

```python
import subprocess
result = subprocess.run(
    ['/home/shangyi/python/venv/bypy_venv/bin/python3', '-m', 'bypy', 'upload',
     '/local/file.jpg', '/dest/'],
    capture_output=True, text=True
)
print(result.stdout)
```

## Batch Upload Recipe

To upload multiple files with logging and error handling:

```bash
BYPY="/home/shangyi/python/venv/bypy_venv/bin/python3 -m bypy"
SRC="/home/shangyi/disk0/aicg/workspace/team45/images"
DEST="/team45"

for f in "$SRC"/*; do
  fname=$(basename "$f")
  echo -n "Uploading $fname ... "
  if $BYPY upload "$f" "$DEST/" 2>&1 | grep -q "Success"; then
    echo "✓"
  else
    echo "✗ (see above)"
  fi
done
```

## Important Notes

| Aspect | Detail |
|--------|--------|
| Upload path prefix | All paths are relative to `/apps/bypy/` on Baidu Netdisk |
| OAuth token location | `~/.bypy/bypy.json` — keep this file safe |
| Upload limit (unauthorized) | 2GB per file |
| Upload limit (authorized) | 2TB per file |
| Bandwidth | No official limit documented; practical throughput ~10MB/s |
| Duplicate upload | If the remote file already exists with the same name, bypy overwrites it silently |
| Concurrent uploads | bypy uploads one file at a time (no parallel upload built-in) |

## Troubleshooting

### `bypy info` returns error or empty quota
- OAuth token may have expired. Re-authenticate: `bypy info`
- Check that `~/.bypy/bypy.json` exists and is valid

### Upload fails with "SOME_ERROR"
- Network connectivity issue — verify internet access to Baidu servers
- File path contains spaces — wrap in quotes
- File does not exist locally — verify with `ls`

### "Move failed: target dir is not existing"
- bypy `upload` requires the parent directory to exist. Create it first or use `syncup` which creates parents automatically.

## One-Shot Recipe

```
User says: "上传 /home/user/images 到百度网盘 /backup"
Steps:
1. Verify bypy is accessible: /home/shangyi/python/venv/bypy_venv/bin/python3 -m bypy info
2. Run: bypy upload /home/user/images /backup/
3. Verify: bypy list /backup
4. Report result
```
