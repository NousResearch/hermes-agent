# MEDIA Delivery Diagnostic Pattern

Use this when MEDIA files show as literal path text in Discord instead of native attachments.

## Quick diagnostic: isolate the failure mode

The fastest way to narrow down which failure mode you're hitting:

```bash
# 1. Copy the file to image_cache (always a hardcoded root AND a known-working extension)
cp /path/to/your/report.html /home/kensei/.hermes/image_cache/test-delivery.png

# 2. Send BOTH paths as MEDIA tags
# If image_cache/.png works but runbooks/.html shows as text → check extension regex first (Failure C), then allowlist (Failure B)
# If both fail → Failure A (file doesn't exist or permissions)
# If .png works in runbooks but .html doesn't → Failure C (extension not in regex)
```

**Key shortcut:** If `MEDIA:/path/file.png` works but `MEDIA:/path/file.html` fails for the SAME directory, the issue is Failure C (regex). Don't waste time on allowlists or permissions.

## Failure A: File doesn't exist at delivery time

The LLM wrote `MEDIA:/absolute/path.html` but the file was never created. Two sub-causes:

### A1: Directory doesn't exist
The LLM cannot `mkdir`. If the cron prompt references `/path/to/reports/2026-05-24/report.html`, the directory `/path/to/reports/` and `/path/to/reports/2026-05-24/` must already exist. Pre-create date-stamped directories in the cron prompt or a pre-flight script.

### A2: File was created but deleted or overwritten
If two crons output to the same path, or a runbook cleanup script runs between creation and delivery. Use date-stamped filenames.

## Failure B: File exists but allowlist rejects it

The gateway checked `validate_media_delivery_path()` and returned `None` because the path isn't in `MEDIA_DELIVERY_SAFE_ROOTS` or `HERMES_MEDIA_ALLOW_DIRS`.

### Two-layer allowlist

| Layer | Where | Scope | Example |
|-------|-------|-------|---------|
| Hardcoded roots | `gateway/platforms/base.py` | ALL contexts (gateway, CLI, tools) | `image_cache`, `audio_cache`, `document_cache`, `runbooks`, `reports` |
| Env var | `HERMES_MEDIA_ALLOW_DIRS` in `.env` | Gateway process only (loads `.env` at startup) | Any custom path |

### Why the env-var-only fix is not enough

The `send_message` tool and `execute_code` tool run in **isolated subprocess contexts**. They DO NOT call `load_hermes_dotenv()`. So `HERMES_MEDIA_ALLOW_DIRS` set in `.env` is invisible to these tools.

This means: cron deliveries work (gateway loaded .env), but if you test MEDIA delivery via `send_message` in a CLI session, the path is silently rejected even though the file exists and the gateway cron delivery works fine.

### Fix for paths used by cron outputs

1. **Add the path to `MEDIA_DELIVERY_SAFE_ROOTS` in `base.py`** (durable, works everywhere)
2. **Also set `HERMES_MEDIA_ALLOW_DIRS` in `.env`** (belt-and-suspenders, helps the gateway path too)

Don't rely on the env var alone for paths that cron jobs consistently output to.

## Failure C: File extension not in extract_media() regex

DISCOVERED 2026-05-27. The most subtle failure mode because the file exists, the path is allowlisted, the gateway is running — but the `MEDIA:` tag is still rendered as raw text.

**Root cause:** `extract_media()` in `gateway/platforms/base.py` line ~2418 uses a regex to identify which file extensions are valid for `MEDIA:` tag processing. If the extension is not in the regex's alternation group, the entire `MEDIA:` line is passed through as plain text — it's never parsed, never validated against the allowlist, never uploaded as an attachment.

The regex (as of the fix) matches:
```
(?:png|jpe?g|gif|webp|mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|txt|csv|html?|apk|ipa)
```

**Before the 2026-05-27 fix**, `html?` was NOT in this group. So every cron producing HTML attachments (kanban digest, calendar brief, system report, content engine, research digest, mailbox cleaner, MrHermagi lesson) had their `MEDIA:` tags rendered as raw text paths.

**Important distinction:** `_LOCAL_MEDIA_EXTS` (a separate list used by `extract_local_files()` for bare path detection) already included `.html` and `.htm`. But `extract_media()` is the entry point for `MEDIA:` tag parsing — it's a different code path. Both must include the extension for end-to-end MEDIA delivery.

### How to diagnose

1. Check if `MEDIA:/path/file.html` shows as text but `MEDIA:/path/file.png` works for the same path prefix → definitely Failure C
2. Grep the regex in the active codebase:
   ```bash
   grep 'html?' /home/kensei/repos/KenseiAgent/gateway/platforms/base.py
   # If no match, the extension is missing from the regex
   ```
3. Check both copies (the git checkout AND the `.hermes/hermes-agent/` deployment copy):
   ```bash
   grep 'html?' /home/kensei/.hermes/hermes-agent/gateway/platforms/base.py
   ```

### Fix

Add `html?` (or whatever extension is missing) to the regex alternation group at line ~2418:

```python
# Before:
(?:png|jpe?g|gif|webp|mp4|...|txt|csv|apk|ipa)

# After:
(?:png|jpe?g|gif|webp|mp4|...|txt|csv|html?|apk|ipa)
```

Patch BOTH copies of `base.py`, then restart all gateway services:

```bash
sudo systemctl restart hermes-gateway hermes-gateway-*
```

### Verification

Send a test message with an HTML file:
```python
# Via Hermes send_message tool, or:
echo '<html><body><h1>Test</h1></body></html>' > /home/kensei/.hermes/runbooks/test.html
# Then send a Discord message containing: MEDIA:/home/kensei/.hermes/runbooks/test.html
# Confirm: file appears as a downloadable .html attachment, not as raw text
```

Or programmatically:
```python
import sys
sys.path.insert(0, '/home/kensei/.hermes/hermes-agent')
from gateway.platforms.base import BasePlatformAdapter
result = BasePlatformAdapter.extract_media('MEDIA:/home/kensei/.hermes/runbooks/test.html')
# Should return a list with one item containing the path
# If empty list, the extension is not in the regex
```

### Stale bytecode trap

After patching `base.py`, Python may load stale `.pyc` from `__pycache__/` instead of the patched `.py`. This is especially likely when the gateway worker process was already importing from the file before the patch.

**Symptom:** You patched `base.py` and restarted the gateway, but `extract_media()` still behaves as if the old regex is in effect. No error is logged — the old compiled bytecode silently wins.

**Fix:** Delete `__pycache__` in both copies, then restart:
```bash
find /home/kensei/repos/KenseiAgent/gateway/platforms/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find /home/kensei/.hermes/hermes-agent/gateway/platforms/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
sudo systemctl restart hermes-gateway hermes-gateway-*
```

This applies to ANY patch to `base.py` or any gateway Python module — always clear `__pycache__` before restarting.

## All-failures checklist

When debugging "MEDIA path shown as text instead of attached file":

| # | Check | Command | Failure mode |
|---|-------|---------|-------------|
| 1 | Does the file exist? | `ls -la /path/from/MEDIA-line` | A — file never written |
| 2 | Is the parent directory writable? | Check parent dir permissions | A — can't create file |
| 3 | Is the path in allowlist? | `grep HERMES_MEDIA_ALLOW_DIRS ~/.hermes/.env` | B — path not allowlisted |
| 4 | Is the path in hardcoded roots? | Check `MEDIA_DELIVERY_SAFE_ROOTS` in base.py | B — path not in hardcoded roots |
| 5 | File readable by gateway user? | `stat -c '%a %U %G' /path/to/file` | A or B — kensei must be able to read |
| 6 | Is the extension in extract_media() regex? | `grep 'html?' gateway/platforms/base.py` | C — extension not recognised |
| 7 | Quick test: .png works but .html doesn't? | Send two MEDIA tags with same path prefix | C confirmed if .png attaches but .html doesn't |