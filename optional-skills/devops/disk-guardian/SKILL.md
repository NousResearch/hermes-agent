---
name: disk-guardian
description: >
  Cleans up ephemeral files Hermes creates during sessions — temp outputs,
  test scripts, test artifacts, cron logs, and stale chrome profiles.
  Auto-deletes test files at the end of every task. Runs automatically when
  disk usage is high or sessions are long. Scoped strictly to HERMES_HOME.
  Triggers: "clean up", "free disk space", "disk is full", "remove test files",
  "what temp files exist", "show disk usage", "run cleanup".
version: 1.2.0
platforms: [linux]
requires_toolsets: [terminal]
metadata:
  hermes:
    tags: [devops, maintenance, cleanup, ephemeral, WSL2, auto]
    category: devops
---

# Disk Guardian

Tracks and removes **ephemeral files Hermes creates** during sessions:
temp outputs, test scripts/artifacts, cron logs, stale chrome profiles,
and large downloads. Scope is strictly `HERMES_HOME` and `/tmp/hermes-*`.
Never touches `~/.hermes/logs/` or any system directory.

## When to Use

**Auto-trigger (no user prompt needed):**
- End of every task that created test files → run `quick` silently
- Session context window exceeds 60% → run `quick` silently
- Any tool call creates a file matching `test_*`, `*.test.*`, `tmp_*` → track it immediately
- User says "done", "finished", "task complete" after a multi-step task

**User-triggered:**
- User says "disk is full", "clean up", "free space", "remove temp files"
- User says "what's taking disk space", "show disk usage"
- User asks for dry-run preview before committing to deletion

## Auto-Run Behavior

This skill runs **silently in the background**. When auto-running:
- Never ask for confirmation for safe categories (temp, test, empty dirs)
- Print one line after: `[disk-guardian] Cleaned 12 files, freed 234 MB`
- Only surface to the user if something needs confirmation (research folders, large files >500 MB)

## Memory — Install Once, Active Forever

On first install, run this to write persistent memory rules so the behavior
carries across all future sessions automatically:

```bash
python3 ~/.hermes/optional-skills/devops/disk-guardian/scripts/disk_guardian.py install-memory
```

This adds to Hermes memory:
- Auto-track every file created under HERMES_HOME during tasks
- Auto-run quick cleanup at end of every task that created test files
- Auto-run quick cleanup if HERMES_HOME disk usage exceeds 5 GB
- Never ask permission for temp/test cleanup — clean silently and report one line

## Quick Reference

```bash
SCRIPT=~/.hermes/optional-skills/devops/disk-guardian/scripts/disk_guardian.py

python3 $SCRIPT install-memory  # write persistent memory (run once after install)
python3 $SCRIPT status          # show breakdown by category + top 10 largest
python3 $SCRIPT dry-run         # preview deletions, touch nothing
python3 $SCRIPT quick           # auto-delete safe files (no prompts)
python3 $SCRIPT deep            # full cleanup, prompt for risky items
python3 $SCRIPT track <path> <category>   # register a file
python3 $SCRIPT forget <path>             # stop tracking a path
```

## Procedure

### Step 1 — Track files as Hermes creates them

Whenever Hermes creates a file via `write_file` or `terminal`, immediately track it:

```bash
# Test script created during a task
python3 $SCRIPT track "/home/user/.hermes/test_output.py" "test"

# Temp output
python3 $SCRIPT track "/home/user/.hermes/cache/run_abc.json" "temp"

# Cron output
python3 $SCRIPT track "/home/user/.hermes/cron/report_2026.md" "cron-output"

# Chrome profile
python3 $SCRIPT track "/home/user/.local/share/notebooklm-mcp/chrome_profile_x" "chrome-profile"
```

Categories: `temp` | `test` | `research` | `download` | `chrome-profile` | `cron-output` | `other`

Always wrap user-supplied paths with `shlex.quote()`.

### Step 2 — Auto-delete test files at end of every task

At the end of **every task** that involved creating or running test files, silently run:

```bash
python3 $SCRIPT quick
```

Output: `[disk-guardian] Cleaned 3 test files, freed 45 KB`

Do not ask the user — just clean and report one line.

### Step 3 — Respond to user cleanup requests

```bash
# Safe, no prompts
python3 $SCRIPT quick

# Full cleanup with confirmation for research/large files
python3 $SCRIPT deep

# Preview only
python3 $SCRIPT dry-run
```

## Cleanup Rules (Deterministic)

| Category | Threshold | Confirmation |
|---|---|---|
| `test` | >0 days — delete at task end | Never |
| `temp` | >7 days since tracked | Never |
| empty dirs under HERMES_HOME | always | Never |
| `cron-output` | >14 days since tracked | Never |
| `research` | >30 days, beyond 10 newest | Always |
| `chrome-profile` | >14 days since tracked | Always |
| `download` / `other` | never auto | Always (deep only) |
| any file >500 MB | never auto | Always (deep only) |

## Pitfalls

- **Never hardcode `~/.hermes`** — always use `HERMES_HOME` env var or `get_hermes_home()`
- **Never touch `~/.hermes/logs/`** — agent debug logs are not ephemeral artifacts
- **Backup/restore scoped to `tracked.json` only** — never agent logs or other Hermes state
- **WSL2: reject Windows mounts** — `/mnt/c/` and all `/mnt/` paths rejected by `_is_safe_path()`
- **Test files are always ephemeral** — delete aggressively, never prompt
- **Silent by default** — only interrupt the user when confirmation is genuinely required

## Verification

```bash
# After quick cleanup:
tail -5 ~/.hermes/disk-guardian/cleanup.log
# Should show DELETED entries for test/temp files

# After install-memory:
# Ask Hermes: "what do you remember about disk cleanup?"
# Should confirm auto-cleanup rules are in memory
```
