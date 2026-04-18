---
name: disk-guardian
description: >
  Keeps Hermes's disk footprint clean. Tracks temp files, test outputs, research
  artifacts, and large downloads created during sessions, then removes stale ones
  safely. Especially useful on WSL2 where disk fills up fast during long agent runs.
version: 1.0.0
metadata:
  hermes:
    tags: [devops, maintenance, disk, cleanup, WSL2]
    category: devops
---

# Disk Guardian

Autonomous disk cleanup for Hermes Agent. Tracks files created during sessions and safely removes stale ones to prevent disk space exhaustion, especially on WSL2 where disk fills up fast during long agent runs.

## When to Use

- User reports disk space issues or slow performance
- Long-running sessions have accumulated temp files
- Research artifacts from deep-research need cleanup
- Chrome debug profiles from NotebookLM authentication are growing
- User wants to see disk usage breakdown by category
- User wants to clean up old test outputs and logs

## Core Behaviors

1. **Silent Tracking** - Log every path Hermes writes to tracked.json with timestamp + category
2. **Safe Auto-Cleanup** - Delete stale files by age/size rules with appropriate safety checks
3. **Status Reporting** - Show disk usage breakdown and largest files

## First-Time Setup

On first run, create the disk-guardian directory and state files:

```bash
# Create directory
mkdir -p "$(get_hermes_home)/disk-guardian"

# Initialize tracking file
echo '[]' > "$(get_hermes_home)/disk-guardian/tracked.json"

# Initialize log file
touch "$(get_hermes_home)/disk-guardian/cleanup.log"

# Optional: Register weekly cronjob (Sunday 3 AM)
# This is optional - skill works without cron
```

The skill uses `get_hermes_home()` to resolve the actual path. Never hardcode `~/.hermes` - the path is resolved by the agent, not hardcoded.

## Silent Tracking Protocol

Track files when Hermes creates them via write_file or terminal:

```bash
# Track a temp file
python disk_guardian.py track "/tmp/hermes-abc123/output.json" "temp"

# Track a research artifact
python disk_guardian.py track "$(get_hermes_home)/research/ai-safety/paper.pdf" "research"

# Track a test output
python disk_guardian.py track "$(get_hermes_home)/test_results/test_001.log" "test"

# Track a download
python disk_guardian.py track "$(get_hermes_home)/downloads/model.gguf" "download"

# Track a chrome profile
python disk_guardian.py track "$(get_hermes_home)/.local/share/notebooklm-mcp/chrome_profile_abc" "chrome-profile"
```

Categories: `temp`, `test`, `research`, `download`, `chrome-profile`, `cron-output`, `other`

Always use `shlex.quote()` when interpolating user input into shell commands.

## Cleanup Rules

### Rule 1: Temp Files (> 7 days)

```bash
find "$(get_hermes_home)/cache/hermes" -type f -mtime +7 -delete
find "/tmp/hermes-*" -type f -mtime +7 -delete
```

Auto-delete without confirmation.

### Rule 2: Test Outputs (> 3 days)

```bash
find "$(get_hermes_home)" -type f \( -name "test_*.py" -o -name "*.test.log" -o -name "tmp_*.json" \) -mtime +3 -delete
```

Auto-delete without confirmation.

### Rule 3: Empty Directories

```bash
find "$(get_hermes_home)" -type d -empty -delete
```

Auto-delete without confirmation.

### Rule 4: Research Folders (keep last 10)

```bash
# List research folders sorted by modification time
ls -td "$(get_hermes_home)/research"/* 2>/dev/null | tail -n +11 | while read dir; do
  echo "Delete old research folder: $dir? [y/N]"
  # Prompt user for confirmation
done
```

Prompt before deleting older than last 10.

### Rule 5: Chrome Debug Profiles (> 14 days)

```bash
find "$(get_hermes_home)/.local/share/notebooklm-mcp" -type d -name "chrome_profile*" -mtime +14
```

Warn + offer to trim.

### Rule 6: Large Files (> 500 MB)

```bash
find "$(get_hermes_home)" -type f -size +500M -exec ls -lh {} \;
```

Warn + offer to delete if looks like temp download.

## Sub-Command Implementations

### /cleanup dry-run

Preview what would be deleted without touching anything:

```bash
python disk_guardian.py dry-run
```

Returns list of files that would be deleted by each rule, with total size.

### /cleanup quick

Safe fast clean, no confirmation needed:

```bash
python disk_guardian.py quick
```

Applies Rules 1-3 (temp, test, empty dirs). Returns summary: "Deleted 15 files, freed 234 MB"

### /cleanup deep

Full scan, confirm before anything > 100 MB or research folders:

```bash
python disk_guardian.py deep
```

Applies all rules. For risky items (research folders, large files, chrome profiles), prompts user for confirmation. Returns detailed breakdown by category.

### /cleanup status

Disk usage breakdown by category + top 10 largest Hermes files:

```bash
python disk_guardian.py status
```

Returns table with categories (temp, test, research, download, chrome-profile, other) and disk usage, plus top 10 largest files.

### /cleanup forget <path>

Remove a path from tracking permanently:

```bash
python disk_guardian.py forget "$(shlex.quote "$path")"
```

Removes entry from tracked.json and logs action.

## Integration with deep-research-monitor

If deep-research-monitor skill is present, offer to clean/archive the research folder after `/deep-research stop <topic>`:

```bash
# After deep-research stops, prompt user:
echo "Research complete. Clean up old research folders? [y/N]"
# If yes, run: python disk_guardian.py deep --category research
```

## Pitfalls to Avoid

1. **Never hardcode `~/.hermes`** - Always use `get_hermes_home()` for path resolution
2. **Always use `shlex.quote()`** - When interpolating user input into shell commands
3. **Don't delete outside Hermes home** - Validate all paths are under Hermes home directory
4. **Don't delete research artifacts without confirmation** - These are valuable user data
5. **Don't delete large files without warning** - User may need them
6. **Don't assume WSL2** - Check `/proc/version` for "microsoft" marker
7. **Don't delete Windows drives in WSL2** - Skip `/mnt/c/` and other Windows mounts
8. **Don't corrupt tracked.json** - Use file locking and atomic writes
9. **Don't ignore errors** - Log all errors and provide user feedback
10. **Don't require cron** - Skill works perfectly without cron integration

## Error Handling

| Error Scenario | Detection | Recovery | User Message |
|----------------|-----------|----------|--------------|
| tracked.json corrupted | JSON parse fails | Backup file restoration | "Tracking file corrupted, restored from backup" |
| Permission denied | OSError on delete | Skip file, log error | "Cannot delete <path>: permission denied" |
| Path doesn't exist | FileNotFoundError | Remove from tracking | "Path <path> no longer exists, removed from tracking" |
| Disk full | OSError on write | Abort operation, report | "Disk full, cannot write to tracking file" |
| Invalid category | Category not in allowed list | Default to "other" | "Unknown category, using 'other'" |
| WSL detection fails | /proc/version unreadable | Assume non-WSL | "Cannot detect WSL, using default paths" |
| Cron registration fails | crontab command fails | Continue without cron | "Cron registration failed, skill works without it" |
| Concurrent cleanup | File lock fails | Wait and retry | "Another cleanup in progress, waiting..." |

## Example Interactions

**User:** "My disk is full, help me clean up"

**Agent:** "I'll run a quick cleanup to remove temp files and test outputs. This won't delete any research artifacts or large files without asking."

```bash
python disk_guardian.py quick
```

**Output:** "Deleted 23 temp files, freed 456 MB. Run `/cleanup deep` for more thorough cleanup including research folders and large files."

---

**User:** "Show me what's taking up space"

**Agent:** "Here's your disk usage breakdown:"

```bash
python disk_guardian.py status
```

**Output:**
```
Category        | Files | Size
----------------|-------|--------
temp            | 156   | 2.3 GB
test            | 45    | 890 MB
research        | 12    | 15.6 GB
download        | 3     | 4.2 GB
chrome-profile  | 2     | 1.1 GB
other           | 89    | 3.4 GB

Top 10 largest files:
1. /home/user/.hermes/research/llm-training/model.gguf (4.2 GB)
2. /home/user/.hermes/cache/hermes/trajectory_cache.db (2.1 GB)
3. /home/user/.hermes/downloads/dataset.zip (1.8 GB)
...
```

---

**User:** "Clean up everything, including research folders"

**Agent:** "I'll run a deep cleanup. This will prompt you before deleting research folders and large files."

```bash
python disk_guardian.py deep
```

**Output:**
```
Found 12 research folders. Keeping last 10 by modification time.
Delete old research folder: /home/user/.hermes/research/old-project? [y/N] y
Deleted: /home/user/.hermes/research/old-project (1.2 GB)

Found 3 files > 500 MB:
1. /home/user/.hermes/downloads/model.gguf (4.2 GB) - looks like temp download
Delete this file? [y/N] y
Deleted: /home/user/.hermes/downloads/model.gguf (4.2 GB)

Summary: Deleted 2 items, freed 5.4 GB
```

---

**User:** "I want to see what would be deleted without actually deleting anything"

**Agent:** "Running dry-run to preview cleanup actions:"

```bash
python disk_guardian.py dry-run
```

**Output:**
```
Would delete 156 temp files (2.3 GB)
Would delete 45 test files (890 MB)
Would delete 23 empty directories
Would prompt for 2 research folders (3.4 GB)
Would prompt for 3 large files (6.0 GB)

Total potential cleanup: 12.5 GB
Run `/cleanup quick` for safe auto-cleanup
Run `/cleanup deep` for full cleanup with confirmation
```
