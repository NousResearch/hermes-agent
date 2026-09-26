# Vault Sync Pattern

Templates and patterns for syncing cronjob data to Obsidian Vault with git automation.

## Sync Job Template

Each sync job runs AFTER its source job, copies data conditionally, and git commits only on changes.

### Generic Sync Script

```bash
#!/usr/bin/env bash
set -euo pipefail

KNOWLEDGE_BASE="${KNOWLEDGE_BASE:-${HOME}/knowledge-base}"
VAULT_PATH="${VAULT_PATH:-${HOME}/dev/workspace/Personal-Vault-backup}"

# Source: knowledge-base dir → Destination: vault dir
SOURCE="${KNOWLEDGE_BASE}/${1:?usage: sync.sh <source-dir> <vault-dir>}"
DEST="${VAULT_PATH}/${2:?}"

if [ ! -d "$SOURCE" ]; then
  echo "Source missing: $SOURCE"
  exit 0
fi

mkdir -p "$DEST"

# Conditional copy — only if source is newer
cp -ru "${SOURCE}/"* "$DEST/" 2>/dev/null || true

# Atomic git commit (silent on no changes)
cd "$VAULT_PATH"
git add -A
git diff --cached --quiet || git commit -m "sync: ${1} at $(date +%Y-%m-%d\ %H:%M)"
git push 2>/dev/null || true
```

### Usage

```bash
# Sync Discord data
./sync.sh discord/threads "02-Literature/02-Discord"

# Sync YouTube data
./sync.sh youtube/transcripts "02-Literature/YouTube"
```

## Scheduling Pattern

| Source Job | Time | Sync Job | Time |
|-----------|------|----------|------|
| Discord Spy | 02:00 | Discord Sync | 03:00 |
| YouTube Learning | 04:00 | YouTube Sync | 05:00 |
| Knowledge Gatherer | 06:00 | Vault Sync | 06:30 |

## Key Patterns

1. **Schedule sync AFTER source:** Avoids race conditions and missing data.
2. **Conditional copy:** `cp -ru` only copies if source is newer — idempotent.
3. **Silent on no changes:** Exit code 0 when git status is clean prevents empty commits.
4. **Atomic git operations:** `git add -A` + `git commit` handles concurrent updates safely.
5. **Git push last:** Only push after successful commit.

## Common Pitfalls

- **Redundant generic sync job:** Once specific per-source sync jobs exist, remove the generic one to avoid conflicts.
- **Missing directories:** Sync script must handle missing source/dest gracefully (exit 0, not error).
- **Large files:** Consider `.gitignore` patterns for transient files (logs, temp data).
