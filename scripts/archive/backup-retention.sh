#!/usr/bin/env bash
# backup-retention.sh — Retain last 7 daily backups, delete older than 14 days
# Logs to /home/kensei/backups/retention.log
set -euo pipefail

BACKUP_DIR="/home/kensei/backups/daily"
LOG_FILE="/home/kensei/backups/retention.log"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN: No files will be deleted." | tee -a "$LOG_FILE"
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Starting backup retention policy" >> "$LOG_FILE"

# Find backup tarballs
mapfile -t BACKUPS < <(find "$BACKUP_DIR" -maxdepth 1 -name 'hermes-*.tar.gz' -type f | sort)

if [[ ${#BACKUPS[@]} -eq 0 ]]; then
    echo "[$TIMESTAMP] No backup files found." >> "$LOG_FILE"
    exit 0
fi

echo "[$TIMESTAMP] Found ${#BACKUPS[@]} backup files." >> "$LOG_FILE"

# Determine files to keep (newest 7)
KEEP_COUNT=7
if [[ ${#BACKUPS[@]} -le $KEEP_COUNT ]]; then
    echo "[$TIMESTAMP] No deletions needed (have ${#BACKUPS[@]} backups, keeping latest $KEEP_COUNT)." >> "$LOG_FILE"
    exit 0
fi

# Arrays for indexing
TO_KEEP=("${BACKUPS[@]: -$KEEP_COUNT}")   # last KEEP_COUNT elements
TO_DELETE=("${BACKUPS[@]:0:${#BACKUPS[@]}-$KEEP_COUNT}")   # all except last KEEP_COUNT

echo "[$TIMESTAMP] Will keep ${#TO_KEEP[@]} newest backups:" >> "$LOG_FILE"
for f in "${TO_KEEP[@]}"; do
    echo "  - $(basename "$f")" >> "$LOG_FILE"
done

echo "[$TIMESTAMP] Will delete ${#TO_DELETE[@]} older backups:" >> "$LOG_FILE"
for f in "${TO_DELETE[@]}"; do
    echo "  - $(basename "$f")" >> "$LOG_FILE"
    if [[ $DRY_RUN == false ]]; then
        rm -f "$f"
        echo "    Deleted." >> "$LOG_FILE"
    else
        echo "    [DRY RUN] Would delete." >> "$LOG_FILE"
    fi
done

# Additionally, delete any files older than 14 days (based on mtime) as safety
MAX_DAYS=14
NOW=$(date +%s)
echo "[$TIMESTAMP] Checking for backups older than $MAX_DAYS days (mtime)..." >> "$LOG_FILE"
OLD_COUNT=0
while IFS= read -r -d '' oldfile; do
    # Only consider if not already in TO_DELETE (to avoid double log)
    if [[ " ${TO_DELETE[*]} " != *" $oldfile "* ]]; then
        OLD_COUNT=$((OLD_COUNT+1))
        echo "[$TIMESTAMP] Found old backup (mtime >$MAX_DAYS days): $(basename "$oldfile")" >> "$LOG_FILE"
        if [[ $DRY_RUN == false ]]; then
            rm -f "$oldfile"
            echo "    Deleted." >> "$LOG_FILE"
        else
            echo "    [DRY RUN] Would delete." >> "$LOG_FILE"
        fi
    fi
done < <(find "$BACKUP_DIR" -maxdepth 1 -name 'hermes-*.tar.gz' -type f -mtime +$MAX_DAYS -print0)

if [[ $OLD_COUNT -eq 0 ]]; then
    echo "[$TIMESTAMP] No backups older than $MAX_DAYS days found." >> "$LOG_FILE"
fi

echo "[$TIMESTAMP] Retention policy finished." >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit 0