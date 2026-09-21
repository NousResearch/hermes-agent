#!/usr/bin/env bash
# backup-health-check.sh — Checks that daily backups are recent and present
# Runs as a no_agent cron at 6am daily, silent when healthy
set -euo pipefail

BACKUP_DIR="/home/kensei/backups/daily"
LOG_FILE="${BACKUP_DIR}/backup.log"
NOW=$(date +%s)

# Find the most recent backup attempt from the log
LAST_YYYYMMDD=$(grep "^Backup:" "${LOG_FILE}" 2>/dev/null | tail -1 | sed 's/.*hermes-//' | sed 's/-.*//')
if [ -z "${LAST_YYYYMMDD}" ]; then
    echo "ALERT: No backup records found — backup may never have run."
    exit 0
fi

# Convert YYYYMMDD to epoch
LAST_EPOCH=$(date -d "${LAST_YYYYMMDD}" +%s 2>/dev/null || echo 0)
GAP_DAYS=$(( (NOW - LAST_EPOCH) / 86400 ))

if [ $GAP_DAYS -ge 2 ]; then
    # Convert to UK date format DD/MM/YYYY for Discord output
    LAST_UK=$(date -d "${LAST_YYYYMMDD}" +%d/%m/%Y 2>/dev/null || echo "${LAST_YYYYMMDD}")
    echo "ALERT: Backup gap — last backup was ${LAST_UK} (${GAP_DAYS} days ago)."
fi

# Check disk usage of backup directory
DISK_GB=$(du -sb "${BACKUP_DIR}" 2>/dev/null | awk '{printf "%.0f", $1/1073741824}')
if [ "${DISK_GB}" -gt 30 ]; then
    echo "WARN: Backup directory is ${DISK_GB}GB — approaching 30GB threshold."
fi

# Count actual backup files
FILE_COUNT=$(ls "${BACKUP_DIR}"/hermes-*.tar.gz 2>/dev/null | wc -l)
if [ "${FILE_COUNT}" -lt 3 ]; then
    echo "WARN: Only ${FILE_COUNT} backup files on disk — fewer than expected."
fi

# Silent exit when healthy (watchdog pattern: no news is good news)
exit 0