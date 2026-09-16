#!/bin/bash
# Hermes Shared Memory - Master Sync Script
# ROLE: MASTER ONLY - 主节点专用，基准权威机器使用
# 同步本地基准数据到GitHub，强制保持main分支与本机一致
# AUTO DETECT: Check if this machine matches the recorded master ID

MASTER_ID=$(cat $(dirname "$0")/.master-id)
CURRENT_ID=$(hostname)

if [ "$CURRENT_ID" != "$MASTER_ID" ]; then
    echo "❌ ERROR: This is NOT the authorized MASTER machine!"
    echo "   Recorded master ID: $MASTER_ID"
    echo "   Current machine ID: $CURRENT_ID"
    echo "   This script can ONLY run on the authoritative master machine."
    echo "   For slave machines, use ./sync-slave.sh or ./sync-slave-push.sh instead."
    exit 1
fi

echo "=== Hermes Master: Syncing all data to GitHub... ==="
echo "ROLE: MASTER - This machine is the AUTHORITY, will maintain main branch"
echo "AUTH MATCH: $CURRENT_ID == $MASTER_ID ✓"
echo ""

# Sync all content from master to local repo
echo "1/4 Copying Hermes memory..."
mkdir -p hermes/
rsync -av --delete /opt/data/memories/* hermes/memory
echo "   Note: config.yaml and .env are NOT synced (machine-specific sensitive), kept out of repo"

echo "2/4 Copying custom skills..."
rsync -av --delete /opt/data/hermes-skills/* hermes-skills/

echo "3/4 Copying MemPalace memory maze..."
rsync -av --delete /opt/data/mempalace/* mempalace/

echo "4/4 Copying Wiki knowledge base..."
rsync -av --delete /opt/data/wikis/* wikis/

echo ""
echo "=== Committing and pushing to GitHub... ==="
git add .
git diff --cached --quiet
if [ $? -eq 0 ]; then
    echo "✅ No changes detected, everything up to date"
    exit 0
fi
git commit -m "Master auto sync: $(date '+%Y-%m-%d %H:%M')"
git push -f origin main

echo ""
echo "✅ Master sync complete! GitHub main is now aligned with this machine."
echo "   Slaves can ./sync-slave.sh pull to get the latest authoritative version."
