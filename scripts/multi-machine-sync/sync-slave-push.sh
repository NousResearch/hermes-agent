#!/bin/bash
# Hermes Shared Memory - Slave Push Script
# ROLE: SLAVE ONLY - 从节点专用，用于推送从节点新增的记忆到GitHub
# 从不强制覆盖，冲突自动提示等待主节点处理

echo "=== Hermes Slave: Pushing local updates to GitHub... ==="
echo "ROLE: SLAVE - This is slave node only, will NOT force push master"
echo ""

# Pull latest changes from GitHub first
git pull origin main

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Merge conflict detected!"
    echo "   Conflict must be resolved by MASTER node"
    echo "   Please wait for master node to sync and force update main"
    echo "   Then you can git pull and try again"
    exit 1
fi

# Sync all content from local to repo
echo ""
echo "1/4 Copying Hermes memory..."
mkdir -p hermes/
rsync -av --delete /opt/data/memories/* hermes/memory/
echo "   Note: config.yaml and .env are NOT synced (machine-specific sensitive), kept out of repo"

echo "2/4 Copying custom skills..."
rsync -av --delete /opt/data/hermes-skills/* hermes-skills/

echo "3/4 Copying MemPalace memory maze..."
rsync -av --delete /opt/data/mempalace/* mempalace/

echo "4/4 Copying Wiki knowledge base..."
rsync -av --delete /opt/data/wikis/* wikis/

echo ""
echo "=== Committing changes... ==="
git add .

# Check if there are changes
git diff --cached --quiet
if [ $? -eq 0 ]; then
    echo "✅ No changes detected, nothing to push"
    exit 0
fi

git commit -m "Slave auto sync: $(date '+%Y-%m-%d %H:%M')"

echo ""
echo "=== Pushing to GitHub... ==="
git push origin main

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Push rejected! Probably remote has newer changes"
    echo "   Please wait for MASTER node to resolve conflict"
    echo "   Then git pull and try pushing again"
    exit 1
fi

echo ""
echo "✅ Slave push complete! Changes are now on GitHub"
echo "   Master node will review and merge to maintain authority"
