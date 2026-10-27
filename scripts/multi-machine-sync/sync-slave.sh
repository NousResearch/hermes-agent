#!/bin/bash
# Hermes Shared Memory - Slave Pull Sync Script
# ROLE: SLAVE ONLY - 从节点专用，只用于拉取最新同步更新本地文件
# 从不推送，只拉取主节点最新基准
# Configure TARGET_ROOT below to match your machine's deployment path

# --------------------------
# CONFIGURE THIS FOR YOUR MACHINE:
# Default: TARGET_ROOT="/opt/data" (for master-style deployment)
# If your Hermes data is under ~/.hermes: TARGET_ROOT="/root/.hermes"
TARGET_ROOT="/root/.hermes"
# --------------------------

echo "=== Hermes Slave: Pulling latest sync from GitHub... ==="
echo "ROLE: SLAVE - This script only pulls updates from GitHub main"
echo "TARGET_ROOT configured as: $TARGET_ROOT"
echo ""

# Pull latest changes from GitHub
git pull origin main

echo ""
echo "=== Copying synced files to target locations... ==="

# Copy to target locations
echo "1/4 Copying Hermes memory..."
mkdir -p $TARGET_ROOT/memories
rsync -av --delete hermes/memory/* $TARGET_ROOT/memories/
echo "   Note: config.yaml and .env are NOT synced (machine-specific), kept your local version"

echo "2/4 Copying custom skills..."
mkdir -p $TARGET_ROOT/hermes-skills
rsync -av --delete hermes-skills/* $TARGET_ROOT/hermes-skills/

echo "3/4 Copying MemPalace memory maze..."
mkdir -p $TARGET_ROOT/mempalace
rsync -av --delete mempalace/* $TARGET_ROOT/mempalace/

echo "4/4 Copying Wiki knowledge base..."
mkdir -p $TARGET_ROOT/wikis
rsync -av --delete wikis/* $TARGET_ROOT/wikis/

echo ""
echo "✅ Slave pull complete! Local memory updated from GitHub main"
echo ""
echo "Next steps: Restart your Hermes Agent to load the new memory:"
echo "   - If running as systemd service: sudo systemctl restart hermes"
echo "   - If running in container: restart the container"
