#!/usr/bin/env bash
# Orca Environment Verification Script
# Author: Rafa Martins <rafacpti@gmail.com>

set -euo pipefail

echo "========================================="
echo "   🐋 Orca Environment & Health Check   "
echo "   Developer: Rafa Martins               "
echo "========================================="

# 1. Check user
echo -n "[1/5] Checking 'orca' system user: "
if id orca >/dev/null 2>&1; then
    echo "✓ Found (uid=$(id -u orca), gid=$(id -g orca))"
else
    echo "✗ Missing 'orca' user"
fi

# 2. Check service
echo -n "[2/5] Checking 'orca-serve.service': "
if systemctl is-active --quiet orca-serve 2>/dev/null; then
    echo "✓ Active (running)"
else
    echo "⚠ Inactive or not installed"
fi

# 3. Check CLI binary
echo -n "[3/5] Checking Orca CLI wrapper: "
if command -v orca >/dev/null 2>&1; then
    echo "✓ Available at $(which orca)"
else
    echo "✗ Not in PATH"
fi

# 4. Check Runtime Status
echo "[4/5] Checking Orca Runtime Engine:"
if command -v orca >/dev/null 2>&1; then
    orca status 2>&1 | sed 's/^/    /' || echo "    ✗ Failed to query status"
fi

# 5. Check Worktrees & Projects
echo "[5/5] Active Worktrees and Repositories:"
if command -v orca >/dev/null 2>&1; then
    echo "  -- Registered Repos --"
    orca repo list 2>&1 | sed 's/^/    /' || true
    echo "  -- Live Worktrees --"
    orca worktree list 2>&1 | sed 's/^/    /' || true
fi

echo "========================================="
echo "Check complete!"
