#!/usr/bin/env bash
# tmp_inventory.sh — safe read-only temp directory analyzer for Termux.
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
cd "$PREFIX/tmp" 2>/dev/null || exit 1
echo "=== Top 12 Largest Items in $PREFIX/tmp ==="
du -sh ./* 2>/dev/null | sort -rh | head -12
echo ""
echo "=== File Count by Age ==="
find . -maxdepth 1 -type f -mtime +7 2>/dev/null | wc -l | xargs echo "Files older than 7 days:"
find . -maxdepth 1 -type f -mtime -1 2>/dev/null | wc -l | xargs echo "Files newer than 1 day:"
echo ""
echo "=== Sample Filenames ==="
ls -p 2>/dev/null | grep -v / | head -10
