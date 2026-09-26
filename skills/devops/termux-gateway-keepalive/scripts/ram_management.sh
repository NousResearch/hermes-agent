#!/usr/bin/env bash
# ram_management.sh — Termux storage and RAM hygiene system.
# Cleans: pip build leftovers, dead tmp files, chromium caches, and telemetry buildup.
# Preserves: .env, configs, vault, credentials, active sessions.
set -u
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
HOME_DIR="$HOME"

before=$(df -k /data 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)

echo "═══ TERMUX HYGIENE — $(date '+%m-%d %H:%M') ═══"

# 1. PREFIX/tmp: pip-install leftovers + tmp* older than 2 hours
if [ -d "$PREFIX/tmp" ]; then
  CNT=$(find "$PREFIX/tmp" -maxdepth 1 -name "pip-install-*" -mmin +120 2>/dev/null | wc -l || echo 0)
  find "$PREFIX/tmp" -maxdepth 1 -name "pip-install-*" -mmin +120 -exec rm -rf {} \; 2>/dev/null || true
  echo "[1] pip-install leftovers removed: $CNT dirs"

  CNT2=$(find "$PREFIX/tmp" -maxdepth 1 -name "tmp*" -mmin +360 2>/dev/null | wc -l || echo 0)
  find "$PREFIX/tmp" -maxdepth 1 -name "tmp*" -mmin +360 -exec rm -rf {} \; 2>/dev/null || true
  echo "[2] old tmp* dirs removed: $CNT2"

  CNT3=$(find "$PREFIX/tmp" -maxdepth 1 -name "hermes-cwd-*" -mtime +1 2>/dev/null | wc -l || echo 0)
  find "$PREFIX/tmp" -maxdepth 1 -name "hermes-cwd-*" -mtime +1 -delete 2>/dev/null || true
  echo "[3] stale hermes-cwd snapshots: $CNT3"

  find "$PREFIX/tmp" -maxdepth 1 -name "adb.*.log" -mtime +3 -delete 2>/dev/null || true
fi

# 2. Chromium profile caches (safe — rebuilt automatically on launch)
CC="$HOME_DIR/.chromium-cdp"
if [ -d "$CC" ]; then
  for d in "Default/Cache" "Default/Code Cache" "Default/GPUCache" "GrShaderCache" "Default/DawnWebGPUCache" "Default/DawnGraphiteCache" "component_crx_cache"; do
    if [ -d "$CC/$d" ]; then
      SZ=$(du -sk "$CC/$d" 2>/dev/null | cut -f1 || echo 0)
      rm -rf "$CC/$d"/* 2>/dev/null || true
      echo "[4] chromium cache $d: ${SZ}KB cleared"
    fi
  done
fi

# 3. Home tmp dir old files
if [ -d "$HOME_DIR/tmp" ]; then
  find "$HOME_DIR/tmp" -type f -mtime +7 -delete 2>/dev/null || true
  echo "[5] ~/tmp files >7d purged"
fi

# 4. Chromium telemetry metrics (.pma files that never rotate offline)
BM="$HOME_DIR/.config/chromium/BrowserMetrics"
DBM="$HOME_DIR/.config/chromium/DeferredBrowserMetrics"
for d in "$BM" "$DBM"; do
  if [ -d "$d" ]; then
    SZ=$(du -sk "$d" 2>/dev/null | cut -f1 || echo 0)
    if [ "${SZ:-0}" -gt 51200 ]; then
      find "$d" -name "*.pma" -delete 2>/dev/null || true
      echo "[6] chromium telemetry purged: ${SZ}KB"
    fi
  fi
done

after=$(df -k /data 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
if [ "$before" -gt 0 ] && [ "$after" -ge "$before" ]; then
  freed_mb=$(( (after - before) / 1024 ))
  echo "────────────────────────────"
  echo "Storage freed: ${freed_mb} MB"
fi
free -m 2>/dev/null | awk 'NR==2{print "RAM available:", $7"MB"}' || true
echo "═══ done ═══"
