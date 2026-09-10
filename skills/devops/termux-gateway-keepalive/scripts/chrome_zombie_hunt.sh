#!/usr/bin/env bash
# chrome_zombie_hunt.sh — kill remaining chrome processes precisely by verifying /proc/$pid/exe path.
KILLED=0
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  exe=$(readlink /proc/$p/exe 2>/dev/null || echo "")
  case "$exe" in
    *chromium/chrome*|*chromium/*)
      kill -9 "$p" 2>/dev/null && KILLED=$((KILLED+1))
      ;;
  esac
done
echo "killed by exe-path: $KILLED"
sleep 1
C=0
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  exe=$(readlink /proc/$p/exe 2>/dev/null || echo "")
  case "$exe" in *chromium/chrome*) C=$((C+1));; esac
done
echo "chrome procs remaining: $C"
