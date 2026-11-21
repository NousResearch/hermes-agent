#!/usr/bin/env bash
# verify_notebooklm.sh — End-to-end NotebookLM auth + connectivity check.
#
# Checks: install, auth (login --check), doctor, and a notebook list.
# Exit codes: 0 ok / 1 not installed / 2 not authenticated / 3 notebook list failed
set -euo pipefail

NLM="${NLM:-nlm}"

echo "=== NotebookLM install ==="
if ! command -v "$NLM" >/dev/null 2>&1; then
  echo "✗ nlm not on PATH. Ask the user to install a pinned release, e.g.: uv tool install notebooklm-mcp-cli==<VERSION>"
  exit 1
fi
echo "✓ nlm found: $(command -v "$NLM")"
"$NLM" --version 2>/dev/null || true

echo ""
echo "=== Doctor ==="
"$NLM" doctor 2>&1 || true

echo ""
echo "=== Auth check ==="
if "$NLM" login --check >/dev/null 2>&1; then
  echo "✓ Authenticated"
  "$NLM" login --check 2>&1 | head -5 || true
else
  echo "✗ Not authenticated (or cookies stale)."
  echo "  Headless options:"
  echo "    1) nlm login --manual --file cookies.txt"
  echo "    2) nlm login --provider openclaw --cdp-url http://127.0.0.1:18800"
  echo "    3) nlm login  (on a desktop with Chrome, then copy auth.json)"
  exit 2
fi

echo ""
echo "=== Notebook list ==="
if "$NLM" notebook list >/dev/null 2>&1; then
  echo "✓ Notebook list reachable"
  "$NLM" notebook list 2>&1 | head -20 || true
  exit 0
else
  echo "✗ Notebook list failed (auth may be stale despite --check passing)."
  echo "  Try: nlm login --manual --file cookies.txt  (re-extract cookies)"
  exit 3
fi
