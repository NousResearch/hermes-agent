#!/bin/bash
# web-3d skill - lightweight setup check

set -euo pipefail

echo "=== web-3d setup check ==="
echo ""

if command -v node >/dev/null 2>&1; then
  echo "[OK] Node.js: $(node -v)"
else
  echo "[WARN] Node.js not found"
fi

if command -v npm >/dev/null 2>&1; then
  echo "[OK] npm: $(npm -v)"
else
  echo "[WARN] npm not found"
fi

if command -v python3 >/dev/null 2>&1; then
  echo "[OK] python3: $(python3 --version 2>&1)"
else
  echo "[WARN] python3 not found"
fi

echo ""
echo "Typical package set:"
echo "  npm install three"
echo "  npm install @react-three/fiber"
echo ""
echo "Install only what the target repo and implementation path actually need."
