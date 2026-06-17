#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CONFIG="$ROOT/.gitleaks.toml"
MODE="${1:-workspace}"

if ! command -v gitleaks >/dev/null 2>&1; then
  cat >&2 <<'EOF'
gitleaks is required for secret scanning.

Install it with:
  brew install gitleaks

Then retry:
  scripts/check-secrets.sh
EOF
  exit 127
fi

case "$MODE" in
  --staged|staged)
    exec gitleaks git "$ROOT" --pre-commit --staged --config "$CONFIG" --redact --no-banner
    ;;
  --history|history)
    exec gitleaks git "$ROOT" --config "$CONFIG" --redact --no-banner
    ;;
  --workspace|workspace|"")
    exec gitleaks dir "$ROOT" --config "$CONFIG" --redact --no-banner
    ;;
  *)
    echo "Usage: scripts/check-secrets.sh [--staged|--workspace|--history]" >&2
    exit 2
    ;;
esac
