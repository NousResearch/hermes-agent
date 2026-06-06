#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
VPS_HOST="${HERMES_VPS_HOST:-linux-nat@103.142.150.185}"
VPS_PATH="${HERMES_VPS_PATH:-/home/linux-nat/projects/hermes-agent/main}"
DRY_RUN=""

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
fi

case "$VPS_PATH" in
  /home/linux-nat/projects/hermes-agent|/srv/projects/hermes-agent)
    echo "Refusing to sync into the project container. Use a real worktree path such as /home/linux-nat/projects/hermes-agent/main." >&2
    exit 2
    ;;
esac

ssh "$VPS_HOST" "mkdir -p '$VPS_PATH'"

rsync -az --delete $DRY_RUN \
  --include='/.env.example' \
  --include='/.envrc' \
  --exclude='/.git' \
  --exclude='/.git/' \
  --exclude='/.env' \
  --exclude='/.env.*' \
  --exclude='/.hermes/.env' \
  --exclude='/venv' \
  --exclude='/venv/' \
  --exclude='/.venv' \
  --exclude='/.venv/' \
  --exclude='/node_modules' \
  --exclude='/node_modules/' \
  --exclude='/web/node_modules' \
  --exclude='/web/node_modules/' \
  --exclude='/ui-tui/node_modules' \
  --exclude='/ui-tui/node_modules/' \
  --exclude='/website/node_modules' \
  --exclude='/website/node_modules/' \
  --exclude='/scripts/whatsapp-bridge/node_modules' \
  --exclude='/scripts/whatsapp-bridge/node_modules/' \
  --exclude='/hermes_cli/web_dist/' \
  --exclude='/ui-tui/dist/' \
  --exclude='/.pytest_cache/' \
  --exclude='/.mypy_cache/' \
  --exclude='/.ruff_cache/' \
  --exclude='/.direnv/' \
  --exclude='/__pycache__/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='/logs/' \
  --exclude='/tmp/' \
  --exclude='/data/' \
  --exclude='/wandb/' \
  --exclude='/result' \
  --exclude='/skills/.hub/' \
  "$SOURCE_DIR/" "$VPS_HOST:$VPS_PATH/"

if [[ -n "$DRY_RUN" ]]; then
  echo "Dry run complete for $SOURCE_DIR -> $VPS_HOST:$VPS_PATH"
else
  echo "Sync complete for $SOURCE_DIR -> $VPS_HOST:$VPS_PATH"
fi
