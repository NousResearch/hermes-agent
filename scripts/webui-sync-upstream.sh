#!/usr/bin/env bash
set -euo pipefail

# Sync YallaPlay's forked Hermes WebUI checkout with upstream.
# Defaults are for this repo's local deployment.
#
# Usage:
#   scripts/webui-sync-upstream.sh
#   scripts/webui-sync-upstream.sh --restart-admin
#   scripts/webui-sync-upstream.sh --no-restart

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBUI_DIR="${WEBUI_DIR:-$ROOT_DIR/.local/hermes-webui}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-master}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-yallaplay/main}"
RESTART_PUBLIC=1
RESTART_ADMIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart-admin)
      RESTART_ADMIN=1
      ;;
    --no-restart)
      RESTART_PUBLIC=0
      RESTART_ADMIN=0
      ;;
    --webui-dir)
      WEBUI_DIR="$2"
      shift
      ;;
    --deploy-branch)
      DEPLOY_BRANCH="$2"
      shift
      ;;
    --upstream-branch)
      UPSTREAM_BRANCH="$2"
      shift
      ;;
    -h|--help)
      sed -n '1,32p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

cd "$WEBUI_DIR"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to sync: WebUI checkout has uncommitted changes:" >&2
  git status --short >&2
  exit 1
fi

if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
  echo "Missing remote '$UPSTREAM_REMOTE'. Expected upstream Hermes WebUI remote." >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$DEPLOY_BRANCH" ]]; then
  git checkout "$DEPLOY_BRANCH"
fi

git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
git fetch origin "$DEPLOY_BRANCH" || true

echo "Merging $UPSTREAM_REMOTE/$UPSTREAM_BRANCH into $DEPLOY_BRANCH..."
git merge "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"

echo "Running verification..."
python3 -m py_compile api/routes.py
node --check static/boot.js
node --check static/panels.js

if [[ "$RESTART_PUBLIC" == "1" ]]; then
  echo "Restarting hermes-webui-public..."
  systemctl --user restart hermes-webui-public
  systemctl --user is-active hermes-webui-public >/dev/null
fi

if [[ "$RESTART_ADMIN" == "1" ]]; then
  echo "Restarting hermes-webui..."
  systemctl --user restart hermes-webui
  systemctl --user is-active hermes-webui >/dev/null
fi

if command -v curl >/dev/null 2>&1 && [[ "$RESTART_PUBLIC" == "1" ]]; then
  echo "Verifying simple UI settings endpoint..."
  settings_file="${TMPDIR:-/tmp}/hermes-webui-public-settings.$$.$RANDOM.json"
  for attempt in $(seq 1 40); do
    if curl -fs http://127.0.0.1:9122/api/settings -o "$settings_file"; then
      break
    fi
    sleep 0.5
  done
  python3 - "$settings_file" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
print({"simple_ui": s.get("simple_ui"), "hidden_tabs": s.get("hidden_tabs")})
if s.get("simple_ui") is not True:
    raise SystemExit("simple_ui was not true")
PY
  rm -f "$settings_file"
fi

echo "WebUI sync complete. Current HEAD: $(git rev-parse --short HEAD)"
