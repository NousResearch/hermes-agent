#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="root@888933.xyz"
REMOTE_DIR="/root/projects/hermes-agent"
REMOTE_HERMES_HOME="/root/.hermes"
REMOTE_PYTHON=""
TAIL_LINES=80
DO_SYNC=1
DO_RESTART=1
FILES=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/deploy_gateway.sh [options] [relative_file ...]

Options:
  --host HOST                 Remote SSH host. Default: root@888933.xyz
  --remote-dir PATH           Remote project directory. Default: /root/projects/hermes-agent
  --remote-hermes-home PATH   Remote HERMES_HOME. Default: /root/.hermes
  --remote-python PATH        Remote Python interpreter. Default: <remote-dir>/venv/bin/python
  --tail N                    Tail N log lines during verification. Default: 80
  --no-sync                   Skip file sync and only restart + verify
  --no-restart                Sync files but skip restart
  --help                      Show this help

Examples:
  bash scripts/deploy_gateway.sh gateway/run.py gateway/platforms/qq_napcat.py
  bash scripts/deploy_gateway.sh --no-sync
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --remote-hermes-home)
      REMOTE_HERMES_HOME="$2"
      shift 2
      ;;
    --remote-python)
      REMOTE_PYTHON="$2"
      shift 2
      ;;
    --tail)
      TAIL_LINES="$2"
      shift 2
      ;;
    --no-sync)
      DO_SYNC=0
      shift
      ;;
    --no-restart)
      DO_RESTART=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      FILES+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${REMOTE_PYTHON}" ]]; then
  REMOTE_PYTHON="${REMOTE_DIR}/venv/bin/python"
fi

if [[ ${DO_SYNC} -eq 1 && ${#FILES[@]} -eq 0 ]]; then
  echo "No files specified; proceeding with restart/verify only."
  DO_SYNC=0
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_root="${REMOTE_DIR}/.deploy-backups/${timestamp}"

echo "Target host: ${HOST}"
echo "Remote dir : ${REMOTE_DIR}"
echo "Hermes home: ${REMOTE_HERMES_HOME}"
echo "Remote py  : ${REMOTE_PYTHON}"

if [[ ${DO_SYNC} -eq 1 ]]; then
  echo "Preparing remote backup root: ${backup_root}"
  ssh "${HOST}" "mkdir -p '${backup_root}'"

  copy_file() {
    local src="$1"
    local dst="$2"
    if command -v rsync >/dev/null 2>&1; then
      rsync -az "${src}" "${HOST}:${dst}"
      return
    fi
    scp -q "${src}" "${HOST}:${dst}"
  }

  for rel_path in "${FILES[@]}"; do
    local_path="${PROJECT_DIR}/${rel_path}"
    if [[ ! -f "${local_path}" ]]; then
      echo "Missing local file: ${rel_path}" >&2
      exit 1
    fi

    remote_path="${REMOTE_DIR}/${rel_path}"
    remote_parent="$(dirname "${remote_path}")"
    backup_path="${backup_root}/${rel_path}"
    backup_parent="$(dirname "${backup_path}")"

    echo "Syncing ${rel_path}"
    ssh "${HOST}" "mkdir -p '${remote_parent}' '${backup_parent}' && if [[ -f '${remote_path}' ]]; then cp '${remote_path}' '${backup_path}'; fi"
    copy_file "${local_path}" "${remote_path}"
  done
fi

if [[ ${DO_RESTART} -eq 1 ]]; then
  echo "Restarting gateway"
  ssh "${HOST}" "nohup '${REMOTE_PYTHON}' -m hermes_cli.main gateway run --replace >/tmp/hermes-gateway.log 2>&1 </dev/null & echo \$!"
fi

echo "Waiting for gateway health"
ssh "${HOST}" "sleep 3; cat '${REMOTE_HERMES_HOME}/gateway_state.json'; printf '\n---\n'; ps -ef | grep -E 'hermes_cli.main gateway run --replace|venv/bin/hermes gateway run' | grep -v grep || true; printf '\n---\n'; tail -n '${TAIL_LINES}' /tmp/hermes-gateway.log"
