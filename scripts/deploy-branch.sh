#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: deploy-branch.sh <branch> [options]

Options:
  --remote <name>        Git remote to fetch from (default: origin)
  --repo <path>          Repo path on server (default: current directory)
  --skip-tests           Skip pytest run
  --restart-gateway      Restart Hermes gateway after deploy
  -h, --help             Show this help
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BRANCH="$1"
shift

REMOTE="origin"
REPO_PATH="$(pwd)"
SKIP_TESTS=0
RESTART_GATEWAY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --repo)
      REPO_PATH="${2:-}"
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --restart-gateway)
      RESTART_GATEWAY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH" >&2
  exit 1
fi

cd "$REPO_PATH"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository: $REPO_PATH" >&2
  exit 1
fi

echo "==> Fetching latest refs from $REMOTE"
git fetch "$REMOTE"

echo "==> Checking out branch $BRANCH from $REMOTE/$BRANCH"
git checkout -B "$BRANCH" "$REMOTE/$BRANCH"

echo "==> Syncing submodules"
git submodule update --init --recursive

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment (.venv)"
  uv venv .venv --python 3.11
fi

echo "==> Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing project dependencies"
uv pip install -e ".[all,dev]"
uv pip install -e "./mini-swe-agent"

if [[ -d "./tinker-atropos" ]]; then
  uv pip install -e "./tinker-atropos"
fi

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "==> Running test suite"
  pytest tests/ -q
else
  echo "==> Skipping tests (--skip-tests)"
fi

echo "==> Running Hermes diagnostics"
hermes doctor

if [[ "$RESTART_GATEWAY" -eq 1 ]]; then
  echo "==> Restarting Hermes gateway"
  hermes gateway restart
  hermes gateway status
fi

echo "Deploy complete for branch: $BRANCH"
