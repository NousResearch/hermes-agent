#!/usr/bin/env bash
# Fast pre-PR validation for agent/developer workflows.
#
# This intentionally stays cheap: run the blocking lint/footgun checks before
# opening a PR, then use scripts/run_tests.sh for full or scoped pytest runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV=""
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  if [ -f "$candidate/bin/python" ]; then
    VENV="$candidate"
    break
  fi
done

if [ -z "$VENV" ]; then
  echo "error: no virtualenv found in $REPO_ROOT/.venv or $REPO_ROOT/venv" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"

cd "$REPO_ROOT"

echo "==> ruff"
"$PYTHON" -m ruff check .

echo "==> windows footguns"
FOOTGUN_BASE="${HERMES_VALIDATE_BASE:-}"
if [ -z "$FOOTGUN_BASE" ]; then
  for candidate in upstream/main origin/main main; do
    if git rev-parse --verify --quiet "$candidate" >/dev/null; then
      FOOTGUN_BASE="$candidate"
      break
    fi
  done
fi

FOOTGUN_PATHS=()
if [ -n "$FOOTGUN_BASE" ] && git merge-base "$FOOTGUN_BASE" HEAD >/dev/null 2>&1; then
  while IFS= read -r -d '' path; do
    FOOTGUN_PATHS+=("$path")
  done < <(git diff -z --name-only --diff-filter=ACMR "$FOOTGUN_BASE"...HEAD)
fi
while IFS= read -r -d '' path; do
  FOOTGUN_PATHS+=("$path")
done < <(git diff -z --name-only --diff-filter=ACMR)
while IFS= read -r -d '' path; do
  FOOTGUN_PATHS+=("$path")
done < <(git ls-files -z --others --exclude-standard)

if [ "${#FOOTGUN_PATHS[@]}" -eq 0 ]; then
  echo "No changed files to scan."
else
  "$PYTHON" scripts/check-windows-footguns.py "${FOOTGUN_PATHS[@]}"
fi

echo "ok: validation passed"
