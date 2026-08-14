#!/usr/bin/env bash
# KENSEI GitOps — pre-commit lint
# 1. Block commits that touch forbidden paths (defence-in-depth against
#    .gitignore misses or `git add -f` abuse).
# 2. Lint any staged *.yaml / *.json so malformed config does not land.
#
# Deployed by: scripts/gitops/install.sh
# Lives at:    ~/.hermes/.git/hooks/pre-commit

set -euo pipefail

# Files that must never be committed, even via `git add -f`.
FORBIDDEN_PATTERNS=(
  '^\.env$'
  '^\.env\..*'
  '^auth\.json$'
  '.*/auth\.json$'
  '^auth\.lock$'
  '.*/auth\.lock$'
  '^credentials\.json$'
  '.*/credentials\.json$'
  '.*/auth\.json\..*'
  '.*/auth\.tmp\..*'
  '.*/auth\.lock\..*'
  '.*/auth\.lock\.tmp\..*'
  '.*/credentials\.json\..*'
  '.*/credentials\.tmp\..*'
  '.*/vault\.key$'
  '^vault\.key$'
  '.*/id_rsa.*'
  '^id_rsa.*'
  '.*/id_ed25519.*'
  '^id_ed25519.*'
  '.*/id_ecdsa.*'
  '^id_ecdsa.*'
  '.*\.lock$'
  '.*\.db$'
  '.*\.db-wal$'
  '.*\.db-shm$'
  '.*\.sqlite$'
  '.*\.sqlite3$'
  '^state\.db'
  '^sessions\.db'
  '^kanban_db\.sqlite'
  '^tasks\.db'
  '^response_store\.db'
  '^cron/cron\.db'
  '^content\.db'
  '^default\.db'
)

STAGED=$(git diff --cached --name-only --diff-filter=ACMRT)
if [[ -z "$STAGED" ]]; then
  exit 0
fi

# Check 1: forbidden path block.
for file in $STAGED; do
  for pat in "${FORBIDDEN_PATTERNS[@]}"; do
    if [[ "$file" =~ $pat ]]; then
      echo "pre-commit: refusing to commit $file (matches forbidden pattern: $pat)" >&2
      echo "  bypass with --no-verify ONLY if you are sure." >&2
      exit 1
    fi
  done
done

# Check 2: YAML / JSON parse lint.
LINT_FAILED=0
for file in $STAGED; do
  case "$file" in
    *.yaml|*.yml)
      if ! python3 -c "import sys, yaml; list(yaml.safe_load_all(open(sys.argv[1])))" "$file" 2>/dev/null; then
        echo "pre-commit: YAML parse error in $file" >&2
        LINT_FAILED=1
      fi
      ;;
    *.json)
      if ! python3 -c "import sys, json; json.load(open(sys.argv[1]))" "$file" 2>/dev/null; then
        echo "pre-commit: JSON parse error in $file" >&2
        LINT_FAILED=1
      fi
      ;;
  esac
done

if [[ $LINT_FAILED -ne 0 ]]; then
  echo "pre-commit: lint failed. Fix or use --no-verify." >&2
  exit 1
fi

exit 0
