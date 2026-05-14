#!/usr/bin/env bash
# Canonical local validation gate for PR handoff.
#
# This is the one command contributors and work-pack agents should run before
# opening or handing off a PR. It mirrors the blocking checks that protect main:
#   * git whitespace/conflict-marker validation
#   * uv.lock / pyproject.toml consistency
#   * blocking ruff rules
#   * Windows cross-platform footgun scanner
#   * hermetic pytest wrapper over the validation smoke tests
#
# Usage:
#   scripts/check.sh
#   scripts/check.sh tests/
#   scripts/check.sh tests/agent/test_foo.py::test_bar

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

VENV=""
PYTHON_VERSION="${HERMES_CHECK_PYTHON:-3.11}"

venv_matches_python() {
  local candidate="$1"
  [ -f "$candidate/bin/activate" ] || return 1
  "$candidate/bin/python" - "$PYTHON_VERSION" <<'PY'
import sys

want = tuple(int(part) for part in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[: len(want)] == want else 1)
PY
}

for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  if venv_matches_python "$candidate"; then
    VENV="$candidate"
    break
  fi
done

if [ -z "$VENV" ]; then
  while IFS= read -r line; do
    case "$line" in
      worktree\ *)
        worktree_path="${line#worktree }"
        for candidate in "$worktree_path/.venv" "$worktree_path/venv"; do
          if venv_matches_python "$candidate"; then
            VENV="$candidate"
            break 2
          fi
        done
        ;;
    esac
  done < <(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null || true)
fi

if [ -z "$VENV" ]; then
  echo "error: no Python $PYTHON_VERSION virtualenv found in $REPO_ROOT/.venv, $REPO_ROOT/venv, or linked worktrees" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $name"
  "$@"
}

run_step "git diff --check (worktree)" git diff --check

BASE_REF="${HERMES_CHECK_BASE_REF:-origin/main}"
if git rev-parse --verify --quiet "$BASE_REF^{commit}" >/dev/null; then
  MERGE_BASE="$(git merge-base "$BASE_REF" HEAD)"
  run_step "git diff --check ($BASE_REF...HEAD)" git diff --check "$MERGE_BASE" HEAD
else
  echo
  echo "error: cannot verify committed diff; base ref '$BASE_REF' was not found" >&2
  echo "  fetch the base branch or set HERMES_CHECK_BASE_REF to a local commit/ref" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  run_step "uv lock --check" uv lock --check
else
  echo
  echo "error: uv is required for the local validation gate" >&2
  echo "  install: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

run_step "ruff check ." "$PYTHON" -m ruff check .
run_step "Windows footgun checker" "$PYTHON" scripts/check-windows-footguns.py --all

if [ "$#" -eq 0 ]; then
  TEST_ARGS=(tests/scripts/test_validation_gate_scripts.py -q)
else
  TEST_ARGS=("$@")
fi

run_step "pytest" scripts/run_tests.sh "${TEST_ARGS[@]}"

echo
echo "Local validation gate passed."
