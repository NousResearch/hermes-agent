#!/usr/bin/env bash
# Bisection: find which test file leaves unwanted files/state behind ("pollution").
# Adapted from obra/superpowers (skills/systematic-debugging/find-polluter.sh).
#
# Usage:
#   ./find-polluter.sh <path_to_check> <test_file_glob> [run_command]
#
# run_command is executed per test file; use {file} as the placeholder.
# Defaults to 'npm test {file}'.
#
# Examples:
#   ./find-polluter.sh '.git' 'src/**/*.test.ts'
#   ./find-polluter.sh '.git' 'src/**/*.test.ts' 'npx vitest run {file}'
#   ./find-polluter.sh 'leftover.db' 'tests/**/*.py' 'pytest {file}'
#   ./find-polluter.sh 'TestResults' 'test/**/*.csproj' 'dotnet test {file}'

set -u

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <path_to_check> <test_file_glob> [run_command]"
  echo "  run_command may contain {file}; default: 'npm test {file}'"
  exit 1
fi

POLLUTION_CHECK="$1"
TEST_PATTERN="$2"
# NB: can't use ${3:-npm test {file}} — the } inside word would misparsed.
if [ $# -ge 3 ]; then
  RUN_CMD="$3"
else
  RUN_CMD='npm test {file}'
fi

# find . emits ./-prefixed paths, so accept the pattern with or without ./.
TEST_PATTERN="${TEST_PATTERN#./}"
# find -path can't match '**/' against zero directory levels, so also try the
# pattern with '**/' collapsed to cover files directly under the base dir.
TEST_FILES=$(find . \( -path "./$TEST_PATTERN" -o -path "./${TEST_PATTERN//\*\*\//}" \) | sort -u)
TOTAL=0
[ -n "$TEST_FILES" ] && TOTAL=$(printf '%s\n' "$TEST_FILES" | wc -l | tr -d ' ')

echo "Searching for the test that creates: $POLLUTION_CHECK"
echo "Test pattern: $TEST_PATTERN ($TOTAL files)"
echo "Run command:  $RUN_CMD"
echo

COUNT=0
for TEST_FILE in $TEST_FILES; do
  COUNT=$((COUNT + 1))
  if [ -e "$POLLUTION_CHECK" ]; then
    echo "[$COUNT/$TOTAL] Pollution already present — skipping $TEST_FILE"
    continue
  fi

  echo "[$COUNT/$TOTAL] $TEST_FILE"
  CMD="${RUN_CMD//\{file\}/$TEST_FILE}"
  sh -c "$CMD" > /dev/null 2>&1 || true

  if [ -e "$POLLUTION_CHECK" ]; then
    echo
    echo "FOUND POLLUTER: $TEST_FILE"
    echo "Created: $POLLUTION_CHECK"
    echo
    ls -la "$POLLUTION_CHECK"
    echo
    echo "Investigate with:"
    echo "  $CMD"
    exit 1
  fi
done

echo
echo "No polluter found — all $COUNT test files ran clean."
exit 0
