#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/5] Checking Hermes CLI"
if ! command -v hermes >/dev/null 2>&1; then
  echo "ERROR: hermes CLI not found in PATH"
  exit 1
fi

echo "[2/5] Checking starter-kit files"
required_files=(
  "$ROOT_DIR/README.md"
  "$ROOT_DIR/prompts/weekly-kickoff.md"
  "$ROOT_DIR/prompts/daily-ceo-review.md"
  "$ROOT_DIR/prompts/evening-doc-sync.md"
  "$ROOT_DIR/prompts/friday-ship-review.md"
  "$ROOT_DIR/templates/weekly-mvp-factory-template.md"
  "$ROOT_DIR/templates/mvp-pipeline-template.md"
  "$ROOT_DIR/templates/ceo-note-template.md"
  "$ROOT_DIR/templates/ship-checklist-template.md"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: missing required file: $file"
    exit 1
  fi
done

echo "[3/5] Checking Hermes home"
HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
echo "Using HERMES_HOME=$HERMES_HOME_DIR"
mkdir -p "$HERMES_HOME_DIR"

echo "[4/5] Printing recommended schedules"
echo "  weekly kickoff     -> 0 9 * * 1"
echo "  daily CEO review   -> 0 9 * * 2-5"
echo "  evening doc sync   -> 0 18 * * 1-5"
echo "  friday ship review -> 0 15 * * 5"

echo "[5/5] Next action"
echo "Create the four jobs with the prompts in $ROOT_DIR/prompts and point them at your project notes."

echo "Preflight OK"
