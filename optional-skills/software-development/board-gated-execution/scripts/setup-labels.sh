#!/usr/bin/env bash
# Create the label vocabulary the execution gate depends on.
#
# The gate reads Issues plus labels rather than a GitHub Project board, because
# Projects require the `read:project` scope that most tokens lack. Issues work
# with plain `repo` scope.
#
# Idempotent: `gh label create` fails if the label exists, so each call falls
# back to `edit`. Safe to re-run.
#
# Usage:  bash setup-labels.sh [owner/repo]
set -uo pipefail

REPO="${1:-}"
REPO_ARG=()
[ -n "$REPO" ] && REPO_ARG=(--repo "$REPO")

label() {
  local name="$1" color="$2" desc="$3"
  if gh label create "$name" "${REPO_ARG[@]}" --color "$color" --description "$desc" >/dev/null 2>&1; then
    echo "  created  $name"
  else
    gh label edit "$name" "${REPO_ARG[@]}" --color "$color" --description "$desc" >/dev/null 2>&1 \
      && echo "  updated  $name" || echo "  FAILED   $name"
  fi
}

# Area labels are examples: rename them to match the project. The gate does not
# read them — they exist so a human can filter the board.
echo "area labels (customise per project):"
label "area:backend"   "0e8a16" "Server, API, data layer"
label "area:frontend"  "5319e7" "UI and client code"
label "area:infra"     "fbca04" "Deploys, CI, hosting"
label "area:docs"      "0075ca" "Documentation"

# State labels ARE read by the gate. Do not rename these without updating
# next-task.py.
echo "state labels (required by the gate):"
label "in-progress"    "fef2c0" "Actively being worked — WIP limit is one"
label "blocked:human"  "e99695" "Needs a decision or credential only a human can supply"
label "needs-tests"    "d4c5f9" "Implementation exists, verification does not"

echo "priority labels (required by the gate):"
label "P0"             "b60205" "Drop everything — broken in production"
label "P1"             "d93f0b" "Next in line"
label "P2"             "fbca04" "Planned"
label "P3"             "c2e0c6" "Someday"

echo "done."
