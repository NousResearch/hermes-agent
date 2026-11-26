#!/usr/bin/env bash
set -euo pipefail

TITLE="${1:?usage: tot-start.sh <title> <rca|feature> <project-root> [parent-task-id]}"
TYPE="${2:?usage: tot-start.sh <title> <rca|feature> <project-root> [parent-task-id]}"
PROJECT_ROOT="${3:?usage: tot-start.sh <title> <rca|feature> <project-root> [parent-task-id]}"
PARENT="${4:-}"

BODY=$(printf 'Type: %s\nProject: %s' "$TYPE" "$PROJECT_ROOT")

PARENT_ARGS=()
if [[ -n "$PARENT" ]]; then
  PARENT_ARGS=(--parent "$PARENT")
fi

ROOT_OUT=$(hermes kanban create "ToT: $TITLE" --assignee default --body "$BODY" --triage "${PARENT_ARGS[@]}")
ROOT_ID=$(echo "$ROOT_OUT" | grep -oE 't_[a-f0-9]+')

echo "Root card: $ROOT_ID"

for role in clio hephaestus solon talaria; do
  hermes kanban create \
    "ToT debate — $role — $TITLE" \
    --assignee "$role" \
    --body "Participate in the Team-of-Thoughts debate on root card $ROOT_ID. Read all prior comments, then post one comment per round. No code." \
    --parent "$ROOT_ID" >/dev/null
  echo "Created $role debate card"
done
