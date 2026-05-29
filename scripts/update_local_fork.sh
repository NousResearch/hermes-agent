#!/usr/bin/env bash
set -euo pipefail

# Rebase the user's local/fork-retained Hermes Agent customization branch on top
# of upstream NousResearch/hermes-agent main, then push the rebased branch to the
# user's fork. This keeps local-only changes reapplied after Hermes updates
# without ad hoc cherry-picking.

BRANCH="${1:-openclaw/local-main}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-origin}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
FORK_REMOTE="${FORK_REMOTE:-fork}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to update with uncommitted changes in $repo_root" >&2
  git status --short
  exit 2
fi

git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
git fetch "$FORK_REMOTE" "$BRANCH" || true

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
elif git show-ref --verify --quiet "refs/remotes/$FORK_REMOTE/$BRANCH"; then
  git switch -c "$BRANCH" --track "$FORK_REMOTE/$BRANCH"
else
  echo "Branch $BRANCH not found locally or on $FORK_REMOTE" >&2
  exit 3
fi

git rebase "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"

# Lightweight smoke check from the local checkout's virtualenv when present.
if [ -x ./venv/bin/python ]; then
  ./venv/bin/python - <<'PY'
from hermes_cli.main import main
print('hermes_cli import ok')
PY
fi

# Keep the fork branch updated so future machines/workers can recover it.
git push --force-with-lease "$FORK_REMOTE" "$BRANCH"

echo "Updated $BRANCH on top of $UPSTREAM_REMOTE/$UPSTREAM_BRANCH and pushed to $FORK_REMOTE/$BRANCH"
