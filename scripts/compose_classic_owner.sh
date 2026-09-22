#!/usr/bin/env bash
set -euo pipefail

BASE_SHA=485d5f6848c2598ca00fa7704e50e8ae66d0983a
FILES_SHA=804ce9124f6669568585a331aeb4ef1b5cce5e29
OUTPUT_SHA=0caf39c3c5e91eb5e0911b72fba120616d0b41e0
RUNTIME_FIRST_SHA=2fb90a347b5a0d7864367b221544dc058c6cf70c
RUNTIME_SECOND_SHA=927ac6476e13d2963d220dec82fc535558326574
RUNTIME_SHA=bd629002a4b8fea665404f233ddb4727082b9587
UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/NousResearch/hermes-agent.git}

usage() {
    printf 'usage: %s DESTINATION [OWNER_REV]\n' "$0" >&2
    printf 'Run this from the #104198 owner checkout; OWNER_REV defaults to HEAD.\n' >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
command -v git >/dev/null || { printf 'git is required\n' >&2; exit 2; }

OWNER_ROOT=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
OWNER_REV=${2:-HEAD}
OWNER_SHA=$(git -C "$OWNER_ROOT" rev-parse "${OWNER_REV}^{commit}")
DEST=$1
case "$DEST" in
    /*) ;;
    *) DEST="$PWD/$DEST" ;;
esac
[[ ! -e "$DEST" ]] || { printf 'destination already exists: %s\n' "$DEST" >&2; exit 2; }

pin() {
    local ref=$1 expected=$2 label=$3 actual
    actual=$(git -C "$DEST" rev-parse "${ref}^{commit}")
    [[ "$actual" == "$expected" ]] || {
        printf '%s moved: expected %s, got %s\n' "$label" "$expected" "$actual" >&2
        exit 1
    }
}

git init -q "$DEST"
git -C "$DEST" remote add upstream "$UPSTREAM_URL"
git -C "$DEST" fetch --no-tags upstream \
    "refs/heads/feat/unified-gateway-runtime:refs/recipe/base" \
    "refs/pull/98072/head:refs/recipe/files" \
    "refs/pull/99159/head:refs/recipe/output" \
    "refs/pull/111216/head:refs/recipe/runtime"

pin refs/recipe/base "$BASE_SHA" '#106742 base'
pin refs/recipe/files "$FILES_SHA" '#98072'
pin refs/recipe/output "$OUTPUT_SHA" '#99159'
pin refs/recipe/runtime "$RUNTIME_SHA" '#111216'
git -C "$DEST" merge-base --is-ancestor "$BASE_SHA" "$FILES_SHA"
git -C "$DEST" merge-base --is-ancestor "$BASE_SHA" "$OUTPUT_SHA"
[[ $(git -C "$DEST" rev-parse "${RUNTIME_SECOND_SHA}^") == "$RUNTIME_FIRST_SHA" ]]
[[ $(git -C "$DEST" rev-parse "${RUNTIME_SHA}^") == "$RUNTIME_SECOND_SHA" ]]

git -C "$DEST" fetch --no-tags "$OWNER_ROOT" \
    "${OWNER_SHA}:refs/recipe/owner"
git -C "$DEST" merge-base --is-ancestor "$BASE_SHA" refs/recipe/owner
[[ -z $(git -C "$DEST" rev-list --merges "$BASE_SHA..refs/recipe/owner") ]] || {
    printf 'owner interval unexpectedly contains a merge commit\n' >&2
    exit 1
}

git -C "$DEST" config user.name 'Classic composition recipe'
git -C "$DEST" config user.email 'classic-composition@example.invalid'
git -C "$DEST" checkout -q -b recipe/classic-104198 "$OUTPUT_SHA"

# #99159 already contains the integrated #98072 source. Record the exact
# public Files head as ancestry without replaying its overlapping delta.
output_tree=$(git -C "$DEST" rev-parse 'HEAD^{tree}')
git -C "$DEST" merge --no-ff -s ours "$FILES_SHA" \
    -m 'build: bind #99159 to the exact #98072 dependency'
[[ $(git -C "$DEST" rev-parse 'HEAD^{tree}') == "$output_tree" ]]

# Apply the three #111216 owner changes exactly once, preserving their public
# authors, author dates, messages, and source trailers.
git -C "$DEST" cherry-pick \
    "$RUNTIME_FIRST_SHA" "$RUNTIME_SECOND_SHA" "$RUNTIME_SHA"

# Apply only the #104198 review commits from its declared #106742 base.
mapfile -t owner_commits < <(
    git -C "$DEST" rev-list --reverse "$BASE_SHA..refs/recipe/owner"
)
[[ ${#owner_commits[@]} -gt 0 ]] || {
    printf 'owner interval is empty\n' >&2
    exit 1
}
git -C "$DEST" cherry-pick "${owner_commits[@]}"

owner_patch=docs/development/patches/classic-owner-dependent-integration.patch
git -C "$DEST" apply --unidiff-zero --check "$owner_patch"
git -C "$DEST" apply --unidiff-zero --index "$owner_patch"
git -C "$DEST" diff --cached --check
git -C "$DEST" commit -m 'build: apply #104198 dependent owner hunks'

[[ -z $(git -C "$DEST" status --porcelain=v1 --untracked-files=all) ]]
printf 'base=%s\nfiles=%s\noutput=%s\nruntime=%s\nowner=%s\nhead=%s\ntree=%s\n' \
    "$BASE_SHA" "$FILES_SHA" "$OUTPUT_SHA" "$RUNTIME_SHA" "$OWNER_SHA" \
    "$(git -C "$DEST" rev-parse HEAD)" \
    "$(git -C "$DEST" rev-parse 'HEAD^{tree}')"
