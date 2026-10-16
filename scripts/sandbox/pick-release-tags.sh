#!/usr/bin/env bash
# Pick the release tags the install/update E2E should update FROM.
#
# Emits a JSON array of tag names on stdout, suitable for a GitHub Actions
# matrix (`fromJSON`). Choosing at runtime rather than hardcoding keeps the
# matrix honest as releases land: a pinned list silently stops covering the
# newest release the day after it ships, and pins the "oldest" forever even
# after it stops being a version anyone still runs.
#
# Selection: the newest tag, the oldest tag, and evenly spaced tags in between.
# Newest catches "did the last release break updating?", oldest is the longest
# upgrade jump anyone can still make, and the spread samples the migrations in
# between (config-schema bumps, venv layout changes, dependency floors).
#
# Usage:
#   scripts/sandbox/pick-release-tags.sh [--count N] [--repo DIR]
#
#   --count   how many tags to emit (default 5, minimum 1). Fewer tags than
#             requested emits all of them.
#   --repo    repository to read tags from (default: this checkout).
#
# Reads tags from the local checkout, so it needs one fetched with tags
# (actions/checkout with fetch-depth: 0, or `fetch-tags: true`). A shallow
# checkout has no tags and emits an empty list with a warning; the caller can
# then run the HEAD -> NEXT self-update leg without inventing an old baseline.
#
# Only stable and canonical canary release tags are considered. A canary is
# valid as an OLD baseline only when it points before the checkout under test.
# The repository may temporarily have no historical baseline (for example when
# a new line has only its current canary tag); callers decide whether that is a
# warning (scheduled/manual) or a release-gate failure.

set -euo pipefail

COUNT=5
# Default to the repository containing this script, resolved through its real
# path so a symlinked or copied script still reads the checkout it lives in
# rather than whatever repo the caller happens to be standing in.
REPO=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --count)
      [ "$#" -ge 2 ] || { echo 'error: --count needs a value' >&2; exit 1; }
      COUNT="$2"; shift 2 ;;
    --repo)
      [ "$#" -ge 2 ] || { echo 'error: --repo needs a value' >&2; exit 1; }
      REPO="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$COUNT" in
  ''|*[!0-9]*) echo "error: --count must be a positive integer: $COUNT" >&2; exit 1 ;;
esac
[ "$COUNT" -ge 1 ] || { echo 'error: --count must be at least 1' >&2; exit 1; }

# Resolve the script's own location through symlinks, then ask git which
# worktree that path belongs to. Deriving the repo from the script rather than
# from $PWD means a copied script cannot silently report a different checkout's
# tags, and --show-toplevel keeps it correct when invoked from a subdirectory.
if [ -z "$REPO" ]; then
  script_path="${BASH_SOURCE[0]}"
  if command -v readlink >/dev/null 2>&1; then
    script_path="$(readlink -f "$script_path" 2>/dev/null || printf '%s' "$script_path")"
  fi
  script_dir="$(cd "$(dirname "$script_path")" && pwd)"
  REPO="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$script_dir")"
fi

# sort -V orders v2026.4.8 before v2026.4.13 (numeric), which a plain
# lexicographic sort gets wrong.
mapfile -t tags < <(
  git -C "$REPO" tag --list 'v*' \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?(\+canary\.20[0-9]{6}T[0-9]{6}Z)?$' \
    | while IFS= read -r tag; do
        # An unresolvable tag (partial clone / missing object) must skip itself,
        # not kill this subshell: under errexit a failing command substitution in
        # an assignment would drop every later tag silently.
        if ! commit="$(git -C "$REPO" rev-list -1 "$tag^{commit}")"; then
          continue
        fi
        head="$(git -C "$REPO" rev-parse HEAD)"
        [ "$commit" != "$head" ] && printf '%s\n' "$tag"
      done | sort -V
)

total="${#tags[@]}"
if [ "$total" -eq 0 ]; then
  echo "warning: no historical release tags found in $REPO" >&2
  printf '[]\n'
  exit 0
fi

if [ "$total" -le "$COUNT" ]; then
  picked=("${tags[@]}")
elif [ "$COUNT" -eq 1 ]; then
  # One slot means the newest release; there is no span to spread across.
  picked=("${tags[$((total - 1))]}")
else
  # Evenly spaced indices across [0, total-1], endpoints included, so the
  # oldest and newest are always present and the rest are spread between them.
  picked=()
  for slot in $(seq 0 $((COUNT - 1))); do
    # Round to nearest rather than truncate, so the spacing does not bunch
    # toward the oldest end.
    index=$(( (slot * (total - 1) * 2 + (COUNT - 1)) / ((COUNT - 1) * 2) ))
    candidate="${tags[$index]}"
    # Guard against a duplicate if rounding lands twice on the same tag.
    case " ${picked[*]-} " in
      *" $candidate "*) continue ;;
    esac
    picked+=("$candidate")
  done
fi

printf '['
for i in "${!picked[@]}"; do
  [ "$i" -eq 0 ] || printf ','
  printf '"%s"' "${picked[$i]}"
done
printf ']\n'
