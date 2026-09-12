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
# Reads tags from the local checkout. A shallow clone may list tag names
# whose commits are missing; this script deepens the graph once and then
# keeps only tags that are ancestors of HEAD, so a rewritten release cannot
# enter the matrix (#100947).
#
# Only vYYYY.M.D[.N] release tags that are ancestors of HEAD are considered;
# the repo also carries backup/* and one-off tags that are not releases.

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
mapfile -t all_tags < <(
  git -C "$REPO" tag --list 'v*' \
    | grep -E '^v[0-9]{4}\.[0-9]+\.[0-9]+(\.[0-9]+)?$' \
    | sort -V
)

# Drop tags that are not ancestors of HEAD. After a history rewrite those
# names still exist but `hermes update` cannot reach them from current main.
collect_reachable() {
  tags=()
  if [ "${#all_tags[@]}" -gt 0 ]; then
    for tag in "${all_tags[@]}"; do
      if git -C "$REPO" merge-base --is-ancestor "${tag}^{}" HEAD 2>/dev/null; then
        tags+=("$tag")
      fi
    done
  fi
}

collect_reachable
listed="${#all_tags[@]}"
# Tags listed but not ancestral: usually a shallow checkout with
# fetch-tags (the GHA picker). Deepen once, then re-filter. Rewritten
# tags stay out even after a full graph is present.
if [ "$listed" -gt "${#tags[@]}" ]; then
  git -C "$REPO" fetch --unshallow --tags >/dev/null 2>&1 \
    || git -C "$REPO" fetch --deepen=2147483647 --tags >/dev/null 2>&1 \
    || true
  mapfile -t all_tags < <(
    git -C "$REPO" tag --list 'v*' \
      | grep -E '^v[0-9]{4}\.[0-9]+\.[0-9]+(\.[0-9]+)?$' \
      | sort -V
  )
  collect_reachable
fi

total="${#tags[@]}"
if [ "$total" -eq 0 ]; then
  echo "error: no reachable release tags found in $REPO" >&2
  echo '       Need tags that are ancestors of HEAD, with a complete commit' >&2
  echo '       graph (fetch-tags plus a non-shallow history).' >&2
  exit 1
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
