#!/usr/bin/env bash
set -euo pipefail

force=0
if [[ "${1:-}" == "--force" ]]; then
  force=1
fi

if ! git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "error: not inside a git repository" >&2
  exit 1
fi

hook_dir="$git_root/.git/hooks"
hook_path="$hook_dir/pre-commit"
start_marker="# >>> hermes self-knowledge hook >>>"
end_marker="# <<< hermes self-knowledge hook <<<"

block="${start_marker}
# Auto-refresh repo-grounded Hermes self-knowledge before commits.
if command -v hermes >/dev/null 2>&1; then
  hermes self-knowledge --refresh
else
  python -m hermes_cli.main self-knowledge --refresh
fi
if ! git diff --quiet -- context/self/hermes-agent.md; then
  git add context/self/hermes-agent.md
fi
${end_marker}"

mkdir -p "$hook_dir"

if [[ -f "$hook_path" ]]; then
  if grep -Fq "$start_marker" "$hook_path"; then
    tmp="$(mktemp)"
    awk -v start="$start_marker" -v end="$end_marker" '
      $0 == start {skip=1; next}
      $0 == end {skip=0; next}
      !skip {print}
    ' "$hook_path" > "$tmp"
    {
      cat "$tmp"
      printf "\n%s\n" "$block"
    } > "$hook_path"
    rm -f "$tmp"
  elif [[ "$force" -eq 1 ]]; then
    tmp="$(mktemp)"
    cat "$hook_path" > "$tmp"
    {
      cat "$tmp"
      printf "\n%s\n" "$block"
    } > "$hook_path"
    rm -f "$tmp"
  else
    echo "error: foreign pre-commit hook exists; rerun with --force to append Hermes block" >&2
    exit 2
  fi
else
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
    printf "%s\n" "$block"
  } > "$hook_path"
fi

chmod +x "$hook_path"
echo "installed Hermes self-knowledge pre-commit hook: $hook_path"
