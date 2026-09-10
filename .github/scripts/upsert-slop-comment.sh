#!/usr/bin/env bash
# Creates or updates the single anti-slop report comment on a PR.
# Inputs: $PR (number), $LINT_EXIT ("0" when the ratchet found nothing net-new, "1"
# otherwise — read from the artifact's slop-exit.txt, since the advisory
# check itself always passes), and slop-report.trimmed.txt from its artifact.
set -euo pipefail

report=slop-report.trimmed.txt
# Slop Lint can conclude failure before oxlint writes anything, leaving the
# artifact empty. Say so, rather than posting an empty fence or dying below.
[ -s "$report" ] || printf 'The lint run produced no output. See the Slop Lint job log.\n' > "$report"

marker='<!-- slop-diff-report -->'
{
  echo "$marker"
  if [ "${LINT_EXIT}" = "0" ]; then
    echo "**Anti-slop lint: no net-new findings** in the JS/TS files this PR touches (baseline: \`tools/oxlint/slop-baseline.json\`). Advisory only; never blocks a merge."
  else
    echo "**Anti-slop lint found net-new findings** in the JS/TS files this PR touches. Advisory only; never blocks a merge."
    echo
    # The report quotes attacker-influenced source text, so the fence must
    # outrun any backtick sequence inside it: one longer than the longest.
    # `|| true` is load-bearing: grep exits 1 on a report holding no backtick
    # at all (a bare git error out of slop-diff.sh, say), and pipefail would
    # then kill this script inside the redirect, emitting nothing anywhere.
    longest=$(grep -o '`\+' "$report" | awk '{ if (length($0) > m) m = length($0) } END { print m + 0 }' || true)
    fence_len=$(( longest >= 3 ? longest + 1 : 3 ))
    fence=$(printf '%*s' "$fence_len" '' | tr ' ' '`')
    echo "$fence"
    cat "$report"
    echo "$fence"
  fi
  echo
  echo '<details><summary>Run it locally</summary>'
  echo
  echo 'From the repo root:'
  echo
  echo '```bash'
  echo 'npm run lint:slop:diff              # lint only JS/TS files changed vs origin/main'
  echo 'SLOP_BASE=origin/some-branch \'
  echo '  npm run lint:slop:diff            # different diff base'
  echo 'npm run lint:slop                   # the whole tree'
  echo 'npm run lint:slop:baseline          # after fixing old hits, lower the baseline (it never goes up without --allow-increase)'
  echo '```'
  echo
  echo 'The rules live in `tools/oxlint/anti-slop/`; the config is `oxlint.config.ts` at the repo root. A finding means the pattern needs a real fix or, for type assertions, a `// SAFETY:` comment stating the checked invariant.'
  echo '</details>'
} > comment-body.md

# No `| head -1`: head closing the pipe early makes gh die of SIGPIPE, which
# pipefail turns into a script abort. Take the first line after the fact so a
# genuine gh failure still aborts and a long list still costs nothing.
existing=$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR}/comments" --paginate \
  --jq ".[] | select(.body | startswith(\"$marker\")) | .id")
existing=${existing%%$'\n'*}

if [ -n "$existing" ]; then
  gh api -X PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${existing}" -F body=@comment-body.md > /dev/null
else
  gh api "repos/${GITHUB_REPOSITORY}/issues/${PR}/comments" -F body=@comment-body.md > /dev/null
fi
echo "comment upserted (existing: ${existing:-none})"
