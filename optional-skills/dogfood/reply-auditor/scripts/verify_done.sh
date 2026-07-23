#!/usr/bin/env bash
# verify_done.sh — audit an assistant message for unsourced "done" claims.
#
# Usage:
#   bash verify_done.sh <file>          # audit a file containing draft reply
#   bash verify_done.sh -               # audit stdin
#   echo "..." | bash verify_done.sh -  # pipe through
#
# Environment:
#   VERIFY_DONE_WINDOW=N    override the +/- N line context window (default 5)
#
# Exits 0 if no unsourced claims, 1 if any found. Prints line numbers.
#
# This is the operational enforcement of Iron Law 1 (Show, don't claim).
#
# Provenance:
#   Originally shipped in hermes-iron-laws v0.1.0 as a log-scraper.
#   That design produced false positives on every INFO line that
#   contained the word "done" or "ready". Rewritten in v0.2.0 to
#   audit only agent-voice messages. Promoted to standalone skill
#   in v1.0.0 here.
#
# Companion: https://github.com/shootzjmr/hermes-iron-laws

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash verify_done.sh <message-file>     # audit a draft reply file
  bash verify_done.sh -                  # audit stdin (interactive)
  echo "..." | bash verify_done.sh -     # pipe a reply

Env:
  VERIFY_DONE_WINDOW=N    line context window (default 5)

Exits:
  0  no unsourced claims (message is safe to send)
  1  unsourced claims found (fix or rephrase)
  2  usage error
USAGE
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

WINDOW="${VERIFY_DONE_WINDOW:-5}"

case "$1" in
    -h|--help) usage; exit 0 ;;
    -)        msg=$(cat) ;;
    *)        if [[ ! -f "$1" ]]; then
                  echo "ERR: no such file: $1" >&2
                  exit 2
              fi
              msg=$(cat "$1") ;;
esac

# Claims phrased in 1st person, present-tense, declarative.
CLAIM_RE='(^|[^a-z])(I |I'\''ve |I have |we |we'\''ve |ya (esta|quedo|quedó)|listo|done|finished|complete|completed|fixed|working|set up|configured|ready to go|all set)(\b|[^a-z])'

# Proof markers — a claim is grounded if any of these appears within
# +/- WINDOW lines.
PROOF_RE='(```|`[A-Za-z][^`]*`|\bexit[_ ]?code[: ]?[ \t]*[0-9]+|\$ (sudo |cd |ls |cat |grep |ssh |curl |docker |systemctl |apt |git |echo |bash |python|python3 |mkdir |cp |mv |chmod |chown |scp |rsync )|HTTP/[0-9.]+ [0-9]+|ENOENT|Permission denied|Connection refused|^\s*✓|^\s*✗|^\s*❌|stdout|stderr|^\s*root@|HTTP %{http_code}|Status: [0-9])'

found=0
idx=0

# Build line array for windowed access
mapfile -t lines <<< "$msg"
nlines=${#lines[@]}

while [[ "$idx" -lt "$nlines" ]]; do
    line="${lines[$idx]}"
    idx=$((idx + 1))

    if ! echo "$line" | grep -qiE "$CLAIM_RE"; then
        continue
    fi

    # Window: idx (1-based) +/- WINDOW
    lo=$((idx > WINDOW ? idx - WINDOW : 1))
    hi=$((idx + WINDOW < nlines ? idx + WINDOW : nlines))

    window=""
    for ((i = lo - 1; i < hi && i < nlines; i++)); do
        window+="${lines[$i]}"$'\n'
    done

    if echo "$window" | grep -qE "$PROOF_RE"; then
        continue
    fi

    echo "[UNSOURCED CLAIM line $idx]: $line"
    found=1
done

if [[ "$found" -ne 0 ]]; then
    echo
    echo "→ Found $found claim(s) without paired proof within +/- $WINDOW lines."
    echo "  Either add the actual output nearby or rephrase without the claim."
    echo "  For longer messages, set VERIFY_DONE_WINDOW=10 to widen the search."
    exit 1
fi

echo "✓ No unsourced completion claims — safe to send"
exit 0
