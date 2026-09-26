#!/usr/bin/env bash
# Proof harness for the disk-guard "top reclaimable" ranking.
#
# CLASS proven: the alert of 2026-09-21T02:01 listed six /private/tmp files of
# ONE du block each (wake.out = 30 bytes, hb.bak = 18KB) as the top reclaimable
# paths. The scan covered the wrong roots (kanban workspaces + /private/tmp only)
# and ranked by du blocks, so an operator following it would delete nothing.
# An alert whose remediation list is noise is worse than no alert.
#
# Requirements encoded here:
#   R1 entries are ordered bytes-descending
#   R2 sub-100MB trivia never appears
#   R3 the real scratch roots (worktrees, pnpm store, hermes logs/sessions) are
#      covered — proven by planting a large dir under a real root and requiring
#      it to surface
#   R4 a path with a LIVE process in it is never listed as reclaimable
#
# Run: bash ~/.hermes/scripts/tests/prove_disk_guard_ranking.sh
set -uo pipefail

GUARD="$HOME/.hermes/scripts/disk-guard.sh"
fail=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1 ($3)"; else echo "FAIL $1: expected '$2' got '$3'"; fail=1; fi; }

SANDBOX=$(mktemp -d "$HOME/.hermes/logs/ranking-proof.XXXXXX")
cleanup() { kill "${live_pid:-}" 2>/dev/null; rm -rf "$SANDBOX"; }
trap cleanup EXIT

# Plant a big dir under a REAL scratch root, plus trivia that must not rank.
BIG="$HOME/workspace/AgentPod-worktrees/zzz-ranking-proof-big"
SMALL="$HOME/workspace/AgentPod-worktrees/zzz-ranking-proof-small"
INUSE="$HOME/workspace/AgentPod-worktrees/zzz-ranking-proof-inuse"
mkdir -p "$BIG" "$SMALL" "$INUSE"
cleanup() { kill "${live_pid:-}" 2>/dev/null; rm -rf "$SANDBOX" "$BIG" "$SMALL" "$INUSE"; }
trap cleanup EXIT

# Real bytes, not sparse: `mkfile -n` / `dd` from /dev/zero can produce a
# sparse or compressed file that du reports as 1 block, which would make R3
# fail for a reason that has nothing to do with the ranking. /dev/urandom is
# incompressible, so the size du sees is the size on disk.
dd if=/dev/urandom of="$BIG/blob" bs=1m count=300 >/dev/null 2>&1
dd if=/dev/urandom of="$INUSE/blob" bs=1m count=220 >/dev/null 2>&1
printf 'tiny\n' > "$SMALL/note.txt"

# A live process holding the path as its cwd (what path_in_use derives on).
# It must outlive the whole du scan across every scratch root, which takes
# minutes on a full disk — a 120s holder dies mid-scan and R4 then fails for
# a harness reason rather than a guard reason.
( cd "$INUSE" && exec sleep 1800 ) &
live_pid=$!
sleep 1
cleanup() { kill "${live_pid:-}" 2>/dev/null; rm -rf "$SANDBOX" "$BIG" "$SMALL" "$INUSE"; }
trap cleanup EXIT

# Fail loudly rather than vacuously if the holder is not actually live.
kill -0 "$live_pid" 2>/dev/null || { echo "FAIL harness: in-use holder never started"; exit 1; }

out=$(bash "$GUARD" --top-reclaimable 2>/dev/null)

if [ -z "$out" ]; then
  echo "FAIL harness: --top-reclaimable produced no output"; exit 1
fi
echo "--- ranking output ---"; printf '%s\n' "$out"; echo "----------------------"

# Anti-vacuity: R1/R2/R4 are all trivially satisfiable by an empty or log-only
# list, which is exactly how a broken ranking would sneak through. Every line
# must be "<integer MB>\t<absolute path>" and there must be at least two.
lines=$(printf '%s\n' "$out" | wc -l | tr -d ' ')
shaped=$(printf '%s\n' "$out" | grep -cE '^[0-9]+[[:space:]]+/')
check "R0 output is a real ranking (>=2 well-formed rows)" "yes" \
      "$([ "$lines" -ge 2 ] && [ "$shaped" = "$lines" ] && echo yes || echo no)"

# R1: bytes-descending. First field is MB.
sizes=$(printf '%s\n' "$out" | awk '{print $1}')
sorted=$(printf '%s\n' "$sizes" | sort -rn)
check "R1 entries are bytes-descending" "$sorted" "$sizes"

# R2: nothing under 100MB
tiny=$(printf '%s\n' "$out" | awk '$1 < 100' | wc -l | tr -d ' ')
check "R2 no sub-100MB trivia listed" "0" "$tiny"

# R3: a large dir under a real scratch root surfaces
hit=$(printf '%s\n' "$out" | grep -c "zzz-ranking-proof-big")
check "R3 large dir under a real scratch root is listed" "1" "$hit"

# R4: an in-use path is never offered for deletion
inuse_hit=$(printf '%s\n' "$out" | grep -c "zzz-ranking-proof-inuse")
check "R4 in-use path is not listed" "0" "$inuse_hit"

exit "$fail"
