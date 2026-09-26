#!/usr/bin/env bash
# disk-guard.sh — fail loudly below a free-space floor, and reclaim space from
# agent scratch that is provably dead (terminal kanban tasks, merged worktrees,
# stale /private/tmp clones).
#
# CLASS this fixes: an ENOSPC host makes every watchdog unreliable while it
# still reports green. Each concurrent agent worker materialises a du-apparent
# clone (kanban scratch workspace, /private/tmp clone, or git worktree) and
# nothing reaps them when the task ends.
#
# Everything checked here is DERIVED from the source of truth at runtime:
#   - live/dead task set  -> sqlite query against ~/.hermes/kanban.db
#   - merged branches     -> git merge-base --is-ancestor against origin/main
#   - in-use directories  -> running process table
# There is no hand-maintained list of paths to keep in sync.
#
# Usage:
#   disk-guard.sh            # check only; exit 1 if below floor
#   disk-guard.sh --reclaim  # check, reclaim dead scratch, re-check
#
# Exit codes: 0 = at or above floor, 1 = below floor after reclaim.

set -uo pipefail

# TWO NUMBERS, TWO JOBS. Conflating them is what produced both failure modes
# this guard has actually shown: a 10Gi floor paged 67 times in 13h with no
# failed write (alert fatigue), and a 0.5Gi floor let the host cross from
# healthy to ENOSPC inside one 900s tick (no warning at all).
#
#   RECLAIM_TARGET_GI  how much free space we try to HOLD. Enforced by evicting
#                      more sanctioned scratch, SILENTLY. Never pages.
#   FLOOR_GI           the only thing that pages. Owner-pinned at 0.5Gi
#                      (Den, 2026-09-22) — page only where writes actually fail.
#
# The 10Gi target is the capacity derivation in disk-guard-cron.sh: 2.0GB worst
# case for 3 concurrent workers on a cold pnpm store + 8GB headroom for the
# non-worker writers on this volume (state.db, postgres, gateway logs).
FLOOR_GI="${DISK_GUARD_FLOOR_GI:-0.5}"
RECLAIM_TARGET_GI="${DISK_GUARD_RECLAIM_TARGET_GI:-10}"
KANBAN_DB="${HERMES_KANBAN_DB:-$HOME/.hermes/kanban.db}"
WORKSPACES="$HOME/.hermes/kanban/workspaces"
TMPDIRS="/private/tmp"
TMP_AGE_SECONDS="${DISK_GUARD_TMP_AGE:-7200}"
RECLAIM=0
[ "${1:-}" = "--reclaim" ] && RECLAIM=1

log() { printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"; }

free_mb() { df -m /System/Volumes/Data 2>/dev/null | awk 'NR==2{print $4}'; }
# FLOAT GiB. An integer `free_mb/1024` floored 1708MiB to "1Gi" and made the
# 0.5 comparison a lie at the only magnitude that matters.
free_gi() { awk -v m="$(free_mb)" 'BEGIN{printf "%.2f", m/1024}'; }
# Float-exact comparison: `[ 0.60 -lt 0.5 ]` is a bash INTEGER error, which
# silently skips the branch and reports green on a dying host.
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 < b+0)}'; }

# --- derived fact 1: which task ids are dead (terminal or absent from the DB)
dead_task() { # $1 = task id -> 0 if dead/unknown
  [ -f "$KANBAN_DB" ] || return 1
  local st
  st=$(sqlite3 -noheader "$KANBAN_DB" \
        "select status from tasks where id='$1';" 2>/dev/null)
  case "$st" in
    ''|done|archived|cancelled) return 0 ;;
    *) return 1 ;;
  esac
}

# --- derived fact 2: is any running process sitting in this path
# Deliberately derives from lsof's OUTPUT, not its exit code: on macOS
# `lsof +D <dir>` exits 1 even when it prints a matching holder (measured
# 2026-09-21: a `sleep` whose cwd was the dir was listed, rc=1). Trusting rc
# made the guard offer a live worker's directory for deletion. The argv check
# alone is also insufficient — a process that cd'd into the dir has a bare
# argv ("sleep 900") that never mentions the path.
path_in_use() { # $1 = abs path
  # SELF-MATCH BUG (found 2026-09-25): this used
  #   ps -Ao args= | grep -Fq -- "$1"
  # and `grep`'s OWN argv contains "$1", so ps listed it and the grep matched
  # itself. path_in_use therefore returned "in use" for EVERY path ever passed:
  # every reclaim class that consults it deleted nothing, and top_reclaimable
  # filtered out every candidate — which is exactly the "No single reclaimable
  # path over 100MB" line the operator kept reading on a host that was filling
  # up. Passing the needle through the ENVIRONMENT keeps it out of argv, so the
  # scan can no longer see itself.
  # (Exported, not a command-prefix assignment: a prefix binds only to the
  # first command of the pipeline -- `ps` -- and awk would see an empty needle,
  # matching every line and reinstating the same always-in-use bug.)
  export DG_NEEDLE="$1"
  ps -Ao args= 2>/dev/null \
    | awk 'index($0, ENVIRON["DG_NEEDLE"]) { found = 1 } END { exit !found }' \
    && { unset DG_NEEDLE; return 0; }
  unset DG_NEEDLE
  [ -n "$(lsof +D "$1" 2>/dev/null | tail -n +2)" ] && return 0
  [ -n "$(lsof -- "$1" 2>/dev/null | tail -n +2)" ] && return 0
  return 1
}

# --- derived fact 3: the scratch roots worth ranking.
# DERIVED from the live filesystem, never a hand-kept list: every git repo's
# worktree parent, the kanban workspaces, the agent caches and the hermes log/
# session dirs. The 2026-09-21 alert ranked /private/tmp files of one du block
# each and would have sent an operator to delete 30 bytes — the roots were
# wrong and the sort was by du blocks.
RANK_MIN_MB="${DISK_GUARD_RANK_MIN_MB:-100}"

scratch_roots() {
  {
    echo "$WORKSPACES"
    echo "$HOME/workspace/AgentPod-worktrees"
    ls -d "$HOME"/workspace/*/.worktrees 2>/dev/null
    echo "$HOME/Library/pnpm/store"
    echo "$HOME/.cache"
    echo "$HOME/.copilot/session-state"
    echo "$HOME/.paperclip/scratch"
    echo "$HOME/.paperclip/cli/installs/git"
    echo "$HOME/.hermes/logs"
    echo "$HOME/.hermes/sessions"
  } | while read -r r; do [ -d "$r" ] && echo "$r"; done
}

# Rank first-level entries of every scratch root by APPARENT SIZE, descending,
# dropping anything below the floor of interest and anything a live process is
# sitting in. Output: "<MB>\t<path>" lines, largest first.
top_reclaimable() {
  local n="${1:-8}"
  scratch_roots | while read -r root; do
    du -sxm "$root"/* 2>/dev/null
  done | sort -rn | awk -v min="$RANK_MIN_MB" '$1 >= min' | \
  while read -r mb path; do
    path_in_use "$path" && continue
    printf '%s\t%s\n' "$mb" "$path"
  done | head -"$n"
}

if [ "${1:-}" = "--top-reclaimable" ]; then
  top_reclaimable "${2:-8}"
  exit 0
fi

# The threshold is ONE decision with ONE owner. The cron wrapper renders a "no
# path over NMB" message and used to carry its own `:-100` default literal; the
# two could drift and the operator would read a threshold the ranking never
# applied. Wrapper asks, guard answers.
if [ "${1:-}" = "--rank-min-mb" ]; then
  printf '%s\n' "$RANK_MIN_MB"
  exit 0
fi

reclaim_workspaces() {
  local freed=0 n=0 id sz
  [ -d "$WORKSPACES" ] || return 0
  for w in "$WORKSPACES"/*/; do
    [ -d "$w" ] || continue
    id=$(basename "$w")
    dead_task "$id" || continue
    path_in_use "${w%/}" && { log "  skip (in use) $w"; continue; }
    sz=$(du -sxm "$w" 2>/dev/null | cut -f1)
    rm -rf "$w" && { freed=$((freed + ${sz:-0})); n=$((n + 1)); }
  done
  log "workspaces: removed $n dead scratch dirs, ${freed}MB"
}

reclaim_merged_worktrees() {
  # Superseded by `hermes kanban reclaim`. Kept as a fallback ONLY for when the
  # hermes venv is unavailable. Its eligibility test (ancestor-of-origin/main)
  # is wrong for a squash-merging repo: a squash-merged branch tip is never an
  # ancestor of main, so this loop removed 0 worktrees over its entire log while
  # reporting success. Do not extend it; extend hermes_cli/kanban_reclaim.py.
  local repo="$1" removed=0 freed=0 p b sha sz id
  [ -d "$repo/.git" ] || [ -f "$repo/.git" ] || return 0
  git -C "$repo" fetch origin main -q 2>/dev/null
  git -C "$repo" worktree prune 2>/dev/null
  git -C "$repo" worktree list --porcelain 2>/dev/null \
    | awk '/^worktree /{w=$2} /^branch /{print w" "$2}' \
    | while read -r p b; do
        [ "$p" = "$repo" ] && continue
        [ -d "$p" ] || continue
        sha=$(git -C "$repo" rev-parse "$b" 2>/dev/null) || continue
        # only reclaim branches fully contained in origin/main
        git -C "$repo" merge-base --is-ancestor "$sha" origin/main 2>/dev/null || continue
        # a worktree named after a live task is off limits
        id=$(basename "$p" | grep -oE 't_[0-9a-f]{8}' | head -1)
        if [ -n "$id" ] && ! dead_task "$id"; then
          log "  skip (live task $id) $p"; continue
        fi
        path_in_use "$p" && { log "  skip (in use) $p"; continue; }
        sz=$(du -sxm "$p" 2>/dev/null | cut -f1)
        git -C "$repo" worktree remove --force "$p" 2>/dev/null || rm -rf "$p"
        log "  removed merged worktree ${sz}MB $p"
      done
  git -C "$repo" worktree prune 2>/dev/null
}

reclaim_tmp() {
  local cutoff freed=0 n=0 m sz e
  cutoff=$(( $(date +%s) - TMP_AGE_SECONDS ))
  for e in "$TMPDIRS"/*; do
    [ -e "$e" ] || continue
    case "$(basename "$e")" in com.apple.*|*.sock) continue ;; esac
    m=$(stat -f %m "$e" 2>/dev/null) || continue
    [ "$m" -lt "$cutoff" ] || continue
    path_in_use "$e" && continue
    sz=$(du -sxm "$e" 2>/dev/null | cut -f1)
    rm -rf "$e" 2>/dev/null && { freed=$((freed + ${sz:-0})); n=$((n + 1)); }
  done
  log "tmp: removed $n stale entries, ${freed}MB"
}

# --- safe classes: caches and agent session-state that regenerate on demand.
# Each entry is DERIVED (a glob/age query against the live filesystem), and
# nothing here is a source of truth for any running job: caches refill, a
# copilot session older than the retention window is never resumed, and the
# pnpm store is rebuilt from the lockfile. Anything whose loss would cost a
# human decision (VM images, repos, databases) is deliberately NOT here.
SAFE_AGE_DAYS="${DISK_GUARD_SAFE_AGE_DAYS:-7}"

reclaim_safe_caches() {
  local before after n=0 d
  before=$(free_mb)

  for d in "$HOME/.cache/uv" "$HOME/Library/Caches/pip" \
           "$HOME/Library/Caches/ms-playwright" "$HOME/Library/Caches/Homebrew"; do
    [ -d "$d" ] || continue
    path_in_use "$d" && { log "  skip (in use) $d"; continue; }
    rm -rf "$d"/* 2>/dev/null && n=$((n + 1))
  done

  # copilot session-state older than the retention window
  if [ -d "$HOME/.copilot/session-state" ]; then
    find "$HOME/.copilot/session-state" -mindepth 1 -maxdepth 1 \
         -mtime "+${SAFE_AGE_DAYS}" -exec rm -rf {} + 2>/dev/null
  fi

  # hermes worker logs beyond rotation (rotated copies only, never the live file)
  find "$HOME/.hermes/logs" "$HOME/.hermes/profiles" -type f \
       \( -name '*.log.[0-9]*' -o -name '*.log.gz' \) \
       -mtime "+${SAFE_AGE_DAYS}" -delete 2>/dev/null

  if command -v pnpm >/dev/null 2>&1; then
    pnpm store prune >/dev/null 2>&1
  fi

  after=$(free_mb)
  log "safe caches: pruned $n cache dirs + aged session-state/logs, $(( after - before ))MB"
}

# --- target enforcement: silent, and strictly more of the SAME sanctioned
# classes. It never introduces a new category of deletion (that would make the
# quiet path riskier than the loud one) and it never pages — being under the
# capacity target is not an incident, it is a reason to work harder quietly.
#
# The per-user temp root is DERIVED from the running shell's TMPDIR, not the
# hardcoded /var/folders/<hash> path of one machine. It held 4.0GB of >1d-old
# agent scratch on 2026-09-25 and nothing was reaping it: reclaim_tmp only ever
# looked at /private/tmp.
user_tmp_root() { printf '%s' "${TMPDIR:-/tmp}" | sed 's:/$::'; }

reclaim_user_tmp() { # $1 = age in days
  local root freed_before freed_after
  root=$(user_tmp_root)
  case "$root" in /tmp|/private/tmp|''|/) return 0 ;; esac
  [ -d "$root" ] || return 0
  freed_before=$(free_mb)
  find "$root" -mindepth 1 -maxdepth 1 -mtime "+$1" \
       ! -name 'com.apple.*' ! -name '*.sock' \
       -exec rm -rf {} + 2>/dev/null
  freed_after=$(free_mb)
  log "user tmp ($root, >${1}d): $(( freed_after - freed_before ))MB"
}

reclaim_to_target() {
  local now
  now=$(free_gi)
  lt "$now" "$RECLAIM_TARGET_GI" || return 0
  log "below_target: ${now}Gi < ${RECLAIM_TARGET_GI}Gi — escalating quietly"
  reclaim_user_tmp 1
  now=$(free_gi)
  lt "$now" "$RECLAIM_TARGET_GI" || return 0
  TMP_AGE_SECONDS=3600 reclaim_tmp
  reclaim_user_tmp 0
  now=$(free_gi)
  lt "$now" "$RECLAIM_TARGET_GI" && \
    log "below_target: still ${now}Gi after escalation; sanctioned scratch is exhausted (silent by design — only FLOOR_GI pages)"
  return 0
}

# --- agent review clones in $HOME. The 2026-09-24 ENOSPC incident: 8 agent
# clones directly under $HOME held 4.3Gi of node_modules while the guard
# reported "the remaining consumers are NOT agent scratch". The class is
# "a git checkout an agent made outside the sanctioned scratch roots".
#
# SELECTION (the same predicate prove_disk_guard_node_modules_scope.sh asserts):
#   include  $HOME/<dir>/ that is a git repo, or a parent of git repos
#   exclude  ~/workspace (human checkouts), ~/Library, and every dotdir —
#            which is what keeps ~/.local/lib/node_modules (the global npm
#            prefix holding the pi/opencode CLIs) and ~/.hermes/hermes-agent
#            (the running agent itself) out of it.
# A first cut without those exclusions would have deleted the agent's own
# runtime, so the narrowing is load-bearing, not tidiness.
reclaim_home_node_modules() {
  local before after n e d
  before=$(free_mb); n=0
  for e in "$HOME"/[!.]*/; do
    case "$e" in "$HOME/workspace/"|"$HOME/Library/") continue ;; esac
    [ -d "${e}.git" ] || [ -f "${e}.git" ] || \
      { ls -d "$e"*/.git >/dev/null 2>&1 || continue; }
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      path_in_use "$d" && continue
      rm -rf "$d" 2>/dev/null && n=$((n + 1))
    done < <(find "$e" -maxdepth 3 -type d -name node_modules -prune -mtime +0 -print 2>/dev/null)
  done
  after=$(free_mb)
  log "node_modules: removed $n agent-clone dirs in \$HOME, $(( after - before ))MB"
}

# --- pytest scratch roots. MEASURED 2026-09-25: the host fell 25Gi -> 16Gi in
# 25 minutes while this card was open, and the consumer was not any sanctioned
# scratch root. Every agent pytest run that builds a throwaway hermes-home
# provisions its OWN 1.8GB Chromium under $TMPDIR/pytest-of-<user>/pytest-N;
# pytest's own retention keeps the last 3 numbered roots and never accounts for
# size, so a burst of test runs adds GB/minute and nothing reaps it. reclaim_tmp
# did not see it (it only ever looked at /private/tmp) and it is not a git
# checkout, so the node_modules sweep did not either.
#
# Liveness, not age, is the predicate: these roots are minutes old by
# construction, so an age rule either deletes a running test's fixture or never
# fires. Keep `pytest-current` (the symlink target of the run in progress) and
# anything a live process holds open; evict the rest.
reclaim_pytest_roots() {
  local root base before after n d
  root=$(user_tmp_root)
  base="$root/pytest-of-$(id -un)"
  [ -d "$base" ] || return 0
  before=$(free_mb); n=0
  for d in "$base"/pytest-*; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in pytest-current) continue ;; esac
    [ "$d" -ef "$base/pytest-current" ] && continue
    path_in_use "$d" && continue
    rm -rf "$d" 2>/dev/null && n=$((n + 1))
  done
  after=$(free_mb)
  log "pytest roots: removed $n idle roots under $base, $(( after - before ))MB"
}

before=$(free_gi)

# Library seam: `DISK_GUARD_LIB=1 source disk-guard.sh` defines the reclaim
# functions and stops, so a proof harness can drive ONE class against its own
# fixture. Without it a harness can only re-implement the predicate in its own
# words, which is how prove_disk_guard_node_modules_scope.sh ended up asserting
# a copy of the selection while the real sweep was missing from the script
# entirely for a full day.
[ "${DISK_GUARD_LIB:-0}" = 1 ] && return 0 2>/dev/null

log "free=${before}Gi floor=${FLOOR_GI}Gi target=${RECLAIM_TARGET_GI}Gi"


if [ "$RECLAIM" = 1 ]; then
  reclaim_workspaces
  reclaim_safe_caches
  # Done-card worktrees: one implementation, in hermes, shared with `kanban gc`.
  # The shell copy below is a fallback only (see reclaim_merged_worktrees).
  HERMES_BIN="${DISK_GUARD_HERMES_BIN:-$HOME/.hermes/hermes-agent/venv/bin/hermes}"
  if [ -x "$HERMES_BIN" ] && "$HERMES_BIN" kanban reclaim --dry-run >/dev/null 2>&1; then
    # --dry-run above is a CAPABILITY PROBE: the installed hermes only grows the
    # worktree-reclaim flags once this change ships. Older builds reject it and
    # we fall through to the shell copy instead of silently reclaiming nothing.
    wt_out=$("$HERMES_BIN" kanban reclaim --logs 2>&1); wt_rc=$?
    if [ "$wt_rc" = 0 ]; then
      log "worktrees: $(printf '%s' "$wt_out" | grep -E '^ *-> ' | tr '\n' ' ')"
    else
      log "worktrees: FAIL hermes reclaim rc=$wt_rc, falling back to the shell copy"
      printf '%s\n' "$wt_out" | tail -3 | while read -r l; do log "  $l"; done
      for repo in $(ls -d "$HOME"/workspace/*/.git 2>/dev/null | xargs -n1 dirname); do
        git -C "$repo" worktree list 2>/dev/null | grep -q . && reclaim_merged_worktrees "$repo"
      done
    fi
  else
    # NOT routine. The shell fallback (reclaim_merged_worktrees) is the function
    # whose squash-merge gate made it a permanent no-op: it has removed 0 dirs
    # over its entire log. Taking this branch means the host is STILL LEAKING
    # and the installed hermes needs upgrading — say FAIL so it is greppable.
    log "worktrees: FAIL hermes reclaim unavailable at $HERMES_BIN (upgrade the installed hermes); the shell fallback below is a known no-op"
    for repo in $(ls -d "$HOME"/workspace/*/.git 2>/dev/null | xargs -n1 dirname); do
      git -C "$repo" worktree list 2>/dev/null | grep -q . && reclaim_merged_worktrees "$repo"
    done
  fi
  reclaim_tmp
  reclaim_home_node_modules
  reclaim_pytest_roots
  # Target enforcement runs LAST: only after every ordinary class has been
  # reclaimed do we decide whether to escalate.
  reclaim_to_target
fi

after=$(free_gi)
log "free=${after}Gi (reclaimed $(awk -v a="$after" -v b="$before" 'BEGIN{printf "%.2f", a-b}')Gi)"

# The ONLY paging decision. RECLAIM_TARGET_GI deliberately does not appear
# below this line: a capacity shortfall is handled silently above, and letting
# the target reach this branch is exactly how the guard paged 67 times in 13h.
if lt "$after" "$FLOOR_GI"; then
  below_floor=1
  log "FAIL: ${after}Gi free is below the ${FLOOR_GI}Gi floor."
  log "An ENOSPC host silently breaks every agent tool call and watchdog."
  log "Top reclaimable, largest first (live-pid paths excluded):"
  top_reclaimable 8 | awk -F'\t' '{printf "  %sMB  %s\n", $1, $2}'
  log "NOTE: figures are du-apparent. On APFS du overstates a worker's MARGINAL"
  log "cost ~25x (pnpm clonefile CoW clones counted in full). Deleting 5 such"
  log "dirs on 2026-09-10 really freed 4.9Gi. Size capacity with scratch-cost.sh,"
  log "never from this list."
  exit 1
fi

log "OK: ${after}Gi free."
exit 0
