# swap-selftest.sh -- posix.sh --self-test-swap-interrupt (tests only). Bash
# 3.2 (macOS /bin/bash) compatible.
# shellcheck shell=bash disable=SC2034  # sets posix.sh globals its mac_swap reads
#
# Sourced by posix.sh; uses its REAL mac_swap and mac_bundle_recover. For each
# interruption point of the swap, against a fixture bundle tree in a fresh
# temp dir (never an installed app):
#   1. run mac_swap in its own process group; it SIGSTOPs itself at the named
#      step boundary (posix.sh swap_checkpoint);
#   2. SIGKILL the whole group there -- or resume it and SIGKILL it while the
#      next step (the ditto copy, the cleanup rm) is half done;
#   3. run mac_bundle_recover, as the next hand-off does at start of run.
# Invariant: the target is the COMPLETE old or the COMPLETE new bundle (the
# whole tree is compared, version marker included), no aside/staged copy is
# left, and a second recovery changes nothing and logs nothing.
# Prints one "ok   [cell]" / "FAIL [cell] why" line per cell; exits 1 naming
# the failed cells.
#
# The swap's interruption points (mac_swap order):
#   before-stage  stopped before the copy        -> old, nothing staged
#   mid-stage     killed during ditto            -> old, partial .new
#   staged        copied, before the first mv    -> old, complete .new
#   aside         between the two renames        -> NO app, .old + .new
#   installed     after the second mv, before rm -> new, complete .old
#   mid-cleanup   killed during `rm -rf .old`    -> new, partial .old
#   complete      control: never interrupted     -> new, nothing left
# A kill between a failed second mv and its rollback mv leaves the `aside`
# state. Each rename is one rename(2) on one volume, so there is no mid-rename
# state to kill into.

SWAP_ST_FILES="${HERMES_SELFTEST_SWAP_FILES:-1500}"
SWAP_ST_FAILED="" SWAP_ST_CELL=""

swap_st_fail() { echo "FAIL [$SWAP_ST_CELL] $*"; SWAP_ST_FAILED="$SWAP_ST_FAILED $SWAP_ST_CELL"; }

swap_st_bundle() { # dir version -- a fixture Hermes.app; every file carries the version
  local d="$1" v="$2" i=0
  mkdir -p "$d/Contents/MacOS" "$d/Contents/Resources/payload" || return 1
  printf '%s\n' "$v" > "$d/Contents/Resources/hermes-version"
  printf '#!/bin/sh\necho %s\n' "$v" > "$d/Contents/MacOS/Hermes" && chmod +x "$d/Contents/MacOS/Hermes"
  # Enough files that a kill lands inside the copy / the cleanup.
  while [ "$i" -lt "$SWAP_ST_FILES" ]; do
    printf '%s %s\n' "$v" "$i" > "$d/Contents/Resources/payload/f$i"; i=$((i + 1))
  done
}

swap_st_digest() { # dir -> one checksum of the tree: every path and every file's bytes
  [ -d "$1" ] || { echo absent; return; }
  ( cd "$1" && { find . -print | LC_ALL=C sort; find . -type f -exec cksum {} + | LC_ALL=C sort; } ) | cksum
}

swap_st_listing() { # dir -> every path below it (the residue / no-op check)
  ( cd "$1" && find . -print | LC_ALL=C sort )
}

swap_st_stopped() { # pid -> 0 iff the process is stopped (job-control stop)
  case "$(ps -o stat= -p "$1" 2>/dev/null)" in *T*) return 0 ;; esac
  return 1
}

# One interruption: run the real swap on a fresh fixture, kill it, leave the
# wreck. 0 = the kill landed where the cell says; 1 = it did not (retryable);
# 2 = the swap never reached the step (a hard failure).
swap_st_interrupt() { # root pause-at mode
  local root="$1" pause="$2" mode="$3" t pid i deadline s sentinels
  rm -rf "$root"
  mkdir -p "$root/Applications" "$root/hermes-agent/apps/desktop/release/mac-arm64" || return 2
  cp -R "$SWAP_ST_ROOT/old.app" "$root/Applications/Hermes.app" || return 2
  cp -R "$SWAP_ST_ROOT/new.app" "$root/hermes-agent/apps/desktop/release/mac-arm64/Hermes.app" || return 2
  INSTALL_ROOT="$root/hermes-agent" RELAUNCH_TARGET="$root/Applications/Hermes.app"
  t="$RELAUNCH_TARGET"
  if [ "$mode" = none ]; then
    ( FINAL_CODE=0 APP_REBUILD_FAILED=0; mac_swap ) </dev/null >/dev/null 2>&1
    return 0
  fi
  # Own process group (set -m): the kill takes the swapper AND its ditto/rm
  # child, like a crash of the whole hand-off.
  set -m
  ( SWAP_PAUSE_AT="$pause" FINAL_CODE=0 APP_REBUILD_FAILED=0; mac_swap ) </dev/null >/dev/null 2>&1 &
  pid=$!
  set +m
  for ((i = 0; i < 600; i++)); do
    swap_st_stopped "$pid" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.05
  done
  if ! swap_st_stopped "$pid"; then
    kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return 2
  fi
  deadline=$((SECONDS + 30))
  case "$mode" in
    mid-stage)  # resume into the copy; kill as soon as the staged dir exists
      kill -CONT -- "-$pid"
      while [ ! -e "$t.new" ] && [ ! -e "$t.old" ] && [ "$SECONDS" -lt "$deadline" ]; do :; done ;;
    mid-cleanup)  # resume into `rm -rf .old`; kill once it has deleted anything
      # Builtin tests only (no fork per poll): a dozen files spread over the
      # tree, whichever the rm reaches first.
      sentinels=("$t.old/Contents/Resources/hermes-version" "$t.old/Contents/MacOS/Hermes")
      for ((i = 0; i < SWAP_ST_FILES; i += SWAP_ST_FILES / 10 + 1)); do
        sentinels+=("$t.old/Contents/Resources/payload/f$i")
      done
      kill -CONT -- "-$pid"
      while [ "$SECONDS" -lt "$deadline" ]; do
        for s in "${sentinels[@]}"; do [ -e "$s" ] || break 2; done
      done ;;
  esac
  kill -KILL -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  case "$mode" in
    mid-stage)  # the staged copy must be there and incomplete
      [ -d "$t.new" ] && [ "$(swap_st_digest "$t.new")" != "$SWAP_ST_NEW" ] || return 1 ;;
    mid-cleanup)  # the aside copy must be there and partly deleted
      [ -d "$t.old" ] && [ "$(swap_st_digest "$t.old")" != "$SWAP_ST_OLD" ] || return 1 ;;
  esac
  return 0
}

swap_st_shape() { # dir complete-digest complete-word -> complete-word | old | new | absent | partial
  # (no case inside $(...): bash 3.2 mis-parses its pattern parens there)
  local d
  d="$(swap_st_digest "$1")"
  if [ "$d" = "$2" ]; then echo "$3"
  elif [ "$d" = "$SWAP_ST_OLD" ]; then echo old
  elif [ "$d" = "$SWAP_ST_NEW" ]; then echo new
  elif [ "$d" = absent ]; then echo absent
  else echo partial; fi
}

swap_st_state() { # target -> the wreck's shape before recovery, for the cell's precondition
  echo "target=$(swap_st_shape "$1" "$SWAP_ST_OLD" old) old=$(swap_st_shape "$1.old" "$SWAP_ST_OLD" complete) new=$(swap_st_shape "$1.new" "$SWAP_ST_NEW" complete)"
}

swap_st_cell() { # cell pause-at mode expected-wreck expected-bundle
  local cell="$1" pause="$2" mode="$3" want_state="$4" want="$5"
  local root="$SWAP_ST_ROOT/$1" t attempt rc state got leftover r before log_lines
  SWAP_ST_CELL="$cell"
  t="$root/Applications/Hermes.app"
  for attempt in 1 2 3 4 5; do
    swap_st_interrupt "$root" "$pause" "$mode"; rc=$?
    [ "$rc" -eq 1 ] || break
  done
  case "$rc" in
    1) swap_st_fail "could not land the kill inside the step in 5 tries"; return ;;
    2) swap_st_fail "the swap never stopped at its '$pause' step"; return ;;
  esac
  state="$(swap_st_state "$t")"
  [ "$state" = "$want_state" ] || { swap_st_fail "wreck before recovery is '$state', expected '$want_state'"; return; }

  mac_bundle_recover "$t" >/dev/null 2>&1

  case "$(swap_st_digest "$t")" in
    "$SWAP_ST_OLD") got=old ;;
    "$SWAP_ST_NEW") got=new ;;
    absent) swap_st_fail "no app at the target after recovery ($state)"; return ;;
    *) swap_st_fail "the target is neither the complete old nor the complete new bundle after recovery ($state)"; return ;;
  esac
  [ "$got" = "$want" ] || { swap_st_fail "recovered to the $got bundle, expected $want ($state)"; return; }
  leftover=""
  for r in .old .new .hermes-update-old .hermes-update-new; do
    [ ! -e "$t$r" ] || leftover="$leftover $t$r"
  done
  [ -z "$leftover" ] || { swap_st_fail "left after recovery:$leftover"; return; }

  before="$(swap_st_listing "$root/Applications")"
  log_lines="$(wc -l < "$LOG")"
  mac_bundle_recover "$t" >/dev/null 2>&1
  [ "$(swap_st_listing "$root/Applications")" = "$before" ] \
    || { swap_st_fail "a second recovery changed the tree"; return; }
  [ "$(wc -l < "$LOG")" -eq "$log_lines" ] \
    || { swap_st_fail "a second recovery was not a no-op: $(tail -n 1 "$LOG")"; return; }
  echo "ok   [$cell] $state -> $got bundle (attempt $attempt)"
  rm -rf "$root"
}

swap_selftest() {
  SWAP_ST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/hermes-swap-selftest.XXXXXX")" || return 1
  LOG="$SWAP_ST_ROOT/handoff.log" STATUS="$SWAP_ST_ROOT/status"
  : > "$LOG"
  if [ ! -x "$MAC_DITTO" ]; then
    # Linux: the same staged copy, by cp (ditto exists only on macOS).
    # shellcheck disable=SC2016  # the shim's own "$1" "$2"
    printf '#!/bin/sh\nexec cp -R "$1" "$2"\n' > "$SWAP_ST_ROOT/ditto" && chmod +x "$SWAP_ST_ROOT/ditto"
    MAC_DITTO="$SWAP_ST_ROOT/ditto"
  fi
  echo "swap self-test: $(uname -s), copy=$MAC_DITTO, root=$SWAP_ST_ROOT"
  if ! swap_st_bundle "$SWAP_ST_ROOT/old.app" old-1.0 || ! swap_st_bundle "$SWAP_ST_ROOT/new.app" new-2.0; then
    echo "FAIL [fixture] could not build the fixture bundles"; return 1
  fi
  SWAP_ST_OLD="$(swap_st_digest "$SWAP_ST_ROOT/old.app")" SWAP_ST_NEW="$(swap_st_digest "$SWAP_ST_ROOT/new.app")"

  #            cell          pause-at   mode         wreck before recovery                     after
  swap_st_cell before-stage  start      kill         "target=old old=absent new=absent"        old
  swap_st_cell mid-stage     start      mid-stage    "target=old old=absent new=partial"       old
  swap_st_cell staged        staged     kill         "target=old old=absent new=complete"      old
  swap_st_cell aside         aside      kill         "target=absent old=complete new=complete" old
  swap_st_cell installed     installed  kill         "target=new old=complete new=absent"      new
  swap_st_cell mid-cleanup   installed  mid-cleanup  "target=new old=partial new=absent"       new
  swap_st_cell complete      ""         none         "target=new old=absent new=absent"        new

  if [ -n "$SWAP_ST_FAILED" ]; then
    echo "swap self-test FAILED:$SWAP_ST_FAILED (fixture kept at $SWAP_ST_ROOT)"
    return 1
  fi
  rm -rf "$SWAP_ST_ROOT"
  echo "swap self-test passed: every interruption recovers to a complete bundle"
}
