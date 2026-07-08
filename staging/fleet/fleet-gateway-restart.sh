#!/usr/bin/env bash
#
# fleet-gateway-restart.sh — restart ALL Hermes gateways and VERIFY each loaded new code.
#
# SPEC: ~/.hermes/plans/2026-07-01_fleet-gateway-restart-SPEC.md
#
# Fixes the recurring partial-restart gap: a by-hand `kickstart` loop misses gateways (2/5 on 2026-07-01),
# and "it restarted" gets asserted without the pid-vs-deploy-mtime check that catches a stale one. This
# enumerates the fleet dynamically, restarts idle siblings directly, defers the self/busy gateway through
# the safe-restart front door, and VERIFIES each: fresh pid + running + started-after-deploy + on runtime tree.
#
# Usage:
#   fleet-gateway-restart.sh [--since <file|epoch>] [--only <label,...>] [--dry-run] [--json]
#                            [--handoff "<self-restart handoff>"]
#   --since    freshness anchor (default: the deploy stamp ~/.hermes/runtime/.deploy-stamp written by
#              deploy.sh after a successful FF; falls back to deepest runtime *.py mtime, then .git/HEAD).
#              A gateway is fresh iff it started at/after this. Pass a file (its mtime) or a raw epoch.
#   --only     restrict to a comma-separated label list (e.g. ai.hermes.gateway-argus). Default: whole fleet.
#   --dry-run  print the plan; restart + verify nothing.
#   --json     machine-readable result.
#   --handoff  handoff for the deferred self restart.
#
# Lib mode (functions only, no main), for tests:  FGR_LIB=1 source fleet-gateway-restart.sh
#
set -uo pipefail

MYUID="$(id -u)"
HERMES="${HERMES_HOME_ROOT:-$HOME/.hermes}"          # ops root (NOT a profile dir)
RUNTIME_TREE="${FGR_RUNTIME_TREE:-$HERMES/runtime/hermes-agent}"
LA_DIR="${FGR_LA_DIR:-$HOME/Library/LaunchAgents}"
SAFE_RESTART="${FGR_SAFE_RESTART:-$HERMES/skills-shared/general/safe-gateway-restart/scripts/safe-restart.py}"
VENV_PY="${FGR_VENV_PY:-$RUNTIME_TREE/venv/bin/python}"
VERIFY_POLLS="${FGR_VERIFY_POLLS:-30}"               # 2s each -> 60s per label
DEFAULT_LABEL="ai.hermes.gateway"

# ---- injectable command hooks (tests override with stub *.sh) ----
FGR_LAUNCHCTL="${FGR_LAUNCHCTL:-launchctl}"
FGR_PS="${FGR_PS:-ps}"
FGR_LSOF="${FGR_LSOF:-lsof}"

# ---------------------------------------------------------------------------
# enumeration (INV-3): gateway plists, minus watchdog / bak / disabled noise
# ---------------------------------------------------------------------------
fgr_labels() {
  local f base
  for f in "$LA_DIR"/ai.hermes.gateway*.plist; do
    [ -e "$f" ] || continue
    base="$(basename "$f" .plist)"
    case "$base" in
      *watchdog*|*.bak*|*.disabled*) continue ;;
    esac
    echo "$base"
  done
}

# profile name from a gateway label: ai.hermes.gateway -> default ; ai.hermes.gateway-aegis -> aegis
fgr_profile_of() {
  local label="$1"
  case "$label" in
    "$DEFAULT_LABEL") echo "default" ;;
    "$DEFAULT_LABEL"-*) echo "${label#"$DEFAULT_LABEL"-}" ;;
    *) echo "" ;;
  esac
}

# state file path for a profile (default lives at HERMES root; others under profiles/<p>/)
fgr_state_path() {
  local profile="$1"
  if [ "$profile" = "default" ]; then echo "$HERMES/gateway_state.json"
  else echo "$HERMES/profiles/$profile/gateway_state.json"; fi
}

# active_agents for a profile: integer, or "unknown" (missing/corrupt) -> caller treats as BUSY (INV-1)
fgr_active_agents() {
  local profile="$1" sp; sp="$(fgr_state_path "$profile")"
  [ -f "$sp" ] || { echo "unknown"; return; }
  python3 - "$sp" <<'PY' 2>/dev/null || echo "unknown"
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    v=d.get("active_agents")
    print(int(v) if v is not None else "unknown")
except Exception:
    print("unknown")
PY
}

fgr_is_idle() { [ "$(fgr_active_agents "$1")" = "0" ]; }

# ---------------------------------------------------------------------------
# pending-restart ledger (A1b): durable memory of a BUSY-SKIPPED gateway so a
# re-drive (runtime-parity-check.py::_redrive_pending_restarts) can bring it
# current the moment it goes idle, instead of it sitting stale until a human
# re-runs. Fail-closed atomic tmp-rename, same pattern as runtime-parity.json.
# Entry shape: {label: {target_sha, since_epoch, first_skipped, last_skipped}}.
# ---------------------------------------------------------------------------
FGR_LEDGER="${FGR_LEDGER:-$HERMES/state/pending-gateway-restart.json}"

# the sha a restart should land the gateway on (runtime tree HEAD). Empty on error.
fgr_target_sha() { git -C "$RUNTIME_TREE" rev-parse HEAD 2>/dev/null | tr -dc '0-9a-f'; }

# append/update a BUSY-SKIPPED label in the ledger (idempotent: refresh
# last_skipped + target_sha, preserve first_skipped). No-op on json error.
fgr_ledger_add() {
  local label="$1" since="$2" tgt; tgt="$(fgr_target_sha)"
  mkdir -p "$(dirname "$FGR_LEDGER")" 2>/dev/null
  python3 - "$FGR_LEDGER" "$label" "$tgt" "$since" "$(date +%s)" <<'PY' 2>/dev/null || true
import json, os, sys
path, label, tgt, since, now = sys.argv[1:6]
try:
    d = json.load(open(path))
    if not isinstance(d, dict): d = {}
except Exception:
    d = {}
e = d.get(label) or {}
d[label] = {
    "target_sha": tgt,
    "since_epoch": since,
    "first_skipped": e.get("first_skipped", now),
    "last_skipped": now,
    "attempts": int(e.get("attempts", 0)),
}
tmp = path + ".tmp.%d" % os.getpid()
with open(tmp, "w") as f: json.dump(d, f, indent=2)
os.replace(tmp, path)
PY
}

# clear a label from the ledger (it VERIFIED). No-op if absent / json error.
fgr_ledger_clear() {
  local label="$1"
  [ -f "$FGR_LEDGER" ] || return 0
  python3 - "$FGR_LEDGER" "$label" <<'PY' 2>/dev/null || true
import json, os, sys
path, label = sys.argv[1:3]
try:
    d = json.load(open(path))
    if not isinstance(d, dict):
        sys.exit(0)
except Exception:
    sys.exit(0)
if label in d:
    del d[label]
    tmp = path + ".tmp.%d" % os.getpid()
    with open(tmp, "w") as f: json.dump(d, f, indent=2)
    os.replace(tmp, path)
PY
}

# which gateway are WE running under (INV-4)? Trust HERMES_PROFILE ONLY if cross-checked against actual
# process ancestry — a bare shell can carry a STALE exported HERMES_PROFILE (pass-2 B-2), which would defer
# the wrong label and (if the real self is idle) plain-kickstart the gateway we're under = self-kill. We
# confirm the env-named gateway's launchd pid is an ANCESTOR of this process; if not, treat as "no self".
fgr_self_label() {
  local p="${HERMES_PROFILE:-}"
  [ -z "$p" ] && { echo ""; return; }
  local label
  if [ "$p" = "default" ]; then label="$DEFAULT_LABEL"; else label="$DEFAULT_LABEL-$p"; fi
  # cross-check: is that label's live pid an ancestor of us? (walk our ppid chain)
  local gw_pid; gw_pid="$(fgr_pid_of "$label")"
  if [ -z "$gw_pid" ]; then echo ""; return; fi   # env names a gateway that isn't loaded -> not self
  if [ "${FGR_SKIP_ANCESTRY:-0}" = "1" ]; then echo "$label"; return; fi  # test hook
  local cur="$$" guard=0
  while [ -n "$cur" ] && [ "$cur" != "0" ] && [ "$cur" != "1" ] && [ "$guard" -lt 40 ]; do
    if [ "$cur" = "$gw_pid" ]; then echo "$label"; return; fi
    cur="$($FGR_PS -o ppid= -p "$cur" 2>/dev/null | tr -dc '0-9')"
    guard=$((guard+1))
  done
  echo ""   # HERMES_PROFILE set but we are NOT under that gateway -> stale env -> treat as no-self
}

fgr_pid_of() { $FGR_LAUNCHCTL print "gui/$MYUID/$1" 2>/dev/null | grep -m1 'pid = ' | tr -dc '0-9'; }
fgr_state_of() { $FGR_LAUNCHCTL print "gui/$MYUID/$1" 2>/dev/null | grep -m1 'state = ' | awk '{print $3}'; }

# process start epoch for a pid (via ps lstart)
fgr_start_epoch() {
  local pid="$1" ls
  ls="$($FGR_PS -o lstart= -p "$pid" 2>/dev/null)"
  [ -n "$ls" ] || { echo 0; return; }
  date -j -f "%a %b %d %T %Y" "$ls" +%s 2>/dev/null || echo 0
}

# resolve --since to an epoch. Precedence (pass-1 B-1: .git/HEAD mtime is a fragile, non-monotonic,
# content-free anchor — a `git checkout <same-sha>`/re-clone/backup-restore resets it to now with ZERO code
# change, false-FAILing every gateway and training the operator to ignore FAILED). So:
#   1. explicit --since (file mtime or raw epoch)
#   2. the deploy stamp deploy.sh writes AFTER a successful FF (content-true: only advances on a real deploy)
#   3. deepest *.py mtime in the runtime tree (content-true fallback if the stamp is absent)
#   4. .git/HEAD mtime (last-resort legacy)
FGR_DEPLOY_STAMP="${FGR_DEPLOY_STAMP:-$HERMES/runtime/.deploy-stamp}"
fgr_since_epoch() {
  local since="${1:-}"
  if [ -n "$since" ]; then
    if [ -f "$since" ]; then stat -f %m "$since" 2>/dev/null || echo 0; else echo "$since"; fi
    return
  fi
  if [ -f "$FGR_DEPLOY_STAMP" ]; then
    # stamp holds an epoch (written by deploy.sh); trust its content, fall back to its mtime
    local v; v="$(tr -dc '0-9' < "$FGR_DEPLOY_STAMP" 2>/dev/null)"
    if [ -n "$v" ]; then echo "$v"; return; fi
    stat -f %m "$FGR_DEPLOY_STAMP" 2>/dev/null && return
  fi
  # deepest .py mtime in the runtime tree (content-true: newest actually-imported source)
  local deep; deep="$(find "$RUNTIME_TREE" -name '*.py' -not -path '*/venv/*' -not -path '*/.git/*' -exec stat -f %m {} + 2>/dev/null | sort -rn | head -1)"
  if [ -n "$deep" ]; then echo "$deep"; return; fi
  stat -f %m "$RUNTIME_TREE/.git/HEAD" 2>/dev/null || echo 0
}

# INV-2 verification: fresh pid (!= old) + running + start >= since + runtime-venv open files > 0
# args: label old_pid since_epoch  -> echoes "VERIFIED pid=.. started=.." or "FAILED <reason>"
fgr_verify() {
  local label="$1" old_pid="$2" since="$3" i pid st start rtfiles
  for ((i=0;i<VERIFY_POLLS;i++)); do
    pid="$(fgr_pid_of "$label")"
    st="$(fgr_state_of "$label")"
    if [ -n "$pid" ] && [ "$pid" != "$old_pid" ] && [ "$st" = "running" ]; then
      start="$(fgr_start_epoch "$pid")"
      if [ "$start" -ge "$since" ] 2>/dev/null; then
        rtfiles="$($FGR_LSOF -p "$pid" 2>/dev/null | grep -c "$RUNTIME_TREE/venv/")"
        if [ "${rtfiles:-0}" -gt 0 ] 2>/dev/null; then
          echo "VERIFIED pid=$pid started=$start"; return 0
        fi
      fi
    fi
    sleep 2
  done
  # one last snapshot for the failure reason
  pid="$(fgr_pid_of "$label")"; st="$(fgr_state_of "$label")"; start="$(fgr_start_epoch "${pid:-0}")"
  if [ -z "$pid" ]; then echo "FAILED not-loaded/no-pid"; return 1; fi
  if [ "$pid" = "$old_pid" ]; then echo "FAILED pid-unchanged($pid)"; return 1; fi
  if [ "$st" != "running" ]; then echo "FAILED state=$st"; return 1; fi
  if [ "$start" -lt "$since" ] 2>/dev/null; then echo "FAILED stale-start($start<$since)"; return 1; fi
  echo "FAILED no-runtime-venv-files"; return 1
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
_fgr_main() {
  local since_arg="" only="" dry=0 json=0 handoff=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --since) since_arg="$2"; shift 2 ;;
      --only) only="$2"; shift 2 ;;
      --dry-run) dry=1; shift ;;
      --json) json=1; shift ;;
      --handoff) handoff="$2"; shift 2 ;;
      -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; return 0 ;;
      *) echo "fleet-gateway-restart: unknown arg: $1" >&2; return 2 ;;
    esac
  done

  local since; since="$(fgr_since_epoch "$since_arg")"
  local self; self="$(fgr_self_label)"
  [ -z "$handoff" ] && handoff="fleet restart to load runtime $(git -C "$RUNTIME_TREE" rev-parse --short HEAD 2>/dev/null); verify + report"

  local all_labels label want
  all_labels=()
  while IFS= read -r label; do [ -n "$label" ] && all_labels+=("$label"); done < <(fgr_labels)

  # apply --only filter
  local labels=()
  for label in ${all_labels[@]+"${all_labels[@]}"}; do
    if [ -n "$only" ]; then
      case ",$only," in *,"$label",*) labels+=("$label") ;; esac
    else labels+=("$label"); fi
  done

  # pass-1 RC-5: --only naming a label that doesn't enumerate = loud FAIL, never a green-on-nothing.
  if [ -n "$only" ]; then
    local req want found
    IFS=',' read -ra _req <<< "$only"
    for want in "${_req[@]}"; do
      [ -z "$want" ] && continue
      found=0
      for label in ${labels[@]+"${labels[@]}"}; do [ "$label" = "$want" ] && found=1; done
      if [ "$found" = "0" ]; then
        echo "fleet-gateway-restart: --only label not found in the fleet: $want" >&2
        echo "  (known: $(echo ${all_labels[@]+"${all_labels[@]}"}))" >&2
        return 2
      fi
    done
  fi

  local results=() n_verified=0 n_failed=0 n_skipped=0 n_deferred=0 self_pending=""
  for label in ${labels[@]+"${labels[@]}"}; do
    local profile; profile="$(fgr_profile_of "$label")"
    if [ "$label" = "$self" ]; then
      self_pending="$label"   # handle LAST (INV-4)
      continue
    fi
    if fgr_is_idle "$profile"; then
      local old_pid; old_pid="$(fgr_pid_of "$label")"
      if [ "$dry" = "1" ]; then
        results+=("$label: WOULD-KICKSTART (idle, old pid ${old_pid:-none})"); continue
      fi
      $FGR_LAUNCHCTL kickstart -k "gui/$MYUID/$label" >/dev/null 2>&1
      local v; v="$(fgr_verify "$label" "${old_pid:-0}" "$since")"
      results+=("$label: $v")
      case "$v" in
        VERIFIED*) n_verified=$((n_verified+1)); fgr_ledger_clear "$label" ;;
        *) n_failed=$((n_failed+1)) ;;
      esac
    else
      results+=("$label: BUSY-SKIPPED (active_agents=$(fgr_active_agents "$profile"))")
      n_skipped=$((n_skipped+1))
      # A1b: remember the skip so the re-drive can bring it current when idle.
      fgr_ledger_add "$label" "$since"
    fi
  done

  # self LAST: deferred safe-restart (INV-4)
  if [ -n "$self_pending" ]; then
    if [ "$dry" = "1" ]; then
      results+=("$self_pending: WOULD-DEFER (self, via safe-restart)")
    elif [ -x "$VENV_PY" ] && [ -f "$SAFE_RESTART" ]; then
      "$VENV_PY" "$SAFE_RESTART" --handoff "$handoff" >/dev/null 2>&1 \
        && { results+=("$self_pending: DEFERRED (safe-restart armed)"); n_deferred=$((n_deferred+1)); } \
        || { results+=("$self_pending: FAILED safe-restart-spawn"); n_failed=$((n_failed+1)); }
    else
      results+=("$self_pending: FAILED safe-restart-or-venv-missing"); n_failed=$((n_failed+1))
    fi
  fi

  # report
  local total=${#results[@]}
  if [ "$json" = "1" ]; then
    printf '{"since":%s,"verified":%d,"failed":%d,"skipped":%d,"deferred":%d,"results":[' \
      "$since" "$n_verified" "$n_failed" "$n_skipped" "$n_deferred"
    local first=1 r
    for r in ${results[@]+"${results[@]}"}; do
      [ "$first" = 1 ] && first=0 || printf ','
      printf '%s' "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$r")"
    done
    printf ']}\n'
  else
    for r in ${results[@]+"${results[@]}"}; do echo "  $r"; done
    echo "fleet-restart: $n_verified verified, $n_failed failed, $n_skipped busy-skipped, $n_deferred deferred (of $total)"
  fi

  # exit 0 iff every non-deferred label VERIFIED (dry-run always 0)
  [ "$dry" = "1" ] && return 0
  { [ "$n_failed" -eq 0 ] && [ "$n_skipped" -eq 0 ]; } && return 0 || return 1
}

if [ "${FGR_LIB:-0}" != "1" ]; then
  _fgr_main "$@"
fi
