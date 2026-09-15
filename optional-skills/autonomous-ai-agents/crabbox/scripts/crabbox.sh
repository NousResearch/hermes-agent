#!/usr/bin/env bash
# Normalize supported cloud-sandbox CLIs behind one guarded interface.

set -euo pipefail

CRABBOX_VERSION="1.2.0"
CRABBOX_BACKEND="${CRABBOX_BACKEND:-islo}"
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_FILE="${SCRIPT_DIR}/../SKILL.md"

cmd_skill() {
  if [ ! -f "$SKILL_FILE" ]; then
    echo "crabbox.sh: SKILL.md not found at $SKILL_FILE" >&2
    exit 1
  fi
  cat -- "$SKILL_FILE"
}
# --- Backend registry -------------------------------------------------------
# Normalize the three workhorse verbs and capability flags per backend. Every
# other verb forwards verbatim. To add a backend, append a case arm; unknown
# backends fall back to the islo-style contract.
load_backend() {
  # islo-style defaults. BK_ID_FLAG empty => box names pass POSITIONALLY (islo).
  BK_NEW="use"; BK_LIST="ls"; BK_RM="rm"; BK_AGENT="yes"; BK_SCHEMA="yes"
  BK_RM_FORCE="yes"
  BK_ID_FLAG=""; BK_JSON_FLAG="--output json"
  BK_DURABLE=""   # backends where bare 'new NAME' is durable need no hint (islo)
  BK_INSTALL='curl -fsSL https://islo.dev/install.sh | sh   # then: islo login'
  case "$CRABBOX_BACKEND" in
    islo) : ;;
    crabbox)
      # rm -> stop (per-box teardown), NOT cleanup (fleet-wide GC sweep).
      # crabbox addresses boxes by --id <lease-id-or-slug>, never positionally.
      BK_NEW="run"; BK_LIST="list"; BK_RM="stop"; BK_AGENT="no"; BK_SCHEMA="no"
      BK_RM_FORCE="no"
      BK_ID_FLAG="--id"; BK_JSON_FLAG="--json"
      # 'run' without a kept lease auto-releases on exit; durable boxes use warmup.
      BK_DURABLE="warmup --slug"
      BK_INSTALL='brew install openclaw/tap/crabbox   # docs: https://crabbox.sh' ;;
    *) : ;;  # unknown backend: assume islo-style verbs/capabilities
  esac
}

# Reshape a leading positional NAME so it is passed via the backend's id flag
# (e.g. '--id NAME'). Only argv[0] can be NAME: later positional values may be
# flag arguments or the remote command. Backends with an empty BK_ID_FLAG
# (islo) keep every argument unchanged. Emits NUL-delimited records on fd1.
inject_id_flag() {
  if [ -z "$BK_ID_FLAG" ]; then
    local a
    for a in "$@"; do printf '%s\0' "$a"; done
    return 0
  fi
  if [ $# -gt 0 ] && [ "$1" != "--" ] && [ "${1#-}" = "$1" ]; then
    printf '%s\0%s\0' "$BK_ID_FLAG" "$1"
    shift
  fi
  local a
  for a in "$@"; do printf '%s\0' "$a"; done
}

# Forward a single-box verb with the box NAME bound to the backend's id flag.
forward_with_id() {
  local verb="$1"; shift
  ensure_backend
  local -a out=()
  while IFS= read -r -d '' tok; do out+=("$tok"); done < <(inject_id_flag "$@")
  exec "$CRABBOX_BACKEND" "$verb" "${out[@]}"
}

# Non-blocking heads-up: on backends where a bare 'new NAME' creates an
# ephemeral, auto-released lease (crabbox), point at the durable-box command.
# Fires only when a NAME is given with no remote command and no keep/slug flag.
durable_hint() {
  [ -n "${BK_DURABLE:-}" ] || return 0
  local a has_name=no
  for a in "$@"; do
    case "$a" in
      --) return 0 ;;                       # a remote command follows: one-shot is intended
      --keep|--keep-on-failure|--slug|--slug=*) return 0 ;;
      -*) ;;
      *)  has_name=yes ;;
    esac
  done
  [ "$has_name" = yes ] || return 0
  echo "crabbox.sh: note — '$CRABBOX_BACKEND' leases auto-release on exit; for a durable, reusable box use '$CRABBOX_BACKEND $BK_DURABLE NAME' (or add --keep). Proceeding." >&2
}

# Translate a canonical workhorse verb to the active backend's spelling.
to_backend_verb() {
  case "$1" in
    new)  printf '%s' "$BK_NEW" ;;
    list) printf '%s' "$BK_LIST" ;;
    rm)   printf '%s' "$BK_RM" ;;
    *)    printf '%s' "$1" ;;
  esac
}

ensure_backend() {
  if ! command -v "$CRABBOX_BACKEND" >/dev/null 2>&1; then
    cat >&2 <<EOF
crabbox.sh: backend '$CRABBOX_BACKEND' not found on PATH.

Install it:
  $BK_INSTALL

Known backends (set CRABBOX_BACKEND to choose):
  islo      agent-capable cloud microVMs (default)   https://islo.dev
  crabbox   openclaw remote run/test boxes           https://crabbox.sh
Other CLIs work too — declare their verb names in load_backend().
EOF
    exit 127
  fi
}

forward() {
  ensure_backend
  exec "$CRABBOX_BACKEND" "$@"
}

# Refuse --agent/--task on a backend that can't run an autonomous agent.
guard_agent_flags() {
  [ "$BK_AGENT" = yes ] && return 0
  local a
  for a in "$@"; do
    case "$a" in
      --agent|--task|--agent=*|--task=*)
        cat >&2 <<EOF
crabbox.sh: backend '$CRABBOX_BACKEND' does not run autonomous agents.
It syncs your working tree to a remote box and runs a command (edit-save-run
on cloud compute) — there is no --agent/--task -> PR flow.

  * Remote test/run (this backend):   crabbox.sh new my-box -- pnpm test
  * Autonomous agent -> PR:            CRABBOX_BACKEND=islo crabbox.sh new ... --agent claude --task "..."

See 'crabbox.sh skill' for backend selection and workflow guidance.
EOF
        exit 2 ;;
    esac
  done
}

# Enforce the cleanup-safety contract the skill documents: no wildcard/bulk
# removes, and a name is required. (-a/--all are matched before generic flags.)
guard_rm() {
  local a have_name=no
  for a in "$@"; do
    case "$a" in
      --all|-a|all)
        echo "crabbox.sh: refusing bulk remove ('$a'). Name exactly one box; run 'crabbox.sh list' first." >&2
        exit 2 ;;
      *'*'*|*'?'*|'~'*)
        echo "crabbox.sh: refusing glob in box name ('$a'). Name exactly one box; run 'crabbox.sh list' first." >&2
        exit 2 ;;
      -*) : ;;            # other flags (e.g. -f) are fine
      *)  have_name=yes ;;
    esac
  done
  if [ "$have_name" = no ]; then
    echo "crabbox.sh: rm needs an explicit box name. Run 'crabbox.sh list' first." >&2
    exit 2
  fi
}

# The wrapper accepts -f/--force for compatibility with islo. Crabbox stop
# is already a direct single-lease teardown and defines neither flag.
forward_rm() {
  local -a out=()
  local a
  for a in "$@"; do
    if [ "$BK_RM_FORCE" = no ] && { [ "$a" = "-f" ] || [ "$a" = "--force" ]; }; then
      continue
    fi
    out+=("$a")
  done
  forward_with_id "$(to_backend_verb rm)" "${out[@]}"
}

cmd_backends() {
  cat <<EOF
crabbox.sh backends — set CRABBOX_BACKEND to choose (current: $CRABBOX_BACKEND)

  islo     [default] agent-capable cloud microVMs. Provision a sandbox, clone a
           repo, run claude/cursor against --task, return a PR.
           verbs:  new->use   list->ls    rm->rm
           addr:   positional box name        json: --output json
           caps:   agent: yes   schema: yes
           install: curl -fsSL https://islo.dev/install.sh | sh && islo login

  crabbox  openclaw/crabbox — "warm a box, sync the diff, run the suite". Rsync
           your dirty checkout to a leased box and run a command. No agent/PR.
           verbs:  new->run   list->list  rm->stop
           addr:   --id NAME (lease-id or slug); NEVER positional
                   single-box verbs: new/status/rm/stop/pause/resume/ssh/share/ports
           json:   --json (per-command, append after positionals)
           caps:   agent: no    schema: no
           notes:  rm->stop (per-box teardown). NOT 'cleanup' (fleet GC sweep,
                   takes no id, refuses under a coordinator). 'release' is a
                   compat alias for stop. 'run' WITHOUT --id is ephemeral and
                   auto-releases on exit; for a durable named box use
                   'crabbox warmup --slug NAME' (or 'run --keep'). 'logs' takes
                   a RUN id (run_<hex>), NOT a box name. pause/resume are
                   provider-gated (islo, codesandbox).
           install: brew install openclaw/tap/crabbox   (https://crabbox.sh)

Patterns needing --agent/--task (Build/Review/Refine) require an agent-capable
backend (islo). Remote test/run (Pattern 8) works on crabbox. Add a backend by
declaring its verb names, addressing (BK_ID_FLAG), and capabilities in
load_backend().
EOF
}

usage() {
  local BK_ADDR_DESC
  if [ -n "$BK_ID_FLAG" ]; then
    BK_ADDR_DESC="$BK_ID_FLAG NAME (injected for single-box verbs)"
  else
    BK_ADDR_DESC="positional NAME"
  fi
  cat <<EOF
crabbox.sh v$CRABBOX_VERSION — Hermes skill sandbox wrapper
active backend: $CRABBOX_BACKEND  (new->$(to_backend_verb new), list->$(to_backend_verb list), rm->$(to_backend_verb rm))
box addressing: ${BK_ADDR_DESC}   json: $BK_JSON_FLAG

Usage:
  crabbox.sh skill                            Print the canonical SKILL.md
  crabbox.sh backends                         List wired-in backends + capabilities
  crabbox.sh new NAME [flags] [-- CMD]        Lease a box (backend: $CRABBOX_BACKEND $(to_backend_verb new))
  crabbox.sh list [flags]                     List boxes
  crabbox.sh status [NAME]                    Show box / auth status
  crabbox.sh logs NAME [flags]                Stream box logs
  crabbox.sh rm NAME -f                       Remove a box (wildcard-guarded; no recycle bin)
  crabbox.sh schema [COMMAND]                 Backend schema (JSON); crabbox: stubbed (use --json/providers/config show)
  crabbox.sh doctor                           Backend system health check
  crabbox.sh pause|resume|stop NAME           Box lifecycle
  crabbox.sh login|logout [--tool ...]        Auth + integration management
  crabbox.sh init|add [TOOL]                  Project setup
  crabbox.sh ssh|share|ports|snapshot ...     Passthrough to backend
  crabbox.sh update                           Update the backend CLI
  crabbox.sh version                          Print crabbox + backend version
  crabbox.sh help                             This message

Environment:
  CRABBOX_BACKEND    sandbox CLI to drive (default: islo; also: crabbox).
                     Verb names + capabilities are declared in load_backend().

Run 'crabbox.sh skill' for workflow guidance and 'crabbox.sh backends'
to compare islo (agent->PR) vs openclaw/crabbox (remote run/test).
EOF
}

load_backend

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  skill|--skill)               cmd_skill ;;
  backends)                    cmd_backends ;;
  help|--help|-h|"")           usage ;;
  version|--version)           ensure_backend
                               echo "crabbox.sh $CRABBOX_VERSION (backend: $CRABBOX_BACKEND $("$CRABBOX_BACKEND" --version 2>/dev/null | head -1))" ;;
  new)                         guard_agent_flags "$@"
                               durable_hint "$@"
                               forward_with_id "$(to_backend_verb new)" "$@" ;;
  list)                        forward "$(to_backend_verb list)" "$@" ;;
  rm)                          guard_rm "$@"
                               forward_rm "$@" ;;
  status|pause|resume|stop|ssh|share|ports)
                               forward_with_id "$(to_backend_verb "$cmd")" "$@" ;;
  schema)
    if [ "$BK_SCHEMA" = yes ]; then
      forward schema "$@"
    else
      ensure_backend
      cat >&2 <<EOF
crabbox.sh: backend '$CRABBOX_BACKEND' is NOT schema-capable — there is no
schema/introspect/completion verb to defer to. Machine-readable discovery is
limited to:
  * per-command --json   (status, list, inspect, ports, doctor, providers, logs, ...)
  * $CRABBOX_BACKEND providers --json
  * $CRABBOX_BACKEND config show --json
Showing '--help' for orientation only (this is NOT a schema):
EOF
      if [ $# -gt 0 ]; then
        sub="$(to_backend_verb "$1")"; shift
        "$CRABBOX_BACKEND" "$sub" "$@" --help >&2 || true
      else
        "$CRABBOX_BACKEND" --help >&2 || true
      fi
      exit 2
    fi ;;
  *)                           forward "$cmd" "$@" ;;
esac
