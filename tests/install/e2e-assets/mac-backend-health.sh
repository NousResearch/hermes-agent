#!/usr/bin/env bash
# Wait for the relaunched app's backend after a Squirrel.Mac update and save its /api/health.
# usage: mac-backend-health.sh --payload <agent-payload dir> --newer-than <app pid> --timeout <s> --out <file>
# Prints the backend's port. Processes are found by executable path, not parentage: the
# relaunched app may still be starting its backend, and 0.23 runs the gateway as a launchd job
# outside the app's process tree. Only processes no older than --newer-than count, because the
# payload path is the same before and after the in-place update.
set -uo pipefail

payload="" ref="" timeout=180 out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --payload) payload="${2%/}/"; shift 2 ;;
    --newer-than) ref="$2"; shift 2 ;;
    --timeout) timeout="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "mac-backend-health: unknown argument $1" >&2; exit 2 ;;
  esac
done
if [ -z "$payload" ] || [ -z "$ref" ] || [ -z "$out" ]; then
  echo "usage: mac-backend-health.sh --payload <dir> --newer-than <pid> --timeout <s> --out <file>" >&2
  exit 2
fi

# ps etime is [[dd-]hh:]mm:ss.
seconds() {
  awk -v e="$1" 'BEGIN { d = 0; if (index(e, "-")) { split(e, p, "-"); d = p[1]; e = p[2] }
    n = split(e, a, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + a[i]; print d * 86400 + s }'
}

candidates() {
  local ref_etime ref_age pid etime args
  ref_etime="$(ps -o etime= -p "$ref" 2>/dev/null | tr -d ' ')"
  [ -n "$ref_etime" ] || return 0
  ref_age="$(seconds "$ref_etime")"
  ps -axo pid=,etime=,args= | while read -r pid etime args; do
    case "$args" in "$payload"*) ;; *) continue ;; esac
    [ "$(seconds "$etime")" -le "$ref_age" ] && echo "$pid"
  done
}

deadline=$((SECONDS + timeout))
while :; do
  for pid in $(candidates); do
    for port in $(lsof -nP -a -p "$pid" -iTCP -sTCP:LISTEN -Fn 2>/dev/null | sed -n 's/^n.*:\([0-9][0-9]*\)$/\1/p' | sort -u); do
      if curl -fsS --max-time 2 -o "$out.tmp" "http://127.0.0.1:$port/api/health" 2>/dev/null; then
        mv "$out.tmp" "$out"
        echo "$port"
        exit 0
      fi
    done
  done
  [ "$SECONDS" -lt "$deadline" ] || break
  sleep 2
done
rm -f "$out.tmp"
echo "mac-backend-health: no process under $payload newer than pid $ref answered /api/health within ${timeout}s" >&2
echo "mac-backend-health: payload processes now (pid, elapsed, command):" >&2
ps -axo pid=,etime=,args= | grep -F -- "$payload" | grep -v -e 'grep -F' -e mac-backend-health >&2
exit 1
