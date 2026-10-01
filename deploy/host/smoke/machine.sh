# shellcheck shell=bash
#
# The machine-topology half of deploy/host/smoke.sh, sourced by it after the
# image is built (`smoke.sh --topology machine`). Two matter slots, a and b,
# run through the real litco-supervisor API in one container
# (FIRM_HOST_WAVE2 section 8.1):
#
#   1. the image: TOPOLOGY is machine, no litco-agent.service, no secret on disk
#   2. the supervisor starts with a test env file and no interface binding;
#      /health answers and GET /slots without the bearer is 401
#   3. PUT /slots/a and /slots/b with the stub model: started, ports 8800 and
#      8801, users m_a and m_b; each slot's /health names its own matter
#   4. systemd-analyze verify litco-agent@a.service
#   5. a turn on each slot; B's matter sent to slot a is 403 matter_mismatch
#   6. litco-slot-probe a b and b a: all four reads denied both ways; and a
#      control: the metadata address answers from outside any slot
#   7. a slow turn held on a, then POST /slots/a/stop: 202; a new turn gets
#      503 draining; the held turn finishes; a reaches stopped; its env file is
#      gone; b still answers
#   8. PUT /slots/a again: same port, same user, home intact
#   9. DELETE /slots/b: user, home and registry entry gone; port 8801 is free
#      (the next new slot gets it)
#  10. MemoryCurrent of each idle slot, printed
#
# Expects from smoke.sh: IMAGE, NAME, KEEP, fail(), step(). Sets its own fail().

SUP_PORT="${LITCO_SMOKE_SUPERVISOR_PORT:-18700}"
PORT_A="${LITCO_SMOKE_SLOT_PORT_A:-18800}"
PORT_B="${LITCO_SMOKE_SLOT_PORT_B:-18801}"
SUP="http://127.0.0.1:${SUP_PORT}"
# Fake secrets for the container; none is real.
SUP_SECRET="smoke-supervisor-secret-not-real-0123456789"
SECRET_A="smoke-host-secret-a-not-real-0123456789abcdef"
SECRET_B="smoke-host-secret-b-not-real-0123456789abcdef"
SECRET_C="smoke-host-secret-c-not-real-0123456789abcdef"
MATTER_A="smoke-matter-a"
MATTER_B="smoke-matter-b"
MATTER_C="smoke-matter-c"
TMPD="$(mktemp -d)"

fail() {
  echo "SMOKE FAIL: $*" >&2
  docker exec "$NAME" journalctl -u litco-supervisor --no-pager -n 60 >&2 2>/dev/null || true
  for slot in a b c; do
    docker exec "$NAME" journalctl -u "litco-agent@${slot}" --no-pager -n 40 >&2 2>/dev/null || true
  done
  exit 1
}
in_box() { docker exec "$NAME" "$@"; }
# py <expr> <json>: evaluate a Python expression over the JSON document `d`.
py() { python3 -c 'import json, sys; d = json.loads(sys.argv[2]); print(eval(sys.argv[1]))' "$1" "$2"; }

# api <method> <path> [json] [token]: prints "<status> <body>".
api() {
  local method="$1" path="$2" data="${3:-}" token="${4-$SUP_SECRET}" code
  local args=(-sS --max-time 60 -X "$method" -o "$TMPD/body" -w '%{http_code}')
  [[ -n "$token" ]] && args+=(-H "Authorization: Bearer ${token}")
  [[ -n "$data" ]] && args+=(-H 'Content-Type: application/json' --data "$data")
  : > "$TMPD/body"
  code="$(curl "${args[@]}" "${SUP}${path}" 2>/dev/null)" || code="${code:-000}"
  printf '%s %s\n' "$code" "$(cat "$TMPD/body")"
}
status_of() { printf '%s' "${1%% *}"; }
body_of() { printf '%s' "${1#* }"; }

slot_body() {  # <matter> <host secret>
  printf '{"matterId":"%s","instanceUrl":"http://127.0.0.1:9","hostSecret":"%s","agentToken":"lkm_%s_not_real","modelProvider":"custom","model":"stub-model","modelBaseUrl":"http://127.0.0.1:18080/v1"}' \
    "$1" "$2" "${1//-/_}"
}

turn_json() {  # <matter> <session> <text>
  printf '{"matterId":"%s","sessionId":"%s","text":"%s","channel":"web","kind":"channel"}' "$1" "$2" "$3"
}

# turn <port> <secret> <matter> <session> <text>: the SSE stream on stdout.
turn() {
  curl -fsS --max-time 240 -N -X POST "http://127.0.0.1:$1/turn" -H "X-Host-Secret: $2" \
    -H "Content-Type: application/json" -d "$(turn_json "$3" "$4" "$5")"
}

# turn_status <port> <secret> <matter> <session> <text>: "<status> <body>" of a refused turn.
turn_status() {
  local code
  : > "$TMPD/turn"
  code="$(curl -sS --max-time 30 -o "$TMPD/turn" -w '%{http_code}' -X POST "http://127.0.0.1:$1/turn" \
    -H "X-Host-Secret: $2" -H "Content-Type: application/json" -d "$(turn_json "$3" "$4" "$5")" 2>/dev/null)" \
    || code="${code:-000}"
  printf '%s %s\n' "$code" "$(cat "$TMPD/turn")"
}

assert_final() {  # <sse text> <label>
  python3 - "$1" "$2" <<'PY' || fail "$2: SSE stream did not end with the stub model's final frame"
import json, sys
frames = [json.loads(line[6:]) for line in sys.argv[1].splitlines() if line.startswith("data: ")]
assert frames and frames[0]["type"] == "goal_accepted", frames[:1]
assert frames[-1]["type"] == "final" and "SMOKE-OK" in frames[-1]["text"], frames[-1:]
print(f"   {sys.argv[2]}: final.text: {frames[-1]['text'][:80]}")
PY
}

health() { curl -fsS --max-time 3 "http://127.0.0.1:$1/health" 2>/dev/null; }

wait_health() {  # <port> <matter>
  local h=""
  for _ in $(seq 1 120); do
    h="$(health "$1" || true)"
    if [[ -n "$h" ]] && [[ "$(py 'd.get("matterId")' "$h")" == "$2" ]]; then
      echo "   :$1 $h"
      return 0
    fi
    sleep 2
  done
  fail "slot on port $1 never answered /health for $2 (last: ${h:-nothing})"
}

slot_state() {  # <id>: the supervisor's state for a slot, or "absent"
  local r
  r="$(api GET /slots)"
  py "next((s['state'] for s in d['slots'] if s['id'] == '$1'), 'absent')" "$(body_of "$r")"
}

wait_slot_state() {  # <id> <state> <seconds>
  local s=""
  for _ in $(seq 1 "$3"); do
    s="$(slot_state "$1")"
    [[ "$s" == "$2" ]] && return 0
    sleep 1
  done
  fail "slot $1 did not reach $2 in $3 s (last: $s)"
}

memory_mib() {  # <id>
  local bytes
  bytes="$(in_box systemctl show -p MemoryCurrent --value "litco-agent@$1.service")"
  if [[ "$bytes" =~ ^[0-9]+$ ]]; then
    python3 -c "print(f'{int(\"$bytes\") / 1048576:.0f} MiB ({$bytes} bytes)')"
  else
    echo "not reported (${bytes:-empty})"
  fi
}

step "boot systemd in the container (machine topology)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --privileged --cgroupns=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw --tmpfs /run --tmpfs /run/lock \
  -p "127.0.0.1:${SUP_PORT}:8700" -p "127.0.0.1:${PORT_A}:8800" -p "127.0.0.1:${PORT_B}:8801" \
  "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
  state="$(in_box systemctl is-system-running 2>/dev/null || true)"
  [[ "$state" == running || "$state" == degraded ]] && break
  sleep 1
done
echo "   systemd: ${state:-unknown}"

step "1. the image: topology machine, no per-matter unit, no secret on disk"
[[ "$(in_box cat /opt/litco-agent/TOPOLOGY)" == machine ]] || fail "/opt/litco-agent/TOPOLOGY is not machine"
in_box test ! -e /etc/systemd/system/litco-agent.service || fail "the machine image has litco-agent.service"
in_box test ! -e /etc/litco-supervisor.env || fail "the image carries /etc/litco-supervisor.env"
in_box test ! -e /etc/litco-agent/env || fail "the image carries /etc/litco-agent/env"
in_box sh -c '! ls /run/litco-agent/*.env >/dev/null 2>&1' || fail "the image carries a slot env file"
for tool in litco-supervisor litco-host-update litco-slot-probe litco-slot-import; do
  in_box test -x "/usr/local/sbin/$tool" || fail "/usr/local/sbin/$tool is missing"
done
echo "   current -> $(in_box readlink /opt/litco-agent/current)"

step "2. supervisor up with a test env file and no interface binding"
printf 'LITCO_SUPERVISOR_SECRET="%s"\nLITCO_SUPERVISOR_INTERFACE=""\n' "$SUP_SECRET" \
  | docker exec -i "$NAME" sh -c 'umask 077 && cat > /etc/litco-supervisor.env'
in_box systemctl restart litco-supervisor.service
sup_health=""
for _ in $(seq 1 30); do
  sup_health="$(curl -fsS --max-time 3 "${SUP}/health" 2>/dev/null || true)"
  [[ -n "$sup_health" ]] && break
  sleep 1
done
[[ -n "$sup_health" ]] || fail "the supervisor's /health never answered"
echo "   $sup_health"
r="$(api GET /slots "" "")"
[[ "$(status_of "$r")" == 401 ]] || fail "GET /slots without the bearer answered $(status_of "$r"), not 401"
echo "   GET /slots without the bearer: 401"

step "3. PUT /slots/a and /slots/b through the real API, with the stub model"
r="$(api PUT /slots/a "$(slot_body "$MATTER_A" "$SECRET_A")")"
[[ "$(status_of "$r")" == 200 ]] || fail "PUT /slots/a answered: $r"
py '(d["started"], d["slot"]["port"], d["slot"]["unixUser"])' "$(body_of "$r")" | grep -qx "(True, 8800, 'm_a')" \
  || fail "PUT /slots/a: $r"
r="$(api PUT /slots/b "$(slot_body "$MATTER_B" "$SECRET_B")")"
[[ "$(status_of "$r")" == 200 ]] || fail "PUT /slots/b answered: $r"
py '(d["started"], d["slot"]["port"], d["slot"]["unixUser"])' "$(body_of "$r")" | grep -qx "(True, 8801, 'm_b')" \
  || fail "PUT /slots/b: $r"
echo "   a: port 8800 user m_a; b: port 8801 user m_b; both started"
wait_health "$PORT_A" "$MATTER_A"
wait_health "$PORT_B" "$MATTER_B"
sleep 5
echo "   idle memory after start: a $(memory_mib a); b $(memory_mib b)"
IDLE_START_A="$(memory_mib a)"; IDLE_START_B="$(memory_mib b)"

step "4. systemd-analyze verify litco-agent@a.service"
in_box systemd-analyze verify litco-agent@a.service || fail "litco-agent@a.service does not verify"

step "5. a turn on each slot; B's matter sent to slot a is refused"
sse="$(turn "$PORT_A" "$SECRET_A" "$MATTER_A" "smoke-a-1" "Say hello to the team.")" || fail "turn on slot a failed"
assert_final "$sse" "slot a"
sse="$(turn "$PORT_B" "$SECRET_B" "$MATTER_B" "smoke-b-1" "Say hello to the team.")" || fail "turn on slot b failed"
assert_final "$sse" "slot b"
r="$(turn_status "$PORT_A" "$SECRET_A" "$MATTER_B" "smoke-x-1" "Wrong matter.")"
[[ "$(status_of "$r")" == 403 && "$(py 'd["error"]["code"]' "$(body_of "$r")")" == matter_mismatch ]] \
  || fail "B's matter on slot a answered: $r"
echo "   B's matter on slot a: 403 matter_mismatch"
sleep 5
IDLE_TURN_A="$(memory_mib a)"; IDLE_TURN_B="$(memory_mib b)"
echo "   idle memory after one turn: a ${IDLE_TURN_A}; b ${IDLE_TURN_B}"

step "6. litco-slot-probe a b and b a"
for pair in "a b" "b a"; do
  # shellcheck disable=SC2086
  if ! out="$(docker exec "$NAME" litco-slot-probe $pair 2>"$TMPD/probe.err")"; then
    cat "$TMPD/probe.err" >&2
    fail "litco-slot-probe $pair: ${out:-no output}"
  fi
  sed 's/^/   /' "$TMPD/probe.err"
  echo "   litco-slot-probe $pair: $out"
  [[ "$(py 'sorted(set(d.values()))' "$out")" == "['denied']" ]] || fail "litco-slot-probe $pair: $out"
done

# Control: from outside any slot the metadata address must answer (here the
# container's network refuses the connection; on a droplet it serves user-data).
# Then the slot's "timed out" is the IPAddressDeny drop, not an absent route.
control="$(docker exec -i "$NAME" python3 - <<'PY'
import socket
s = socket.socket()
s.settimeout(3)
try:
    s.connect(("169.254.169.254", 80))
    print("connected")
except OSError as exc:
    print(type(exc).__name__)
PY
)"
echo "   control, outside any slot: 169.254.169.254:80 -> ${control}"
[[ "$control" != TimeoutError && "$control" != timeout ]] \
  || fail "the metadata address times out outside the slots too, so the probe's metadata verdict proves nothing"

step "7. drain under a held turn on a"
in_box sh -c 'echo kept > /srv/litco/m/a/matter/smoke-marker && chown m_a:m_a /srv/litco/m/a/matter/smoke-marker'
turn "$PORT_A" "$SECRET_A" "$MATTER_A" "smoke-a-slow" "Hold this turn open: SMOKE-SLOW-30" > "$TMPD/held.sse" 2>&1 &
HELD=$!
for _ in $(seq 1 60); do
  [[ "$(py 'd["activeTurns"]' "$(health "$PORT_A" || echo '{"activeTurns": 0}')")" == 1 ]] && break
  sleep 0.5
done
[[ "$(py 'd["activeTurns"]' "$(health "$PORT_A")")" == 1 ]] || fail "the held turn never started on a"
r="$(api POST /slots/a/stop)"
[[ "$(status_of "$r")" == 202 && "$(py 'd["slot"]["state"]' "$(body_of "$r")")" == stopping ]] \
  || fail "POST /slots/a/stop answered: $r"
echo "   POST /slots/a/stop: 202 stopping"
for _ in $(seq 1 40); do
  [[ "$(py 'd["draining"]' "$(health "$PORT_A" || echo '{"draining": false}')")" == True ]] && break
  sleep 0.5
done
r="$(turn_status "$PORT_A" "$SECRET_A" "$MATTER_A" "smoke-a-2" "A new turn while draining.")"
[[ "$(status_of "$r")" == 503 && "$(py 'd["error"]["code"]' "$(body_of "$r")")" == draining ]] \
  || fail "a new turn while a drains answered: $r"
echo "   a new turn while a drains: 503 draining"
wait "$HELD" || fail "the held turn's stream failed: $(tail -5 "$TMPD/held.sse")"
assert_final "$(cat "$TMPD/held.sse")" "held turn on a"
wait_slot_state a stopped 90
in_box test ! -e /run/litco-agent/a.env || fail "a's env file survived the stop"
in_box journalctl -u litco-agent@a --no-pager | grep -q "litco-agent-drain" || fail "the drain did not run on a"
health "$PORT_B" >/dev/null || fail "slot b stopped answering while a drained"
echo "   a stopped, env file gone, drain logged; b still answers"

step "8. PUT /slots/a again: same port, same user, home intact"
r="$(api PUT /slots/a "$(slot_body "$MATTER_A" "$SECRET_A")")"
py '(d["created"], d["userCreated"], d["started"], d["slot"]["port"], d["slot"]["unixUser"])' "$(body_of "$r")" \
  | grep -qx "(False, False, True, 8800, 'm_a')" || fail "PUT /slots/a again: $r"
wait_health "$PORT_A" "$MATTER_A"
[[ "$(in_box cat /srv/litco/m/a/matter/smoke-marker)" == kept ]] || fail "a's home lost its files"
echo "   a is back on 8800 as m_a with its home"

step "9. DELETE /slots/b"
r="$(api DELETE /slots/b)"
[[ "$(status_of "$r")" == 202 ]] || fail "DELETE /slots/b answered: $r"
r2="$(api PUT /slots/b "$(slot_body "$MATTER_B" "$SECRET_B")")"
[[ "$(status_of "$r2")" == 409 ]] || echo "   (the removal finished before the PUT; 409 not observed: $(status_of "$r2"))"
wait_slot_state b absent 120
in_box getent passwd m_b >/dev/null && fail "user m_b still exists"
in_box test ! -e /srv/litco/m/b || fail "/srv/litco/m/b still exists"
in_box test ! -e /run/litco-agent/b.env || fail "b's env file survived"
in_box python3 -c 'import json, sys; sys.exit("b" in json.load(open("/var/lib/litco-supervisor/slots.json"))["slots"])' \
  || fail "the registry still lists b"
r="$(api PUT /slots/c "$(slot_body "$MATTER_C" "$SECRET_C")")"
[[ "$(py 'd["slot"]["port"]' "$(body_of "$r")")" == 8801 ]] || fail "the next new slot did not get port 8801: $r"
echo "   b removed (user, home, env file, registry entry); the next new slot c got port 8801"
api DELETE /slots/c >/dev/null
wait_slot_state c absent 120

step "10. idle memory per slot (MemoryCurrent)"
echo "   after start:    a ${IDLE_START_A}; b ${IDLE_START_B}"
echo "   after one turn: a ${IDLE_TURN_A}; b ${IDLE_TURN_B}"
sleep 5
echo "   a after restart: $(memory_mib a)"

step "no secret in the supervisor's journal"
if in_box journalctl -u litco-supervisor --no-pager | grep -q -e "$SUP_SECRET" -e "$SECRET_A" -e "$SECRET_B" -e "$SECRET_C" -e "lkm_smoke"; then
  fail "a secret reached the supervisor's journal"
fi

rm -rf "$TMPD"
echo "SMOKE OK"
