#!/usr/bin/env bash
# Live-daemon repro for duplicate jail mounts (t_44fdd37a): one identity, the reuse fingerprint,
# names the container (hermes-<fp12>). Touches only containers it created that mount a path under
# its own throwaway root. Exit: 0 pass, 1 fail, 77 skip.
#   A: a respawn of one configuration attaches to the container under its canonical name.
#   B: another task bucket sharing a running jail's path RW is refused, naming the holder, in
#      either order (default holds and forge is refused; in a fresh root, the reverse).
#   C: a same-bucket spawn whose volumes diverge while sharing the jail RW is refused.
#   D: per-task buckets sharing only sandbox dirs coexist.
#   E: daemon-restart ordering — with the default jail stopped, forge comes up first; starting the
#      stopped default jail beside it is refused, never a second holder of the path.
#   E2: exec recovery inside a live process restarts its own container by name.
#   F: two envs share one container with no host volumes; after an out-of-band `docker rm` the
#      first recreates it under the canonical name and the second's recovery attaches to it.
#   G': a respawn that adds a named volume is a new fingerprint, so a new name: the stale container
#      is neither adopted nor removed (it is the orphan reaper's once it exits).
#   H: two concurrent spawns of one configuration converge on exactly one container, and both
#      processes can exec in it. Exactly one racer must log the sibling attach, or the spawns
#      serialized and the race was never exercised: retried, then a failure, never a pass.
set -u
cd "$(dirname "$0")/.."
PY=${HERMES_PYTHON:-.venv/bin/python}
export IMAGE=bash:5  # alpine + bash: execute() runs every command through bash (E2, F, H)

skip() { echo "SKIP: $*"; exit 77; }
fail() { echo "FAIL: $*"; exit 1; }

docker info >/dev/null 2>&1 || skip "no daemon/image"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull -q "$IMAGE" >/dev/null 2>&1 \
  || skip "no daemon/image"

T=$(realpath "$(mktemp -d)")
GVOL=jaildup_g_$(basename "$T")  # scenario G's named volume, unique to this run
A=$T/jailA B=$T/jailB C=$T/jailC E=$T/jailE
mkdir -p "$A" "$B" "$C" "$E" "$T/tmp"
export CREATED_IDS=$T/created.ids
: >"$CREATED_IDS"

cleanup() {
  local rc=$? id
  while read -r id; do
    docker inspect --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$id" 2>/dev/null \
      | grep -q "^$T/" && docker rm -f "$id" >/dev/null
  done <"$CREATED_IDS"
  docker volume rm "$GVOL" >/dev/null 2>&1
  # The containers ran as root and wrote into their bind-mounted sandbox dirs.
  rm -rf "$T" 2>/dev/null || { docker run --rm -v "$T:/t" "$IMAGE" find /t -mindepth 1 -delete; rm -rf "$T"; }
  exit "$rc"
}
trap cleanup EXIT

# HERMES_HOME is a fingerprint input, so every name this run derives is unique to $T.
export HERMES_HOME=$T/home TERMINAL_SANDBOX_DIR=$T/sandboxes
# Python's tempdir must not contain the jails: process-tempdir mounts are volatile by design
# and carry no jail identity, which would hide every mount under $T from the guard.
export TMPDIR=$T/tmp

# spawn <task_id> <volume>...: one process per spawn (the cross-process case), printing the
# container id or "REFUSED <message>". The id is recorded at once so a later crash cannot leak it.
spawn() {
  "$PY" -c '
import os, sys
from tools.environments.docker import DockerEnvironment
try:
    env = DockerEnvironment(image=os.environ["IMAGE"], cwd="/", task_id=sys.argv[1],
                            volumes=sys.argv[2:], persistent_filesystem=True)
except RuntimeError as e:
    print("REFUSED", e)
else:
    with open(os.environ["CREATED_IDS"], "a") as f:
        f.write(env._container_id + "\n")
    print(env._container_id)' "$@"
}

same_container() { [ -n "$1" ] && [ "$1" = "$2" ]; }

running() { [ "$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

canonical_name() {  # prints the container's name; non-zero (reason on stderr) unless it is hermes-<fp12>
  local name
  name=$(docker inspect --format '{{.Name}}' "$1") || { echo "cannot inspect $1" >&2; return 1; }
  [[ $name =~ ^/hermes-[0-9a-f]{12}$ ]] || { echo "container $1 is named $name, not hermes-<fp12>" >&2; return 1; }
  echo "${name#/}"
}

holders() {  # hermes containers bind-mounting $1: running ones, or all with -a as $2
  local n=0 id state=(--filter status=running)
  [ "${2:-}" = -a ] && state=(-a)
  for id in $(docker ps -q --filter label=hermes-agent=1 "${state[@]}"); do
    docker inspect --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$id" | grep -qxF "$1" && n=$((n + 1))
  done
  echo "$n"
}

assert_single_jail() {
  local jail n
  for jail in "$@"; do
    n=$(holders "$jail")
    [ "$n" = 1 ] || fail "$n running containers mount $jail (want 1)"
  done
  echo "PASS single-jail-invariant"
}

expect_refusal() {  # expect_refusal <scenario> <holder id> <spawn output>
  case "$3" in
    "REFUSED "*"already bind-mounted read-write"*"${2:0:12}"*) echo "PASS $1" ;;
    *) fail "$1: expected a refusal naming ${2:0:12}, got: $3" ;;
  esac
}

# A: one configuration, two processes, one container under the canonical name.
id1=$(spawn default "$A:/home/bot") || fail "scenario A default spawn crashed"
case "$id1" in REFUSED*) fail "scenario A default spawn refused: $id1" ;; esac
canonical_name "$id1" >/dev/null || fail "scenario A container is not canonically named"
id2=$(spawn default "$A:/home/bot") || fail "scenario A respawn crashed"
same_container "$id1" "$id2" || fail "respawn got $id2, not the named container $id1"
echo "PASS respawn-attaches-by-name"
assert_single_jail "$A"

# B: the live leak shape — "default" holds the jail, a "profile:forge" spawn is another identity.
out=$(spawn profile:forge "$A:/home/bot") || fail "scenario B forge spawn crashed: $out"
expect_refusal refuse-other-bucket-on-held-jail "$id1" "$out"
# Reverse order in a fresh sandbox root: forge holds the jail, default is refused.
export TERMINAL_SANDBOX_DIR=$T/sbxT
idF=$(spawn profile:forge "$C:/home/bot") || fail "scenario B forge-first spawn crashed"
out=$(spawn default "$C:/home/bot") || fail "scenario B default spawn crashed: $out"
expect_refusal refuse-default-on-forge-held-jail "$idF" "$out"
assert_single_jail "$A" "$C"

# C: same bucket, volumes diverge while sharing jail A read-write: a new fingerprint, refused.
export TERMINAL_SANDBOX_DIR=$T/sandboxes
out=$(spawn default "$A:/home/bot" "$B:/extra") || fail "scenario C spawn crashed: $out"
expect_refusal refuse-divergent-volumes "$id1" "$out"
[ "$(holders "$B" -a)" = 0 ] || fail "refused spawn left a container on $B"
assert_single_jail "$A"

# D: rollouts under one profile, no volumes — only per-task sandbox dirs, which coexist.
export TERMINAL_SANDBOX_DIR=$T/sbxD
id3=$(spawn rollout:one) || fail "scenario D rollout:one spawn crashed"
id4=$(spawn rollout:two) || fail "scenario D rollout:two spawn crashed"
[ "$id3" != "$id4" ] && running "$id3" && running "$id4" \
  || fail "rollouts must get running containers of their own, got $id3 / $id4"
echo "PASS rollout-isolation"

# E: daemon restart leaves the default jail stopped and forge recovers first (nothing running holds
# the path, so it starts). Starting the stopped default jail beside it would double-mount the path.
export TERMINAL_SANDBOX_DIR=$T/sbxE
idX=$(spawn default "$E:/home/bot") || fail "scenario E default spawn crashed"
docker stop "$idX" >/dev/null || fail "could not stop $idX"
idY=$(spawn profile:forge "$E:/home/bot") || fail "scenario E forge spawn crashed"
case "$idY" in REFUSED*) fail "forge must start fresh with nothing running, got: $idY" ;; esac
same_container "$idX" "$idY" && fail "forge started the stopped default jail $idX"
running "$idY" || fail "forge container $idY not running"
out=$(spawn default "$E:/home/bot") || fail "scenario E default respawn crashed: $out"
expect_refusal restart-order-refuses-second-holder "$idY" "$out"
running "$idX" && fail "the refused default jail $idX was started"
assert_single_jail "$E"

# E2: the daemon restarts under a live process: recovery restarts the container by name, not anew.
export TERMINAL_SANDBOX_DIR=$T/sbxE2
out=$("$PY" -c '
import os, subprocess
from tools.environments.docker import DockerEnvironment
env = DockerEnvironment(image=os.environ["IMAGE"], cwd="/", task_id="default", volumes=[], persistent_filesystem=True)
with open(os.environ["CREATED_IDS"], "a") as f:
    f.write(env._container_id + "\n")
before = env._container_id
subprocess.run(["docker", "stop", before], check=True, capture_output=True)
result = env.execute("echo alive")
print(before, env._container_id, "alive" in result.get("output", ""))') \
  || fail "scenario E2 crashed: $out"
read -r before after alive <<<"$out"
[ "$alive" = True ] || fail "exec after recovery did not run: $out"
[ "$before" = "$after" ] || fail "recovery switched $before to $after instead of restarting it"
echo "PASS recovery-restarts-named-container"
assert_single_jail "$A" "$C" "$E"

# F: a refused recovery would leave B's every later exec asserting "Container not started".
export TERMINAL_SANDBOX_DIR=$T/sbxF
out=$("$PY" -c '
import os, subprocess
from tools.environments.docker import DockerEnvironment
kw = dict(image=os.environ["IMAGE"], cwd="/", task_id="default", volumes=[], persistent_filesystem=True)
a = DockerEnvironment(**kw)
with open(os.environ["CREATED_IDS"], "a") as f:
    f.write(a._container_id + "\n")
b = DockerEnvironment(**kw)
assert a._container_id == b._container_id, (a._container_id, b._container_id)
subprocess.run(["docker", "rm", "-f", a._container_id], check=True, capture_output=True)
ra = a.execute("echo alive")
rb = b.execute("echo alive")
print(a._container_id, b._container_id, "alive" in ra.get("output", ""), "alive" in rb.get("output", ""))') \
  || fail "scenario F crashed: $out"
read -r idA idB aliveA aliveB <<<"$out"
printf '%s\n' "$idA" "$idB" >>"$CREATED_IDS"
[ "$aliveA" = True ] && [ "$aliveB" = True ] || fail "exec after recovery did not run in both envs: $out"
same_container "$idA" "$idB" || fail "B recovered into $idB, not A's recreated container $idA: $out"
canonical_name "$idA" >/dev/null || fail "scenario F recreated container is not canonically named"
echo "PASS no-host-volume-twin-recovers-by-name"
assert_single_jail "$T/sbxF/docker/default/home" "$T/sbxF/docker/default/workspace"

# G': a named volume is invisible to the bind model; only the fingerprint sees the drift. The new
# configuration gets its own name; the stale container is left running, neither adopted nor removed.
export TERMINAL_SANDBOX_DIR=$T/sbxG
idOld=$(spawn default) || fail "scenario G' default spawn crashed"
idG=$(spawn default "$GVOL:/data") || fail "scenario G' drifted spawn crashed: $idG"
case "$idG" in REFUSED*) fail "named-volume drift shares only sandbox dirs, got: $idG" ;; esac
[ "$idG" != "$idOld" ] || fail "drifted config adopted the stale container $idOld"
nameG=$(canonical_name "$idG") || fail "scenario G' drifted container is not canonically named"
nameOld=$(canonical_name "$idOld") || fail "scenario G' stale container is not canonically named"
[ "$nameG" != "$nameOld" ] || fail "drifted config kept the stale name $nameOld"
running "$idOld" || fail "drift spawn stopped or removed the stale container $idOld"
docker inspect --format '{{range .Mounts}}{{.Type}}:{{.Destination}}{{"\n"}}{{end}}' "$idG" | grep -qx volume:/data \
  || fail "new container $idG does not carry the named volume"
echo "PASS fingerprint-drift-gets-new-name"

# H: two processes of one configuration spawn at the same instant (both probe before either runs);
# the name makes `docker run` the atomic duplicate test, so the loser attaches to the winner.
race() {  # race <attempt> <racer>
  "$PY" -c '
import logging, os, sys, time
from tools.environments.docker import DockerEnvironment
logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # the loser logs its sibling attach
time.sleep(max(0.0, int(os.environ["GO_AT"]) - time.time()))
env = DockerEnvironment(image=os.environ["IMAGE"], cwd="/", task_id="default", volumes=[], persistent_filesystem=True)
with open(os.environ["CREATED_IDS"], "a") as f:
    f.write(env._container_id + "\n")
print(env._container_id, "alive" in env.execute("echo alive").get("output", ""))' >"$T/race.$1.$2" 2>"$T/race.$1.$2.err"
}
raced=
for attempt in 1 2 3; do
  # A fresh sandbox root per attempt is a fresh fingerprint, so a fresh name to race for.
  export TERMINAL_SANDBOX_DIR=$T/sbxH$attempt GO_AT=$(($(date +%s) + 3))
  race "$attempt" 1 & pid1=$!
  race "$attempt" 2 & pid2=$!
  wait "$pid1" || fail "scenario H racer 1 crashed: $(tail -5 "$T/race.$attempt.1.err")"
  wait "$pid2" || fail "scenario H racer 2 crashed: $(tail -5 "$T/race.$attempt.2.err")"
  read -r idH1 aliveH1 <"$T/race.$attempt.1"
  read -r idH2 aliveH2 <"$T/race.$attempt.2"
  same_container "$idH1" "$idH2" || fail "racers got $idH1 / $idH2, not one container"
  [ "$aliveH1" = True ] && [ "$aliveH2" = True ] || fail "exec failed in a racer: $aliveH1 / $aliveH2"
  n=$(holders "$TERMINAL_SANDBOX_DIR/docker/default/home" -a)
  [ "$n" = 1 ] || fail "$n containers (any state) mount the raced sandbox (want 1)"
  attached=$(($(grep -l "was taken by a sibling process" "$T/race.$attempt".[12].err | wc -l)))
  case "$attached" in
    1) raced=$attempt; break ;;
    0) echo "scenario H attempt $attempt: spawns serialized (no sibling attach logged), retrying" ;;
    *) fail "scenario H: both racers logged the sibling attach" ;;
  esac
done
[ -n "$raced" ] || { echo "FAIL scenario H: race not exercised (spawns serialized)"; exit 1; }
echo "PASS concurrent-spawn-converges (race fired on attempt $raced)"

echo "ALL PASS"
