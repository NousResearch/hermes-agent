#!/usr/bin/env bash
#
# smoke.sh: local container smoke test of the matter host.
#
# Builds deploy/host/smoke/Dockerfile (the same install-host.sh the droplet
# image runs, minus Tailscale and ufw), boots systemd in the container, writes
# a fake /etc/litco-agent/env the way cloud-init would, restarts
# litco-agent.service, and then:
#   1. waits for GET /health,
#   2. checks the rendered profile (config.yaml, SOUL.md) and that no secret
#      reached $HERMES_HOME,
#   3. posts one POST /turn and asserts the SSE stream ends with a `final`
#      frame carrying the stub model's answer. The model is a stub
#      OpenAI-compatible server inside the container, so the turn runs through
#      the real gateway, litco_turn platform, HermesTurnRunner and AIAgent,
#   4. stops the unit and checks the drain ran.
#
# Needs a running Docker daemon (Docker Desktop, colima, ...). Runs nothing
# against any cloud account.
#
# Usage: deploy/host/smoke.sh [--keep] [--no-build]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMAGE="litco-agent-host-smoke:local"
NAME="litco-agent-host-smoke"
PORT="${LITCO_SMOKE_PORT:-18765}"
SECRET="smoke-host-secret-not-real-0123456789abcdef"
KEEP=false
BUILD=true
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=true ;;
    --no-build) BUILD=false ;;
    *) echo "smoke: unknown argument $arg" >&2; exit 2 ;;
  esac
done

fail() { echo "SMOKE FAIL: $*" >&2; docker logs "$NAME" 2>&1 | tail -20 >&2 || true;
         docker exec "$NAME" journalctl -u litco-agent --no-pager -n 80 >&2 2>/dev/null || true; exit 1; }
step() { echo "== $*"; }

if ! docker info >/dev/null 2>&1; then
  echo "smoke: no Docker daemon reachable (docker info failed); start Docker or colima and rerun" >&2
  exit 3
fi

cleanup() { [[ "$KEEP" == true ]] || docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "${CTX:-}"; }
trap cleanup EXIT

if [[ "$BUILD" == true ]]; then
  step "export the working tree (tracked + untracked, minus ignored files)"
  CTX="$(mktemp -d)"
  mkdir -p "$CTX/src"
  (cd "$REPO" && git ls-files -co --exclude-standard -z \
    | python3 -c 'import os,sys; sys.stdout.buffer.write(b"".join(p+b"\0" for p in sys.stdin.buffer.read().split(b"\0") if p and os.path.lexists(p)))' \
    | tar --null -T - -cf - | tar -xf - -C "$CTX/src")
  cp "$HERE/smoke/Dockerfile" "$CTX/Dockerfile"
  step "docker build $IMAGE (this runs install-host.sh; the first build takes several minutes)"
  docker build -t "$IMAGE" "$CTX"
fi

step "boot systemd in the container"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --privileged --cgroupns=host \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw --tmpfs /run --tmpfs /run/lock \
  -p "127.0.0.1:${PORT}:8765" "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
  state="$(docker exec "$NAME" systemctl is-system-running 2>/dev/null || true)"
  [[ "$state" == running || "$state" == degraded ]] && break
  sleep 1
done
echo "   systemd: ${state:-unknown}"

step "systemd-analyze verify the unit (systemd 255, as on Ubuntu 24.04)"
docker exec "$NAME" systemd-analyze verify /etc/systemd/system/litco-agent.service || fail "unit does not verify"

step "unit is baked enabled and does not start without an env file"
[[ "$(docker exec "$NAME" systemctl is-enabled litco-agent)" == enabled ]] || fail "litco-agent not enabled"
docker exec "$NAME" systemctl is-active litco-agent >/dev/null 2>&1 && fail "litco-agent ran without /etc/litco-agent/env"

step "write /etc/litco-agent/env as cloud-init would (root:root 0600), restart the unit"
docker exec -i "$NAME" sh -c 'install -d -m 0755 /etc/litco-agent && cat > /etc/litco-agent/env && chown root:root /etc/litco-agent/env && chmod 600 /etc/litco-agent/env' < "$HERE/smoke/env.smoke"
docker exec "$NAME" install -d -o hermes -g hermes -m 0750 /home/hermes/matter
docker exec "$NAME" systemctl restart litco-agent

step "wait for GET /health"
health=""
for _ in $(seq 1 90); do
  health="$(curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)"
  [[ -n "$health" ]] && break
  sleep 2
done
[[ -n "$health" ]] || fail "/health never answered"
echo "   $health"
python3 -c 'import json,sys; h=json.loads(sys.argv[1]); assert h["ok"] and h["matterId"]=="smoke-matter-0001", h' "$health" \
  || fail "/health payload wrong"

step "profile rendered from non-secret env only"
docker exec "$NAME" test -f /home/hermes/.hermes/config.yaml || fail "config.yaml not rendered"
docker exec "$NAME" grep -q "smoke-matter-0001" /home/hermes/.hermes/SOUL.md || fail "SOUL.md not rendered"
docker exec "$NAME" test ! -e /home/hermes/.hermes/.env || fail "an .env file appeared in HERMES_HOME"
if docker exec "$NAME" grep -rqs -e "$SECRET" -e lkm_smoke_not_real /home/hermes/.hermes/config.yaml /home/hermes/.hermes/SOUL.md; then
  fail "a secret reached the rendered profile"
fi
[[ "$(docker exec "$NAME" stat -c '%U %a' /etc/litco-agent/env)" == "root 600" ]] || fail "env file not root 0600"
docker exec -u hermes "$NAME" cat /etc/litco-agent/env >/dev/null 2>&1 && fail "hermes user can read the env file"

step "POST /turn (stub model) and read the SSE stream"
sse="$(curl -fsS --max-time 180 -N -X POST "http://127.0.0.1:${PORT}/turn" \
  -H "X-Host-Secret: ${SECRET}" -H "Content-Type: application/json" \
  -d '{"matterId":"smoke-matter-0001","sessionId":"smoke-thread-1","text":"Say hello to the team.","channel":"web","kind":"channel"}')" \
  || fail "/turn request failed"
printf '%s\n' "$sse" | grep '^event:' | sort | uniq -c | sed 's/^/   /'
python3 - "$sse" <<'PY' || fail "SSE stream did not end with the expected final frame"
import json, sys
frames = [json.loads(line[6:]) for line in sys.argv[1].splitlines() if line.startswith("data: ")]
assert frames, "no frames"
assert frames[0]["type"] == "goal_accepted", frames[0]
final = frames[-1]
assert final["type"] == "final", final
assert "SMOKE-OK" in final["text"], final
print("   final.text:", final["text"][:120])
print("   final.usage:", final.get("usage"), "modelUsed:", final.get("modelUsed"))
PY

step "Chromium from the PM store renders a page as hermes"
# Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor, so
# Chromium's own sandbox cannot start; Hermes detects this and passes
# --no-sandbox (tools/browser_tool_session.py). The smoke does the same.
docker exec -u hermes -e HOME=/home/hermes "$NAME" sh -c \
  'timeout 60 /opt/litco-agent/tools/chromium-*/chrome-linux/chrome --headless=new --no-sandbox --disable-gpu --dump-dom "data:text/html,<p>browser-ok</p>" 2>/dev/null' \
  | grep -q browser-ok || fail "headless Chromium did not render"

step "stop drains and exits cleanly"
docker exec "$NAME" systemctl stop litco-agent
docker exec "$NAME" journalctl -u litco-agent --no-pager -n 50 | grep -q "litco-agent-drain" || fail "drain did not run on stop"

echo "SMOKE OK"
