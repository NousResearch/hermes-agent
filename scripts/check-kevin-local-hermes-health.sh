#!/bin/zsh
set -u

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
COMPOSE_FILE="$PROJECT_DIR/compose.hermes.local.yml"
DASHBOARD_URL="${HERMES_DASHBOARD_URL:-http://127.0.0.1:9119/chat}"
CHECK_GIT_SAFETY="${CHECK_GIT_SAFETY:-1}"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

exit_code=0

info() {
  print -r -- "[INFO] $*"
}

ok() {
  print -r -- "[OK] $*"
}

warn() {
  print -r -- "[WARN] $*"
}

fail() {
  print -r -- "[FAIL] $*" >&2
  exit_code=1
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    fail "missing command: $cmd"
    return 1
  fi
  return 0
}

cd "$PROJECT_DIR" || {
  print -r -- "[FAIL] cannot cd to project: $PROJECT_DIR" >&2
  exit 1
}

info "project: $PROJECT_DIR"
info "compose: $COMPOSE_FILE"
info "dashboard: $DASHBOARD_URL"

[[ -f "$COMPOSE_FILE" ]] || fail "missing compose file: $COMPOSE_FILE"

require_command docker
require_command curl

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    ok "Docker daemon is reachable"
  else
    fail "Docker daemon is not reachable"
  fi
fi

if [[ -f "$COMPOSE_FILE" ]] && command -v docker >/dev/null 2>&1; then
  info "docker compose ps"
  if docker compose -f "$COMPOSE_FILE" ps; then
    ok "docker compose ps completed"
  else
    fail "docker compose ps failed"
  fi
fi

if command -v curl >/dev/null 2>&1; then
  http_code="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "$DASHBOARD_URL" 2>/dev/null || true)"
  if [[ "$http_code" == "200" ]]; then
    ok "dashboard HTTP returned 200"
  else
    fail "dashboard HTTP expected 200, got ${http_code:-none}"
  fi
fi

if command -v python3 >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
  ws_output="$(python3 - "$DASHBOARD_URL" <<'PY'
import asyncio
import re
import sys
import urllib.parse
import urllib.request

dashboard_url = sys.argv[1]

try:
    import websockets
except Exception:
    print("SKIP websockets package is not installed")
    sys.exit(10)

try:
    html = urllib.request.urlopen(dashboard_url, timeout=5).read().decode("utf-8", "replace")
except Exception as exc:
    print(f"FAIL dashboard fetch failed: {exc}")
    sys.exit(11)

token_match = re.search(r'window\.__HERMES_SESSION_TOKEN__\s*=\s*"([^"]+)"', html)
if not token_match:
    print("SKIP dashboard session token was not found")
    sys.exit(12)

parts = urllib.parse.urlsplit(dashboard_url)
scheme = "wss" if parts.scheme == "https" else "ws"
query = urllib.parse.urlencode({
    "token": token_match.group(1),
    "channel": "healthcheck",
})
ws_url = urllib.parse.urlunsplit((scheme, parts.netloc, "/api/pty", query, ""))

async def main() -> None:
    async with websockets.connect(ws_url, open_timeout=5) as ws:
        await asyncio.wait_for(ws.recv(), timeout=8)

try:
    asyncio.run(main())
except Exception as exc:
    print(f"FAIL websocket probe failed: {exc}")
    sys.exit(13)

print("OK websocket opened and received data")
PY
)"
  ws_status=$?
  case "$ws_status" in
    0)
      ok "$ws_output"
      ;;
    10|12)
      warn "$ws_output"
      ;;
    *)
      fail "$ws_output"
      ;;
  esac
else
  warn "python3 is missing; skipping WebSocket probe"
fi

if [[ "$CHECK_GIT_SAFETY" == "1" ]]; then
  if [[ -x "$PROJECT_DIR/scripts/check-git-remote-safety.sh" ]]; then
    info "scripts/check-git-remote-safety.sh"
    if "$PROJECT_DIR/scripts/check-git-remote-safety.sh"; then
      ok "git remote safety check passed"
    else
      fail "git remote safety check failed"
    fi
  else
    fail "missing executable git safety script"
  fi
else
  warn "git remote safety check skipped because CHECK_GIT_SAFETY=$CHECK_GIT_SAFETY"
fi

exit "$exit_code"
