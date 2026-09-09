#!/usr/bin/env bash
# Start/stop the mock inference server and point the app at it as a REAL
# configured provider.
#
# Why: the desktop app boots to the onboarding overlay when no provider is
# configured - a fullscreen div that intercepts every click, which killed
# the app-update legs. Seeding a fake key makes the overlay vanish but
# leaves a lying app; this makes the app GENUINELY configured (config.yaml
# provider + key env, exactly what tests-js/scripts/mock-server.ts's
# dev:mock flow writes) with a real, chat-capable backend.
#
# Usage (sourced from a driver):
#   mock_start <workroot>      start, write config into $HERMES_HOME
#   mock_stop                  kill the background server
#
# Requires: ASSETS (e2e-assets dir), LOG_DIR, HERMES_HOME, ok/fail helpers.

MOCK_PIDFILE=""
MOCK_URLFILE=""

mock_start() {
  local workroot="${1:?mock_start needs a workroot}"
  MOCK_PIDFILE="$workroot/mock.pid"
  MOCK_URLFILE="$workroot/mock.url"
  rm -f "$MOCK_PIDFILE" "$MOCK_URLFILE"

  # Bare `node file.ts` type-stripping works on node >=22.18 (the images
  # ship 22.22+; the installed managed node is >=26) - same contract as
  # the repo's own `dev:mock` script.
  node "$ASSETS/mock-provider.mjs" "$MOCK_URLFILE" > "$LOG_DIR/mock.log" 2>&1 &
  echo $! > "$MOCK_PIDFILE"

  local _i
  for _i in 1 2 3 4 5 6 7 8 9 10; do
    [ -s "$MOCK_URLFILE" ] && break
    sleep 0.2
  done
  if [ ! -s "$MOCK_URLFILE" ]; then
    log_group "mock server transcript" "$LOG_DIR/mock.log"
    fail "mock inference server did not come up; transcript above"
  fi
  local url
  url="$(cat "$MOCK_URLFILE")"
  ok "mock inference server: $url"

  # The provider config, byte-compatible with writeMockConfig() in
  # tests-js/scripts/mock-server.ts.
  cat > "$HERMES_HOME/config.yaml" <<EOF
model:
  default: mock-model
  provider: mock
providers:
  mock:
    api: $url/v1
    name: Mock
    api_mode: chat_completions
    key_env: MOCK_API_KEY
    models:
      mock-model: {}
    context_length: 4096
EOF
  printf 'MOCK_API_KEY=e2e-mock-key\n' >> "$HERMES_HOME/.env"
  ok "provider 'mock' configured in $HERMES_HOME (api $url/v1)"
}

mock_stop() {
  if [ -n "$MOCK_PIDFILE" ] && [ -f "$MOCK_PIDFILE" ]; then
    kill "$(cat "$MOCK_PIDFILE")" 2>/dev/null || true
    rm -f "$MOCK_PIDFILE"
  fi
}
