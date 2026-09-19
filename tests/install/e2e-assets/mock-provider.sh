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

  # Idempotent: a mock that is still live from earlier in this run is REUSED. Its
  # URL is what the install is already configured with, so re-writing the config
  # below leaves .env byte-identical -- a NEW instance would take a new port, and
  # the user-state verifier would report that as the upgrade rewriting .env
  # (an equal-size OPENAI_BASE_URL swap, invisible in the file hash).
  if [ -s "$MOCK_URLFILE" ] && [ -s "$MOCK_PIDFILE" ] \
      && kill -0 "$(cat "$MOCK_PIDFILE" 2>/dev/null)" 2>/dev/null; then
    local live_url
    live_url="$(cat "$MOCK_URLFILE")"
    export HERMES_E2E_MOCK_URL="$live_url"
    ok "mock inference server already live: $live_url"
    # The config write is NOT optional: this is what re-points the app at a live
    # endpoint, and callers that only want that (the app-update flow, whose app
    # reads config.yaml/.env rather than the env var) depend on it.
    mock_configure_provider "$live_url"
    return 0
  fi

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
  export HERMES_E2E_MOCK_URL="$url"
  ok "mock inference server: $url"
  mock_configure_provider "$url"
}

mock_configure_provider() {
  local url="${1:?mock_configure_provider needs a url}"
  node "$ASSETS/../../../tests-js/scripts/mock-provider-config.ts" "$HERMES_HOME" "$url" || fail "mock provider config failed"
  ok "configured in $HERMES_HOME as an OpenAI-compatible endpoint (api $url/v1)"
}

mock_stop() {
  if [ -n "$MOCK_PIDFILE" ] && [ -f "$MOCK_PIDFILE" ]; then
    kill "$(cat "$MOCK_PIDFILE")" 2>/dev/null || true
    rm -f "$MOCK_PIDFILE"
  fi
}
