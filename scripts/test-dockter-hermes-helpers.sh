#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/dockter-hermes-test.$$"
LOG_FILE="${TMP_DIR}/commands.log"
REAL_DOCKER_BIN="$(command -v docker 2>/dev/null || true)"
REAL_HOME="$HOME"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TMP_DIR/bin" "$TMP_DIR/home/.hermes/dockter-hermes" \
  "$TMP_DIR/home/.hermes/workspace" "$TMP_DIR/hermes-agent"
touch "$LOG_FILE"
touch "$TMP_DIR/hermes-agent/Dockerfile"
touch "$TMP_DIR/hermes-agent/docker-compose.yml"
touch "$TMP_DIR/hermes-agent/docker-compose.windows.yml"
touch "$TMP_DIR/hermes-agent/docker-compose.extra.yml"
touch "$TMP_DIR/home/.hermes/dockter-hermes/docker-compose.override.yml"

cat > "$TMP_DIR/home/.hermes/config.yaml" <<'YAML'
model:
  provider: test
YAML

cat > "$TMP_DIR/home/.hermes/.env" <<'ENV'
API_SERVER_KEY=secret-value
OPENROUTER_API_KEY=another-secret
ENV

cat > "$TMP_DIR/bin/docker" <<'SH'
#!/usr/bin/env bash
printf 'docker %q' "$1" >> "$DOCKTER_TEST_LOG"
shift || true
for arg in "$@"; do
  printf ' %q' "$arg" >> "$DOCKTER_TEST_LOG"
done
printf '\n' >> "$DOCKTER_TEST_LOG"
if [[ -n "${DOCKTER_TEST_FAIL_MATCH:-}" && "$*" == *"$DOCKTER_TEST_FAIL_MATCH"* ]]; then
  printf 'docker stub failure\n' >&2
  exit "${DOCKTER_TEST_FAIL_STATUS:-42}"
fi
case "$*" in
  *"gateway status --deep"*)
    printf 'gateway ok\n'
    ;;
  *"--format"*)
    printf 'NAMES\tIMAGE\tSTATUS\tPORTS\nhermes\thermes-agent\tUp\t9119\n'
    ;;
  *)
    printf 'docker stub ok\n'
    ;;
esac
SH
chmod +x "$TMP_DIR/bin/docker"

cat > "$TMP_DIR/bin/git" <<'SH'
#!/usr/bin/env bash
printf 'git' >> "$DOCKTER_TEST_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$DOCKTER_TEST_LOG"
done
printf '\n' >> "$DOCKTER_TEST_LOG"
printf 'git stub ok\n'
SH
chmod +x "$TMP_DIR/bin/git"

cat > "$TMP_DIR/bin/curl" <<'SH'
#!/usr/bin/env bash
printf 'curl' >> "$DOCKTER_TEST_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$DOCKTER_TEST_LOG"
done
printf '\n' >> "$DOCKTER_TEST_LOG"
printf '{"status":"ok"}\n'
SH
chmod +x "$TMP_DIR/bin/curl"

cat > "$TMP_DIR/bin/open" <<'SH'
#!/usr/bin/env bash
printf 'open' >> "$DOCKTER_TEST_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$DOCKTER_TEST_LOG"
done
printf '\n' >> "$DOCKTER_TEST_LOG"
printf 'open stub ok\n'
SH
chmod +x "$TMP_DIR/bin/open"

cat > "$TMP_DIR/bin/xdg-open" <<'SH'
#!/usr/bin/env bash
printf 'xdg-open' >> "$DOCKTER_TEST_LOG"
for arg in "$@"; do
  printf ' %q' "$arg" >> "$DOCKTER_TEST_LOG"
done
printf '\n' >> "$DOCKTER_TEST_LOG"
printf 'xdg-open stub ok\n'
SH
chmod +x "$TMP_DIR/bin/xdg-open"

export HOME="$TMP_DIR/home"
export PATH="$TMP_DIR/bin:$PATH"
export DOCKTER_TEST_LOG="$LOG_FILE"
export DOCKTER_HERMES_DIR="$TMP_DIR/hermes-agent"
export HERMES_HOME="$TMP_DIR/home/.hermes"
export DOCKTER_HERMES_DOCKER_BIN="$TMP_DIR/bin/docker"

# shellcheck source=/dev/null
source "$ROOT_DIR/dockter-hermes-helpers.sh"

command_line() {
  local command="$1"
  shift
  printf '%s' "$command"
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
}

compose_line() {
  command_line docker compose \
    -f "$TMP_DIR/hermes-agent/docker-compose.yml" \
    -f "$TMP_DIR/hermes-agent/docker-compose.extra.yml" \
    -f "$TMP_DIR/home/.hermes/dockter-hermes/docker-compose.override.yml" \
    "$@"
}

windows_compose_line() {
  command_line docker compose \
    -f "$TMP_DIR/hermes-agent/docker-compose.windows.yml" \
    -f "$TMP_DIR/hermes-agent/docker-compose.extra.yml" \
    -f "$TMP_DIR/home/.hermes/dockter-hermes/docker-compose.override.yml" \
    "$@"
}

assert_log() {
  local name="$1"
  local expected="$2"
  local actual
  actual=$(< "$LOG_FILE")
  if [[ "$actual" != "$expected" ]]; then
    printf 'Unexpected command log for %s\nExpected:\n%s\nActual:\n%s\n' \
      "$name" "$expected" "$actual" >&2
    return 1
  fi
}

run_case() {
  local name="$1"
  shift
  printf 'TEST %s\n' "$name"
  : > "$LOG_FILE"
  ( "$@" ) >/dev/null
  assert_log "$name" ""
}

run_logged_case() {
  local name="$1"
  local expected="$2"
  shift 2
  printf 'TEST %s\n' "$name"
  : > "$LOG_FILE"
  ( "$@" ) >/dev/null
  assert_log "$name" "$expected"
}

run_failure_case() {
  local name="$1"
  shift
  printf 'TEST %s\n' "$name"
  : > "$LOG_FILE"
  if ( "$@" ) >/dev/null 2>&1; then
    printf 'Expected failure but command passed: %s\n' "$name" >&2
    return 1
  fi
  assert_log "$name" ""
}

run_update_failure_case() {
  local expected="$1"
  local output status
  printf 'TEST update-recreate-failure\n'
  : > "$LOG_FILE"
  export DOCKTER_TEST_FAIL_MATCH='up -d gateway dashboard'
  export DOCKTER_TEST_FAIL_STATUS=42
  set +e
  output=$(dockter-hermes-update 2>&1)
  status=$?
  set -e
  unset DOCKTER_TEST_FAIL_MATCH DOCKTER_TEST_FAIL_STATUS

  if [[ $status -ne 42 ]]; then
    printf 'Expected recreation status 42, got %s\n' "$status" >&2
    return 1
  fi
  if [[ "$output" == *"Update complete."* ]]; then
    printf 'Failed update reported success:\n%s\n' "$output" >&2
    return 1
  fi
  if [[ "$output" != *"Container recreation failed"* ]]; then
    printf 'Failed update did not report recreation failure:\n%s\n' "$output" >&2
    return 1
  fi
  assert_log update-recreate-failure "$expected"
}

run_dashboard_failure_case() {
  local expected="$1"
  local output status
  printf 'TEST dashboard-not-ready\n'
  : > "$LOG_FILE"
  set +e
  output=$(run_windows_dashboard_not_ready 2>&1)
  status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    printf 'Expected dashboard readiness failure\n' >&2
    return 1
  fi
  if [[ "$output" == *"Opening:"* ]]; then
    printf 'Unready dashboard claimed it was opening:\n%s\n' "$output" >&2
    return 1
  fi
  if [[ "$output" != *"dockter-hermes-cli dashboard register"* ]]; then
    printf 'Windows dashboard failure omitted auth guidance:\n%s\n' "$output" >&2
    return 1
  fi
  assert_log dashboard-not-ready "$expected"
}

run_clean() {
  printf 'y\n' | dockter-hermes-clean
}

run_dashboard() {
  DOCKTER_HERMES_DASHBOARD_PORT=19119 dockter-hermes-dashboard
}

run_windows_start() {
  _dockter_hermes_is_native_windows() { return 0; }
  dockter-hermes-start
}

run_windows_dashboard_not_ready() {
  _dockter_hermes_is_native_windows() { return 0; }
  _dockter_hermes_wait_for_dashboard() { return 1; }
  dockter-hermes-dashboard
}

run_windows_config() {
  _dockter_hermes_is_native_windows() { return 0; }
  dockter-hermes-config | grep -F "$TMP_DIR/hermes-agent/docker-compose.windows.yml" >/dev/null
}

run_windows_help() {
  _dockter_hermes_is_native_windows() { return 0; }
  dockter-hermes-help | grep -F 'dockter-hermes-cli dashboard register' >/dev/null
}

run_windows_data_dir() {
  _dockter_hermes_is_native_windows() { return 0; }
  cygpath() {
    [[ "$1" == "-am" && "$2" == "$HERMES_HOME" ]] || return 1
    printf 'C:/Users/test/.hermes'
  }
  [[ "$(_dockter_hermes_data_dir)" == 'C:/Users/test/.hermes' ]]
}

run_windows_compose_model_case() {
  if [[ -z "$REAL_DOCKER_BIN" ]] || ! HOME="$REAL_HOME" "$REAL_DOCKER_BIN" compose version >/dev/null 2>&1; then
    printf 'SKIP windows-compose-model (Docker Compose unavailable)\n'
    return 0
  fi

  local config manual_images
  config=$(DOCKTER_HERMES_DATA_DIR="$TMP_DIR/windows-data" \
    DOCKTER_HERMES_DASHBOARD_PORT=19119 \
    DOCKTER_HERMES_IMAGE=hermes-agent \
    HERMES_UID=501 HERMES_GID=20 \
    HOME="$REAL_HOME" "$REAL_DOCKER_BIN" compose \
      -f "$ROOT_DIR/../docker-compose.windows.yml" \
      config --format json)
  manual_images=$(unset DOCKTER_HERMES_IMAGE; \
    USERPROFILE="$TMP_DIR/windows-profile" HOME="$REAL_HOME" \
    "$REAL_DOCKER_BIN" compose \
      -f "$ROOT_DIR/../docker-compose.windows.yml" \
      config --images)

  [[ "$config" == *'"build":'* ]] || {
    printf 'Windows Compose model did not inherit the local image build\n' >&2
    return 1
  }
  [[ "$config" == *'"image": "hermes-agent"'* ]] || {
    printf 'Windows Compose model did not select the helper local image\n' >&2
    return 1
  }
  [[ "$config" != *'"network_mode"'* ]] || {
    printf 'Windows Compose model retained host networking\n' >&2
    return 1
  }
  [[ "$config" == *'"published": "19119"'* ]] || {
    printf 'Windows Compose model did not publish the configured dashboard port\n' >&2
    return 1
  }
  [[ "$config" == *'"HERMES_UID": "10000"'* ]] || {
    printf 'Windows Compose model did not replace host UID mapping\n' >&2
    return 1
  }
  [[ "$manual_images" == *'nousresearch/hermes-agent:latest'* ]] || {
    printf 'Standalone Windows Compose no longer defaults to the published image\n' >&2
    return 1
  }
  printf 'TEST windows-compose-model\n'
}

run_case help dockter-hermes-help
run_case help-windows run_windows_help
run_case config dockter-hermes-config
run_case config-windows run_windows_config
run_case data-dir-windows run_windows_data_dir
run_case show-config dockter-hermes-show-config
run_case cd dockter-hermes-cd
run_case home dockter-hermes-home
run_case workspace dockter-hermes-workspace

run_logged_case start-default "$(compose_line up -d gateway dashboard)" dockter-hermes-start
run_logged_case start-gateway "$(compose_line up -d gateway)" dockter-hermes-start gateway
run_logged_case start-windows "$(windows_compose_line up -d gateway dashboard)" run_windows_start
run_logged_case restart-default "$(compose_line restart gateway)" dockter-hermes-restart
run_logged_case restart-all "$(compose_line restart gateway dashboard)" dockter-hermes-restart all
run_logged_case stop "$(compose_line down)" dockter-hermes-stop
run_logged_case logs-default "$(compose_line logs -f gateway)" dockter-hermes-logs
run_logged_case logs-dashboard "$(compose_line logs -f dashboard --tail 5)" dockter-hermes-logs dashboard --tail 5
run_logged_case status "$(compose_line ps)
$(command_line docker ps -a --filter name=hermes --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}')" dockter-hermes-status

run_logged_case shell "$(compose_line exec gateway bash)" dockter-hermes-shell gateway
run_logged_case exec "$(compose_line exec gateway hermes --version)" dockter-hermes-exec gateway hermes --version
run_logged_case exec-default-service "$(compose_line exec gateway hermes --help)" dockter-hermes-exec hermes --help
run_failure_case exec-missing-command dockter-hermes-exec
run_logged_case cli "$(compose_line run --rm gateway hermes --help)" dockter-hermes-cli hermes --help
run_logged_case chat "$(compose_line run --rm gateway chat)" dockter-hermes-chat
run_logged_case setup "$(compose_line run --rm gateway setup)" dockter-hermes-setup

run_logged_case dashboard "$(compose_line up -d dashboard)
$(command_line curl -fsS --max-time 2 http://127.0.0.1:19119/api/status)
$(command_line open http://127.0.0.1:19119/)" run_dashboard
run_dashboard_failure_case "$(windows_compose_line up -d dashboard)
$(windows_compose_line logs --tail 50 dashboard)"
run_logged_case health "$(command_line docker exec hermes hermes gateway status --deep)" dockter-hermes-health

run_logged_case rebuild "$(compose_line build gateway)
$(compose_line up -d --force-recreate gateway dashboard)" dockter-hermes-rebuild
run_logged_case rebuild-no-cache-service-first "$(compose_line build --no-cache gateway)
$(compose_line up -d --force-recreate gateway dashboard)" dockter-hermes-rebuild all --no-cache
run_failure_case rebuild-options-before-service dockter-hermes-rebuild --no-cache all
update_log="$(command_line git -C "$TMP_DIR/hermes-agent" pull)
$(compose_line build gateway)
$(compose_line up -d gateway dashboard)"
run_logged_case update "$update_log" dockter-hermes-update
run_update_failure_case "$update_log"
run_logged_case clean "$(compose_line down -v --remove-orphans)" run_clean
run_windows_compose_model_case
