#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="${TMPDIR:-/tmp}/dockter-hermes-test.$$"
LOG_FILE="${TMP_DIR}/commands.log"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$TMP_DIR/bin" "$TMP_DIR/home/.hermes/workspace" "$TMP_DIR/hermes-agent"
touch "$LOG_FILE"
touch "$TMP_DIR/hermes-agent/Dockerfile"
touch "$TMP_DIR/hermes-agent/docker-compose.yml"
touch "$TMP_DIR/hermes-agent/docker-compose.extra.yml"

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

run_clean() {
  printf 'y\n' | dockter-hermes-clean
}

run_case help dockter-hermes-help
run_case config dockter-hermes-config
run_case show-config dockter-hermes-show-config
run_case cd dockter-hermes-cd
run_case home dockter-hermes-home
run_case workspace dockter-hermes-workspace

run_logged_case start-default "$(compose_line up -d gateway dashboard)" dockter-hermes-start
run_logged_case start-gateway "$(compose_line up -d gateway)" dockter-hermes-start gateway
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
$(command_line open http://127.0.0.1:9119/)" dockter-hermes-dashboard
run_logged_case health "$(command_line docker exec hermes hermes gateway status --deep)" dockter-hermes-health

run_logged_case rebuild "$(compose_line build gateway)
$(compose_line up -d --force-recreate gateway dashboard)" dockter-hermes-rebuild
update_log="$(command_line git -C "$TMP_DIR/hermes-agent" pull)
$(compose_line build gateway)
$(compose_line up -d gateway dashboard)"
run_logged_case update "$update_log" dockter-hermes-update
run_update_failure_case "$update_log"
run_logged_case clean "$(compose_line down -v --remove-orphans)" run_clean
