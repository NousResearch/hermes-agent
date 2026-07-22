#!/usr/bin/env bash
# Dockter Hermes - Docker helpers for Hermes Agent
# Inspired with love from Openclaw's clawdock and modified to work with Hermes
#
# Installation:
#   copy this file someplace (ie ~/.dockter-hermes/dockter-hermes-helpers.sh)
#   source ~/.dockter-hermes/dockter-hermes-helpers.sh
#   # optional:
#   echo 'source ~/.dockter-hermes/dockter-hermes-helpers.sh' >> ~/.zshrc
#
# Usage:
#   dockter-hermes-help

# =============================================================================
# Colors
# =============================================================================
_HD_CLR_RESET='\033[0m'
_HD_CLR_BOLD='\033[1m'
_HD_CLR_DIM='\033[2m'
_HD_CLR_GREEN='\033[0;32m'
_HD_CLR_YELLOW='\033[1;33m'
_HD_CLR_BLUE='\033[0;34m'
_HD_CLR_MAGENTA='\033[0;35m'
_HD_CLR_CYAN='\033[0;36m'
_HD_CLR_RED='\033[0;31m'

_hd_cmd() {
  echo "${_HD_CLR_GREEN}${_HD_CLR_BOLD}$1${_HD_CLR_RESET}"
}

# =============================================================================
# Config
# =============================================================================
DOCKTER_HERMES_STATE_DIR="${HERMES_HOME:-${HOME}/.hermes}/dockter-hermes"
DOCKTER_HERMES_CONFIG="${DOCKTER_HERMES_STATE_DIR}/config"
DOCKTER_HERMES_LOCAL_CONFIG="${HOME}/.dockter-hermes/config"
DOCKTER_HERMES_USER_COMPOSE="${DOCKTER_HERMES_STATE_DIR}/docker-compose.override.yml"
DOCKTER_HERMES_DOCKER_BIN="${DOCKTER_HERMES_DOCKER_BIN:-}"

DOCKTER_HERMES_COMMON_PATHS=(
  "${HOME}/Source/hermes-agent"
  "${HOME}/hermes-agent"
  "${HOME}/workspace/hermes-agent"
  "${HOME}/projects/hermes-agent"
  "${HOME}/dev/hermes-agent"
  "${HOME}/code/hermes-agent"
  "${HOME}/src/hermes-agent"
)

_dockter_hermes_filter_warnings() {
  grep -v "^WARN\|^time="
}

_dockter_hermes_trim_quotes() {
  local value="$1"
  value="${value#\"}"
  value="${value%\"}"
  printf "%s" "$value"
}

_dockter_hermes_mask_value() {
  local value="$1"
  local length=${#value}
  if (( length == 0 )); then
    printf "%s" "<empty>"
    return 0
  fi
  printf "<redacted:%s chars>" "$length"
}

_dockter_hermes_read_config_dir() {
  local config_file="$DOCKTER_HERMES_CONFIG"
  if [[ ! -f "$config_file" && -f "$DOCKTER_HERMES_LOCAL_CONFIG" ]]; then
    config_file="$DOCKTER_HERMES_LOCAL_CONFIG"
  fi
  if [[ ! -f "$config_file" ]]; then
    return 1
  fi
  local raw
  raw=$(sed -n 's/^DOCKTER_HERMES_DIR=//p' "$config_file" | head -n 1)
  if [[ -z "$raw" ]]; then
    return 1
  fi
  _dockter_hermes_trim_quotes "$raw"
}

_dockter_hermes_config_file() {
  if [[ -f "$DOCKTER_HERMES_CONFIG" ]]; then
    printf "%s" "$DOCKTER_HERMES_CONFIG"
    return 0
  fi
  if [[ -f "$DOCKTER_HERMES_LOCAL_CONFIG" ]]; then
    printf "%s" "$DOCKTER_HERMES_LOCAL_CONFIG"
    return 0
  fi
  return 1
}

_dockter_hermes_active_config_source() {
  if [[ -n "${DOCKTER_HERMES_DIR:-}" && -f "${DOCKTER_HERMES_DIR}/docker-compose.yml" ]]; then
    printf "%s" "environment"
    return 0
  fi
  local config_file
  config_file=$(_dockter_hermes_config_file) || {
    printf "%s" "<not saved>"
    return 0
  }
  local config_dir
  config_dir=$(_dockter_hermes_read_config_dir) || {
    printf "%s" "$config_file"
    return 0
  }
  if [[ -f "${config_dir}/docker-compose.yml" ]]; then
    printf "%s" "$config_file"
    return 0
  fi
  printf "%s" "${config_file} (invalid path)"
}

_dockter_hermes_save_dir() {
  if [[ ! -d "$DOCKTER_HERMES_STATE_DIR" ]]; then
    /bin/mkdir -p "$DOCKTER_HERMES_STATE_DIR"
  fi
  echo "DOCKTER_HERMES_DIR=\"$DOCKTER_HERMES_DIR\"" > "$DOCKTER_HERMES_CONFIG"
  echo "Saved Dockter Hermes config to $DOCKTER_HERMES_CONFIG"
}

_dockter_hermes_ensure_docker() {
  local candidate
  for candidate in \
    "$DOCKTER_HERMES_DOCKER_BIN" \
    "$(command -v docker 2>/dev/null)" \
    /usr/local/bin/docker \
    /opt/homebrew/bin/docker \
    /Applications/Docker.app/Contents/Resources/bin/docker
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      DOCKTER_HERMES_DOCKER_BIN="$candidate"
      return 0
    fi
  done

  echo "Error: docker was not found. Set DOCKTER_HERMES_DOCKER_BIN=/path/to/docker if Docker is installed elsewhere."
  return 1
}

_dockter_hermes_docker() {
  _dockter_hermes_ensure_docker || return 1
  "$DOCKTER_HERMES_DOCKER_BIN" "$@"
}

_dockter_hermes_is_native_windows() {
  case "$(command uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

_dockter_hermes_data_dir() {
  local data_dir="${HERMES_HOME:-${HOME}/.hermes}"
  if _dockter_hermes_is_native_windows && command -v cygpath >/dev/null 2>&1; then
    cygpath -am "$data_dir"
    return $?
  fi
  printf '%s' "$data_dir"
}

_dockter_hermes_ensure_dir() {
  if [[ -n "$DOCKTER_HERMES_DIR" && -f "${DOCKTER_HERMES_DIR}/docker-compose.yml" ]]; then
    return 0
  fi

  local config_dir
  config_dir=$(_dockter_hermes_read_config_dir)
  if [[ -n "$config_dir" && -f "${config_dir}/docker-compose.yml" ]]; then
    DOCKTER_HERMES_DIR="$config_dir"
    return 0
  fi

  local found_path=""
  for path in "${DOCKTER_HERMES_COMMON_PATHS[@]}"; do
    if [[ -f "${path}/docker-compose.yml" && -f "${path}/Dockerfile" ]]; then
      found_path="$path"
      break
    fi
  done

  if [[ -n "$found_path" ]]; then
    echo ""
    echo "Found Hermes Agent at: $found_path"
    echo -n "Use this location? [Y/n] "
    read -r response
    if [[ "$response" =~ ^[Nn] ]]; then
      echo ""
      echo "Set DOCKTER_HERMES_DIR manually:"
      echo "  export DOCKTER_HERMES_DIR=/path/to/hermes-agent"
      return 1
    fi
    DOCKTER_HERMES_DIR="$found_path"
    _dockter_hermes_save_dir
    echo ""
    return 0
  fi

  echo ""
  echo "Hermes Agent was not found in common locations."
  echo ""
  echo "Set DOCKTER_HERMES_DIR manually:"
  echo "  export DOCKTER_HERMES_DIR=/path/to/hermes-agent"
  echo ""
  return 1
}

_dockter_hermes_uid() {
  if [[ -n "${HERMES_UID:-}" ]]; then
    printf "%s" "$HERMES_UID"
    return 0
  fi
  if [[ -x /usr/bin/id ]]; then
    /usr/bin/id -u
    return $?
  fi
  if command -v id >/dev/null 2>&1; then
    command id -u
    return $?
  fi
  echo "Error: id was not found. Set HERMES_UID manually." >&2
  return 1
}

_dockter_hermes_gid() {
  if [[ -n "${HERMES_GID:-}" ]]; then
    printf "%s" "$HERMES_GID"
    return 0
  fi
  if [[ -x /usr/bin/id ]]; then
    /usr/bin/id -g
    return $?
  fi
  if command -v id >/dev/null 2>&1; then
    command id -g
    return $?
  fi
  echo "Error: id was not found. Set HERMES_GID manually." >&2
  return 1
}

_dockter_hermes_compose_files() {
  local base_file
  if _dockter_hermes_is_native_windows; then
    base_file="${DOCKTER_HERMES_DIR}/docker-compose.windows.yml"
  else
    base_file="${DOCKTER_HERMES_DIR}/docker-compose.yml"
  fi
  if [[ ! -f "$base_file" ]]; then
    echo "Error: Compose file not found: $base_file" >&2
    return 1
  fi
  printf '%s\n' "$base_file"

  if [[ -f "${DOCKTER_HERMES_DIR}/docker-compose.extra.yml" ]]; then
    printf '%s\n' "${DOCKTER_HERMES_DIR}/docker-compose.extra.yml"
  fi
  if [[ -f "$DOCKTER_HERMES_USER_COMPOSE" ]]; then
    printf '%s\n' "$DOCKTER_HERMES_USER_COMPOSE"
  fi
}

_dockter_hermes_compose() {
  _dockter_hermes_ensure_docker || return 1
  _dockter_hermes_ensure_dir || return 1

  local compose_files
  compose_files=$(_dockter_hermes_compose_files) || return 1
  local compose_args=()
  local compose_file
  while IFS= read -r compose_file; do
    [[ -n "$compose_file" ]] && compose_args+=(-f "$compose_file")
  done <<< "$compose_files"

  local data_dir uid gid
  data_dir=$(_dockter_hermes_data_dir) || return 1
  uid=$(_dockter_hermes_uid) || return 1
  gid=$(_dockter_hermes_gid) || return 1

  DOCKTER_HERMES_DATA_DIR="$data_dir" \
    DOCKTER_HERMES_DASHBOARD_PORT="${DOCKTER_HERMES_DASHBOARD_PORT:-9119}" \
    DOCKTER_HERMES_IMAGE="hermes-agent" \
    HERMES_UID="$uid" HERMES_GID="$gid" \
    "$DOCKTER_HERMES_DOCKER_BIN" compose "${compose_args[@]}" "$@"
}

_dockter_hermes_default_service() {
  local service="${1:-gateway}"
  case "$service" in
    gateway|dashboard)
      printf "%s" "$service"
      ;;
    all)
      printf "%s" "all"
      ;;
    *)
      echo "Unknown service: $service" >&2
      echo "Expected: gateway, dashboard, or all" >&2
      return 1
      ;;
  esac
}

_dockter_hermes_container_name() {
  local service="${1:-gateway}"
  case "$service" in
    gateway)
      printf "%s" "${DOCKTER_HERMES_GATEWAY_CONTAINER:-hermes}"
      ;;
    dashboard)
      printf "%s" "${DOCKTER_HERMES_DASHBOARD_CONTAINER:-hermes-dashboard}"
      ;;
    *)
      printf "%s" "$service"
      ;;
  esac
}

_dockter_hermes_show_env_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo -e "${_HD_CLR_YELLOW}No ${file} found${_HD_CLR_RESET}"
    return 0
  fi

  echo -e "${_HD_CLR_BOLD}${file}${_HD_CLR_RESET}"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$line" ]]; then
      echo -e "${_HD_CLR_DIM}${line}${_HD_CLR_RESET}"
    elif [[ "$line" == *=* ]]; then
      local key="${line%%=*}"
      local val="${line#*=}"
      echo -e "${_HD_CLR_CYAN}${key}${_HD_CLR_RESET}=${_HD_CLR_DIM}$(_dockter_hermes_mask_value "$val")${_HD_CLR_RESET}"
    else
      echo -e "${_HD_CLR_DIM}${line}${_HD_CLR_RESET}"
    fi
  done < "$file"
}

# =============================================================================
# Basic operations
# =============================================================================
dockter-hermes-start() {
  local service
  service=$(_dockter_hermes_default_service "${1:-all}") || return 1
  if [[ "$service" == "all" ]]; then
    _dockter_hermes_compose up -d gateway dashboard
  else
    _dockter_hermes_compose up -d "$service"
  fi
}

dockter-hermes-stop() {
  _dockter_hermes_compose down
}

dockter-hermes-restart() {
  local service
  service=$(_dockter_hermes_default_service "${1:-gateway}") || return 1
  if [[ "$service" == "all" ]]; then
    _dockter_hermes_compose restart gateway dashboard
  else
    _dockter_hermes_compose restart "$service"
  fi
}

dockter-hermes-logs() {
  local service
  service=$(_dockter_hermes_default_service "${1:-gateway}") || return 1
  shift 2>/dev/null || true
  if [[ "$service" == "all" ]]; then
    _dockter_hermes_compose logs -f "$@"
  else
    _dockter_hermes_compose logs -f "$service" "$@"
  fi
}

dockter-hermes-status() {
  _dockter_hermes_compose ps
  echo ""
  echo -e "${_HD_CLR_BOLD}Hermes-looking Docker containers:${_HD_CLR_RESET}"
  _dockter_hermes_docker ps -a \
    --filter "name=hermes" \
    --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
}

# =============================================================================
# Navigation and config
# =============================================================================
dockter-hermes-cd() {
  _dockter_hermes_ensure_dir || return 1
  cd "$DOCKTER_HERMES_DIR" || return 1
}

dockter-hermes-config() {
  local config_source
  config_source=$(_dockter_hermes_active_config_source)
  _dockter_hermes_ensure_dir || true
  _dockter_hermes_ensure_docker >/dev/null 2>&1 || true

  local hermes_home="${HERMES_HOME:-${HOME}/.hermes}"
  if [[ "$config_source" == "<not saved>" && -n "${DOCKTER_HERMES_DIR:-}" ]]; then
    config_source="auto-detected and saved to ${DOCKTER_HERMES_CONFIG}"
  fi

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_CYAN}Dockter Hermes helper config${_HD_CLR_RESET}"
  echo ""
  echo -e "${_HD_CLR_BOLD}Source:${_HD_CLR_RESET} ${_HD_CLR_CYAN}${config_source}${_HD_CLR_RESET}"
  echo -e "${_HD_CLR_BOLD}DOCKTER_HERMES_DIR:${_HD_CLR_RESET} ${_HD_CLR_CYAN}${DOCKTER_HERMES_DIR:-<not set>}${_HD_CLR_RESET}"
  echo -e "${_HD_CLR_BOLD}HERMES_HOME:${_HD_CLR_RESET} ${_HD_CLR_CYAN}${hermes_home}${_HD_CLR_RESET}"
  echo -e "${_HD_CLR_BOLD}Docker:${_HD_CLR_RESET} ${_HD_CLR_CYAN}${DOCKTER_HERMES_DOCKER_BIN:-<not found>}${_HD_CLR_RESET}"
  echo ""
  echo -e "${_HD_CLR_BOLD}Compose files:${_HD_CLR_RESET}"
  if [[ -n "${DOCKTER_HERMES_DIR:-}" ]]; then
    local compose_files compose_file
    if compose_files=$(_dockter_hermes_compose_files); then
      while IFS= read -r compose_file; do
        [[ -n "$compose_file" ]] && echo "  $compose_file"
      done <<< "$compose_files"
    else
      echo "  <unable to resolve Compose files>"
    fi
  else
    echo "  <project directory not set>"
  fi
  echo ""
  echo -e "${_HD_CLR_DIM}Temporary override:${_HD_CLR_RESET} export DOCKTER_HERMES_DIR=/path/to/hermes-agent"
  echo -e "${_HD_CLR_DIM}Re-detect:${_HD_CLR_RESET} unset DOCKTER_HERMES_DIR && rm \"${DOCKTER_HERMES_CONFIG}\""
}

dockter-hermes-home() {
  cd "${HERMES_HOME:-${HOME}/.hermes}" || return 1
}

dockter-hermes-workspace() {
  cd "${HERMES_HOME:-${HOME}/.hermes}/workspace" || return 1
}

dockter-hermes-show-config() {
  _dockter_hermes_ensure_dir >/dev/null 2>&1 || true
  local hermes_home="${HERMES_HOME:-${HOME}/.hermes}"
  echo -e "${_HD_CLR_BOLD}Hermes data directory:${_HD_CLR_RESET} ${_HD_CLR_CYAN}${hermes_home}${_HD_CLR_RESET}"
  echo -e "${_HD_CLR_BOLD}Hermes project directory:${_HD_CLR_RESET} ${_HD_CLR_CYAN}${DOCKTER_HERMES_DIR:-<not set>}${_HD_CLR_RESET}"
  echo ""

  if [[ -f "${hermes_home}/config.yaml" ]]; then
    echo -e "${_HD_CLR_BOLD}${hermes_home}/config.yaml${_HD_CLR_RESET}"
    sed -n '1,220p' "${hermes_home}/config.yaml"
  else
    echo -e "${_HD_CLR_YELLOW}No ${hermes_home}/config.yaml found${_HD_CLR_RESET}"
  fi
  echo ""

  _dockter_hermes_show_env_file "${hermes_home}/.env"
  echo ""

  if [[ -n "$DOCKTER_HERMES_DIR" && -f "${DOCKTER_HERMES_DIR}/.env" ]]; then
    _dockter_hermes_show_env_file "${DOCKTER_HERMES_DIR}/.env"
    echo ""
  fi
}

# =============================================================================
# Container access
# =============================================================================
dockter-hermes-shell() {
  local service="${1:-gateway}"
  service=$(_dockter_hermes_default_service "$service") || return 1
  if [[ "$service" == "all" ]]; then
    service="gateway"
  fi
  _dockter_hermes_compose exec "$service" bash
}

dockter-hermes-exec() {
  local service="gateway"
  if [[ "${1:-}" == "gateway" || "${1:-}" == "dashboard" ]]; then
    service="$1"
    shift
  fi
  if [[ $# -eq 0 ]]; then
    echo -e "Usage: $(_hd_cmd 'dockter-hermes-exec [gateway|dashboard] <command>')"
    return 1
  fi
  _dockter_hermes_compose exec "$service" "$@"
}

dockter-hermes-cli() {
  _dockter_hermes_compose run --rm gateway "$@"
}

dockter-hermes-chat() {
  _dockter_hermes_compose run --rm gateway chat "$@"
}

dockter-hermes-setup() {
  _dockter_hermes_compose run --rm gateway setup
}

# =============================================================================
# Dashboard and health
# =============================================================================
_dockter_hermes_dashboard_url() {
  printf 'http://127.0.0.1:%s/' "${DOCKTER_HERMES_DASHBOARD_PORT:-9119}"
}

_dockter_hermes_wait_for_dashboard() {
  local url="$1"
  if ! command -v curl >/dev/null 2>&1; then
    return 2
  fi

  local attempt
  for (( attempt = 0; attempt < 15; attempt++ )); do
    if curl -fsS --max-time 2 "${url}api/status" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

dockter-hermes-dashboard() {
  _dockter_hermes_compose up -d dashboard || return 1
  local url wait_status
  url=$(_dockter_hermes_dashboard_url)
  if _dockter_hermes_wait_for_dashboard "$url"; then
    wait_status=0
  else
    wait_status=$?
  fi
  if (( wait_status == 1 )); then
    echo "Dashboard did not become ready at ${url}" >&2
    _dockter_hermes_compose logs --tail 50 dashboard >&2 || true
    if _dockter_hermes_is_native_windows; then
      echo "" >&2
      echo "Docker Desktop bridge networking requires a non-loopback bind inside the container," >&2
      echo "so Hermes requires a dashboard auth provider. Configure one, then restart dashboard:" >&2
      echo "  OAuth:             dockter-hermes-cli dashboard register" >&2
      echo "  Username/password: configure HERMES_DASHBOARD_BASIC_AUTH_* in ~/.hermes/.env" >&2
      echo "  Restart:           dockter-hermes-restart dashboard" >&2
    fi
    return 1
  fi
  echo -e "Opening: ${_HD_CLR_CYAN}${url}${_HD_CLR_RESET}"
  open "$url" 2>/dev/null || xdg-open "$url" 2>/dev/null || \
    echo -e "Open manually: ${_HD_CLR_CYAN}${url}${_HD_CLR_RESET}"
}

dockter-hermes-health() {
  local container
  container=$(_dockter_hermes_container_name gateway)
  if _dockter_hermes_docker exec "$container" hermes gateway status --deep; then
    return 0
  fi

  echo ""
  echo "Container gateway status failed. If the API server is enabled, trying HTTP health..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${DOCKTER_HERMES_API_PORT:-8642}/health" || return 1
    echo ""
  else
    echo "curl is not available"
    return 1
  fi
}

# =============================================================================
# Maintenance
# =============================================================================
dockter-hermes-rebuild() {
  local service
  service=$(_dockter_hermes_default_service "${1:-all}") || return 1
  _dockter_hermes_compose build gateway || return 1
  if [[ "$service" == "all" ]]; then
    _dockter_hermes_compose up -d --force-recreate gateway dashboard
  else
    _dockter_hermes_compose up -d --force-recreate "$service"
  fi
}

dockter-hermes-update() {
  _dockter_hermes_ensure_dir || return 1

  echo "Updating Hermes Agent Docker instance..."
  echo ""
  echo "Pulling latest source..."
  git -C "$DOCKTER_HERMES_DIR" pull || { echo "git pull failed"; return 1; }

  echo ""
  echo "Rebuilding Docker image..."
  _dockter_hermes_compose build gateway || { echo "Docker build failed"; return 1; }

  echo ""
  echo "Recreating gateway and dashboard containers..."
  local compose_output compose_status
  if compose_output=$(_dockter_hermes_compose up -d gateway dashboard 2>&1); then
    compose_status=0
  else
    compose_status=$?
  fi
  if [[ -n "$compose_output" ]]; then
    printf '%s\n' "$compose_output" | _dockter_hermes_filter_warnings || true
  fi
  if (( compose_status != 0 )); then
    echo "Container recreation failed"
    return "$compose_status"
  fi

  echo ""
  echo "Update complete."
  echo -e "Verify: $(_hd_cmd dockter-hermes-status)"
}

dockter-hermes-clean() {
  echo "This removes Hermes Docker containers created by this compose file."
  echo "Your host Hermes data directory is bind-mounted and is not removed by compose volumes."
  echo -n "Continue? [y/N] "
  read -r response
  if [[ ! "$response" =~ ^[Yy] ]]; then
    echo "Cancelled."
    return 1
  fi
  _dockter_hermes_compose down -v --remove-orphans
}

# =============================================================================
# Help
# =============================================================================
dockter-hermes-help() {
  echo -e "\n${_HD_CLR_BOLD}${_HD_CLR_CYAN}Dockter Hermes - Docker Helpers for Hermes Agent${_HD_CLR_RESET}\n"

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_MAGENTA}Basic Operations${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-start) ${_HD_CLR_CYAN}[gateway|dashboard|all]${_HD_CLR_RESET}     ${_HD_CLR_DIM}Start containers (default: all)${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-stop)                         ${_HD_CLR_DIM}Stop the compose stack${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-restart) ${_HD_CLR_CYAN}[gateway|dashboard|all]${_HD_CLR_RESET}   ${_HD_CLR_DIM}Restart containers${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-status)                       ${_HD_CLR_DIM}Check container status${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-logs) ${_HD_CLR_CYAN}[gateway|dashboard|all]${_HD_CLR_RESET}      ${_HD_CLR_DIM}View live logs${_HD_CLR_RESET}"
  echo ""

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_MAGENTA}Container Access${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-shell) ${_HD_CLR_CYAN}[gateway|dashboard]${_HD_CLR_RESET}         ${_HD_CLR_DIM}Open a shell in a container${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-exec) ${_HD_CLR_CYAN}[service] <cmd>${_HD_CLR_RESET}              ${_HD_CLR_DIM}Run a command in a container${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-cli) ${_HD_CLR_CYAN}<args>${_HD_CLR_RESET}                        ${_HD_CLR_DIM}Run hermes inside a one-off container${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-chat)                         ${_HD_CLR_DIM}Open interactive Hermes chat${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-setup)                        ${_HD_CLR_DIM}Run the setup wizard${_HD_CLR_RESET}"
  echo ""

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_MAGENTA}Dashboard & Health${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-dashboard)                    ${_HD_CLR_DIM}Start/open the dashboard${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-health)                       ${_HD_CLR_DIM}Run gateway status/health checks${_HD_CLR_RESET}"
  echo ""

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_MAGENTA}Config & Navigation${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-cd)                           ${_HD_CLR_DIM}Jump to the Hermes project directory${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-config)                       ${_HD_CLR_DIM}Print Dockter Hermes helper config${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-home)                         ${_HD_CLR_DIM}Jump to the Hermes data directory${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-workspace)                    ${_HD_CLR_DIM}Jump to the Hermes workspace directory${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-show-config)                  ${_HD_CLR_DIM}Print Hermes config with secrets redacted${_HD_CLR_RESET}"
  echo ""

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_MAGENTA}Maintenance${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-update)                       ${_HD_CLR_DIM}Pull, rebuild, and recreate containers${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-rebuild) ${_HD_CLR_CYAN}[gateway|dashboard|all]${_HD_CLR_RESET}   ${_HD_CLR_DIM}Rebuild image and recreate containers (default: all)${_HD_CLR_RESET}"
  echo -e "  $(_hd_cmd dockter-hermes-clean)                        ${_HD_CLR_RED}Remove compose containers and volumes${_HD_CLR_RESET}"
  echo ""

  echo -e "${_HD_CLR_BOLD}${_HD_CLR_CYAN}First Time Setup${_HD_CLR_RESET}"
  echo -e "  ${_HD_CLR_CYAN}1.${_HD_CLR_RESET} $(_hd_cmd dockter-hermes-setup)       ${_HD_CLR_DIM}# Configure ~/.hermes inside Docker${_HD_CLR_RESET}"
  if _dockter_hermes_is_native_windows; then
    echo -e "  ${_HD_CLR_CYAN}2.${_HD_CLR_RESET} $(_hd_cmd 'dockter-hermes-cli dashboard register') ${_HD_CLR_DIM}# Configure OAuth (or basic auth in ~/.hermes/.env)${_HD_CLR_RESET}"
    echo -e "  ${_HD_CLR_CYAN}3.${_HD_CLR_RESET} $(_hd_cmd dockter-hermes-start)     ${_HD_CLR_DIM}# Start gateway and dashboard${_HD_CLR_RESET}"
    echo -e "  ${_HD_CLR_CYAN}4.${_HD_CLR_RESET} $(_hd_cmd dockter-hermes-dashboard)   ${_HD_CLR_DIM}# Open http://127.0.0.1:${DOCKTER_HERMES_DASHBOARD_PORT:-9119}${_HD_CLR_RESET}"
  else
    echo -e "  ${_HD_CLR_CYAN}2.${_HD_CLR_RESET} $(_hd_cmd dockter-hermes-start)     ${_HD_CLR_DIM}# Start gateway and dashboard${_HD_CLR_RESET}"
    echo -e "  ${_HD_CLR_CYAN}3.${_HD_CLR_RESET} $(_hd_cmd dockter-hermes-dashboard)   ${_HD_CLR_DIM}# Open http://127.0.0.1:${DOCKTER_HERMES_DASHBOARD_PORT:-9119}${_HD_CLR_RESET}"
  fi
  echo ""
}
