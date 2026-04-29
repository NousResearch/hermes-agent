#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEAM_PREFIX="${TEAM_PREFIX:-feishu}"
SOURCE_PROFILE="${SOURCE_PROFILE:-default}"
PROFILES_ROOT="${HERMES_PROFILES_ROOT:-$HOME/.hermes/profiles}"
LOG_DIR="${FEISHU_TEAM_LOG_DIR:-$PROJECT_ROOT/.feishu-team-logs}"
PYTHON_BIN="${HERMES_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

roles=(coordinator researcher builder reviewer)

profile_name() {
  printf '%s-%s' "$TEAM_PREFIX" "$1"
}

profile_dir() {
  printf '%s/%s' "$PROFILES_ROOT" "$(profile_name "$1")"
}

bot_display_name() {
  case "$1" in
    coordinator) printf 'Coordinator' ;;
    researcher) printf 'Researcher' ;;
    builder) printf 'Builder' ;;
    reviewer) printf 'Reviewer' ;;
    *) printf '%s' "$1" ;;
  esac
}

hermes_cmd() {
  (cd "$PROJECT_ROOT" && "$PYTHON_BIN" "$PROJECT_ROOT/hermes" "$@")
}

usage() {
  cat <<'EOF'
Usage:
  scripts/start_feishu_team.sh init
  scripts/start_feishu_team.sh check
  scripts/start_feishu_team.sh start
  scripts/start_feishu_team.sh stop
  scripts/start_feishu_team.sh status
  scripts/start_feishu_team.sh logs [role]

Environment overrides:
  TEAM_PREFIX=feishu
  SOURCE_PROFILE=default
  HERMES_PYTHON=/path/to/python
  HERMES_PROFILES_ROOT=$HOME/.hermes/profiles
  FEISHU_TEAM_LOG_DIR=./.feishu-team-logs

After init, fill each profile .env:
  ~/.hermes/profiles/feishu-coordinator/.env
  ~/.hermes/profiles/feishu-researcher/.env
  ~/.hermes/profiles/feishu-builder/.env
  ~/.hermes/profiles/feishu-reviewer/.env
EOF
}

write_soul() {
  local role="$1"
  local dir
  dir="$(profile_dir "$role")"
  local name
  name="$(bot_display_name "$role")"
  case "$role" in
    coordinator)
      cat > "$dir/SOUL.md" <<'EOF'
You are the Coordinator agent in a Feishu multi-agent team.

Your job is to understand the user's goal, split work into clear assignments, mention the right teammate bots, track progress, prevent loops, and synthesize final decisions.

When coordinating in a Feishu group:
- Only respond when explicitly mentioned or directly assigned.
- Assign short, concrete tasks to Researcher, Builder, and Reviewer.
- Include a stable task label when asking teammates to collaborate.
- Do not repeatedly mention teammates if they have already responded.
- Summarize the team result in the user's language.
EOF
      ;;
    researcher)
      cat > "$dir/SOUL.md" <<'EOF'
You are the Researcher agent in a Feishu multi-agent team.

Your job is to gather relevant facts, constraints, options, and open questions.

When collaborating in a Feishu group:
- Only respond when explicitly mentioned or directly assigned.
- Keep answers concise and structured.
- Do not make code changes.
- Address Coordinator when reporting results.
EOF
      ;;
    builder)
      cat > "$dir/SOUL.md" <<'EOF'
You are the Builder agent in a Feishu multi-agent team.

Your job is to inspect implementation details, propose or make the smallest justified change, and report exact files and verification steps.

When collaborating in a Feishu group:
- Only respond when explicitly mentioned or directly assigned.
- Prefer minimal, reversible implementation steps.
- State whether you changed files.
- Address Coordinator when reporting results.
EOF
      ;;
    reviewer)
      cat > "$dir/SOUL.md" <<'EOF'
You are the Reviewer agent in a Feishu multi-agent team.

Your job is to review correctness, safety, maintainability, test coverage, operational risk, and cost/runtime risk.

When collaborating in a Feishu group:
- Only respond when explicitly mentioned or directly assigned.
- Do not make code changes unless explicitly requested.
- Prioritize blockers first, then recommendations.
- Address Coordinator when reporting results.
EOF
      ;;
  esac
  printf '%s\n' "Wrote $dir/SOUL.md for $name"
}

write_config() {
  local role="$1"
  local dir
  dir="$(profile_dir "$role")"
  "$PYTHON_BIN" - "$dir" <<'PY'
from pathlib import Path
import sys
import yaml

profile_dir = Path(sys.argv[1])
config_path = profile_dir / "config.yaml"
if config_path.exists():
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
else:
    data = {}

platforms = data.setdefault("platforms", {})
feishu = platforms.setdefault("feishu", {})
feishu["enabled"] = True
extra = feishu.setdefault("extra", {})
extra.setdefault("domain", "feishu")
extra.setdefault("connection_mode", "websocket")
extra.setdefault("default_group_policy", "open")

data["group_sessions_per_user"] = False
data["thread_sessions_per_user"] = False
data["unauthorized_dm_behavior"] = "ignore"

config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY
  printf '%s\n' "Updated $dir/config.yaml"
}

ensure_env_key() {
  local file="$1"
  local key="$2"
  local value="${3:-}"
  if [[ ! -f "$file" ]]; then
    touch "$file"
  fi
  if ! grep -q "^${key}=" "$file"; then
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

set_env_key() {
  local file="$1"
  local key="$2"
  local value="${3:-}"
  if [[ ! -f "$file" ]]; then
    touch "$file"
  fi
  "$PYTHON_BIN" - "$file" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
prefix = key + "="
updated = False
out = []
for line in lines:
    if line.startswith(prefix):
        if not updated:
            out.append(prefix + value)
            updated = True
        continue
    out.append(line)
if not updated:
    out.append(prefix + value)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
}

write_env_templates() {
  local role="$1"
  local dir
  dir="$(profile_dir "$role")"
  local name
  name="$(bot_display_name "$role")"
  cat > "$dir/.env.feishu-team.example" <<EOF
FEISHU_APP_ID=
FEISHU_APP_SECRET=
FEISHU_BOT_OPEN_ID=
FEISHU_BOT_USER_ID=
FEISHU_BOT_NAME=$name
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket
FEISHU_GROUP_POLICY=open
FEISHU_ALLOWED_USERS=
FEISHU_HOME_CHANNEL=
FEISHU_HOME_CHANNEL_NAME=Feishu Team Room
EOF
  local env_file="$dir/.env"
  set_env_key "$env_file" "FEISHU_APP_ID"
  set_env_key "$env_file" "FEISHU_APP_SECRET"
  set_env_key "$env_file" "FEISHU_BOT_OPEN_ID"
  set_env_key "$env_file" "FEISHU_BOT_USER_ID"
  set_env_key "$env_file" "FEISHU_BOT_NAME" "$name"
  set_env_key "$env_file" "FEISHU_DOMAIN" "feishu"
  set_env_key "$env_file" "FEISHU_CONNECTION_MODE" "websocket"
  set_env_key "$env_file" "FEISHU_GROUP_POLICY" "open"
  set_env_key "$env_file" "FEISHU_ALLOWED_USERS"
  set_env_key "$env_file" "FEISHU_HOME_CHANNEL"
  set_env_key "$env_file" "FEISHU_HOME_CHANNEL_NAME" "Feishu Team Room"
  chmod 600 "$env_file" 2>/dev/null || true
  printf '%s\n' "Prepared $env_file and $dir/.env.feishu-team.example"
}

init_profiles() {
  for role in "${roles[@]}"; do
    local profile
    profile="$(profile_name "$role")"
    local dir
    dir="$(profile_dir "$role")"
    if [[ ! -d "$dir" ]]; then
      hermes_cmd profile create "$profile" --clone --clone-from "$SOURCE_PROFILE" --no-alias
    else
      printf '%s\n' "Profile exists: $profile"
    fi
    write_soul "$role"
    write_config "$role"
    write_env_templates "$role"
  done
}

value_from_env_file() {
  local file="$1"
  local key="$2"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2); exit}' "$file"
}

check_profile() {
  local role="$1"
  local dir
  dir="$(profile_dir "$role")"
  local env_file="$dir/.env"
  local ok=0
  printf '%s\n' "[$(profile_name "$role")] $dir"
  if [[ ! -d "$dir" ]]; then
    printf '  missing profile\n'
    return 1
  fi
  for key in FEISHU_APP_ID FEISHU_APP_SECRET FEISHU_BOT_NAME; do
    local value
    value="$(value_from_env_file "$env_file" "$key" | tr -d '[:space:]')"
    if [[ -n "$value" ]]; then
      printf '  %s=ok\n' "$key"
    else
      printf '  %s=missing\n' "$key"
      ok=1
    fi
  done
  return "$ok"
}

check_all() {
  local failed=0
  for role in "${roles[@]}"; do
    check_profile "$role" || failed=1
  done
  return "$failed"
}

start_one() {
  local role="$1"
  check_profile "$role" >/dev/null || {
    printf '%s\n' "Cannot start $(profile_name "$role"): missing required Feishu values in $(profile_dir "$role")/.env"
    return 1
  }
  mkdir -p "$LOG_DIR"
  local profile
  profile="$(profile_name "$role")"
  local log_file="$LOG_DIR/$profile.log"
  local pid_file="$LOG_DIR/$profile.pid"
  nohup "$PYTHON_BIN" "$PROJECT_ROOT/hermes" -p "$profile" gateway run --replace > "$log_file" 2>&1 &
  printf '%s\n' "$!" > "$pid_file"
  printf '%s\n' "Started $profile pid=$(cat "$pid_file") log=$log_file"
}

start_all() {
  local failed=0
  for role in "${roles[@]}"; do
    start_one "$role" || failed=1
  done
  return "$failed"
}

stop_one() {
  local role="$1"
  local profile
  profile="$(profile_name "$role")"
  local pid_file="$LOG_DIR/$profile.pid"
  if [[ ! -f "$pid_file" ]]; then
    printf '%s\n' "No pid file for $profile"
    return 0
  fi
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    printf '%s\n' "Stopped $profile pid=$pid"
  else
    printf '%s\n' "$profile pid=$pid is not running"
  fi
  rm -f "$pid_file"
}

stop_all() {
  for role in "${roles[@]}"; do
    stop_one "$role"
  done
}

status_all() {
  for role in "${roles[@]}"; do
    local profile
    profile="$(profile_name "$role")"
    printf '\n== %s ==\n' "$profile"
    hermes_cmd -p "$profile" gateway status || true
  done
}

logs_cmd() {
  mkdir -p "$LOG_DIR"
  if [[ "${1:-}" ]]; then
    tail -n 80 -f "$LOG_DIR/$(profile_name "$1").log"
  else
    for role in "${roles[@]}"; do
      local profile
      profile="$(profile_name "$role")"
      local log_file="$LOG_DIR/$profile.log"
      if [[ -f "$log_file" ]]; then
        printf '\n== %s ==\n' "$profile"
        tail -n 40 "$log_file"
      fi
    done
  fi
}

cmd="${1:-}"
case "$cmd" in
  init)
    init_profiles
    ;;
  check)
    check_all
    ;;
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  status)
    status_all
    ;;
  logs)
    shift || true
    logs_cmd "${1:-}"
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
