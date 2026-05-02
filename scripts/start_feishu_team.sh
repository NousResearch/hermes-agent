#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEAM_PREFIX="${TEAM_PREFIX:-feishu}"
SOURCE_PROFILE="${SOURCE_PROFILE:-default}"
PROFILES_ROOT="${HERMES_PROFILES_ROOT:-$HOME/.hermes/profiles}"
LOG_DIR="${FEISHU_TEAM_LOG_DIR:-$PROJECT_ROOT/.feishu-team-logs}"
PYTHON_BIN="${HERMES_PYTHON:-}"
FEISHU_TEAM_MEMORY_PROVIDER="${FEISHU_TEAM_MEMORY_PROVIDER-memtensor}"
FEISHU_TEAM_MEMORY_ENABLED="${FEISHU_TEAM_MEMORY_ENABLED-true}"
FEISHU_TEAM_USER_PROFILE_ENABLED="${FEISHU_TEAM_USER_PROFILE_ENABLED-true}"
FEISHU_TEAM_MEMORY_REQUIRED="${FEISHU_TEAM_MEMORY_REQUIRED-false}"
MEMOS_STATE_DIR_DEFAULT="${MEMOS_STATE_DIR:-$HOME/.hermes/memos-state}"
MEMOS_DAEMON_PORT_DEFAULT="${MEMOS_DAEMON_PORT:-18992}"
MEMOS_VIEWER_PORT_DEFAULT="${MEMOS_VIEWER_PORT:-18901}"
MEMOS_EMBEDDING_PROVIDER_DEFAULT="${MEMOS_EMBEDDING_PROVIDER:-local}"

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
  scripts/start_feishu_team.sh memory-install

Environment overrides:
  TEAM_PREFIX=feishu
  SOURCE_PROFILE=default
  HERMES_PYTHON=/path/to/python
  HERMES_PROFILES_ROOT=$HOME/.hermes/profiles
  FEISHU_TEAM_LOG_DIR=./.feishu-team-logs
  FEISHU_TEAM_MEMORY_PROVIDER=memtensor
  FEISHU_TEAM_MEMORY_ENABLED=true
  FEISHU_TEAM_USER_PROFILE_ENABLED=true
  FEISHU_TEAM_MEMORY_REQUIRED=false
  MEMOS_STATE_DIR=$HOME/.hermes/memos-state
  MEMOS_DAEMON_PORT=18992
  MEMOS_VIEWER_PORT=18901
  MEMOS_EMBEDDING_PROVIDER=local

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
  "$PYTHON_BIN" - "$dir" "$FEISHU_TEAM_MEMORY_PROVIDER" "$FEISHU_TEAM_MEMORY_ENABLED" "$FEISHU_TEAM_USER_PROFILE_ENABLED" <<'PY'
from pathlib import Path
import sys
import yaml

profile_dir = Path(sys.argv[1])
memory_provider = sys.argv[2].strip()
memory_enabled = sys.argv[3].strip().lower() not in {"", "0", "false", "no", "off"}
user_profile_enabled = sys.argv[4].strip().lower() not in {"", "0", "false", "no", "off"}
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

memory = data.setdefault("memory", {})
memory["memory_enabled"] = memory_enabled
memory["user_profile_enabled"] = user_profile_enabled
if memory_provider:
    memory["provider"] = memory_provider
else:
    memory.pop("provider", None)

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
FEISHU_TEAMROOM_ROLE=$role
FEISHU_TEAMROOM_REPLY_TIMEOUT_SECONDS=300
EOF
  if [[ "$FEISHU_TEAM_MEMORY_PROVIDER" == "memtensor" ]]; then
    cat >> "$dir/.env.feishu-team.example" <<EOF
MEMOS_STATE_DIR=$MEMOS_STATE_DIR_DEFAULT
MEMOS_DAEMON_PORT=$MEMOS_DAEMON_PORT_DEFAULT
MEMOS_VIEWER_PORT=$MEMOS_VIEWER_PORT_DEFAULT
MEMOS_EMBEDDING_PROVIDER=$MEMOS_EMBEDDING_PROVIDER_DEFAULT
MEMOS_EMBEDDING_API_KEY=
MEMOS_EMBEDDING_ENDPOINT=
MEMOS_BRIDGE_CONFIG=
MEMOS_BRIDGE_SCRIPT=
EOF
  fi
  if [[ "$role" == "coordinator" ]]; then
    for teammate in researcher builder reviewer; do
      local upper
      upper="$(printf '%s' "$teammate" | tr '[:lower:]' '[:upper:]')"
      cat >> "$dir/.env.feishu-team.example" <<EOF
FEISHU_TEAMROOM_${upper}_NAME=$(bot_display_name "$teammate")
FEISHU_TEAMROOM_${upper}_OPEN_ID=
FEISHU_TEAMROOM_${upper}_USER_ID=
EOF
    done
  fi
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
  set_env_key "$env_file" "FEISHU_TEAMROOM_ROLE" "$role"
  set_env_key "$env_file" "FEISHU_TEAMROOM_REPLY_TIMEOUT_SECONDS" "300"
  if [[ "$FEISHU_TEAM_MEMORY_PROVIDER" == "memtensor" ]]; then
    ensure_env_key "$env_file" "MEMOS_STATE_DIR" "$MEMOS_STATE_DIR_DEFAULT"
    ensure_env_key "$env_file" "MEMOS_DAEMON_PORT" "$MEMOS_DAEMON_PORT_DEFAULT"
    ensure_env_key "$env_file" "MEMOS_VIEWER_PORT" "$MEMOS_VIEWER_PORT_DEFAULT"
    ensure_env_key "$env_file" "MEMOS_EMBEDDING_PROVIDER" "$MEMOS_EMBEDDING_PROVIDER_DEFAULT"
    ensure_env_key "$env_file" "MEMOS_EMBEDDING_API_KEY"
    ensure_env_key "$env_file" "MEMOS_EMBEDDING_ENDPOINT"
    ensure_env_key "$env_file" "MEMOS_BRIDGE_CONFIG"
    ensure_env_key "$env_file" "MEMOS_BRIDGE_SCRIPT"
  fi
  if [[ "$role" == "coordinator" ]]; then
    for teammate in researcher builder reviewer; do
      local upper
      upper="$(printf '%s' "$teammate" | tr '[:lower:]' '[:upper:]')"
      set_env_key "$env_file" "FEISHU_TEAMROOM_${upper}_NAME" "$(bot_display_name "$teammate")"
      set_env_key "$env_file" "FEISHU_TEAMROOM_${upper}_OPEN_ID"
      set_env_key "$env_file" "FEISHU_TEAMROOM_${upper}_USER_ID"
    done
  fi
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

check_memory_provider() {
  local role="$1"
  local provider="$2"
  local dir
  dir="$(profile_dir "$role")"
  (cd "$PROJECT_ROOT" && HERMES_HOME="$dir" "$PYTHON_BIN" - "$provider" <<'PY'
import sys

provider_name = sys.argv[1]
try:
    from plugins.memory import load_memory_provider
    provider = load_memory_provider(provider_name)
    if not provider:
        print("missing")
        raise SystemExit(2)
    if not provider.is_available():
        print("unavailable")
        raise SystemExit(3)
    print("ok")
except SystemExit:
    raise
except Exception:
    print("error")
    raise SystemExit(4)
PY
  )
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
  if [[ -n "$FEISHU_TEAM_MEMORY_PROVIDER" ]]; then
    local provider_status
    provider_status="$(check_memory_provider "$role" "$FEISHU_TEAM_MEMORY_PROVIDER" 2>/dev/null || true)"
    if [[ "$provider_status" == "ok" ]]; then
      printf '  memory.provider=%s ok\n' "$FEISHU_TEAM_MEMORY_PROVIDER"
    else
      printf '  memory.provider=%s %s\n' "$FEISHU_TEAM_MEMORY_PROVIDER" "${provider_status:-missing}"
      printf '  install MemOS: curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/openclaw-local-plugin-20260408/apps/memos-local-plugin/install.sh | bash\n'
      if [[ "${FEISHU_TEAM_MEMORY_REQUIRED,,}" =~ ^(1|true|yes|on)$ ]]; then
        ok=1
      else
        printf '  memory.provider warning only; set FEISHU_TEAM_MEMORY_REQUIRED=true to enforce\n'
      fi
    fi
  fi
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
    printf '%s\n' "Cannot start $(profile_name "$role"): missing required values. Run: scripts/start_feishu_team.sh check"
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

memory_install_cmd() {
  cat <<'EOF'
MemOS / MemTensor memory provider is configured as:
  memory.provider: memtensor

This command only prints instructions; it does not install MemOS.

Install the official local plugin with:
  curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/openclaw-local-plugin-20260408/apps/memos-local-plugin/install.sh | bash

Then run:
  scripts/start_feishu_team.sh init
  scripts/start_feishu_team.sh check
  scripts/start_feishu_team.sh start

Memory Viewer:
  http://127.0.0.1:18901
EOF
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
  memory-install)
    memory_install_cmd
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
