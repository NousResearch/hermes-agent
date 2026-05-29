#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$PACKAGE_ROOT/../.." 2>/dev/null && pwd || printf '%s\n' "$PACKAGE_ROOT")"

target="${1:-$PACKAGE_ROOT}"
if [ -d "$target/config" ]; then
  CONFIG_DIR="$target/config"
else
  CONFIG_DIR="$target"
fi

ENV_EXAMPLE_FILE="$CONFIG_DIR/.env.example"
ENV_RUNTIME_FILE="$CONFIG_DIR/.env"
if [ -f "$ENV_RUNTIME_FILE" ]; then
  ENV_FILE="$ENV_RUNTIME_FILE"
  ENV_MODE="runtime"
else
  ENV_FILE="$ENV_EXAMPLE_FILE"
  ENV_MODE="template"
fi

CONFIG_FILE="$CONFIG_DIR/config.example.yaml"
SOUL_FILE="$CONFIG_DIR/SOUL.example.md"
TOOL_POLICY_FILE="$CONFIG_DIR/tool-policy.example.yaml"

failures=0

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  failures=$((failures + 1))
}

pass() {
  printf 'OK: %s\n' "$1"
}

active_lines() {
  sed -e '/^[[:space:]]*#/d' -e 's/[[:space:]]#.*$//' "$1"
}

normalize_value() {
  printf '%s' "$1" |
    sed -e 's/^[[:space:]]*//' \
      -e 's/[[:space:]]*$//' \
      -e 's/^"//' \
      -e 's/"$//' \
      -e "s/^'//" \
      -e "s/'$//"
}

lower_value() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

env_value() {
  key="$1"
  raw="$(
    awk -v key="$key" '
      /^[[:space:]]*($|#)/ { next }
      {
        line = $0
        sub(/^[[:space:]]*/, "", line)
        if (index(line, key "=") == 1) {
          print substr(line, length(key) + 2)
          found = 1
          exit
        }
      }
      END { if (!found) exit 1 }
    ' "$ENV_FILE"
  )" || return 1
  normalize_value "$raw"
}

is_placeholder_value() {
  value="$(normalize_value "$1")"
  lower="$(lower_value "$value")"

  if [ -z "$value" ]; then
    return 0
  fi
  case "$value" in
    \<*\>)
      return 0
      ;;
  esac
  case "$lower" in
    replace-with*|*replace-with*|changeme|*changeme*|example|*example*|*placeholder*|dummy|*dummy*|fake|*fake*|todo|*todo*|your-*|*your-*)
      return 0
      ;;
  esac
  return 1
}

is_angle_placeholder() {
  value="$(normalize_value "$1")"
  case "$value" in
    \<*\>)
      return 0
      ;;
  esac
  return 1
}

require_file() {
  file="$1"
  if [ -f "$file" ]; then
    pass "found $(basename "$file")"
  else
    fail "missing $file"
  fi
}

required_env_value() {
  key="$1"
  if [ ! -f "$ENV_FILE" ]; then
    fail "missing env file: $ENV_FILE"
    return 1
  fi
  if value="$(env_value "$key")"; then
    if [ -n "$value" ]; then
      printf '%s' "$value"
      return 0
    fi
  fi
  fail "missing or empty env key: $key"
  return 1
}

require_env_key() {
  key="$1"
  if required_env_value "$key" >/dev/null; then
    pass "env key present: $key"
  fi
}

require_boolean_env() {
  key="$1"
  expected="$2"
  if ! value="$(required_env_value "$key")"; then
    return 0
  fi
  lower="$(lower_value "$value")"
  case "$expected:$lower" in
    true:true|true:1|true:yes)
      pass "$key enabled"
      ;;
    false:false|false:0|false:no)
      pass "$key disabled"
      ;;
    *)
      fail "$key must be $expected"
      ;;
  esac
}

require_runtime_value() {
  key="$1"
  if ! value="$(required_env_value "$key")"; then
    return 0
  fi
  if is_placeholder_value "$value"; then
    fail "$key still has a placeholder or example value"
  else
    pass "$key has a non-placeholder value"
  fi
}

require_template_placeholder() {
  key="$1"
  if ! value="$(required_env_value "$key")"; then
    return 0
  fi
  if is_angle_placeholder "$value"; then
    pass "$key uses an angle-bracket placeholder"
  else
    fail "$key must use an angle-bracket placeholder in .env.example"
  fi
}

contains_broad_value() {
  value="$(lower_value "$1")"
  printf '%s\n' "$value" | tr ',; ' '\n' |
    grep -Eiq '^(\*|all|any|everyone|@everyone|public|0|none)$'
}

require_discord_id_list() {
  key="$1"
  if ! value="$(required_env_value "$key")"; then
    return 0
  fi
  if [ "$ENV_MODE" = "template" ] && is_angle_placeholder "$value"; then
    pass "$key uses an angle-bracket allowlist placeholder"
    return 0
  fi
  if contains_broad_value "$value"; then
    fail "$key contains a broad allowlist value"
  elif [ "$ENV_MODE" = "runtime" ] && ! printf '%s\n' "$value" | grep -Eq '[0-9]{15,25}'; then
    fail "$key must contain explicit Discord snowflake IDs"
  else
    pass "$key uses explicit allowlist values"
  fi
}

check_webhook_secret() {
  if ! secret="$(required_env_value WEBHOOK_SECRET)"; then
    return 0
  fi
  if [ "$ENV_MODE" = "template" ]; then
    require_template_placeholder WEBHOOK_SECRET
    return 0
  fi
  if is_placeholder_value "$secret"; then
    fail "WEBHOOK_SECRET still has a placeholder or example value"
    return 0
  fi
  if [ "${#secret}" -lt 32 ]; then
    fail "WEBHOOK_SECRET must be at least 32 characters"
  elif printf '%s\n' "$secret" | grep -Eq '^(.)\1+$'; then
    fail "WEBHOOK_SECRET must not repeat one character"
  elif printf '%s\n' "$(lower_value "$secret")" | grep -Eiq '(webhook|secret|hmac|changeme|example|placeholder|replace|dummy|fake|test)'; then
    fail "WEBHOOK_SECRET is an obvious placeholder"
  else
    pass "WEBHOOK_SECRET strength check passed"
  fi
}

check_hermes_home() {
  if [ "$ENV_MODE" = "template" ]; then
    require_template_placeholder HERMES_HOME
    return 0
  fi
  if ! home="$(required_env_value HERMES_HOME)"; then
    return 0
  fi
  if is_placeholder_value "$home"; then
    fail "HERMES_HOME still has a placeholder or example value"
    return 0
  fi
  case "$home" in
    "~/.hermes"|~/.hermes/*|"\$HOME/.hermes"|"\${HOME}/.hermes"|/Users/leo/.hermes|/Users/leo/.hermes/*|/root|/root/*|/)
      fail "HERMES_HOME points at an unsafe or live runtime path"
      return 0
      ;;
  esac
  if [ "${home#/}" = "$home" ] && [ "${home#\~}" = "$home" ]; then
    fail "HERMES_HOME must be an absolute fresh package path"
    return 0
  fi
  if [ -e "$home" ]; then
    resolved="$(CDPATH= cd -- "$home" 2>/dev/null && pwd -P || printf '%s\n' "$home")"
  else
    resolved="$home"
  fi
  case "$resolved/" in
    "$REPO_ROOT/"*|"$PACKAGE_ROOT/"*|"$(pwd -P)/"*)
      fail "HERMES_HOME must not point inside the current repo or package"
      ;;
    *)
      pass "HERMES_HOME path is isolated from live and repo paths"
      ;;
  esac
}

check_no_legacy_template_placeholders() {
  for file in "$ENV_EXAMPLE_FILE" "$CONFIG_FILE"; do
    if [ ! -f "$file" ]; then
      continue
    fi
    if active_lines "$file" | grep -Eiq 'replace-with|changeme|example\.invalid|placeholder'; then
      fail "$(basename "$file") contains legacy placeholder text"
    else
      pass "$(basename "$file") uses delivery-safe placeholders"
    fi
  done
}

check_config_allowlists() {
  if [ ! -f "$CONFIG_FILE" ]; then
    return 0
  fi
  if active_lines "$CONFIG_FILE" | grep -Eiq '(allow_all|allow_all_users)[[:space:]]*:[[:space:]]*(true|1|yes)'; then
    fail "allow-all config value is enabled"
  else
    pass "allow-all config values disabled"
  fi
  if active_lines "$CONFIG_FILE" | grep -Eiq 'require_mention[[:space:]]*:[[:space:]]*(false|0|no)'; then
    fail "Discord mention requirement is disabled in config template"
  else
    pass "Discord mention requirement enabled in config template"
  fi
  if active_lines "$CONFIG_FILE" | grep -Eiq '(passive_ingestion|message_history_ingestion|listen_all)[[:space:]]*:[[:space:]]*(true|1|yes)'; then
    fail "passive ingestion is enabled"
  else
    pass "passive ingestion disabled"
  fi
  if active_lines "$CONFIG_FILE" | grep -Eiq 'free_response_channels[[:space:]]*:[[:space:]]*[^[:space:]\[]'; then
    fail "free-response channels are enabled in config template"
  else
    pass "free-response channels empty in config template"
  fi
  if awk '
    /^[^[:space:]][^:]*:/ { section = "" }
    /^[[:space:]]*(allowed_users|allowed_channels):/ { section = $1 }
    section != "" && /^[[:space:]]*-[[:space:]]*/ {
      value = tolower($0)
      if (value ~ /["'\'']?(\*|all|any|everyone|@everyone|public)["'\'']?/) found = 1
    }
    END { exit found ? 0 : 1 }
  ' "$CONFIG_FILE"; then
    fail "config allowlists contain broad user or channel values"
  else
    pass "config allowlists avoid broad user and channel values"
  fi
}

check_attachment_caps() {
  if [ ! -f "$CONFIG_FILE" ]; then
    return 0
  fi
  large_caps="$(
    awk -F: '
      /max_.*bytes[[:space:]]*:/ {
        value = $2
        gsub(/[^0-9]/, "", value)
        if (value != "" && value + 0 > 33554432) {
          print $1 ":" value
        }
      }
    ' "$CONFIG_FILE"
  )"
  if [ -n "$large_caps" ]; then
    fail "attachment byte cap exceeds 33554432: $large_caps"
  else
    pass "attachment byte caps are bounded"
  fi
}

config_number() {
  key="$1"
  awk -F: -v key="$key" '
    $1 ~ key {
      value = $2
      gsub(/[^0-9]/, "", value)
      if (value != "") print value
      exit
    }
  ' "$CONFIG_FILE"
}

check_webhook_policy() {
  if [ ! -f "$CONFIG_FILE" ]; then
    return 0
  fi
  if active_lines "$CONFIG_FILE" | grep -Eq 'require_signature:[[:space:]]*true' &&
    active_lines "$CONFIG_FILE" | grep -Eq 'signature_algorithm:[[:space:]]*"hmac-sha256"' &&
    active_lines "$CONFIG_FILE" | grep -Eq 'signature_header:[[:space:]]*"X-Dobby-Signature"' &&
    active_lines "$CONFIG_FILE" | grep -Eq 'timestamp_header:[[:space:]]*"X-Dobby-Timestamp"' &&
    active_lines "$CONFIG_FILE" | grep -Eq 'unsigned_requests:[[:space:]]*"deny"' &&
    active_lines "$CONFIG_FILE" | grep -Eq 'replay_cache_required:[[:space:]]*true'; then
    pass "webhook signature, timestamp, replay, and unsigned-deny policy present"
  else
    fail "webhook policy must require HMAC signature, timestamp, replay cache, and unsigned-deny"
  fi
  replay_window="$(config_number 'replay_window_seconds')"
  if [ -n "$replay_window" ] && [ "$replay_window" -gt 0 ] && [ "$replay_window" -le 300 ]; then
    pass "webhook replay window is bounded"
  else
    fail "webhook replay window must be between 1 and 300 seconds"
  fi
  max_body="$(config_number 'max_body_bytes')"
  if [ -n "$max_body" ] && [ "$max_body" -gt 0 ] && [ "$max_body" -le 1048576 ]; then
    pass "webhook body size is bounded"
  else
    fail "webhook body size must be between 1 and 1048576 bytes"
  fi
  if awk '
    /^[^[:space:]][^:]*:/ && $1 !~ /^webhook:/ { in_webhook = 0 }
    /^webhook:/ { in_webhook = 1 }
    in_webhook && /^[[:space:]]*allowed_routes:/ { in_routes = 1; next }
    in_routes && /^[[:space:]]*-[[:space:]]*["'\'']\/[^"'\'']+["'\'']/ { found = 1 }
    in_routes && /^[[:space:]]*-[[:space:]]*["'\'']?\*["'\'']?/ { broad = 1 }
    END { exit (found && !broad) ? 0 : 1 }
  ' "$CONFIG_FILE"; then
    pass "webhook route allowlist present"
  else
    fail "webhook route allowlist must contain explicit non-wildcard routes"
  fi
}

check_tool_policy() {
  if [ ! -f "$TOOL_POLICY_FILE" ]; then
    return 0
  fi
  if active_lines "$TOOL_POLICY_FILE" | grep -Eiq 'default_action[[:space:]]*:[[:space:]]*"?allow"?'; then
    fail "tool policy must deny by default"
  else
    pass "tool policy denies by default"
  fi
  if active_lines "$TOOL_POLICY_FILE" | grep -Eiq 'default_enabled[[:space:]]*:[[:space:]]*(true|1|yes)'; then
    fail "tool policy must not enable capabilities by default"
  else
    pass "tool policy keeps broad capabilities default-off"
  fi
  if active_lines "$TOOL_POLICY_FILE" | grep -Eiq '(premium|experimental)[^#]*(enabled|enabled:)[[:space:]]*(true|1|yes)'; then
    fail "premium or experimental capability is enabled"
  else
    pass "premium and experimental capabilities disabled"
  fi
  for required in broad_oauth github notion google slack home_automation autonomous_actions purchase trade send_email post_public_message merge_pull_request deploy browser_automation default_enabled; do
    if active_lines "$TOOL_POLICY_FILE" | grep -Eq "$required"; then
      pass "tool policy contains $required"
    else
      fail "tool policy missing $required deny/default-off control"
    fi
  done
}

check_redaction() {
  if [ ! -f "$SCRIPT_DIR/redaction-check.sh" ]; then
    fail "missing redaction-check.sh"
    return 0
  fi
  paths=()
  for file in "$ENV_EXAMPLE_FILE" "$CONFIG_FILE" "$SOUL_FILE" "$TOOL_POLICY_FILE"; do
    if [ -f "$file" ]; then
      paths+=("$file")
    fi
  done
  bash "$SCRIPT_DIR/redaction-check.sh" "${paths[@]}"
}

require_file "$ENV_EXAMPLE_FILE"
if [ "$ENV_FILE" != "$ENV_EXAMPLE_FILE" ]; then
  require_file "$ENV_FILE"
fi
require_file "$CONFIG_FILE"
require_file "$SOUL_FILE"
require_file "$TOOL_POLICY_FILE"
pass "preflight env mode: $ENV_MODE"

for key in \
  HERMES_HOME \
  HERMES_INFERENCE_PROVIDER \
  OPENAI_BASE_URL \
  OPENAI_API_KEY \
  HERMES_MODEL \
  DISCORD_CLIENT_ID \
  DISCORD_BOT_TOKEN \
  DISCORD_HOME_CHANNEL \
  DISCORD_ALLOWED_USERS \
  DISCORD_ALLOWED_CHANNELS \
  DISCORD_REQUIRE_MENTION \
  DISCORD_ALLOW_ALL_USERS \
  GATEWAY_ALLOW_ALL_USERS \
  WEBHOOK_SECRET \
  HERMES_REDACT_SECRETS; do
  require_env_key "$key"
done

if [ "$ENV_MODE" = "template" ]; then
  for key in \
    HERMES_HOME \
    OPENAI_BASE_URL \
    OPENAI_API_KEY \
    HERMES_MODEL \
    DISCORD_CLIENT_ID \
    DISCORD_BOT_TOKEN \
    DISCORD_HOME_CHANNEL \
    DISCORD_ALLOWED_USERS \
    DISCORD_ALLOWED_CHANNELS \
    HERMES_WRITE_SAFE_ROOT; do
    require_template_placeholder "$key"
  done
else
  for key in \
    HERMES_HOME \
    OPENAI_BASE_URL \
    OPENAI_API_KEY \
    HERMES_MODEL \
    DISCORD_CLIENT_ID \
    DISCORD_BOT_TOKEN \
    DISCORD_HOME_CHANNEL \
    DISCORD_ALLOWED_USERS \
    DISCORD_ALLOWED_CHANNELS; do
    require_runtime_value "$key"
  done
fi

require_boolean_env DISCORD_REQUIRE_MENTION true
require_boolean_env DISCORD_ALLOW_ALL_USERS false
require_boolean_env GATEWAY_ALLOW_ALL_USERS false
require_boolean_env HERMES_REDACT_SECRETS true
require_discord_id_list DISCORD_ALLOWED_USERS
require_discord_id_list DISCORD_ALLOWED_CHANNELS
check_webhook_secret
check_hermes_home
check_no_legacy_template_placeholders
check_config_allowlists
check_attachment_caps
check_webhook_policy
check_tool_policy

if [ -f "$SOUL_FILE" ] && active_lines "$SOUL_FILE" | grep -Eiq 'my name is|home address|phone number|birthday|ssn|social security'; then
  fail "SOUL template appears to contain personal-fact language"
else
  pass "SOUL template contains no obvious personal facts"
fi

check_redaction

if [ "$failures" -ne 0 ]; then
  printf 'preflight: %s check(s) failed\n' "$failures" >&2
  exit 1
fi

printf 'preflight: all local %s checks passed\n' "$ENV_MODE"
