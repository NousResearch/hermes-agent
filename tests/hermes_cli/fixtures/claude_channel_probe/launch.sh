#!/bin/sh
set -eu

# Launch exactly one reviewed foreground session. Custom development Channels
# require an interactive Claude confirmation before they register, so this
# process must remain attached to an operator terminal until that gate is
# accepted. Only then may the separate sender submit the bounded question.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
: "${ESN0407_MCP_CONFIG:?set ESN0407_MCP_CONFIG to the absolute disposable config path}"
: "${ESN0407_WORK_DIR:?set ESN0407_WORK_DIR to the absolute stable empty managed working directory}"
: "${ESN0407_SESSION_ID:?set ESN0407_SESSION_ID to the pre-registered provider session UUID}"
: "${ESN0407_PROFILE_ROOT:?set ESN0407_PROFILE_ROOT to the reviewed durable provider-auth profile}"
: "${CLAUDE_CONFIG_DIR:?set CLAUDE_CONFIG_DIR to the reviewed durable Claude config directory}"

case "$ESN0407_MCP_CONFIG" in /*) ;; *) echo "MCP config path must be absolute" >&2; exit 2 ;; esac
case "$ESN0407_WORK_DIR" in /*) ;; *) echo "work directory must be absolute" >&2; exit 2 ;; esac
node -e 'if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(process.argv[1])) process.exit(2)' \
  "$ESN0407_SESSION_ID" || { echo "session id must be a lowercase UUIDv4" >&2; exit 2; }
[ -f "$ESN0407_MCP_CONFIG" ] || { echo "MCP config is not a regular file" >&2; exit 2; }
[ -d "$ESN0407_WORK_DIR" ] || { echo "work directory does not exist" >&2; exit 2; }
[ -z "$(find "$ESN0407_WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ] || {
  echo "work directory must be empty" >&2
  exit 2
}
node "$script_dir/preflight.mjs" profile \
  "$ESN0407_PROFILE_ROOT" "$CLAUDE_CONFIG_DIR" "$ESN0407_WORK_DIR"

lock_dir="$ESN0407_PROFILE_ROOT/.esn0407-channel.lock"
owner_file="$lock_dir/owner"
self_owner="$$ $ESN0407_SESSION_ID"
if ! mkdir "$lock_dir" 2>/dev/null; then
  [ ! -L "$lock_dir" ] && [ -d "$lock_dir" ] && [ ! -L "$owner_file" ] && [ -f "$owner_file" ] || {
    echo "managed profile lease is malformed; refusing recovery" >&2
    exit 2
  }
  old_owner="$(cat "$owner_file")"
  old_pid="${old_owner%% *}"
  case "$old_pid" in *[!0-9]*|'') echo "managed profile lease has an invalid owner" >&2; exit 2 ;; esac
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "managed provider-auth profile already has a live launcher" >&2
    exit 2
  fi
  stale_dir="$ESN0407_PROFILE_ROOT/.esn0407-channel.stale.$$"
  if ! mv "$lock_dir" "$stale_dir" 2>/dev/null; then
    echo "managed profile lease changed during stale-owner recovery" >&2
    exit 2
  fi
  [ ! -L "$stale_dir/owner" ] && [ "$(cat "$stale_dir/owner")" = "$old_owner" ] || {
    echo "stale managed profile lease changed during recovery" >&2
    exit 2
  }
  rm "$stale_dir/owner"
  rmdir "$stale_dir"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "another launcher acquired the managed profile during recovery" >&2
    exit 2
  fi
fi
chmod 700 "$lock_dir"
(umask 077 && printf '%s\n' "$self_owner" > "$owner_file")
cleanup_profile_lease() {
  if [ ! -L "$owner_file" ] && [ -f "$owner_file" ] && [ "$(cat "$owner_file")" = "$self_owner" ]; then
    rm "$owner_file"
    rmdir "$lock_dir"
  fi
}
claude_pid=""
terminate_launcher() {
  code="$1"
  trap - EXIT HUP INT TERM
  if [ -n "$claude_pid" ] && kill -0 "$claude_pid" 2>/dev/null; then
    kill -TERM "$claude_pid" 2>/dev/null || true
    wait "$claude_pid" 2>/dev/null || true
  fi
  cleanup_profile_lease
  exit "$code"
}
trap cleanup_profile_lease EXIT
trap 'terminate_launcher 129' HUP
trap 'terminate_launcher 130' INT
trap 'terminate_launcher 143' TERM

unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
unset GOOGLE_APPLICATION_CREDENTIALS ANTHROPIC_VERTEX_PROJECT_ID
unset AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_TENANT_ID

auth_json="$(claude auth status --json)"
printf '%s' "$auth_json" | node "$script_dir/preflight.mjs" auth
active_sessions_json="$(claude agents --json)"
printf '%s' "$active_sessions_json" | node "$script_dir/preflight.mjs" sessions

cd "$ESN0407_WORK_DIR"
# An asynchronous command in a non-interactive POSIX shell may otherwise read
# from /dev/null. Preserve the launcher's terminal input explicitly so Claude's
# mandatory development-Channel confirmation remains a human interaction.
exec 3<&0
claude --restricted --permission-mode dontAsk \
  --session-id "$ESN0407_SESSION_ID" \
  --setting-sources "" --strict-mcp-config \
  --mcp-config "$ESN0407_MCP_CONFIG" \
  --settings '{"crossSessionInbound":"refuse"}' \
  --tools mcp__esn0407_channel__reply \
  --allowedTools mcp__esn0407_channel__reply \
  --disable-slash-commands --no-chrome \
  --dangerously-load-development-channels server:esn0407_channel \
  --name esn-0407-disposable-channel <&3 &
claude_pid="$!"
exec 3<&-
if wait "$claude_pid"; then
  claude_status=0
else
  claude_status="$?"
fi
claude_pid=""
exit "$claude_status"
