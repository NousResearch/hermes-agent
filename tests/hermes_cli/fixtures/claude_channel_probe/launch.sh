#!/bin/sh
set -eu

# Launch exactly one reviewed disposable session. This script must be run only
# after prepare.sh and independent review have passed.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
: "${ESN0407_MCP_CONFIG:?set ESN0407_MCP_CONFIG to the absolute disposable config path}"
: "${ESN0407_WORK_DIR:?set ESN0407_WORK_DIR to the absolute empty disposable working directory}"

case "$ESN0407_MCP_CONFIG" in /*) ;; *) echo "MCP config path must be absolute" >&2; exit 2 ;; esac
case "$ESN0407_WORK_DIR" in /*) ;; *) echo "work directory must be absolute" >&2; exit 2 ;; esac
[ -f "$ESN0407_MCP_CONFIG" ] || { echo "MCP config is not a regular file" >&2; exit 2; }
[ -d "$ESN0407_WORK_DIR" ] || { echo "work directory does not exist" >&2; exit 2; }
[ -z "$(find "$ESN0407_WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ] || {
  echo "work directory must be empty" >&2
  exit 2
}

unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
unset GOOGLE_APPLICATION_CREDENTIALS ANTHROPIC_VERTEX_PROJECT_ID
unset AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_TENANT_ID

auth_json="$(claude auth status --json)"
printf '%s' "$auth_json" | node "$script_dir/preflight.mjs" auth

cd "$ESN0407_WORK_DIR"
exec claude --bg --restricted --permission-mode dontAsk \
  --setting-sources "" --strict-mcp-config \
  --mcp-config "$ESN0407_MCP_CONFIG" \
  --settings '{"crossSessionInbound":"refuse"}' \
  --tools mcp__esn0407_channel__reply \
  --disable-slash-commands --no-chrome \
  --dangerously-load-development-channels server:esn0407_channel \
  --name esn-0407-disposable-channel
