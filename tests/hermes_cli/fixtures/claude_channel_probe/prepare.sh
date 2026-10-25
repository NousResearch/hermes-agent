#!/bin/sh
set -eu

# Run only inside the private disposable copy after independent review passes.
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$script_dir"
node ./preflight.mjs location

unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
unset CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
unset GOOGLE_APPLICATION_CREDENTIALS ANTHROPIC_VERTEX_PROJECT_ID
unset AZURE_CLIENT_ID AZURE_CLIENT_SECRET AZURE_TENANT_ID

auth_json="$(claude auth status --json)"
printf '%s' "$auth_json" | node ./preflight.mjs auth
node ./preflight.mjs lock
npm ci --ignore-scripts --no-audit --no-fund
node ./preflight.mjs lock
