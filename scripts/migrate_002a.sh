#!/usr/bin/env bash
# migrate_002a.sh — Phase 002-A: Create canonical ~/.hermes/users/{id}/ directory tree.
#
# What this does:
#   1. Creates the full user-namespaced directory tree under ~/.hermes/users/{USER_ID}/
#   2. Creates shared system directories (system/skills/, runtime/sessions/, teams/)
#   3. Copies (not moves) memory files to the new canonical location
#      — the legacy ~/.hermes/memories/ path is preserved as a fallback
#   4. Writes ~/.hermes/system/mcp-servers.json stub (populated in Phase C)
#   5. Writes credentials/README.md explaining the ref-only contract
#
# Usage:
#   bash scripts/migrate_002a.sh
#   HERMES_USER_ID=alice bash scripts/migrate_002a.sh
#
# Idempotent: safe to run multiple times. Existing files are never overwritten.
#
# After running this script, set HERMES_USER_ID in your environment (or
# config.yaml) to activate the new canonical memory path automatically.

set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
USER_ID="${HERMES_USER_ID:-blake}"
USER_HOME="$HERMES_HOME/users/$USER_ID"

echo "=== Hermes Plan 002-A Migration ==="
echo "HERMES_HOME : $HERMES_HOME"
echo "USER_ID     : $USER_ID"
echo "USER_HOME   : $USER_HOME"
echo ""

# ── 1. Create user namespace directory tree ───────────────────────────────────
echo "Creating user directory tree..."
mkdir -p \
  "$USER_HOME/memory" \
  "$USER_HOME/skills" \
  "$USER_HOME/plans/active" \
  "$USER_HOME/plans/archive" \
  "$USER_HOME/artifacts/sessions" \
  "$USER_HOME/artifacts/projects" \
  "$USER_HOME/credentials"

# ── 2. Create shared system directories ──────────────────────────────────────
echo "Creating system directories..."
mkdir -p \
  "$HERMES_HOME/system/skills" \
  "$HERMES_HOME/runtime/sessions" \
  "$HERMES_HOME/teams"

# ── 3. Copy memory files (not move — legacy path preserved as fallback) ───────
LEGACY_MEMORIES="$HERMES_HOME/memories"
CANONICAL_MEMORY="$USER_HOME/memory"

if [ -d "$LEGACY_MEMORIES" ]; then
  copied=0
  for fname in MEMORY.md USER.md; do
    src="$LEGACY_MEMORIES/$fname"
    dst="$CANONICAL_MEMORY/$fname"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
      cp "$src" "$dst"
      echo "  Copied: memories/$fname → users/$USER_ID/memory/$fname"
      copied=$((copied + 1))
    elif [ -f "$dst" ]; then
      echo "  Skipped: users/$USER_ID/memory/$fname already exists"
    fi
  done
  if [ "$copied" -eq 0 ]; then
    echo "  No new memory files to copy (all already present or source missing)"
  fi
  echo "  Legacy path preserved: $LEGACY_MEMORIES"
else
  echo "  No legacy memories/ directory found — skipping memory copy"
fi

# ── 4. Write system/mcp-servers.json stub ────────────────────────────────────
MCP_SERVERS_FILE="$HERMES_HOME/system/mcp-servers.json"
if [ ! -f "$MCP_SERVERS_FILE" ]; then
  printf '{}' > "$MCP_SERVERS_FILE"
  echo "Created: system/mcp-servers.json (empty stub, populated in Phase C)"
else
  echo "Skipped: system/mcp-servers.json already exists"
fi

# ── 5. Write credentials/README.md ───────────────────────────────────────────
CREDS_README="$USER_HOME/credentials/README.md"
if [ ! -f "$CREDS_README" ]; then
  cat > "$CREDS_README" << 'CREDS_EOF'
# Credentials Directory — Refs Only

This directory contains ONLY credential references (URIs pointing to
Keychain or Secrets Manager entries) — never actual secret values.

## Format

Each `.ref` file is a JSON object specifying how to resolve the credential:

```json
{ "type": "keychain", "service": "hermes/blake/gmail", "account": "blake" }
{ "type": "env",      "var": "GMAIL_TOKEN" }
{ "type": "secrets-manager", "arn": "arn:aws:secretsmanager:us-east-1:..." }
```

File names correspond to MCP server names (e.g. `gmail.ref`, `github.ref`).

## Contract

- CredentialResolver (Phase 002-C) reads these refs and resolves them
  to live values in-process memory only.
- Resolved values are NEVER written to disk during a session.
- On session close, the in-memory cache is shredded.
- These files are safe to commit — they contain no secret values.

## Adding a new credential

```bash
# Keychain (macOS local)
echo '{"type":"keychain","service":"hermes/blake/mymcp","account":"blake"}' \
  > ~/.hermes/users/blake/credentials/mymcp.ref

# Environment variable (useful for CI/CD)
echo '{"type":"env","var":"MY_MCP_TOKEN"}' \
  > ~/.hermes/users/blake/credentials/mymcp.ref
```
CREDS_EOF
  echo "Created: users/$USER_ID/credentials/README.md"
else
  echo "Skipped: credentials/README.md already exists"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 002-A migration complete ==="
echo "Tree rooted at: $USER_HOME"
echo ""
echo "Next step: set HERMES_USER_ID=$USER_ID in your shell or config.yaml"
echo "to activate the new canonical memory path."
echo ""
echo "Verify with:"
echo "  find $USER_HOME -type d | sort"
echo "  python3 -c \"from hermes_constants import get_user_home; print(get_user_home('$USER_ID'))\""
