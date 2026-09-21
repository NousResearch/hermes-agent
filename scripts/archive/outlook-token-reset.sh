#!/usr/bin/env bash
set -e

NODE="/home/kensei/.local/bin/node"
SERVER="/home/kensei/.hermes/node/bin/ms-365-mcp-server"
CONFIGURED_CACHE="$HOME/.config/ms-365-mcp-server/token-cache.json"
FALLBACK_CACHE="/home/kensei/.hermes/node/lib/node_modules/@softeria/ms-365-mcp-server/.token-cache.json"
SELECTED_ACCOUNT="$HOME/.config/ms-365-mcp-server/.selected-account.json"

export MS365_MCP_TOKEN_CACHE_PATH=~/.config/ms-365-mcp-server/token-cache.json

echo "======================================"
echo "  Outlook MCP — Full Token Reset"
echo "======================================"
echo ""
echo "This will clear all cached tokens and re-authenticate"
echo "all 4 Outlook accounts with device code flow."
echo ""
echo "For each account you will need to:"
echo "  1. Open https://login.microsoft.com/device"
echo "  2. Sign in with the target email (switch account if needed)"
echo "  3. Enter the code shown below"
echo "  4. Approve ALL requested scopes (including User.Read)"
echo ""
echo "Press Enter to begin..."
read -r

# --- Step 1: Clear everything ---
echo ""
echo "--- Clearing token caches and account selection ---"
> "$CONFIGURED_CACHE" && echo "Cleared: $CONFIGURED_CACHE"
> "$FALLBACK_CACHE" && echo "Cleared: $FALLBACK_CACHE"
> "$SELECTED_ACCOUNT" && echo "Cleared: $SELECTED_ACCOUNT"

echo ""
echo "Cache cleared. Starting device code auth for each account..."
echo ""

# --- Step 2: Re-auth all 4 accounts ---
ACCOUNTS=(
  "sahil_ss@outlook.com"
  "sahil_ss9@hotmail.com"
  "sahil_saghir@hotmail.co.uk"
  "matchdaymaestro@outlook.com"
)

for email in "${ACCOUNTS[@]}"; do
  echo ""
  echo "╔══════════════════════════════════════════╗"
  echo "║  Account: $email"
  echo "╚══════════════════════════════════════════╝"
  echo ""
  echo "Starting device code login..."
  echo "⚠️  IMPORTANT: sign in as $email (switch account if you're logged into another one)"
  echo "⚠️  Approve ALL requested scopes — including User.Read"
  echo ""

  "$NODE" "$SERVER" --login --preset mail,calendar \
    --enabled-tools "mail|calendar|user|get-current-user" -v

  echo ""
  echo "✅ Auth complete for $email"
  echo ""

  # Verify login for this account
  echo "--- Verifying login for $email ---"
  MS365_MCP_TOKEN_CACHE_PATH=~/.config/ms-365-mcp-server/token-cache.json \
    "$NODE" "$SERVER" --verify-login --preset mail,calendar -v || echo "⚠️  Verify returned non-zero (check output above)"
  echo ""

  # Confirm with user before continuing
  echo "Press Enter to continue to the next account, or Ctrl+C to stop..."
  read -r
done

# --- Step 3: Final summary ---
echo ""
echo "======================================"
echo "  All accounts authenticated"
echo "======================================"
echo ""
echo "Listing cached accounts:"
MS365_MCP_TOKEN_CACHE_PATH=~/.config/ms-365-mcp-server/token-cache.json \
  "$NODE" "$SERVER" --list-accounts --preset mail,calendar 2>/dev/null || echo "(list-accounts failed — run manually to verify)"
echo ""
echo "DONE. Restart the Hermes gateway to pick up the new tokens:"
echo "  hermes restart"
