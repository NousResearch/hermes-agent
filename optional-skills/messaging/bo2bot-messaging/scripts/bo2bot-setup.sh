#!/bin/bash
# Bo2bot Credential Setup Script
# Run this interactively to set up or update your Bo2bot credentials
# Usage: bash ~/.hermes/skills/messaging/bo2bot-messaging/scripts/bo2bot-setup.sh

set -e

CREDS_FILE=~/.hermes/secrets/bo2bot.env

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         BO2BOT CREDENTIAL SETUP FOR HERMES AGENT           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if credentials already exist
if [ -f "$CREDS_FILE" ]; then
    echo "Existing credentials found at: $CREDS_FILE"
    echo ""
    read -p "Do you want to update them? (y/n) " -n 1 -r UPDATE
    echo ""
    if [[ ! $UPDATE =~ ^[Yy]$ ]]; then
        echo "✅ Keeping existing credentials."
        exit 0
    fi
fi

echo "Please provide your Bo2bot credentials:"
echo "(Leave empty to keep existing value, if any)"
echo ""

# Get handle
if [ -f "$CREDS_FILE" ]; then
    CURRENT_HANDLE=$(grep "^BO2BOT_HANDLE=" "$CREDS_FILE" | cut -d'=' -f2)
fi
read -p "Bot handle (e.g., @yourname) [$CURRENT_HANDLE]: " HANDLE
HANDLE=${HANDLE:-$CURRENT_HANDLE}

# Get public address
if [ -f "$CREDS_FILE" ]; then
    CURRENT_ADDRESS=$(grep "^BO2BOT_PUBLIC_ADDRESS=" "$CREDS_FILE" | cut -d'=' -f2)
fi
read -p "Public address (e.g., yourname@bo2bot.com) [$CURRENT_ADDRESS]: " ADDRESS
ADDRESS=${ADDRESS:-$CURRENT_ADDRESS}

# Get account ID
if [ -f "$CREDS_FILE" ]; then
    CURRENT_ACCOUNT_ID=$(grep "^BO2BOT_ACCOUNT_ID=" "$CREDS_FILE" | cut -d'=' -f2)
fi
read -p "Account ID (acct_...) [$CURRENT_ACCOUNT_ID]: " ACCOUNT_ID
ACCOUNT_ID=${ACCOUNT_ID:-$CURRENT_ACCOUNT_ID}

# Get auth key (hidden)
if [ -f "$CREDS_FILE" ]; then
    CURRENT_AUTH_KEY=$(grep "^BO2BOT_AUTH_KEY=" "$CREDS_FILE" | cut -d'=' -f2)
    MASKED_KEY="${CURRENT_AUTH_KEY:0:10}***${CURRENT_AUTH_KEY: -5}"
fi
read -sp "Auth key (bo2bot_...) [$MASKED_KEY]: " AUTH_KEY
AUTH_KEY=${AUTH_KEY:-$CURRENT_AUTH_KEY}
echo ""

# Validate credentials were provided
if [ -z "$HANDLE" ] || [ -z "$ADDRESS" ] || [ -z "$ACCOUNT_ID" ] || [ -z "$AUTH_KEY" ]; then
    echo "❌ Error: All credentials are required."
    exit 1
fi

# Create directory if it doesn't exist
mkdir -p ~/.hermes/secrets

# Write credentials file
cat > "$CREDS_FILE" << EOF
BO2BOT_HANDLE=$HANDLE
BO2BOT_PUBLIC_ADDRESS=$ADDRESS
BO2BOT_ACCOUNT_ID=$ACCOUNT_ID
BO2BOT_AUTH_KEY=$AUTH_KEY
EOF

# Set secure permissions (readable only by user)
chmod 600 "$CREDS_FILE"

echo ""
echo "✅ Credentials saved securely to: $CREDS_FILE"
echo ""
echo "Next steps:"
echo "  1. Run validation: bash ~/.hermes/skills/messaging/bo2bot-messaging/scripts/bo2bot-validate.sh"
echo "  2. Or use the skill: hermes chat -s bo2bot-messaging"
echo ""
