#!/usr/bin/env bash
# Setup Ori Mnemos for Hermes Agent
# Usage: bash setup_ori.sh [vault_path]
#
# Installs ori-memory globally, initializes a vault, and runs the Hermes bridge.
# Restart Hermes after running.

set -euo pipefail

VAULT="${1:-$HOME/brain}"

echo "=== Ori Mnemos Setup for Hermes Agent ==="
echo ""

# Check Node.js
if ! command -v node &>/dev/null; then
  echo "Error: Node.js is required. Install from https://nodejs.org"
  exit 1
fi

# Install ori-memory
if ! command -v ori &>/dev/null; then
  echo "Installing ori-memory..."
  npm install -g ori-memory
else
  echo "ori-memory already installed: $(ori --version)"
fi

# Initialize vault
if [ -d "$VAULT/.ori" ]; then
  echo "Vault already exists at $VAULT"
else
  echo "Initializing vault at $VAULT..."
  ori init "$VAULT"
fi

# Run bridge
echo "Installing Hermes bridge..."
ori bridge hermes --vault "$VAULT"

echo ""
echo "Done. Restart Hermes to activate Ori memory."
echo "Vault: $VAULT"
