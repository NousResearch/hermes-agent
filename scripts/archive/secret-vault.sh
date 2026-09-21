#!/usr/bin/env bash
set -euo pipefail

# Thin stable wrapper for the local Hermes secret vault.
# Values are never logged by this wrapper. Prefer interactive prompt or --stdin.

exec python3 "$HOME/.hermes/secret-vault/cli.py" "$@"
