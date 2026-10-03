#!/bin/bash
# Copy this shim physically into ~/.hermes/scripts/ for native no-agent jobs.
set -euo pipefail
exec /usr/bin/python3 "$HOME/HermesCompanion/scripts/refresh_hermes_phone.py"
