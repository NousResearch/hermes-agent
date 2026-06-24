#!/usr/bin/env bash
set -euo pipefail

python3 "${HERMES_HOME:-$HOME/.hermes}/scripts/improvement_queue.py" digest
