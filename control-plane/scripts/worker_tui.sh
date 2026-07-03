#!/usr/bin/env bash
# Pretty full-screen Hermes TUI bound to ONE employee's home, with that
# employee's tokens loaded from secrets.json.
#
# NOTE: this runs UNCONFINED (no seatbelt). The TUI needs node/npm writes that
# the sandbox blocks (npm install fails under seatbelt). The confined + pretty
# combo is the job of the docker backend (container = boundary, normal FS
# inside). Use this only for a local look on your own machine.
set -euo pipefail
EMP="${1:?usage: worker_tui.sh <employee-id>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="$ROOT/data/employees/$EMP"
HERMES="${HERMES_BIN:-/Users/stanislav/hermes-agent/.venv/bin/hermes}"
[ -d "$HOME_DIR" ] || { echo "no provisioned home at $HOME_DIR"; exit 1; }
export PATH="$HOME/.local/bin:$(dirname "$HERMES"):$PATH"

# Load this employee's tokens (TUI is one process → read once at launch). Use the
# _HERMES_FORCE_ prefix so Hermes passes these through to tool subprocesses (it
# strips well-known credential names like GITHUB_TOKEN otherwise).
if [ -f "$HOME_DIR/secrets.json" ]; then
  while IFS= read -r kv; do [ -n "$kv" ] && export "$kv"; done < <(
    python3 -c "import json;[print(f'_HERMES_FORCE_{k}={v}') for k,v in json.load(open('$HOME_DIR/secrets.json')).items()]" 2>/dev/null)
fi
export GITHUB_API_URL="${GITHUB_API_URL:-https://api.github.com}"
# So a skill can mint a secure entry link instead of asking for a token in chat.
export ORCHARD_API="${ORCHARD_API:-http://127.0.0.1:8700}"
export ORCHARD_EMPLOYEE_ID="$EMP"

# Refresh base skills + operating rules (SOUL.md) into this tenant's home, reusing
# the same orchard logic the daemon path uses, so the agent behaves identically.
# Run from repo root so the config's relative paths resolve correctly.
( cd "$ROOT" && "$ROOT/.venv/bin/python" -c "from orchard.config import Settings; from orchard.provisioner import sync_shared_skills, write_soul, write_hermes_config, install_fetch_helper; s=Settings.load('scripts/demo.config.yaml'); sync_shared_skills(s,'$EMP'); write_soul(s,'$EMP'); write_hermes_config(s, s.paths.home_for('$EMP')); install_fetch_helper(s,'$EMP')" ) 2>/dev/null || true

cd "$HOME_DIR/workspace"
echo "opening Hermes TUI as $EMP (UNCONFINED dev) — GITHUB_TOKEN $([ -n "${GITHUB_TOKEN:-}" ] && echo set || echo MISSING)…"
# No --yolo here: this launcher runs UNCONFINED (seatbelt breaks the TUI), so
# auto-approving tool calls would let the agent do anything on the host. Keep
# approvals ON for the unconfined TUI. --yolo is only safe on CONFINED workers
# (the daemon path, or the docker backend).
exec env HERMES_HOME="$HOME_DIR" PATH="$PATH" "$HERMES" --tui
