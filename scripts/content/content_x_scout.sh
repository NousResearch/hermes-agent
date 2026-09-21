#!/usr/bin/env bash
# X auto-suggest scout: search Sahil niches and stage repurpose suggestions
# for approval via the G03 content gate. Dormant until the X API spend cap
# resets (2026-06-14); skips safely.
# P13: CONTENT_SCOUT_DRY_RUN=1 short-circuits before xurl/Discord invocation.
set -euo pipefail

if [[ "${CONTENT_SCOUT_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run: would run x_scout (xurl fetch + gate approval cards)"
  exit 0
fi

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
CE="/home/kensei/repos/KenseiAgent/content_engine"
cd "$CE"
set -a
# shellcheck disable=SC1091
source "${HERMES_HOME_DIR}/.env"
set +a
export PYTHONPATH="${CE}:${PYTHONPATH:-}"

# G03 gate check: verify the gate module is importable and functional before
# running the scout. If the gate is broken, fail closed (exit non-zero).
/home/kensei/repos/KenseiAgent/.venv/bin/python -c "
import sys
sys.path.insert(0, '$CE')
from content_gate import init_gate_db, get_pending
init_gate_db()
assert callable(get_pending)
"

/home/kensei/repos/KenseiAgent/.venv/bin/python x_scout.py
