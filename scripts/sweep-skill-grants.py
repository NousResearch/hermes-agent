#!/usr/bin/env python3
"""Cron entrypoint: TTL fallback sweep for temporary skill grants.

Thin wrapper over tools.skill_grants.sweep_expired_grants (repo is source of
truth). Portable: sys.path is derived from __file__ so the wrapper runs from
any checkout location, not a hardcoded absolute path.

P13 isolation: pass --dry-run to suppress the sweep (no revoke events are
appended to the profile activity ledger). The wrapper prints what would run
and exits 0 without touching the ledger. HERMES_HOME (read by the ledger via
the config layer) selects the active home for any read path.
"""
import json
import os
import sys
from pathlib import Path

# Prefer the scheduler-provided canonical runtime root.  When invoked
# standalone or copied for a relocation test, retain self-contained discovery.
RUNTIME_ROOT = Path(os.environ.get("HERMES_AGENT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(RUNTIME_ROOT))
from tools.skill_grants import sweep_expired_grants

if "--dry-run" in sys.argv:
    print(json.dumps({"dry_run": True, "ttl_hours": 24, "action": "sweep_expired_grants"}))
    sys.exit(0)

n = sweep_expired_grants(ttl_hours=24)
if n:
    print(json.dumps({"expired_grants_revoked": n, "ttl_hours": 24}))
