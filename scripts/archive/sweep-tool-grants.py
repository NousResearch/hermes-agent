#!/usr/bin/env python3
"""Cron entrypoint: TTL fallback sweep for temporary tool grants.
Thin wrapper over tools.tool_grants.sweep_expired_grants (repo is source of truth)."""
import json
import sys
sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
from tools.tool_grants import sweep_expired_grants

n = sweep_expired_grants(ttl_hours=24)
if n:
    print(json.dumps({"expired_grants_revoked": n, "ttl_hours": 24}))
