"""Runtime health checks for Trend Discovery Center."""

from __future__ import annotations

import json
import socket
from urllib import request

from .store import TrendDiscoveryStore


def check_localhost_url(url: str, timeout: int = 3) -> dict:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return {"ok": True, "status_code": resp.status}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def health_check(store: TrendDiscoveryStore, *, check_dashboard: bool = False) -> dict:
    store.init()
    snapshot = store.status_snapshot()
    checks = {
        "database": {"ok": store.path.exists(), "path": str(store.path)},
        "hostname": {"ok": True, "value": socket.gethostname()},
        "phases_seeded": {"ok": len(snapshot["phases"]) >= 5, "count": len(snapshot["phases"])},
        "issues_seeded": {"ok": len(snapshot["issues"]) >= 50, "count": len(snapshot["issues"])},
        "sources_seeded": {"ok": len(snapshot["sources"]) >= 5, "count": len(snapshot["sources"])},
    }
    if check_dashboard:
        url = store.get_config("health.localhost_url", "http://127.0.0.1:9119")
        checks["localhost_dashboard"] = check_localhost_url(url)
    ok = all(item.get("ok") for item in checks.values())
    return {"ok": ok, "checks": checks}


def health_json(store: TrendDiscoveryStore, *, check_dashboard: bool = False) -> str:
    return json.dumps(health_check(store, check_dashboard=check_dashboard), indent=2, sort_keys=True)
