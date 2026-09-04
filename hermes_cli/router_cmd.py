"""Read-only model-router inspection commands."""
from __future__ import annotations

import json
from pathlib import Path


def _router_db_path():
    from agent.model_router.pipeline import default_db_path
    return default_db_path()


def _load_cfg():
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        return cfg.get("model_router", {}) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def cmd_router(args):
    from agent.model_router.state import RouterStateStore
    from agent.model_router.telemetry import RouterTelemetry

    action = getattr(args, "router_action", None) or "status"
    cfg = _load_cfg()
    db_path = _router_db_path()
    telemetry = RouterTelemetry(db_path)
    state = RouterStateStore(db_path)

    if action == "status":
        payload = {
            "mode": cfg.get("mode", "off"),
            "candidate_count": len(cfg.get("candidates", [])) if isinstance(cfg.get("candidates", []), list) else 0,
            "pinned_sessions": len(state.list_pins()),
            "db_path": str(db_path),
            "db_size": db_path.stat().st_size if db_path.exists() else 0,
        }
    elif action == "history":
        payload = telemetry.history(
            limit=max(1, min(1000, int(getattr(args, "limit", 20)))),
            session_id=getattr(args, "session", "") or "",
        )
    elif action == "stats":
        payload = telemetry.stats()
    else:
        raise ValueError(f"unknown router action: {action}")

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
