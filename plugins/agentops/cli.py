"""Operator-only CLI for the Phase 1 observe-only control plane."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

from plugins.agentops.control.api import request_health
from plugins.agentops.control.config import default_config_path, load_agentops_config
from plugins.agentops.control.daemon import run_daemon
from plugins.agentops.control.store import inspect_store


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="agentops_command")
    daemon = subs.add_parser("daemon", help="Run the observe-only local AgentOps daemon")
    daemon.add_argument("--config", type=Path, default=default_config_path())
    doctor = subs.add_parser("doctor", help="Inspect AgentOps state without repair")
    doctor.add_argument("--config", type=Path, default=default_config_path())
    doctor.add_argument("--json", action="store_true", dest="json")
    subparser.set_defaults(func=agentops_command)


def doctor_report(config_path: Path) -> dict[str, Any]:
    config = load_agentops_config(Path(config_path))
    inspection = inspect_store(config.sqlite_path)
    daemon_health: dict[str, Any] | None = None
    if config.state_dir_safe and config.socket_path.exists():
        try:
            daemon_health = request_health(config.socket_path)
        except (OSError, RuntimeError, ValueError):
            daemon_health = {"available": False}
    daemon_ready = bool(
        daemon_health
        and daemon_health.get("ready") is True
        and daemon_health.get("store_available") is True
        and daemon_health.get("audit_chain_valid") is True
        and daemon_health.get("spool_healthy") is True
        and not daemon_health.get("safe_start_reasons")
    )
    status = "ok" if (
        not config.safe_start_reasons
        and inspection.exists
        and inspection.integrity_ok is True
        and inspection.audit_chain_valid is True
        and daemon_ready
    ) else "degraded"
    return {
        "status": status,
        "authority_mode": "observe_only",
        "global_write_enabled": False,
        "config_path": str(config.config_path),
        "safe_start_reasons": list(config.safe_start_reasons),
        "store": {
            "exists": inspection.exists,
            "schema_version": inspection.schema_version,
            "audit_chain_valid": inspection.audit_chain_valid,
            "event_count": inspection.event_count,
            "integrity_ok": inspection.integrity_ok,
            "error": inspection.error,
        },
        "daemon": daemon_health,
    }


def agentops_command(args: argparse.Namespace) -> int:
    command = getattr(args, "agentops_command", None)
    config_path = Path(getattr(args, "config", default_config_path()))
    if command == "doctor":
        report = doctor_report(config_path)
        if getattr(args, "json", False):
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        else:
            print(f"agentops doctor: {report['status']}")
            print(f"  authority: {report['authority_mode']}")
            print(f"  store: {'present' if report['store']['exists'] else 'missing'}")
        return 0 if report["status"] == "ok" else 1
    if command == "daemon":
        try:
            return run_daemon(load_agentops_config(config_path), threading.Event())
        except KeyboardInterrupt:
            return 0
    print("usage: hermes agentops {daemon,doctor}")
    return 2


def agentops_main_command(args: argparse.Namespace) -> None:
    """Preserve meaningful exit status even on Hermes CLI versions ignoring returns."""
    code = agentops_command(args)
    if code:
        raise SystemExit(code)
