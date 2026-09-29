"""Agent Client Protocol (ACP) Bridge for External IDEs (`samagent/acp_bridge.py`).

Allows external IDEs that connect via `acp_adapter` (VS Code ACP, Cursor, Zed, JetBrains) to invoke
SamAgent's Spec Contract, L0–L4 Pre-Production Verification, and Production Release Bundler directly
via slash commands (`/samagent-verify`, `/samagent-preprod`, `/samagent-promote`) or programmatic RPC.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from samagent.conductor.verify import VerificationRunner
from samagent.ide_bridge import evaluate_pre_production_gate
from samagent.ide_watcher import check_and_sync_external_edits
from samagent.platform_installer import get_default_projects_root
from samagent.prod_bundler import promote_to_production
from samagent.spec.models import SpecDocument


def resolve_active_workspace(cwd: Optional[str] = None) -> Path:
    if cwd:
        p = Path(cwd).resolve()
        if (p / ".samagent" / "spec.yaml").exists():
            return p
    env_ws = os.environ.get("SAMAGENT_WORKSPACE_DIR")
    if env_ws:
        return Path(env_ws).resolve()
    return (get_default_projects_root() / "yoga-studio-local").resolve()


def run_acp_samagent_command(command: str, args_text: str = "", cwd: Optional[str] = None) -> Dict[str, Any]:
    """Execute a SamAgent IDE slash command (`samagent-verify`, `samagent-preprod`, `samagent-promote`)."""
    ws = resolve_active_workspace(cwd)
    cmd = command.strip().lstrip("/").lower()

    if not (ws / ".samagent" / "spec.yaml").exists():
        return {
            "ok": False,
            "workspace": str(ws),
            "message": f"No .samagent/spec.yaml found in {ws}. Open the SamAgent Local Platform at http://127.0.0.1:8080 to initialize.",
        }

    if cmd in ("samagent-verify", "verify"):
        sync_info = check_and_sync_external_edits(ws)
        spec = SpecDocument.load(ws)
        report = VerificationRunner(ws, spec).run_all()
        gate = evaluate_pre_production_gate(ws, {"verification": report.to_dict()})
        return {
            "ok": report.passed,
            "command": "samagent-verify",
            "workspace": str(ws),
            "ide_autosync": sync_info,
            "verification_passed": report.passed,
            "failed_levels": report.failed_levels,
            "pre_prod_gate": gate,
            "message": (
                f"SamAgent L0–L4 Verification PASSED in {ws.name}. Pre-Production Gate: READY."
                if gate["ready_for_production"]
                else f"SamAgent Pre-Production Gate BLOCKED in {ws.name}: {', '.join(gate['blockers'])}"
            ),
        }

    if cmd in ("samagent-preprod", "preprod"):
        spec = SpecDocument.load(ws)
        report = VerificationRunner(ws, spec).run_all()
        gate = evaluate_pre_production_gate(ws, {"verification": report.to_dict()})
        return {
            "ok": gate["ready_for_production"],
            "command": "samagent-preprod",
            "workspace": str(ws),
            "pre_prod_gate": gate,
            "message": json.dumps(gate, indent=2),
        }

    if cmd in ("samagent-promote", "promote"):
        promo = promote_to_production(ws, release_tag=args_text.strip() or None)
        return {
            "ok": promo["promoted"],
            "command": "samagent-promote",
            "workspace": str(ws),
            "promotion": promo,
            "message": (
                f"Promoted {ws.name} to Production Release {promo.get('release_id')} ({promo.get('manifest_path')})"
                if promo["promoted"]
                else f"Production Promotion BLOCKED: {', '.join(promo.get('blockers') or [])}"
            ),
        }

    return {"ok": False, "workspace": str(ws), "message": f"Unknown SamAgent ACP command: {command}"}
