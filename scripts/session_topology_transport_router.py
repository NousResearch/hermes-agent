#!/usr/bin/env python3
"""Deterministic transport-topology router CLI (Telegram lane/topic/session routing).

This is the operator-facing CLI wrapper around:
`ops/openclaw/continuity/session_topology_router.py`.

Note: route-policy model selection uses `scripts/session_topology_router.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
TRANSPORT_ROUTER_PATH = REPO_ROOT / "ops" / "openclaw" / "continuity" / "session_topology_router.py"


def load_transport_router_module():
    spec = importlib.util.spec_from_file_location("openclaw_transport_topology_router", TRANSPORT_ROUTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("transport_router_module_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _operator_diagnostics(*, decision: str, block_reason: Optional[str]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "router": "transport_topology",
        "router_module": "ops/openclaw/continuity/session_topology_router.py",
        "docs_ref": "docs/ops/SESSION_TOPOLOGY.md",
        "disambiguation": {
            "transport_router_cli": "scripts/session_topology_transport_router.py",
            "route_policy_router_cli": "scripts/session_topology_router.py",
        },
        "suggested_commands": [
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-transport-route --topology docs/ops/templates/session_topology_transport_contract.template.json --request docs/ops/templates/session_topology_transport_route_request.template.json --json > /tmp/transport_decision.json",
            "bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology docs/ops/templates/session_topology_contract.template.json --request docs/ops/templates/session_route_request.template.json --qualification-decision <model_rollout_gate_decision.json> --transport-decision /tmp/transport_decision.json --json",
        ],
    }

    if decision == "BLOCK":
        extra: list[str] = []
        reason = str(block_reason or "")
        if reason == "route_lock_mismatch":
            extra.append("Adjust or remove request.route_lock so it matches the resolved transport tuple (agent/lane/thread/session_key).")
        elif reason in {"chat_id_missing", "chat_scope_invalid"}:
            extra.append("Fix request.chat fields (id + scope) using docs/ops/templates/session_topology_transport_route_request.template.json.")
        elif reason == "only_telegram_channel_supported":
            extra.append("Use this router only for Telegram transport contracts; other channels need their own transport router.")
        elif reason == "route_lock_invalid_message_thread_id":
            extra.append("Set route_lock.message_thread_id to a positive integer or use null/main for non-topic routing.")
        if extra:
            diagnostics["next_steps"] = extra

    return diagnostics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic transport-topology router (Telegram lane/topic/session routing)")
    ap.add_argument("--topology", required=True, help="Path to session topology transport contract JSON")
    ap.add_argument("--request", required=True, help="Path to transport routing request JSON")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    topology_path = Path(args.topology).expanduser().resolve()
    request_path = Path(args.request).expanduser().resolve()

    module = load_transport_router_module()

    try:
        topology = load_json(topology_path)
        request = load_json(request_path)
    except Exception as exc:  # pragma: no cover
        reason = "transport_route_evaluation_failed"
        out = {
            "schema": "clawd.session_topology_transport_routing.decision.v1",
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_reason": reason,
            "error": str(exc),
            "topology_path": str(topology_path),
            "request_path": str(request_path),
            "operator_diagnostics": _operator_diagnostics(decision="BLOCK", block_reason=reason),
        }
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(out, ensure_ascii=False, sort_keys=True))
        return 2

    route_payload = module.evaluate_session_topology_route(request, topology)
    decision = str(route_payload.get("decision") or "BLOCK")
    final_state = str(route_payload.get("final_state") or ("ROUTED" if decision == "PASS" else "BLOCKED"))
    block_reason = str(route_payload.get("block_reason") or "").strip() or None

    out = {
        "schema": "clawd.session_topology_transport_routing.decision.v1",
        "decision": decision,
        "final_state": final_state,
        "block_gate": route_payload.get("block_gate"),
        "block_reason": block_reason,
        "topology_path": str(topology_path),
        "request_path": str(request_path),
        "route": route_payload,
        "operator_diagnostics": _operator_diagnostics(decision=decision, block_reason=block_reason),
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
