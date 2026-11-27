"""Bounded Mission Control contracts for the Forge/Atlas SaaS bundle.

This module is deliberately a pure projection layer. It does not read tenant
data, billing state, credentials, prompts, or Hermes session memory, and it
does not execute work. Runtime adapters may pass in already-authorized health
facts and publish the resulting contract to Mission Control.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

CONTRACT_VERSION = "hermes-os-forge-atlas-operating-layer-v1"
CAPABILITY_REGISTRY = {
    "forge": {
        "player_execution": ("tenant_safe_player_authority",),
        "runs_tasks": ("governed_run_authority",),
        "usage_cost": ("existing_saas_usage_authority",),
        "connections": ("existing_connection_authority",),
        "approvals": ("governed_approval_authority",),
        "outcomes": ("existing_outcome_authority",),
        "sanitized_health": ("atlas_contract",),
    },
    "atlas": {
        "worker_process_health": ("atlas_runtime",),
        "host_metrics": ("atlas_runtime",),
        "deployment_state": ("atlas_runtime",),
        "self_healing": ("atlas_runtime",),
        "runtime_failures": ("atlas_runtime",),
    },
}
FORBIDDEN = {
    "tenant_id", "company_id", "customer_id", "entitlement", "wallet",
    "ledger", "credential", "secret", "api_key", "prompt", "memory",
    "filesystem_path", "raw_output", "session_data",
}


def _safe_text(value: Any, fallback: str = "unknown") -> str:
    return value if isinstance(value, str) and len(value) <= 120 else fallback


def _assert_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert str(key).lower() not in FORBIDDEN, f"forbidden field: {key}"
            _assert_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_safe(child)


def sanitize_atlas_health(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Project Atlas health to fields safe for Forge, MC, and Locker Room."""
    checks = raw.get("checks") if isinstance(raw.get("checks"), Mapping) else {}
    result = {
        "status": _safe_text(raw.get("status"), "unavailable"),
        "version": _safe_text(raw.get("version")),
        "service": _safe_text(raw.get("service")),
        "checks": {
            name: _safe_text(checks.get(name), "unknown")
            for name in ("gateway", "configuration", "telemetry")
        },
    }
    _assert_safe(result)
    return result


def build_consumer_contract(
    role: str,
    health: Mapping[str, Any],
    *,
    atlas_health: Mapping[str, Any] | None = None,
    forge_health: Mapping[str, Any] | None = None,
    paco_connected: bool = False,
) -> dict[str, Any]:
    """Build a role-scoped operating-layer projection with fail-safe status."""
    if role not in {"forge", "atlas"}:
        raise ValueError("role must be forge or atlas")
    status = _safe_text(health.get("status"), "unavailable")
    result: dict[str, Any] = {
        "schema": CONTRACT_VERSION,
        "role": role,
        "status": status if status in {"healthy", "degraded", "unavailable"} else "unavailable",
        "runtime": {
            "version": _safe_text(health.get("version")),
            "service": _safe_text(health.get("service")),
        },
        "paco": "connected" if paco_connected else "disconnected",
        "mission_control": "saas_forge" if role == "forge" else "saas_atlas",
        "capability_manifest": build_capability_manifest(role),
    }
    if role == "atlas":
        # Atlas remains capable of infra monitoring when Forge is degraded.
        result["dependencies"] = {"forge": "unavailable"}
        if forge_health is not None:
            result["dependencies"]["forge"] = _safe_text(forge_health.get("status"), "unavailable")
        result["capabilities"] = ["infrastructure_health", "recovery_protection", "sanitized_incidents"]
        result["workload_policy"] = "monitor_and_protect"
    else:
        atlas = sanitize_atlas_health(atlas_health or {"status": "unavailable"})
        result["dependencies"] = {"atlas": atlas["status"]}
        result["capabilities"] = ["business_control", "approvals", "sanitized_usage", "customer_state_preservation"]
        result["workload_policy"] = "fail_safe" if atlas["status"] != "healthy" else "normal"
    _assert_safe(result)
    return result


def build_capability_manifest(role: str, observed: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Expose routing metadata while keeping execution and authority elsewhere."""
    if role not in CAPABILITY_REGISTRY:
        raise ValueError("role must be forge or atlas")
    observed = observed or {}
    capabilities = []
    for name, authorities in CAPABILITY_REGISTRY[role].items():
        status = str(observed.get(name, "declared")).lower()
        if status not in {"declared", "observed", "degraded", "unavailable"}:
            status = "unavailable"
        capabilities.append({"name": name, "status": status, "authorities": list(authorities)})
    return {"schema": CONTRACT_VERSION, "role": role, "capabilities": capabilities,
            "execution_enabled": False, "source_of_truth": "existing_authorities"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit a sanitized MC operating-layer projection")
    parser.add_argument("--role", choices=("forge", "atlas"), required=True)
    parser.add_argument("--health", default="{}", help="JSON health facts supplied by the runtime adapter")
    args = parser.parse_args()
    print(json.dumps(build_consumer_contract(args.role, json.loads(args.health)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
