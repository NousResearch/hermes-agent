"""Bridge imported session-topology routing policy into Hermes-native routing governance."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from typing import Any

from agent.routing_governance import read_rollout_state, route_matches_qualified
from hermes_constants import get_hermes_home

_POLICY_MODULE_NAME = "session_topology_routing_policy_contract"
_COST_TIER_BY_FAMILY = {
    "codex": "premium",
    "claude": "premium",
    "gemini": "balanced",
    "deepseek": "balanced",
    "kimi": "economy",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snapshot_path() -> Path:
    return get_hermes_home() / "routing_governance" / "session_topology_snapshot.json"


def _load_policy_contract(repo_root: Path):
    module_path = repo_root / "scripts" / "session_topology_routing_policy_contract.py"
    spec = importlib.util.spec_from_file_location(_POLICY_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load routing policy contract from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy_paths(repo_root: Path) -> tuple[Path, Path]:
    return (
        repo_root / "docs" / "ops" / "session_topology_routing_policy_v1.json",
        repo_root / "docs" / "ops" / "schemas" / "session_topology_routing_policy.schema.json",
    )


def _family_from_route(provider: str | None, model: str | None) -> str:
    provider_token = str(provider or "").strip().lower()
    model_token = str(model or "").strip().lower()
    combined = f"{provider_token} {model_token}"
    if "codex" in combined or provider_token == "openai-codex":
        return "Codex"
    if "gemini" in combined or provider_token in {"google", "gemini"}:
        return "Gemini"
    if "deepseek" in combined:
        return "DeepSeek"
    if "kimi" in combined or provider_token in {"moonshot", "kimi"}:
        return "Kimi"
    if "claude" in combined or provider_token == "anthropic":
        return "Claude"
    return "Other"


def _cost_tier(family: str) -> str:
    return _COST_TIER_BY_FAMILY.get(str(family or "").strip().lower(), "unknown")


def _dedupe_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for route in routes:
        provider = str(route.get("provider") or "").strip().lower()
        model = str(route.get("model") or "").strip()
        if not provider or not model:
            continue
        key = (provider, model)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                **route,
                "provider": provider,
                "model": model,
                "sources": list(route.get("sources") or []),
            }
            continue
        existing_sources = set(existing.get("sources") or [])
        for source in route.get("sources") or []:
            if source not in existing_sources:
                existing.setdefault("sources", []).append(source)
                existing_sources.add(source)
    return list(merged.values())


def _normalize_probe_summary(summary: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(summary, dict):
        return rows
    for item in summary.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "").strip().lower()
        model = str(item.get("model") or "").strip()
        if provider and model:
            rows[(provider, model)] = item
    return rows


def _build_available_routes(
    *,
    primary_route: dict[str, Any],
    fallback_routes: list[dict[str, Any]],
    rollout_state: dict[str, Any],
    probe_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    probe_index = _normalize_probe_summary(probe_summary)
    raw_routes: list[dict[str, Any]] = []

    primary = {
        "provider": primary_route.get("provider"),
        "model": primary_route.get("model"),
        "sources": ["primary"],
    }
    raw_routes.append(primary)

    for item in fallback_routes or []:
        raw_routes.append({
            "provider": item.get("provider"),
            "model": item.get("model"),
            "sources": ["fallback"],
        })

    current = rollout_state.get("current_route") or {}
    if isinstance(current, dict):
        raw_routes.append({
            "provider": current.get("provider"),
            "model": current.get("model"),
            "sources": ["current_rollout"],
        })

    for item in rollout_state.get("qualified_routes", []) or []:
        if not isinstance(item, dict):
            continue
        raw_routes.append({
            "provider": item.get("provider"),
            "model": item.get("model"),
            "sources": ["qualified"],
        })

    routes = _dedupe_routes(raw_routes)
    for route in routes:
        route["family"] = _family_from_route(route.get("provider"), route.get("model"))
        route["heuristic_cost_tier"] = _cost_tier(route["family"])
        route["qualified"] = route_matches_qualified(route.get("provider", ""), route.get("model", ""))
        probe = probe_index.get((route.get("provider", ""), route.get("model", "")), {})
        route["health"] = {
            "ok": bool(probe.get("ok")),
            "classification": probe.get("classification"),
            "probed": bool(probe),
        }
    return routes


def load_session_topology_policy(repo_root: Path) -> dict[str, Any]:
    contract = _load_policy_contract(repo_root)
    policy_path, schema_path = _policy_paths(repo_root)
    ok, reason, details, policy_doc = contract.load_routing_policy(policy_path, schema_path)
    return {
        "ok": ok,
        "reason": reason,
        "details": details,
        "policy": policy_doc,
        "contract": contract,
        "policy_path": str(policy_path),
        "schema_path": str(schema_path),
    }


def _select_candidates(routes: list[dict[str, Any]], families: list[str]) -> list[dict[str, Any]]:
    source_priority = {"current_rollout": 0, "primary": 1, "qualified": 2, "fallback": 3}

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for family in families:
        family_routes = [route for route in routes if route.get("family") == family]
        family_routes.sort(
            key=lambda route: (
                0 if route.get("health", {}).get("ok") else 1,
                min(source_priority.get(source, 9) for source in route.get("sources", []) or ["fallback"]),
                route.get("provider", ""),
                route.get("model", ""),
            )
        )
        for route in family_routes:
            key = (route.get("provider", ""), route.get("model", ""))
            if key in seen:
                continue
            selected.append(route)
            seen.add(key)
    return selected


def build_task_class_route_plan(
    *,
    task_class: str,
    primary_route: dict[str, Any],
    fallback_routes: list[dict[str, Any]] | None,
    repo_root: Path,
    probe_summary: dict[str, Any] | None = None,
    rollout_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rollout_state = rollout_state or read_rollout_state()
    loaded = load_session_topology_policy(repo_root)
    if not loaded["ok"]:
        raise RuntimeError(f"routing policy unavailable: {loaded['reason'] or loaded['details']}")

    contract = loaded["contract"]
    policy = loaded["policy"]
    default_family, fallback_families = contract.routing_policy_task_family(policy, task_class)
    routes = _build_available_routes(
        primary_route=primary_route,
        fallback_routes=list(fallback_routes or []),
        rollout_state=rollout_state,
        probe_summary=probe_summary,
    )
    preferred_families = [default_family, *fallback_families]
    selected = _select_candidates(routes, preferred_families)
    selected_route = selected[0] if selected else None
    coding_task = task_class == "implementation" or task_class.startswith("code:")

    plan = {
        "task_class": task_class,
        "preferred_families": preferred_families,
        "available_families": sorted({route.get("family") for route in routes if route.get("family")}),
        "selected_route": selected_route,
        "candidate_routes": selected,
        "missing_preferred_families": [family for family in preferred_families if not any(route.get("family") == family for route in routes)],
        "parity": {
            "default_family_available": any(route.get("family") == default_family for route in routes),
            "has_any_policy_candidate": bool(selected),
        },
    }

    if coding_task:
        qualified = bool(selected_route and selected_route.get("qualified"))
        plan["coding_policy"] = {
            "require_qualification_signal": bool(contract.routing_policy_coding_require_qualification_signal(policy)),
            "min_score_high": contract.routing_policy_coding_min_score(policy, "high"),
            "min_score_critical": contract.routing_policy_coding_min_score(policy, "critical"),
            "allowed_readiness_high": sorted(contract.routing_policy_coding_allowed_readiness(policy, "high")),
            "allowed_readiness_critical": sorted(contract.routing_policy_coding_allowed_readiness(policy, "critical")),
            "route_is_qualified": qualified,
            "eligible_for_high_risk_rollout": qualified or not contract.routing_policy_coding_require_qualification_signal(policy),
        }
    return plan


def build_routing_governance_snapshot(
    *,
    primary_route: dict[str, Any],
    fallback_routes: list[dict[str, Any]] | None,
    repo_root: Path,
    probe_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rollout_state = read_rollout_state()
    loaded = load_session_topology_policy(repo_root)
    if not loaded["ok"]:
        raise RuntimeError(f"routing policy unavailable: {loaded['reason'] or loaded['details']}")

    contract = loaded["contract"]
    policy = loaded["policy"]
    routes = _build_available_routes(
        primary_route=primary_route,
        fallback_routes=list(fallback_routes or []),
        rollout_state=rollout_state,
        probe_summary=probe_summary,
    )
    known_task_classes = sorted(contract.routing_policy_known_task_classes(policy))
    plans = [
        build_task_class_route_plan(
            task_class=task_class,
            primary_route=primary_route,
            fallback_routes=fallback_routes,
            repo_root=repo_root,
            probe_summary=probe_summary,
            rollout_state=rollout_state,
        )
        for task_class in known_task_classes
    ]

    health_counts: dict[str, int] = {}
    for route in routes:
        classification = str(route.get("health", {}).get("classification") or "unknown")
        health_counts[classification] = health_counts.get(classification, 0) + 1

    family_counts: dict[str, int] = {}
    cost_tier_counts: dict[str, int] = {}
    for route in routes:
        family = str(route.get("family") or "Other")
        family_counts[family] = family_counts.get(family, 0) + 1
        tier = str(route.get("heuristic_cost_tier") or "unknown")
        cost_tier_counts[tier] = cost_tier_counts.get(tier, 0) + 1

    snapshot = {
        "schema": "hermes.session_topology_routing_snapshot.v1",
        "generated_at": _utc_now_iso(),
        "policy": {
            "policy_id": policy.get("policy_id"),
            "policy_path": loaded["policy_path"],
            "schema_path": loaded["schema_path"],
        },
        "rollout": rollout_state,
        "available_routes": routes,
        "task_class_plans": plans,
        "parity_validation": {
            "known_task_classes": known_task_classes,
            "tasks_with_selected_route": sum(1 for plan in plans if plan.get("selected_route")),
            "tasks_missing_default_family": [plan["task_class"] for plan in plans if not plan["parity"]["default_family_available"]],
            "tasks_without_any_policy_candidate": [plan["task_class"] for plan in plans if not plan["parity"]["has_any_policy_candidate"]],
        },
        "cost_governance": {
            "heuristic": "family_cost_tier",
            "family_counts": family_counts,
            "cost_tier_counts": cost_tier_counts,
            "rollout_mode": rollout_state.get("rollout", {}).get("mode"),
            "rollout_max_percent": rollout_state.get("rollout", {}).get("max_percent"),
            "qualified_route_count": len(rollout_state.get("qualified_routes", []) or []),
        },
        "health_summary": {
            "probe_results_seen": sum(1 for route in routes if route.get("health", {}).get("probed")),
            "classification_counts": health_counts,
        },
    }

    path = _snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    snapshot["snapshot_path"] = str(path)
    return snapshot
