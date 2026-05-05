#!/usr/bin/env python3
"""Write observe-only Hermes tool registry/profile snapshot artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.tool_profile_snapshot import (  # noqa: E402
    SCHEMA_VERSION,
    build_tool_loader_contract,
    build_tool_profile_snapshot,
    build_tool_registry,
    build_tool_risk_summary,
    normalize_tool_schema,
    render_tool_risk_markdown,
    tool_name_from_schema,
)


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _load_builtin_tool_schemas() -> dict[str, dict[str, Any]]:
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    schemas: dict[str, dict[str, Any]] = {}
    for name in sorted(registry.get_all_tool_names()):
        schema = registry.get_schema(name)
        if schema:
            schemas[name] = normalize_tool_schema(schema)
    return schemas


def _load_brainstack_tool_schemas() -> dict[str, dict[str, Any]]:
    try:
        from plugins.memory.brainstack.tool_schemas import build_tool_schemas
    except Exception:
        return {}

    schemas = build_tool_schemas(
        capture_schema_version="brainstack.explicit_capture.v1",
        maintenance_schema_version="brainstack.maintenance.v1",
        maintenance_class_semantic_index="semantic_index",
        owner_user_project="user_project",
        owner_agent_assignment="agent_assignment",
        source_explicit="explicit",
        source_manual_migration="manual_migration",
        runtime_handoff_update_model_callable=False,
    )
    normalized = [normalize_tool_schema(schema) for schema in schemas]
    return {tool_name_from_schema(schema): schema for schema in normalized}


def _toolset_memberships() -> dict[str, list[str]]:
    import toolsets

    memberships: dict[str, set[str]] = {}
    for toolset_name in sorted(toolsets.get_all_toolsets()):
        try:
            names = toolsets.resolve_toolset(toolset_name)
        except Exception:
            names = []
        for name in names:
            memberships.setdefault(name, set()).add(toolset_name)
    memberships.update(
        {
            "brainstack_recall": {"memory_provider:brainstack"},
            "brainstack_inspect": {"memory_provider:brainstack"},
            "brainstack_stats": {"memory_provider:brainstack"},
            "brainstack_remember": {"memory_provider:brainstack"},
            "brainstack_supersede": {"memory_provider:brainstack"},
            "brainstack_workstream_recap": {"memory_provider:brainstack"},
            "brainstack_consolidate": {"memory_provider:brainstack"},
        }
    )
    return {name: sorted(values) for name, values in sorted(memberships.items())}


def _schemas_for(names: list[str], schemas_by_name: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [schemas_by_name[name] for name in sorted(set(names)) if name in schemas_by_name]


def _profile_tools(schemas_by_name: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    all_names = sorted(schemas_by_name)
    conversation_tools = [
        name
        for name in ["brainstack_recall", "session_search", "clarify"]
        if name in schemas_by_name
    ]
    read_like_prefixes = ("browser_", "feishu_drive_list", "feishu_drive_read", "rl_get", "rl_list")
    heavy_read = sorted(
        name
        for name in all_names
        if (
            name in conversation_tools
            or name in {"read_file", "search_files", "web_search", "web_extract", "brainstack_inspect", "brainstack_stats"}
            or name.startswith(read_like_prefixes)
        )
    )
    heavy_work = all_names
    return {
        "conversation_direct": [],
        "conversation_tools": conversation_tools,
        "heavy_read": heavy_read,
        "heavy_action": heavy_work,
        "heavy_work": heavy_work,
        "heavy_full_debug": heavy_work,
    }


def build_artifacts() -> dict[str, Any]:
    schemas_by_name = _load_builtin_tool_schemas()
    schemas_by_name.update(_load_brainstack_tool_schemas())
    memberships = _toolset_memberships()
    registry = build_tool_registry(
        [schemas_by_name[name] for name in sorted(schemas_by_name)],
        toolset_memberships=memberships,
    )
    profiles = _profile_tools(schemas_by_name)

    snapshots = []
    loaders = []
    risk_summaries = []
    for profile_name, names in sorted(profiles.items()):
        tool_schemas = _schemas_for(names, schemas_by_name)
        static_prefix = f"hermes-profile:{profile_name};tools:{','.join(sorted(names))}"
        snapshot = build_tool_profile_snapshot(
            profile_name,
            tool_schemas,
            registry,
            system_prompt=f"Hermes gateway profile {profile_name}",
            static_prefix=static_prefix,
            provider_cache_observable=False,
        )
        loader = build_tool_loader_contract(profile_name, names, registry)
        risk = build_tool_risk_summary(profile_name, registry, names)
        snapshots.append(snapshot)
        loaders.append(loader)
        risk_summaries.append(risk)

    registry_payload = {
        "schema": SCHEMA_VERSION,
        "tool_count": len(registry),
        "tools": {name: meta.to_dict() for name, meta in registry.items()},
        "unknown_metadata_tools": [
            name for name, meta in registry.items() if meta.unknown_metadata
        ],
        "notes": [
            "observe_only",
            "static_capability_metadata_not_natural_language_router",
            "brainstack_provider_tools_included_as provider-injected metadata",
        ],
    }
    snapshot_payload = {
        "schema": SCHEMA_VERSION,
        "profiles": {snapshot.profile_name: snapshot.to_dict() for snapshot in snapshots},
    }
    loader_payload = {
        "schema": SCHEMA_VERSION,
        "contracts": {loader.profile_name: loader.to_dict() for loader in loaders},
    }
    risk_payload = {
        "schema": SCHEMA_VERSION,
        "summaries": {summary.profile_name: summary.to_dict() for summary in risk_summaries},
    }
    return {
        "registry": registry_payload,
        "snapshots": snapshot_payload,
        "loaders": loader_payload,
        "risk": risk_payload,
        "risk_markdown": render_tool_risk_markdown(risk_summaries),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fail-on-unknown", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts()

    _json_dump(output_dir / "151-TOOL-REGISTRY.json", artifacts["registry"])
    _json_dump(output_dir / "151-TOOL-PROFILE-SNAPSHOTS.json", artifacts["snapshots"])
    _json_dump(output_dir / "151-TOOL-LOADER-CONTRACT.json", artifacts["loaders"])
    _json_dump(output_dir / "151-TOOL-RISK.json", artifacts["risk"])
    (output_dir / "151-TOOL-RISK.md").write_text(artifacts["risk_markdown"] + "\n")

    unknown = artifacts["registry"]["unknown_metadata_tools"]
    print(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "tool_count": artifacts["registry"]["tool_count"],
                "profiles": sorted(artifacts["snapshots"]["profiles"]),
                "unknown_metadata_tools": unknown,
            },
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_unknown and unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
